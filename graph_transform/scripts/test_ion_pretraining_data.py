#!/usr/bin/env python3
"""Bounded synthetic tests: never read the real MSP or exclusion datasets."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from graph_transform.pretraining import msp_dataset as data


SEQ = "PEPTIDER"


def peak(seq=SEQ, ion="b", ordinal=2, charge=1, intensity=10, ppm=0, annotation=None):
    mz = data.fragment_mz(seq, ion, ordinal, charge) * (1 + ppm * 1e-6)
    if annotation is None:
        annotation = f"{ion}{ordinal}" + (f"^{charge}" if charge != 1 else "") + "/-999ppm"
    return f'{mz:.10f}\t{intensity}\t"{annotation}"'


def record(peaks, name=SEQ + "/4"):
    return data.MSPRecord(name, 1, 1, mods=0, declared_peaks=len(peaks), peaks=peaks)


def spectrum(seq, peaks, charge=4, parent="garbage-ignored", mods=0):
    return (f"Name: {seq}/{charge}\nComment: Parent={parent} Mods={mods} Modstring={seq}///{charge}\n"
            f"Num peaks: {len(peaks)}\n" + "\n".join(peaks) + "\n\n")


def split_examples():
    examples = {}
    for suffix in itertools.product("ACDEFG", repeat=3):
        seq = "PEPT" + "".join(suffix)
        examples.setdefault(data.split_for_sequence(seq), seq)
        if len(examples) == 2:
            return examples
    raise AssertionError("Synthetic hash search did not find both splits")


class FragmentTests(unittest.TestCase):
    def test_standard_mass_and_alignment(self):
        # Independent known residue calculation: y1(R) is R + water + proton.
        self.assertAlmostEqual(data.fragment_mz(SEQ, "y", 1, 1), 175.11895217407, places=9)
        self.assertEqual(data.fragment_slot(8, "y", 1, 2), (6, 3))
        self.assertEqual(data.fragment_slot(8, "b", 2, 2), (1, 1))
        self.assertAlmostEqual(data.precursor_mz("AAAAAAA", 4),
                               (7 * 71.03711378471 + 18.0105646837 + 4 * 1.00727646677) / 4)
        for args in ((8, "b", 0, 1), (8, "y", 8, 1), (8, "b", 2, 3)):
            with self.assertRaises(ValueError):
                data.fragment_slot(*args)

    def test_charge_is_caret_not_ppm_suffix(self):
        parsed = data.parse_annotation('"y3^2/10.51ppm"')
        self.assertEqual((parsed.category, parsed.ion, parsed.ordinal, parsed.charge), ("standard", "y", 3, 2))
        self.assertEqual(data.parse_annotation("b2/-1.5ppm").charge, 1)
        self.assertEqual(data.parse_annotation("b2/2").category, "unresolved")
        self.assertEqual(data.parse_annotation("b2^3/1ppm").category, "out_of_scope_charge")
        self.assertEqual(data.parse_annotation("b2+1/1ppm").category, "unresolved")

    def test_four_channel_alignment_duplicate_max_normalization_and_missing(self):
        peaks = [peak(ion="b", ordinal=2, charge=1, intensity=10),
                 peak(ion="b", ordinal=2, charge=2, intensity=20),
                 peak(ion="y", ordinal=1, charge=1, intensity=30),
                 peak(ion="y", ordinal=1, charge=2, intensity=40),
                 peak(ion="b", ordinal=2, charge=1, intensity=15)]
        qa = data.qa_spectrum(SEQ, record(peaks))
        self.assertIsNone(qa.reason)
        self.assertEqual(qa.intensities[1], [.375, .5, 0, 0])
        self.assertEqual(qa.intensities[6], [0, 0, .75, 1])
        self.assertEqual(qa.mask[1], [True, True, False, False])
        self.assertEqual(qa.mask[0], [False] * 4)
        self.assertEqual(qa.counts["duplicate_valid_slot_assignments"], 1)

    def test_neutral_loss_never_supervises_standard_channel(self):
        qa = data.qa_spectrum(SEQ, record([peak(ion="y", ordinal=1),
                                           peak(annotation="b2-H2O^2/0ppm"),
                                           peak(annotation="b2^2-NH3/0ppm")]))
        self.assertIsNone(qa.reason)
        self.assertFalse(qa.mask[1][1])
        self.assertEqual(sum(sum(row) for row in qa.mask), 1)
        self.assertEqual(qa.counts["annotation_neutral_loss"], 2)

    def test_ambiguous_slot_stays_masked_even_with_valid_duplicate(self):
        peaks = [peak(), peak(annotation="b2/0ppm,y3^2/0ppm"), peak(ion="y", ordinal=1)]
        for ordered in (peaks, list(reversed(peaks))):
            qa = data.qa_spectrum(SEQ, record(ordered))
            self.assertIsNone(qa.reason)
            self.assertFalse(qa.mask[1][0])
            self.assertFalse(qa.mask[4][3])
            self.assertTrue(qa.mask[6][2])
        unknown = data.qa_spectrum(SEQ, record([peak(), peak(annotation="b2/0ppm,?unresolved")]))
        self.assertIn("unlocalizable", unknown.reason)

    def test_mass_error_is_recomputed_and_failing_slot_blocked(self):
        passing = data.qa_spectrum(SEQ, record([peak(ppm=49, annotation="b2/999999ppm")]))
        self.assertIsNone(passing.reason)
        failing = data.qa_spectrum(SEQ, record([peak(ppm=51, annotation="b2/0ppm"), peak(),
                                               peak(ion="y", ordinal=1)]))
        self.assertIsNone(failing.reason)
        self.assertFalse(failing.mask[1][0])
        self.assertEqual(failing.counts["mass_outside_50ppm"], 1)

    def test_invalid_numeric_peak_fails_closed(self):
        for mz, intensity in (("nan", "1"), ("inf", "1"), ("-1", "2"), ("100", "nan"),
                              ("100", "inf"), ("100", "0"), ("100", "-1")):
            with self.subTest(mz=mz, intensity=intensity):
                qa = data.qa_spectrum(SEQ, record([peak(), f'{mz} {intensity} "b3/0ppm"']))
                self.assertIn("invalid_numeric", qa.reason)
                self.assertFalse(any(any(row) for row in qa.mask))

    def test_missing_unresolved_out_of_range_and_empty_supervision(self):
        for annotation in ("", "b2+1/0ppm", "b99/0ppm", "?", "b2/1e999ppm"):
            qa = data.qa_spectrum(SEQ, record([peak(), peak(annotation=annotation)]))
            self.assertIsNotNone(qa.reason)
        qa = data.qa_spectrum(SEQ, record([peak(annotation="b2-H2O/0ppm")]))
        self.assertEqual(qa.reason, "no_valid_unambiguous_slot")
        malformed = record([peak()])
        malformed.declared_peaks = 2
        self.assertEqual(data.qa_spectrum(SEQ, malformed).reason, "malformed_header_or_peak_count")


class ConstructionTests(unittest.TestCase):
    def test_hash_and_il_identity_are_fixed(self):
        expected = hashlib.sha256(b"A\nB\n").hexdigest()
        self.assertEqual(data.sequence_set_sha256(["B", "A", "A"]), expected)
        self.assertEqual(data.sequence_set_sha256([]), hashlib.sha256(b"").hexdigest())
        self.assertEqual(data.sequence_identity(" ilboxz "), "LLBOXZ")
        self.assertEqual(data.split_for_sequence("PEPTIDE"), data.split_for_sequence("PEPTLDE"))

    def test_seq_only_exclusions(self):
        with tempfile.TemporaryDirectory(prefix="ion4_unit_seqonly_") as temporary:
            path = Path(temporary) / "exclude.csv"
            path.write_text("seq,label,prediction\nPEPTIDE,do-not-consume,do-not-consume\npeptide,x,y\n", encoding="utf-8")
            sequences, sources = data.read_sequence_columns([path])
            self.assertEqual(sequences, {"PEPTIDE"})
            self.assertEqual(sources[0]["columns_read"], ["seq"])
            self.assertEqual(sources[0]["rows"], 2)
            self.assertNotIn("label", sources[0])

    def test_single_pass_build_first_passing_dedup_isolation_and_provenance(self):
        examples = split_examples()
        train, val = examples["train"], examples["val"]
        text = spectrum(train, [peak(train, ppm=200)])  # rejected; must not reserve key
        text += spectrum(train, [peak(train, intensity=10), peak(train, ion="y", ordinal=1, intensity=40)])
        text += spectrum(train, [peak(train, intensity=40), peak(train, ion="y", ordinal=1, intensity=10)])
        text += spectrum(train, [peak(train)], charge=5)  # same seq must keep same split
        text += spectrum(val, [peak(val)])
        text += spectrum("PEPTIDE", [peak("PEPTIDE")])  # exact NIST
        text += spectrum("PEPTLDE", [peak("PEPTLDE")])  # additional I/L
        text += spectrum("ACDEFGH", [peak("ACDEFGH")])  # exact MiPD
        text += spectrum("PEPBOXZ", ["100 1 b2/0ppm"])  # special chemistry excluded
        text += spectrum(train, [peak(train)], mods=1)
        with tempfile.TemporaryDirectory(prefix="ion4_unit_build_") as temporary:
            base = Path(temporary)
            source, output = base / "fixture.msp", base / "output"
            source.write_bytes(text.encode())
            manifest = data.build_dataset(source, output, {"PEPTIDE"}, {"ACDEFGH"},
                                          expected_msp_sha256=hashlib.sha256(text.encode()).hexdigest())
            self.assertEqual(manifest["source"]["streaming_passes"], 1)
            self.assertEqual(manifest["source"]["sha256"], data.file_sha256(source))
            self.assertEqual(manifest["outputs"]["train"]["rows"], 2)
            self.assertEqual(manifest["outputs"]["val"]["rows"], 1)
            self.assertEqual(manifest["outputs"]["train"]["sequences"], [train])
            self.assertEqual(manifest["outputs"]["val"]["sequences"], [val])
            self.assertEqual(manifest["statistics"]["counts"]["candidate_spectra_fragment_qa"], 5)
            self.assertEqual(manifest["statistics"]["rejections"]["later_qa_passing_duplicate_seq_charge"], 1)
            self.assertEqual(manifest["statistics"]["rejections"]["excluded_nist_additional_il_identity"], 1)
            for split in ("train", "val"):
                metadata = manifest["outputs"][split]
                path = output / metadata["csv"]
                self.assertFalse(Path(metadata["csv"]).is_absolute())
                self.assertEqual(metadata["sha256"], data.file_sha256(path))
                with path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                for row in rows:
                    targets, mask = json.loads(row["ion_intensities_json"]), json.loads(row["ion_mask_json"])
                    self.assertEqual(len(targets), len(row["seq"]) - 1)
                    self.assertEqual(row["true_multi"], ";".join(["0"] * (len(row["seq"]) - 1)))
                    self.assertEqual(row["nce"], "0")
                    self.assertAlmostEqual(float(row["pep_mass"]), data.precursor_mz(row["seq"], int(row["charge"])))
                    self.assertTrue(all(math.isfinite(value) and (0 < value <= 1 if valid else value == 0)
                                        for values, masks in zip(targets, mask) for value, valid in zip(values, masks)))
                    if row["seq"] == train and row["charge"] == "4":
                        self.assertEqual(targets[1][0], .25)  # first passing, not last/best
            identity = manifest["exclusion"]["identity_file"]
            self.assertEqual(data.file_sha256(output / identity["path"]), identity["sha256"])
            self.assertFalse(manifest["exclusion"]["certification"]["certified_full_exclusion_sets"])
            self.assertEqual(json.loads((output / "build_status.json").read_text())["status"], "completed")
            with self.assertRaises(FileExistsError):
                data.build_dataset(source, output, {"PEPTIDE"}, {"ACDEFGH"})

    def test_parent_is_not_consumed(self):
        with tempfile.TemporaryDirectory(prefix="ion4_unit_parent_") as temporary:
            base = Path(temporary)
            outputs = []
            for index, parent in enumerate(("not-numeric", "-999", "999999999")):
                path = base / f"fixture{index}.msp"
                path.write_text(spectrum(SEQ, [peak()], parent=parent), encoding="utf-8")
                records = list(data.MSPStream(path))
                qa = data.qa_spectrum(SEQ, records[0])
                outputs.append((qa.intensities, qa.mask, records[0].mods, records[0].modstring))
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(outputs[1], outputs[2])

    def test_wrong_msp_hash_never_emits_complete_manifest(self):
        examples = split_examples()
        with tempfile.TemporaryDirectory(prefix="ion4_unit_hash_") as temporary:
            source, output = Path(temporary) / "fixture.msp", Path(temporary) / "output"
            source.write_text("".join(spectrum(seq, [peak(seq)]) for seq in examples.values()), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA mismatch"):
                data.build_dataset(source, output, {"PEPTIDE"}, {"ACDEFGH"}, expected_msp_sha256="0" * 64)
            self.assertFalse((output / "manifest.json").exists())
            self.assertEqual(json.loads((output / "build_status.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
