"""Conservative, one-pass MSP library-annotation intensity data construction.

This is an auxiliary target, not a cleavage classification dataset or a claim
that complete experimental spectra have been validated. Unobserved channels are
unknown, never negative. Parent, iRT, acquisition energy and precursor intensity
are neither parsed nor used. Only sequence identities are read from exclusions.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Iterable


CHANNEL_ORDER = ["b1", "b2", "y1", "y2"]
CSV_COLUMNS = ["name", "seq", "charge", "pep_mass", "intensity", "nce",
               "scan_num", "rt", "fbr", "tb", "fb", "mb", "true_multi",
               "ion_intensities_json", "ion_mask_json"]
IDENTITY_RULE = "uppercase_base_sequence_I_to_L"
SPLIT_SALT = "ion4-seq-split-v1:"
MAX_PPM = 50.0
PROTON = 1.00727646677
WATER = 18.0105646837
# Standard monoisotopic residue masses (amino acid minus water), matching
# pyteomics.mass.std_aa_mass. Local constants avoid PBCLA's mutated global table.
RESIDUE_MASS = {
    "A": 71.03711378471, "C": 103.00918478471, "D": 115.02694302383001,
    "E": 129.04259308796998, "F": 147.06841391298997, "G": 57.02146372057,
    "H": 137.05891185845002, "I": 113.08406397713001, "K": 128.09496301399997,
    "L": 113.08406397713001, "M": 131.04048491299, "N": 114.04292744114001,
    "P": 97.05276384885, "Q": 128.05857750527997, "R": 156.10111102359997,
    "S": 87.03202840427001, "T": 101.04767846841, "V": 99.06841391299,
    "W": 186.07931294985997, "Y": 163.06332853254997,
}
FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
ERROR_SUFFIX = r"(?:/(" + FLOAT_PATTERN + r")ppm)?"
STANDARD_ION = re.compile(r"([by])([1-9]\d*)(?:\^([1-9]\d*))?" + ERROR_SUFFIX)
# Narrow, explicit loss grammar: unsupported notation is rejected, not guessed.
LOSS_ION = re.compile(r"([by])([1-9]\d*)(?:-(H2O|NH3)(?:\^([1-9]\d*))?|"
                      r"\^([1-9]\d*)-(H2O|NH3))" + ERROR_SUFFIX)
MODS = re.compile(r"(?:^|\s)Mods=(\d+)(?=[/\s]|$)")
MODSTRING = re.compile(r"(?:^|\s)Modstring=([^\s]+)")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sequence_identity(seq: str) -> str:
    """Only I/L mass equivalence is collapsed; B/O/X/Z are never substituted."""
    return seq.strip().upper().replace("I", "L")


def set_bytes(values: Iterable[str]) -> bytes:
    ordered = sorted(set(values))
    return (("\n".join(ordered) + "\n") if ordered else "").encode("utf-8")


def sequence_set_sha256(values: Iterable[str]) -> str:
    return hashlib.sha256(set_bytes(values)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def split_for_sequence(seq: str) -> str:
    digest = hashlib.sha256((SPLIT_SALT + sequence_identity(seq)).encode("utf-8")).digest()
    return "val" if int.from_bytes(digest, "big") % 10 == 0 else "train"


def read_sequence_columns(paths: Iterable[Path]) -> tuple[set[str], list[dict]]:
    """Read only seq, never exclusion data labels/predictions/observed features."""
    import pandas as pd
    sequences, sources = set(), []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        before = path.stat()
        local, rows, blanks = set(), 0, 0
        for chunk in pd.read_csv(path, usecols=["seq"], dtype={"seq": str},
                                 keep_default_na=False, chunksize=100_000):
            values = chunk["seq"].str.strip().str.upper()
            rows += len(values)
            blanks += int((values == "").sum())
            local.update(value for value in values.unique() if value)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Sequence source changed: {path}")
        if blanks or not local:
            raise ValueError(f"Blank or empty exclusion sequences: {path}")
        sequences.update(local)
        sources.append({"path": str(path), "columns_read": ["seq"], "rows": rows,
                        "blank_sequences": blanks, "unique_sequences": len(local),
                        "sequence_set_sha256": sequence_set_sha256(local),
                        "size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns})
    if not sources:
        raise ValueError("At least one exclusion source is required")
    return sequences, sources


def neutral_mass(sequence: str) -> float:
    return math.fsum(RESIDUE_MASS[aa] for aa in sequence) + WATER


def precursor_mz(sequence: str, charge: int) -> float:
    if charge <= 0:
        raise ValueError("Charge must be positive")
    return (neutral_mass(sequence) + charge * PROTON) / charge


def fragment_slot(length: int, ion: str, ordinal: int, charge: int) -> tuple[int, int]:
    if ion not in ("b", "y") or not 1 <= ordinal < length or charge not in (1, 2):
        raise ValueError("Fragment outside the four-channel target")
    return (ordinal - 1 if ion == "b" else length - ordinal - 1,
            (0 if ion == "b" else 2) + charge - 1)


def fragment_mz(sequence: str, ion: str, ordinal: int, charge: int) -> float:
    fragment_slot(len(sequence), ion, ordinal, charge)
    residues = sequence[:ordinal] if ion == "b" else sequence[-ordinal:]
    mass = math.fsum(RESIDUE_MASS[aa] for aa in residues)
    return (mass + (WATER if ion == "y" else 0) + charge * PROTON) / charge


@dataclass(frozen=True)
class Annotation:
    category: str
    ion: str | None = None
    ordinal: int | None = None
    charge: int | None = None
    reported_ppm: float | None = None  # parsed for grammar only, NEVER used for QA


def parse_annotation(text: str) -> Annotation:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        text = text[1:-1].strip()
    match = STANDARD_ION.fullmatch(text)
    if match:
        charge = int(match[3] or 1)
        ppm = float(match[4]) if match[4] is not None else None
        if ppm is not None and not math.isfinite(ppm):
            return Annotation("unresolved")
        return Annotation("standard" if charge in (1, 2) else "out_of_scope_charge",
                          match[1], int(match[2]), charge, ppm)
    match = LOSS_ION.fullmatch(text)
    if match:
        return Annotation("neutral_loss", match[1], int(match[2]),
                          int(match[4] or match[5] or 1))
    return Annotation("missing" if not text else "unresolved")


@dataclass
class MSPRecord:
    name: str
    source_record: int
    source_line: int
    mods: int | None = None
    modstring: str | None = None
    declared_peaks: int | None = None
    peaks: list[str] = field(default_factory=list)
    header_error: bool = False


class MSPStream:
    """Read and hash original bytes together, retaining only one spectrum."""
    def __init__(self, path: Path):
        self.path = Path(path)
        self.sha256 = None
        self.lines_read = 0
        self.records_read = 0
        self.before = None

    def __iter__(self):
        if self.before is not None:
            raise RuntimeError("An MSPStream can be consumed only once")
        self.before = self.path.stat()
        digest, record, in_peaks = hashlib.sha256(), None, False
        with self.path.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                digest.update(raw)
                self.lines_read = line_number
                line = raw.decode("utf-8", errors="replace").strip()
                if line.startswith("Name:"):
                    if record is not None:
                        yield record
                    self.records_read += 1
                    record = MSPRecord(line[5:].strip(), self.records_read, line_number)
                    in_peaks = False
                elif record is not None and line.startswith("Comment:") and not in_peaks:
                    matches = list(MODS.finditer(line))
                    if len(matches) != 1 or record.mods is not None:
                        record.header_error = True
                    else:
                        record.mods = int(matches[0][1])
                    modstrings = list(MODSTRING.finditer(line))
                    if len(modstrings) > 1:
                        record.header_error = True
                    elif modstrings:
                        record.modstring = modstrings[0][1]
                elif record is not None and line.lower().startswith("num peaks:"):
                    if record.declared_peaks is not None:
                        record.header_error = True
                    try:
                        record.declared_peaks = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        record.header_error = True
                    in_peaks = True
                elif record is not None and in_peaks and line:
                    record.peaks.append(line)
            if record is not None:
                yield record
        after = self.path.stat()
        if (self.before.st_size, self.before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("MSP changed during the only streaming pass")
        self.sha256 = digest.hexdigest()


@dataclass
class SpectrumQA:
    intensities: list[list[float]]
    mask: list[list[bool]]
    reason: str | None
    counts: Counter = field(default_factory=Counter)
    ppm: list[float] = field(default_factory=list)
    raw_valid_intensities: list[float] = field(default_factory=list)


def qa_spectrum(sequence: str, record: MSPRecord) -> SpectrumQA:
    """Ambiguous identified slots remain masked even if another peak names them.

    Unknown/missing annotation, invalid numeric peak or unlocalizable ambiguity
    rejects the spectrum. Explicit neutral losses/charges >2 are outside scope.
    A mass-failing standard assignment masks its slot permanently. Duplicated
    valid, unambiguous assignments use max intensity before spectrum normalization.
    """
    result = SpectrumQA([[0.0] * 4 for _ in range(len(sequence) - 1)],
                        [[False] * 4 for _ in range(len(sequence) - 1)], None)
    if record.header_error or record.declared_peaks != len(record.peaks):
        result.reason = "malformed_header_or_peak_count"
        return result
    valid, blocked = {}, set()
    rejection_reasons = set()
    for peak in record.peaks:
        result.counts["peak_lines"] += 1
        parts = peak.split(None, 2)
        try:
            observed, intensity = float(parts[0]), float(parts[1])
            if not math.isfinite(observed) or not math.isfinite(intensity) or observed <= 0 or intensity <= 0:
                raise ValueError("Nonpositive/nonfinite peak")
        except (ValueError, IndexError):
            result.counts["invalid_numeric_peaks"] += 1
            rejection_reasons.add("invalid_numeric_peak")
            continue
        annotation = parts[2].strip() if len(parts) > 2 else ""
        if len(annotation) >= 2 and annotation[0] == annotation[-1] == '"':
            annotation = annotation[1:-1].strip()
        components = re.split(r"[,;|]", annotation)
        ambiguous = len(components) != 1
        if ambiguous:
            result.counts["multiple_annotation_peaks"] += 1
        for component in components:
            parsed = parse_annotation(component)
            result.counts["annotation_" + parsed.category] += 1
            if parsed.category in ("unresolved", "missing"):
                rejection_reasons.add("unlocalizable_or_missing_annotation")
                continue
            if parsed.category in ("neutral_loss", "out_of_scope_charge"):
                continue
            try:
                slot = fragment_slot(len(sequence), parsed.ion, parsed.ordinal, parsed.charge)
            except ValueError:
                rejection_reasons.add("invalid_standard_ion_ordinal")
                continue
            if ambiguous:
                blocked.add(slot)
                result.counts["ambiguous_slot_mentions"] += 1
                continue
            theoretical = fragment_mz(sequence, parsed.ion, parsed.ordinal, parsed.charge)
            ppm = (observed - theoretical) / theoretical * 1e6
            result.ppm.append(ppm)
            result.counts["unambiguous_standard_peaks_evaluated"] += 1
            if abs(ppm) > MAX_PPM:
                blocked.add(slot)
                result.counts["mass_outside_50ppm"] += 1
                continue
            result.counts["mass_within_50ppm"] += 1
            if slot in valid:
                result.counts["duplicate_valid_slot_assignments"] += 1
            valid[slot] = max(valid.get(slot, 0.0), intensity)
    result.counts["blocked_slots"] = len(blocked)
    if rejection_reasons:
        result.reason = "+".join(sorted(rejection_reasons))
        return result
    valid = {slot: intensity for slot, intensity in valid.items() if slot not in blocked}
    if not valid:
        result.reason = "no_valid_unambiguous_slot"
        return result
    maximum = max(valid.values())
    for (bond, channel), intensity in sorted(valid.items()):
        result.intensities[bond][channel] = intensity / maximum
        result.mask[bond][channel] = True
        result.raw_valid_intensities.append(intensity)
    result.counts["valid_slots_after_masking"] = len(valid)
    return result


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    quantiles = {}
    for name, probability in (("min", 0), ("median", .5), ("p95", .95), ("p99", .99), ("max", 1)):
        index = (len(ordered) - 1) * probability
        low, high = math.floor(index), math.ceil(index)
        quantiles[name] = ordered[low] + (ordered[high] - ordered[low]) * (index - low)
    return {"count": len(values), "mean": math.fsum(values) / len(values), **quantiles}


def _sequence_summary(sequences: set[str]) -> dict:
    identities = {sequence_identity(seq) for seq in sequences}
    return {"unique_sequences": len(sequences), "sequence_set_sha256": sequence_set_sha256(sequences),
            "identity_set_sha256": sequence_set_sha256(identities),
            "sequences": sorted(sequences), "sequence_identities": sorted(identities)}


def build_dataset(msp_path: Path, output_dir: Path, nist_sequences: set[str], mipd_sequences: set[str],
                  *, exclusion_sources: dict | None = None, certification: dict | None = None,
                  expected_msp_sha256: str | None = None, source_files: Iterable[Path] = (),
                  invocation: dict | None = None) -> dict:
    """Create a NEW artifact directory. CLI supplies certified real exclusions.

    Generic small sets are supported only for unit fixtures; their manifest is
    explicitly uncertified. An incomplete/failed build never emits manifest.json.
    """
    msp_path, output_dir = Path(msp_path).resolve(), Path(output_dir).resolve()
    if not nist_sequences or not mipd_sequences:
        raise ValueError("Both exclusion sets must be nonempty")
    if any(seq != seq.strip().upper() or not seq for seq in nist_sequences | mipd_sequences):
        raise ValueError("Exclusion sequences must already be canonical uppercase nonempty strings")
    output_dir.mkdir(parents=True, exist_ok=False)
    started, monotonic_start = utc_now(), time.monotonic()
    status_path = output_dir / "build_status.json"
    atomic_json(status_path, {"status": "running", "started_at": started})
    atomic_json(output_dir / "build_command.json", invocation or {"api": "build_dataset", "fixture": True})
    try:
        code_hashes = {str(Path(path).resolve()): file_sha256(Path(path)) for path in source_files}
        nist_ids = {sequence_identity(seq) for seq in nist_sequences}
        mipd_ids = {sequence_identity(seq) for seq in mipd_sequences}
        excluded_ids = nist_ids | mipd_ids
        identity_path = output_dir / "excluded_sequence_identities.txt"
        identity_path.write_bytes(set_bytes(excluded_ids))
        counts, rejections, qa_counts = Counter(), Counter(), Counter()
        accepted_channels, charges, lengths = Counter(), Counter(), Counter()
        candidates, accepted_keys = set(), set()
        split_sequences = {"train": set(), "val": set()}
        split_rows, split_slots = Counter(), Counter()
        all_ppm, accepted_raw, accepted_normalized = [], [], []
        accepted_possible_slots = 0
        stream = MSPStream(msp_path)
        with (output_dir / "train.csv").open("w", encoding="utf-8", newline="") as train_file, \
             (output_dir / "val.csv").open("w", encoding="utf-8", newline="") as val_file:
            writers = {"train": csv.DictWriter(train_file, CSV_COLUMNS, lineterminator="\n"),
                       "val": csv.DictWriter(val_file, CSV_COLUMNS, lineterminator="\n")}
            for writer in writers.values():
                writer.writeheader()
            for record in stream:
                counts["all_msp_records"] += 1
                try:
                    sequence, charge_text = record.name.rsplit("/", 1)
                    sequence, charge = sequence.strip().upper(), int(charge_text)
                except ValueError:
                    rejections["invalid_name_or_charge"] += 1
                    continue
                if sequence in nist_sequences:
                    rejections["excluded_nist_exact_sequence"] += 1
                    continue
                if sequence in mipd_sequences:
                    rejections["excluded_mipd_exact_sequence"] += 1
                    continue
                if not sequence or not set(sequence) <= RESIDUE_MASS.keys():
                    rejections["nonstandard_residue"] += 1
                    continue
                if not 7 <= len(sequence) <= 30:
                    rejections["length_outside_7_30"] += 1
                    continue
                if charge not in range(2, 7):
                    rejections["charge_outside_2_6"] += 1
                    continue
                if record.mods != 0:
                    rejections["modified_or_missing_mods"] += 1
                    continue
                counts["standard_unmodified_candidates_before_il_exclusion"] += 1
                identity = sequence_identity(sequence)
                if identity in nist_ids:
                    rejections["excluded_nist_additional_il_identity"] += 1
                    continue
                if identity in mipd_ids:
                    rejections["excluded_mipd_additional_il_identity"] += 1
                    continue
                counts["candidates_after_all_exclusions"] += 1
                candidates.add(sequence)
                # Mods=0 Modstring must agree if supplied; no Parent is consumed.
                if record.modstring is not None and record.modstring != f"{sequence}///{charge}":
                    rejections["inconsistent_unmodified_modstring"] += 1
                    continue
                counts["candidate_spectra_fragment_qa"] += 1
                qa = qa_spectrum(sequence, record)
                qa_counts.update(qa.counts)
                all_ppm.extend(qa.ppm)
                if qa.reason is not None:
                    rejections["fragment_qa:" + qa.reason] += 1
                    continue
                counts["qa_passing_spectra_before_dedup"] += 1
                key = (sequence, charge)
                if key in accepted_keys:
                    rejections["later_qa_passing_duplicate_seq_charge"] += 1
                    continue
                accepted_keys.add(key)
                split = split_for_sequence(sequence)
                split_sequences[split].add(sequence)
                split_rows[split] += 1
                counts["accepted_spectra"] += 1
                charges[str(charge)] += 1
                lengths[str(len(sequence))] += 1
                accepted_possible_slots += (len(sequence) - 1) * 4
                accepted_raw.extend(qa.raw_valid_intensities)
                for bond, row in enumerate(qa.mask):
                    for channel, valid in enumerate(row):
                        if valid:
                            accepted_channels[CHANNEL_ORDER[channel]] += 1
                            split_slots[split] += 1
                            accepted_normalized.append(qa.intensities[bond][channel])
                row = dict.fromkeys(CSV_COLUMNS, 0)
                row.update(name=f"ion4:{record.source_record}:line{record.source_line}:{record.name}",
                           seq=sequence, charge=charge, pep_mass=precursor_mz(sequence, charge),
                           tb=len(sequence) - 1, true_multi=";".join(["0"] * (len(sequence) - 1)),
                           ion_intensities_json=json.dumps(qa.intensities, separators=(",", ":"), allow_nan=False),
                           ion_mask_json=json.dumps(qa.mask, separators=(",", ":")))
                writers[split].writerow(row)
        if expected_msp_sha256 is not None and stream.sha256 != expected_msp_sha256:
            raise RuntimeError(f"MSP SHA mismatch: {stream.sha256}")
        if not split_sequences["train"] or not split_sequences["val"]:
            raise RuntimeError("Fixed sequence split produced an empty train or validation set")
        train_ids = {sequence_identity(seq) for seq in split_sequences["train"]}
        val_ids = {sequence_identity(seq) for seq in split_sequences["val"]}
        if train_ids & val_ids or (train_ids | val_ids) & excluded_ids:
            raise RuntimeError("Sequence identity isolation failed")
        if counts["all_msp_records"] != sum(rejections.values()) + counts["accepted_spectra"]:
            raise RuntimeError("Record accounting is not exhaustive")
        if any(file_sha256(Path(path)) != digest for path, digest in code_hashes.items()):
            raise RuntimeError("Builder source changed during construction")
        sources = exclusion_sources or {}
        exclusion = {"identity_rule": IDENTITY_RULE, "set_hash_rule": "UTF-8 sorted unique lines with trailing LF; empty set uses empty bytes",
                     "certification": certification or {"certified_full_exclusion_sets": False, "scope": "synthetic/unit fixture"},
                     "nist": {"sources": sources.get("nist", []), "unique_sequences": len(nist_sequences),
                              "sequence_set_sha256": sequence_set_sha256(nist_sequences),
                              "identity_set_sha256": sequence_set_sha256(nist_ids)},
                     "mipd": {"sources": sources.get("mipd", []), "unique_sequences": len(mipd_sequences),
                              "sequences": sorted(mipd_sequences), "sequence_set_sha256": sequence_set_sha256(mipd_sequences),
                              "identity_set_sha256": sequence_set_sha256(mipd_ids)},
                     "identity_file": {"path": identity_path.name, "sha256": file_sha256(identity_path), "rows": len(excluded_ids)}}
        outputs = {}
        for split in ("train", "val"):
            path = output_dir / f"{split}.csv"
            outputs[split] = {"csv": path.name, "sha256": file_sha256(path), "rows": split_rows[split],
                              "valid_slots": split_slots[split], **_sequence_summary(split_sequences[split])}
        manifest = {
            "schema_version": 1, "status": "completed", "dataset_kind": "ion_intensity_external",
            "purpose": "library-annotation masked relative-intensity auxiliary pretraining",
            "started_at": started, "completed_at": utc_now(), "elapsed_seconds": time.monotonic() - monotonic_start,
            "source": {"path": str(msp_path), "sha256": stream.sha256, "size_bytes": stream.before.st_size,
                       "mtime_ns": stream.before.st_mtime_ns, "lines_read": stream.lines_read, "streaming_passes": 1,
                       "source_unchanged_during_read": True, "expected_sha256": expected_msp_sha256},
            "source_sha256": code_hashes, "channel_order": CHANNEL_ORDER, "exclusion": exclusion, "outputs": outputs,
            "split": {"unit": "I/L-normalized sequence identity", "salt": SPLIT_SALT,
                      "rule": "int.from_bytes(sha256(UTF8(salt+identity)), 'big') % 10 == 0 -> val, else train",
                      "target_train_fraction": .9, "target_val_fraction": .1,
                      "sequence_overlap": 0, "identity_overlap": 0, "excluded_identity_overlap": 0},
            "protocol": {
                "sequence_alphabet": "ACDEFGHIKLMNPQRSTVWY", "length_range": [7, 30], "precursor_charge_range": [2, 6],
                "modifications": "explicit Mods=0; optional Modstring must agree with sequence and charge",
                "reported_charge_provenance": "Name suffix, consistency with Modstring when present; not independently measured",
                "precursor_mz": "(sum standard residue masses + H2O + charge*proton) / charge; Parent never consumed",
                "nce": {"known": False, "csv_placeholder": 0, "required_input_mask": False,
                        "filename_not_used_to_infer_energy": True},
                "observed_features": "intensity, scan_num, rt all 0 placeholders and must be masked",
                "ion_alignment": "bN -> row N-1; yN -> row L-N-1; columns b1,b2,y1,y2",
                "annotation": "single standard b/yN optional ^charge optional /signed_errorppm; ^ defines charge, /ppm is never used for QA",
                "mass_ppm": MAX_PPM, "mass_constants": {"residue_mass": RESIDUE_MASS, "proton": PROTON, "water": WATER},
                "ambiguity_policy": "multi annotations block all explicitly implicated standard slots; unlocalizable/missing annotations reject whole spectrum",
                "out_of_scope_policy": "explicit H2O/NH3 losses and fragment charges >2 are ignored, never positive/negative labels; unresolved isotope/adduct notation rejects spectrum",
                "mass_failure_policy": "mask mass-failing slot permanently, even if another peak names the same slot",
                "numeric_peak_policy": "all peak m/z and intensities must be finite and positive, otherwise reject spectrum",
                "duplicate_slot_policy": "max raw intensity among valid assignments, after ambiguity masks",
                "normalization": "divide each valid slot by maximum valid retained slot intensity per spectrum",
                "missing_channel_policy": "intensity 0 and mask false; never a negative label",
                "deduplication": "exact (sequence, charge), first QA-passing spectrum in source order; all candidates QA before dedup",
                "minimum_valid_slots": 1, "true_multi": "all-zero compatibility placeholder, NOT classification labels; never train/evaluate on it",
            },
            "statistics": {"counts": dict(counts), "rejections": dict(rejections), "candidate_unique_sequences": len(candidates),
                           "qa_peak_counts_all_candidates": dict(qa_counts), "precursor_charge_counts_accepted": dict(charges),
                           "length_counts_accepted": dict(lengths), "accepted_possible_slots": accepted_possible_slots,
                           "accepted_valid_slots": len(accepted_normalized),
                           "accepted_channel_valid_slots": {key: accepted_channels[key] for key in CHANNEL_ORDER},
                           "accepted_channel_coverage": {key: accepted_channels[key] / (accepted_possible_slots / 4) for key in CHANNEL_ORDER},
                           "all_candidate_evaluated_signed_ppm": _distribution(all_ppm),
                           "all_candidate_evaluated_absolute_ppm": _distribution([abs(value) for value in all_ppm]),
                           "accepted_raw_valid_slot_intensity": _distribution(accepted_raw),
                           "accepted_normalized_valid_slot_intensity": _distribution(accepted_normalized)},
            "limitations": ["Annotation-conditioned auxiliary targets do not establish complete spectrum or absent-fragment validity.",
                            "Parent convention and per-spectrum NCE remain unknown and are not validated or imputed.",
                            "Relative intensities are normalized selected annotated slots, not total ion-current-normalized spectra.",
                            "Exact and I/L sequence isolation do not establish homology separation or independent source libraries.",
                            "Standard natural-residue library supervision does not validate D stereochemistry or B/O/X/Z chemistry."],
        }
        atomic_json(output_dir / "manifest.json", manifest)
        atomic_json(status_path, {"status": "completed", "started_at": started, "completed_at": utc_now(),
                                  "manifest": "manifest.json", "manifest_sha256": file_sha256(output_dir / "manifest.json")})
        return manifest
    except BaseException as error:
        atomic_json(status_path, {"status": "failed", "started_at": started, "failed_at": utc_now(),
                                  "error": f"{type(error).__name__}: {error}"})
        raise
