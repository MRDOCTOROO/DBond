"""Synthetic ion4 tests only. Optional --device cuda exercises CUDA FP16 as well."""
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.theory_features import RESIDUE_MONO_MASS
from models.utils import ModelConfig, build_model_config
from pretraining.intensity_model import (
    CHANNELS, PROTON_MASS, WATER_MASS, IonIntensityDataset, IonIntensityModel,
    collate_ion_intensity, make_pretraining_config, masked_l1, masked_l1_per_spectrum,
    parse_ion_targets,
)
import pretrain_ion_intensity as driver


DEVICE = "cpu"


def tiny_config(global_node=True):
    config = make_pretraining_config()
    config.update_config({"hidden_dim": 16, "aa_embedding_dim": 8,
                          "position_embedding_dim": 4, "physicochemical_dim": 4,
                          "edge_embedding_dim": 4, "distance_embedding_dim": 4,
                          "num_gcn_layers": 1, "num_gat_layers": 1,
                          "dropout": 0.0, "use_global_node": global_node})
    return config


def write_csv(path, specs):
    rows = []
    for sequence, charge in specs:
        count = len(sequence) - 1
        values = [[(b + c + 1) / (count + 3) for c in range(4)] for b in range(count)]
        mask = [[True, b % 2 == 0, True, b % 3 == 0] for b in range(count)]
        rows.append({"seq": sequence, "charge": charge,
                     "ion_intensities_json": json.dumps(values), "ion_mask_json": json.dumps(mask),
                     # Deliberately invalid production labels/observations must not be read.
                     "true_multi": "NOT_A_LABEL", "soft_multi": "NOT_A_TARGET", "nce": "NaN",
                     "pep_mass": -999.0, "intensity": "NaN", "scan_num": "NaN", "rt": "NaN"})
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(directory, train, val):
    identities = directory / "excluded_sequence_identities.txt"
    identities.write_text("YYYY\nYYYYY\n", encoding="utf-8")
    result = {"status": "completed", "dataset_kind": "ion_intensity_external", "channel_order": CHANNELS,
              "exclusion": {"identity_rule": "uppercase_base_sequence_I_to_L",
                            "certification": {"certified_full_exclusion_sets": True},
                            "nist": {"unique_sequences": 1, "sequence_set_sha256": driver.sequence_set_sha256(["YYYY"])},
                            "mipd": {"unique_sequences": 1, "sequences": ["YYYYY"],
                                     "sequence_set_sha256": driver.sequence_set_sha256(["YYYYY"]),
                                     "identity_set_sha256": driver.sequence_set_sha256(["YYYYY"])},
                            "identity_file": {"path": identities.name,
                                              "sha256": driver.sha256_file(identities), "rows": 2}},
              "outputs": {}}
    for name, dataset in (("train", train), ("val", val)):
        sequences = sorted(set(dataset.data["seq"]))
        ids = sorted({seq.replace("I", "L") for seq in sequences})
        path = Path(dataset.csv_path)
        result["outputs"][name] = {
            "csv": path.name, "sha256": driver.sha256_file(path), "rows": len(dataset),
            "unique_sequences": len(sequences), "sequence_set_sha256": driver.sequence_set_sha256(sequences),
            "identity_set_sha256": driver.sequence_set_sha256(ids),
            "sequences": sequences, "sequence_identities": ids,
        }
    path = directory / "manifest.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return path


class IonIntensityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.exclusion_patch = mock.patch.object(driver, "EXPECTED_EXCLUSION_SETS", {
            "nist": (1, driver.sequence_set_sha256(["YYYY"])),
            "mipd": (1, driver.sequence_set_sha256(["YYYYY"])),
        })
        self.exclusion_patch.start()
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.train_path, self.val_path = self.directory / "train.csv", self.directory / "val.csv"
        write_csv(self.train_path, [("AGK", 2), ("PEPTIDE", 4), ("AGK", 4)])
        write_csv(self.val_path, [("ACDK", 2), ("VAGWQ", 3)])
        self.config = tiny_config()
        self.train = IonIntensityDataset(self.train_path, self.config)
        self.val = IonIntensityDataset(self.val_path, self.config, "val")

    def tearDown(self):
        self.exclusion_patch.stop()
        self.temp.cleanup()

    def batch(self, indices=(0, 1), dataset=None):
        dataset = dataset or self.train
        return driver.move_batch(collate_ion_intensity([dataset[i] for i in indices]), torch.device(DEVICE))

    def test_real_control_geometry_and_input_contract(self):
        control_path = Path(__file__).resolve().parents[1] / "config/pre_synthesis_theory_film_cond3_control.yaml"
        control = build_model_config(yaml.safe_load(control_path.read_text(encoding="utf-8")))
        config = make_pretraining_config()
        allowed = {"state_feature_mask", "env_feature_mask", "use_env_features", "use_condition_film"}
        for key, value in control.to_dict().items():
            if key not in allowed:
                self.assertEqual(config.to_dict()[key], value, key)
        self.assertEqual(config.state_feature_mask, [True, True, False])
        self.assertEqual(config.env_feature_mask, [False, False])
        self.assertFalse(config.use_env_features)
        self.assertFalse(config.use_condition_film)
        for index, charge in ((0, 2), (2, 4)):
            sample = self.train[index]
            neutral = sum(RESIDUE_MONO_MASS[aa] for aa in "AGK") + WATER_MASS
            self.assertAlmostEqual(sample["pep_mass"], (neutral + charge * PROTON_MASS) / charge)
            self.assertEqual(sample["nce"], 0.0)
            self.assertEqual(sample["intensity"], 0.0)
            self.assertEqual(sample["env_vars"].tolist(), [0.0, 0.0])
        batch = self.batch()
        self.assertNotIn("labels", batch)
        self.assertNotIn("soft_labels", batch)
        self.assertFalse(batch["ion_mask"][0, 2:].any())

    def test_masked_l1_equal_spectrum_weight_and_missing_gradients(self):
        pred = torch.zeros(2, 3, 4, requires_grad=True)
        targets = torch.full_like(pred, float("nan"))
        mask = torch.zeros_like(pred, dtype=torch.bool)
        mask[0, 0, 0] = True
        mask[1, :2] = True
        targets[0, 0, 0] = 0.2
        targets[1, :2] = 0.6
        losses = masked_l1_per_spectrum(pred, targets, mask)
        torch.testing.assert_close(losses, torch.tensor([0.2, 0.6]))
        loss = losses.mean()
        self.assertAlmostEqual(float(loss.detach()), 0.4, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertEqual(int(torch.count_nonzero(pred.grad[~mask])), 0)
        self.assertAlmostEqual(float(pred.grad[0, 0, 0]), -0.5)
        self.assertAlmostEqual(float(pred.grad[1, 0, 0]), -0.0625)
        with self.assertRaises(ValueError):
            masked_l1(pred, targets, torch.zeros_like(mask))
        values, parsed = parse_ion_targets('[[0.5, NaN, 0.8, 0.0]]', '[[1,0,1,0]]', 1)
        self.assertTrue(torch.isfinite(values).all())
        self.assertEqual(parsed.sum(), 2)
        with self.assertRaises(ValueError):
            parse_ion_targets('[[0,0,0,0]]', '[[0,0,0,0]]', 1)

    def test_forward_padding_batch_order_and_unknown_condition_invariance(self):
        for global_node in (False, True):
            config = tiny_config(global_node)
            dataset = IonIntensityDataset(self.train_path, config)
            model = IonIntensityModel(config).to(DEVICE).eval()
            batch = self.batch(dataset=dataset)
            with torch.no_grad():
                actual = model(batch)
                single = model(self.batch((0,), dataset=dataset))
                reversed_batch = model(self.batch((1, 0), dataset=dataset))
                changed = copy.deepcopy(batch)
                for key in ("nces", "intensities", "rts", "secondary_envs", "env_vars"):
                    changed[key].fill_(float("nan"))
                changed["state_vars"][:, 2] = float("nan")
                changed["labels"] = torch.full(actual.shape[:2], float("nan"), device=DEVICE)
                changed["soft_labels"] = torch.ones(actual.shape[:2], device=DEVICE)
                perturbed = model(changed)
            self.assertEqual(actual.shape, (2, 6, 4))
            self.assertTrue(torch.isfinite(actual).all())
            self.assertEqual(int(torch.count_nonzero(actual[0, 2:])), 0)
            torch.testing.assert_close(actual[0, :2], single[0], rtol=1e-4, atol=1e-6)
            torch.testing.assert_close(actual, reversed_batch.flip(0), rtol=1e-4, atol=1e-6)
            torch.testing.assert_close(actual, perturbed, rtol=0, atol=0)
            self.assertIsNone(model._bond_hidden)

    def test_real_backbone_and_ion_head_gradients_amp(self):
        model = IonIntensityModel(self.config).to(DEVICE).train()
        batch = self.batch()
        dtype = torch.float16 if DEVICE.startswith("cuda") else torch.bfloat16
        with torch.autocast(device_type=torch.device(DEVICE).type, dtype=dtype):
            output = model(batch)
            output.retain_grad()
            loss = masked_l1(output, batch["ion_targets"], batch["ion_mask"])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(output.grad).all())
        self.assertEqual(int(torch.count_nonzero(output.grad[~batch["ion_mask"]])), 0)
        for name, parameter in (("ion_head", model.ion_head.weight),
                                ("bond_hidden", model.backbone.bond_head[0].weight),
                                ("embedding", model.backbone.node_encoder.aa_embedding.weight)):
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0, name)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.backbone.gcn_layers.parameters()))
        self.assertIsNone(model.backbone.bond_head[3].weight.grad)

    def test_checkpoint_safe_serialization_strict_roundtrip(self):
        model = IonIntensityModel(self.config).to(DEVICE).eval()
        batch = self.batch()
        with torch.no_grad():
            expected = model(batch)
        payload = driver.checkpoint_payload(model, {"train_sequences": ["AGK", "PEPTIDE"]}, 3, 0.2, 0.4, 10)
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=True, map_location="cpu")
        clone = IonIntensityModel(ModelConfig(restored["model_config"])).to(DEVICE).eval()
        clone.backbone.load_state_dict(restored["model_state_dict"], strict=True)
        clone.ion_head.load_state_dict(restored["ion_head_state_dict"], strict=True)
        with torch.no_grad():
            torch.testing.assert_close(clone(batch), expected, rtol=0, atol=0)
        self.assertEqual(restored["schema_version"], 1)
        self.assertEqual(restored["channels"], CHANNELS)

    def test_manifest_identity_and_provenance_guards(self):
        path = write_manifest(self.directory, self.train, self.val)
        info = driver.validate_dataset_manifest(path, self.train_path, self.val_path, self.train, self.val)
        self.assertEqual(info["trained_position_count"], 7)
        self.assertEqual(info["trained_amino_acids"], sorted(set("AGKPEPTIDE")))
        self.assertEqual(info["dataset_manifest_canonical_sha256"], driver.canonical_sha256(info["dataset_manifest"]))
        manifest = json.loads(path.read_text())
        manifest["outputs"]["train"]["sha256"] = "0" * 64
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
            driver.validate_dataset_manifest(path, self.train_path, self.val_path, self.train, self.val)
        # I/L variants must also collide with the external exclusion identities.
        path = write_manifest(self.directory, self.train, self.val)
        exclusion_path = self.directory / "excluded_sequence_identities.txt"
        exclusion_path.write_text("PEPTLDE\nYYYYY\n", encoding="utf-8")
        manifest = json.loads(path.read_text())
        manifest["exclusion"]["identity_file"]["sha256"] = driver.sha256_file(exclusion_path)
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "intersects"):
            driver.validate_dataset_manifest(path, self.train_path, self.val_path, self.train, self.val)

    def test_driver_short_synthetic_run_status_history_and_best(self):
        manifest_path = write_manifest(self.directory, self.train, self.val)
        output = self.directory / "run"
        args = SimpleNamespace(train_csv=str(self.train_path), val_csv=str(self.val_path),
                               dataset_manifest=str(manifest_path), output_dir=str(output),
                               device=DEVICE, num_workers=0)
        protocol = dict(driver.PROTOCOL, max_epochs=3, warmup_epochs=1, batch_size=2)
        source = {"git_revision": "synthetic_fixture_only", "source_sha256": {
            "graph_transform/pretraining/intensity_model.py": driver.sha256_file(
                driver.REPO_ROOT / "graph_transform/pretraining/intensity_model.py")}}
        with mock.patch.object(driver, "PROTOCOL", protocol), \
                mock.patch.object(driver, "make_pretraining_config", return_value=self.config), \
                mock.patch.object(driver, "source_identity", return_value=source):
            status = driver.run(args)
        self.assertEqual(status["status"], "completed")
        history = json.loads((output / "history.json").read_text())
        self.assertEqual(len(history["epochs"]), 3)
        best = min(row["val"]["masked_l1"] for row in history["epochs"])
        self.assertEqual(status["val_loss"], best)
        saved = torch.load(output / "best_ion_intensity.pt", weights_only=True, map_location="cpu")
        self.assertEqual(saved["val_loss"], best)
        self.assertEqual(saved["initial_val_loss"], history["initial_validation"]["masked_l1"])
        self.assertIsNotNone(history["constant_baseline"]["val_masked_l1"])
        self.assertEqual(status["train_spectra_seen"], 9)
        self.assertEqual(status["optimizer_updates"], 6)
        with self.assertRaises(FileExistsError):
            driver.run(args)
        args.output_dir = str(self.directory / "failed")
        args.train_csv = str(self.directory / "absent.csv")
        with self.assertRaises(FileNotFoundError):
            driver.run(args)
        failed = json.loads((Path(args.output_dir) / "status.json").read_text())
        self.assertEqual(failed["status"], "failed")
        self.assertIn("FileNotFoundError", failed["error"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args, unittest_args = parser.parse_known_args()
    DEVICE = args.device
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    unittest.main(argv=[sys.argv[0]] + unittest_args)
