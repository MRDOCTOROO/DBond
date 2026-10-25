"""Synthetic, CPU-only tests for audited ion4 initialization; no training/data scan."""
from argparse import Namespace
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.graph_transformer import GraphTransformer
from models.utils import ModelConfig, build_model_config
from pretraining.intensity_model import IonIntensityModel, make_pretraining_config
from pretraining import transfer
from pretrain_ion_intensity import checkpoint_payload
from train_graph_model import load_config, save_checkpoint_with_selection
from training.checkpoint_selection import CheckpointSelection


def sha(data):
    return hashlib.sha256(data).hexdigest()


def seqsha(sequences):
    return sha(("\n".join(sorted(set(sequences))) + "\n").encode())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def snapshot(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


class PretrainedTransferTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.source_config = make_pretraining_config()
        self.source_config.hidden_dim = 16
        self.source_config.alphabet = "#ZYXWVTSRQOPNMLKIHGFEDCBA"  # Different semantic row order.
        self.target_config = ModelConfig(self.source_config.to_dict())
        self.target_config.alphabet = "#ABCDEFGHIKLMNOPQRSTVWXYZ"
        self.target_config.use_env_features = True
        self.target_config.env_feature_mask = [True, False]
        self.target_config.use_condition_film = True
        torch.manual_seed(17)
        self.source = IonIntensityModel(self.source_config)
        with torch.no_grad():
            for index, parameter in enumerate(self.source.backbone.parameters()):
                parameter.add_(0.2 + index * 0.001)
        self.target = GraphTransformer(self.target_config)
        self.before = snapshot(self.target)
        self.source_hashes = {}
        for relative in transfer.SOURCE_FILES:
            file = self.repo / relative
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(relative + "\n", encoding="utf-8")
            self.source_hashes[relative] = sha(file.read_bytes())
        # This is a synthetic identity fixture, not the production exclusion corpus.
        targets = {"ACDEFGL"}
        for index in range(512):
            number, chars = index, []
            for _ in range(5):
                chars.append("ACDE"[number % 4])
                number //= 4
            targets.add("B" + "".join(chars))
        self.target_sequences = sorted(targets)
        self.assertEqual(len(targets), 513)
        pinned = dict(transfer.EXPECTED_EXCLUSION_SETS, mipd=(513, seqsha(targets)))
        fixture_pins = patch.object(transfer, "EXPECTED_EXCLUSION_SETS", pinned)
        fixture_pins.start()
        self.addCleanup(fixture_pins.stop)
        self.target_paths = {}
        for role, sequences in (("train", self.target_sequences[:410]), ("test", self.target_sequences[410:])):
            file = self.repo / (role + ".csv")
            file.write_text("seq,unused_label\n" + "".join(s + ",do-not-read\n" for s in sequences), encoding="utf-8")
            self.target_paths[role] = str(file)
        self.external = {"train": ["ACDEFGHIKLMNPQRSTV", "VVVAAAA"], "val": ["WYWWW"]}
        self.checkpoint_path = self.repo / "ion4.pt"
        self.rebuild_checkpoint()

    def rebuild_checkpoint(self):
        identities = sorted(transfer._identities(self.target_sequences))
        identity_file = self.repo / "excluded_sequence_identities.txt"
        identity_file.write_text("\n".join(identities) + "\n", encoding="utf-8")
        exclusion = {
            "identity_rule": "uppercase_base_sequence_I_to_L",
            "certification": {"certified_full_exclusion_sets": True},
            "mipd": {"sequences": self.target_sequences, "unique_sequences": 513,
                     "sequence_set_sha256": seqsha(self.target_sequences),
                     "identity_set_sha256": seqsha(identities)},
            "nist": {"unique_sequences": 238178,
                     "sequence_set_sha256": "aad4fe6b2cca66bdaab09fca6aad49db5e3923bb27aa9bd9f8de4c86de596e84",
                     "identity_set_sha256": "1" * 64},
            "identity_file": {"path": identity_file.name, "sha256": sha(identity_file.read_bytes()), "rows": len(identities)},
        }
        outputs, inputs = {}, {}
        for split, sequences in self.external.items():
            sequences = sorted(set(sequences))
            self.external[split] = sequences
            file = self.repo / ("external_" + split + ".csv")
            file.write_text("seq\n" + "\n".join(sequences) + "\n", encoding="utf-8")
            inputs[split + "_csv"] = sha(file.read_bytes())
            outputs[split] = {"csv": file.name, "sha256": sha(file.read_bytes()), "rows": len(sequences),
                              "sequences": sequences, "unique_sequences": len(sequences),
                              "sequence_set_sha256": seqsha(sequences),
                              "sequence_identities": sorted(transfer._identities(sequences)),
                              "identity_set_sha256": seqsha(transfer._identities(sequences))}
        manifest = {"status": "completed", "dataset_kind": "ion_intensity_external",
                    "channel_order": ["b1", "b2", "y1", "y2"], "exclusion": exclusion, "outputs": outputs}
        manifest_file = self.repo / "dataset_manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        metadata = {
            "source_sha256": self.source_hashes, "input_sha256": inputs,
            "dataset_manifest_path": str(manifest_file),
            "dataset_manifest_sha256": sha(manifest_file.read_bytes()),
            "dataset_manifest_canonical_sha256": sha(canonical(manifest)),
            "dataset_manifest": manifest, "exclusion_identity": exclusion,
            "train_sequences": self.external["train"], "val_sequences": self.external["val"],
            "train_sequences_sha256": seqsha(self.external["train"]),
            "val_sequences_sha256": seqsha(self.external["val"]),
            "trained_amino_acids": sorted(set("".join(self.external["train"]))),
            "trained_position_count": max(map(len, self.external["train"])),
        }
        # Use the actual driver's serializer, preventing schema drift between modules.
        self.payload = checkpoint_payload(self.source, metadata, 1, 0.1, 0.2, 1)
        self.rewrite_checkpoint()

    def rewrite_checkpoint(self):
        torch.save(self.payload, self.checkpoint_path)
        self.training = {"pretrained_transfer_scope": "encoder", "pretrained_backbone_path": str(self.checkpoint_path),
                         "pretrained_backbone_sha256": sha(self.checkpoint_path.read_bytes())}

    def load(self, model=None, training=None):
        return transfer.load_pretrained_backbone(model or self.target, training or self.training,
                                                self.target_paths, repo_root=self.repo)

    def assert_unchanged(self, model=None, expected=None):
        for key, tensor in (model or self.target).state_dict().items():
            self.assertTrue(torch.equal(tensor, (expected or self.before)[key]), key)

    def test_disabled_has_no_io_mutation_or_rng_change(self):
        rng = torch.get_rng_state().clone()
        self.assertIsNone(transfer.load_pretrained_backbone(self.target, {}, {}, repo_root="nonexistent"))
        self.assertIsNone(transfer.write_transfer_provenance(None, self.repo / "absent"))
        self.assertFalse((self.repo / "absent").exists())
        self.assert_unchanged()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for config in ({"pretrained_transfer_scope": "typo"},
                       {"pretrained_transfer_scope": "none", "pretrained_backbone_path": "bad"},
                       {"pretrained_transfer_scope": "encoder", "pretrained_backbone_path": "bad"}):
            with self.assertRaises(ValueError):
                transfer.validate_transfer_request(config)
        with self.assertRaisesRegex(ValueError, "resume"):
            transfer.validate_transfer_request(self.training, "old.pt")

    def test_allowlist_partial_rows_columns_and_rng(self):
        rng = torch.get_rng_state().clone()
        provenance = self.load()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        after, source = self.target.state_dict(), self.source.backbone.state_dict()
        for key in provenance["fully_copied_keys"]:
            self.assertTrue(torch.equal(after[key], source[key]), key)
        for key in provenance["retained_entire_keys"]:
            self.assertTrue(torch.equal(after[key], self.before[key]), key)
        for item in provenance["aa_row_mapping"]:
            self.assertTrue(torch.equal(after[transfer.AA_KEY][item["target_row"]], source[transfer.AA_KEY][item["source_row"]]))
        copied = {item["target_row"] for item in provenance["aa_row_mapping"]}
        for row in set(range(after[transfer.AA_KEY].shape[0])) - copied:
            self.assertTrue(torch.equal(after[transfer.AA_KEY][row], self.before[transfer.AA_KEY][row]))
        for residue in "BOXZWY":  # W/Y exist only in pretraining validation, never train.
            self.assertNotIn(residue, {item["residue"] for item in provenance["aa_row_mapping"]})
        count = provenance["copied_position_interval"][1]
        self.assertTrue(torch.equal(after[transfer.POSITION_KEY][:count], source[transfer.POSITION_KEY][:count]))
        self.assertTrue(torch.equal(after[transfer.POSITION_KEY][count:], self.before[transfer.POSITION_KEY][count:]))
        columns = provenance["copied_node_fusion_column_interval"][1]
        self.assertTrue(torch.equal(after[transfer.FUSION_KEY][:, :columns], source[transfer.FUSION_KEY][:, :columns]))
        self.assertTrue(torch.equal(after[transfer.FUSION_KEY][:, columns:], self.before[transfer.FUSION_KEY][:, columns:]))

    def test_same_checkpoint_scope_difference_is_exactly_context(self):
        other = GraphTransformer(self.target_config)
        other.load_state_dict(self.before)
        encoder_provenance = self.load()
        config = dict(self.training, pretrained_transfer_scope="bond_context")
        context_provenance = self.load(other, config)
        changed = {key for key, value in self.target.state_dict().items() if not torch.equal(value, other.state_dict()[key])}
        self.assertEqual(changed, transfer.CONTEXT_KEYS)
        self.assertEqual(set(context_provenance["fully_copied_keys"]) - set(encoder_provenance["fully_copied_keys"]), transfer.CONTEXT_KEYS)
        self.assertEqual(encoder_provenance["checkpoint_sha256"], context_provenance["checkpoint_sha256"])

    def test_rejections_are_atomic(self):
        original = deepcopy(self.payload)
        mutations = {
            "schema": lambda p: p.update(schema_version=2),
            "conflicting schema": lambda p: p.update(schema=2),
            "task": lambda p: p.update(task="classification"),
            "source hash": lambda p: p["source_sha256"].update({transfer.SOURCE_FILES[0]: "0" * 64}),
            "unknown config": lambda p: p["model_config"].update(unknown=1),
            "observed state": lambda p: p["model_config"].update(state_feature_mask=[True, True, True]),
            "environment": lambda p: p["model_config"].update(use_env_features=True),
            "unknown state": lambda p: p["model_state_dict"].update(unknown=torch.ones(1)),
            "shape": lambda p: p["model_state_dict"].update({"bond_head.3.weight": torch.ones(1, 3)}),
            "nonfinite": lambda p: p["model_state_dict"]["bond_head.3.bias"].fill_(float("nan")),
            "manifest canonical": lambda p: p.update(dataset_manifest_canonical_sha256="0" * 64),
            "input hash": lambda p: p["input_sha256"].update(train_csv="0" * 64),
            "sequence hash": lambda p: p.update(train_sequences_sha256="0" * 64),
            "trained aa": lambda p: p.update(trained_amino_acids=["W"]),
            "positions": lambda p: p.update(trained_position_count=32),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                self.payload = deepcopy(original)
                mutate(self.payload)
                self.rewrite_checkpoint()
                with self.assertRaises(ValueError):
                    self.load()
                self.assert_unchanged()
        bad_hash = dict(self.training, pretrained_backbone_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "checkpoint SHA256"):
            self.load(training=bad_hash)
        self.assert_unchanged()

    def test_source_and_target_leakage_rejected_by_sequence_only(self):
        self.external["val"] = ["ACDEFGI"]  # I/L canonical collision with target ACDEFGL.
        self.rebuild_checkpoint()
        with self.assertRaisesRegex(ValueError, "overlap excluded"):
            self.load()
        self.assert_unchanged()
        self.external["val"] = ["WYWWW"]
        self.rebuild_checkpoint()
        Path(self.target_paths["test"]).write_text("seq,unused_label\nBUNDECLARED,never-read\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "absent from full MiPD"):
            self.load()
        self.assert_unchanged()

    def test_manifest_and_csv_declarations_are_bound_to_actual_files(self):
        file = self.repo / "external_train.csv"
        file.write_text("seq\nACDEFGH\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "CSV SHA256"):
            self.load()
        self.assert_unchanged()
        self.rebuild_checkpoint()
        (self.repo / "dataset_manifest.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest file SHA256"):
            self.load()
        self.assert_unchanged()

    def test_provenance_persists_in_json_and_all_checkpoint_routes(self):
        provenance = self.load()
        file = transfer.write_transfer_provenance(provenance, self.repo / "run")
        self.assertEqual(json.loads(file.read_text(encoding="utf-8")), provenance)
        optimizer = torch.optim.AdamW(self.target.parameters())
        selection = CheckpointSelection("max_f1", 0.0)
        selection.update(0.8, 1)
        for name, selected in (("best_model.pt", selection), ("checkpoint_epoch_1.pt", selection), ("best_model_q.pt", None)):
            path = self.repo / name
            save_checkpoint_with_selection(self.target, optimizer, 1, 0.2, {"f1": 0.8}, str(path), selected,
                                           extra_state={"pretrained_transfer_provenance": provenance})
            loaded = torch.load(path, weights_only=True)
            self.assertEqual(loaded["extra_state"]["pretrained_transfer_provenance"], provenance)
            self.assertEqual(CheckpointSelection.STATE_KEY in loaded, selected is not None)
        path = self.repo / "disabled.pt"
        save_checkpoint_with_selection(self.target, optimizer, 1, 0.2, {}, str(path), selection)
        self.assertNotIn("extra_state", torch.load(path, weights_only=True))

    def test_actual_production_configuration_and_state_key_contract(self):
        repo = Path(__file__).resolve().parents[2]
        args = Namespace(epochs=None, batch_size=None, learning_rate=None, device=None)
        target_config = build_model_config(load_config(repo / "graph_transform/config/pre_synthesis_fold1222_theory_film.yaml", args))
        source_config = make_pretraining_config()
        transfer._verify_structure(source_config.to_dict(), target_config.to_dict())
        for config in (source_config, target_config):
            model = GraphTransformer(config)
            self.assertEqual(set(model.state_dict()), transfer._model_keys(config.to_dict()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
