"""Small CPU tests for verified ion4 discriminative learning rates; no training run."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.utils import LearningRateScheduler
from pretraining import transfer
from training.checkpoint_selection import CheckpointSelection
from training.transfer_optimizer import (
    validate_transfer_optimizer_request, describe_transfer_optimizer,
    get_optimizer_provenance, write_optimizer_provenance,
)
from train_graph_model import create_optimizer, create_scheduler, save_checkpoint_with_selection
import test_pretrained_transfer as fixtures


def config_for(training, scale=1.0):
    return {
        "optimizer": {"type": "adamw", "betas": [0.9, 0.999], "eps": "1e-8"},
        "training": {**training, "transfer_encoder_lr_scale": scale,
                     "learning_rate": 1e-4, "weight_decay": 1e-4, "scheduler": "cosine",
                     "warmup_epochs": 40, "epochs": 300, "min_lr": 1e-5},
    }


class TransferOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        # Reuse the existing synthetic source/513-sequence fixture, but execute
        # the real transfer loader, including its source/manifest validations.
        self.fixture = fixtures.PretrainedTransferTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.training = dict(self.fixture.training, pretrained_transfer_scope="bond_context")
        self.model = self.fixture.target
        self.provenance = self.fixture.load(training=self.training)
        self.config = config_for(self.training, 0.1)

    def assert_recursive_equal(self, actual, expected):
        if torch.is_tensor(actual):
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(actual, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                self.assert_recursive_equal(actual[key], expected[key])
        elif isinstance(actual, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_recursive_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    def test_default_and_explicit_one_exact_legacy_optimizer_scheduler_rng(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                config = config_for({}, 1.0)
                if not explicit:
                    del config["training"]["transfer_encoder_lr_scale"]
                actual_model, legacy_model = deepcopy(self.model), deepcopy(self.model)
                rng = torch.get_rng_state().clone()
                actual = create_optimizer(actual_model, config)
                actual_scheduler = create_scheduler(actual, config)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                legacy = torch.optim.AdamW(legacy_model.parameters(), lr=1e-4, weight_decay=1e-4,
                                          betas=[0.9, 0.999], eps=1e-8)
                legacy_scheduler = LearningRateScheduler.cosine_annealing_with_warmup(legacy, 40, 300, 1e-5)
                self.assert_recursive_equal(actual.state_dict(), legacy.state_dict())
                self.assert_recursive_equal(actual_scheduler.state_dict(), legacy_scheduler.state_dict())
                self.assertEqual([id(p) for p in actual.param_groups[0]["params"]],
                                 [id(p) for p in actual_model.parameters()])
                self.assertNotIn("name", actual.param_groups[0])
                self.assertNotIn("lr_scale", actual.param_groups[0])
                for _ in range(3):
                    for left, right in zip(actual_model.parameters(), legacy_model.parameters()):
                        left.grad = torch.full_like(left, 0.2)
                        right.grad = torch.full_like(right, 0.2)
                    actual.step()
                    legacy.step()
                    actual_scheduler.step()
                    legacy_scheduler.step()
                self.assert_recursive_equal(actual_model.state_dict(), legacy_model.state_dict())
                self.assert_recursive_equal(actual.state_dict(), legacy.state_dict())
                self.assert_recursive_equal(actual_scheduler.state_dict(), legacy_scheduler.state_dict())

    def test_verified_groups_full_coverage_order_and_partial_parameters_fast(self):
        rng = torch.get_rng_state().clone()
        optimizer = create_optimizer(self.model, self.config, self.provenance)
        scheduler = create_scheduler(optimizer, self.config)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        record = describe_transfer_optimizer(self.model, optimizer, self.config, self.provenance)
        fast, slow = record["groups"]
        expected_slow = set(self.provenance["fully_copied_keys"]) - transfer.CONTEXT_KEYS
        self.assertEqual(slow["num_tensors"], 48)
        self.assertEqual(set(slow["param_names"]), expected_slow)
        original_names = [name for name, _ in self.model.named_parameters()]
        self.assertEqual(fast["param_names"], [name for name in original_names if name not in expected_slow])
        self.assertEqual(slow["param_names"], [name for name in original_names if name in expected_slow])
        self.assertFalse(set(fast["param_names"]) & set(slow["param_names"]))
        self.assertEqual(set(fast["param_names"]) | set(slow["param_names"]), set(original_names))
        self.assertEqual(record["total_numel"], sum(p.numel() for p in self.model.parameters()))
        self.assertEqual(sum(g["numel"] for g in record["groups"]), record["total_numel"])
        for name in self.provenance["partially_copied_keys"] + sorted(transfer.CONTEXT_KEYS):
            self.assertIn(name, fast["param_names"])
        for name in original_names:
            if name.startswith(("edge_encoder.", "global_", "film_mlp.", "node_encoder.state_encoder.",
                                "node_encoder.env_encoder.", "bond_head.3.")):
                self.assertIn(name, fast["param_names"])
        self.assertEqual(record["base_lr_partial_parameter_names"], self.provenance["partially_copied_keys"])
        self.assertEqual([group["name"] for group in optimizer.param_groups], ["base", "pretrained_encoder"])
        self.assertEqual([group["initial_lr"] for group in optimizer.param_groups], scheduler.base_lrs)
        self.assertEqual([group["initial_lr"] for group in record["groups"]], scheduler.get_last_lr())

    def test_actual_adamw_delta_ratio_and_special_rows_keep_full_updates(self):
        # float64 isolates the optimizer's LR effect from subtraction roundoff;
        # this is a real AdamW update, not a gradient-scaling simulation.
        grouped_model = deepcopy(self.model).double()
        ordinary_model = deepcopy(grouped_model)
        grouped = create_optimizer(grouped_model, self.config, self.provenance)
        ordinary = create_optimizer(ordinary_model, config_for(self.training, 1.0), self.provenance)
        before = {name: parameter.detach().clone() for name, parameter in grouped_model.named_parameters()}
        for parameter in grouped_model.parameters():
            parameter.grad = torch.full_like(parameter, 0.3)
        for parameter in ordinary_model.parameters():
            parameter.grad = torch.full_like(parameter, 0.3)
        grouped.step()
        ordinary.step()
        slow = set(self.provenance["fully_copied_keys"]) - transfer.CONTEXT_KEYS
        other = dict(ordinary_model.named_parameters())
        for name, parameter in grouped_model.named_parameters():
            actual_delta = before[name] - parameter.detach()
            reference_delta = before[name] - other[name].detach()
            torch.testing.assert_close(actual_delta, reference_delta * (0.1 if name in slow else 1.0),
                                       rtol=1e-10, atol=1e-15, msg=name)
        aa = dict(grouped_model.named_parameters())[transfer.AA_KEY]
        for residue in "BOXZ":
            row = grouped_model.config.alphabet.index(residue) + 1
            self.assertTrue(torch.equal(aa[row], other[transfer.AA_KEY][row]), residue)
        # Equal gradients give equal Adam moments, despite different LR groups.
        slow_name = next(iter(slow))
        grouped_parameter = dict(grouped_model.named_parameters())[slow_name]
        self.assertTrue(torch.equal(grouped.state[grouped_parameter]["exp_avg"],
                                    ordinary.state[other[slow_name]]["exp_avg"]))

    def test_warmup_cosine_floor_preserves_ratio_and_legacy_normal_curve(self):
        grouped = create_optimizer(self.model, self.config, self.provenance)
        scheduler = create_scheduler(grouped, self.config)
        uniform = create_optimizer(deepcopy(self.model), config_for(self.training, 1.0), self.provenance)
        legacy = create_scheduler(uniform, config_for(self.training, 1.0))
        for step in range(304):
            self.assertEqual(grouped.param_groups[0]["lr"], uniform.param_groups[0]["lr"], step)
            self.assertAlmostEqual(grouped.param_groups[1]["lr"] / grouped.param_groups[0]["lr"], 0.1, places=14)
            if step >= 300:
                self.assertAlmostEqual(grouped.param_groups[0]["lr"], 1e-5, places=18)
                self.assertAlmostEqual(grouped.param_groups[1]["lr"], 1e-6, places=18)
            grouped.step()
            uniform.step()
            scheduler.step()
            legacy.step()
        self.assertEqual(scheduler.last_epoch, legacy.last_epoch)

    def test_bad_scale_scope_resume_and_unverified_provenance_rejected(self):
        for value in (0, -0.1, 1.1, float("nan"), float("inf"), True, "0.1"):
            with self.subTest(scale=value), self.assertRaises(ValueError):
                validate_transfer_optimizer_request({"transfer_encoder_lr_scale": value})
        with self.assertRaisesRegex(ValueError, "resume"):
            validate_transfer_optimizer_request(self.config["training"], "prior.pt")
        for scope in ("none", "encoder"):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                validate_transfer_optimizer_request(dict(self.config["training"], pretrained_transfer_scope=scope))
        mutations = [None, {}, dict(self.provenance, scope="encoder"),
                     dict(self.provenance, fully_copied_keys=self.provenance["fully_copied_keys"] + [transfer.AA_KEY]),
                     dict(self.provenance, partially_copied_keys=[]),
                     dict(self.provenance, retained_entire_keys=[]),
                     dict(self.provenance, checkpoint_sha256="0" * 64),
                     dict(self.provenance, dataset_binding={})]
        before = {name: tensor.clone() for name, tensor in self.model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        for index, provenance in enumerate(mutations):
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                create_optimizer(self.model, self.config, provenance)
        self.assert_recursive_equal(self.model.state_dict(), before)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for section, key, value in (("optimizer", "type", "adam"), ("training", "scheduler", "plateau")):
            config = deepcopy(self.config)
            config[section][key] = value
            with self.assertRaises(ValueError):
                create_optimizer(self.model, config, self.provenance)

    def test_opt_in_metadata_and_all_checkpoint_roles_safe_roundtrip(self):
        old_config = config_for({})
        del old_config["training"]["transfer_encoder_lr_scale"]
        rng = torch.get_rng_state().clone()
        self.assertIsNone(get_optimizer_provenance(None, None, old_config))
        absent = self.fixture.repo / "legacy_has_no_metadata"
        self.assertIsNone(write_optimizer_provenance(None, absent))
        self.assertFalse(absent.exists())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for scale, transferred in ((1.0, False), (1.0, True), (0.1, True)):
            with self.subTest(scale=scale, transferred=transferred):
                provenance = self.provenance if transferred else None
                config = config_for(self.training if transferred else {}, scale)
                optimizer = create_optimizer(self.model, config, provenance)
                create_scheduler(optimizer, config)
                record = get_optimizer_provenance(self.model, optimizer, config, provenance)
                directory = self.fixture.repo / f"metadata_{scale}_{transferred}"
                path = write_optimizer_provenance(record, directory)
                self.assertEqual(json.loads(path.read_text(encoding="utf-8")), record)
                extra = {"optimizer_provenance": record}
                if transferred:
                    extra["pretrained_transfer_provenance"] = provenance
                selection = CheckpointSelection("max_f1", 0.001)
                selection.update(0.7, 1)
                for filename in ("best_model.pt", "best_model_q.pt", "checkpoint_epoch_1.pt"):
                    saved = directory / filename
                    save_checkpoint_with_selection(self.model, optimizer, 0, 0.2, {"f1": 0.7}, str(saved),
                                                   None if "_q" in filename else selection, extra_state=extra)
                    checkpoint = torch.load(saved, map_location="cpu", weights_only=True)
                    self.assertEqual(checkpoint["extra_state"], extra)
                    self.assertEqual(len(checkpoint["optimizer_state_dict"]["param_groups"]), len(record["groups"]))
                resumed = get_optimizer_provenance(None, None, old_config, resume_checkpoint={"extra_state": extra})
                self.assertEqual(resumed, record)
                self.assertIsNot(resumed, record)

    def test_actual_three_configs_match_except_declared_arm_fields(self):
        directory = Path(__file__).resolve().parents[1] / "config"
        configs = [yaml.safe_load((directory / f"pre_synthesis_theory_film_lr5_{arm}.yaml").read_text(encoding="utf-8"))
                   for arm in ("control", "uniform", "slow")]
        self.assertEqual([cfg["training"]["transfer_encoder_lr_scale"] for cfg in configs], [1.0, 1.0, 0.1])
        self.assertEqual([cfg["training"]["pretrained_transfer_scope"] for cfg in configs],
                         ["none", "bond_context", "bond_context"])
        for key in ("pretrained_backbone_path", "pretrained_backbone_sha256"):
            self.assertEqual(configs[1]["training"][key], configs[2]["training"][key])
            self.assertIsNone(configs[0]["training"][key])
        allowed = {"experiment": ("name", "description"), "data": ("cache_dir",),
                   "debug": ("diagnostic_dir",), "logging": ("log_dir", "tensorboard_log_dir"),
                   "evaluation": ("output_metric_dir", "output_pred_dir"),
                   "training": ("checkpoint_dir", "pretrained_backbone_path", "pretrained_backbone_sha256",
                                "pretrained_transfer_scope", "transfer_encoder_lr_scale")}
        for config in configs:
            for section, keys in allowed.items():
                for key in keys:
                    config.get(section, {}).pop(key, None)
        self.assertEqual(configs[0], configs[1])
        self.assertEqual(configs[1], configs[2])


if __name__ == "__main__":
    unittest.main()
