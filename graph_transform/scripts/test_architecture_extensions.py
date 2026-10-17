"""Small CPU regression tests for the opt-in architecture branches.

Run from the repository root:
    .venv/Scripts/python.exe graph_transform/scripts/test_architecture_extensions.py

Synthetic tensors only: no dataset, training loop, optimizer, or GPU is required.
"""

from __future__ import annotations

import io
import itertools
from pathlib import Path
import sys
import tempfile
import unittest
from copy import deepcopy

import torch
import yaml
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.graph_builder import GraphBuilder
from data.graph_dataset import GraphDataLoader
from data.theory_features import RESIDUE_MONO_MASS, compute_bond_theory
from models.fragment_readout import fragment_means
from models.graph_transformer import GraphTransformer
from models.residue_chemistry import (
    CHEMISTRY_FEATURE_NAMES,
    RESIDUE_COMPOSITION,
    ResidueChemistryEncoder,
    residue_descriptor,
)
from models.utils import CheckpointManager, build_model_config


BRANCH_PREFIXES = ("fragment_readout.", "residue_chemistry.")
SAMPLE_SPECS = (("AG", 2.0, 25.0), ("BOPX", 3.0, 30.0), ("ZKRPGH", 2.0, 35.0))


def make_config(fragment=False, chemistry=False, global_node=True):
    return build_model_config({"model": {
        "alphabet": "#ABCDEFGHIKLMNOPQRSTVWXYZ", "pad_char": "#",
        "hidden_dim": 16, "num_attention_heads": 4,
        "aa_embedding_dim": 8, "position_embedding_dim": 4,
        "physicochemical_dim": 4, "edge_embedding_dim": 4,
        "distance_embedding_dim": 4, "max_seq_len": 12,
        "num_gcn_layers": 1, "num_gat_layers": 1, "num_classes": 1,
        "dropout": 0.0, "max_distance": 6,
        "use_global_node": global_node, "use_long_range_edges": False,
        "use_theory_features": True, "use_condition_film": True,
        "state_feature_mask": [True, True, False],
        "env_feature_mask": [True, False],
        "env_feature_name": "scan_num", "env_feature_scale": 1.0,
        "use_fragment_readout": fragment, "fragment_readout_dim": 8,
        "use_residue_chemistry": chemistry, "residue_chemistry_dim": 8,
    }})


def make_samples(config, observed_scale=1.0):
    builder = GraphBuilder(config)
    samples = []
    for index, (sequence, charge, nce) in enumerate(SAMPLE_SPECS):
        features = {
            "charge": charge, "nce": nce,
            "pep_mass": sum(RESIDUE_MONO_MASS[aa] for aa in sequence) + 18.010565,
            "intensity": (100.0 + index) * observed_scale,
            "scan_num": (10.0 + index) * observed_scale,
            "rt": (2.0 + index) * observed_scale,
        }
        graph = builder.build_graph(sequence, features, "distance")
        samples.append({
            **graph, **features, "sequence": sequence,
            "labels": torch.tensor([float(i % 2) for i in range(len(sequence) - 1)]),
            "env_feature_value": features["scan_num"],
            "state_vars": torch.tensor([charge, features["pep_mass"], features["intensity"]]),
            "env_vars": torch.tensor([nce, features["scan_num"]]),
            "seq_len": len(sequence),
            "node_len": len(sequence) + int(config.use_global_node),
            "bond_theory": compute_bond_theory(sequence),
        })
    return samples


def collate(samples):
    return next(iter(GraphDataLoader(
        dataset=samples, batch_size=len(samples), shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False,
    )))


def make_model(config, seed=2048):
    torch.manual_seed(seed)
    return GraphTransformer(config).eval()


def activate_branches(model):
    """Use nonzero corrections so invariance tests cannot pass vacuously."""
    with torch.no_grad():
        if model.use_fragment_readout:
            model.fragment_readout.head[-1].weight.fill_(0.04)
            model.fragment_readout.head[-1].bias.fill_(0.02)
        if model.use_residue_chemistry:
            layer = model.residue_chemistry.project[-1]
            values = torch.linspace(-0.06, 0.08, layer.weight.numel())
            layer.weight.copy_(values.reshape_as(layer.weight))
            layer.bias.copy_(torch.linspace(-0.02, 0.02, layer.bias.numel()))


class ArchitectureExtensionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_default_flags_and_all_forward_combinations(self):
        defaults = build_model_config({"model": {}})
        self.assertFalse(defaults.use_fragment_readout)
        self.assertFalse(defaults.use_residue_chemistry)
        for fragment, chemistry, global_node in itertools.product((False, True), repeat=3):
            with self.subTest(fragment=fragment, chemistry=chemistry, global_node=global_node):
                config = make_config(fragment, chemistry, global_node)
                batch = collate(make_samples(config))
                with torch.no_grad():
                    output = make_model(config)(batch)
                self.assertEqual(tuple(output.shape), (3, 5))
                self.assertTrue(torch.isfinite(output).all())
                self.assertEqual(int(batch["label_mask"].sum()), 9)
                self.assertEqual(torch.count_nonzero(output[~batch["label_mask"].bool()]), 0)

    def test_zero_initialization_preserves_backbone_and_predictions(self):
        for global_node in (False, True):
            baseline_config = make_config(global_node=global_node)
            baseline = make_model(baseline_config)
            batch = collate(make_samples(baseline_config))
            with torch.no_grad():
                expected = baseline(batch)
            baseline_state = baseline.state_dict()
            for fragment, chemistry in ((True, False), (False, True), (True, True)):
                with self.subTest(fragment=fragment, chemistry=chemistry, global_node=global_node):
                    model = make_model(make_config(fragment, chemistry, global_node))
                    state = model.state_dict()
                    self.assertEqual(
                        {key for key in state if not key.startswith(BRANCH_PREFIXES)},
                        set(baseline_state),
                    )
                    for key, value in baseline_state.items():
                        torch.testing.assert_close(state[key], value, rtol=0, atol=0)
                    with torch.no_grad():
                        torch.testing.assert_close(model(batch), expected, rtol=0, atol=0)

    def test_new_branch_gradients_before_and_after_activation(self):
        config = make_config(True, True)
        model = make_model(config)
        batch = collate(make_samples(config))
        mask = batch["label_mask"].bool()
        for active in (False, True):
            with self.subTest(active=active):
                if active:
                    activate_branches(model)
                model.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(model(batch)[mask], batch["labels"][mask])
                loss.backward()
                for branch in (model.fragment_readout.head, model.residue_chemistry.project):
                    gradient = branch[-1].weight.grad
                    self.assertIsNotNone(gradient)
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertGreater(float(gradient.abs().sum()), 0.0)
                if active:
                    for branch in (model.fragment_readout.project, model.residue_chemistry.project):
                        gradient = branch[0].weight.grad
                        self.assertIsNotNone(gradient)
                        self.assertTrue(torch.isfinite(gradient).all())
                        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_active_branches_preserve_batch_and_observed_input_invariance(self):
        for fragment, chemistry in ((True, False), (False, True), (True, True)):
            for global_node in (False, True):
                with self.subTest(fragment=fragment, chemistry=chemistry, global_node=global_node):
                    config = make_config(fragment, chemistry, global_node)
                    model = make_model(config)
                    samples = make_samples(config)
                    batch = collate(samples)
                    with torch.no_grad():
                        zero_output = model(batch)
                        activate_branches(model)
                        output = model(batch)
                        self.assertGreater(float((output - zero_output).abs().max()), 1e-6)
                        for i, sample in enumerate(samples):
                            alone = model(collate([sample]))
                            torch.testing.assert_close(
                                output[i, :sample["seq_len"] - 1], alone[0], rtol=2e-5, atol=2e-6)
                        perturbed = model(collate(make_samples(config, observed_scale=1e5)))
                        torch.testing.assert_close(output, perturbed, rtol=0, atol=0)

    def test_cpu_bfloat16_forward_backward_is_finite(self):
        for enabled, global_node in itertools.product((False, True), repeat=2):
            with self.subTest(enabled=enabled, global_node=global_node):
                config = make_config(enabled, enabled, global_node)
                model = make_model(config)
                activate_branches(model)
                batch = collate(make_samples(config))
                mask = batch["label_mask"].bool()
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    output = model(batch)
                    loss = F.binary_cross_entropy_with_logits(output[mask], batch["labels"][mask])
                self.assertEqual(output.dtype, torch.float32)
                self.assertTrue(torch.isfinite(output).all())
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_strict_state_dict_serialization_roundtrip(self):
        for fragment, chemistry in itertools.product((False, True), repeat=2):
            with self.subTest(fragment=fragment, chemistry=chemistry):
                config = make_config(fragment, chemistry)
                model = make_model(config)
                activate_branches(model)
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                restored = make_model(config, seed=42)
                restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
                batch = collate(make_samples(config))
                with torch.no_grad():
                    torch.testing.assert_close(restored(batch), model(batch), rtol=0, atol=0)

    def test_checkpoint_manager_rejects_branch_mismatch_both_directions(self):
        with tempfile.TemporaryDirectory(prefix="dbond_arch_test_") as directory:
            for fragment, chemistry in ((True, False), (False, True), (True, True)):
                for enabled_source in (False, True):
                    with self.subTest(fragment=fragment, chemistry=chemistry, enabled_source=enabled_source):
                        source_config = make_config(fragment, chemistry) if enabled_source else make_config()
                        target_config = make_config() if enabled_source else make_config(fragment, chemistry)
                        source = make_model(source_config)
                        target = make_model(target_config)
                        path = Path(directory) / "checkpoint.pt"
                        torch.save({"model_state_dict": source.state_dict(),
                                    "config": source_config.to_dict()}, path)
                        with self.assertRaises(ValueError):
                            CheckpointManager.load_checkpoint(str(path), target, device=torch.device("cpu"))

    def test_fragment_means_expected_values_masks_and_gradients(self):
        # Global sentinels include NaN/Inf: neither they nor other samples may enter the means.
        nodes = torch.tensor([[1.0], [3.0], [5.0], [float("nan")],
                              [10.0], [20.0], [float("inf")]], requires_grad=True)
        lengths = torch.tensor([3, 2])
        offsets = torch.tensor([0, 4])
        prefix, suffix = fragment_means(nodes, lengths, offsets, max_seq_len=5)
        expected_prefix = torch.tensor([[[1.0], [2.0], [0.0], [0.0]],
                                        [[10.0], [0.0], [0.0], [0.0]]])
        expected_suffix = torch.tensor([[[4.0], [5.0], [0.0], [0.0]],
                                        [[20.0], [0.0], [0.0], [0.0]]])
        torch.testing.assert_close(prefix, expected_prefix)
        torch.testing.assert_close(suffix, expected_suffix)
        (prefix.sum() + suffix.sum()).backward()
        torch.testing.assert_close(nodes.grad[:, 0], torch.tensor([1.5, 1.0, 1.5, 0.0, 1.0, 1.0, 0.0]))

    def test_fragment_means_reverse_exchanges_ordered_n_c_sides(self):
        residues = torch.tensor([[1.0, 4.0], [3.0, 9.0], [8.0, 2.0], [6.0, 5.0]])
        lengths, offsets = torch.tensor([4]), torch.tensor([0])
        prefix, suffix = fragment_means(residues, lengths, offsets, 4)
        rev_prefix, rev_suffix = fragment_means(residues.flip([0]), lengths, offsets, 4)
        torch.testing.assert_close(rev_prefix, suffix.flip([1]))
        torch.testing.assert_close(rev_suffix, prefix.flip([1]))
        empty_n, empty_c = fragment_means(residues[:1], torch.tensor([1]), offsets, 1)
        self.assertEqual(tuple(empty_n.shape), (1, 0, 2))
        self.assertEqual(tuple(empty_c.shape), (1, 0, 2))

    def test_chemistry_compositions_match_theory_masses(self):
        # Monoisotopic neutral atom masses; residue formulas have already lost H2O.
        atom_masses = (12.0, 1.00782503223, 14.00307400443, 15.99491461957, 31.9720711744)
        self.assertEqual(set(RESIDUE_COMPOSITION), set(RESIDUE_MONO_MASS))
        for residue, composition in RESIDUE_COMPOSITION.items():
            with self.subTest(residue=residue):
                mass = sum(count * atom for count, atom in zip(composition, atom_masses))
                self.assertAlmostEqual(mass, RESIDUE_MONO_MASS[residue], delta=1e-5)

    def test_chemistry_boxz_functional_groups_and_padding(self):
        def groups(residue):
            return dict(zip(CHEMISTRY_FEATURE_NAMES, residue_descriptor(residue)))

        for residue in "BO":
            self.assertEqual(groups(residue)["primary_amine"], 1.0)
            self.assertEqual(groups(residue)["guanidino"], 0.0)
            self.assertEqual(groups(residue)["N"], 2)
        self.assertEqual(groups("X")["pyridyl"], 1.0)
        self.assertEqual(groups("X")["aromatic"], 1.0)
        self.assertEqual(groups("X")["primary_amine"], 0.0)
        self.assertEqual(groups("Z")["saturated_carbocycle"], 1.0)
        self.assertEqual(groups("Z")["aromatic"], 0.0)
        self.assertEqual(groups("P")["backbone_N_substituted"], 1.0)
        self.assertEqual(groups("W")["aromatic"], 2.0)
        with self.assertRaises(ValueError):
            residue_descriptor("?")

        mapping = {"#": 0, "B": 1, "O": 2, "X": 3, "Z": 4}
        encoder = ResidueChemistryEncoder(mapping, 5, hidden_dim=4, chemistry_dim=3)
        self.assertNotIn("lookup", encoder.state_dict())
        torch.testing.assert_close(encoder.lookup[0], torch.zeros_like(encoder.lookup[0]))
        with torch.no_grad():
            encoder.project[-1].weight.fill_(0.1)
            encoder.project[-1].bias.fill_(2.0)
        correction = encoder(torch.tensor([[0, 1, 2, 3, 4, 0]]))
        self.assertTrue(torch.isfinite(correction).all())
        self.assertGreater(float(correction[:, 1:5].detach().abs().sum()), 0.0)
        torch.testing.assert_close(correction[:, [0, 5]], torch.zeros((1, 2, 4)), rtol=0, atol=0)

    def test_branches_keep_initialization_and_rng_when_combined(self):
        models = []
        states = []
        for fragment, chemistry in ((False, False), (True, False), (False, True), (True, True)):
            models.append(make_model(make_config(fragment, chemistry)))
            states.append(torch.get_rng_state())
        for state in states[1:]:
            torch.testing.assert_close(state, states[0], rtol=0, atol=0)
        for index, prefix in ((1, "fragment_readout."), (2, "residue_chemistry.")):
            for key, value in models[index].state_dict().items():
                if key.startswith(prefix):
                    torch.testing.assert_close(value, models[3].state_dict()[key], rtol=0, atol=0)

    def test_experiment_configs_differ_only_in_declared_variables(self):
        directory = Path(__file__).resolve().parents[1] / "config"
        base = yaml.safe_load((directory / "pre_synthesis_fold1222_theory_film.yaml").read_text(encoding="utf-8"))
        output_fields = {
            "training": ["checkpoint_dir"],
            "logging": ["log_dir", "tensorboard_log_dir"],
            "evaluation": ["output_metric_dir", "output_pred_dir"],
            "debug": ["diagnostic_dir"], "data": ["cache_dir"],
        }
        paths = set()
        for arm in ("control", "fragment", "chemistry"):
            config = yaml.safe_load((directory / f"pre_synthesis_theory_film_arch_{arm}.yaml").read_text(encoding="utf-8"))
            parsed = build_model_config(config)
            self.assertEqual(parsed.use_fragment_readout, arm == "fragment")
            self.assertEqual(parsed.use_residue_chemistry, arm == "chemistry")
            normalized = deepcopy(config)
            for key in ("use_fragment_readout", "fragment_readout_dim", "use_residue_chemistry", "residue_chemistry_dim"):
                normalized["model"].pop(key)
            for key in ("name", "description"):
                normalized["experiment"][key] = base["experiment"][key]
            for section, keys in output_fields.items():
                for key in keys:
                    self.assertNotEqual(config[section][key], base[section][key])
                    self.assertNotIn(config[section][key], paths)
                    paths.add(config[section][key])
                    normalized[section][key] = base[section][key]
            self.assertEqual(normalized, base)


if __name__ == "__main__":
    unittest.main(verbosity=2)
