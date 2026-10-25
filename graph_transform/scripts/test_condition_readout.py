"""Synthetic late-condition tests; --device cuda also checks FP16 autocast."""

import argparse
import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.graph_builder import GraphBuilder
from models.condition_readout import BondConditionReadout
from models.utils import CheckpointManager, build_model_config
from test_architecture_extensions import make_config, make_samples, collate, make_model


DEVICE = 'cpu'
BRANCH_PREFIX = 'bond_condition_readout.'


def configuration(mode='none', global_node=True):
    config = make_config(global_node=global_node)
    config.bond_condition_mode = mode
    config.bond_condition_dim = 8
    return config


def move_batch(batch):
    return {key: value.to(DEVICE) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def activate(branch):
    with torch.no_grad():
        branch.head[-1].weight.copy_(
            torch.linspace(0.05, 0.20, branch.head[-1].in_features,
                           device=branch.head[-1].weight.device).unsqueeze(0))
        branch.head[-1].bias.fill_(0.03)


def assert_same(actual, expected):
    tolerance = 1e-6 if DEVICE.startswith('cuda') else 0
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def change_conditions(samples, config, charge=None, nce=None):
    changed = copy.deepcopy(samples)
    builder = GraphBuilder(config)
    for sample in changed:
        if charge is not None:
            sample['charge'] = charge
            sample['state_vars'][0] = charge
        if nce is not None:
            sample['nce'] = nce
            sample['env_vars'][0] = nce
        sample.update(builder.build_graph(sample['sequence'], sample, 'distance'))
    return changed


class ConditionReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        cls.previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)
        torch.backends.cudnn.allow_tf32 = cls.previous_cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = cls.previous_matmul_tf32

    def test_defaults_validation_and_initial_shared_weights_outputs_rng(self):
        defaults = build_model_config({})
        self.assertEqual(defaults.bond_condition_mode, 'none')
        self.assertEqual(defaults.bond_condition_dim, 32)
        with self.assertRaises(ValueError):
            make_model(configuration('invalid'))
        for invalid in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                BondConditionReadout(16, 'concat', invalid)
        for global_node in (False, True):
            baseline_config = configuration(global_node=global_node)
            batch = move_batch(collate(make_samples(baseline_config)))
            baseline = make_model(baseline_config).to(DEVICE)
            creation_rng = torch.get_rng_state()
            self.assertFalse(any(key.startswith(BRANCH_PREFIX) for key in baseline.state_dict()))
            for mode in ('concat', 'gated'):
                model = make_model(configuration(mode, global_node)).to(DEVICE)
                torch.testing.assert_close(torch.get_rng_state(), creation_rng, rtol=0, atol=0)
                for key, value in baseline.state_dict().items():
                    torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
                for training in (False, True):
                    baseline.train(training)
                    model.train(training)
                    torch.manual_seed(91)
                    with torch.no_grad():
                        expected = baseline(batch)
                    expected_rng = torch.get_rng_state()
                    expected_cuda_rng = torch.cuda.get_rng_state(DEVICE) if DEVICE.startswith('cuda') else None
                    torch.manual_seed(91)
                    with torch.no_grad():
                        actual = model(batch)
                    assert_same(actual, expected)
                    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
                    if expected_cuda_rng is not None:
                        torch.testing.assert_close(torch.cuda.get_rng_state(DEVICE), expected_cuda_rng,
                                                   rtol=0, atol=0)

    def test_active_module_uses_both_conditions_and_masks_nan_before_arithmetic(self):
        for mode in ('concat', 'gated'):
            torch.manual_seed(41)
            branch = BondConditionReadout(16, mode, 8).to(DEVICE)
            activate(branch)
            hidden = torch.randn(4, 16, device=DEVICE)
            owners = torch.tensor([0, 0, 1, 1], device=DEVICE)
            charges = torch.tensor([2., 3.], device=DEVICE)
            nces = torch.tensor([20., 30.], device=DEVICE)
            expected = branch(hidden, charges, nces, owners)
            self.assertGreater(float((branch(hidden, charges + 2, nces, owners) - expected).detach().abs().max()), 1e-7)
            self.assertGreater(float((branch(hidden, charges, nces + 20, owners) - expected).detach().abs().max()), 1e-7)
            for use_charge, use_nce in ((False, True), (True, False), (False, False)):
                masked = BondConditionReadout(16, mode, 8, use_charge, use_nce).to(DEVICE)
                # condition_mask is a configuration buffer, deliberately absent from checkpoints.
                masked.load_state_dict(branch.state_dict(), strict=True)
                bad_charge = charges.clone() if use_charge else torch.full_like(charges, float('nan'))
                bad_nce = nces.clone() if use_nce else torch.full_like(nces, float('inf'))
                bad_charge.requires_grad_()
                bad_nce.requires_grad_()
                out = masked(hidden, bad_charge, bad_nce, owners)
                torch.testing.assert_close(out, masked(hidden, charges, nces, owners), rtol=0, atol=0)
                self.assertTrue(torch.isfinite(out).all())
                out.sum().backward()
                if not use_charge:
                    self.assertEqual(bad_charge.grad.count_nonzero().item(), 0)
                if not use_nce:
                    self.assertEqual(bad_nce.grad.count_nonzero().item(), 0)

    def test_active_full_model_batch_order_padding_global_and_observed_invariance(self):
        for mode in ('concat', 'gated'):
            for global_node in (False, True):
                config = configuration(mode, global_node)
                model = make_model(config).to(DEVICE)
                activate(model.bond_condition_readout)
                samples = make_samples(config)
                batch = move_batch(collate(samples))
                with torch.no_grad():
                    out = model(batch)
                    self.assertEqual(out[~batch['label_mask'].bool()].count_nonzero().item(), 0)
                    for index, sample in enumerate(samples):
                        single = model(move_batch(collate([sample])))
                        torch.testing.assert_close(out[index, :sample['seq_len'] - 1], single[0],
                                                   rtol=1e-4, atol=2e-6)
                    order = [2, 0, 1]
                    reordered = model(move_batch(collate([samples[index] for index in order])))
                    torch.testing.assert_close(reordered, out[order], rtol=1e-4, atol=2e-6)
                    observed = model(move_batch(collate(make_samples(config, observed_scale=float('nan')))))
                    assert_same(observed, out)

    def test_new_branch_and_full_model_respect_feature_switches_without_legacy_film(self):
        # Legacy FiLM intentionally remains unchanged and reads raw conditions.
        # Turn it off here to test whole-model invariance under the feature masks.
        cases = [('use_state_features', False, 'charge'), ('use_env_features', False, 'nce'),
                 ('state_feature_mask', [False, True, False], 'charge'),
                 ('env_feature_mask', [False, False], 'nce')]
        for mode in ('concat', 'gated'):
            for name, value, blocked in cases:
                config = configuration(mode)
                config.use_condition_film = False
                setattr(config, name, value)
                model = make_model(config).to(DEVICE)
                activate(model.bond_condition_readout)
                self.assertFalse(bool(model.bond_condition_readout.condition_mask[0 if blocked == 'charge' else 1]))
                samples = make_samples(config)
                bad = change_conditions(samples, config, **{blocked: float('nan')})
                with torch.no_grad():
                    out = model(move_batch(collate(samples)))
                    altered = model(move_batch(collate(bad)))
                self.assertTrue(torch.isfinite(altered).all())
                assert_same(altered, out)

    def test_module_gradients_and_equal_budget_distinct_interaction(self):
        branches = []
        for mode in ('concat', 'gated'):
            torch.manual_seed(7)
            branch = BondConditionReadout(16, mode, 8).to(DEVICE)
            hidden = torch.randn(5, 16, device=DEVICE, requires_grad=True)
            owners = torch.tensor([0, 0, 1, 1, 1], device=DEVICE)
            charges = torch.tensor([2., 4.], device=DEVICE)
            nces = torch.tensor([20., 50.], device=DEVICE)
            for active in (False, True):
                branch.zero_grad(set_to_none=True)
                if active:
                    activate(branch)
                out = branch(hidden, charges, nces, owners)
                if not active:
                    self.assertEqual(out.count_nonzero().item(), 0)
                (out - torch.linspace(.1, .9, 5, device=DEVICE)).square().mean().backward()
                self.assertGreater(float(branch.head[-1].weight.grad.abs().sum()), 0)
                if active:
                    for parameter in branch.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                        self.assertGreater(float(parameter.grad.abs().sum()), 0)
            branches.append(branch)
        self.assertEqual(sum(p.numel() for p in branches[0].parameters()),
                         sum(p.numel() for p in branches[1].parameters()))
        for (_, left), (_, right) in zip(branches[0].named_parameters(), branches[1].named_parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertGreater(float((branches[0](hidden, charges, nces, owners)
                                  - branches[1](hidden, charges, nces, owners)).detach().abs().max()), 1e-7)

    def test_full_model_autocast_backward_and_strict_roundtrip(self):
        device_type = 'cuda' if DEVICE.startswith('cuda') else 'cpu'
        dtype = torch.float16 if device_type == 'cuda' else torch.bfloat16
        for mode in ('concat', 'gated'):
            for global_node in (False, True):
                config = configuration(mode, global_node)
                model = make_model(config).to(DEVICE).train()
                batch = move_batch(collate(make_samples(config)))
                mask = batch['label_mask'].bool()
                for active in (False, True):
                    if active:
                        activate(model.bond_condition_readout)
                    model.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device_type, dtype=dtype):
                        out = model(batch)
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(out[mask], batch['labels'][mask])
                    self.assertEqual(out.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(out).all())
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    self.assertGreater(float(model.bond_condition_readout.head[-1].weight.grad.abs().sum()), 0)
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            self.assertTrue(torch.isfinite(parameter.grad).all())
                stream = io.BytesIO()
                torch.save(model.state_dict(), stream)
                stream.seek(0)
                restored = make_model(config, seed=12).to(DEVICE)
                restored.load_state_dict(torch.load(stream, weights_only=True, map_location=DEVICE), strict=True)
                model.eval()
                with torch.no_grad():
                    assert_same(restored(batch), model(batch))

    def test_checkpoint_modes_reject_mismatch_including_same_shapes(self):
        with tempfile.TemporaryDirectory(prefix='dbond_condition_test_') as directory:
            path = Path(directory) / 'checkpoint.pt'
            for source_mode in ('none', 'concat', 'gated'):
                source = make_model(configuration(source_mode))
                if source_mode != 'none':
                    activate(source.bond_condition_readout)
                # No config metadata: actual state keys/mode tag must suffice.
                torch.save({'model_state_dict': source.state_dict()}, path)
                for target_mode in ('none', 'concat', 'gated'):
                    target = make_model(configuration(target_mode), seed=77)
                    if source_mode == target_mode:
                        CheckpointManager.load_checkpoint(path, target, device=torch.device('cpu'))
                        for key, value in source.state_dict().items():
                            torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)
                    else:
                        with self.assertRaises(ValueError):
                            CheckpointManager.load_checkpoint(path, target, device=torch.device('cpu'))
                        with self.assertRaises(RuntimeError):
                            target.load_state_dict(source.state_dict(), strict=True)
                        if target_mode != 'none':
                            self.assertEqual(int(target.bond_condition_readout.mode_code),
                                             BondConditionReadout.MODES[target_mode])
                            with self.assertRaises(RuntimeError):
                                target.load_state_dict(source.state_dict(), strict=True)
            source = make_model(configuration('concat'))
            incomplete = dict(source.state_dict())
            incomplete.pop(BRANCH_PREFIX + 'mode_code')
            torch.save({'model_state_dict': incomplete}, path)
            with self.assertRaises(ValueError):
                CheckpointManager.load_checkpoint(path, make_model(configuration('concat')), device=torch.device('cpu'))

    def test_formal_configs_are_paired_and_have_exact_equal_parameter_counts(self):
        directory = Path(__file__).resolve().parents[1] / 'config'
        configs, counts = [], []
        for arm, mode in (('control', 'none'), ('concat', 'concat'), ('gated', 'gated')):
            cfg = yaml.safe_load((directory / f'pre_synthesis_theory_film_cond3_{arm}.yaml').read_text(encoding='utf-8'))
            self.assertEqual(cfg['training']['checkpoint_selection'], 'max_f1')
            self.assertEqual(cfg['model']['bond_condition_mode'], mode)
            self.assertEqual(cfg['model']['bond_condition_dim'], 32)
            self.assertEqual(cfg['model']['sequence_adapter_type'], 'none')
            self.assertFalse(cfg['model']['use_fragment_readout'])
            self.assertFalse(cfg['model']['use_residue_chemistry'])
            model = make_model(build_model_config(cfg))
            counts.append(sum(parameter.numel() for parameter in model.parameters()))
            cfg['model']['bond_condition_mode'] = 'none'
            for section, keys in {
                'data': ['cache_dir'], 'debug': ['diagnostic_dir'], 'training': ['checkpoint_dir'],
                'logging': ['log_dir', 'tensorboard_log_dir'],
                'evaluation': ['output_metric_dir', 'output_pred_dir'],
                'experiment': ['name', 'description'],
            }.items():
                for key in keys:
                    cfg[section].pop(key)
            configs.append(cfg)
        self.assertEqual(configs[0], configs[1])
        self.assertEqual(configs[0], configs[2])
        self.assertEqual(counts[1] - counts[0], 11489)
        self.assertEqual(counts[2] - counts[0], 11489)
        print(f'Parameter counts: control={counts[0]}, concat={counts[1]}, gated={counts[2]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    args, remaining = parser.parse_known_args()
    DEVICE = args.device
    if DEVICE.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable')
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
