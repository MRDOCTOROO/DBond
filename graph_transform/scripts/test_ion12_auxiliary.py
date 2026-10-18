"""Focused ion12 loss/gradient/legacy tests; execute on the training pod."""
from copy import deepcopy
import math
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.graph_transformer import GraphTransformer
from models.utils import CheckpointManager
from training.ion12_loss import ion12_condition_bce
from training.loss_functions import BinaryBondLoss
from training.trainer import Trainer
from test_architecture_extensions import make_config, make_samples, collate


def model(enabled, seed=819):
    config = make_config()
    config.use_ion12_aux_head = enabled
    config.dropout = 0.1
    torch.manual_seed(seed)
    return GraphTransformer(config), config


class Ion12AuxiliaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_condition_weighting_not_bond_or_group_n_weighting(self):
        mask = torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.bool)
        targets = torch.full((2, 3, 12), float('nan'))
        targets[mask] = 0
        logits = torch.tensor([0., 2., 2., 2.]).unsqueeze(1).expand(-1, 12).clone().requires_grad_()
        loss = ion12_condition_bce(logits, targets, mask)
        expected = (math.log(2) + math.log1p(math.exp(2))) / 2
        self.assertAlmostEqual(float(loss.detach()), expected, places=6)
        self.assertNotAlmostEqual(float(loss.detach()), (math.log(2) + 3 * math.log1p(math.exp(2))) / 4, places=3)
        loss.backward()
        self.assertAlmostEqual(float(logits.grad[0, 0]), 0.5 / (2 * 12), places=7)
        self.assertAlmostEqual(float(logits.grad[1, 0]), torch.sigmoid(torch.tensor(2.)).item() / (2 * 3 * 12), places=7)

    def test_invalid_targets_and_alignment_fail(self):
        mask = torch.ones(1, 2, dtype=torch.bool)
        targets = torch.zeros(1, 2, 12)
        logits = torch.zeros(2, 12)
        for value in (-0.1, 1.1, float('nan')):
            bad = targets.clone()
            bad[0, 0, 0] = value
            with self.assertRaises(ValueError):
                ion12_condition_bce(logits, bad, mask)
        for bad_logits, bad_target, bad_mask in (
            (logits[:1], targets, mask),
            (logits, targets[:, :, :11], mask),
            (logits, targets, torch.zeros_like(mask)),
        ):
            with self.assertRaises(ValueError):
                ion12_condition_bce(bad_logits, bad_target, bad_mask)

    def test_shared_initialization_rng_forward_and_main_gradients(self):
        legacy, config = model(False)
        rng = torch.get_rng_state().clone()
        auxiliary, _ = model(True)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(set(auxiliary.state_dict()) - set(legacy.state_dict()), {'ion12_head.weight', 'ion12_head.bias'})
        for name, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, auxiliary.state_dict()[name]), name)
        batch = collate(make_samples(config))
        for training in (True, False):
            legacy.zero_grad(set_to_none=True)
            auxiliary.zero_grad(set_to_none=True)
            legacy.train(training)
            auxiliary.train(training)
            torch.manual_seed(917)
            old = legacy(batch)
            old_rng = torch.get_rng_state().clone()
            torch.manual_seed(917)
            new = auxiliary(batch)
            self.assertTrue(torch.equal(old, new))
            self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
            if training:
                packed = auxiliary.last_aux_outputs['ion12_logits']
                self.assertEqual(packed.shape, (int(batch['label_mask'].sum()), 12))
            else:
                self.assertIsNone(auxiliary.last_aux_outputs)
            mask = batch['label_mask'].bool()
            F.binary_cross_entropy_with_logits(old[mask], batch['labels'][mask]).backward()
            F.binary_cross_entropy_with_logits(new[mask], batch['labels'][mask]).backward()
            for name, parameter in legacy.named_parameters():
                other = dict(auxiliary.named_parameters())[name]
                if parameter.grad is None:
                    self.assertIsNone(other.grad)
                else:
                    torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)

    def test_auxiliary_reaches_head_and_backbone_and_targets_are_not_inputs(self):
        net, config = model(True)
        batch = collate(make_samples(config))
        batch['ion12_targets'] = torch.full((*batch['labels'].shape, 12), 0.25)
        torch.manual_seed(112)
        output = net(batch)
        auxiliary = net.last_aux_outputs['ion12_logits']
        changed = dict(batch, ion12_targets=1 - batch['ion12_targets'])
        torch.manual_seed(112)
        repeated = net(changed)
        self.assertTrue(torch.equal(output, repeated))
        loss = ion12_condition_bce(auxiliary, batch['ion12_targets'], batch['label_mask'])
        loss.backward()
        for parameter in (net.ion12_head.weight, net.bond_head[0].weight, net.gat_layers[0].W):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertIsNone(net.bond_head[3].weight.grad)

    def test_trainer_lambda_zero_exact_main_and_positive_composition(self):
        with tempfile.TemporaryDirectory() as folder:
            for weight in (0., 0.1):
                net, config = model(True)
                batch = collate(make_samples(config))
                batch['ion12_targets'] = torch.full((*batch['labels'].shape, 12), 0.2)
                batch['sample_weights'] = torch.ones(batch['labels'].shape[0])
                criterion = BinaryBondLoss({'main_loss': 'binary_cross_entropy'})
                trainer = Trainer(net, torch.optim.AdamW(net.parameters(), lr=1e-4), criterion,
                    config={'loss': {'ion12_aux_weight': weight}, 'debug': {'diagnostic_dir': folder}})
                torch.manual_seed(223)
                predicted = net(batch)
                mask = batch['label_mask'].bool()
                main = criterion(predicted[mask], batch['labels'][mask], weights=torch.ones_like(predicted[mask]))
                aux = ion12_condition_bce(net.last_aux_outputs['ion12_logits'], batch['ion12_targets'], mask)
                torch.manual_seed(223)
                actual, stats = trainer._forward_pass(batch)
                torch.testing.assert_close(actual, main + weight * aux, rtol=0, atol=0)
                self.assertAlmostEqual(stats['main_loss'], float(main.detach()))
                actual.backward()
                if weight == 0:
                    self.assertIsNone(net.ion12_head.weight.grad)
                else:
                    self.assertGreater(float(net.ion12_head.weight.grad.abs().sum()), 0)
                with self.assertRaises(ValueError):
                    trainer._forward_pass({k: v for k, v in batch.items() if k != 'ion12_targets'})

    def test_checkpoint_enabled_disabled_rejected_and_roundtrip(self):
        legacy, _ = model(False)
        auxiliary, _ = model(True)
        for dest, source in ((legacy, auxiliary), (auxiliary, legacy)):
            with self.assertRaisesRegex(RuntimeError, 'ion12'):
                dest.load_state_dict(source.state_dict(), strict=False)
        copy_model, _ = model(True, seed=820)
        copy_model.load_state_dict(auxiliary.state_dict())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'checkpoint.pt'
            torch.save({'model_state_dict': auxiliary.state_dict()}, path)
            CheckpointManager.load_checkpoint(str(path), copy_model)
            with self.assertRaisesRegex(ValueError, 'ion12'):
                CheckpointManager.load_checkpoint(str(path), legacy)

    def test_epoch_loss_keeps_distinct_main_and_auxiliary_denominators(self):
        config = make_config()
        config.use_ion12_aux_head = True
        net = GraphTransformer(config)
        samples = make_samples(config)
        batches = [collate(samples[:1]), collate(samples[1:])]
        for batch in batches:
            batch['ion12_targets'] = torch.full((*batch['labels'].shape, 12), 0.2)
        with tempfile.TemporaryDirectory() as folder:
            trainer = Trainer(net, torch.optim.AdamW(net.parameters(), lr=0.), BinaryBondLoss({}),
                config={'loss': {'ion12_aux_weight': 0.1}, 'debug': {'diagnostic_dir': folder}})
            # Legacy state/env encoders have fixed dropout=0.1. Match their RNG
            # across both passes; lr=0 prevents parameter updates in this check.
            torch.manual_seed(556)
            values = [trainer._forward_pass(batch)[1] for batch in batches]
            expected_main = sum(v['main_loss'] * v['valid_bond_count'] for v in values) / sum(v['valid_bond_count'] for v in values)
            expected_aux = sum(v['ion12_aux_loss'] * v['sample_count'] for v in values) / sum(v['sample_count'] for v in values)
            torch.manual_seed(556)
            metrics = trainer.train_epoch(batches, epoch=0)
            self.assertAlmostEqual(metrics['main_loss'], expected_main, places=6)
            self.assertAlmostEqual(metrics['ion12_aux_loss'], expected_aux, places=6)
            self.assertAlmostEqual(metrics['loss'], expected_main + 0.1 * expected_aux, places=6)

    def test_invalid_weight_or_missing_head_fails_before_training(self):
        for enabled, weight in ((True, None), (True, -0.1), (True, float('nan')), (False, 0.1)):
            net, _ = model(enabled)
            config = {'loss': {'ion12_aux_weight': weight}}
            with self.assertRaises(ValueError):
                Trainer(net, torch.optim.AdamW(net.parameters()), BinaryBondLoss({}), config=config)


if __name__ == '__main__':
    unittest.main()
