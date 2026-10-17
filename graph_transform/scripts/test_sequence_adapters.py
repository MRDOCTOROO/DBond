"""Small synthetic checks; --device cuda also checks native FP16 autocast."""
import argparse
import io
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.sequence_adapter import SequenceAdapter
from models.utils import CheckpointManager, build_model_config
from test_architecture_extensions import make_config, make_samples, collate, make_model

DEVICE = 'cpu'


def configuration(kind, global_node=True):
    config = make_config(global_node=global_node)
    config.sequence_adapter_type = kind
    config.sequence_adapter_dim = 12
    return config


def batch_to_device(batch):
    return {key: value.to(DEVICE) if torch.is_tensor(value) else value for key, value in batch.items()}


class SequenceAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_zero_start_preserves_shared_weights_outputs_and_rng(self):
        config = configuration('none')
        batch = batch_to_device(collate(make_samples(config)))
        baseline = make_model(config).to(DEVICE)
        expected_state = torch.get_rng_state()
        with torch.no_grad():
            expected = baseline(batch)
        for kind in ['multiscale', 'bigru']:
            model = make_model(configuration(kind)).to(DEVICE)
            torch.testing.assert_close(torch.get_rng_state(), expected_state, rtol=0, atol=0)
            for key, value in baseline.state_dict().items():
                torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
            with torch.no_grad():
                torch.testing.assert_close(model(batch), expected, rtol=0, atol=0)

    def test_padding_length_one_batch_order_and_active_gradients(self):
        for kind in ['multiscale', 'bigru']:
            torch.manual_seed(11)
            adapter = SequenceAdapter(8, 16, kind, 12).to(DEVICE).eval()
            with torch.no_grad():
                adapter.output.weight.normal_(std=0.05)
            x = torch.randn(3, 7, 8, device=DEVICE, requires_grad=True)
            lengths = [3, 7, 1]
            out = adapter(x, lengths)
            for i, n in enumerate(lengths):
                single = adapter(x[i:i+1, :n], [n])
                torch.testing.assert_close(out[i, :n], single[0], rtol=1e-4, atol=1e-5)
                self.assertEqual(out[i, n:].count_nonzero().item(), 0)
            corrupt = x.detach().clone()
            for i, n in enumerate(lengths):
                corrupt[i, n:] = float('nan')
            torch.testing.assert_close(adapter(corrupt, lengths), out, rtol=0, atol=0)
            out.square().sum().backward()
            for p in adapter.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
            for i, n in enumerate(lengths):
                self.assertEqual(x.grad[i, n:].count_nonzero().item(), 0)

    def test_full_model_autocast_backward_and_roundtrip(self):
        for kind in ['multiscale', 'bigru']:
            for global_node in [False, True]:
                config = configuration(kind, global_node)
                model = make_model(config).to(DEVICE)
                batch = batch_to_device(collate(make_samples(config)))
                mask = batch['label_mask'].bool()
                for active in [False, True]:
                    if active:
                        with torch.no_grad():
                            model.sequence_adapter.output.weight.normal_(std=0.02)
                    model.zero_grad(set_to_none=True)
                    dtype = torch.float16 if DEVICE.startswith('cuda') else torch.bfloat16
                    with torch.autocast(device_type='cuda' if DEVICE.startswith('cuda') else 'cpu', dtype=dtype):
                        out = model(batch)
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(out[mask], batch['labels'][mask])
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    gradient = model.sequence_adapter.output.weight.grad
                    self.assertGreater(gradient.abs().sum().item(), 0)
                    for param in model.parameters():
                        if param.grad is not None:
                            self.assertTrue(torch.isfinite(param.grad).all())
                stream = io.BytesIO()
                torch.save(model.state_dict(), stream)
                stream.seek(0)
                restored = make_model(config, seed=17).to(DEVICE)
                restored.load_state_dict(torch.load(stream, weights_only=True, map_location=DEVICE), strict=True)
                with torch.no_grad():
                    torch.testing.assert_close(model(batch), restored(batch), rtol=0, atol=0)

    def test_checkpoint_kind_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'model.pt'
            for source_kind, target_kind in [('none', 'bigru'), ('bigru', 'none'), ('bigru', 'multiscale')]:
                source = make_model(configuration(source_kind))
                torch.save({'model_state_dict': source.state_dict()}, checkpoint)
                with self.assertRaises(ValueError):
                    CheckpointManager.load_checkpoint(checkpoint, make_model(configuration(target_kind)), device=torch.device('cpu'))

    def test_formal_configs_share_protocol_and_parameter_budget(self):
        directory = Path(__file__).resolve().parents[1] / 'config'
        configs = []
        counts = []
        for arm in ['control', 'multiscale', 'bigru']:
            config = yaml.safe_load((directory / f'pre_synthesis_theory_film_seq2_{arm}.yaml').read_text(encoding='utf-8'))
            self.assertEqual(config['training']['checkpoint_selection'], 'max_f1')
            self.assertFalse(config['model']['use_fragment_readout'])
            self.assertFalse(config['model']['use_residue_chemistry'])
            counts.append(sum(p.numel() for p in make_model(build_model_config(config)).parameters()))
            config['model']['sequence_adapter_type'] = 'none'
            for section, keys in {'data':['cache_dir'], 'debug':['diagnostic_dir'], 'training':['checkpoint_dir'], 'logging':['log_dir','tensorboard_log_dir'], 'evaluation':['output_metric_dir','output_pred_dir'], 'experiment':['name','description']}.items():
                for key in keys:
                    config[section].pop(key)
            configs.append(config)
        self.assertEqual(configs[0], configs[1])
        self.assertEqual(configs[0], configs[2])
        self.assertEqual(counts[1] - counts[0], 55648)
        self.assertEqual(counts[2] - counts[0], 57664)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    args, remaining = parser.parse_known_args()
    DEVICE = args.device
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
