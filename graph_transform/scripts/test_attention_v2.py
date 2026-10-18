"""Focused GATv2 tests on synthetic graphs; --device cuda adds FP16 AMP.

No training loop or production dataset is used. TF32 is disabled/restored only
inside these numerical checks; CUDA scatter roundoff is compared with tolerance.
"""

import argparse
import io
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.attention_layers import GraphAttentionLayer
from models.attention_v2 import ATTENTION_MODE_KEY, GraphAttentionV2Layer
from models.graph_transformer import GraphTransformer
from models.utils import CheckpointManager, ModelConfig, build_model_config
from test_architecture_extensions import make_config, make_samples, collate


DEVICE = 'cpu'


def configuration(mode='gat', global_node=True):
    config = make_config(global_node=global_node)
    config.gat_attention_type = mode
    return config


def make_model(config, seed=315):
    torch.manual_seed(seed)
    return GraphTransformer(config).to(DEVICE).eval()


def batch(samples):
    return {key: value.to(DEVICE) if torch.is_tensor(value) else value
            for key, value in collate(samples).items()}


def assert_outputs(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


class AttentionV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        cls.matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        cls.cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        torch.backends.cuda.matmul.allow_tf32 = cls.matmul_tf32
        torch.backends.cudnn.allow_tf32 = cls.cudnn_tf32

    def test_legacy_default_shared_initialization_and_rng(self):
        self.assertEqual(ModelConfig().gat_attention_type, 'gat')
        for mode in ('GATv2', 'unknown', None):
            with self.assertRaises(ValueError):
                ModelConfig({'gat_attention_type': mode, 'num_gat_layers': 0})
            bad = configuration(mode)
            bad.num_gat_layers = 0
            with self.assertRaises(ValueError):
                GraphTransformer(bad)
        legacy_config = configuration()
        del legacy_config.gat_attention_type
        legacy = make_model(legacy_config)
        legacy_rng = torch.get_rng_state().clone()
        explicit = make_model(configuration())
        self.assertTrue(torch.equal(legacy_rng, torch.get_rng_state()))
        self.assertEqual(set(legacy.state_dict()), set(explicit.state_dict()))
        self.assertNotIn(ATTENTION_MODE_KEY, legacy.state_dict())
        self.assertIs(type(legacy.gat_layers[0]), GraphAttentionLayer)
        for key, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, explicit.state_dict()[key]), key)
        inputs = batch(make_samples(legacy_config))
        for training in (False, True):
            legacy.train(training)
            explicit.train(training)
            torch.manual_seed(672)
            out = legacy(inputs)
            torch.manual_seed(672)
            reference = explicit(inputs)
            if DEVICE.startswith('cuda'):
                assert_outputs(out, reference)
            else:
                self.assertTrue(torch.equal(out, reference))
        v2 = make_model(configuration('gatv2'))
        self.assertTrue(torch.equal(legacy_rng, torch.get_rng_state()))
        if DEVICE.startswith('cuda'):
            # Constructors create CPU tensors; transfer to CUDA consumes no RNG.
            make_model(configuration())
            after_gat = torch.cuda.get_rng_state().clone()
            make_model(configuration('gatv2'))
            self.assertTrue(torch.equal(after_gat, torch.cuda.get_rng_state()))
        expected_new = {ATTENTION_MODE_KEY, 'gat_layers.0.a_v2'}
        self.assertEqual(set(v2.state_dict()) - set(legacy.state_dict()), expected_new)
        self.assertEqual(set(legacy.state_dict()) - set(v2.state_dict()), {'gat_layers.0.a'})
        for key in set(legacy.state_dict()) & set(v2.state_dict()):
            self.assertTrue(torch.equal(legacy.state_dict()[key], v2.state_dict()[key]), key)
        self.assertFalse(hasattr(v2.gat_layers[0], 'a'))

    def test_query_content_reverses_neighbor_ranking_and_edge_bias_position(self):
        config = configuration('gatv2')
        config.hidden_dim = 2
        config.num_attention_heads = 1
        v2 = GraphAttentionV2Layer(config).to(DEVICE).double().eval()
        old = GraphAttentionLayer(config).to(DEVICE).double().eval()
        with torch.no_grad():
            for layer in (v2, old):
                layer.W.copy_(torch.eye(2, device=DEVICE, dtype=torch.float64).unsqueeze(0))
                layer.edge_attention_bias.weight.fill_(0.5)
                layer.edge_attention_bias.bias.fill_(0.2)
            v2.a_v2.copy_(torch.tensor([[[1.0], [-1.0]]], device=DEVICE))
            old.a.copy_(torch.tensor([[[1.0], [1.0], [1.0], [-1.0]]], device=DEVICE))
        # SAME two sources and SAME edge features, only target content changes.
        nodes = torch.tensor([[1., 1.], [0., 0.], [0., -2.], [-2., 0.]],
                             dtype=torch.float64, device=DEVICE)
        edges = torch.tensor([[0, 1, 0, 1], [2, 2, 3, 3]], device=DEVICE)
        attrs = torch.full((4, 2), 0.3, dtype=torch.float64, device=DEVICE)
        _, weights = v2(nodes, edges, attrs, return_attention=True)
        _, old_weights = old(nodes, edges, attrs, return_attention=True)
        self.assertGreater(float(weights[0].detach()), float(weights[1].detach()))
        self.assertLess(float(weights[2].detach()), float(weights[3].detach()))
        self.assertGreater(float(old_weights[0].detach()), float(old_weights[1].detach()))
        self.assertGreater(float(old_weights[2].detach()), float(old_weights[3].detach()))
        expected = torch.tensor([[1.2, 0.4], [-1.2, -0.4]], dtype=torch.float64,
                                device=DEVICE).softmax(dim=1).reshape(4, 1)
        torch.testing.assert_close(weights, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(v2.get_attention_weights(nodes, edges, attrs), weights)
        # Nonuniform edge bias distinguishes AFTER-score placement from the
        # legacy inside-LeakyReLU placement. Softmax groups by destination.
        edge_bias = torch.tensor([[-4.], [1.], [2.], [-3.]], device=DEVICE, dtype=torch.float64)
        projected = nodes.unsqueeze(1)
        actual = v2._compute_attention(projected[edges[0]], projected[edges[1]],
                                       edges[1], edge_bias, 4)
        expected = (torch.tensor([[1.2, 0.4], [-1.2, -0.4]], device=DEVICE,
                                 dtype=torch.float64) + edge_bias.reshape(2, 2)).softmax(1)
        torch.testing.assert_close(actual, expected.reshape(4, 1), rtol=1e-12, atol=1e-12)
        empty = v2._compute_attention(projected[:0], projected[:0], edges[1, :0], None, 4)
        self.assertEqual(empty.shape, (0, 1))
        v2.reset_parameters()
        self.assertTrue(torch.isfinite(v2.a_v2).all())
        self.assertFalse(hasattr(v2, 'a'))

    def test_graph_batch_order_padding_observed_invariance_and_gradients(self):
        for global_node in (False, True):
            config = configuration('gatv2', global_node)
            model = make_model(config)
            samples = make_samples(config)
            inputs = batch(samples)
            mask = inputs['label_mask'].bool()
            output = model(inputs)
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(torch.equal(output[~mask], torch.zeros_like(output[~mask])))
            with torch.no_grad():
                repeated = model(inputs)
                reordered = model(batch([samples[2], samples[0], samples[1]]))
                changed = model(batch(make_samples(config, observed_scale=1000)))
                assert_outputs(output, repeated)
                assert_outputs(output, reordered[[1, 2, 0]])
                assert_outputs(output, changed)
                for index, sample in enumerate(samples):
                    single = model(batch([sample]))
                    assert_outputs(output[index, :sample['seq_len'] - 1], single[0])
            output.retain_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output[mask], inputs['labels'][mask])
            loss.backward()
            self.assertTrue(torch.equal(output.grad[~mask], torch.zeros_like(output.grad[~mask])))
            layer = model.gat_layers[0]
            for parameter in (layer.a_v2, layer.W, layer.edge_message_gate.weight,
                              layer.edge_attention_bias.weight, model.bond_head[0].weight):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_amp_backward_finite(self):
        config = configuration('gatv2')
        model = make_model(config).train()
        inputs = batch(make_samples(config))
        mask = inputs['label_mask'].bool()
        device_type = 'cuda' if DEVICE.startswith('cuda') else 'cpu'
        dtype = torch.float16 if device_type == 'cuda' else torch.bfloat16
        with torch.autocast(device_type=device_type, dtype=dtype):
            output = model(inputs)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output[mask], inputs['labels'][mask])
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(model.gat_layers[0].a_v2.grad.abs().sum()), 0)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_checkpoint_modes_legacy_roundtrip_and_rejection_before_copy(self):
        with tempfile.TemporaryDirectory(prefix='dbond_gatv2_') as directory:
            path = Path(directory) / 'checkpoint.pt'
            for mode in ('gat', 'gatv2'):
                config = configuration(mode)
                source = make_model(config)
                stream = io.BytesIO()
                torch.save(source.state_dict(), stream)
                stream.seek(0)
                state = torch.load(stream, weights_only=True, map_location=DEVICE)
                target = make_model(config, seed=77)
                target.load_state_dict(state, strict=True)
                for key, value in state.items():
                    self.assertTrue(torch.equal(value, target.state_dict()[key]), key)
                assert_outputs(source(batch(make_samples(config))), target(batch(make_samples(config))))
                for saved_config in (None, config.to_dict(), {'model': config.to_dict()}):
                    torch.save({'model_state_dict': state, 'config': saved_config}, path)
                    CheckpointManager.load_checkpoint(path, target, device=torch.device(DEVICE))
                    wrong = make_model(configuration('gatv2' if mode == 'gat' else 'gat'), seed=90)
                    untouched = {key: value.clone() for key, value in wrong.state_dict().items()}
                    for _ in range(2):
                        with self.assertRaises(ValueError):
                            CheckpointManager.load_checkpoint(path, wrong, device=torch.device(DEVICE))
                        with self.assertRaises(RuntimeError):
                            wrong.load_state_dict(state, strict=False)
                        for key, value in untouched.items():
                            self.assertTrue(torch.equal(value, wrong.state_dict()[key]), key)
                conflicting_config = config.to_dict()
                conflicting_config['gat_attention_type'] = 'gatv2' if mode == 'gat' else 'gat'
                torch.save({'model_state_dict': state, 'config': conflicting_config}, path)
                with self.assertRaises(ValueError):
                    CheckpointManager.load_checkpoint(path, target, device=torch.device(DEVICE))
            legacy = make_model(configuration())
            legacy_config = legacy.config.to_dict()
            legacy_config.pop('gat_attention_type')
            torch.save({'model_state_dict': legacy.state_dict(), 'config': legacy_config}, path)
            CheckpointManager.load_checkpoint(path, legacy, device=torch.device(DEVICE))
            v2 = make_model(configuration('gatv2'))
            malformed = []
            for marker in (None, torch.tensor(1.), torch.tensor([1]), torch.tensor(0)):
                state = dict(v2.state_dict())
                if marker is None:
                    state.pop(ATTENTION_MODE_KEY)
                else:
                    state[ATTENTION_MODE_KEY] = marker
                malformed.append(state)
            state = dict(v2.state_dict())
            state.pop('gat_layers.0.a_v2')
            malformed.append(state)
            state = dict(v2.state_dict())
            state['gat_layers.0.a_v2'] = torch.zeros(1)
            malformed.append(state)
            for state in malformed:
                torch.save({'model_state_dict': state}, path)
                with self.assertRaises(ValueError):
                    CheckpointManager.load_checkpoint(path, v2, device=torch.device(DEVICE))
                with self.assertRaises(RuntimeError):
                    v2.load_state_dict(state, strict=True)

    def test_formal_configs_only_declared_changes_and_parameter_counts(self):
        directory = Path(__file__).resolve().parents[1] / 'config'
        normalized, counts, states, rng_states = [], [], [], []
        for arm, mode in (('control', 'gat'), ('gatv2', 'gatv2')):
            config = yaml.safe_load((directory / f'pre_synthesis_theory_film_attn6_{arm}.yaml').read_text(encoding='utf-8'))
            self.assertEqual(config['model']['gat_attention_type'], mode)
            self.assertEqual(config['training']['checkpoint_selection'], 'max_f1')
            self.assertEqual(config['training'].get('pretrained_transfer_scope', 'none'), 'none')
            self.assertIsNone(config['training'].get('pretrained_backbone_path'))
            self.assertIsNone(config['training'].get('pretrained_backbone_sha256'))
            model = make_model(build_model_config(config))
            counts.append(sum(parameter.numel() for parameter in model.parameters()))
            states.append(model.state_dict())
            rng_states.append(torch.get_rng_state().clone())
            config['model'].pop('gat_attention_type')
            for section, keys in {
                'data': ['cache_dir'], 'debug': ['diagnostic_dir'], 'training': ['checkpoint_dir'],
                'logging': ['log_dir', 'tensorboard_log_dir'],
                'evaluation': ['output_metric_dir', 'output_pred_dir'],
                'experiment': ['name', 'description'],
            }.items():
                for key in keys:
                    config[section].pop(key)
            normalized.append(config)
        self.assertEqual(normalized[0], normalized[1])
        self.assertEqual(counts, [1706153, 1705641])
        self.assertTrue(torch.equal(rng_states[0], rng_states[1]))
        for key in set(states[0]) & set(states[1]):
            self.assertTrue(torch.equal(states[0][key], states[1][key]), key)
        self.assertEqual(set(states[0]) - set(states[1]), {'gat_layers.0.a', 'gat_layers.1.a'})
        self.assertEqual(set(states[1]) - set(states[0]), {
            ATTENTION_MODE_KEY, 'gat_layers.0.a_v2', 'gat_layers.1.a_v2'})
        print(f'Parameter counts: gat={counts[0]}, gatv2={counts[1]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    args, remaining = parser.parse_known_args()
    DEVICE = args.device
    if DEVICE.startswith('cuda') and not torch.cuda.is_available():
        raise SystemExit('CUDA was requested but is unavailable')
    unittest.main(argv=[sys.argv[0]] + remaining)
