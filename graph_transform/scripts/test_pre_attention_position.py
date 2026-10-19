"""Position routing, packed alignment and checkpoint regression tests (run on pod)."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.graph_transformer import GraphTransformer
from models.position_encoding import add_packed_residue_positions
from models.utils import ModelConfig
from test_architecture_extensions import make_config, make_samples, collate


def configuration(global_node=True):
    c = make_config(global_node=global_node)
    c.position_embedding_mode = 'pre_attention_add'
    c.num_gat_layers = 2
    return c


class PreAttentionPositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_configuration_rejects_silent_noops(self):
        self.assertEqual(ModelConfig().position_embedding_mode, 'input_concat')
        with self.assertRaises(ValueError):
            ModelConfig({'position_embedding_mode': 'invalid'})
        for key, value in [('num_gat_layers', 0), ('use_position_embedding', False)]:
            c = configuration()
            setattr(c, key, value)
            with self.assertRaises(ValueError):
                GraphTransformer(c)

    def test_legacy_default_schema_and_rng_unchanged(self):
        c = make_config()
        del c.position_embedding_mode
        torch.manual_seed(67)
        m0 = GraphTransformer(c)
        rng = torch.get_rng_state().clone()
        c.position_embedding_mode = 'input_concat'
        torch.manual_seed(67)
        m1 = GraphTransformer(c)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(set(m0.state_dict()), set(m1.state_dict()))
        for key, tensor in m0.state_dict().items():
            self.assertTrue(torch.equal(tensor, m1.state_dict()[key]), key)
        self.assertNotIn('position_embedding_mode_code', m0.state_dict())

    def test_packed_positions_exclude_global_and_keep_gradients(self):
        for global_node in (False, True):
            lens = torch.tensor([2, 4])
            node_lens = lens + int(global_node)
            positions = torch.arange(12, dtype=torch.float32).reshape(4, 3).requires_grad_()
            h = torch.ones(int(node_lens.sum()), 3, requires_grad=True)
            out = add_packed_residue_positions(h, positions, lens, node_lens)
            offset = 0
            for length in lens.tolist():
                torch.testing.assert_close(out[offset:offset+length], 1 + positions[:length])
                if global_node:
                    torch.testing.assert_close(out[offset+length], h[offset+length])
                offset += length + int(global_node)
            out.sum().backward()
            torch.testing.assert_close(positions.grad, torch.tensor([[2.]*3, [2.]*3, [1.]*3, [1.]*3]))

    def test_injected_once_after_gcn_before_first_gat(self):
        for global_node in (False, True):
            c = configuration(global_node)
            m = GraphTransformer(c).eval()
            batch = collate(make_samples(c))
            seen = {}
            def capture(key, input_value=False):
                def hook(module, args, output=None):
                    seen[key] = (args[0] if input_value else output).detach().clone()
                return hook
            hooks = [m.gcn_layers[0].register_forward_pre_hook(capture('gcn_in', True)),
                     m.gcn_layers[-1].register_forward_hook(capture('gcn_out')),
                     m.gat_layers[0].register_forward_pre_hook(capture('gat_in', True)),
                     m.gat_layers[0].register_forward_hook(capture('gat_out')),
                     m.gat_layers[1].register_forward_pre_hook(capture('gat2_in', True))]
            with torch.no_grad():
                m(batch)
                first_gcn = seen['gcn_in'].clone()
                positions = m.pre_attention_position_proj(m.node_encoder.position_embedding(
                    torch.arange(max(map(len, batch['sequences'])))))
                expected = add_packed_residue_positions(seen['gcn_out'], positions, batch['seq_lens'], batch['node_lens'])
                torch.testing.assert_close(seen['gat_in'], expected)
                torch.testing.assert_close(seen['gat2_in'], seen['gat_out'])
                self.assertGreater((seen['gat_in']-seen['gcn_out']).abs().max().item(), 0.)
                m.node_encoder.position_embedding.weight.zero_()
                m(batch)
                torch.testing.assert_close(seen['gcn_in'], first_gcn, rtol=0, atol=0)
                torch.testing.assert_close(seen['gat_in'], seen['gcn_out'], rtol=0, atol=0)
            for hook in hooks:
                hook.remove()
            self.assertEqual(m.node_encoder.node_encoder[0].in_features,
                             c.aa_embedding_dim+c.physicochemical_dim+64)

    def test_variable_length_batch_invariance_and_padding(self):
        for global_node in (False, True):
            c = configuration(global_node)
            m = GraphTransformer(c).eval()
            samples = make_samples(c)
            batch = collate(samples)
            with torch.no_grad():
                together = m(batch)
                self.assertTrue(torch.isfinite(together).all())
                self.assertTrue((together[~batch['label_mask'].bool()] == 0).all())
                for i, sample in enumerate(samples):
                    alone = m(collate([sample]))
                    torch.testing.assert_close(together[i, :len(sample['sequence'])-1], alone[0], rtol=1e-5, atol=1e-6)

    def test_checkpoint_roundtrip_and_cross_mode_rejected_before_mutation(self):
        legacy_config = make_config()
        legacy_config.num_gat_layers = 2
        legacy = GraphTransformer(legacy_config)
        late = GraphTransformer(configuration())
        for model, wrong in [(legacy, late.state_dict()), (late, legacy.state_dict())]:
            before = deepcopy(model.state_dict())
            with self.assertRaisesRegex(RuntimeError, 'position_embedding_mode'):
                model.load_state_dict(wrong, strict=False)
            for key, tensor in before.items():
                self.assertTrue(torch.equal(model.state_dict()[key], tensor), key)
        copy = GraphTransformer(configuration())
        copy.load_state_dict(late.state_dict())
        broken = deepcopy(late.state_dict())
        broken['position_embedding_mode_code'] = torch.tensor(2)
        with self.assertRaisesRegex(RuntimeError, 'position_embedding_mode_code'):
            copy.load_state_dict(broken, strict=False)
        broken = deepcopy(late.state_dict())
        del broken['pre_attention_position_proj.weight']
        with self.assertRaisesRegex(RuntimeError, 'position projection'):
            copy.load_state_dict(broken, strict=False)


if __name__ == '__main__':
    unittest.main()
