"""Bond-local cross-attention routing and legacy regression checks; run on pod."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.bond_cross_attention import BondCrossAttentionHead
from models.graph_transformer import GraphTransformer
from models.utils import ModelConfig
from test_architecture_extensions import make_config, make_samples, collate


def config(cross=True, global_node=True):
    c = make_config(global_node=global_node)
    c.bond_head_type = 'cross_attention_mlp' if cross else 'concat_mlp'
    return c


class BondCrossAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_explicit_cross_attention_math_and_source_gradients(self):
        torch.manual_seed(15)
        head = BondCrossAttentionHead(16, 4, 0., [0, 1, 2, 3]).eval()
        parts = [torch.randn(3, 16, requires_grad=True) for _ in range(6)]
        q = torch.stack(parts[:2], dim=1)
        memory = torch.stack(parts[2:], dim=1) + head.memory_type_embedding
        attention = head.cross_attention
        wq, wk, wv = attention.in_proj_weight.chunk(3)
        bq, bk, bv = attention.in_proj_bias.chunk(3)
        def split(value):
            return value.reshape(3, -1, 4, 4).transpose(1, 2)
        projected_q = split(F.linear(q, wq, bq))
        projected_k = split(F.linear(memory, wk, bk))
        projected_v = split(F.linear(memory, wv, bv))
        weights = (projected_q @ projected_k.transpose(-2, -1) / 2).softmax(-1)
        context = (weights @ projected_v).transpose(1, 2).reshape(3, 2, 16)
        reference = head.mlp(head.layer_norm(q + attention.out_proj(context)).flatten(1))
        actual = head(parts)
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
        actual.square().sum().backward()
        for value in parts + [head.memory_type_embedding, attention.in_proj_weight]:
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0.)

    def test_no_attention_between_different_bonds(self):
        head = BondCrossAttentionHead(16, 4, 0., [0, 1, 2, 3]).eval()
        parts = [torch.randn(5, 16) for _ in range(6)]
        with torch.no_grad():
            expected = head(parts)
            independently = torch.cat([head([p[i:i+1] for p in parts]) for i in range(5)])
            torch.testing.assert_close(expected, independently, rtol=1e-5, atol=1e-6)
            order = torch.tensor([3, 0, 4, 1, 2])
            torch.testing.assert_close(head([p[order] for p in parts]), expected[order], rtol=1e-5, atol=1e-6)

    def test_default_legacy_and_shared_backbone_rng(self):
        self.assertEqual(ModelConfig().bond_head_type, 'concat_mlp')
        c = config(False)
        del c.bond_head_type
        torch.manual_seed(23)
        old = GraphTransformer(c)
        old_rng = torch.get_rng_state().clone()
        c.bond_head_type = 'concat_mlp'
        torch.manual_seed(23)
        explicit = GraphTransformer(c)
        self.assertEqual(set(old.state_dict()), set(explicit.state_dict()))
        for k, v in old.state_dict().items():
            self.assertTrue(torch.equal(v, explicit.state_dict()[k]), k)
        torch.manual_seed(23)
        cross = GraphTransformer(config())
        self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
        for k, v in old.state_dict().items():
            if not k.startswith('bond_head.'):
                self.assertTrue(torch.equal(v, cross.state_dict()[k]), k)
        self.assertNotIn('bond_head.0.weight', cross.state_dict())

    def test_model_batch_padding_and_observed_feature_invariance(self):
        for global_node in (False, True):
            c = config(global_node=global_node)
            model = GraphTransformer(c).eval()
            samples = make_samples(c)
            batch = collate(samples)
            with torch.no_grad():
                logits = model(batch)
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue((logits[~batch['label_mask'].bool()] == 0).all())
                torch.testing.assert_close(logits, model(collate(make_samples(c, observed_scale=999))), rtol=0, atol=0)
                for i, sample in enumerate(samples):
                    independent = model(collate([sample]))
                    torch.testing.assert_close(logits[i, :len(sample['sequence'])-1], independent[0], rtol=1e-5, atol=1e-6)

    def test_cpu_amp_and_finite_full_model_gradient(self):
        c = config()
        model = GraphTransformer(c)
        batch = collate(make_samples(c))
        with torch.autocast('cpu', dtype=torch.bfloat16):
            logits = model(batch)
            loss = logits[batch['label_mask'].bool()].square().mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for value in (model.bond_head.cross_attention.in_proj_weight,
                      model.bond_head.memory_type_embedding,
                      model.bond_theory_proj.weight, model.gat_layers[0].W):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0.)

    def test_configuration_rejects_invalid_or_unsupported_combinations(self):
        with self.assertRaises(ValueError):
            ModelConfig({'bond_head_type': 'unknown'})
        for field, value in [('bond_cross_attention_heads', 3), ('bond_cross_attention_heads', 0),
                             ('bond_condition_mode', 'concat'), ('use_ion12_aux_head', True)]:
            c = config()
            setattr(c, field, value)
            with self.assertRaises(ValueError):
                GraphTransformer(c)
        c = config()
        for field in ('bond_use_edge_repr', 'bond_use_diff_feature', 'bond_use_product_feature', 'use_theory_features'):
            setattr(c, field, False)
        with self.assertRaisesRegex(ValueError, 'memory source'):
            GraphTransformer(c)
        c.use_theory_features = True
        self.assertEqual(GraphTransformer(c).bond_head.memory_role_ids.tolist(), [3])

    def test_checkpoint_rejects_cross_mode_partial_or_semantic_mismatch_before_mutation(self):
        legacy, cross = GraphTransformer(config(False)), GraphTransformer(config())
        copy = GraphTransformer(config())
        copy.load_state_dict(cross.state_dict())
        cases = [(legacy, cross.state_dict()), (cross, legacy.state_dict())]
        for key, value in [('bond_head_mode_code', torch.tensor(2)),
                           ('bond_head.num_heads_code', torch.tensor(2)),
                           ('bond_head.memory_role_ids', torch.tensor([1, 0, 2, 3]))]:
            wrong = deepcopy(cross.state_dict())
            wrong[key] = value
            cases.append((cross, wrong))
        wrong = deepcopy(cross.state_dict())
        del wrong['bond_head.cross_attention.in_proj_weight']
        cases.append((cross, wrong))
        for model, state in cases:
            before = deepcopy(model.state_dict())
            with self.assertRaisesRegex(RuntimeError, 'bond.head|bond_head'):
                model.load_state_dict(state, strict=False)
            for k, v in before.items():
                self.assertTrue(torch.equal(v, model.state_dict()[k]), k)


if __name__ == '__main__':
    unittest.main()
