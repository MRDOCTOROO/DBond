"""Pod-only bounded preflight for four-state terminal fusion heads.

No epochs or outer-test metrics. Full-model empty-edge GAT behavior is outside
scope; mixed short chains and head-only empty valid-bond tensors are checked.
"""
import argparse
import copy
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
import types

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'graph_transform'), str(ROOT / 'graph_transform/scripts')]
from data.row_validation import file_sha
from models.graph_transformer import GraphTransformer
from models.bond_fusion_heads import validate_bond_fusion_checkpoint
from models.four_state import union_logit
from models.utils import build_model_config
from training.four_state_loss import four_state_cross_entropy
from training.loss_functions import BinaryBondLoss
from training.trainer import Trainer
from test_four_state_weight import check_loss
from test_four_state_width import (collate, longest_indices, masked_metric_check,
                                  memory, to_cuda, topology_proof)
from test_four_state_dynamic_edges import seed, rng_state, same_rng, cpu_state, same_tensors, toy_batch
from train_graph_model import load_config, create_datasets, create_optimizer, create_scheduler
from train_5fold import make_fold_config, _stable_fold_seed
import train_graph_model as train_script

MODES = ('concat', 'cross_attention', 'gated', 'residual_mlp')
FOLDS = ('1222', '2252', '3514', '6072', '9075')
LEGACY_REF = 'fb30252'
LEGACY_FILE = 'graph_transform/models/graph_transformer.py'
PRIOR = Path('/mnt/pvc/graphtrans/four_state_ops/smoke.json')
WIDTH_PRIOR = Path('/mnt/pvc/graphtrans/four_state_width_ops/smoke.json')
LOGGER = logging.getLogger('four_state_head_fusion_preflight')
train_script.logger = LOGGER


def config_path(mode):
    return ROOT / f'graph_transform/config/pre_synthesis_four_state_head_fusion_{mode}.yaml'


def normalized(config, identity):
    result = copy.deepcopy(config)
    result['experiment'].pop('description', None)
    result['model'].pop('bond_fusion_mode', None)
    result['model'].setdefault('bond_cross_attention_heads', 4)
    result['model']['edge_state_mode'] = 'static'
    result['model']['theory_fusion_stage'] = 'late'
    result['model'].setdefault('use_charge_partition_features', False)
    result['loss'].setdefault('four_state_conditional_weight', 1.)
    result['loss'].setdefault('four_state_union_brier_weight', 0.)
    result['training'].pop('validation_split_base_seed', None)
    result['training'].pop('validation_split_seed', None)
    return json.dumps(result, sort_keys=True).replace(identity, 'four_state_HEAD_FUSION')


def legacy_class():
    source = subprocess.check_output(['git', 'show', f'{LEGACY_REF}:{LEGACY_FILE}'], cwd=ROOT)
    revision = subprocess.check_output(['git', 'rev-parse', LEGACY_REF], cwd=ROOT).decode().strip()
    module = types.ModuleType('models._legacy_head_fusion_reference')
    module.__package__ = 'models'
    module.__file__ = f'git:{revision}:{LEGACY_FILE}'
    sys.modules[module.__name__] = module
    exec(compile(source.decode('utf-8'), module.__file__, 'exec'), module.__dict__)
    return module.GraphTransformer, {'commit': revision, 'path': LEGACY_FILE,
                                   'sha256': hashlib.sha256(source).hexdigest()}


def legacy_compatibility(config, batch):
    Legacy, proof = legacy_class()
    seed()
    old = Legacy(build_model_config(config)).cpu()
    expected_rng = rng_state()
    seed()
    current = GraphTransformer(build_model_config(config)).cpu()
    same_rng(expected_rng, rng_state())
    same_tensors(cpu_state(old), cpu_state(current))
    current.load_state_dict(cpu_state(old), strict=True)
    results = []
    for model in (old, current):
        model.train()
        model.zero_grad(set_to_none=True)
        seed()
        output = model(batch)
        states = model.last_four_state_logits
        output[batch['label_mask'].bool()].square().mean().backward()
        gradients = {name: value.grad.detach().clone() for name, value in model.named_parameters()
                     if value.grad is not None}
        results.append((output.detach(), states.detach(), gradients, rng_state()))
    assert torch.equal(results[0][0], results[1][0]), 'Frozen concat predictions changed'
    assert torch.equal(results[0][1], results[1][1]), 'Frozen four-state logits changed'
    same_tensors(results[0][2], results[1][2])
    same_rng(results[0][3], results[1][3])
    proof.update(initial_state_and_schema_exact=True, initialization_rng_exact=True,
        train_forward_backward_exact=True, forward_rng_exact=True,
        old_checkpoint_strict_load=True)
    return proof


def backbone_state(state):
    return {name: value for name, value in state.items()
            if not name.startswith('bond_head.') and not name.startswith('bond_fusion_')}


def legacy_binary_heads_check(config, batch):
    """The already-tested binary concat and binary cross routes remain readable."""
    Legacy, reference = legacy_class()
    proof = {'reference': reference, 'heads': {}}
    for head_type in ('concat_mlp', 'cross_attention_mlp'):
        runtime = copy.deepcopy(config)
        runtime['model']['bond_head_type'] = head_type
        runtime['model']['bond_fusion_mode'] = 'concat'
        seed()
        old = Legacy(build_model_config(runtime)).cpu().eval()
        expected_rng = rng_state()
        seed()
        current = GraphTransformer(build_model_config(runtime)).cpu().eval()
        same_rng(expected_rng, rng_state())
        same_tensors(cpu_state(old), cpu_state(current))
        current.load_state_dict(cpu_state(old), strict=True)
        validate_bond_fusion_checkpoint(current, cpu_state(old), runtime)
        with torch.no_grad():
            assert torch.equal(old(batch), current(batch)), f'Existing binary route changed: {head_type}'
        proof['heads'][head_type] = {'schema_state_and_rng_exact': True,
            'strict_old_checkpoint_load': True, 'cpu_eval_forward_exact': True}
    return proof


def output_layer(head):
    candidates = [module for module in head.modules()
                  if isinstance(module, nn.Linear) and module.out_features == 4]
    assert len(candidates) == 1, 'Every four-state head must have one explicit four-logit output'
    return candidates[0]


def feature_parts(value):
    if isinstance(value, (list, tuple)):
        assert len(value) == 6
        return list(value)
    assert value.ndim == 2 and value.shape[-1] == 1536
    return list(value.chunk(6, dim=-1))


def fusion_call(model, parts):
    """Execute the real constituent modules for head-only batch-isolation tests."""
    mode, head = model.bond_fusion_mode, model.bond_head
    if mode == 'cross_attention':
        return head(parts)
    x = torch.cat(parts, dim=-1)
    if mode == 'gated':
        gates = 6 * model.bond_fusion_gate(x).softmax(-1)
        x = (torch.stack(parts, dim=1) * gates.unsqueeze(-1)).flatten(start_dim=1)
    elif mode == 'residual_mlp':
        z = head[1](head[0](x))
        z = model.bond_fusion_residual(z)
        return head[3](head[2](z))
    return head(x)


def linear_mlp_reference(mlp, x):
    assert not mlp.training
    x = F.relu(F.linear(x, mlp[0].weight, mlp[0].bias))
    return F.linear(x, mlp[3].weight, mlp[3].bias)


def attention_reference(head, parts):
    """Literal Q/K/V projections, scaled dot-product softmax and ordered pooling."""
    query = torch.stack(parts[:2], dim=1)
    memory_ = torch.stack(parts[2:], dim=1) + head.memory_type_embedding.unsqueeze(0)
    attention = head.cross_attention
    q_w, k_w, v_w = attention.in_proj_weight.chunk(3, dim=0)
    q_b, k_b, v_b = attention.in_proj_bias.chunk(3, dim=0)
    q = F.linear(query, q_w, q_b)
    k, v = F.linear(memory_, k_w, k_b), F.linear(memory_, v_w, v_b)
    count, query_count, hidden = q.shape
    heads = attention.num_heads
    width = hidden // heads
    q = q.reshape(count, query_count, heads, width).transpose(1, 2)
    k = k.reshape(count, 4, heads, width).transpose(1, 2)
    v = v.reshape(count, 4, heads, width).transpose(1, 2)
    weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(width), dim=-1)
    mixed = torch.matmul(weights, v).transpose(1, 2).reshape(count, query_count, hidden)
    mixed = F.linear(mixed, attention.out_proj.weight, attention.out_proj.bias)
    normalized = F.layer_norm(query + mixed, head.layer_norm.normalized_shape,
        head.layer_norm.weight, head.layer_norm.bias, head.layer_norm.eps)
    return linear_mlp_reference(head.mlp, normalized.flatten(start_dim=1)), weights


def head_reference(model, parts):
    mode, head = model.bond_fusion_mode, model.bond_head
    x = torch.cat(parts, dim=-1)
    if mode == 'concat':
        return linear_mlp_reference(head, x)
    if mode == 'cross_attention':
        return attention_reference(head, parts)[0]
    if mode == 'gated':
        gate = model.bond_fusion_gate
        gates = 6 * torch.softmax(F.linear(x, gate.weight, gate.bias), dim=-1)
        weighted = torch.stack(parts, dim=1) * gates.unsqueeze(-1)
        return linear_mlp_reference(head, weighted.flatten(start_dim=1))
    base, residual = head, model.bond_fusion_residual
    z = F.relu(F.linear(x, base[0].weight, base[0].bias))
    normalized = F.layer_norm(z, residual.normalization.normalized_shape,
        residual.normalization.weight, residual.normalization.bias, residual.normalization.eps)
    correction = F.linear(F.relu(F.linear(normalized,
        residual.down.weight, residual.down.bias)), residual.up.weight, residual.up.bias)
    return F.linear(z + correction, base[3].weight, base[3].bias)


def nonzero_fixture(model):
    """Only a CPU test replica is perturbed; production starts remain untouched."""
    with torch.no_grad():
        target = (model.bond_fusion_gate.weight if model.bond_fusion_mode == 'gated'
                  else model.bond_fusion_residual.up.weight)
        target.copy_(torch.linspace(-.01, .01, target.numel()).reshape_as(target))


def head_unit_check(model, parts):
    model.eval()
    mode, head = model.bond_fusion_mode, model.bond_head
    parts = [value[:8].detach().clone() for value in parts]
    assert len(parts) == 6 and parts[0].shape == (8, 256)
    with torch.no_grad():
        original = fusion_call(model, parts)
        reference = head_reference(model, parts)
        assert torch.allclose(original, reference, atol=3e-6, rtol=1e-5)
        if mode == 'gated':
            gate = model.bond_fusion_gate
            assert torch.count_nonzero(gate.weight) == torch.count_nonzero(gate.bias) == 0
            gates = 6 * torch.softmax(gate(torch.cat(parts, dim=-1)), dim=-1)
            assert torch.equal(gates, torch.ones_like(gates))
            assert torch.equal(original, linear_mlp_reference(head, torch.cat(parts, dim=-1)))
        elif mode == 'residual_mlp':
            residual = model.bond_fusion_residual
            assert torch.count_nonzero(residual.up.weight) == torch.count_nonzero(residual.up.bias) == 0
            assert torch.equal(original, linear_mlp_reference(head, torch.cat(parts, dim=-1)))
        if mode in ('gated', 'residual_mlp'):
            nonzero_fixture(model)
        baseline = fusion_call(model, parts)
        reference = head_reference(model, parts)
        assert torch.allclose(baseline, reference, atol=3e-6, rtol=1e-5)
        # Every bond is an independent batch item; alter one bond only.
        changed = [value.clone() for value in parts]
        changed[2][0] += torch.linspace(-.5, .4, 256)
        perturbed = fusion_call(model, changed)
        assert torch.equal(baseline[1:], perturbed[1:]), 'Head mixes different bond/sample items'
        assert not torch.equal(baseline[0], perturbed[0]), 'Edge source does not reach head output'
        individual = torch.cat([fusion_call(model, [p[i:i+1] for p in parts]) for i in range(8)])
        assert torch.allclose(baseline, individual, atol=3e-6, rtol=1e-5)
        permuted = list(parts)
        permuted[2], permuted[3] = permuted[3], permuted[2]
        assert not torch.equal(baseline, fusion_call(model, permuted)), 'Source identity/order lost'
        empty = [value[:0] for value in parts]
        assert fusion_call(model, empty).shape == (0, 4)
        if mode == 'cross_attention':
            weights = attention_reference(head, parts)[1]
            assert weights.shape == (8, 4, 2, 4)
            assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)))
    differentiable = [value.clone().requires_grad_(True) for value in parts]
    output = fusion_call(model, differentiable)
    objective = (output * torch.tensor([.7, 1.1, 1.3, 1.9])).square().mean()
    gradients = torch.autograd.grad(objective, differentiable)
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    return {'independent_functional_formula': True, 'all_six_sources_receive_finite_gradient': True,
        'local_valid_bond_batch_isolation': True, 'ordered_source_roles_distinguishable': True,
        'head_only_empty_valid_bonds_supported': True,
        'source_gradient_l1': [float(g.abs().sum()) for g in gradients]}


def trace(model, batch):
    record = {'nodes': [], 'edge': [], 'theory': [], 'head': []}
    def capture_head(module, args):
        record['head'].append([value.detach().clone() for value in feature_parts(args[0])])
    handles = [model.gat_layers[-1].register_forward_hook(
        lambda module, args, output: record['nodes'].append(output.detach().clone())),
        model.edge_encoder.register_forward_hook(
        lambda module, args, output: record['edge'].append(output.detach().clone())),
        model.bond_theory_proj.register_forward_hook(
        lambda module, args, output: record['theory'].append(output.detach().clone()))]
    point_of_use = (model.bond_fusion_gate if model.bond_fusion_mode == 'gated'
                    else model.bond_head[0] if model.bond_fusion_mode == 'residual_mlp'
                    else model.bond_head)
    handles.append(point_of_use.register_forward_pre_hook(capture_head))
    try:
        with torch.no_grad():
            prediction = model(batch)
    finally:
        for handle in handles:
            handle.remove()
    return prediction.detach().clone(), record


def no_bond_batch():
    """Two L1 peptides retain global edges, so the legacy GAT never sees E0."""
    return {'sequences': ['A', 'A'], 'seq_lens': torch.tensor([1, 1]),
        'node_lens': torch.tensor([2, 2]), 'max_node_len': 2, 'max_bonds': 0,
        'edge_index': torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
        'edge_types': torch.full((4,), 4), 'edge_distances': torch.ones(4, dtype=torch.long),
        'edge_attr': torch.zeros(4, 8), 'bond_edge_map': torch.empty(0, dtype=torch.long),
        'bond_theory': torch.empty(2, 0, 15), 'charges': torch.full((2,), 2.),
        'nces': torch.full((2,), 30.), 'state_vars': torch.tensor([[2., 200., 0.]]).repeat(2, 1),
        'env_vars': torch.tensor([[30., 0.]]).repeat(2, 1),
        'label_mask': torch.zeros(2, 0, dtype=torch.bool), 'labels': torch.zeros(2, 0)}


def routing_check(model, batch):
    assert next(model.parameters()).device.type == 'cpu'
    model.eval()
    features = {key: value for key, value in batch.items()
                if key not in ('labels', 'label_mask', 'soft_labels', 'four_state_targets')}
    prediction, record = trace(model, features)
    assert all(len(record[key]) == 1 for key in record)
    parts, nodes, edges = record['head'][0], record['nodes'][0], record['edge'][0]
    left, right = [], []
    offset = 0
    for length, node_length in zip(batch['seq_lens'].tolist(), batch['node_lens'].tolist()):
        for position in range(max(length - 1, 0)):
            left.append(offset + position)
            right.append(offset + position + 1)
        offset += node_length
    left, right = torch.tensor(left), torch.tensor(right)
    mapping = batch['bond_edge_map']
    assert torch.equal(batch['edge_index'][0, mapping], left)
    assert torch.equal(batch['edge_index'][1, mapping], right)
    expected = [nodes[left], nodes[right], edges[mapping],
                nodes[left] - nodes[right], nodes[left] * nodes[right], record['theory'][0]]
    assert all(torch.equal(value, reference) for value, reference in zip(parts, expected)), 'Six-route order changed'
    independent_logits = head_reference(model, parts)
    assert torch.allclose(prediction[batch['label_mask'].bool()], union_logit(independent_logits),
                          atol=3e-6, rtol=1e-5), 'Production terminal fusion differs from independent reference'
    poisoned = dict(features)
    poisoned['bond_theory'] = batch['bond_theory'].clone()
    poisoned['bond_theory'][~batch['label_mask'].bool()] = float('nan')
    poisoned_prediction, _ = trace(model, poisoned)
    assert torch.equal(prediction, poisoned_prediction), 'Invalid theoretical padding enters fusion'
    for fixture in (toy_batch(), no_bond_batch()):
        output, _ = trace(model, fixture)
        assert output.shape == fixture['label_mask'].shape and torch.isfinite(output).all()
    return parts, {'actual_six_source_route_order_verified': True,
        'theory_projected_once_after_graph': True, 'head_receives_e0': True,
        'target_fields_not_used': True, 'padding_poison_invariant': True,
        'mixed_l0_l1_l2_nonempty_graph_finite': True,
        'empty_valid_bonds_with_nonempty_global_edges_supported': True}


def diagnostic_check(model, batch):
    model.eval()
    model.enable_bond_fusion_diagnostics = False
    baseline, _ = trace(model, batch)
    assert model.last_bond_fusion_diagnostics == {}
    before = rng_state()
    model.enable_bond_fusion_diagnostics = True
    try:
        prediction, record = trace(model, batch)
        values = copy.deepcopy(model.last_bond_fusion_diagnostics)
    finally:
        model.enable_bond_fusion_diagnostics = False
    same_rng(before, rng_state())
    assert torch.equal(baseline, prediction), 'Diagnostic switch changed production predictions'
    parts, mode = record['head'][0], model.bond_fusion_mode
    assert values['mode'] == mode and values['num_valid_bonds'] == parts[0].shape[0]
    assert values['sources'] == ['left', 'right', 'edge', 'difference', 'product', 'theory']
    norms = [float(part.float().norm(dim=-1).mean()) for part in parts]
    assert all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
               for a, b in zip(values['source_l2_mean'], norms))
    with torch.no_grad():
        if mode == 'cross_attention':
            attention = attention_reference(model.bond_head, parts)[1]
            per_head = attention.mean(0)
            per_query = per_head.mean(0)
            source = per_query.mean(0)
            for key, expected in (('attention_mean_per_head_query_source', per_head),
                                  ('attention_mean_per_query_source', per_query),
                                  ('source_attention_mean', source)):
                assert torch.allclose(torch.tensor(values[key]), expected, atol=1e-6, rtol=1e-5)
            assert values['query_sources'] == ['left', 'right']
            assert values['memory_sources'] == ['edge', 'difference', 'product', 'theory']
            assert values['attention_heads'] == 4
        elif mode == 'gated':
            gate = model.bond_fusion_gate
            weights = F.linear(torch.cat(parts, dim=-1), gate.weight, gate.bias).softmax(-1)
            for key, expected in (('source_weight_mean', weights.mean(0)),
                                  ('source_weight_min', weights.amin(0)),
                                  ('source_weight_max', weights.amax(0)),
                                  ('scaled_gate_mean', 6 * weights.mean(0))):
                assert torch.allclose(torch.tensor(values[key]), expected, atol=1e-6, rtol=1e-5)
        elif mode == 'residual_mlp':
            head, residual = model.bond_head, model.bond_fusion_residual
            z = F.relu(F.linear(torch.cat(parts, dim=-1), head[0].weight, head[0].bias))
            normed = F.layer_norm(z, residual.normalization.normalized_shape,
                residual.normalization.weight, residual.normalization.bias, residual.normalization.eps)
            correction = F.linear(F.relu(F.linear(normed, residual.down.weight, residual.down.bias)),
                                  residual.up.weight, residual.up.bias)
            corrected = z + correction
            # The recorded delta is from actual post-addition values, including FP32 rounding.
            delta, input_norm = (corrected - z).norm(dim=-1), z.norm(dim=-1)
            expected = {'residual_delta_l2_mean': float(delta.mean()),
                'residual_input_l2_mean': float(input_norm.mean()),
                'residual_output_l2_mean': float(corrected.norm(dim=-1).mean()),
                'residual_relative_l2_mean': float((delta / input_norm.clamp_min(torch.finfo(z.dtype).eps)).mean())}
            assert all(math.isclose(values[key], value, rel_tol=1e-6, abs_tol=1e-6)
                       for key, value in expected.items())
    return {'opt_in_only': True, 'predictions_rng_unchanged': True,
            'source_norms_and_mode_specific_formula_verified': True}


def backbone_shape_hooks(model, observations):
    handles = []
    for name, layers in (('gcn', model.gcn_layers), ('gat', model.gat_layers)):
        for index, layer in enumerate(layers):
            def record(module, args, output, key=f'{name}_{index + 1}'):
                assert args[0].shape[-1] == output.shape[-1] == args[2].shape[-1] == 256
                assert args[0].shape[0] == output.shape[0]
                observations.setdefault(key, {'input': list(args[0].shape), 'output': list(output.shape)})
            handles.append(layer.register_forward_hook(record))
    return handles


def smoke_update(model, trainer, batch):
    model.train()
    classifier = output_layer(model.bond_head)
    before = classifier.weight.detach().clone()
    captured = []
    handle = model.register_forward_hook(lambda module, args, output: captured.append(output))
    seed()
    try:
        loss, stats = trainer._forward_pass(batch)
    finally:
        handle.remove()
    assert len(captured) == 1 and torch.isfinite(loss)
    states = model.last_four_state_logits
    mask = batch['label_mask'].bool()
    assert states.shape == (int(mask.sum()), 4)
    assert torch.equal(captured[0][mask].float(), union_logit(states)), 'Four-state union logit path changed'
    assert torch.allclose(captured[0][mask].float().sigmoid(), 1 - states.float().softmax(-1)[:, 0], atol=1e-6)
    expected = four_state_cross_entropy(states, batch['four_state_targets'], mask,
        trainer._masked_row_weights(batch), conditional_weight=1., union_brier_weight=0.)
    assert torch.equal(loss, expected), 'Actual production trainer does not use alpha1/lambda0 four-state CE'
    states.retain_grad()
    trainer._backward_pass(loss)
    assert not torch.equal(before, classifier.weight), 'AMP optimizer step was skipped'
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert states.grad is not None and torch.isfinite(states.grad).all()
    assert (states.grad.abs().sum(0) > 0).all()
    result = {'loss': float(loss.detach()), 'valid_bonds': stats['valid_bond_count'],
        'gradient_norm': trainer.last_grad_norm, 'amp': trainer.use_amp,
        'four_state_ce_and_union_q_verified': True}
    assert math.isfinite(result['gradient_norm'])
    trainer.optimizer.zero_grad(set_to_none=True)
    model.last_four_state_logits = None
    trainer.metrics_calculator.reset()
    return result


def checkpoint_checks(configs, states, folder):
    result = {}
    for mode in MODES:
        path = folder / f'{mode}.pt'
        torch.save({'model_state_dict': states[mode], 'config': configs[mode]}, path)
        model = GraphTransformer(build_model_config(configs[mode])).cpu()
        saved = torch.load(path, map_location='cpu', weights_only=True)
        validate_bond_fusion_checkpoint(model, saved['model_state_dict'], saved['config'])
        model.load_state_dict(saved['model_state_dict'], strict=True)
        same_tensors(states[mode], cpu_state(model))
        rejected = []
        for other in MODES:
            if other == mode:
                continue
            before = cpu_state(model)
            for strict in (True, False):
                try:
                    model.load_state_dict(states[other], strict=strict)
                except (RuntimeError, ValueError):
                    pass
                else:
                    raise AssertionError(f'Cross-fusion checkpoint accepted: {other}->{mode}')
                same_tensors(before, cpu_state(model))
            rejected.append(other)
        bad = copy.deepcopy(configs[mode])
        bad['model']['bond_fusion_mode'] = MODES[(MODES.index(mode) + 1) % len(MODES)]
        try:
            validate_bond_fusion_checkpoint(model, states[mode], bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Saved mode/config mismatch accepted')
        for key in ('four_state_mode_code', 'bond_fusion_source_roles', 'bond_fusion_heads_code'):
            if key not in states[mode]:
                continue
            tampered = {name: tensor.clone() for name, tensor in states[mode].items()}
            tampered[key] = tampered[key].flip(0) if tampered[key].ndim else tampered[key] + 1
            before = cpu_state(model)
            try:
                model.load_state_dict(tampered, strict=False)
            except (RuntimeError, ValueError):
                pass
            else:
                raise AssertionError(f'State/source/head-count semantic mismatch accepted: {key}')
            same_tensors(before, cpu_state(model))
        result[mode] = {'same_mode_strict_file_restore': True,
            'cross_modes_rejected_before_copy': rejected, 'saved_config_mismatch_rejected': True,
            'source_output_headcount_semantics_rejected_before_copy': True}
    return result


def run(gpu=0):
    assert sys.platform.startswith('linux') and str(ROOT).startswith('/mnt/pvc/'), 'Pod only'
    assert torch.cuda.is_available() and isinstance(gpu, int) and not isinstance(gpu, bool)
    torch.cuda.set_device(gpu)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    check_loss()
    args = argparse.Namespace(epochs=None, batch_size=None, learning_rate=None, device=None)
    configs = {mode: load_config(config_path(mode), args) for mode in MODES}
    original = load_config(ROOT / 'graph_transform/config/pre_synthesis_four_state_joint.yaml', args)
    canonical = normalized(original, 'four_state_joint')
    for mode, config in configs.items():
        assert normalized(config, f'four_state_head_fusion_{mode}') == canonical, f'Unexpected change: {mode}'
        mc = build_model_config(config)
        assert mc.bond_fusion_mode == mode and mc.bond_head_type == 'four_state_mlp'
        assert mc.hidden_dim == 256 and mc.num_attention_heads == 4 and mc.concat_heads
        assert mc.num_gcn_layers == 3 and mc.num_gat_layers == 2
        assert mc.theory_fusion_stage == 'late' and mc.edge_state_mode == 'static'
        assert mc.use_theory_features and mc.use_global_node and mc.use_condition_film
        assert not mc.use_charge_partition_features and mc.sequence_adapter_type == 'none'
        assert mc.state_feature_mask == [True, True, False] and mc.env_feature_mask == [True, False]
        assert config['loss']['four_state_conditional_weight'] == 1.
        assert config['loss']['four_state_union_brier_weight'] == 0.
        assert config['experiment']['seed'] == config['training']['validation_split_base_seed'] == 42
        assert config['training']['batch_size'] == 128
        assert not config['training'].get('pretrained_backbone_path')
        assert not config['training'].get('pretrained_backbone_sha256')
    prior, width = json.loads(PRIOR.read_text()), json.loads(WIDTH_PRIOR.read_text())
    assert prior['status'] == width['status'] == 'passed' and prior['folds'] == width['folds']
    report = {'status': 'running', 'configs': {mode: file_sha(config_path(mode)) for mode in MODES},
        'folds': {}, 'variants': {}, 'arms': {}, 'resources': {},
        'prior_validation_proof': str(PRIOR), 'prior_validation_proof_sha256': file_sha(PRIOR),
        'width_validation_proof': str(WIDTH_PRIOR), 'width_validation_proof_sha256': file_sha(WIDTH_PRIOR),
        'device': torch.cuda.get_device_name(gpu), 'gpu': gpu,
        'checks': [], 'scope': 'Pod-only bounded preflight, no epochs/outer-test metrics. '
            'All-empty-edge historical GAT is outside scope; empty valid-bond heads and nonempty-global-edge full batches are checked.'}
    with tempfile.TemporaryDirectory(prefix='four_state_head_fusion_') as directory:
        temp = Path(directory)
        first = None
        for fold in FOLDS:
            fold_configs = {}
            for mode, config in configs.items():
                resolved, _, _ = make_fold_config(config, fold, str(ROOT / 'dataset/5fold_folded'),
                    str(temp / 'paired'), _stable_fold_seed(42, fold))
                assert resolved['experiment']['seed'] == resolved['training']['validation_split_seed'] == _stable_fold_seed(42, fold)
                resolved['data']['cache_graphs'] = False
                resolved['data']['cache_dir'] = str(temp / fold / 'cache')
                fold_configs[mode] = resolved
            base = normalized(fold_configs['concat'], 'four_state_head_fusion_concat')
            assert all(normalized(c, f'four_state_head_fusion_{m}') == base for m, c in fold_configs.items())
            train, val, test = create_datasets(fold_configs['concat'])
            sets = [set(dataset.data['seq']) for dataset in (train, val, test)]
            assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            proof = {'train_rows': len(train), 'val_rows': len(val),
                'train_sequences': len(sets[0]), 'val_sequences': len(sets[1]),
                'train_sha256': file_sha(train.csv_path), 'val_sha256': file_sha(val.csv_path),
                'source_sha256': val.row_validation_provenance['source_sha256']}
            assert proof == prior['folds'][fold], f'Historical split changed: {fold}'
            assert val.row_validation_provenance['outer_test_used_for_internal_validation'] is False
            assert 'four_state_targets' not in val[0] and 'four_state_targets' not in test[0]
            proof.update(training_seed=fold_configs['concat']['experiment']['seed'],
                         split_seed=fold_configs['concat']['training']['validation_split_seed'])
            report['folds'][fold] = proof
            if first is None:
                first = (fold_configs, train, val)
        fold_configs, train, val = first
        assert len(train) >= 128 and len(val) >= 512
        indices = list(range(8)) + list(range(len(train) - 8, len(train)))
        small_cpu = collate(train, indices)
        report['global_topology'] = topology_proof(small_cpu)
        report['legacy_compatibility'] = legacy_compatibility(fold_configs['concat'], small_cpu)
        report['legacy_binary_head_compatibility'] = legacy_binary_heads_check(fold_configs['concat'], small_cpu)
        small = to_cuda(small_cpu)
        initial_bone = initial_rng = concat_initial_prediction = None
        saved_states = {}
        for mode in MODES:
            config = copy.deepcopy(fold_configs[mode])
            config['debug']['diagnostic_dir'] = str(temp / mode / 'diagnostics')
            seed(config['experiment']['seed'])
            model = GraphTransformer(build_model_config(config)).cuda()
            state, current_rng = cpu_state(model), rng_state()
            if initial_bone is None:
                initial_bone, initial_rng = backbone_state(state), current_rng
            else:
                same_tensors(initial_bone, backbone_state(state))
                same_rng(initial_rng, current_rng)
            with torch.random.fork_rng(devices=[]):
                causal = GraphTransformer(build_model_config(config)).cpu()
                causal.load_state_dict(state, strict=True)
                causal.eval()
                initial_prediction, _ = trace(causal, small_cpu)
                if mode == 'concat':
                    concat_initial_prediction = initial_prediction
                elif mode in ('gated', 'residual_mlp'):
                    assert torch.equal(concat_initial_prediction, initial_prediction), 'Zero gate/residual not function preserving'
                parts, routing = routing_check(causal, small_cpu)
                units = head_unit_check(causal, parts)
                diagnostics = diagnostic_check(causal, small_cpu)
                del causal
            observed = {}
            shape_handles = backbone_shape_hooks(model, observed)
            optimizer = create_optimizer(model, config)
            trainer = Trainer(model, optimizer, BinaryBondLoss(config['loss']),
                create_scheduler(optimizer, config), torch.device(f'cuda:{gpu}'), config, LOGGER)
            assert trainer.use_amp and trainer.four_state_conditional_weight == 1. and trainer.four_state_union_brier_weight == 0.
            gradients, gradient_handles = {}, []
            watched = {}
            if mode == 'gated':
                watched = {'gate': model.bond_fusion_gate.weight}
            elif mode == 'residual_mlp':
                watched = {'up': model.bond_fusion_residual.up.weight,
                           'down': model.bond_fusion_residual.down.weight}
            for key, parameter in watched.items():
                gradients[key] = []
                def capture(value, name=key):
                    assert torch.isfinite(value).all()
                    gradients[name].append(float(value.detach().float().abs().sum()))
                gradient_handles.append(parameter.register_hook(capture))
            torch.cuda.reset_peak_memory_stats()
            updates = [smoke_update(model, trainer, small), smoke_update(model, trainer, small)]
            small_memory = memory()
            if mode == 'gated':
                assert gradients['gate'][0] > 0 and gradients['gate'][1] > 0
            elif mode == 'residual_mlp':
                assert gradients['up'][0] > 0 and gradients['down'][0] == 0 and gradients['down'][1] > 0
            model.eval()
            with torch.no_grad():
                prediction = model(small)
            metrics = masked_metric_check(prediction, small)
            parameters = sum(p.numel() for p in model.parameters())
            entry = {'bond_fusion_mode': mode, 'bond_head_type': 'four_state_mlp',
                'hidden_dim': 256, 'theory_fusion_stage': 'late', 'edge_state_mode': 'static',
                'parameter_count': parameters, 'parameters': parameters,
                'fusion_parameter_count': sum(p.numel() for name, p in model.named_parameters()
                    if name.startswith(('bond_head.', 'bond_fusion_'))),
                'shared_backbone_initialization_exact': True, 'initialization_rng_exact': True,
                'initial_function_equal_to_concat': mode != 'cross_attention',
                'amp': trainer.use_amp, 'optimizer_updates': 2, 'small_batch_size': len(indices),
                'small_batch_updates': updates, 'new_parameter_gradient_l1': gradients,
                'routing': routing, 'unit_checks': units, 'diagnostics': diagnostics,
                'masked_metrics': metrics, 'shapes': observed,
                'peak_cuda_allocated_bytes': small_memory['allocated_bytes'],
                'peak_cuda_reserved_bytes': small_memory['reserved_bytes'],
                'training_small_batch_peak_allocated_mb': small_memory['allocated_bytes'] / 1024 ** 2,
                'training_small_batch_peak_reserved_mb': small_memory['reserved_bytes'] / 1024 ** 2}
            if mode != 'concat':
                full_train = to_cuda(collate(train, longest_indices(train, 128)))
                torch.cuda.reset_peak_memory_stats()
                entry['full_training_batch_update'] = smoke_update(model, trainer, full_train)
                resource = memory()
                entry.update(full_training_batch_size=128,
                    full_training_batch_peak_allocated_mb=resource['allocated_bytes'] / 1024 ** 2,
                    full_training_batch_peak_reserved_mb=resource['reserved_bytes'] / 1024 ** 2)
                entry['peak_cuda_allocated_bytes'] = max(entry['peak_cuda_allocated_bytes'], resource['allocated_bytes'])
                entry['peak_cuda_reserved_bytes'] = max(entry['peak_cuda_reserved_bytes'], resource['reserved_bytes'])
                entry['optimizer_updates'] += 1
                del full_train
                gc.collect()
                torch.cuda.empty_cache()
                full_val = to_cuda(collate(val, longest_indices(val, 512)))
                assert 'four_state_targets' not in full_val
                torch.cuda.reset_peak_memory_stats()
                model.eval()
                with torch.no_grad():
                    prediction = model(full_val)
                assert prediction.shape == full_val['labels'].shape and torch.isfinite(prediction).all()
                resource = memory()
                entry.update(validation_batch_size=512,
                    validation_batch_peak_allocated_mb=resource['allocated_bytes'] / 1024 ** 2,
                    validation_batch_peak_reserved_mb=resource['reserved_bytes'] / 1024 ** 2)
                entry['peak_cuda_allocated_bytes'] = max(entry['peak_cuda_allocated_bytes'], resource['allocated_bytes'])
                entry['peak_cuda_reserved_bytes'] = max(entry['peak_cuda_reserved_bytes'], resource['reserved_bytes'])
                del full_val
            saved_states[mode] = cpu_state(model)
            for handle in [*shape_handles, *gradient_handles]:
                handle.remove()
            report['variants'][mode] = report['arms'][mode] = entry
            report['resources'][mode] = {key: value for key, value in entry.items()
                if 'peak_' in key or key in ('parameter_count', 'fusion_parameter_count')}
            del model, trainer, optimizer, prediction, state
            gc.collect()
            torch.cuda.empty_cache()
        checkpoints = checkpoint_checks(fold_configs, saved_states, temp)
        for mode, proof in checkpoints.items():
            report['variants'][mode]['checkpoint_roundtrip'] = proof
    report['fusion_checks'] = {
        'frozen_concat_schema_state_rng_cpu_forward_backward_exact': report['legacy_compatibility']['train_forward_backward_exact'],
        'legacy_binary_concat_and_cross_checkpoint_compatible': True,
        'shared_backbone_initialization_and_rng': True, 'gated_residual_zero_initial_function_equal': True,
        'four_state_production_ce_and_union_q_actual': True,
        'all_six_source_order_and_gradients': True, 'independent_attention_gate_residual_formula': True,
        'local_key_sample_isolation': True, 'padding_poison_metrics_invariant': True,
        'opt_in_descriptive_diagnostics_formula_and_rng_verified': True,
        'mixed_shortchains_and_empty_valid_bonds': True,
        'same_mode_restore_cross_mode_precopy_reject': True,
        'two_real_amp_updates_and_zero_branch_gradient_staging': True,
        'all_three_candidates_longest_train128_validation512': True,
        'all_five_historical_train_val_source_shas_match': True}
    assert all(report['fusion_checks'].values())
    report['checks'] = report['fusion_checks']
    report['status'] = 'passed'
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    try:
        result = run(args.gpu)
    except Exception as error:
        failure = {'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                   'traceback': traceback.format_exc()}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(failure, indent=2) + '\n')
        raise
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'output': args.output}))
