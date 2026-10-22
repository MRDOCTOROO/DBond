"""Bounded pod-only preflight for static/fixed/dynamic edge-state ablations.

No epoch, outer-test metric, or local numerical execution is performed here.
The ops runner selects an idle GPU and calls ``run(0)`` before training.
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
import random
import subprocess
import sys
import tempfile
import traceback
import types

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'graph_transform'), str(ROOT / 'graph_transform/scripts')]
from data.row_validation import file_sha
from models.graph_transformer import GraphTransformer
from models.dynamic_edges import SharedEdgeStateUpdater, validate_edge_state_checkpoint
from models.utils import build_model_config
from training.loss_functions import BinaryBondLoss
from training.trainer import Trainer
from test_four_state_weight import check_loss
from test_four_state_width import (collate, longest_indices, masked_metric_check,
                                   memory, shape_hooks, smoke_update, to_cuda,
                                   topology_proof)
from train_graph_model import load_config, create_datasets, create_optimizer, create_scheduler
from train_5fold import make_fold_config, _stable_fold_seed
import train_graph_model as train_script

MODES = ('static', 'fixed_context', 'dynamic_context')
FOLDS = ('1222', '2252', '3514', '6072', '9075')
LEGACY_REF = '967bd43c51513f8a0779e5fd1d20dbb1fec955f4'
LEGACY_FILE = 'graph_transform/models/graph_transformer.py'
PRIOR = Path('/mnt/pvc/graphtrans/four_state_ops/smoke.json')
WIDTH_PRIOR = Path('/mnt/pvc/graphtrans/four_state_width_ops/smoke.json')
MODE_KEY = 'edge_state_mode_code'
UPDATER_PREFIX = 'edge_state_updater.'
LOGGER = logging.getLogger('four_state_dynamic_edges_preflight')
train_script.logger = LOGGER


def seed(value=42):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def config_path(mode):
    return ROOT / f'graph_transform/config/pre_synthesis_four_state_dynamic_edge_{mode}.yaml'


def normalized(config, identity):
    config = copy.deepcopy(config)
    # Human-readable arm descriptions are identity metadata, not a model input.
    config['experiment'].pop('description', None)
    config['model'].pop('edge_state_mode', None)
    config['model']['theory_fusion_stage'] = 'late'
    config['model'].setdefault('use_charge_partition_features', False)
    config['loss'].setdefault('four_state_conditional_weight', 1.)
    config['loss'].setdefault('four_state_union_brier_weight', 0.)
    config['training'].pop('validation_split_base_seed', None)
    config['training'].pop('validation_split_seed', None)
    return json.dumps(config, sort_keys=True).replace(identity, 'four_state_EDGE')


def rng_state():
    return {'cpu': torch.get_rng_state().clone(),
            'cuda': [value.clone() for value in torch.cuda.get_rng_state_all()]}


def same_rng(left, right):
    assert torch.equal(left['cpu'], right['cpu']), 'CPU initialization RNG changed'
    assert len(left['cuda']) == len(right['cuda'])
    assert all(torch.equal(a, b) for a, b in zip(left['cuda'], right['cuda'])), 'CUDA RNG changed'


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def same_tensors(left, right):
    assert left.keys() == right.keys(), (left.keys() - right.keys(), right.keys() - left.keys())
    assert all(torch.equal(value, right[name]) for name, value in left.items()), 'Tensor mismatch'


def shared_state(state):
    return {name: value for name, value in state.items()
            if name != MODE_KEY and not name.startswith(UPDATER_PREFIX)}


def legacy_class():
    source = subprocess.check_output(['git', 'show', f'{LEGACY_REF}:{LEGACY_FILE}'], cwd=ROOT)
    module = types.ModuleType('models._legacy_dynamic_edge_reference')
    module.__package__ = 'models'
    module.__file__ = f'git:{LEGACY_REF}:{LEGACY_FILE}'
    sys.modules[module.__name__] = module
    exec(compile(source.decode('utf-8'), module.__file__, 'exec'), module.__dict__)
    return module.GraphTransformer, {'commit': LEGACY_REF, 'path': LEGACY_FILE,
                                   'sha256': hashlib.sha256(source).hexdigest()}


def legacy_compatibility(config, batch):
    Legacy, proof = legacy_class()
    seed()
    old = Legacy(build_model_config(config)).cpu()
    old_rng = rng_state()
    seed()
    current = GraphTransformer(build_model_config(config)).cpu()
    same_rng(old_rng, rng_state())
    same_tensors(cpu_state(old), cpu_state(current))
    assert MODE_KEY not in current.state_dict()
    assert not any(name.startswith(UPDATER_PREFIX) for name in current.state_dict())
    current.load_state_dict(cpu_state(old), strict=True)
    fixture = {key: value.cpu() if torch.is_tensor(value) else value for key, value in batch.items()}
    outputs = []
    for model in (old, current):
        model.train()
        model.zero_grad(set_to_none=True)
        seed()
        prediction = model(fixture)
        state_logits = model.last_four_state_logits
        prediction[fixture['label_mask'].bool()].square().mean().backward()
        gradients = {name: value.grad.detach().clone() for name, value in model.named_parameters()
                     if value.grad is not None}
        outputs.append((prediction.detach(), state_logits.detach(), gradients, rng_state()))
    assert torch.equal(outputs[0][0], outputs[1][0]), 'Static frozen forward changed'
    assert torch.equal(outputs[0][1], outputs[1][1]), 'Static state logits changed'
    same_tensors(outputs[0][2], outputs[1][2])
    same_rng(outputs[0][3], outputs[1][3])
    proof.update(initial_tensors_exact=True, initialization_rng_exact=True,
        train_forward_and_gradients_bitwise_exact=True, forward_rng_exact=True,
        old_checkpoint_without_marker_loads_strictly=True)
    del old, current, outputs
    return proof


def make_nonzero(updater):
    """Deterministic test fixture, never used by a production training model."""
    with torch.no_grad():
        updater.up.weight.copy_(torch.linspace(-.01, .01, updater.up.weight.numel(),
            device=updater.up.weight.device).reshape_as(updater.up.weight))
        updater.up.bias.zero_()


def updater_reference(updater, edge, source, target):
    """Independent literal formula using functional operators, not forward()."""
    joined = torch.cat([source, target, edge], dim=-1)
    normalized = F.layer_norm(joined, updater.input_norm.normalized_shape,
        updater.input_norm.weight, updater.input_norm.bias, updater.input_norm.eps)
    hidden = F.silu(F.linear(normalized, updater.down.weight, updater.down.bias))
    return edge + F.linear(hidden, updater.up.weight, updater.up.bias)


def toy_batch():
    """Hand-written packed L=2/1/0 graph with no dataset or graph-builder code."""
    lengths, sequences = [2, 1, 0], ['AA', 'A', '']
    edges = torch.tensor([[0, 1, 0, 2, 1, 2, 3, 4], [1, 0, 2, 0, 2, 1, 4, 3]])
    types_ = torch.tensor([0, 0, 4, 4, 4, 4, 4, 4])
    mapping = torch.tensor([0])
    count = len(lengths)
    max_bonds = max(max(length - 1, 0) for length in lengths)
    mask = torch.arange(max_bonds)[None, :] < (torch.tensor(lengths) - 1).clamp_min(0)[:, None]
    theory = torch.full((count, max_bonds, 15), float('nan'))
    theory[mask] = torch.arange(15, dtype=torch.float32) / 15.
    return {'sequences': sequences, 'seq_lens': torch.tensor(lengths),
        'node_lens': torch.tensor(lengths) + 1, 'max_node_len': max(lengths) + 1,
        'max_bonds': max_bonds, 'edge_index': edges, 'edge_types': types_,
        'edge_distances': torch.ones(edges.shape[1], dtype=torch.long),
        'edge_attr': torch.zeros(edges.shape[1], 8), 'bond_edge_map': mapping,
        'bond_theory': theory, 'charges': torch.full((count,), 2.),
        'nces': torch.full((count,), 30.), 'state_vars': torch.tensor([[2., 250., 0.]]).repeat(count, 1),
        'env_vars': torch.tensor([[30., 0.]]).repeat(count, 1),
        'label_mask': mask, 'labels': torch.zeros_like(mask, dtype=torch.float32)}


def toy_updater_check():
    with torch.random.fork_rng(devices=[]):
        updater = SharedEdgeStateUpdater(8).cpu()
    batch = toy_batch()
    edge_index = batch['edge_index']
    nodes = torch.arange(6 * 8, dtype=torch.float32).reshape(6, 8) / 11.
    edge = torch.arange(8 * 8, dtype=torch.float32).reshape(8, 8) / 13.
    assert torch.equal(updater(edge, nodes[edge_index[0]], nodes[edge_index[1]]), edge)
    make_nonzero(updater)
    observed = updater(edge, nodes[edge_index[0]], nodes[edge_index[1]])
    expected = torch.cat([updater_reference(updater, edge[i:i+1],
        nodes[edge_index[0, i:i+1]], nodes[edge_index[1, i:i+1]]) for i in range(edge.shape[0])])
    assert torch.allclose(observed, expected, atol=1e-6, rtol=1e-5)
    changed = nodes.clone()
    changed[2] += torch.linspace(-.4, .3, 8)  # first graph's global node only
    perturbed = updater(edge, changed[edge_index[0]], changed[edge_index[1]])
    incident = (edge_index == 2).any(0)
    assert torch.equal(observed[~incident], perturbed[~incident])
    assert not torch.equal(observed[incident], perturbed[incident]), 'Global edges did not update'
    owner = torch.tensor([0, 0, 0, 1, 1, 2])
    assert torch.equal(owner[edge_index[0]], owner[edge_index[1]])
    empty = edge[:0]
    assert updater(empty, nodes[:0], nodes[:0]).shape == (0, 8)
    assert torch.equal(updater(empty, nodes[:0], nodes[:0]), empty)
    return {'independent_functional_and_row_loop_reference': True,
            'zero_residual_exact_identity': True, 'global_endpoint_influences_only_incident_edges': True,
            'no_cross_graph_edges_or_context': True, 'length_zero_one_two_packed_fixture': True,
            'empty_edge_updater_only': True}


def trace(model, batch, perturb=False):
    record = {'encoded': [], 'updates': [], 'layers': [], 'head': []}
    handles = [model.edge_encoder.register_forward_hook(
        lambda module, args, output: record['encoded'].append(output.detach().clone()))]
    if model.edge_state_mode != 'static':
        def update_hook(module, args, output):
            record['updates'].append({'edge': args[0].detach().clone(),
                'source': args[1].detach().clone(), 'target': args[2].detach().clone(),
                'output': output.detach().clone()})
        handles.append(model.edge_state_updater.register_forward_hook(update_hook))
    for layer in [*model.gcn_layers, *model.gat_layers]:
        def layer_hook(module, args):
            record['layers'].append({'node': args[0].detach().clone(),
                'edge_index': args[1].detach().clone(), 'edge': args[2].detach().clone()})
        handles.append(layer.register_forward_pre_hook(layer_hook))
    handles.append(model.bond_head[0].register_forward_pre_hook(
        lambda module, args: record['head'].append(args[0].detach().clone())))
    if perturb:
        def perturb_hook(module, args, output):
            delta = torch.zeros_like(output)
            delta[0] = torch.linspace(-.3, .4, output.shape[-1], device=output.device)
            return output + delta
        handles.append(model.gcn_layers[0].register_forward_hook(perturb_hook))
    try:
        with torch.no_grad():
            prediction = model(batch)
    finally:
        for handle in handles:
            handle.remove()
    assert len(record['encoded']) == 1 and len(record['layers']) == 5
    assert all(torch.equal(entry['edge_index'], batch['edge_index']) for entry in record['layers'])
    assert len(record['updates']) == (0 if model.edge_state_mode == 'static' else 5)
    return prediction.detach().clone(), record


def route_check(model, batch):
    assert next(model.parameters()).device.type == 'cpu'
    model.eval()
    mode = model.edge_state_mode
    if mode != 'static':
        make_nonzero(model.edge_state_updater)
    features = {key: value for key, value in batch.items()
                if key not in ('labels', 'label_mask', 'soft_labels', 'four_state_targets')}
    baseline, observed = trace(model, features)
    first_nodes = observed['layers'][0]['node']
    edge_index = batch['edge_index']
    initial_edge = observed['encoded'][0]
    owner = torch.repeat_interleave(torch.arange(len(batch['sequences'])), batch['node_lens'])
    assert torch.equal(owner[edge_index[0]], owner[edge_index[1]])
    if mode != 'static':
        prior = initial_edge
        for index, update in enumerate(observed['updates']):
            context = first_nodes if mode == 'fixed_context' else observed['layers'][index]['node']
            assert torch.equal(update['edge'], prior)
            assert torch.equal(update['source'], context[edge_index[0]])
            assert torch.equal(update['target'], context[edge_index[1]])
            expected = updater_reference(model.edge_state_updater, prior,
                context[edge_index[0]], context[edge_index[1]])
            assert torch.allclose(update['output'], expected, atol=1e-6, rtol=1e-5)
            assert torch.equal(observed['layers'][index]['edge'], update['output'])
            prior = update['output']
        assert not torch.equal(initial_edge, prior), 'Nonzero fixture never changed edge state'
    else:
        assert all(torch.equal(entry['edge'], initial_edge) for entry in observed['layers'])
    assert len(observed['head']) == 1
    head_edge = observed['head'][0][:, 2 * model.hidden_dim:3 * model.hidden_dim]
    assert torch.equal(head_edge, initial_edge[batch['bond_edge_map']]), 'Head readout must retain e0'
    changed, perturbed = trace(model, features, perturb=True)
    if mode != 'static':
        assert torch.equal(observed['updates'][0]['output'], perturbed['updates'][0]['output'])
        responds = not torch.equal(observed['updates'][1]['output'], perturbed['updates'][1]['output'])
        assert responds == (mode == 'dynamic_context'), 'Second updater ignored or leaked layer context'
    poisoned = dict(features)
    poisoned['bond_theory'] = batch['bond_theory'].clone()
    poisoned['bond_theory'][~batch['label_mask'].bool()] = float('nan')
    invariant, _ = trace(model, poisoned)
    assert torch.equal(baseline, invariant), 'Invalid theoretical padding changed predictions'
    assert torch.isfinite(invariant).all()
    # The unmodified historical GAT does not accept an entirely edge-empty
    # batch. Exercise L=0/1/2 together in a nonempty packed graph; E=0 is
    # deliberately covered only by the new updater unit reference above.
    fixture = toy_batch()
    prediction, traces = trace(model, fixture)
    assert prediction.shape == fixture['label_mask'].shape and torch.isfinite(prediction).all()
    assert all(entry['edge'].shape[0] == fixture['edge_index'].shape[1] for entry in traces['layers'])
    return {'device': 'cpu', 'five_updates_and_fixed_topology': True,
            'context_matches_declared_mode': True, 'head_uses_original_e0_in_order': True,
            'second_update_responds_only_in_dynamic_mode': True,
            'no_target_fields_used': True, 'poisoned_padding_invariant': True,
            'global_edges_retained': True, 'no_cross_batch_edges': True,
            'mixed_l2_l1_l0_full_model': True, 'empty_edge_updater_only': True}


def diagnostic_check(model, batch):
    """Verify opt-in summaries against the actual gate projections independently."""
    assert next(model.parameters()).device.type == 'cpu'
    model.eval()
    model.enable_edge_state_diagnostics = False
    baseline, _ = trace(model, batch)
    assert model.last_edge_state_diagnostics == []
    before = rng_state()
    model.enable_edge_state_diagnostics = True
    try:
        prediction, recorded = trace(model, batch)
        diagnostics = copy.deepcopy(model.last_edge_state_diagnostics)
    finally:
        model.enable_edge_state_diagnostics = False
    same_rng(before, rng_state())
    assert torch.equal(baseline, prediction)
    assert len(diagnostics) == 5
    initial = recorded['encoded'][0].float()
    previous = initial
    for index, (layer, values, observed) in enumerate(zip(
            [*model.gcn_layers, *model.gat_layers], recorded['layers'], diagnostics)):
        kind, number = ('gcn', index) if index < 3 else ('gat', index - 3)
        assert observed['layer'] == f'{kind}_{number}'
        current = values['edge'].float()
        count = current.shape[0]
        assert observed['edge_count'] == count
        for key, tensor in (('step_delta_l2_mean', current - previous),
                            ('cumulative_delta_l2_mean', current - initial),
                            ('edge_state_l2_mean', current), ('initial_edge_l2_mean', initial)):
            reference = float(tensor.norm(dim=-1).mean()) if count else 0.
            assert math.isclose(observed[key], reference, rel_tol=1e-6, abs_tol=1e-6)
        projection = layer.gcn.edge_transform if kind == 'gcn' else layer.edge_message_gate
        gate = torch.sigmoid(F.linear(current, projection.weight, projection.bias))
        low = float((gate <= .01).float().mean()) if count else 0.
        high = float((gate >= .99).float().mean()) if count else 0.
        assert observed['gate_saturation_lower'] == .01 and observed['gate_saturation_upper'] == .99
        assert math.isclose(observed['gate_low_fraction'], low, abs_tol=1e-7)
        assert math.isclose(observed['gate_high_fraction'], high, abs_tol=1e-7)
        assert math.isclose(observed['gate_saturation_fraction'], low + high, abs_tol=1e-7)
        previous = current
    return {'opt_in_only': True, 'predictions_and_rng_unchanged': True,
            'five_layer_norms_match_reference': True, 'actual_gcn_gat_gate_saturation_matches': True}


def checkpoint_checks(configs, states, folder):
    proofs = {}
    for mode in MODES:
        path = folder / f'edge_state_{mode}.pt'
        torch.save({'model_state_dict': states[mode], 'config': configs[mode]}, path)
        restored = GraphTransformer(build_model_config(configs[mode])).cpu()
        saved = torch.load(path, map_location='cpu', weights_only=True)
        validate_edge_state_checkpoint(restored, saved['model_state_dict'], saved['config'])
        restored.load_state_dict(saved['model_state_dict'], strict=True)
        same_tensors(states[mode], cpu_state(restored))
        rejected = []
        for other in MODES:
            if other == mode:
                continue
            before = cpu_state(restored)
            for strict in (True, False):
                try:
                    restored.load_state_dict(states[other], strict=strict)
                except (RuntimeError, ValueError) as error:
                    assert 'edge' in str(error).lower()
                else:
                    raise AssertionError(f'Cross-mode checkpoint accepted: {other} -> {mode}')
                same_tensors(before, cpu_state(restored))
            rejected.append(other)
        bad_config = copy.deepcopy(configs[mode])
        bad_config['model']['edge_state_mode'] = MODES[(MODES.index(mode) + 1) % len(MODES)]
        try:
            validate_edge_state_checkpoint(restored, states[mode], bad_config)
        except ValueError:
            pass
        else:
            raise AssertionError('Checkpoint config/state edge routing mismatch accepted')
        proofs[mode] = {'same_mode_saved_roundtrip_exact': True,
            'cross_mode_rejected_before_copy': rejected, 'strict_false_cannot_bypass_mode': True,
            'saved_config_state_mismatch_rejected': True}
    return proofs


def run(gpu=0):
    assert sys.platform.startswith('linux') and str(ROOT).startswith('/mnt/pvc/'), 'Pod execution only'
    assert torch.cuda.is_available(), 'Available pod GPU required'
    assert isinstance(gpu, int) and not isinstance(gpu, bool) and gpu >= 0
    torch.cuda.set_device(gpu)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    check_loss()
    args = argparse.Namespace(epochs=None, batch_size=None, learning_rate=None, device=None)
    configs = {mode: load_config(config_path(mode), args) for mode in MODES}
    original = load_config(ROOT / 'graph_transform/config/pre_synthesis_four_state_joint.yaml', args)
    canonical = normalized(original, 'four_state_joint')
    for mode, config in configs.items():
        assert normalized(config, f'four_state_dynamic_edge_{mode}') == canonical, f'Unexpected config difference: {mode}'
        mc = build_model_config(config)
        assert mc.edge_state_mode == mode and mc.theory_fusion_stage == 'late'
        assert mc.hidden_dim == 256 and mc.num_attention_heads == 4 and mc.concat_heads
        assert mc.num_gcn_layers == 3 and mc.num_gat_layers == 2
        assert mc.bond_head_type == 'four_state_mlp' and mc.use_theory_features
        assert mc.use_global_node and mc.use_condition_film and not mc.use_charge_partition_features
        assert mc.state_feature_mask == [True, True, False] and mc.env_feature_mask == [True, False]
        assert config['loss']['four_state_conditional_weight'] == 1.
        assert config['loss']['four_state_union_brier_weight'] == 0.
        assert config['experiment']['seed'] == 42
        assert config['training']['validation_split_base_seed'] == 42
        assert config['training']['batch_size'] == 128
        assert not config['training'].get('pretrained_backbone_path')
        assert not config['training'].get('pretrained_backbone_sha256')
    prior, width_prior = json.loads(PRIOR.read_text()), json.loads(WIDTH_PRIOR.read_text())
    assert prior['status'] == width_prior['status'] == 'passed'
    assert prior['folds'] == width_prior['folds']
    report = {'status': 'running', 'configs': {m: file_sha(config_path(m)) for m in MODES},
        'folds': {}, 'arms': {}, 'variants': {}, 'resources': {},
        'prior_validation_proof': str(PRIOR), 'prior_validation_proof_sha256': file_sha(PRIOR),
        'width_validation_proof': str(WIDTH_PRIOR), 'width_validation_proof_sha256': file_sha(WIDTH_PRIOR),
        'device': torch.cuda.get_device_name(gpu), 'gpu': gpu,
        'toy_updater': toy_updater_check(),
        'checks': ['All configurations differ only by edge-state mode and output/cache identity',
            'Original H256, 3GCN+2GAT, four heads, late theory, four-state alpha1/lambda0 retained'],
        'scope': 'Bounded pod-only preflight, not epoch training and no outer-test metrics. '
                 'Full-model short-chain fixture is mixed L0/L1/L2 with nonempty packed edges; '
                 'E0 support is tested only for the new updater, not the unchanged historical GAT.'}
    with tempfile.TemporaryDirectory(prefix='four_state_dynamic_edges_') as folder:
        temp = Path(folder)
        first = None
        for fold in FOLDS:
            fold_configs = {}
            for mode, config in configs.items():
                resolved, _, _ = make_fold_config(config, fold, str(ROOT / 'dataset/5fold_folded'),
                    str(temp / 'paired'), _stable_fold_seed(42, fold))
                assert resolved['experiment']['seed'] == resolved['training']['validation_split_seed'] == _stable_fold_seed(42, fold)
                resolved['data']['cache_dir'] = str(temp / fold / 'cache')
                resolved['data']['cache_graphs'] = False
                fold_configs[mode] = resolved
            base = normalized(fold_configs['static'], 'four_state_dynamic_edge_static')
            assert all(normalized(c, f'four_state_dynamic_edge_{m}') == base for m, c in fold_configs.items())
            train, val, test = create_datasets(fold_configs['static'])
            seqs = [set(dataset.data['seq']) for dataset in (train, val, test)]
            assert not (seqs[0] & seqs[1] or seqs[0] & seqs[2] or seqs[1] & seqs[2])
            proof = {'train_rows': len(train), 'val_rows': len(val),
                'train_sequences': len(seqs[0]), 'val_sequences': len(seqs[1]),
                'train_sha256': file_sha(train.csv_path), 'val_sha256': file_sha(val.csv_path),
                'source_sha256': val.row_validation_provenance['source_sha256']}
            assert proof == prior['folds'][fold], f'Historical train/val split changed: {fold}'
            assert val.row_validation_provenance['outer_test_used_for_internal_validation'] is False
            assert 'four_state_targets' not in val[0] and 'four_state_targets' not in test[0]
            proof.update(training_seed=fold_configs['static']['experiment']['seed'],
                         split_seed=fold_configs['static']['training']['validation_split_seed'])
            report['folds'][fold] = proof
            if first is None:
                first = (fold_configs, train, val)
        fold_configs, train, val = first
        assert len(train) >= 128 and len(val) >= 512
        indices = list(range(8)) + list(range(len(train) - 8, len(train)))
        small_cpu = collate(train, indices)
        report['global_topology'] = topology_proof(small_cpu)
        report['legacy_compatibility'] = legacy_compatibility(fold_configs['static'], small_cpu)
        small = to_cuda(small_cpu)
        shared = initial_rng = None
        initialized_updater = None
        zero_prediction = None
        initial_first_update = None
        saved_states = {}
        for mode in MODES:
            config = copy.deepcopy(fold_configs[mode])
            config['debug']['diagnostic_dir'] = str(temp / mode / 'diagnostics')
            seed(config['experiment']['seed'])
            model = GraphTransformer(build_model_config(config)).cuda()
            current, current_rng = cpu_state(model), rng_state()
            if shared is None:
                shared, initial_rng = shared_state(current), current_rng
            else:
                same_tensors(shared, shared_state(current))
                same_rng(initial_rng, current_rng)
            updater_parameters = 0
            if mode != 'static':
                updater_parameters = sum(p.numel() for p in model.edge_state_updater.parameters())
                updater_state = {k: v for k, v in current.items() if k.startswith(UPDATER_PREFIX)}
                if initialized_updater is None:
                    initialized_updater = updater_state
                else:
                    same_tensors(initialized_updater, updater_state)
                assert torch.count_nonzero(model.edge_state_updater.up.weight) == 0
                assert torch.count_nonzero(model.edge_state_updater.up.bias) == 0
            with torch.random.fork_rng(devices=[]):
                causal = GraphTransformer(build_model_config(config)).cpu()
                causal.load_state_dict(current, strict=True)
                causal.eval()
                zero, zero_trace = trace(causal, small_cpu)
                if zero_prediction is None:
                    zero_prediction = zero
                else:
                    assert torch.equal(zero_prediction, zero), 'Zero residual changed initial predictions'
                if mode != 'static':
                    first_update = zero_trace['updates'][0]
                    if initial_first_update is None:
                        initial_first_update = first_update
                    else:
                        same_tensors(initial_first_update, first_update)
                routing = route_check(causal, small_cpu)
                diagnostics = diagnostic_check(causal, small_cpu)
                del causal
            observations = {}
            handles = shape_hooks(model, 256, observations)
            optimizer = create_optimizer(model, config)
            trainer = Trainer(model, optimizer, BinaryBondLoss(config['loss']),
                create_scheduler(optimizer, config), torch.device(f'cuda:{gpu}'), config, LOGGER)
            assert trainer.use_amp
            assert trainer.four_state_conditional_weight == 1. and trainer.four_state_union_brier_weight == 0.
            gradients = {'up': [], 'down': []}
            gradient_handles = []
            if mode != 'static':
                for name in gradients:
                    def record_gradient(value, key=name):
                        assert torch.isfinite(value).all()
                        gradients[key].append(float(value.detach().float().abs().sum()))
                    gradient_handles.append(getattr(model.edge_state_updater, name).weight.register_hook(record_gradient))
            torch.cuda.reset_peak_memory_stats()
            updates = [smoke_update(model, trainer, small), smoke_update(model, trainer, small)]
            train_memory = memory()
            if mode != 'static':
                assert len(gradients['up']) == len(gradients['down']) == 2
                assert gradients['up'][0] > 0 and gradients['down'][0] == 0
                assert gradients['down'][1] > 0, 'Updater preceding layer did not learn after zero-output layer moved'
            model.eval()
            with torch.no_grad():
                prediction = model(small)
            assert prediction.shape == small['labels'].shape and torch.isfinite(prediction).all()
            metrics = masked_metric_check(prediction, small)
            count = sum(p.numel() for p in model.parameters())
            entry = {'edge_state_mode': mode, 'theory_fusion_stage': 'late', 'hidden_dim': 256,
                'gat_heads': 4, 'head_dim': 64, 'parameter_count': count, 'parameters': count,
                'updater_parameter_count': updater_parameters,
                'trainable_parameter_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'shared_parameter_initialization_exact': True, 'initialization_rng_exact': True,
                'zero_residual_initial_prediction_exact': True, 'routing': routing, 'diagnostics': diagnostics,
                'amp': trainer.use_amp, 'optimizer_updates': 2, 'small_batch_size': len(indices),
                'small_batch_updates': updates, 'updater_gradient_l1_by_update': gradients,
                'peak_cuda_allocated_bytes': train_memory['allocated_bytes'],
                'peak_cuda_reserved_bytes': train_memory['reserved_bytes'],
                'training_small_batch_peak_allocated_mb': train_memory['allocated_bytes'] / 1024 ** 2,
                'training_small_batch_peak_reserved_mb': train_memory['reserved_bytes'] / 1024 ** 2,
                'shapes': observations, 'masked_metrics': metrics}
            if mode == 'dynamic_context':
                full_train = to_cuda(collate(train, longest_indices(train, 128)))
                torch.cuda.reset_peak_memory_stats()
                entry['full_training_batch_update'] = smoke_update(model, trainer, full_train)
                full_memory = memory()
                entry.update(full_training_batch_size=128,
                    full_training_batch_peak_allocated_mb=full_memory['allocated_bytes'] / 1024 ** 2,
                    full_training_batch_peak_reserved_mb=full_memory['reserved_bytes'] / 1024 ** 2)
                entry['peak_cuda_allocated_bytes'] = max(entry['peak_cuda_allocated_bytes'], full_memory['allocated_bytes'])
                entry['peak_cuda_reserved_bytes'] = max(entry['peak_cuda_reserved_bytes'], full_memory['reserved_bytes'])
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
                val_memory = memory()
                entry.update(validation_batch_size=512,
                    validation_batch_peak_allocated_mb=val_memory['allocated_bytes'] / 1024 ** 2,
                    validation_batch_peak_reserved_mb=val_memory['reserved_bytes'] / 1024 ** 2)
                entry['peak_cuda_allocated_bytes'] = max(entry['peak_cuda_allocated_bytes'], val_memory['allocated_bytes'])
                entry['peak_cuda_reserved_bytes'] = max(entry['peak_cuda_reserved_bytes'], val_memory['reserved_bytes'])
                del full_val
            saved_states[mode] = cpu_state(model)
            for handle in [*handles, *gradient_handles]:
                handle.remove()
            report['arms'][mode] = entry
            report['variants'][mode] = entry
            report['resources'][mode] = {key: value for key, value in entry.items()
                if 'peak_' in key or key in ('parameter_count', 'updater_parameter_count')}
            del model, trainer, optimizer, prediction, current
            gc.collect()
            torch.cuda.empty_cache()
        checkpoints = checkpoint_checks(fold_configs, saved_states, temp)
        for mode, proof in checkpoints.items():
            report['arms'][mode]['checkpoint_roundtrip'] = proof
    report['dynamic_checks'] = {
        'legacy_static_exact': report['legacy_compatibility']['train_forward_and_gradients_bitwise_exact'],
        'shared_initialization_and_rng_exact': True,
        'zero_update_initial_predictions_equal': True,
        'fixed_dynamic_initial_update_equal': True,
        'independent_updater_reference': report['toy_updater']['independent_functional_and_row_loop_reference'],
        'five_updates_same_topology_global_and_batch_boundaries': True,
        'current_layer_context_only_dynamic': True, 'head_keeps_e0': True,
        'opt_in_edge_norms_and_actual_gate_diagnostics_verified': True,
        'mixed_l0_l1_l2_full_model': True, 'empty_edge_updater_only': True,
        'padding_invariant': True,
        'same_mode_roundtrip_cross_mode_rejected_before_copy': True,
        'two_amp_updates_upstream_gradient_after_zero_last_layer': True,
        'historical_full_spectrum_validation_shas_match': True,
        'longest_real_train128_val512_finite': True}
    assert all(report['dynamic_checks'].values())
    report['routing_checks'] = report['dynamic_checks']
    report['checks'] += ['Frozen static source matches state schema, RNG, CPU training forward/backward exactly',
        'Shared parameters and RNG identical; fixed/dynamic updater parameters identical; zero update preserves prediction',
        'Independent functional updater reference checks edge/context order and global incident-edge isolation',
        'CPU hooks verify all five updates, fixed vs evolving node context, unmodified topology and original e0 head slot',
        'Second-layer perturbation affects only dynamic-context updater; mixed L0/L1/L2 full-model batch remains finite',
        'Empty-edge shapes/formula are verified only for the new updater; unchanged historical GAT E0 support is outside scope',
        'Each arm completes two production AMP updates; zero-last-layer then upstream gradients checked',
        'Historical folded training/full-spectrum internal validation CSV SHAs match all five splits',
        'Metrics ignore poisoned padding; strict checkpoint restore and cross-mode pre-copy rejection pass',
        'Dynamic arm longest-real-row train128/validation512 fit on selected GPU']
    report['status'] = 'passed'
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output', required=True)
    arguments = parser.parse_args()
    os.chdir(ROOT)
    try:
        result = run(arguments.gpu)
    except Exception as error:
        result = {'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                  'traceback': traceback.format_exc()}
        Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
        Path(arguments.output).write_text(json.dumps(result, indent=2) + '\n')
        raise
    Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
    Path(arguments.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'output': arguments.output}))
