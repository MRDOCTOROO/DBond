"""Pod-only bounded preflight for the fixed four-state node Conv1D experiment.

No epochs or outer-test metrics are evaluated. Entirely edge-empty legacy GAT
batches are outside scope; branch B0/L0 and mixed short graph batches are tested.
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'graph_transform'), str(ROOT / 'graph_transform/scripts')]
from data.row_validation import file_sha
from models.graph_transformer import GraphTransformer
from models.node_context_conv import NodeContextConv, validate_node_conv_checkpoint
from models.utils import build_model_config
from training.loss_functions import BinaryBondLoss
from training.trainer import Trainer
from test_four_state_weight import check_loss
from test_four_state_width import (collate, longest_indices, masked_metric_check,
                                  memory, to_cuda, topology_proof)
from test_four_state_dynamic_edges import seed, rng_state, same_rng, cpu_state, same_tensors, toy_batch
from test_four_state_head_fusion import smoke_update, backbone_shape_hooks, no_bond_batch
from train_graph_model import load_config, create_datasets, create_optimizer, create_scheduler
from train_5fold import make_fold_config, _stable_fold_seed
import train_graph_model as train_script

MODES = ('none', 'single_k5', 'multiscale_357')
FOLDS = ('1222', '2252', '3514', '6072', '9075')
LEGACY_REF = 'd4a2d1c'
LEGACY_FILE = 'graph_transform/models/graph_transformer.py'
PRIOR = Path('/mnt/pvc/graphtrans/four_state_ops/smoke.json')
WIDTH_PRIOR = Path('/mnt/pvc/graphtrans/four_state_width_ops/smoke.json')
LOGGER = logging.getLogger('four_state_node_conv_preflight')
train_script.logger = LOGGER


def config_path(mode):
    return ROOT / f'graph_transform/config/pre_synthesis_four_state_node_conv_{mode}.yaml'


def normalized(config, identity):
    value = copy.deepcopy(config)
    value['experiment'].pop('description', None)
    value['model'].pop('node_conv_mode', None)
    value['model'].setdefault('bond_fusion_mode', 'concat')
    value['model'].setdefault('bond_cross_attention_heads', 4)
    value['model'].setdefault('edge_state_mode', 'static')
    value['model'].setdefault('theory_fusion_stage', 'late')
    value['model'].setdefault('use_charge_partition_features', False)
    value['loss'].setdefault('four_state_conditional_weight', 1.)
    value['loss'].setdefault('four_state_union_brier_weight', 0.)
    value['training'].pop('validation_split_base_seed', None)
    value['training'].pop('validation_split_seed', None)
    return json.dumps(value, sort_keys=True).replace(identity, 'four_state_NODE_CONV')


def legacy_class():
    source = subprocess.check_output(['git', 'show', f'{LEGACY_REF}:{LEGACY_FILE}'], cwd=ROOT)
    revision = subprocess.check_output(['git', 'rev-parse', LEGACY_REF], cwd=ROOT).decode().strip()
    module = types.ModuleType('models._legacy_node_conv_reference')
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
    old_config = copy.deepcopy(config)
    old_config['model'].pop('node_conv_mode', None)
    validate_node_conv_checkpoint(current, cpu_state(old), old_config)
    results = []
    for model in (old, current):
        model.train()
        model.zero_grad(set_to_none=True)
        seed()
        output = model(batch)
        states = model.last_four_state_logits
        output[batch['label_mask'].bool()].square().mean().backward()
        gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()
                     if p.grad is not None}
        results.append((output.detach(), states.detach(), gradients, rng_state()))
    assert torch.equal(results[0][0], results[1][0]), 'Frozen none predictions changed'
    assert torch.equal(results[0][1], results[1][1]), 'Frozen none four-state logits changed'
    same_tensors(results[0][2], results[1][2])
    same_rng(results[0][3], results[1][3])
    proof.update(initial_state_and_schema_exact=True, initialization_rng_exact=True,
                 train_forward_backward_exact=True, forward_rng_exact=True,
                 old_checkpoint_strict_load=True)
    return proof


def shared_state(state):
    return {key: value for key, value in state.items()
            if not key.startswith(('node_context_conv.', 'node_conv_'))}


def valid_mask(features, lengths):
    return torch.arange(features.shape[1], device=features.device)[None, :] < lengths[:, None]


def position_reference(module, features, lengths):
    """Independent literal per-position affine kernels, without calling Conv1d."""
    mask = valid_mask(features, lengths)
    clean = features.masked_fill(~mask[..., None], 0.)
    activations = []
    for branch in module.branches:
        kernel = branch.kernel_size[0]
        radius = kernel // 2
        rows = []
        for position in range(features.shape[1]):
            patch = [clean[:, position + offset] if 0 <= position + offset < features.shape[1]
                     else clean.new_zeros((features.shape[0], features.shape[2]))
                     for offset in range(-radius, radius + 1)]
            patch = torch.stack(patch, dim=-1).flatten(start_dim=1)
            rows.append(F.silu(F.linear(patch, branch.weight.flatten(start_dim=1), branch.bias)))
        if rows:
            activation = torch.stack(rows, dim=1)
        else:
            activation = clean.new_empty((features.shape[0], 0, branch.out_channels))
        activations.append(activation.masked_fill(~mask[..., None], 0.))
    joined = torch.cat(activations, dim=-1)
    delta = F.linear(joined, module.projection.weight, module.projection.bias)
    delta = delta.masked_fill(~mask[..., None], 0.)
    return (clean + delta).masked_fill(~mask[..., None], 0.), activations, delta


def activate_nonzero_projection(module):
    """Only a disposable CPU test replica is changed."""
    with torch.no_grad():
        projection = module.projection
        projection.weight.copy_(torch.linspace(-.02, .03, projection.weight.numel()).reshape_as(projection.weight))
        projection.bias.copy_(torch.linspace(-.01, .01, projection.bias.numel()))


def receptive_field_check(module):
    fixture = copy.deepcopy(module).cpu().eval()
    with torch.no_grad():
        for branch in fixture.branches:
            branch.weight.fill_(.01)
            branch.bias.zero_()
        fixture.projection.weight.fill_(.02)
        fixture.projection.bias.zero_()
        baseline = torch.zeros(2, 15, 256)
        lengths = torch.tensor([15, 11])
        changed = baseline.clone()
        changed[0, 7, 0] = 1.
        captured = []
        handles = [branch.register_forward_hook(
            lambda layer, args, output: captured.append(output.detach().clone()))
            for branch in fixture.branches]
        before = fixture(baseline, lengths)
        original = captured[:]
        captured.clear()
        after = fixture(changed, lengths)
        for handle in handles:
            handle.remove()
        for index, branch in enumerate(fixture.branches):
            difference = captured[index] - original[index]
            actual = difference[0].abs().sum(0) > 0
            radius = branch.kernel_size[0] // 2
            expected = (torch.arange(15) - 7).abs() <= radius
            assert torch.equal(actual, expected), 'Kernel support differs from exact sequence receptive field'
            assert torch.equal(difference[1], torch.zeros_like(difference[1])), 'Conv mixes peptide samples'
        radius = max(branch.kernel_size[0] for branch in fixture.branches) // 2
        assert torch.equal((after[0] - before[0]).abs().sum(-1) > 0,
                           (torch.arange(15) - 7).abs() <= radius)
        assert torch.equal(before[1], after[1])
    return {'per_branch_exact_receptive_field': True, 'combined_exact_receptive_field': True,
            'different_samples_not_neighbors': True}


def branch_unit_check(module):
    module = copy.deepcopy(module).cpu().eval()
    expected_spec = ((5,), (192,)) if module.mode == 'single_k5' else ((3, 5, 7), (64, 64, 64))
    assert (tuple(branch.kernel_size[0] for branch in module.branches),
            tuple(branch.out_channels for branch in module.branches)) == expected_spec
    assert torch.count_nonzero(module.projection.weight) == torch.count_nonzero(module.projection.bias) == 0
    for branch in module.branches:
        assert branch.stride == branch.dilation == (1,) and branch.groups == 1
        assert branch.bias is not None and branch.padding == (branch.kernel_size[0] // 2,)
        assert branch.in_channels == 256 and torch.count_nonzero(branch.weight) > 0
    lengths = torch.tensor([11, 5, 2, 1, 0])
    features = torch.linspace(-.8, .7, 5 * 11 * 256).reshape(5, 11, 256)
    mask = valid_mask(features, lengths)
    with torch.no_grad():
        initial = module(features, lengths)
        assert torch.equal(initial, features.masked_fill(~mask[..., None], 0.))
        activate_nonzero_projection(module)
        observed = module(features, lengths)
        reference, _, _ = position_reference(module, features, lengths)
        assert torch.allclose(observed, reference, atol=5e-6, rtol=1e-5), 'Conv differs from literal position reference'
        poisoned = features.clone()
        poisoned[~mask] = float('nan')
        assert torch.equal(observed, module(poisoned, lengths)), 'NaN padding leaks into convolution'
        poisoned[~mask] = float('inf')
        assert torch.equal(observed, module(poisoned, lengths)), 'Inf padding leaks into convolution'
        extended = F.pad(features, (0, 0, 0, 7), value=float('nan'))
        longer = module(extended, lengths)
        assert torch.allclose(observed, longer[:, :11], atol=5e-6, rtol=1e-5)
        assert torch.equal(longer[:, 11:], torch.zeros_like(longer[:, 11:]))
        for row, length in enumerate(lengths.tolist()):
            single = module(features[row:row + 1, :length], lengths[row:row + 1])
            assert torch.allclose(observed[row:row + 1, :length], single, atol=5e-6, rtol=1e-5)
        changed = features.clone()
        changed[0, 3] += 3.
        assert torch.equal(observed[1:], module(changed, lengths)[1:])
        assert module(torch.empty(0, 7, 256), torch.empty(0, dtype=torch.long)).shape == (0, 7, 256)
        assert module(torch.empty(3, 0, 256), torch.zeros(3, dtype=torch.long)).shape == (3, 0, 256)
    support = receptive_field_check(module)
    return {'literal_position_formula_verified': True, 'zero_projection_identity': True,
            'poison_padding_and_extended_padding_invariant': True,
            'short_l0_l1_l2_and_b0_supported': True, 'single_vs_mixed_sample_isolation': True,
            **support}


def trace(model, batch, transform=None):
    record = {'encoder': [], 'branch_input': [], 'branch_output': [], 'packed': [], 'global_projection': []}
    def encoder_hook(module, args, output):
        result = transform(output) if transform is not None else output
        record['encoder'].append(result.detach().clone())
        return result if transform is not None else None
    handles = [model.node_encoder.register_forward_hook(encoder_hook),
        model.gcn_layers[0].register_forward_pre_hook(
            lambda module, args: record['packed'].append(args[0].detach().clone())),
        model.global_node_proj.register_forward_hook(
            lambda module, args, output: record['global_projection'].append(output.detach().clone()))]
    if model.node_conv_mode != 'none':
        handles += [model.node_context_conv.register_forward_pre_hook(
            lambda module, args: record['branch_input'].append((args[0].detach().clone(), args[1].detach().clone()))),
            model.node_context_conv.register_forward_hook(
            lambda module, args, output: record['branch_output'].append(output.detach().clone()))]
    try:
        with torch.no_grad():
            output = model(batch)
    finally:
        for handle in handles:
            handle.remove()
    return output.detach().clone(), record


def route_check(model, batch):
    model.eval()
    assert model.node_conv_mode != 'none'
    activate_nonzero_projection(model.node_context_conv)
    inputs = {key: value for key, value in batch.items()
              if key not in ('labels', 'label_mask', 'soft_labels', 'four_state_targets')}
    prediction, record = trace(model, inputs)
    assert all(len(values) == 1 for values in record.values())
    raw = record['encoder'][0]
    branch_input, lengths = record['branch_input'][0]
    assert raw.ndim == 3 and raw.shape[-1] == 256
    assert torch.equal(raw, branch_input) and torch.equal(lengths, batch['seq_lens'])
    expected, _, _ = position_reference(model.node_context_conv, raw, lengths)
    assert torch.allclose(record['branch_output'][0], expected, atol=5e-6, rtol=1e-5)
    globals_ = model.global_node_embedding + record['global_projection'][0]
    packed = torch.cat([torch.cat((expected[i, :length], globals_[i:i + 1]), dim=0)
                        for i, length in enumerate(lengths.tolist())], dim=0)
    with torch.no_grad():
        film_input = torch.stack([batch['charges'].float() * .1, batch['nces'].float() * .01], dim=1)
        gamma, beta = model.film_mlp(film_input).chunk(2, -1)
        sample = torch.repeat_interleave(torch.arange(len(lengths)), batch['node_lens'])
        packed = packed * (1 + gamma[sample]) + beta[sample]
    assert torch.allclose(record['packed'][0], packed, atol=6e-6, rtol=1e-5), 'Conv/global/FiLM order changed'
    mask = valid_mask(raw, lengths)
    def poison(value):
        return value.masked_fill(~mask[..., None], float('nan'))
    poisoned, _ = trace(model, inputs, poison)
    assert torch.equal(prediction, poisoned), 'Node padding enters valid graph/head predictions'
    extended = dict(inputs)
    extended['max_node_len'] = int(batch['max_node_len']) + 7
    prolonged, _ = trace(model, extended, lambda value: F.pad(value, (0, 0, 0, 7), value=float('nan')))
    assert torch.allclose(prediction, prolonged, atol=6e-6, rtol=1e-5), 'Extra padded length changes valid graph predictions'
    with torch.no_grad():
        original_global = model.global_node_embedding.detach().clone()
        model.global_node_embedding.add_(torch.linspace(-.2, .3, 256))
    try:
        _, altered = trace(model, inputs)
        assert torch.equal(record['branch_input'][0][0], altered['branch_input'][0][0])
        assert torch.equal(record['branch_output'][0], altered['branch_output'][0]), 'Global enters residue convolution'
    finally:
        with torch.no_grad():
            model.global_node_embedding.copy_(original_global)
    for fixture in (toy_batch(), no_bond_batch()):
        output, _ = trace(model, fixture)
        assert output.shape == fixture['label_mask'].shape and torch.isfinite(output).all()
    return {'actual_node_encoder_to_conv_to_global_pack_to_film_order': True,
            'global_never_enters_residue_branch': True, 'targets_not_used_as_inputs': True,
            'valid_prediction_padding_poison_and_extension_invariant': True,
            'mixed_short_full_model_and_empty_valid_bonds_finite': True}


def diagnostic_check(model, batch):
    model.eval()
    model.enable_node_conv_diagnostics = False
    prediction, _ = trace(model, batch)
    assert model.last_node_conv_diagnostics == {}
    before = rng_state()
    model.enable_node_conv_diagnostics = True
    try:
        observed, record = trace(model, batch)
        values = copy.deepcopy(model.last_node_conv_diagnostics)
    finally:
        model.enable_node_conv_diagnostics = False
    same_rng(before, rng_state())
    assert torch.equal(prediction, observed), 'Diagnostic switch changes forward function'
    assert values['mode'] == model.node_conv_mode
    assert values['num_valid_residues'] == int(batch['seq_lens'].sum())
    if model.node_conv_mode == 'none':
        assert values['kernel_sizes'] == values['branch_channels'] == values['branch_activation_l2_mean'] == []
        raw = record['encoder'][0]
        mask = valid_mask(raw, batch['seq_lens'])
        expected_norm = float(raw[mask].float().norm(dim=-1).mean())
        assert values['projection_residual_l2_mean'] == values['residual_relative_l2_mean'] == 0.
        assert math.isclose(values['input_l2_mean'], expected_norm, rel_tol=1e-6, abs_tol=1e-6)
        assert math.isclose(values['output_l2_mean'], expected_norm, rel_tol=1e-6, abs_tol=1e-6)
        return {'opt_in_only_and_rng_exact': True, 'none_has_no_branch_and_zero_residual_diagnostics': True}
    assert values['kernel_sizes'] == [branch.kernel_size[0] for branch in model.node_context_conv.branches]
    assert values['branch_channels'] == [branch.out_channels for branch in model.node_context_conv.branches]
    raw, lengths = record['branch_input'][0]
    expected, branches, projected = position_reference(model.node_context_conv, raw, lengths)
    mask = valid_mask(raw, lengths)
    clean = raw.masked_fill(~mask[..., None], 0.)
    delta = (expected - clean)[mask].float().norm(dim=-1)
    input_norm = clean[mask].float().norm(dim=-1)
    reference = {'branch_activation_l2_mean': [float(value[mask].float().norm(dim=-1).mean()) for value in branches],
        'projection_residual_l2_mean': float(projected[mask].float().norm(dim=-1).mean()),
        'input_l2_mean': float(input_norm.mean()),
        'output_l2_mean': float(expected[mask].float().norm(dim=-1).mean()),
        'residual_relative_l2_mean': float((delta / input_norm.clamp_min(torch.finfo(torch.float32).eps)).mean())}
    for key, value in reference.items():
        actual = torch.tensor(values[key])
        assert torch.allclose(actual, torch.tensor(value), atol=2e-6, rtol=1e-5), f'Diagnostic reference differs: {key}'
    json.dumps(values, allow_nan=False)
    return {'opt_in_only_and_rng_exact': True, 'bounded_internal_descriptive_formula_verified': True}


def checkpoint_checks(configs, states, folder):
    result = {}
    for mode in MODES:
        path = folder / f'{mode}.pt'
        torch.save({'model_state_dict': states[mode], 'config': configs[mode]}, path)
        model = GraphTransformer(build_model_config(configs[mode])).cpu()
        saved = torch.load(path, map_location='cpu', weights_only=True)
        validate_node_conv_checkpoint(model, saved['model_state_dict'], saved['config'])
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
                    raise AssertionError(f'Cross-mode checkpoint accepted: {other}->{mode}')
                same_tensors(before, cpu_state(model))
            rejected.append(other)
        bad = copy.deepcopy(configs[mode])
        bad['model']['node_conv_mode'] = MODES[(MODES.index(mode) + 1) % len(MODES)]
        try:
            validate_node_conv_checkpoint(model, states[mode], bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Saved node-conv mode/config mismatch accepted')
        marker_keys = [name for name in states[mode] if name.startswith('node_conv_')]
        for key in marker_keys:
            tampered = {name: tensor.clone() for name, tensor in states[mode].items()}
            tampered[key] = tampered[key] + 1
            before = cpu_state(model)
            try:
                model.load_state_dict(tampered, strict=False)
            except (RuntimeError, ValueError):
                pass
            else:
                raise AssertionError(f'Node-conv semantic marker mismatch accepted: {key}')
            same_tensors(before, cpu_state(model))
        result[mode] = {'same_mode_strict_file_restore': True,
            'cross_modes_rejected_before_copy': rejected, 'saved_config_mismatch_rejected': True,
            'all_marker_semantics_rejected_before_copy': True}
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
        assert normalized(config, f'four_state_node_conv_{mode}') == canonical, f'Unexpected config: {mode}'
        mc = build_model_config(config)
        assert mc.node_conv_mode == mode and mc.bond_fusion_mode == 'concat'
        assert mc.bond_head_type == 'four_state_mlp' and mc.hidden_dim == 256
        assert mc.num_attention_heads == 4 and mc.concat_heads
        assert mc.num_gcn_layers == 3 and mc.num_gat_layers == 2
        assert mc.theory_fusion_stage == 'late' and mc.edge_state_mode == 'static'
        assert mc.use_theory_features and mc.use_global_node and mc.use_condition_film
        assert mc.sequence_adapter_type == 'none' and not mc.use_charge_partition_features
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
        'scope': 'Pod-only bounded preflight; no epochs or outer-test metrics. '
                 'Branch B0/L0 are checked; all-empty-edge legacy GAT is outside scope.'}
    with tempfile.TemporaryDirectory(prefix='four_state_node_conv_') as directory:
        temp, first = Path(directory), None
        for fold in FOLDS:
            fold_configs = {}
            for mode, config in configs.items():
                resolved, _, _ = make_fold_config(config, fold, str(ROOT / 'dataset/5fold_folded'),
                    str(temp / 'paired'), _stable_fold_seed(42, fold))
                assert resolved['experiment']['seed'] == resolved['training']['validation_split_seed'] == _stable_fold_seed(42, fold)
                resolved['data']['cache_graphs'] = False
                resolved['data']['cache_dir'] = str(temp / fold / 'cache')
                fold_configs[mode] = resolved
            baseline = normalized(fold_configs['none'], 'four_state_node_conv_none')
            assert all(normalized(config, f'four_state_node_conv_{mode}') == baseline
                       for mode, config in fold_configs.items())
            train, val, test = create_datasets(fold_configs['none'])
            sets = [set(dataset.data['seq']) for dataset in (train, val, test)]
            assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            proof = {'train_rows': len(train), 'val_rows': len(val),
                'train_sequences': len(sets[0]), 'val_sequences': len(sets[1]),
                'train_sha256': file_sha(train.csv_path), 'val_sha256': file_sha(val.csv_path),
                'source_sha256': val.row_validation_provenance['source_sha256']}
            assert proof == prior['folds'][fold], f'Historical split changed: {fold}'
            assert val.row_validation_provenance['outer_test_used_for_internal_validation'] is False
            assert 'four_state_targets' not in val[0] and 'four_state_targets' not in test[0]
            proof.update(training_seed=fold_configs['none']['experiment']['seed'],
                         split_seed=fold_configs['none']['training']['validation_split_seed'])
            report['folds'][fold] = proof
            if first is None:
                first = (fold_configs, train, val)
        fold_configs, train, val = first
        assert len(train) >= 128 and len(val) >= 512
        indices = list(range(8)) + list(range(len(train) - 8, len(train)))
        small_cpu = collate(train, indices)
        report['global_topology'] = topology_proof(small_cpu)
        report['legacy_compatibility'] = legacy_compatibility(fold_configs['none'], small_cpu)
        small = to_cuda(small_cpu)
        initial_shared = initial_rng = initial_prediction = None
        saved_states, counts = {}, {}
        for mode in MODES:
            config = copy.deepcopy(fold_configs[mode])
            config['debug']['diagnostic_dir'] = str(temp / mode / 'diagnostics')
            seed(config['experiment']['seed'])
            model = GraphTransformer(build_model_config(config)).cuda()
            state, current_rng = cpu_state(model), rng_state()
            if initial_shared is None:
                initial_shared, initial_rng = shared_state(state), current_rng
            else:
                same_tensors(initial_shared, shared_state(state))
                same_rng(initial_rng, current_rng)
            with torch.random.fork_rng(devices=[]):
                causal = GraphTransformer(build_model_config(config)).cpu().eval()
                causal.load_state_dict(state, strict=True)
                with torch.no_grad():
                    prediction = causal(small_cpu)
                if mode == 'none':
                    initial_prediction = prediction
                    units = routing = {'not_applicable_to_none': True}
                    diagnostics = diagnostic_check(causal, small_cpu)
                else:
                    assert torch.equal(initial_prediction, prediction), 'Zero branch changes initial complete-model function'
                    units = branch_unit_check(causal.node_context_conv)
                    routing = route_check(causal, small_cpu)
                    diagnostics = diagnostic_check(causal, small_cpu)
                del causal
            observed = {}
            shape_handles = backbone_shape_hooks(model, observed)
            optimizer = create_optimizer(model, config)
            trainer = Trainer(model, optimizer, BinaryBondLoss(config['loss']),
                create_scheduler(optimizer, config), torch.device(f'cuda:{gpu}'), config, LOGGER)
            assert trainer.use_amp and trainer.four_state_conditional_weight == 1. and trainer.four_state_union_brier_weight == 0.
            gradients, gradient_handles = {}, []
            if mode != 'none':
                watched = {'projection': model.node_context_conv.projection.weight,
                           **{f'conv_{index}': branch.weight for index, branch in enumerate(model.node_context_conv.branches)}}
                for key, parameter in watched.items():
                    gradients[key] = []
                    def capture(value, name=key):
                        assert torch.isfinite(value).all()
                        gradients[name].append(float(value.detach().float().abs().sum()))
                    gradient_handles.append(parameter.register_hook(capture))
            torch.cuda.reset_peak_memory_stats()
            updates = [smoke_update(model, trainer, small), smoke_update(model, trainer, small)]
            small_memory = memory()
            if mode != 'none':
                assert gradients['projection'][0] > 0 and gradients['projection'][1] > 0
                assert all(values[0] == 0 and values[1] > 0 for key, values in gradients.items() if key.startswith('conv_'))
            model.eval()
            with torch.no_grad():
                prediction = model(small)
            metrics = masked_metric_check(prediction, small)
            counts[mode] = sum(p.numel() for p in model.parameters())
            branch_count = sum(p.numel() for p in model.node_context_conv.parameters()) if mode != 'none' else 0
            if mode != 'none':
                assert branch_count == 256 * 192 * 5 + 192 + 192 * 256 + 256
            entry = {'node_conv_mode': mode, 'bond_head_type': 'four_state_mlp', 'bond_fusion_mode': 'concat',
                'hidden_dim': 256, 'theory_fusion_stage': 'late', 'edge_state_mode': 'static',
                'parameter_count': counts[mode], 'parameters': counts[mode], 'node_conv_parameter_count': branch_count,
                'shared_backbone_initialization_exact': True, 'initialization_rng_exact': True,
                'initial_function_equal_to_none': True, 'amp': trainer.use_amp, 'optimizer_updates': 2,
                'small_batch_size': len(indices), 'small_batch_updates': updates,
                'new_parameter_gradient_l1': gradients, 'unit_checks': units, 'routing': routing,
                'diagnostics': diagnostics, 'masked_metrics': metrics, 'shapes': observed,
                'peak_cuda_allocated_bytes': small_memory['allocated_bytes'],
                'peak_cuda_reserved_bytes': small_memory['reserved_bytes'],
                'training_small_batch_peak_allocated_mb': small_memory['allocated_bytes'] / 1024 ** 2,
                'training_small_batch_peak_reserved_mb': small_memory['reserved_bytes'] / 1024 ** 2}
            if mode != 'none':
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
                if 'peak_' in key or key in ('parameter_count', 'node_conv_parameter_count')}
            del model, trainer, optimizer, prediction, state
            gc.collect()
            torch.cuda.empty_cache()
        assert counts['single_k5'] == counts['multiscale_357']
        assert counts['single_k5'] - counts['none'] == 256 * 192 * 5 + 192 + 192 * 256 + 256
        report['matched_candidate_capacity'] = {'parameter_count': counts['single_k5'], 'verified': True}
        checkpoints = checkpoint_checks(fold_configs, saved_states, temp)
        for mode, proof in checkpoints.items():
            report['variants'][mode]['checkpoint_roundtrip'] = proof
    report['conv_checks'] = {
        'frozen_none_schema_state_rng_cpu_forward_backward_and_restore_exact': True,
        'shared_backbone_initialization_rng_and_initial_functions_exact': True,
        'candidate_parameter_budgets_exactly_equal': True,
        'independent_position_convolution_and_exact_kernel_support': True,
        'single_vs_mixed_sample_isolation': True,
        'nan_inf_padding_and_extended_padding_valid_predictions_invariant': True,
        'global_excluded_and_conv_before_global_pack_and_film': True,
        'short_chains_mixed_l0_l1_l2_and_branch_b0_l0': True,
        'real_four_state_ce_union_q_and_padding_mask_metrics': True,
        'two_amp_steps_projection_then_all_convolutions_receive_gradient': True,
        'two_candidates_longest_train128_and_validation512': True,
        'all_five_historical_train_val_source_shas_match': True,
        'same_mode_restore_cross_mode_config_and_marker_precopy_reject': True,
        'opt_in_descriptive_diagnostics_formula_and_rng': True}
    assert all(report['conv_checks'].values())
    report['checks'] = report['conv_checks']
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
        result = {'status': 'failed', 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
        raise
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'output': args.output}))
