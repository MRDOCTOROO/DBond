"""Pod-only, bounded preflight for equal-width theory-fusion routing ablations.

Install under graph_transform/scripts and run only after the preceding width
batch has a completed event. No epoch or outer-test evaluation is performed.
"""
import argparse
import copy
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
import types

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'graph_transform'), str(ROOT / 'graph_transform/scripts')]
from data.row_validation import file_sha
from models.graph_transformer import GraphTransformer
from models.theory_fusion import (THEORY_FUSION_MODE_KEY, add_packed_bond_theory,
                                  validate_theory_fusion_checkpoint)
from models.utils import build_model_config
from training.loss_functions import BinaryBondLoss
from training.trainer import Trainer
from test_four_state import seed
from test_four_state_weight import check_loss
from test_four_state_width import (collate, longest_indices, masked_metric_check,
                                   memory, shape_hooks, smoke_update, to_cuda,
                                   topology_proof)
from train_graph_model import load_config, create_datasets, create_optimizer, create_scheduler
from train_5fold import make_fold_config, _stable_fold_seed
import train_graph_model as train_script

STAGES = ('late', 'early_only', 'early_plus_late')
FOLDS = ('1222', '2252', '3514', '6072', '9075')
PRIOR = Path('/mnt/pvc/graphtrans/four_state_ops/smoke.json')
LEGACY_REF = '08a8cbc771090814d791b60336053bc94f65cc52'
LEGACY_FILE = 'graph_transform/models/graph_transformer.py'
LOGGER = logging.getLogger('four_state_fusion_stage_test')
train_script.logger = LOGGER


def config_path(stage):
    return ROOT / f'graph_transform/config/pre_synthesis_four_state_fusion_{stage}.yaml'


def normalized(config, identity):
    """Only route/identity paths and explicit established defaults may differ."""
    config = copy.deepcopy(config)
    config['model']['theory_fusion_stage'] = 'late'
    config['model'].setdefault('use_charge_partition_features', False)
    config['loss'].setdefault('four_state_conditional_weight', 1.)
    config['loss'].setdefault('four_state_union_brier_weight', 0.)
    return json.dumps(config, sort_keys=True).replace(identity, 'four_state_FUSION')


def rng_state():
    return {'cpu': torch.get_rng_state().clone(),
            'cuda': [state.clone() for state in torch.cuda.get_rng_state_all()]}


def same_rng(left, right):
    assert torch.equal(left['cpu'], right['cpu']), 'CPU RNG differs across routes'
    assert len(left['cuda']) == len(right['cuda'])
    assert all(torch.equal(a, b) for a, b in zip(left['cuda'], right['cuda'])), 'CUDA RNG differs across routes'


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def shared_state(state):
    return {key: value for key, value in state.items() if key != THEORY_FUSION_MODE_KEY}


def same_tensors(left, right):
    assert left.keys() == right.keys()
    assert all(torch.equal(value, right[key]) for key, value in left.items()), 'Shared initialized tensors differ'


def legacy_class():
    """Read an immutable existing Git object; never alter the checkout or ref."""
    source = subprocess.check_output(['git', 'show', f'{LEGACY_REF}:{LEGACY_FILE}'], cwd=ROOT)
    digest = hashlib.sha256(source).hexdigest()
    module = types.ModuleType('models._legacy_fusion_reference')
    module.__package__ = 'models'
    module.__file__ = f'git:{LEGACY_REF}:{LEGACY_FILE}'
    sys.modules[module.__name__] = module
    exec(compile(source.decode('utf-8'), module.__file__, 'exec'), module.__dict__)
    return module.GraphTransformer, {'commit': LEGACY_REF, 'path': LEGACY_FILE, 'sha256': digest}


def incident_reference(node_lens, seq_lens, vectors):
    """Independent Python-loop endpoint average, deliberately not index_add_."""
    vectors = vectors.detach().cpu()
    result = vectors.new_zeros((sum(node_lens), vectors.shape[-1]))
    offset = cursor = 0
    for nodes, length in zip(node_lens, seq_lens):
        for residue in range(length):
            incident = []
            if residue > 0:
                incident.append(vectors[cursor + residue - 1])
            if residue + 1 < length:
                incident.append(vectors[cursor + residue])
            if incident:
                result[offset + residue] = torch.stack(incident).mean(0)
        cursor += max(length - 1, 0)
        offset += nodes
    assert cursor == vectors.shape[0]
    return result


def toy_incidence_check():
    lengths = [4, 1, 0, 3]
    node_lens = [length + 1 for length in lengths]
    mask = torch.arange(3).unsqueeze(0) < (torch.tensor(lengths) - 1).clamp_min(0).unsqueeze(1)
    padded = torch.full((4, 3, 2), float('nan'))
    padded[mask] = torch.tensor([[1., 10.], [2., 20.], [3., 30.], [4., 40.], [5., 50.]])
    src, dst, global_indices = [], [], []
    offset = 0
    for length, nodes in zip(lengths, node_lens):
        src.extend(offset + i for i in range(max(length - 1, 0)))
        dst.extend(offset + i + 1 for i in range(max(length - 1, 0)))
        global_indices.append(offset + length)
        offset += nodes
    assert all(b == a + 1 for a, b in zip(src, dst))
    assert not (set(src) | set(dst)) & set(global_indices)
    expected = incident_reference(node_lens, lengths, padded[mask])
    for device in ('cpu', 'cuda:0'):
        vectors = padded[mask].to(device).detach().requires_grad_(True)
        nodes = torch.zeros(offset, 2, device=device, requires_grad=True)
        result = add_packed_bond_theory(nodes, vectors,
            torch.tensor(src, device=device), torch.tensor(dst, device=device))
        assert torch.equal(result.detach().cpu(), expected)
        assert torch.isfinite(result).all() and (result[global_indices] == 0).all()
        # Length one contributes a residue plus a global; length zero only a global.
        assert (result[5:8] == 0).all()
        result.square().sum().backward()
        assert vectors.grad is not None and torch.isfinite(vectors.grad).all()
        assert (vectors.grad.abs().sum(-1) > 0).all()
        empty = nodes.new_empty((0, 2))
        endpoints = torch.empty(0, dtype=torch.long, device=device)
        assert torch.equal(add_packed_bond_theory(nodes, empty, endpoints, endpoints), nodes)
    return {'independent_incident_mean': True, 'only_adjacent_bond_endpoints': True,
            'global_direct_addition_zero': True, 'poisoned_padding_excluded': True,
            'length_zero_one_and_empty_bonds': True, 'finite_endpoint_gradients': True}


def legacy_compatibility(config, batch):
    Legacy, proof = legacy_class()
    seed()
    legacy = Legacy(build_model_config(config)).cpu()
    before_rng = rng_state()
    seed()
    current = GraphTransformer(build_model_config(config)).cpu()
    after_rng = rng_state()
    same_rng(before_rng, after_rng)
    same_tensors(cpu_state(legacy), cpu_state(current))
    assert THEORY_FUSION_MODE_KEY not in current.state_dict()
    current.load_state_dict(cpu_state(legacy), strict=True)
    fixture = {key: value.cpu() if torch.is_tensor(value) else value for key, value in batch.items()}
    # CPU execution avoids nondeterministic CUDA scatter accumulation in this
    # bitwise regression of the unmodified historical forward/backward route.
    results = []
    for model in (legacy, current):
        model.train()
        model.zero_grad(set_to_none=True)
        seed()
        prediction = model(fixture)
        state_logits = model.last_four_state_logits
        loss = prediction[fixture['label_mask'].bool()].square().mean()
        loss.backward()
        gradients = {key: value.grad.detach().clone() for key, value in model.named_parameters()
                     if value.grad is not None}
        results.append((prediction.detach(), state_logits.detach(), gradients, rng_state()))
    assert torch.equal(results[0][0], results[1][0]), 'Late forward changed from frozen source'
    assert torch.equal(results[0][1], results[1][1]), 'Late four-state logits changed'
    same_tensors(results[0][2], results[1][2])
    same_rng(results[0][3], results[1][3])
    proof.update(initial_tensors_exact=True, initialization_rng_exact=True,
        late_train_forward_and_gradients_bitwise_exact=True, forward_rng_exact=True,
        old_checkpoint_without_marker_loads_strictly=True, fixture_rows=len(batch['sequences']))
    del legacy, current, results
    return proof


def route_trace(model, batch):
    traces = {'projected': [], 'gcn_input': [], 'gcn_output': [], 'head_input': []}
    handles = [
        model.bond_theory_proj.register_forward_hook(
            lambda module, args, output: traces['projected'].append(output.detach().clone())),
        model.gcn_layers[0].register_forward_pre_hook(
            lambda module, args: traces['gcn_input'].append(args[0].detach().clone())),
        model.gcn_layers[0].register_forward_hook(
            lambda module, args, output: traces['gcn_output'].append(output.detach().clone())),
        model.bond_head[0].register_forward_pre_hook(
            lambda module, args: traces['head_input'].append(args[0].detach().clone())),
    ]
    try:
        with torch.no_grad():
            prediction = model(batch)
    finally:
        for handle in handles:
            handle.remove()
    assert all(len(values) == 1 for values in traces.values()), 'A route duplicated its projector or graph pass'
    return prediction.detach().clone(), {key: values[0] for key, values in traces.items()}


def backbone_theory_gradient_check(model, batch, stage):
    """Differentiate a pre-GCN objective with respect to theory, not labels."""
    assert next(model.parameters()).device.type == 'cpu', 'Causal routing checks require CPU execution'
    model.eval()
    features = {key: value for key, value in batch.items()
                if key not in ('labels', 'label_mask', 'soft_labels', 'four_state_targets')}
    features['bond_theory'] = batch['bond_theory'].detach().clone().requires_grad_(True)
    captured = []
    handle = model.gcn_layers[0].register_forward_pre_hook(
        lambda module, args: captured.append(args[0]))
    try:
        prediction = model(features)
    finally:
        handle.remove()
    assert len(captured) == 1
    gradient, = torch.autograd.grad(captured[0].square().sum(), features['bond_theory'], allow_unused=True)
    if stage == 'late':
        assert gradient is None, 'Late theory unexpectedly influences the graph backbone'
        total = 0.
    else:
        mask = batch['label_mask'].bool()
        assert gradient is not None and torch.isfinite(gradient).all()
        assert (gradient[~mask] == 0).all()
        total = float(gradient[mask].abs().sum())
        assert total > 0, 'Early theory has no differentiable path into GCN1'
    del prediction, captured, gradient
    return {'pre_gcn_theory_gradient_absent': stage == 'late',
            'pre_gcn_theory_gradient_present': stage != 'late',
            'pre_gcn_theory_gradient_l1': total, 'padding_gradient_zero': True}


def attention_helpers_reject_early(model):
    """An old replay must reject before reading even an empty batch."""
    from utils.attention_extractor import AttentionExtractor
    extractor = AttentionExtractor(model, torch.device('cuda:0'))
    for call in (lambda: model.get_attention_weights({}),
                 lambda: extractor.extract_attention_weights({})):
        try:
            call()
        except NotImplementedError:
            pass
        else:
            raise AssertionError('Legacy attention replay accepted an early theory route')
    return True


def routing_check(model, batch, stage):
    """Changing theory must affect GCN1 only when theory enters before GCN1."""
    assert next(model.parameters()).device.type == 'cpu', 'Causal routing checks require CPU execution'
    model.eval()
    # Inference takes only input features. No target field can affect the trace.
    feature_batch = {key: value for key, value in batch.items()
                     if key not in ('labels', 'label_mask', 'soft_labels', 'four_state_targets')}
    baseline, before = route_trace(model, feature_batch)
    mask = batch['label_mask'].bool()
    perturbed = dict(feature_batch)
    perturbed['bond_theory'] = batch['bond_theory'].clone()
    delta = torch.linspace(.01, .15, 15, device=mask.device)
    perturbed['bond_theory'][mask] += delta
    changed, after = route_trace(model, perturbed)
    assert not torch.equal(before['projected'], after['projected'])
    assert not torch.allclose(baseline[mask], changed[mask], atol=1e-8, rtol=1e-7)
    if stage == 'late':
        assert torch.equal(before['gcn_input'], after['gcn_input'])
        assert torch.equal(before['gcn_output'], after['gcn_output'])
        addition_max = 0.
    else:
        actual = after['gcn_input'] - before['gcn_input']
        expected = incident_reference(batch['node_lens'].tolist(), batch['seq_lens'].tolist(),
                                      after['projected'] - before['projected']).to(actual.device)
        assert torch.allclose(actual, expected, atol=2e-6, rtol=1e-5)
        assert not torch.allclose(before['gcn_input'], after['gcn_input'], atol=1e-8, rtol=1e-7)
        assert not torch.allclose(before['gcn_output'], after['gcn_output'], atol=1e-8, rtol=1e-7)
        offsets = torch.cumsum(torch.cat((batch['node_lens'].new_zeros(1), batch['node_lens'][:-1])), 0)
        globals_ = offsets + batch['seq_lens']
        assert torch.equal(before['gcn_input'][globals_], after['gcn_input'][globals_])
        addition_max = float(actual.abs().max())
    for trace in (before, after):
        slot = trace['head_input'][:, -model.hidden_dim:]
        if stage == 'early_only':
            assert torch.count_nonzero(slot) == 0
        else:
            assert torch.equal(slot, trace['projected'])
    poisoned = dict(feature_batch)
    poisoned['bond_theory'] = batch['bond_theory'].clone()
    poisoned['bond_theory'][~mask] = float('nan')
    invariant, poison_trace = route_trace(model, poisoned)
    assert torch.allclose(baseline, invariant, atol=1e-6, rtol=1e-5)
    assert torch.equal(before['gcn_input'], poison_trace['gcn_input'])
    assert torch.isfinite(invariant).all()
    return {'device': 'cpu', 'projector_calls_per_forward': 1, 'theory_changes_prediction': True,
            'theory_changes_first_gcn_input': stage != 'late',
            'first_gcn_delta_matches_independent_incident_reference': stage != 'late',
            'global_node_has_no_direct_theory_addition': True,
            'head_theory_slot_zero': stage == 'early_only',
            'poisoned_padding_invariant': True, 'no_target_fields_used': True,
            'max_first_gcn_input_change': addition_max}


def checkpoint_checks(configs, states, folder):
    result = {}
    paths = {}
    for stage in STAGES:
        paths[stage] = folder / f'{stage}.pt'
        torch.save({'model_state_dict': states[stage], 'config': configs[stage]}, paths[stage])
    for stage in STAGES:
        model = GraphTransformer(build_model_config(configs[stage]))
        saved = torch.load(paths[stage], map_location='cpu', weights_only=True)
        validate_theory_fusion_checkpoint(model, saved['model_state_dict'], saved['config'])
        model.load_state_dict(saved['model_state_dict'], strict=True)
        same_tensors(states[stage], cpu_state(model))
        rejected = []
        for other in STAGES:
            if other == stage:
                continue
            before = cpu_state(model)
            for strict in (True, False):
                try:
                    model.load_state_dict(states[other], strict=strict)
                except RuntimeError as error:
                    assert 'theory_fusion' in str(error)
                else:
                    raise AssertionError(f'Cross-stage checkpoint accepted: {other} -> {stage}')
                same_tensors(before, cpu_state(model))
            rejected.append(other)
        bad_config = copy.deepcopy(configs[stage])
        bad_config['model']['theory_fusion_stage'] = STAGES[(STAGES.index(stage) + 1) % len(STAGES)]
        try:
            validate_theory_fusion_checkpoint(model, states[stage], bad_config)
        except ValueError:
            pass
        else:
            raise AssertionError('Checkpoint config/state routing mismatch accepted')
        result[stage] = {'same_stage_saved_file_roundtrip_exact': True,
                        'cross_stage_rejected_before_copy': rejected,
                        'strict_false_cannot_bypass_stage': True,
                        'saved_config_state_mismatch_rejected': True}
    return result


def run(output):
    assert sys.platform.startswith('linux') and str(ROOT).startswith('/mnt/pvc/'), 'Pod execution only'
    assert torch.cuda.is_available(), 'An available pod GPU is required'
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    check_loss()
    toy = toy_incidence_check()
    args = argparse.Namespace(epochs=None, batch_size=None, learning_rate=None, device=None)
    configs = {stage: load_config(config_path(stage), args) for stage in STAGES}
    original = load_config(ROOT / 'graph_transform/config/pre_synthesis_four_state_joint.yaml', args)
    canonical = normalized(original, 'four_state_joint')
    for stage, config in configs.items():
        assert normalized(config, f'four_state_fusion_{stage}') == canonical, f'Unexpected config change: {stage}'
        model_config = build_model_config(config)
        assert model_config.theory_fusion_stage == stage
        assert model_config.hidden_dim == 256 and model_config.num_attention_heads == 4 and model_config.concat_heads
        assert model_config.bond_head_type == 'four_state_mlp' and model_config.use_theory_features
        assert model_config.use_global_node and model_config.use_condition_film
        assert model_config.use_charge_partition_features is False
        assert model_config.state_feature_mask == [True, True, False]
        assert model_config.env_feature_mask == [True, False]
        assert config['loss']['four_state_conditional_weight'] == 1.
        assert config['loss']['four_state_union_brier_weight'] == 0.
        assert config['training']['batch_size'] == 128
        assert not config['training'].get('pretrained_backbone_path')
        assert not config['training'].get('pretrained_backbone_sha256')
    prior = json.loads(PRIOR.read_text())
    assert prior['status'] == 'passed'
    report = {'status': 'running', 'folds': {}, 'variants': {}, 'arms': {},
              'configs': {stage: file_sha(config_path(stage)) for stage in STAGES},
              'prior_validation_proof': str(PRIOR), 'prior_validation_proof_sha256': file_sha(PRIOR),
              'device': torch.cuda.get_device_name(), 'incident_mean_fixture': toy,
              'checks': ['Only theory routing and output/cache identity paths change from the four-state joint model',
                         'All routes retain D=256, four heads, six-D readout, alpha=1, lambda=0, no charge residual',
                         'Independent toy incidence reference includes L=0/1, empty bonds, global nodes and NaN padding'],
              'scope': 'Bounded pod-only preflight; no epochs and no outer-test metrics'}
    with tempfile.TemporaryDirectory(prefix='four_state_fusion_') as folder:
        temp = Path(folder)
        first = None
        for fold in FOLDS:
            fold_configs = {}
            for stage, config in configs.items():
                fold_config, _, _ = make_fold_config(config, fold, str(ROOT / 'dataset/5fold_folded'),
                    str(temp / 'paired'), _stable_fold_seed(42, fold))
                fold_config['data']['cache_dir'] = str(temp / fold / 'cache')
                fold_config['data']['cache_graphs'] = False
                fold_configs[stage] = fold_config
            base = normalized(fold_configs['late'], 'four_state_fusion_late')
            assert all(normalized(c, f'four_state_fusion_{stage}') == base for stage, c in fold_configs.items())
            train, val, test = create_datasets(fold_configs['late'])
            seqs = [set(ds.data['seq']) for ds in (train, val, test)]
            assert not (seqs[0] & seqs[1] or seqs[0] & seqs[2] or seqs[1] & seqs[2])
            proof = {'train_rows': len(train), 'val_rows': len(val), 'train_sequences': len(seqs[0]),
                     'val_sequences': len(seqs[1]), 'train_sha256': file_sha(train.csv_path),
                     'val_sha256': file_sha(val.csv_path),
                     'source_sha256': val.row_validation_provenance['source_sha256']}
            assert proof == prior['folds'][fold], f'Established split changed: {fold}'
            assert 'four_state_targets' not in val[0] and 'four_state_targets' not in test[0]
            assert val.row_validation_provenance['outer_test_used_for_internal_validation'] is False
            report['folds'][fold] = proof
            if first is None:
                first = (fold_configs, train, val)
        fold_configs, train, val = first
        assert len(train) >= 128 and len(val) >= 512
        small_indices = list(range(8)) + list(range(len(train) - 8, len(train)))
        small_cpu = collate(train, small_indices)
        report['global_topology'] = topology_proof(small_cpu)
        report['legacy_compatibility'] = legacy_compatibility(fold_configs['late'], small_cpu)
        small = to_cuda(small_cpu)
        initial = initial_rng = None
        initialized_params = None
        states = {}
        for stage in STAGES:
            config = copy.deepcopy(fold_configs[stage])
            config['debug']['diagnostic_dir'] = str(temp / stage / 'diagnostics')
            seed()
            model = GraphTransformer(build_model_config(config)).cuda()
            current_rng = rng_state()
            current = cpu_state(model)
            parameters = sum(p.numel() for p in model.parameters())
            if initial is None:
                initial, initial_rng, initialized_params = shared_state(current), current_rng, parameters
            else:
                same_tensors(initial, shared_state(current))
                same_rng(initial_rng, current_rng)
                assert initialized_params == parameters
            assert model.node_encoder.state_feature_mask.tolist() == [1., 1., 0.]
            assert model.node_encoder.env_feature_mask.tolist() == [1., 0.]
            observations = {}
            handles = shape_hooks(model, 256, observations)
            # Repeated CUDA scatter reductions need not be bitwise identical.
            # Preserve exact causal-routing assertions on an independent CPU
            # replica rather than changing production determinism or tolerances.
            # Construction and checks cannot consume the paired GPU model's CPU
            # RNG stream, and the CPU replica receives the exact same tensors.
            with torch.random.fork_rng(devices=[]):
                routing_model = GraphTransformer(build_model_config(config)).cpu()
                routing_model.load_state_dict(current, strict=True)
                routing = routing_check(routing_model, small_cpu, stage)
                routing.update(backbone_theory_gradient_check(routing_model, small_cpu, stage))
                del routing_model
            if stage != 'late':
                routing['legacy_attention_helpers_reject_before_forward'] = attention_helpers_reject_early(model)
            optimizer = create_optimizer(model, config)
            trainer = Trainer(model, optimizer, BinaryBondLoss(config['loss']),
                create_scheduler(optimizer, config), torch.device('cuda:0'), config, LOGGER)
            assert trainer.four_state_conditional_weight == 1. and trainer.four_state_union_brier_weight == 0.
            projector_gradients = []
            def record_gradient(gradient):
                assert torch.isfinite(gradient).all()
                projector_gradients.append(float(gradient.detach().abs().sum()))
            gradient_handle = model.bond_theory_proj.weight.register_hook(record_gradient)
            torch.cuda.reset_peak_memory_stats()
            update = smoke_update(model, trainer, small)
            train_memory = memory()
            assert projector_gradients and projector_gradients[-1] > 0, 'Theory projector receives no gradient'
            with torch.no_grad():
                model.eval()
                prediction = model(small)
            assert prediction.shape == small['labels'].shape and torch.isfinite(prediction).all()
            metric_proof = masked_metric_check(prediction, small)
            entry = {'theory_fusion_stage': stage, 'hidden_dim': 256, 'gat_heads': 4, 'head_dim': 64,
                     'parameter_count': parameters, 'parameters': parameters,
                     'trainable_parameter_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
                     'shared_parameter_initialization_exact': True, 'cpu_cuda_initialization_rng_exact': True,
                     'amp': trainer.use_amp, 'optimizer_updates': 1, 'valid_bonds': update['valid_bonds'],
                     'small_batch_size': len(small_indices), 'small_batch_update': update,
                     'projector_gradient_l1': projector_gradients[-1], 'routing': routing,
                     'peak_cuda_allocated_bytes': train_memory['allocated_bytes'],
                     'peak_cuda_reserved_bytes': train_memory['reserved_bytes'],
                     'training_small_batch_peak_allocated_mb': train_memory['allocated_bytes'] / 1024 ** 2,
                     'training_small_batch_peak_reserved_mb': train_memory['reserved_bytes'] / 1024 ** 2,
                     'shapes': observations, 'masked_metrics': metric_proof}
            if stage == 'early_plus_late':
                full_train = to_cuda(collate(train, longest_indices(train, 128)))
                torch.cuda.reset_peak_memory_stats()
                entry['full_training_batch_update'] = smoke_update(model, trainer, full_train)
                assert projector_gradients[-1] > 0
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
            states[stage] = cpu_state(model)
            gradient_handle.remove()
            for handle in handles:
                handle.remove()
            report['variants'][stage] = entry
            report['arms'][stage] = entry
            del model, trainer, optimizer, prediction, current
            gc.collect()
            torch.cuda.empty_cache()
        checkpoints = checkpoint_checks(fold_configs, states, temp)
        for stage, proof in checkpoints.items():
            report['variants'][stage]['checkpoint_roundtrip'] = proof
    report['routing_checks'] = {
        'causal_routing_checks_on_cpu': all(report['variants'][stage]['routing']['device'] == 'cpu' for stage in STAGES),
        'late_backbone_theory_gradient_absent': report['variants']['late']['routing']['pre_gcn_theory_gradient_absent'],
        'early_backbone_theory_gradient_present': all(report['variants'][stage]['routing']['pre_gcn_theory_gradient_present']
                                                     for stage in STAGES if stage != 'late'),
        'padding_excluded': all(report['variants'][stage]['routing']['poisoned_padding_invariant'] for stage in STAGES),
        'global_direct_injection_zero': all(report['variants'][stage]['routing']['global_node_has_no_direct_theory_addition'] for stage in STAGES),
        'initialization_equal': all(report['variants'][stage]['shared_parameter_initialization_exact'] and
                                    report['variants'][stage]['cpu_cuda_initialization_rng_exact'] for stage in STAGES),
        'legacy_late_exact': report['legacy_compatibility']['late_train_forward_and_gradients_bitwise_exact'],
        'checkpoint_cross_mode_rejected_before_copy': all(len(proof['cross_stage_rejected_before_copy']) == 2
                                                         for proof in checkpoints.values()),
    }
    assert all(report['routing_checks'].values())
    report['checks'] += ['Frozen immutable Git reference matches late initialization, CPU forward, backward and RNG exactly',
                         'All routes have identical learned parameter count, shared initialized tensors and CPU/CUDA RNG',
                         'Independent CPU replicas with shared initialized tensors isolate causal routing from CUDA scatter nondeterminism',
                         'Real input-only theory perturbations reach GCN1 exactly on early routes; global direct injection is zero',
                         'Projection executes once per route; early-only final theory slot is zero; early-plus-late reuses projection',
                         'Every route completes finite production AMP loss/backward/update with nonzero theory-projector gradient',
                         'Five sequence-disjoint split/validation hashes match the established original-spectrum proof',
                         'Metrics are invariant to poisoned padding; accuracy/F1/q-Brier independently checked',
                         'Saved checkpoints roundtrip and cross-stage loads fail before copying even with strict=False',
                         'Early-plus-late passes longest-real-row training batch 128 and FP32 validation batch 512']
    report['status'] = 'passed'
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    try:
        run(args.output)
    except Exception as error:
        failure = {'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                   'traceback': traceback.format_exc()}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(failure, indent=2) + '\n')
        raise
