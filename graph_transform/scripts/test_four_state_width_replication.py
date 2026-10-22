"""Pod-only preflight for a fixed-four-head four-state hidden-width sweep.

Run this after installing the file under graph_transform/scripts on the pod.
It performs bounded smoke updates, never an epoch or an outer-test evaluation.
"""
import argparse
import copy
import gc
import json
import logging
import math
import os
import random
from pathlib import Path
import sys
import tempfile
import traceback

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'graph_transform'), str(ROOT / 'graph_transform/scripts')]
from data.graph_dataset import GraphDataLoader
from data.row_validation import file_sha
from models.four_state import union_logit
from models.graph_transformer import GraphTransformer
from models.utils import build_model_config
from training.four_state_loss import four_state_cross_entropy
from training.loss_functions import BinaryBondLoss
from training.metrics import BinaryBondMetrics, batch_group_keys
from training.trainer import Trainer
from test_four_state_weight import check_loss
from train_graph_model import load_config, create_datasets, create_optimizer, create_scheduler, _validation_split_seed
from train_5fold import make_fold_config, _stable_fold_seed
import train_graph_model as train_script

WIDTHS = (128, 256)
ORIGINAL_BASE_SEED = 42
NEW_BASE_SEED = 43
FOLDS = ('1222', '2252', '3514', '6072', '9075')
PRIOR = Path('/mnt/pvc/graphtrans/four_state_ops/smoke.json')
DISCOVERY = Path('/mnt/pvc/graphtrans/four_state_width_ops/smoke.json')
LOGGER = logging.getLogger('four_state_width_test')
train_script.logger = LOGGER


def config_path(width):
    return ROOT / f'graph_transform/config/pre_synthesis_four_state_width_replication_{width}.yaml'


def normalized(config, identity):
    """Normalize declared width, RNG split separation, identity, and old defaults."""
    config = copy.deepcopy(config)
    config['experiment']['seed'] = ORIGINAL_BASE_SEED
    config['training'].pop('validation_split_base_seed', None)
    config['training'].pop('validation_split_seed', None)
    config['model']['hidden_dim'] = 256
    config['model'].setdefault('use_charge_partition_features', False)
    config['loss'].setdefault('four_state_conditional_weight', 1.0)
    config['loss'].setdefault('four_state_union_brier_weight', 0.0)
    return json.dumps(config, sort_keys=True).replace(identity, 'four_state_WIDTH')


def seed(value=NEW_BASE_SEED):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def initialization_seed_check(config):
    """Same-width starts reproduce exactly, and the new RNG changes weights."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_stable_fold_seed(ORIGINAL_BASE_SEED, FOLDS[0]))
        original = GraphTransformer(build_model_config(config))
        old = original.node_encoder.node_encoder[0].weight.detach().clone()
        torch.manual_seed(_stable_fold_seed(NEW_BASE_SEED, FOLDS[0]))
        new = GraphTransformer(build_model_config(config))
        new_weights = new.node_encoder.node_encoder[0].weight.detach().clone()
        torch.manual_seed(_stable_fold_seed(NEW_BASE_SEED, FOLDS[0]))
        repeated = GraphTransformer(build_model_config(config))
        assert torch.equal(new_weights, repeated.node_encoder.node_encoder[0].weight)
        assert not torch.equal(old, new_weights)
    return {'new_seed_changes_initial_weights': True, 'same_seed_reproduces_initial_weights': True}


def split_option_check(config, temp):
    """Legacy fallback, integer zero, invalid types, and local split RNG isolation."""
    old = copy.deepcopy(config)
    old['training'].pop('validation_split_base_seed', None)
    old['training'].pop('validation_split_seed', None)
    legacy, _, _ = make_fold_config(old, FOLDS[0], 'unused', str(temp), 123)
    assert legacy['experiment']['seed'] == 123 and 'validation_split_seed' not in legacy['training']
    assert _validation_split_seed(legacy) == 123
    explicit_zero = copy.deepcopy(legacy)
    explicit_zero['training']['validation_split_seed'] = 0
    torch_before = torch.random.get_rng_state().clone()
    numpy_before = np.random.get_state()
    assert _validation_split_seed(explicit_zero) == 0
    numpy_after = np.random.get_state()
    assert torch.equal(torch_before, torch.random.get_rng_state())
    assert numpy_before[0] == numpy_after[0] and np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    zero = copy.deepcopy(old)
    zero['training']['validation_split_base_seed'] = 0
    resolved, _, _ = make_fold_config(zero, FOLDS[0], 'unused', str(temp), 123)
    assert resolved['training']['validation_split_seed'] == _stable_fold_seed(0, FOLDS[0])
    for bad in (True, False, None, '42', 42.5):
        invalid = copy.deepcopy(old)
        invalid['training']['validation_split_base_seed'] = bad
        try:
            make_fold_config(invalid, FOLDS[0], 'unused', str(temp), 123)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid split seed accepted: {bad!r}')
        invalid['training'].pop('validation_split_base_seed')
        invalid['training']['validation_split_seed'] = bad
        try:
            _validation_split_seed(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid explicit split seed accepted: {bad!r}')
    try:
        create_datasets(config)
    except ValueError as error:
        assert 'Unresolved validation_split_base_seed' in str(error)
    else:
        raise AssertionError('Direct CLI silently ignored unresolved fold split base seed')
    return {'legacy_fallback_unchanged': True, 'zero_not_falsy_fallback': True,
            'invalid_types_rejected': True, 'unresolved_base_seed_rejected_before_io': True,
            'production_seed_resolution_preserves_global_rng': True}


def to_cuda(batch):
    return {key: value.cuda() if torch.is_tensor(value) else value
            for key, value in batch.items()}


def collate(dataset, indices):
    return GraphDataLoader._default_collate_fn(None, [dataset[i] for i in indices])


def longest_indices(dataset, count):
    sequences = dataset.data['seq'].tolist()
    assert len(sequences) == len(dataset) and len(sequences) >= count
    return sorted(range(len(sequences)), key=lambda i: len(str(sequences[i])), reverse=True)[:count]


def topology_proof(batch):
    """Check true global bidirectionality and exact valid-bond bookkeeping."""
    seq_lens = batch['seq_lens'].cpu()
    node_lens = batch['node_lens'].cpu()
    assert torch.equal(node_lens, seq_lens + 1)
    edge_index = batch['edge_index'].cpu()
    edge_types = batch['edge_types'].cpu()
    offset = 0
    for length, node_length in zip(seq_lens.tolist(), node_lens.tolist()):
        global_index = offset + length
        residues = torch.arange(offset, global_index)
        incoming = edge_index[1] == global_index
        outgoing = edge_index[0] == global_index
        assert torch.equal(edge_index[0, incoming].sort().values, residues)
        assert torch.equal(edge_index[1, outgoing].sort().values, residues)
        assert (edge_types[incoming | outgoing] == 4).all()
        offset += node_length
    positions = torch.arange(batch['label_mask'].shape[1])
    expected = positions.unsqueeze(0) < (seq_lens - 1).clamp_min(0).unsqueeze(1)
    assert torch.equal(batch['label_mask'].bool().cpu(), expected)
    assert batch['bond_theory'].shape == (*expected.shape, 15)
    assert torch.allclose(1 - batch['four_state_targets'][..., 0][expected],
                          batch['soft_labels'][expected], atol=1e-7, rtol=0)
    return {'global_nodes': len(seq_lens), 'packed_nodes': offset,
            'valid_bonds': int(expected.sum()), 'bidirectional_all_residues': True}


def shape_hooks(model, width, observations):
    handles = []

    def record(name, in_last=None, out_last=None, graph=False):
        def hook(module, args, output):
            assert torch.is_tensor(output), name
            if in_last is not None:
                assert args[0].shape[-1] == in_last, (name, args[0].shape)
            if out_last is not None:
                assert output.shape[-1] == out_last, (name, output.shape)
            if graph:
                assert args[0].ndim == output.ndim == 2
                assert args[0].shape[0] == output.shape[0]
                assert args[2].shape[-1] == width
            entry = {'output_shape': list(output.shape)}
            if args and torch.is_tensor(args[0]):
                entry['input_shape'] = list(args[0].shape)
            observations.setdefault(name, entry)
        return hook

    handles.append(model.node_encoder.register_forward_hook(record('node_encoder', out_last=width)))
    handles.append(model.node_encoder.node_encoder[0].register_forward_hook(
        record('raw_node_projection', in_last=192, out_last=width)))
    handles.append(model.edge_encoder.register_forward_hook(record('edge_encoder', out_last=width)))
    handles.append(model.global_node_proj.register_forward_hook(
        record('global_condition_projection', in_last=64, out_last=width)))
    handles.append(model.film_mlp.register_forward_hook(record('film', in_last=2, out_last=2 * width)))
    handles.append(model.bond_theory_proj.register_forward_hook(
        record('theory_projection', in_last=15, out_last=width)))
    handles.append(model.bond_head[0].register_forward_hook(
        record('bond_fusion', in_last=6 * width, out_last=width)))
    handles.append(model.bond_head[3].register_forward_hook(
        record('four_state_head', in_last=width, out_last=4)))
    assert tuple(model.global_node_embedding.shape) == (1, width)
    assert len(model.gcn_layers) == 3 and len(model.gat_layers) == 2
    for name, layers in (('gcn', model.gcn_layers), ('gat', model.gat_layers)):
        for index, layer in enumerate(layers):
            if name == 'gat':
                assert layer.num_heads == 4 and layer.head_dim == width // 4 and layer.concat
            handles.append(layer.register_forward_hook(
                record(f'{name}_{index + 1}', in_last=width, out_last=width, graph=True)))
    return handles


def masked_metric_check(predictions, batch):
    """Poison old and newly appended padding; production metrics must not change."""
    logits = predictions.detach().float().cpu()
    labels = batch['labels'].detach().cpu()
    soft = batch['soft_labels'].detach().cpu()
    mask = batch['label_mask'].bool().cpu()
    keys = batch_group_keys(batch)

    def compute(values, targets, valid, q):
        calculator = BinaryBondMetrics({'threshold': .5, 'threshold_strategy': 'fixed'},
                                       allow_target_aware_threshold=False)
        calculator.update(values, targets, valid, from_logits=True, soft_targets=q,
                          sequences=batch['sequences'], group_keys=keys)
        metrics = calculator.compute()
        assert calculator.last_probabilities.size == int(mask.sum())
        assert calculator.last_targets.size == int(mask.sum())
        return metrics

    baseline = compute(logits, labels, mask, soft)
    pad_shape = (logits.shape[0], 3)
    padded_logits = torch.cat((logits, torch.full(pad_shape, float('nan'))), dim=1)
    padded_labels = torch.cat((labels, torch.ones(pad_shape)), dim=1)
    padded_soft = torch.cat((soft, torch.full(pad_shape, float('nan'))), dim=1)
    padded_mask = torch.cat((mask, torch.zeros(pad_shape, dtype=torch.bool)), dim=1)
    padded_logits[~padded_mask] = float('nan')
    padded_labels[~padded_mask] = 1
    padded_soft[~padded_mask] = float('nan')
    poisoned = compute(padded_logits, padded_labels, padded_mask, padded_soft)
    assert baseline.keys() == poisoned.keys()
    for key, value in baseline.items():
        assert math.isfinite(float(value)), (key, value)
        assert float(value) == float(poisoned[key]), (key, value, poisoned[key])
    predicted = logits[mask] > 0
    target = labels[mask].bool()
    accuracy = (predicted == target).double().mean().item()
    tp = int((predicted & target).sum())
    fp = int((predicted & ~target).sum())
    fn = int((~predicted & target).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    assert abs(accuracy - baseline['accuracy']) < 1e-12
    assert abs(f1 - baseline['f1']) < 1e-12
    brier = ((logits[mask].sigmoid().double() - soft[mask].double()) ** 2).mean().item()
    assert abs(brier - baseline['q_brier']) < 1e-12
    names = ('accuracy', 'f1', 'auc', 'ex_precision', 'ex_recall', 'q_brier', 'q_spearman_pep_seq')
    result = {name: float(baseline[name]) for name in names if name in baseline}
    precision, recall = baseline['ex_precision'], baseline['ex_recall']
    result['spectrum_pr_harmonic_f1'] = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    result.update(padding_free=True, poisoned_padding_invariant=True, valid_bonds=int(mask.sum()))
    return result


def memory():
    torch.cuda.synchronize()
    return {'allocated_bytes': torch.cuda.max_memory_allocated(),
            'reserved_bytes': torch.cuda.max_memory_reserved()}


def smoke_update(model, trainer, batch):
    model.train()
    before = model.bond_head[3].weight.detach().clone()
    seed()
    loss, stats = trainer._forward_pass(batch)
    assert torch.isfinite(loss)
    logits = model.last_four_state_logits
    assert logits.shape == (int(batch['label_mask'].sum()), 4)
    expected = four_state_cross_entropy(logits, batch['four_state_targets'],
        batch['label_mask'], trainer._masked_row_weights(batch),
        conditional_weight=1., union_brier_weight=0.)
    assert torch.equal(loss, expected), 'Production CE differs from alpha=1/lambda=0'
    logits.retain_grad()
    trainer._backward_pass(loss)
    assert not torch.equal(before, model.bond_head[3].weight), 'AMP optimizer update skipped'
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert (logits.grad.abs().sum(0) > 0).all(), 'One state has no gradient'
    result = {'loss': float(loss.detach()), 'valid_bonds': stats['valid_bond_count'],
              'gradient_norm': trainer.last_grad_norm, 'amp': trainer.use_amp}
    assert math.isfinite(result['gradient_norm'])
    del loss, expected, logits, before
    trainer.optimizer.zero_grad(set_to_none=True)
    model.last_four_state_logits = None
    trainer.metrics_calculator.reset()
    return result


def checkpoint_roundtrip(model, config, folder, width, reference_path):
    path = folder / f'width_{width}.pt'
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save({'model_state_dict': state, 'hidden_dim': width}, path)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    restored = GraphTransformer(build_model_config(config))
    restored.load_state_dict(saved['model_state_dict'], strict=True)
    assert saved['hidden_dim'] == width
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in state.items())
    cross_width_rejected = None
    if reference_path is not None:
        before = {key: value.clone() for key, value in restored.state_dict().items()}
        incompatible = torch.load(reference_path, map_location='cpu', weights_only=True)
        try:
            restored.load_state_dict(incompatible['model_state_dict'], strict=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Cross-width checkpoint accepted')
        assert all(torch.equal(value, restored.state_dict()[key]) for key, value in before.items())
        cross_width_rejected = True
    del restored, state, saved
    return path, {'same_width_file_roundtrip_exact': True,
                  'cross_width_rejected_before_copy': cross_width_rejected}


def run(output):
    assert sys.platform.startswith('linux') and str(ROOT).startswith('/mnt/pvc/'), 'Pod execution only'
    assert torch.cuda.is_available(), 'An available pod GPU is required'
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    check_loss()
    args = argparse.Namespace(epochs=None, batch_size=None, learning_rate=None, device=None)
    configs = {str(w): load_config(config_path(w), args) for w in WIDTHS}
    original = load_config(ROOT / 'graph_transform/config/pre_synthesis_four_state_joint.yaml', args)
    canonical = normalized(original, 'four_state_joint')
    for arm, config in configs.items():
        assert normalized(config, f'four_state_width_replication_{arm}') == canonical, f'Unexpected config change: {arm}'
        old_config = load_config(ROOT / f'graph_transform/config/pre_synthesis_four_state_width_{arm}.yaml', args)
        assert old_config['experiment']['seed'] == ORIGINAL_BASE_SEED
        assert config['experiment']['seed'] == NEW_BASE_SEED
        assert config['training']['validation_split_base_seed'] == ORIGINAL_BASE_SEED
        assert 'validation_split_seed' not in config['training']
        assert normalized(config, f'four_state_width_replication_{arm}') == normalized(old_config, f'four_state_width_{arm}')
        model_config = build_model_config(config)
        assert model_config.hidden_dim == int(arm)
        assert model_config.num_attention_heads == 4 and model_config.concat_heads
        assert model_config.hidden_dim % model_config.num_attention_heads == 0
        assert model_config.bond_head_type == 'four_state_mlp'
        assert model_config.use_charge_partition_features is False
        assert config['loss']['four_state_conditional_weight'] == 1.
        assert config['loss']['four_state_union_brier_weight'] == 0.
        assert config['training']['batch_size'] == 128
        assert model_config.state_feature_mask == [True, True, False]
        assert model_config.env_feature_mask == [True, False]
        assert not config['training'].get('pretrained_backbone_path')
        assert not config['training'].get('pretrained_backbone_sha256')
        assert model_config.theory_fusion_stage == 'late'
    prior = json.loads(PRIOR.read_text())
    assert prior['status'] == 'passed'
    discovery = json.loads(DISCOVERY.read_text())
    assert discovery['status'] == 'passed' and discovery['folds'] == prior['folds']
    report = {'status': 'running', 'folds': {}, 'widths': {}, 'arms': {},
              'configs': {str(w): file_sha(config_path(w)) for w in WIDTHS},
              'prior_validation_proof': str(PRIOR), 'prior_validation_proof_sha256': file_sha(PRIOR),
              'discovery_validation_proof': str(DISCOVERY), 'discovery_validation_proof_sha256': file_sha(DISCOVERY),
              'device': torch.cuda.get_device_name(),
              'training_base_seed': NEW_BASE_SEED, 'validation_split_base_seed': ORIGINAL_BASE_SEED,
              'fold_seeds': {}, 'initialization': {},
              'checks': ['Only hidden width, declared new training seed, fixed split RNG and identity paths change',
                         'Both widths keep four concatenated GAT heads, alpha=1, lambda=0, no charge residual',
                         'Pre-synthesis masks exclude intensity and scan number in every width',
                         'Same five sequence-disjoint splits and full original-spectrum internal validation hashes',
                         'Prior synthetic masked CE/default-gradient/decomposition fixtures pass'],
              'scope': 'Bounded pod-only preflight; not training results and no outer-test metrics'}
    with tempfile.TemporaryDirectory(prefix='four_state_width_') as folder:
        temp = Path(folder)
        report['split_options'] = split_option_check(configs['256'], temp)
        first = None
        for fold in FOLDS:
            fold_configs = {}
            for arm, config in configs.items():
                fold_config, _, _ = make_fold_config(config, fold, str(ROOT / 'dataset/5fold_folded'),
                    str(temp / 'paired'), _stable_fold_seed(43, fold))
                train_seed = _stable_fold_seed(NEW_BASE_SEED, fold)
                split_seed = _stable_fold_seed(ORIGINAL_BASE_SEED, fold)
                assert train_seed != split_seed
                assert fold_config['experiment']['seed'] == train_seed
                assert fold_config['training']['validation_split_seed'] == split_seed
                assert 'validation_split_base_seed' not in fold_config['training']
                fold_config['data']['cache_dir'] = str(temp / fold / 'cache')
                fold_config['data']['cache_graphs'] = False
                fold_configs[arm] = fold_config
            report['fold_seeds'][fold] = {'training': train_seed, 'split': split_seed}
            base = normalized(fold_configs['256'], 'four_state_width_replication_256')
            assert all(normalized(c, f'four_state_width_replication_{a}') == base for a, c in fold_configs.items())
            train, val, test = create_datasets(fold_configs['256'])
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
        small = to_cuda(small_cpu)
        mask = small['label_mask'].bool()
        reference_checkpoint = None
        for width in WIDTHS:
            arm = str(width)
            config = copy.deepcopy(fold_configs[arm])
            config['debug']['diagnostic_dir'] = str(temp / arm / 'diagnostics')
            report['initialization'][arm] = initialization_seed_check(config)
            seed(config['experiment']['seed'])
            model = GraphTransformer(build_model_config(config)).cuda()
            assert model.node_encoder.state_feature_mask.tolist() == [1., 1., 0.]
            assert model.node_encoder.env_feature_mask.tolist() == [1., 0.]
            observations = {}
            handles = shape_hooks(model, width, observations)
            optimizer = create_optimizer(model, config)
            trainer = Trainer(model, optimizer, BinaryBondLoss(config['loss']),
                create_scheduler(optimizer, config), torch.device('cuda:0'), config, LOGGER)
            assert trainer.four_state_conditional_weight == 1.
            assert trainer.four_state_union_brier_weight == 0.
            torch.cuda.reset_peak_memory_stats()
            update = smoke_update(model, trainer, small)
            train_memory = memory()
            with torch.no_grad():
                model.eval()
                prediction = model(small)
            assert prediction.shape == small['labels'].shape and torch.isfinite(prediction).all()
            assert int(mask.sum()) == update['valid_bonds']
            metric_proof = masked_metric_check(prediction, small)
            del prediction
            checkpoint, checkpoint_proof = checkpoint_roundtrip(model, config, temp, width, reference_checkpoint)
            if reference_checkpoint is None:
                reference_checkpoint = checkpoint
            entry = {'hidden_dim': width, 'gat_heads': 4, 'head_dim': width // 4,
                     'parameter_count': sum(p.numel() for p in model.parameters()),
                     'trainable_parameter_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
                     'parameters': sum(p.numel() for p in model.parameters()),
                     'amp': trainer.use_amp, 'optimizer_updates': 1, 'valid_bonds': update['valid_bonds'],
                     'small_batch_size': len(small_indices), 'small_batch_update': update,
                     'peak_cuda_allocated_bytes': train_memory['allocated_bytes'],
                     'peak_cuda_reserved_bytes': train_memory['reserved_bytes'],
                     'training_small_batch_peak_allocated_mb': train_memory['allocated_bytes'] / 1024 ** 2,
                     'training_small_batch_peak_reserved_mb': train_memory['reserved_bytes'] / 1024 ** 2,
                     'shapes': observations, 'checkpoint_roundtrip': checkpoint_proof,
                     'masked_metrics': metric_proof}
            if width == WIDTHS[-1]:
                # Longest real rows stress the unchanged production batch sizes.
                training_indices = longest_indices(train, 128)
                full_train = to_cuda(collate(train, training_indices))
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
                validation_indices = longest_indices(val, 512)
                full_val = to_cuda(collate(val, validation_indices))
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
                del full_val, prediction
            for handle in handles:
                handle.remove()
            report['widths'][arm] = entry
            report['arms'][arm] = entry
            del model, trainer, optimizer
            gc.collect()
            torch.cuda.empty_cache()
    report['checks'] += ['Node/edge/global/FiLM/theory and each GCN/GAT shape verified on actual forwards',
                         'Every width completes production AMP CE, finite backward and a real optimizer step',
                         'Production metrics invariant to poisoned padding; accuracy/F1/q-Brier independently checked',
                         'Same-width saved checkpoint reload exact and cross-width checkpoints rejected before copying',
                         '256 width passes longest-real-row training batch 128 and FP32 validation batch 512',
                         'New per-fold training RNG differs; fixed per-fold split RNG reproduces all five original train/val SHAs']
    report['independent_training_rng_same_splits'] = True
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
