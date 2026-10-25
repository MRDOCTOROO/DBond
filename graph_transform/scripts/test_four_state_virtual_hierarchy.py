"""Bounded pod-only preflight for k5 plus hierarchical virtual graph nodes.

No epoch loop, outer-test metric, or training-progress polling is performed.
Only synthetic topology and fixed internal training/validation subsets are used.
"""
import argparse
import copy
import csv
import gc
import hashlib
import itertools
import json
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
from data.graph_builder import GraphBuilder
from data.graph_dataset import CachedGraphDataset, GraphDataset, GraphDataLoader
from data.row_validation import file_sha
from models.graph_transformer import GraphTransformer, _pool_virtual_groups
from models.utils import build_model_config, validate_virtual_node_checkpoint
from training.loss_functions import BinaryBondLoss
from training.trainer import Trainer
from test_four_state_width import collate, longest_indices, masked_metric_check, memory, to_cuda
from test_four_state_dynamic_edges import seed, rng_state, same_rng, cpu_state, same_tensors
from test_four_state_head_fusion import backbone_shape_hooks
from test_four_state_weight import check_loss
from test_four_state_width_replication import split_option_check
from train_graph_model import (load_config, create_datasets, create_optimizer,
                               create_scheduler, save_checkpoint_with_selection)
from train_5fold import make_fold_config, _stable_fold_seed
import test_four_state_head_fusion as fusion_fixtures
import train_graph_model as train_script
import four_state_virtual_hierarchy_ops as ops

MODES = ('single', 'grouped_skip', 'grouped_strict')
FOLDS = ('1222', '2252', '3514', '6072', '9075')
PRIOR = Path('/mnt/pvc/graphtrans/four_state_ops/smoke.json')
WIDTH_PRIOR = Path('/mnt/pvc/graphtrans/four_state_width_ops/smoke.json')
K5_PRIOR = Path('/mnt/pvc/graphtrans/four_state_node_conv_replication_ops/smoke.json')
OLD_COMMIT = '8ec60bca58b879cba5632379141238d44cc2074e'
OLD_CLASS_FILE = 'graph_transform/models/graph_transformer.py'
train_script.logger = __import__('logging').getLogger('virtual_hierarchy_preflight')


def config_path(mode):
    return ROOT / f'graph_transform/config/pre_synthesis_four_state_virtual_hierarchy_{mode}.yaml'


def normalized(config, mode):
    """Remove topology, then replace only mode-specific artifact path/name tokens."""
    value = copy.deepcopy(config)
    value['experiment'].pop('description', None)
    value['model'].pop('virtual_node_mode', None)
    value['model'].pop('virtual_group_size', None)
    value['training'].pop('validation_split_seed', None)
    token = f'four_state_virtual_hierarchy_{mode}'
    return json.dumps(value, sort_keys=True).replace(token, 'four_state_virtual_hierarchy_ARM')


def old_class():
    source = subprocess.check_output(['git', 'show', f'{OLD_COMMIT}:{OLD_CLASS_FILE}'], cwd=ROOT)
    module = types.ModuleType('models._legacy_virtual_hierarchy_control')
    module.__package__, module.__file__ = 'models', f'git:{OLD_COMMIT}:{OLD_CLASS_FILE}'
    sys.modules[module.__name__] = module
    exec(compile(source.decode('utf-8'), module.__file__, 'exec'), module.__dict__)
    return module.GraphTransformer, hashlib.sha256(source).hexdigest()


def old_k5_compatibility(config, batch):
    Legacy, old_sha = old_class()
    current_cfg = build_model_config(config)
    with torch.random.fork_rng(devices=[]):
        seed(config['experiment']['seed'])
        old = Legacy(current_cfg).cpu()
        old_rng = rng_state()
        seed(config['experiment']['seed'])
        current = GraphTransformer(current_cfg).cpu()
        same_rng(old_rng, rng_state())
        same_tensors(cpu_state(old), cpu_state(current))
        current.load_state_dict(cpu_state(old), strict=True)
        outputs = []
        for model in (old, current):
            model.train()
            model.zero_grad(set_to_none=True)
            seed(config['experiment']['seed'])
            result = model(batch)
            logits = model.last_four_state_logits.detach().clone()
            result[batch['label_mask'].bool()].square().mean().backward()
            grads = {name: value.grad.detach().clone() for name, value in model.named_parameters()
                     if value.grad is not None}
            outputs.append((result.detach().clone(), logits, grads, rng_state()))
        assert torch.equal(outputs[0][0], outputs[1][0])
        assert torch.equal(outputs[0][1], outputs[1][1])
        same_tensors(outputs[0][2], outputs[1][2])
        same_rng(outputs[0][3], outputs[1][3])
    return {'old_commit': OLD_COMMIT, 'old_class_sha256': old_sha,
            'single_state_rng_forward_backward_exact': True, 'old_checkpoint_strict_load': True}


def graph_edges(config, length):
    builder = GraphBuilder(config)
    edge_index, types, distances, bond_map = builder._get_or_build_edges(length, 'distance')
    edges = [tuple(row) for row in edge_index.t().tolist()]
    return builder, edges, types.tolist(), distances.tolist(), bond_map.tolist()


def topology_check(configs):
    proof = {}
    for length in (0, 1, 4, 5, 6, 11):
        original = None
        for mode in MODES:
            builder, edges, kinds, distances, bond_map = graph_edges(configs[mode], length)
            groups = 0 if mode == 'single' else (length + 4) // 5
            top = length + groups
            assert builder.node_count(length) == top + 1
            assert len(edges) == len(set(edges))
            expected = set()
            if mode == 'single':
                for i in range(length):
                    expected.update(((i, top), (top, i)))
            else:
                for i in range(length):
                    group = length + i // 5
                    expected.update(((i, group), (group, i)))
                    if mode == 'grouped_skip':
                        expected.update(((i, top), (top, i)))
                for g in range(groups):
                    expected.update(((length + g, top), (top, length + g)))
            actual = {pair for pair in edges if pair[0] >= length or pair[1] >= length}
            assert actual == expected, f'{mode}/L{length}: virtual edges differ'
            assert all(kinds[i] == 4 and distances[i] == 0 for i, pair in enumerate(edges)
                       if pair in actual), f'{mode}/L{length}: virtual edge code differs'
            assert all(edges[index] == (i, i + 1) for i, index in enumerate(bond_map))
            residue_edges = {pair for pair in edges if pair[0] < length and pair[1] < length}
            if original is None:
                original = residue_edges
            else:
                assert residue_edges == original, 'Original residue topology changed between arms'
            proof[f'{mode}_L{length}'] = {'node_count': builder.node_count(length),
                                         'virtual_edge_count': len(actual),
                                         'bond_count': len(bond_map)}
    keys = [GraphBuilder(configs[mode])._edge_cache_key(11, 'distance') for mode in MODES]
    assert len(set(keys)) == len(keys), 'Graph cache does not separate the modes'
    changed = copy.deepcopy(configs['grouped_skip'])
    changed['model']['virtual_group_size'] = 4
    assert GraphBuilder(changed)._edge_cache_key(11, 'distance') != keys[1]
    return proof


def cache_check(configs, source_csv, folder):
    """Create real-row cached datasets in one directory and reject cross-mode reuse."""
    result = {}
    subset = folder / 'bounded_internal_val16.csv'
    folder.mkdir(parents=True, exist_ok=True)
    with Path(source_csv).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(itertools.islice(reader, 16))
    assert fields and len(rows) == 16
    with subset.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for mode in MODES:
        runtime = copy.deepcopy(configs[mode])
        dataset = CachedGraphDataset(str(subset), runtime, cache_dir=str(folder),
            max_seq_len=runtime['data']['max_seq_len'], graph_strategy='distance',
            augmentation=False, split='val', cache_full_graphs=True)
        plain = GraphDataset(str(subset), runtime,
            max_seq_len=runtime['data']['max_seq_len'], graph_strategy='distance',
            augmentation=False, split='val')
        for row_index in (0, len(dataset) - 1):
            sample, uncached = dataset[row_index], plain[row_index]
            assert sample['node_len'] == dataset.graph_builder.node_count(sample['seq_len'])
            for field in ('edge_index', 'edge_attr', 'edge_types', 'edge_distances',
                          'bond_edge_map', 'labels', 'node_len', 'seq_len'):
                left, right = sample[field], uncached[field]
                assert torch.equal(left, right) if isinstance(left, torch.Tensor) else left == right, \
                    f'{mode}/{row_index}: cached and uncached {field} differ'
        result[mode] = {'edge_cache': dataset.edge_cache_file, 'full_graph_cache': dataset.cache_file,
                        'edge_meta': dataset._edge_cache_meta()}
        assert result[mode]['edge_meta']['virtual_node_mode'] == mode
        assert result[mode]['edge_meta']['virtual_group_size'] == 5
    assert len({item['edge_cache'] for item in result.values()}) == len(MODES)
    assert len({item['full_graph_cache'] for item in result.values()}) == len(MODES)
    changed = copy.deepcopy(configs['grouped_skip'])
    changed['model']['virtual_group_size'] = 4
    dataset = CachedGraphDataset(str(subset), changed, cache_dir=str(folder),
        max_seq_len=changed['data']['max_seq_len'], graph_strategy='distance',
        augmentation=False, split='val', cache_full_graphs=True)
    assert dataset.edge_cache_file != result['grouped_skip']['edge_cache']
    assert dataset.cache_file != result['grouped_skip']['full_graph_cache']
    result['group_size_4_cache_isolated'] = True
    return result


def pooling_padding_check():
    """Poison and extend padding while preserving valid group means, including L=0/1."""
    with torch.random.fork_rng(devices=[]):
        seed(314159)
        features = torch.randn(4, 11, 256)
    lengths = torch.tensor([0, 1, 6, 11], dtype=torch.long)
    original, counts = _pool_virtual_groups(features, lengths, 5)
    assert counts.tolist() == [0, 1, 2, 3]
    poison = features.clone()
    poison[torch.arange(11)[None, :] >= lengths[:, None]] = float('nan')
    changed, changed_counts = _pool_virtual_groups(poison, lengths, 5)
    assert torch.equal(counts, changed_counts) and torch.equal(original, changed)
    longer = torch.cat((features, torch.full((4, 7, 256), float('inf'))), dim=1)
    extended, extended_counts = _pool_virtual_groups(longer, lengths, 5)
    assert torch.equal(counts, extended_counts)
    assert torch.equal(original, extended[:, :original.shape[1]])
    for row, count in enumerate(counts.tolist()):
        if count:
            assert torch.equal(original[row, :count], extended[row, :count])
    return {'lengths': lengths.tolist(), 'poisoned_padding_masked': True,
            'extended_padding_preserves_valid_group_means': True}


def model_batch_isolation(model, dataset):
    """A peptide is independent of real mixed-batch padding and another peptide."""
    short = min((i for i in range(len(dataset)) if len(dataset.data.iloc[i]['seq']) >= 2),
                key=lambda i: len(dataset.data.iloc[i]['seq']))
    long = max(range(len(dataset)), key=lambda i: len(dataset.data.iloc[i]['seq']))
    original = collate(dataset, [short])
    mixed = collate(dataset, [short, long])
    assert int(mixed['seq_lens'][1]) > int(mixed['seq_lens'][0])
    altered_other = copy.deepcopy(mixed)
    altered_other['state_vars'][1] += 3
    altered_other['env_vars'][1] -= 2
    altered_other['charges'][1] += 1
    altered_other['nces'][1] += 10
    model.eval()
    with torch.no_grad():
        base, together, other_changed = model(original), model(mixed), model(altered_other)
        def poison_padding(module, args, output):
            changed = output.clone()
            changed[0, int(mixed['seq_lens'][0]):] = float('nan')
            return changed
        hook = model.node_encoder.register_forward_hook(poison_padding)
        try:
            poisoned = model(mixed)
        finally:
            hook.remove()
    bonds = max(int(original['seq_lens'][0]) - 1, 0)
    assert bonds > 0 and torch.allclose(together[0, :bonds], poisoned[0, :bonds],
                                        atol=1e-5, rtol=1e-5)
    assert torch.allclose(base[0, :bonds], together[0, :bonds], atol=1e-5, rtol=1e-5)
    assert torch.allclose(together[0, :bonds], other_changed[0, :bonds], atol=1e-5, rtol=1e-5)
    return {'selected_short_length': int(original['seq_lens'][0]),
            'selected_other_length': int(mixed['seq_lens'][1]),
            'poisoned_encoder_padding_invariant': True, 'mixed_peptide_independent': True,
            'other_peptide_conditions_independent': True}


def virtual_edge_response(model, batch):
    """Graph message path actually responds when all virtual edges are removed."""
    model.eval()
    with torch.no_grad():
        original = model(batch)
        total = int(batch['node_lens'].sum())
        residue = torch.zeros(total, dtype=torch.bool)
        offset = 0
        for length, node_len in zip(batch['seq_lens'].tolist(), batch['node_lens'].tolist()):
            residue[offset:offset + length] = True
            offset += node_len
        edge_index = batch['edge_index']
        keep = residue[edge_index[0]] & residue[edge_index[1]]
        assert 0 < int(keep.sum()) < keep.numel()
        altered = copy.deepcopy(batch)
        for key in ('edge_index', 'edge_attr', 'edge_types', 'edge_distances'):
            altered[key] = batch[key][:, keep] if key == 'edge_index' else batch[key][keep]
        old_to_new = torch.cumsum(keep.long(), dim=0) - 1
        altered['bond_edge_map'] = old_to_new[batch['bond_edge_map']]
        with torch.no_grad():
            without_virtual = model(altered)
    valid = batch['label_mask'].bool()
    delta = (original[valid] - without_virtual[valid]).abs().max().item()
    assert delta > 1e-8 and torch.isfinite(without_virtual).all(), \
        'Valid bond output did not respond to removal of virtual graph edges'
    return {'max_valid_bond_logit_change': delta, 'virtual_edges_removed': int((~keep).sum())}


def mode_checkpoint_check(configs, states, folder):
    result = {}
    for mode in MODES:
        path = folder / f'{mode}.pt'
        torch.save({'model_state_dict': states[mode], 'config': configs[mode]}, path)
        saved = torch.load(path, map_location='cpu', weights_only=True)
        model = GraphTransformer(build_model_config(configs[mode])).cpu()
        validate_virtual_node_checkpoint(model, saved['model_state_dict'], saved['config']['model'])
        model.load_state_dict(saved['model_state_dict'], strict=True)
        same_tensors(states[mode], cpu_state(model))
        for other in MODES:
            if other == mode:
                continue
            target = GraphTransformer(build_model_config(configs[other])).cpu()
            before = cpu_state(target)
            for strict in (True, False):
                try:
                    target.load_state_dict(saved['model_state_dict'], strict=strict)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError(f'Cross-mode {mode}->{other} load accepted, strict={strict}')
                same_tensors(before, cpu_state(target))
        result[mode] = {'same_mode_strict_restore': True, 'cross_mode_precopy_reject': True,
                        'checkpoint_sha256': file_sha(path)}
    return result


def smoke_update(model, trainer, batch):
    old = fusion_fixtures.seed
    try:
        fusion_fixtures.seed = lambda: seed(trainer.config['experiment']['seed'])
        return fusion_fixtures.smoke_update(model, trainer, batch)
    finally:
        fusion_fixtures.seed = old


def prior_prediction_reconstruction():
    """Exercise the new wide-prediction audit on one already completed pod fold."""
    root = Path(ops.PREVIOUS_ROOT)
    manifest = ops.read_json(root / 'manifest.json')
    arm = manifest['arms'][0]
    with Path(arm['metrics_csv']).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    row = next(value for value in rows if value['fold_id'] == FOLDS[0])
    fold_folder = Path(arm['cv_root']) / f'fold_{FOLDS[0]}'
    runtime = ops.yaml.safe_load((fold_folder / 'config.yaml').read_text(encoding='utf-8'))
    checkpoints = list(fold_folder.glob('checkpoints/**/best_model.pt'))
    assert len(checkpoints) == 1
    expected = {key: float(row[key]) for key in
                ('f1', 'accuracy', 'auc', 'ex_precision', 'ex_recall', 'q_brier')}
    expected['q_seq'] = float(row['q_spearman_pep_seq'])
    expected['max_seq_len'] = runtime['data']['max_seq_len']
    pred_path = ops.prediction_path_for_fold(fold_folder, Path(row['metric_csv_path']))
    return ops.recompute_prediction_metrics(pred_path,
        Path(runtime['data']['test_csv_path']), checkpoints[0], expected)


def run(gpu=0):
    assert sys.platform.startswith('linux') and str(ROOT).startswith('/mnt/pvc/'), 'Pod only'
    assert torch.cuda.is_available() and isinstance(gpu, int) and not isinstance(gpu, bool)
    torch.cuda.set_device(gpu)
    torch.set_num_threads(2)
    check_loss()
    args = argparse.Namespace(epochs=None, batch_size=None, learning_rate=None, device=None)
    configs = {mode: load_config(config_path(mode), args) for mode in MODES}
    canonical = normalized(configs['single'], 'single')
    assert all(normalized(config, mode) == canonical for mode, config in configs.items()), \
        'Arms differ beyond virtual topology or description/output identity'
    for mode, config in configs.items():
        ops.validate_runtime(config, mode)
        assert config['experiment']['seed'] == 44
        assert config['training']['validation_split_base_seed'] == 42
        assert not config['training'].get('pretrained_backbone_path')
        assert not config['training'].get('pretrained_backbone_sha256')
    report = {'status': 'running', 'configs': {mode: file_sha(config_path(mode)) for mode in MODES},
              'folds': {}, 'variants': {}, 'resources': {}, 'hierarchy_checks': {},
              'training_base_seed': 44, 'validation_split_base_seed': 42,
              'device': torch.cuda.get_device_name(gpu), 'gpu': gpu,
              'scope': 'Bounded pod-only preflight; no epochs or outer-test metrics.'}
    old_proof = json.loads(K5_PRIOR.read_text(encoding='utf-8'))
    old_width = json.loads(WIDTH_PRIOR.read_text(encoding='utf-8'))
    old_four = json.loads(PRIOR.read_text(encoding='utf-8'))
    assert old_proof['status'] == old_width['status'] == old_four['status'] == 'passed'
    report['topology'] = topology_check(configs)
    report['pooling_padding'] = pooling_padding_check()
    report['prediction_reconstruction_prior_fold'] = prior_prediction_reconstruction()
    with tempfile.TemporaryDirectory(prefix='four_state_virtual_hierarchy_') as temporary:
        temp = Path(temporary)
        report['split_options'] = split_option_check(configs['single'], temp)
        first = None
        for fold in FOLDS:
            resolved = {}
            for mode, config in configs.items():
                actual, _, _ = make_fold_config(config, fold, str(ROOT / 'dataset/5fold_folded'),
                    str(temp / 'paired'), _stable_fold_seed(44, fold))
                assert actual['experiment']['seed'] == _stable_fold_seed(44, fold)
                assert actual['training']['validation_split_seed'] == _stable_fold_seed(42, fold)
                actual['data']['cache_graphs'] = False
                actual['data']['cache_dir'] = str(temp / fold / mode / 'cache')
                resolved[mode] = actual
            train, val, test = create_datasets(resolved['single'])
            sets = [set(dataset.data['seq']) for dataset in (train, val, test)]
            assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            proof = {'train_rows': len(train), 'val_rows': len(val),
                     'train_sequences': len(sets[0]), 'val_sequences': len(sets[1]),
                     'train_sha256': file_sha(train.csv_path), 'val_sha256': file_sha(val.csv_path),
                     'source_sha256': val.row_validation_provenance['source_sha256']}
            assert proof == old_width['folds'][fold]
            assert proof == {key: old_proof['folds'][fold][key] for key in proof}
            assert val.row_validation_provenance['outer_test_used_for_internal_validation'] is False
            assert 'four_state_targets' not in val[0] and 'four_state_targets' not in test[0]
            proof.update(training_seed=_stable_fold_seed(44, fold), split_seed=_stable_fold_seed(42, fold))
            report['folds'][fold] = proof
            if first is None:
                first = (resolved, train, val)
        resolved, train, val = first
        assert len(train) >= 128 and len(val) >= 512
        indices = list(range(8)) + list(range(len(train) - 8, len(train)))
        control_cpu = collate(train, indices)
        report['old_k5_compatibility'] = old_k5_compatibility(resolved['single'], control_cpu)
        report['cache_isolation'] = cache_check(resolved, str(val.csv_path), temp / 'shared_cache')
        states, counts, predictions = {}, {}, {}
        for mode in MODES:
            runtime = copy.deepcopy(resolved[mode])
            runtime['debug']['diagnostic_dir'] = str(temp / mode / 'diagnostics')
            seed(runtime['experiment']['seed'])
            model = GraphTransformer(build_model_config(runtime)).cuda()
            counts[mode] = sum(p.numel() for p in model.parameters())
            assert counts[mode] == 2002284
            state = cpu_state(model)
            assert ('virtual_node_mode_code' in state) == (mode != 'single')
            states[mode] = state
            with torch.random.fork_rng(devices=[]):
                reference = GraphTransformer(build_model_config(runtime)).cpu().eval()
                reference.load_state_dict(state, strict=True)
                train_arm, val_arm, _ = create_datasets(runtime)
                real_cpu = collate(train_arm, indices)
                ops.verify_virtual_batch(real_cpu, mode)
                with torch.no_grad():
                    first_prediction, _ = ops.verify_virtual_packing(reference, real_cpu, mode)
                assert first_prediction.shape == real_cpu['label_mask'].shape
                assert torch.isfinite(first_prediction).all()
                predictions[mode] = first_prediction
                report.setdefault('batch_isolation', {})[mode] = model_batch_isolation(reference, train_arm)
                report.setdefault('virtual_edge_response', {})[mode] = virtual_edge_response(reference, real_cpu)
            observed = {}
            hooks = backbone_shape_hooks(model, observed)
            optimizer = create_optimizer(model, runtime)
            trainer = Trainer(model, optimizer, BinaryBondLoss(runtime['loss']),
                create_scheduler(optimizer, runtime), torch.device(f'cuda:{gpu}'), runtime, train_script.logger)
            assert trainer.use_amp and trainer.four_state_conditional_weight == 1.
            assert trainer.four_state_union_brier_weight == 0.
            small_mode = to_cuda(real_cpu)
            torch.cuda.reset_peak_memory_stats()
            updates = [smoke_update(model, trainer, small_mode), smoke_update(model, trainer, small_mode)]
            short_memory = memory()
            model.eval()
            with torch.no_grad():
                prediction = model(small_mode)
            quality = masked_metric_check(prediction, small_mode)
            torch.cuda.reset_peak_memory_stats()
            full_train = to_cuda(collate(train_arm, longest_indices(train_arm, 128)))
            full_update = smoke_update(model, trainer, full_train)
            train_memory = memory()
            del full_train
            gc.collect()
            torch.cuda.empty_cache()
            full_val = to_cuda(collate(val_arm, longest_indices(val_arm, 512)))
            torch.cuda.reset_peak_memory_stats()
            model.eval()
            with torch.no_grad():
                validation = model(full_val)
            assert validation.shape == full_val['labels'].shape and torch.isfinite(validation).all()
            val_memory = memory()
            del full_val
            for hook in hooks:
                hook.remove()
            states[mode] = cpu_state(model)
            report['variants'][mode] = {'parameter_count': counts[mode], 'virtual_node_mode': mode,
                'node_conv_mode': 'single_k5', 'virtual_group_size': 5, 'optimizer_updates': 3,
                'small_batch_updates': updates, 'full_training_batch_update': full_update,
                'masked_metrics': quality, 'shapes': observed,
                'training_small_batch_peak_allocated_mb': short_memory['allocated_bytes'] / 1024**2,
                'full_training_batch_peak_allocated_mb': train_memory['allocated_bytes'] / 1024**2,
                'validation_batch_peak_allocated_mb': val_memory['allocated_bytes'] / 1024**2,
                'peak_cuda_allocated_bytes': max(short_memory['allocated_bytes'],
                                                train_memory['allocated_bytes'], val_memory['allocated_bytes']),
                'peak_cuda_reserved_bytes': max(short_memory['reserved_bytes'],
                                               train_memory['reserved_bytes'], val_memory['reserved_bytes'])}
            report['resources'][mode] = {key: value for key, value in report['variants'][mode].items()
                if 'peak_' in key or key == 'parameter_count'}
            del model, trainer, optimizer, prediction, validation, reference, real_cpu, small_mode, train_arm, val_arm
            gc.collect()
            torch.cuda.empty_cache()
        assert len(set(counts.values())) == 1
        report['parameter_budget'] = counts
        report['checkpoint_modes'] = mode_checkpoint_check(resolved, states, temp)
        report['production_checkpoint_rng'] = {}
        for mode in MODES:
            runtime = resolved[mode]
            model = GraphTransformer(build_model_config(runtime)).cpu()
            model.load_state_dict(states[mode], strict=True)
            optimizer = create_optimizer(model, runtime)
            protocol = {'training_seed': runtime['experiment']['seed'],
                'validation_split_seed': runtime['training']['validation_split_seed'],
                'hidden_dim': 256, 'bond_head_type': 'four_state_mlp', 'theory_fusion_stage': 'late',
                'four_state_conditional_weight': 1., 'four_state_union_brier_weight': 0.}
            path = temp / f'{mode}_rng.pt'
            save_checkpoint_with_selection(model, optimizer, 0, 0., {'f1': 0.}, str(path), None,
                extra_state={'training_rng_protocol': protocol})
            saved = torch.load(path, map_location='cpu', weights_only=False)
            assert saved['extra_state']['training_rng_protocol'] == protocol
            report['production_checkpoint_rng'][mode] = True
    report['hierarchy_checks'] = {
        'legacy_single_k5_state_rng_cpu_forward_backward_exact': True,
        'three_arm_configuration_diff_only_topology': True,
        'synthetic_l0_l1_l4_l5_l6_l11_graph_node_edge_bond_properties': True,
        'graph_and_full_cache_mode_group_size_isolation': True,
        'bounded_cached_uncached_graph_tensor_equality': True,
        'poisoned_extended_padding_and_mixed_batch_invariance': True,
        'all_arm_outputs_respond_to_virtual_edges': True,
        'prior_completed_fold_prediction_row_metric_reconstruction': True,
        'same_train_val_source_shas_all_five_splits': True,
        'training_seed44_split_base42_separate': True,
        'same_trainable_parameter_budget_and_checkpoint_cross_mode_reject': True,
        'real_four_state_ce_union_q_amp_and_padding_metrics': True,
        'two_small_and_one_full_training_updates_finite': True,
        'all_three_longest_train128_validation512_memory': True,
        'production_checkpoint_rng_metadata_roundtrip': True,
    }
    assert all(report['hierarchy_checks'].values())
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
        result = {'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                  'traceback': traceback.format_exc()}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
        raise
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'output': args.output}))
