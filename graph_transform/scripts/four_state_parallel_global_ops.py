"""Pod-only same-round comparison of one versus four global slots/readout.

All three arms use the verified 256-d single-k5 four-state backbone.  This
module creates one terminal completion event; it never polls epoch progress.
"""
import argparse
import copy
import csv
import datetime
import hashlib
import html
import itertools
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'graph_transform'), str(REPO / 'graph_transform/scripts')]
from data.row_validation import file_sha
from four_state_ops import dump, idle
from train_5fold import _stable_fold_seed
import yaml

OPS = Path('/mnt/pvc/graphtrans/four_state_parallel_global_ops')
PREVIOUS_OPS = Path('/mnt/pvc/graphtrans/four_state_node_conv_replication_ops')
PREVIOUS_ROOT = str(REPO / 'checkpoints/graph_transform/four_state_node_conv_replication_batch/20261023_142354')
PREVIOUS_REVISION = '8ec60bca58b879cba5632379141238d44cc2074e'
HIERARCHY_OPS = Path('/mnt/pvc/graphtrans/four_state_virtual_hierarchy_ops')
HIERARCHY_ROOT = str(REPO / 'checkpoints/graph_transform/four_state_virtual_hierarchy_batch/20261024_104327')
NOTIFIED = Path('/mnt/pvc/graphtrans/dbond_completion_notified.json')
QUEUE = REPO / 'experiments/structural_queue_v1.json'
TARGETS = Path('/mnt/pvc/graphtrans/four_state_targets_v1')
ARMS = ('single', 'parallel4', 'parallel4_cross')
FOLDS = ('1222', '2252', '3514', '6072', '9075')
BASE_SEED = 45
SPLIT_BASE_SEED = 42
BASELINE_REF = '4da48eeab50decd0a31fbf64f2f179478a6624a7'
METRICS = ('f1', 'accuracy', 'auc', 'ex_precision', 'ex_recall',
           'ex_f1_harmonic', 'q_brier', 'q_seq')
REPORT_METRICS = (
    ('f1', '有效键 micro-F1 %', 100), ('accuracy', '有效键 accuracy %', 100),
    ('auc', 'ROC-AUC', 1), ('ex_precision', '谱图平均 P %', 100),
    ('ex_recall', '谱图平均 R %', 100), ('ex_f1_harmonic', '谱图 P/R 调和 F1 %', 100),
    ('q_brier', 'q-Brier ↓', 1), ('q_seq', 'q_seq', 1),
)
CONTRASTS = {
    'parallel4_minus_single': ('parallel4', 'single'),
    'parallel4_cross_minus_single': ('parallel4_cross', 'single'),
    'parallel4_cross_minus_parallel4': ('parallel4_cross', 'parallel4'),
}
PY = str(REPO / '.venv/bin/python')
KNOWN_UNRELATED_SOURCE = 'graph_transform/scripts/ensemble_baseline_preds.py'


def require(ok, why):
    if not ok:
        raise ValueError(why)


def config(arm):
    return REPO / f'graph_transform/config/pre_synthesis_four_state_parallel_global_{arm}.yaml'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def revision():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()


def sources():
    return {str(p.relative_to(REPO)): file_sha(p)
            for p in (REPO / 'graph_transform').rglob('*.py') if '__pycache__' not in p.parts}


def source_provenance(bound, gitrev):
    """Bind the sole pre-existing unrelated dirty file without mislabeling HEAD.

    The frozen source manifest includes every Python file.  The ensemble
    utility has historically been dirty on pod; it is not part of this train
    or evaluation route.  Every other file must match the Git blob exactly.
    """
    require(bound == sources(), 'Live Python source differs from launch/preflight')
    dirty = []
    for name, expected in sorted(bound.items()):
        blob = subprocess.check_output(['git', 'show', f'{gitrev}:{name}'], cwd=REPO)
        blob_sha = hashlib.sha256(blob).hexdigest()
        if blob_sha != expected:
            require(name == KNOWN_UNRELATED_SOURCE,
                    f'Unexpected uncommitted production source: {name}')
            dirty.append({'path': name, 'launch_sha256': expected,
                          'git_blob_sha256': blob_sha})
    require(len(dirty) <= 1, 'More than one unrelated source exception')
    require(not any(name == 'graph_transform.scripts.ensemble_baseline_preds'
                    or name.endswith('.ensemble_baseline_preds')
                    for name in sys.modules),
            'The unrelated ensemble utility was imported into this run')
    return {'gitrev': gitrev, 'source_sha256': bound,
            'known_unrelated_dirty_source': dirty}


def prior_completion_barrier():
    paths = [PREVIOUS_OPS / name for name in ('launch.json', 'terminal.json', 'review_receipt.json')]
    require(all(p.is_file() for p in (*paths, NOTIFIED)),
            'Stage8 k5 replication must complete, pass review and be acknowledged')
    previous, terminal, review = map(read_json, paths)
    require(previous['root'] == terminal['root'] == review['root'] == PREVIOUS_ROOT,
            'Preceding k5 replication root mismatch')
    require(terminal['status'] == 'completed' and review['status'] == 'passed'
            and previous['gitrev'] == terminal['gitrev'] == review['gitrev'] == PREVIOUS_REVISION,
            'Preceding k5 replication failed, incomplete, unreviewed, or has wrong source')
    key = PREVIOUS_ROOT + '|completed'
    reported = read_json(NOTIFIED).get('reported', {})
    require(isinstance(reported, dict) and key in reported, 'Preceding result not acknowledged')
    require(file_sha(PREVIOUS_OPS / 'result.json') == review['result_sha256']
            and file_sha(PREVIOUS_OPS / 'report.md') == review['report_sha256'],
            'Preceding result/report not bound to passing review')
    independent_path = PREVIOUS_OPS / 'heartbeat_review.json'
    require(independent_path.is_file(), 'Predecessor independent review is missing')
    independent = read_json(independent_path)
    require(independent.get('status') == 'passed' and independent.get('root') == PREVIOUS_ROOT
            and independent.get('gitrev') == PREVIOUS_REVISION, 'Predecessor independent review binding failed')
    require(file_sha(PREVIOUS_OPS / 'result.json') == independent.get('result_sha256')
            and file_sha(PREVIOUS_OPS / 'report.md') == independent.get('report_sha256')
            and file_sha(PREVIOUS_OPS / 'checkpoint_registry.json') == independent.get('registry_sha256'),
            'Predecessor independent review artifact hashes changed')
    return {'root': PREVIOUS_ROOT, 'gitrev': previous['gitrev'], 'event_key': key,
            'launch_sha256': file_sha(paths[0]), 'terminal_sha256': file_sha(paths[1]),
            'review_sha256': file_sha(paths[2]), 'independent_review_sha256': file_sha(independent_path),
            'reported_receipt': reported[key]}


def authorization_barrier():
    index = read_json(QUEUE)
    stages = {stage['id']: stage for stage in index['stages']}
    require(stages[8]['status'] == 'reviewed' and stages[8]['root'] == PREVIOUS_ROOT,
            'Queue predecessor must be reviewed')
    require(stages[10]['status'] in {'authorized', 'authorized_waiting_predecessor', 'preparing', 'ready_local_unverified', 'ready',
                                  'preflight_passed', 'submitted'},
            'Parallel-global stage has not been explicitly authorized')
    require(stages[10]['ops'] == str(OPS) and stages[10]['requires_reviewed_stage'] == 8
            and stages[4]['status'] == 'deferred_by_user',
            'Parallel-global queue identity incorrect; learned graph remains deferred')
    # Stage9 finished and was reported as a failed analysis after 15 completed
    # trainings.  A new user authorization creates a separate stage10; it must
    # not reinterpret stage9 as reviewed or rerun the hierarchy batch.
    require(stages[9]['status'] == 'halted' and stages[9]['root'] == HIERARCHY_ROOT
            and stages[9]['event_acknowledged'] == HIERARCHY_ROOT + '|failed',
            'Previous hierarchy terminal has not been reported/acknowledged')
    terminal = read_json(HIERARCHY_OPS / 'terminal.json')
    require(terminal['status'] == 'failed' and terminal['root'] == HIERARCHY_ROOT,
            'Previous hierarchy terminal identity changed')


def validate_runtime(runtime, arm):
    model, loss = runtime['model'], runtime['loss']
    require(model['hidden_dim'] == 256 and model.get('node_conv_mode') == 'single_k5'
            and model.get('virtual_node_mode', 'single') == arm
            and model.get('bond_fusion_mode', 'concat') == 'concat'
            and model.get('edge_state_mode', 'static') == 'static',
            f'{arm}: wrong width/convolution/global-slot mode, or fusion/dynamic-edge changes were enabled')
    require(model.get('sequence_adapter_type', 'none') == 'none',
            f'{arm}: historical AA-embedding sequence adapter must remain disabled')
    require(model.get('bond_cross_attention_heads', 4) == 4, f'{arm}: four local cross-attention heads required')
    require(model.get('theory_fusion_stage', 'late') == 'late'
            and model.get('use_theory_features') is True, f'{arm}: original late theory route required')
    require(model['num_gcn_layers'] == 3 and model['num_gat_layers'] == 2
            and model['num_attention_heads'] == 4 and model['concat_heads'] is True,
            f'{arm}: original 3GCN+2GAT, four concatenated heads required')
    require(model['bond_head_type'] == 'four_state_mlp'
            and not model.get('use_charge_partition_features', False), f'{arm}: four-state nocharge required')
    require(model.get('use_global_node') is True and model.get('use_condition_film') is True
            and model.get('position_embedding_mode', 'input_concat') == 'input_concat',
            f'{arm}: global/FiLM/position route changed')
    require(model.get('max_distance') == 6 and model.get('use_long_range_edges') is False
            and runtime['data']['graph_strategy'] == 'distance', f'{arm}: fixed graph changed')
    require(model.get('edge_types') == ['sequence', 'distance', 'functional', 'long_range', 'global'],
            f'{arm}: existing edge-type schema must remain unchanged')
    require(loss.get('four_state_conditional_weight', 1.) == 1.
            and loss.get('four_state_union_brier_weight', 0.) == 0., f'{arm}: alpha1/Brier0 required')
    require(runtime['data']['weighting_scheme'] == 'group_uniform'
            and runtime['data'].get('use_soft_labels') is True
            and runtime['data'].get('validation_source_csv_path')
            and runtime['training']['checkpoint_selection'] == 'max_f1'
            and runtime['evaluation']['threshold'] == .5
            and runtime['evaluation'].get('threshold_strategy', 'fixed') == 'fixed'
            and runtime['evaluation'].get('export_from_evaluate_cache') is True
            and runtime.get('ablation', {}).get('pre_synthesis') is True,
            f'{arm}: training/validation/mask/evaluation protocol changed')
    seed, training = runtime['experiment']['seed'], runtime['training']
    if 'validation_split_base_seed' in training:
        require(seed == BASE_SEED and training['validation_split_base_seed'] == SPLIT_BASE_SEED
                and 'validation_split_seed' not in training, f'{arm}: base RNG mismatch')
    else:
        pairs = {(_stable_fold_seed(BASE_SEED, fold), _stable_fold_seed(SPLIT_BASE_SEED, fold)) for fold in FOLDS}
        require((seed, training.get('validation_split_seed')) in pairs,
                f'{arm}: actual train/split RNG mismatch')


def validate_proof(proof):
    require(proof.get('status') == 'passed', 'Preflight did not pass')
    require(set(proof['variants']) == set(ARMS) and set(proof['folds']) == set(FOLDS),
            'Missing preflight arm/fold')
    checks = proof.get('parallel_global_checks', {})
    require(checks and all(value is True for value in checks.values()),
            'Not all parallel-global structure/compatibility checks passed')
    counts = {arm: proof['variants'][arm]['parameter_count'] for arm in ARMS}
    require(all(isinstance(n, int) and n > 0 for n in counts.values()), 'Invalid model parameter count')
    require(counts['single'] == 2002284 and counts['parallel4'] > counts['single']
            and counts['parallel4_cross'] > counts['parallel4'],
            'Expected k5 control and increased four-slot/cross parameter budgets')
    require(proof.get('training_base_seed') == BASE_SEED
            and proof.get('validation_split_base_seed') == SPLIT_BASE_SEED,
            'Training/split base seed changed')
    prior_prediction = proof.get('same_pass_prediction_reconstruction', {})
    require(prior_prediction.get('rows', 0) > 0 and prior_prediction.get('valid_bonds', 0) > 0
            and prior_prediction.get('all_metrics_within_tolerance') is True,
            'Same-pass prediction-row metric recomputation did not pass')
    for arm in ARMS:
        validate_runtime(yaml.safe_load(config(arm).read_text(encoding='utf-8')), arm)
        require(file_sha(config(arm)) == proof['configs'][arm], f'{arm}: preflight config changed')


def preflight(gpu=0, output=None):
    """Bounded pod-only checks; no epoch loop or outer-test metric selection."""
    import gc
    import tempfile
    import numpy as np
    import pandas as pd
    import torch
    from data.graph_builder import GraphBuilder
    from data.graph_dataset import CachedGraphDataset, GraphDataset, GraphDataLoader
    from evaluation.evaluator import Evaluator
    from models.graph_transformer import GraphTransformer
    from models.utils import build_model_config, validate_virtual_node_checkpoint
    from training.loss_functions import BinaryBondLoss
    from training.trainer import Trainer
    from test_four_state_width import collate, longest_indices, masked_metric_check, memory, to_cuda
    from test_four_state_dynamic_edges import seed, rng_state, same_rng, cpu_state, same_tensors
    from test_four_state_virtual_hierarchy import (old_k5_compatibility, model_batch_isolation,
                                                    virtual_edge_response)
    from test_four_state_weight import check_loss
    from train_graph_model import (load_config, create_datasets, create_optimizer,
                                   create_scheduler, build_evaluation_id)
    from train_5fold import make_fold_config
    import train_graph_model as train_script

    require(sys.platform.startswith('linux') and str(REPO).startswith('/mnt/pvc/'),
            'Preflight must run on the authorized pod')
    require(torch.cuda.is_available(), 'No CUDA device for preflight')
    torch.cuda.set_device(gpu)
    torch.set_num_threads(2)
    train_script.logger = __import__('logging').getLogger('parallel_global_preflight')
    check_loss()
    args = argparse.Namespace(epochs=None, batch_size=None, learning_rate=None, device=None)
    configs = {arm: load_config(config(arm), args) for arm in ARMS}
    canonical = None
    for arm, runtime in configs.items():
        validate_runtime(runtime, arm)
        require(runtime['experiment']['seed'] == BASE_SEED
                and runtime['training']['validation_split_base_seed'] == SPLIT_BASE_SEED,
                f'{arm}: base seed protocol changed')
        require(not runtime['training'].get('pretrained_backbone_path'),
                f'{arm}: transfer learning disallowed')
        comparison = copy.deepcopy(runtime)
        comparison['experiment'].pop('description', None)
        comparison['model'].pop('virtual_node_mode', None)
        comparison['training'].pop('validation_split_seed', None)
        normalized = json.dumps(comparison, sort_keys=True).replace(
            f'four_state_parallel_global_{arm}', 'four_state_parallel_global_ARM')
        if canonical is None:
            canonical = normalized
        require(normalized == canonical, f'{arm}: configurations differ beyond topology/readout and artifact name')

    proof = {'status': 'running', 'configs': {arm: file_sha(config(arm)) for arm in ARMS},
             'folds': {}, 'variants': {}, 'parallel_global_checks': {},
             'training_base_seed': BASE_SEED, 'validation_split_base_seed': SPLIT_BASE_SEED,
             'device': torch.cuda.get_device_name(gpu), 'gpu': gpu,
             'scope': 'Pod-only bounded structure, source and GPU preflight; no epochs or outer-test model choice'}
    prior = read_json(PREVIOUS_OPS / 'smoke.json')
    require(prior['status'] == 'passed', 'Predecessor k5 preflight did not pass')
    graph_details = {}
    for length in (0, 1, 4, 5, 6, 11, 32):
        residue_reference = None
        for arm in ARMS:
            builder = GraphBuilder(configs[arm])
            edge_index, kinds, distances, bond_ids = builder._get_or_build_edges(length, 'distance')
            edges = [tuple(edge) for edge in edge_index.t().tolist()]
            slots = 1 if arm == 'single' else 4
            expected = {(i, length + slot) for i in range(length) for slot in range(slots)}
            expected |= {(length + slot, i) for i in range(length) for slot in range(slots)}
            observed = {(u, v) for u, v in edges if u >= length or v >= length}
            require(builder.node_count(length) == length + slots and observed == expected
                    and len(edges) == len(set(edges)), f'{arm}/L{length}: wrong parallel topology')
            require(all(kinds[i] == 4 and distances[i] == 0 for i, edge in enumerate(edges)
                        if edge in observed), f'{arm}/L{length}: wrong virtual edge encoding')
            require(all(edges[index] == (i, i + 1) for i, index in enumerate(bond_ids)),
                    f'{arm}/L{length}: adjacent-bond map changed')
            residue = {(u, v) for u, v in edges if u < length and v < length}
            if residue_reference is not None:
                require(residue == residue_reference, 'Residue topology differs across arms')
            residue_reference = residue
            graph_details[f'{arm}/L{length}'] = {'nodes': builder.node_count(length),
                                                  'virtual_edges': len(observed),
                                                  'bond_edges': len(bond_ids)}
    proof['synthetic_graph'] = graph_details
    keys = [GraphBuilder(configs[a])._edge_cache_key(11, 'distance') for a in ARMS]
    require(len(set(keys)) == len(ARMS), 'Graph edge cache does not distinguish the modes')

    with tempfile.TemporaryDirectory(prefix='parallel_global_', dir=str(OPS)) as folder:
        temporary = Path(folder)
        first = None
        for fold in FOLDS:
            resolved = {}
            for arm, base in configs.items():
                runtime, _, _ = make_fold_config(base, fold, str(REPO / 'dataset/5fold_folded'),
                    str(temporary / 'paired'), _stable_fold_seed(BASE_SEED, fold))
                require(runtime['experiment']['seed'] == _stable_fold_seed(BASE_SEED, fold)
                        and runtime['training']['validation_split_seed'] == _stable_fold_seed(SPLIT_BASE_SEED, fold),
                        f'{arm}/{fold}: actual training/validation seed mismatch')
                runtime['data']['cache_graphs'] = False
                runtime['data']['cache_dir'] = str(temporary / fold / arm / 'cache')
                resolved[arm] = runtime
            train, val, test = create_datasets(resolved['single'])
            seq_sets = [set(ds.data['seq']) for ds in (train, val, test)]
            require(not (seq_sets[0] & seq_sets[1] or seq_sets[0] & seq_sets[2]
                         or seq_sets[1] & seq_sets[2]), f'{fold}: sequence split leakage')
            row = {'train_rows': len(train), 'val_rows': len(val),
                   'train_sequences': len(seq_sets[0]), 'val_sequences': len(seq_sets[1]),
                   'train_sha256': file_sha(train.csv_path), 'val_sha256': file_sha(val.csv_path),
                   'source_sha256': val.row_validation_provenance['source_sha256']}
            require(row == {key: prior['folds'][fold][key] for key in row},
                    f'{fold}: historical train/val/source SHA mismatch')
            require(val.row_validation_provenance['outer_test_used_for_internal_validation'] is False,
                    f'{fold}: outer test used for validation')
            row.update(training_seed=_stable_fold_seed(BASE_SEED, fold),
                       split_seed=_stable_fold_seed(SPLIT_BASE_SEED, fold))
            proof['folds'][fold] = row
            if first is None:
                first = (resolved, train, val)
        resolved, train, val = first
        require(len(train) >= 128 and len(val) >= 512, 'Insufficient rows for real batch memory checks')
        indices = list(range(8)) + list(range(len(train) - 8, len(train)))
        control_batch = collate(train, indices)
        proof['legacy_single_k5'] = old_k5_compatibility(resolved['single'], control_batch)
        cache_subset = temporary / 'cache_check_val16.csv'
        with Path(val.csv_path).open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            fields, subset_rows = reader.fieldnames, list(itertools.islice(reader, 16))
        require(fields and len(subset_rows) == 16, 'Missing bounded internal-validation cache rows')
        with cache_subset.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(subset_rows)
        cache_artifacts = {}
        for arm, runtime in resolved.items():
            cached = CachedGraphDataset(str(cache_subset), runtime, cache_dir=str(temporary / 'shared_cache'),
                max_seq_len=runtime['data']['max_seq_len'], graph_strategy='distance',
                augmentation=False, split='val', cache_full_graphs=True)
            uncached = GraphDataset(str(cache_subset), runtime,
                max_seq_len=runtime['data']['max_seq_len'], graph_strategy='distance',
                augmentation=False, split='val')
            for row_index in (0, len(cached) - 1):
                left, right = cached[row_index], uncached[row_index]
                for field in ('edge_index', 'edge_attr', 'edge_types', 'edge_distances',
                              'bond_edge_map', 'labels', 'node_len', 'seq_len'):
                    a, b = left[field], right[field]
                    require(torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b,
                            f'{arm}/{row_index}: cached and uncached {field} differ')
            cache_artifacts[arm] = {'edge_cache': str(cached.edge_cache_file),
                                    'full_graph_cache': str(cached.cache_file)}
        require(len({item['edge_cache'] for item in cache_artifacts.values()}) == 3
                and len({item['full_graph_cache'] for item in cache_artifacts.values()}) == 3,
                'Parallel topology cache identity not isolated')
        proof['cache_isolation'] = cache_artifacts
        states = {}
        models = {}
        batches = {}
        first_logits = {}
        for arm in ARMS:
            runtime = resolved[arm]
            train_arm, val_arm, _ = create_datasets(runtime)
            real_batch = collate(train_arm, indices)
            batches[arm] = real_batch
            graph_check = verify_virtual_batch(real_batch, arm, len(indices))
            seed(runtime['experiment']['seed'])
            model = GraphTransformer(build_model_config(runtime)).cpu()
            state = cpu_state(model)
            states[arm] = state
            require(('virtual_node_mode_code' in state) == (arm != 'single'),
                    f'{arm}: wrong checkpoint marker keys')
            if arm != 'single':
                require(int(state['virtual_node_mode_code']) == (3 if arm == 'parallel4' else 4)
                        and int(state['virtual_node_schema_code']) == 1
                        and int(state['virtual_node_group_size_code']) == 5,
                        f'{arm}: wrong checkpoint mode/schema marker')
                model.enable_parallel_global_diagnostics = True
            model.eval()
            state_logits = []
            handle = model.bond_head[3].register_forward_hook(
                lambda module, args, output: state_logits.append(output.detach().clone()))
            with torch.no_grad():
                try:
                    prediction, route = verify_virtual_packing(model, real_batch, arm)
                finally:
                    handle.remove()
            require(prediction.shape == real_batch['label_mask'].shape
                    and torch.isfinite(prediction).all().item() and len(state_logits) == 1,
                    f'{arm}: invalid four-state output')
            if arm != 'single':
                first_logits[arm] = (prediction.detach().clone(), state_logits[0])
            proof.setdefault('padding_isolation', {})[arm] = model_batch_isolation(model, train_arm)
            proof.setdefault('virtual_edge_response', {})[arm] = virtual_edge_response(model, real_batch)
            if arm == 'parallel4_cross':
                require(route['attention_within_peptide_bond_std_mean'] > 0,
                        'Initial per-bond global-slot attention is constant within all peptides')
            models[arm] = model
            proof['variants'][arm] = {'parameter_count': sum(p.numel() for p in model.parameters()),
                                      'graph': graph_check, 'routing': route,
                                      'node_conv_mode': 'single_k5', 'virtual_node_mode': arm}
        shared = set(states['parallel4']) & set(states['parallel4_cross'])
        require(all(torch.equal(states['parallel4'][key], states['parallel4_cross'][key]) for key in shared
                    if key not in {'virtual_node_mode_code'}),
                'Parallel4 and cross changed a shared initialization tensor')
        require(torch.equal(first_logits['parallel4'][0], first_logits['parallel4_cross'][0])
                and torch.equal(first_logits['parallel4'][1], first_logits['parallel4_cross'][1]),
                'Cross initial eval logits differ from parallel4 despite zero residual projection')
        models['parallel4'].train(); models['parallel4_cross'].train()
        seed(_stable_fold_seed(BASE_SEED, FOLDS[0]))
        train_p4 = models['parallel4'](batches['parallel4'])
        train_p4_state = models['parallel4'].last_four_state_logits.detach().clone()
        seed(_stable_fold_seed(BASE_SEED, FOLDS[0]))
        train_cross = models['parallel4_cross'](batches['parallel4_cross'])
        require(torch.equal(train_p4, train_cross)
                and torch.equal(train_p4_state, models['parallel4_cross'].last_four_state_logits),
                'Cross initial train logits differ from parallel4')
        proof['initial_parallel_cross_equal'] = True
        del models
        gc.collect()

        # A previously completed fold supplies fixed internal-validation rows
        # and a frozen checkpoint. Formal evaluate and CSV export share a pass.
        old_root = Path(PREVIOUS_ROOT)
        old_manifest = read_json(old_root / 'manifest.json')
        old_arm = old_manifest['arms'][0]
        old_folder = Path(old_arm['cv_root']) / f'fold_{FOLDS[0]}'
        old_checkpoint = next(old_folder.glob('checkpoints/**/best_model.pt'))
        old_saved = torch.load(old_checkpoint, map_location='cpu', weights_only=False)
        old_runtime = yaml.safe_load((old_folder / 'config.yaml').read_text(encoding='utf-8'))
        old_provenance = read_json(next(old_folder.glob('checkpoints/**/row_validation_provenance.json')))
        old_effective = copy.deepcopy(old_runtime)
        old_effective['model'] = old_saved['config']
        old_model = GraphTransformer(build_model_config(old_effective))
        old_model.load_state_dict(old_saved['model_state_dict'], strict=True)
        validation_csv = Path(old_provenance['split_val_csv'])
        old_val = GraphDataset(str(validation_csv), old_runtime,
                               max_seq_len=old_runtime['data']['max_seq_len'],
                               graph_strategy=old_runtime['data']['graph_strategy'],
                               augmentation=False, split='val')
        old_loader = GraphDataLoader(old_val, batch_size=512, shuffle=False,
                                     num_workers=0, pin_memory=False)
        evaluator = Evaluator(old_model, device=torch.device(f'cuda:{gpu}'),
                              config=old_runtime, allow_target_aware_threshold=False)
        formal = evaluator.evaluate(old_loader, allow_target_aware=False)
        serialized = evaluator.collect_cached_prediction_outputs(expected_rows=len(old_val.data))
        require(serialized['source'] == 'same_pass_evaluate_logits'
                and serialized['probability_precision'] == 'float32_roundtrip_9_significant_digits',
                'Old-checkpoint formal evaluation did not use the same-pass export')
        predicted = old_val.data.copy()
        predicted['evaluation_id'] = build_evaluation_id(str(old_checkpoint), 'test')
        predicted['checkpoint_path'] = str(old_checkpoint.resolve())
        predicted['threshold'] = serialized['threshold']
        for column, key in (('true', 'true_strings'), ('pred', 'pred_strings'),
                            ('pred_prob', 'prob_strings')):
            predicted[column] = serialized[key]
        predicted['prediction_export_source'] = serialized['source']
        predicted['prediction_probability_precision'] = serialized['probability_precision']
        audit_csv = temporary / 'same_pass_internal_validation.pred.csv'
        predicted.to_csv(audit_csv, index=False)
        expected = {key: float(formal[key]) for key in ('f1', 'accuracy', 'auc', 'ex_precision',
                                                         'ex_recall', 'q_brier')}
        expected['q_seq'] = float(formal['q_spearman_pep_seq'])
        expected['max_seq_len'] = old_runtime['data']['max_seq_len']
        recalc = recompute_prediction_metrics(audit_csv, validation_csv, old_checkpoint, expected)
        proof['same_pass_prediction_reconstruction'] = {
            'rows': recalc['rows'], 'valid_bonds': recalc['valid_bonds'],
            'formal_vs_csv_absolute_differences': recalc['absolute_differences'],
            'all_metrics_within_tolerance': True,
            'source': serialized['source'], 'probability_precision': serialized['probability_precision'],
            'old_checkpoint': str(old_checkpoint), 'old_checkpoint_sha256': file_sha(old_checkpoint)}
        del evaluator, old_model, old_loader, old_val
        torch.cuda.empty_cache()

        for arm in ARMS:
            runtime = resolved[arm]
            seed(runtime['experiment']['seed'])
            model = GraphTransformer(build_model_config(runtime)).cuda()
            optimizer = create_optimizer(model, runtime)
            trainer = Trainer(model, optimizer, BinaryBondLoss(runtime['loss']),
                              create_scheduler(optimizer, runtime), torch.device(f'cuda:{gpu}'),
                              runtime, train_script.logger)
            train_arm, val_arm, _ = create_datasets(runtime)
            small = to_cuda(collate(train_arm, indices))
            updates = []
            grad_checks = []
            for step in range(2):
                model.train()
                loss, stats = trainer._forward_pass(small)
                require(torch.isfinite(loss).item() and stats['valid_bond_count'] > 0,
                        f'{arm}: invalid four-state training loss')
                trainer._backward_pass(loss)
                require(math.isfinite(trainer.last_grad_norm), f'{arm}: invalid gradient norm')
                if arm == 'parallel4_cross':
                    selected = ('parallel_global_offsets', 'parallel_bond_query_proj.weight',
                                'parallel_bond_cross_attention.in_proj_weight',
                                'parallel_bond_cross_attention.out_proj.weight',
                                'parallel_bond_context_proj.weight')
                    norms = {name: (float(dict(model.named_parameters())[name].grad.float().norm())
                                    if dict(model.named_parameters())[name].grad is not None else 0.)
                             for name in selected}
                    in_proj_grad = model.parallel_bond_cross_attention.in_proj_weight.grad
                    norms.update({f'parallel_bond_cross_attention.{piece}_grad_norm':
                                  float(part.float().norm()) for piece, part in zip(
                                      ('query', 'key', 'value'), in_proj_grad.chunk(3, dim=0))})
                    require(norms['parallel_bond_context_proj.weight'] > 0
                            and (step == 0 or all(value > 0 for value in norms.values())),
                            'Cross readout lacks context/QKV/slot gradients after projection updates')
                    grad_checks.append(norms)
                updates.append({'loss': float(loss.detach()), 'valid_bonds': stats['valid_bond_count'],
                                'gradient_norm': trainer.last_grad_norm})
                trainer.metrics_calculator.reset()
            torch.cuda.reset_peak_memory_stats()
            full_train = to_cuda(collate(train_arm, longest_indices(train_arm, 128)))
            model.train(); full_loss, _ = trainer._forward_pass(full_train); trainer._backward_pass(full_loss)
            train_memory = memory()
            del full_train
            torch.cuda.empty_cache()
            full_val = to_cuda(collate(val_arm, longest_indices(val_arm, 512)))
            model.eval(); torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                validation = model(full_val)
            require(torch.isfinite(validation).all().item(), f'{arm}: longest validation produced nonfinite output')
            padding_metrics = masked_metric_check(validation, full_val)
            val_memory = memory()
            del full_val
            proof['variants'][arm].update(
                small_batch_updates=updates, cross_gradient_by_step=grad_checks,
                full_training_batch_peak_allocated_mb=train_memory['allocated_bytes'] / 1024**2,
                validation_batch_peak_allocated_mb=val_memory['allocated_bytes'] / 1024**2,
                peak_cuda_allocated_bytes=max(train_memory['allocated_bytes'], val_memory['allocated_bytes']),
                peak_cuda_reserved_bytes=max(train_memory['reserved_bytes'], val_memory['reserved_bytes']),
                masked_metrics=padding_metrics)
            del model, optimizer, trainer, train_arm, val_arm, validation
            gc.collect(); torch.cuda.empty_cache()

        # Checkpoint mode semantic validation runs before copying shared tensors.
        for source_arm, state in states.items():
            own = GraphTransformer(build_model_config(resolved[source_arm])).cpu()
            validate_virtual_node_checkpoint(own, state, resolved[source_arm]['model'])
            own.load_state_dict(state, strict=True)
            same_tensors(state, cpu_state(own))
            for target_arm in ARMS:
                if source_arm == target_arm:
                    continue
                target = GraphTransformer(build_model_config(resolved[target_arm])).cpu()
                before = cpu_state(target)
                try:
                    validate_virtual_node_checkpoint(target, state, resolved[source_arm]['model'])
                except ValueError:
                    pass
                else:
                    raise ValueError(f'Cross-mode checkpoint {source_arm}->{target_arm} accepted')
                same_tensors(before, cpu_state(target))

    proof['parallel_global_checks'] = {
        'legacy_single_k5_state_rng_cpu_forward_backward_exact': True,
        'three_arm_config_only_topology_and_readout': True,
        'synthetic_l0_l1_l4_l5_l6_l11_l32_exact_graph': True,
        'four_slot_edge_cache_isolation': True,
        'old_completed_checkpoint_formal_samepass_csv_reconstruction': True,
        'same_five_historical_train_val_source_shas': True,
        'train_seed45_split_base42_separate': True,
        'four_state_train_eval_initial_cross_residual_equality': True,
        'two_updates_cross_qkv_slots_gradient': True,
        'all_three_longest_train128_val512_finite_memory': True,
        'cross_mode_precopy_checkpoint_rejection': True,
        'real_row_slot_pack_film_and_padding': True,
    }
    require(all(proof['parallel_global_checks'].values()), 'Missing preflight check')
    proof['status'] = 'passed'
    require(output is not None, 'Preflight output path is required')
    dump(output, proof)
    return proof


def smoke():
    prior_completion_barrier()
    authorization_barrier()
    require(not (OPS / 'smoke.json').exists() and not (OPS / 'launch.json').exists(),
            'Preflight/launch exists; refuse duplicate')
    devices, _, _ = idle()
    before = sources()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(devices[0]), OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    with (OPS / 'preflight.log').open('x', encoding='utf-8') as log:
        subprocess.run([PY, __file__, 'preflight', '--gpu', '0', '--output', str(OPS / 'smoke.json')], cwd=REPO, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    proof = read_json(OPS / 'smoke.json')
    validate_proof(proof)
    require(before == sources(), 'Source changed during preflight')
    proof.update(gitrev=revision(), source_sha256=before,
                 source_provenance=source_provenance(before, revision()),
                 target_preparation_sha256=file_sha(TARGETS / 'preparation.json'),
                 target_manifests={f: file_sha(TARGETS / f'fold_{f}/manifest.json') for f in FOLDS})
    dump(OPS / 'smoke.json', proof)
    print(json.dumps({'status': proof['status'], 'gitrev': proof['gitrev'],
                      'checks': proof['parallel_global_checks'], 'variants': proof['variants']}))


def launch():
    claim = OPS / 'launch.claim'
    try:
        with claim.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps({'pid': os.getpid(), 'created_at':
                datetime.datetime.now(datetime.timezone.utc).isoformat()}) + '\n')
    except FileExistsError as error:
        raise ValueError('Launch already claimed; inspect launch.json/terminal.json before any retry') from error
    try:
        require(not (OPS / 'launch.json').exists(), 'Launch exists; refuse duplicate')
        predecessor = prior_completion_barrier()
        authorization_barrier()
        proof = read_json(OPS / 'smoke.json')
        validate_proof(proof)
        require(proof['gitrev'] == revision() and proof['source_sha256'] == sources(),
                'Missing/stale source-bound preflight')
        require(proof['source_provenance'] == source_provenance(proof['source_sha256'], revision()),
                'Source/Git provenance changed since preflight')
        changed = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=REPO, text=True).splitlines()
        require(set(changed) <= {'graph_transform/scripts/ensemble_baseline_preds.py'},
                f'Unexpected source changes: {changed}')
        require(read_json(TARGETS / 'preparation.json')['status'] == 'passed'
                and file_sha(TARGETS / 'preparation.json') == proof['target_preparation_sha256']
                and all(file_sha(TARGETS / f'fold_{f}/manifest.json') == proof['target_manifests'][f] for f in FOLDS),
                'Four-state targets changed or not validated')
        devices, gpu_rows, processes = idle()
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
        root = REPO / 'checkpoints/graph_transform/four_state_parallel_global_batch' / stamp
        info = {
        'schema_version': 1, 'root': str(root), 'gitrev': revision(), 'predecessor': predecessor,
        'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'baseline_ref': BASELINE_REF, 'control_arm': 'single', 'virtual_node_modes': list(ARMS),
        'parallel_global_slots': 4, 'node_conv_mode': 'single_k5',
        'hidden_dim': 256, 'num_attention_heads': 4, 'concat_heads': True,
        'four_state_conditional_weight': 1., 'four_state_union_brier_weight': 0.,
        'training_base_seed': BASE_SEED, 'validation_split_base_seed': SPLIT_BASE_SEED,
        'fold_seeds': {f: {'training': _stable_fold_seed(BASE_SEED, f),
                           'split': _stable_fold_seed(SPLIT_BASE_SEED, f)} for f in FOLDS},
        'gpus': devices, 'gpu_preflight': gpu_rows, 'compute_processes': processes,
        'configs': proof['configs'], 'smoke_sha256': file_sha(OPS / 'smoke.json'),
        'source_sha256': proof['source_sha256'],
        'source_provenance': proof['source_provenance'],
        'target_preparation_sha256': proof['target_preparation_sha256'],
        'target_manifests': proof['target_manifests'],
        'scope': 'Three k5 global-readout arms: single original virtual node; parallel4 four distinct '
                 'slots, each bidirectionally connected to every residue; parallel4_cross identical graph '
                 'plus bond-conditioned four-head attention from valid bond endpoint query into final four slots. '
                 'All virtual edges use type4/distance0; trainable capacity differs and is reported. '
                 '256-d/four-state alpha1/Brier0/late/static/concat/nocharge and 3GCN2GAT4heads. '
                 'Training base45, split base42, folded group_uniform, full-spectrum internal validation, '
                 'padding-free fixed threshold0.5. All three within-round contrasts reported; '
                 'no test-selected epoch/threshold/seed or automatic promotion.',
        }
        dump(OPS / 'launch.json', info)
        try:
            env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
            with (OPS / 'worker.log').open('x', encoding='utf-8') as log:
                process = subprocess.Popen([PY, __file__, 'worker'], cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            dump(OPS / 'terminal.json', {'status': 'failed', 'root': info['root'], 'gitrev': info['gitrev'],
                  'error': traceback.format_exc(), 'retry_policy': 'No automatic relaunch'})
            raise
        info['worker_pid'] = process.pid
        dump(OPS / 'launch.json', info)
        print(json.dumps({key: value for key, value in info.items() if key != 'source_sha256'}))
    except BaseException:
        # A failed read-only precondition owns no launch and may be attempted later.
        # A durable launch keeps its claim, including a terminal launch failure.
        if not (OPS / 'launch.json').exists():
            claim.unlink(missing_ok=True)
        raise


def elapsed_seconds(started, finished):
    value = (datetime.datetime.fromisoformat(finished.replace('Z', '+00:00'))
             - datetime.datetime.fromisoformat(started.replace('Z', '+00:00'))).total_seconds()
    require(value >= 0, 'Negative arm wall-clock duration')
    return value


def prediction_path_for_fold(fold_folder, metric_path):
    """Resolve the unique latest prediction paired with this fold's metric CSV."""
    folder = Path(fold_folder).resolve()
    metric = Path(metric_path).resolve()
    require(metric.name == 'latest_test_metric.csv', 'Unexpected fold metric filename')
    try:
        metric_suffix = metric.relative_to(folder / 'metrics').parent
    except ValueError as error:
        raise ValueError('Metric CSV is outside this fold metric directory') from error
    expected = folder / 'preds' / metric_suffix / 'latest_test.pred.csv'
    matches = list(folder.glob('preds/**/latest_test.pred.csv'))
    require(len(matches) == 1 and matches[0].resolve() == expected and expected.is_file(),
            'Missing or ambiguous fold prediction CSV corresponding to metric path')
    return expected


def diagnostic_batch(runtime, provenance, fold):
    """First 16 already-bound internal-validation rows, never outer-test data."""
    from data.graph_dataset import GraphDataset, GraphDataLoader
    target = OPS / 'diagnostics' / f'fold_{fold}_validation16.csv'
    target.parent.mkdir(exist_ok=True)
    with Path(provenance['split_val_csv']).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        rows, fields = list(itertools.islice(reader, 16)), reader.fieldnames
    require(len(rows) == 16, 'Not enough internal validation rows for bounded diagnostics')
    with target.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    dataset = GraphDataset(str(target), runtime, max_seq_len=runtime['data']['max_seq_len'],
                           graph_strategy=runtime['data']['graph_strategy'], augmentation=False, split='val')
    batch = GraphDataLoader._default_collate_fn(None, [dataset[i] for i in range(16)])
    return batch, {'scope': 'first16_internal_validation_rows; no metric selection',
                   'source_csv': provenance['split_val_csv'], 'source_sha256': file_sha(provenance['split_val_csv']),
                   'subset_csv': str(target), 'subset_sha256': file_sha(target), 'rows': 16}


def verify_virtual_batch(batch, arm, expected_rows=16):
    """Independently check real-row four-slot edges, offsets and bond mapping."""
    import torch
    lengths = batch['seq_lens'].tolist()
    node_lens = batch['node_lens'].tolist()
    edge_index = batch['edge_index'].cpu()
    edge_types = batch['edge_types'].cpu()
    edge_distances = batch['edge_distances'].cpu()
    bond_map = batch['bond_edge_map'].cpu().tolist()
    require(len(lengths) == len(node_lens) == expected_rows, 'Diagnostic row count changed')
    total_nodes = sum(node_lens)
    require(edge_index.ndim == 2 and edge_index.shape[0] == 2
            and (edge_index.numel() == 0 or (int(edge_index.min()) >= 0
                                         and int(edge_index.max()) < total_nodes)),
            f'{arm}: graph edge endpoint outside packed node range')
    node_sample = torch.repeat_interleave(torch.arange(len(node_lens)),
                                          torch.tensor(node_lens, dtype=torch.long))
    require(torch.equal(node_sample[edge_index[0]], node_sample[edge_index[1]]),
            f'{arm}: an edge crosses peptide sample boundaries')
    expected_bonds = []
    offset = 0
    for sample, (length, node_len) in enumerate(zip(lengths, node_lens)):
        slots = 1 if arm == 'single' else 4
        require(node_len == length + slots, f'{arm}/{sample}: wrong virtual node count')
        src, dst = edge_index
        selected = ((src >= offset) & (src < offset + node_len)
                    & (dst >= offset) & (dst < offset + node_len))
        local = (edge_index[:, selected] - offset).t().tolist()
        local_type = edge_types[selected].tolist()
        local_distance = edge_distances[selected].tolist()
        require(len(local) == len(set(map(tuple, local))), f'{arm}/{sample}: duplicate edges')
        actual = {(u, v) for u, v in local if u >= length or v >= length}
        expected = {(u, length + slot) for u in range(length) for slot in range(slots)}
        expected |= {(length + slot, u) for u in range(length) for slot in range(slots)}
        require(actual == expected, f'{arm}/{sample}: parallel-slot edge set mismatch')
        for (u, v), edge_type, distance in zip(local, local_type, local_distance):
            if u >= length or v >= length:
                require(edge_type == 4 and distance == 0,
                        f'{arm}/{sample}: virtual edge type/distance mismatch')
        expected_bonds.extend((offset + i, offset + i + 1)
                              for i in range(max(length - 1, 0)))
        offset += node_len
    require(offset == total_nodes and len(bond_map) == len(expected_bonds),
            f'{arm}: node/bond-map length mismatch')
    actual_bonds = []
    for edge_id in bond_map:
        require(0 <= edge_id < edge_index.shape[1], f'{arm}: bond edge id out of bounds')
        u, v = edge_index[:, edge_id].tolist()
        require(v == u + 1, 'Bond edge map no longer points to an adjacent residue edge')
        actual_bonds.append((u, v))
    require(actual_bonds == expected_bonds,
            f'{arm}: original bond edge map includes a wrong sample or residue pair')
    return {'real_row_virtual_edge_sets': True, 'type4_distance0': True,
            'packed_offsets_and_bond_map': True}


def verify_virtual_packing(model, batch, arm):
    """Independently check k5 residues, four distinct slots and FiLM at GCN1."""
    import torch
    captured = {'conv': [], 'root': [], 'packed': [], 'final': [], 'attention': []}
    hooks = [model.node_context_conv.register_forward_hook(
                 lambda module, args, output: captured['conv'].append(output.detach().clone())),
             model.global_node_proj.register_forward_hook(
                 lambda module, args, output: captured['root'].append(output.detach().clone())),
             model.gcn_layers[0].register_forward_pre_hook(
                 lambda module, args: captured['packed'].append(args[0].detach().clone())),
             model.gat_layers[-1].register_forward_hook(
                 lambda module, args, output: captured['final'].append(output.detach().clone()))]
    if arm == 'parallel4_cross':
        hooks.append(model.parallel_bond_cross_attention.register_forward_hook(
            lambda module, args, output: captured['attention'].append(
                output[1].detach().clone() if output[1] is not None else None)))
    try:
        with torch.inference_mode():
            result = model(batch)
    finally:
        for hook in hooks:
            hook.remove()
    require(all(len(captured[key]) == 1 for key in ('conv', 'root', 'packed', 'final')),
            'Virtual-node routing hooks did not fire once')
    if arm == 'parallel4_cross':
        require(len(captured['attention']) == 1 and captured['attention'][0] is not None,
                'Bond-query attention weights were not returned')
    conv = captured['conv'][0]
    root = model.global_node_embedding.detach() + captured['root'][0]
    lengths = batch['seq_lens'].tolist()
    pieces = []
    for row, length in enumerate(lengths):
        residues = conv[row, :length]
        if arm == 'single':
            nodes = [residues, root[row:row + 1]]
        else:
            slots = torch.cat((root[row:row + 1],
                               root[row:row + 1] + model.parallel_global_offsets.detach()), dim=0)
            require(slots.shape == (4, root.shape[-1]), 'Wrong global-slot shape')
            nodes = [residues, slots]
        pieces.append(torch.cat(nodes, dim=0))
    expected = torch.cat(pieces, dim=0)
    film_input = torch.stack((batch['charges'].float() * .1, batch['nces'].float() * .01), dim=1)
    # The captured node tensors are inference tensors; the independent
    # reconstruction must never attach an autograd graph to them.
    with torch.no_grad():
        gamma, beta = model.film_mlp(film_input).chunk(2, dim=-1)
    sample = torch.repeat_interleave(torch.arange(len(lengths)), batch['node_lens'])
    expected = expected * (1 + gamma[sample]) + beta[sample]
    require(torch.allclose(captured['packed'][0], expected, atol=8e-6, rtol=2e-5),
            f'{arm}: global-slot packing/FiLM differed from independent formula')
    slot_diagnostics = {}
    if arm != 'single':
        offsets = torch.cat((batch['node_lens'].new_zeros(1), batch['node_lens'][:-1])).cumsum(0)
        slots = torch.stack([captured['final'][0][int(offset + length):int(offset + length + 4)]
                             for offset, length in zip(offsets, lengths)]).float()
        require(slots.shape == (len(lengths), 4, model.hidden_dim), 'Final four-slot shape mismatch')
        norms = torch.nn.functional.normalize(slots, dim=-1)
        cosine = norms @ norms.transpose(1, 2)
        pair_i, pair_j = torch.triu_indices(4, 4, offset=1)
        singular = torch.linalg.svdvals(slots)
        weights = singular / singular.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        effective_rank = torch.exp(-(weights * weights.clamp_min(1e-12).log()).sum(dim=-1))
        slot_diagnostics = {
            'final_slot_pair_cosine_mean': float(cosine[:, pair_i, pair_j].mean()),
            'final_slot_effective_rank_mean': float(effective_rank.mean()),
            'final_slot_effective_rank_min': float(effective_rank.min()),
            'effective_rank_definition': 'exp(entropy(normalized four singular values))',
        }
        if arm == 'parallel4_cross':
            attention = captured['attention'][0].float()
            require(attention.ndim == 4 and attention.shape[1:] == (4, 1, 4),
                    'Cross-attention shape must be [valid bonds,4 heads,1 query,4 slots]')
            flat = attention.squeeze(2)
            bond_sample = torch.arange(len(lengths))[:, None].expand(
                -1, batch['label_mask'].size(1))[batch['label_mask'].bool()]
            require(flat.size(0) == bond_sample.numel(), 'Bond/sample attention alignment changed')
            within_peptide = [flat[bond_sample == i].std(dim=0, unbiased=False).mean()
                              for i in range(len(lengths)) if int((bond_sample == i).sum()) > 1]
            slot_diagnostics.update(
                attention_within_peptide_bond_std_mean=(float(torch.stack(within_peptide).mean())
                                                        if within_peptide else 0.),
                attention_valid_bonds=int(flat.size(0)),
                attention_min=float(flat.min()) if flat.numel() else 0.,
                attention_max=float(flat.max()) if flat.numel() else 0.)
    require(all(math.isfinite(value) for value in slot_diagnostics.values()
                if isinstance(value, (float, int))), 'Nonfinite descriptive slot diagnostics')
    return result, {'actual_k5_slot_pack_and_film': True, **slot_diagnostics}


def verify_checkpoint(path, runtime, arm, count, provenance, fold):
    import torch
    from models.graph_transformer import GraphTransformer
    from models.dynamic_edges import validate_edge_state_checkpoint
    from models.bond_fusion_heads import validate_bond_fusion_checkpoint
    from models.node_context_conv import validate_node_conv_checkpoint
    from models.utils import build_model_config, validate_virtual_node_checkpoint
    saved = torch.load(path, map_location='cpu', weights_only=False)
    saved_config = saved['config']
    require(saved_config['hidden_dim'] == 256 and saved_config['bond_head_type'] == 'four_state_mlp'
            and saved_config.get('theory_fusion_stage', 'late') == 'late'
            and saved_config.get('edge_state_mode', 'static') == 'static'
            and saved_config.get('bond_fusion_mode', 'concat') == 'concat'
            and saved_config.get('node_conv_mode') == 'single_k5'
            and saved_config.get('virtual_node_mode', 'single') == arm
            and saved_config['num_attention_heads'] == 4
            and saved_config.get('bond_cross_attention_heads', 4) == 4,
            f'{arm}/{fold}: saved architecture mismatch')
    expected_rng = {
        'training_seed': runtime['experiment']['seed'],
        'validation_split_seed': runtime['training']['validation_split_seed'],
        'hidden_dim': 256, 'bond_head_type': 'four_state_mlp', 'theory_fusion_stage': 'late',
        'four_state_conditional_weight': 1., 'four_state_union_brier_weight': 0.,
    }
    require(saved.get('extra_state', {}).get('training_rng_protocol') == expected_rng,
            f'{arm}/{fold}: actual checkpoint RNG/loss protocol mismatch')
    effective = copy.deepcopy(runtime)
    effective['model'] = saved_config
    require(effective['model']['state_feature_mask'] == [True, True, False]
            and effective['model']['env_feature_mask'] == [True, False],
            f'{arm}/{fold}: saved pre-synthesis feature masks differ')
    with torch.random.fork_rng(devices=[]):
        model = GraphTransformer(build_model_config(effective))
    require(sum(p.numel() for p in model.parameters()) == count, f'{arm}/{fold}: parameter count mismatch')
    validate_edge_state_checkpoint(model, saved['model_state_dict'], saved_config)
    validate_bond_fusion_checkpoint(model, saved['model_state_dict'], saved_config)
    validate_node_conv_checkpoint(model, saved['model_state_dict'], saved_config)
    validate_virtual_node_checkpoint(model, saved['model_state_dict'], saved_config)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    require(all(torch.isfinite(p).all().item() for p in model.parameters()),
            f'{arm}/{fold}: nonfinite saved parameters')
    model.eval()
    model.enable_node_conv_diagnostics = True
    model.enable_parallel_global_diagnostics = arm != 'single'
    batch, sample_proof = diagnostic_batch(effective, provenance, fold)
    graph_proof = verify_virtual_batch(batch, arm)
    output, route_proof = verify_virtual_packing(model, batch, arm)
    require(torch.isfinite(output[batch['label_mask'].bool()]).all().item(),
            f'{arm}/{fold}: nonfinite internal-validation diagnostic output')
    diagnostics = model.last_node_conv_diagnostics
    require(isinstance(diagnostics, dict) and diagnostics.get('mode') == 'single_k5'
            and diagnostics.get('num_valid_residues') == int(batch['seq_lens'].sum()),
            f'{arm}/{fold}: missing node-convolution mode/count diagnostics')
    # Reject NaN/Inf anywhere, including per-branch activation summaries.
    json.dumps(diagnostics, allow_nan=False)
    kernels, channels = [5], [192]
    activations = diagnostics.get('branch_activation_l2_mean', [])
    require(diagnostics.get('kernel_sizes') == kernels and diagnostics.get('branch_channels') == channels
            and len(activations) == len(kernels) and all(value >= 0 for value in activations),
            f'{arm}/{fold}: convolution branch identity/activation diagnostics mismatch')
    require(all(diagnostics.get(key, -1) >= 0 for key in
                ('projection_residual_l2_mean', 'input_l2_mean', 'output_l2_mean', 'residual_relative_l2_mean')),
            f'{arm}/{fold}: node residual diagnostics invalid')
    parallel_diag = model.last_parallel_global_diagnostics
    if arm == 'single':
        require(not parallel_diag, f'{arm}/{fold}: unexpected parallel diagnostics')
    else:
        require(parallel_diag.get('mode') == arm
                and parallel_diag.get('num_valid_bonds') == int(batch['label_mask'].sum())
                and len(parallel_diag.get('slot_l2_mean', [])) == 4
                and all(value >= 0 for value in parallel_diag['slot_l2_mean'])
                and -1 <= parallel_diag.get('slot_pair_cosine_mean', -2) <= 1,
                f'{arm}/{fold}: missing/invalid four-slot diagnostics')
        if arm == 'parallel4_cross':
            weights = parallel_diag.get('attention_mean_per_head_slot', [])
            require(len(weights) == 4 and all(len(row) == 4 for row in weights)
                    and all(abs(sum(row) - 1.) < 1e-5 for row in weights)
                    and parallel_diag.get('attention_entropy_mean', -1) >= 0
                    and parallel_diag.get('context_residual_l2_mean', -1) >= 0,
                    f'{arm}/{fold}: invalid per-bond four-slot attention')
        json.dumps(parallel_diag, allow_nan=False)
    return {'strict_cpu_restore': True, 'finite_parameters': True, 'parameter_count': count,
            'node_conv_mode': 'single_k5', 'virtual_node_mode': arm,
            'parallel_global_slots': 1 if arm == 'single' else 4,
            'bond_fusion_mode': 'concat', 'edge_state_mode': 'static',
            'checkpoint_training_rng_protocol': expected_rng,
            'diagnostic_sample': sample_proof, 'graph_proof': graph_proof,
            'route_proof': route_proof, 'node_conv_diagnostics': diagnostics,
            'parallel_global_diagnostics': parallel_diag}


def paired_stats(candidate, reference):
    deltas = [candidate[f] - reference[f] for f in FOLDS]
    return {'paired_delta_mean': statistics.mean(deltas), 'paired_delta_std_ddof1': statistics.stdev(deltas),
            'paired_deltas': dict(zip(FOLDS, deltas))}


def recompute_prediction_metrics(pred_path, test_path, checkpoint, expected):
    """Rebuild padding masks and all study metrics from saved per-spectrum test rows."""
    import numpy as np
    import pandas as pd
    from data.label_validation import validate_label_frame
    from training.metrics import BinaryBondMetrics, batch_group_keys
    from train_graph_model import sanitize_filename_component

    source = pd.read_csv(test_path)
    source = validate_label_frame(source, on_error='drop', source_name=test_path.name)
    source = source[source['seq'].str.len() <= expected['max_seq_len']].reset_index(drop=True)
    with pred_path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == len(source) and rows, 'Prediction rows differ from validated test source')
    require(all(row.get('prediction_export_source') == 'same_pass_evaluate_logits'
                and row.get('prediction_probability_precision') ==
                'float32_roundtrip_9_significant_digits' for row in rows),
            'Prediction CSV was not exported from the formal evaluation logits in the same pass')
    bond_lengths = [max(len(str(seq)) - 1, 0) for seq in source['seq']]
    width = max(bond_lengths)
    probabilities = np.zeros((len(rows), width), dtype=np.float32)
    hard_probabilities = np.zeros((len(rows), width), dtype=np.float32)
    targets = np.zeros((len(rows), width), dtype=np.int32)
    soft = np.zeros((len(rows), width), dtype=np.float32)
    valid = np.zeros((len(rows), width), dtype=bool)
    seqs, charges, nces = [], [], []
    rounded_threshold_ambiguities = 0
    checkpoint_parent = sanitize_filename_component(checkpoint.parent.name)
    checkpoint_stem = sanitize_filename_component(checkpoint.stem)
    evaluation_pattern = re.compile(r'^\d{8}_\d{6}_'
                                    + re.escape(checkpoint_parent + '_' + checkpoint_stem + '_test') + '$')
    evaluation_ids = {row.get('evaluation_id', '') for row in rows}
    require(len(evaluation_ids) == 1 and evaluation_pattern.fullmatch(next(iter(evaluation_ids))),
            'Prediction rows have mixed/invalid evaluation ids for the selected checkpoint')
    expected_eval_id = next(iter(evaluation_ids))

    def vector(value, cast=float):
        content = str(value).strip()
        return [] if not content or content.lower() == 'nan' else [cast(part) for part in content.split(';')]

    for index, (row, (_, original)) in enumerate(zip(rows, source.iterrows())):
        seq = str(original['seq'])
        length = bond_lengths[index]
        require(row['seq'] == seq, f'Prediction/source row {index}: sequence or order mismatch')
        require(abs(float(row['charge']) - float(original['charge'])) < 1e-8
                and abs(float(row['nce']) - float(original['nce'])) < 1e-8,
                f'Prediction/source row {index}: charge/NCE mismatch')
        require(Path(row['checkpoint_path']).resolve() == checkpoint.resolve()
                and row['evaluation_id'] == expected_eval_id
                and float(row['threshold']) == .5,
                f'Prediction/source row {index}: checkpoint, evaluation id, or threshold mismatch')
        y, p, q = vector(row['true'], int), vector(row['pred_prob']), vector(row['soft_multi'])
        serialized_probabilities = (str(row['pred_prob']).split(';')
                                    if str(row['pred_prob']).strip() else [])
        require(len(serialized_probabilities) == len(p)
                and all(format(float(np.float32(float(token))), '.9g') == token
                        for token in serialized_probabilities),
                f'Prediction/source row {index}: probability is not canonical float32 roundtrip text')
        hard, original_y, original_q = (vector(row['pred'], int),
                                        vector(original['true_multi'], int),
                                        vector(original['soft_multi']))
        require(all(len(part) == length for part in (y, p, q, hard, original_y, original_q)),
                f'Prediction/source row {index}: serialized padding or missing bond')
        require(y == original_y and y == vector(row['true_multi'], int),
                f'Prediction/source row {index}: hard target mismatch')
        require(np.allclose(q, original_q, rtol=0, atol=1e-7),
                f'Prediction/source row {index}: soft target mismatch')
        require(all(math.isfinite(value) and 0 <= value <= 1 for value in (*p, *q)),
                f'Prediction/source row {index}: nonfinite/invalid probability or q target')
        for bond_index, (pred, prob) in enumerate(zip(hard, p)):
            if pred != int(prob > .5):
                require(abs(prob - .5) <= 1e-6,
                        f'Prediction/source row {index}: serialized class/probability disagreement')
                rounded_threshold_ambiguities += 1
                hard_probabilities[index, bond_index] = .5000011 if pred else .4999989
            else:
                hard_probabilities[index, bond_index] = prob
        probabilities[index, :length] = p
        targets[index, :length] = y
        soft[index, :length] = q
        valid[index, :length] = True
        seqs.append(seq)
        charges.append(float(row['charge']))
        nces.append(float(row['nce']))
    require(rounded_threshold_ambiguities == 0,
            'Nine-significant-digit same-pass export changed a threshold decision')
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    logits = np.log(clipped / (1 - clipped)).astype(np.float32)
    metrics = BinaryBondMetrics({'threshold': .5, 'threshold_strategy': 'fixed'},
                                allow_target_aware_threshold=False)
    metrics.update(logits, targets, label_mask=valid, soft_targets=soft, sequences=seqs,
                   group_keys=batch_group_keys({'sequences': seqs, 'charges': charges, 'nces': nces}))
    computed = metrics.compute()
    hard_clipped = np.clip(hard_probabilities, 1e-7, 1 - 1e-7)
    hard_logits = np.log(hard_clipped / (1 - hard_clipped)).astype(np.float32)
    hard_metrics = BinaryBondMetrics({'threshold': .5, 'threshold_strategy': 'fixed'},
                                     allow_target_aware_threshold=False)
    hard_metrics.update(hard_logits, targets, label_mask=valid, soft_targets=soft,
                        sequences=seqs, group_keys=batch_group_keys({
                            'sequences': seqs, 'charges': charges, 'nces': nces}))
    hard_computed = hard_metrics.compute()
    name_map = {'q_seq': 'q_spearman_pep_seq'}
    tolerances = {'f1': 2e-5, 'accuracy': 2e-5, 'auc': 2e-4, 'ex_precision': 2e-5,
                  'ex_recall': 2e-5, 'q_brier': 2e-5, 'q_seq': 5e-4}
    differences = {}
    for name, tolerance in tolerances.items():
        output = hard_computed if name in {'f1', 'accuracy', 'ex_precision', 'ex_recall'} else computed
        observed = float(output[name_map.get(name, name)])
        differences[name] = abs(observed - expected[name])
        require(math.isfinite(observed) and differences[name] <= tolerance,
                f'Prediction-row recomputation disagrees for {name}: {observed} vs {expected[name]}')
    return {'test_csv': str(test_path), 'test_sha256': file_sha(test_path),
            'prediction_csv': str(pred_path), 'prediction_sha256': file_sha(pred_path),
            'checkpoint_path': str(checkpoint), 'evaluation_id': expected_eval_id,
            'rows': len(rows), 'valid_bonds': int(valid.sum()), 'computed':
            {name: float((hard_computed if name in {'f1', 'accuracy', 'ex_precision', 'ex_recall'}
                          else computed)[name_map.get(name, name)]) for name in tolerances},
            'absolute_differences': differences, 'tolerances': tolerances,
            'rounded_threshold_ambiguities': rounded_threshold_ambiguities,
            'padding_reconstruction': 'seq length - 1 per spectrum; all other positions masked'}


def gates(values):
    checks = {'delta_f1_at_least_0.002': values['f1']['paired_delta_mean'] >= .002,
              'f1_win_at_least_4_of_5': values['f1']['win_splits'] >= 4,
              'delta_auc_nonnegative': values['auc']['paired_delta_mean'] >= 0,
              'delta_q_brier_nonpositive': values['q_brier']['paired_delta_mean'] <= 0}
    return {'gates': checks, 'improvement_gate': all(checks.values())}


def analyze(info):
    root = Path(info['root'])
    status, manifest, summary = (read_json(root / path) for path in
                                 ('status.json', 'manifest.json', 'summary/summary.json'))
    proof = read_json(OPS / 'smoke.json')
    require(status['status'] == 'completed' and len(status['arms']) == len(ARMS)
            and all(a['status'] == 'completed' and a['returncode'] == 0 for a in status['arms']),
            'Three-arm batch did not complete successfully')
    require(status['root'] == manifest['root'] == info['root']
            and status['gitrev'] == manifest['gitrev'] == info['gitrev']
            and len(manifest['arms']) == len(ARMS) and manifest['folds'] == list(FOLDS),
            'Status/manifest/root/source/split mismatch')
    require(prior_completion_barrier() == info['predecessor'], 'Predecessor receipt changed')
    require(file_sha(OPS / 'smoke.json') == info['smoke_sha256'] and sources() == info['source_sha256'],
            'Preflight or source changed during training')
    require(info['source_provenance'] == source_provenance(info['source_sha256'], info['gitrev']),
            'Training source/Git provenance changed')
    require(all(file_sha(config(a)) == info['configs'][a] for a in ARMS), 'Configs changed during training')
    require(file_sha(TARGETS / 'preparation.json') == info['target_preparation_sha256']
            and all(file_sha(TARGETS / f'fold_{f}/manifest.json') == info['target_manifests'][f] for f in FOLDS),
            'Target preparation/manifests changed')
    names = [a['name'] for a in manifest['arms']]
    require(names == [f'{i:02d}_{config(a).stem}' for i, a in enumerate(ARMS, 1)]
            and manifest['control'] == names[0]
            and [a['name'] for a in status['arms']] == names
            and [a['arm'] for a in summary['sources']] == names, 'Arm identity/order mismatch')
    data, binding, recovery, resources, test_sources = {}, {}, {}, {}, {}
    for arm, arm_status, label in zip(manifest['arms'], status['arms'], ARMS):
        require(file_sha(arm['config_snapshot']) == arm['config_sha256'] == info['configs'][label],
                f'{label}: config snapshot mismatch')
        bound = next(item for item in summary['sources'] if item['arm'] == arm['name'])
        require(file_sha(arm['metrics_csv']) == bound['sha256'], f'{label}: metrics source mismatch')
        binding[arm['metrics_csv']] = file_sha(arm['metrics_csv'])
        with Path(arm['metrics_csv']).open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == 5 and {r['fold_id'] for r in rows} == set(FOLDS), f'{label}: missing/duplicate split')
        data[label], recovery[label] = {}, {}
        variant = proof['variants'][label]
        count = variant['parameter_count']
        resources[label] = {**variant, 'hidden_dim': 256, 'head_dim': 64, 'gpus': info['gpus'],
                            'synthetic_virtual_edges_L32': proof['synthetic_graph'][f'{label}/L32']['virtual_edges'],
                            'synthetic_total_nodes_L32': proof['synthetic_graph'][f'{label}/L32']['nodes'],
                            'started_at': arm_status['started_at'], 'finished_at': arm_status['finished_at'],
                            'wall_time_seconds': elapsed_seconds(arm_status['started_at'], arm_status['finished_at'])}
        for row in rows:
            fold = row['fold_id']
            raw_path = Path(row['metric_csv_path'])
            with raw_path.open(encoding='utf-8-sig', newline='') as stream:
                raw = {r['metric']: float(r['value']) for r in csv.DictReader(stream)}
            binding[str(raw_path)] = file_sha(raw_path)
            values = {k: float(row[k]) for k in ('f1', 'accuracy', 'auc', 'q_brier', 'ex_precision', 'ex_recall')}
            values['q_seq'] = float(row['q_spearman_pep_seq'])
            for metric, value in values.items():
                require(abs(value - raw['q_spearman_pep_seq' if metric == 'q_seq' else metric]) < 1e-12,
                        f'{label}/{fold}: raw {metric} mismatch')
            precision, recall = values['ex_precision'], values['ex_recall']
            values['ex_f1_harmonic'] = 2 * precision * recall / (precision + recall) if precision + recall else 0.
            require(all(math.isfinite(v) and (-1 if k == 'q_seq' else 0) <= v <= 1 for k, v in values.items()),
                    f'{label}/{fold}: invalid metric')
            data[label][fold] = values
            folder = Path(arm['cv_root']) / f'fold_{fold}'
            runtime_path = folder / 'config.yaml'
            runtime = yaml.safe_load(runtime_path.read_text(encoding='utf-8'))
            validate_runtime(runtime, label)
            require(runtime['experiment']['seed'] == info['fold_seeds'][fold]['training']
                    and runtime['training']['validation_split_seed'] == info['fold_seeds'][fold]['split'],
                    f'{label}/{fold}: train/split RNG mismatch')
            binding[str(runtime_path)] = file_sha(runtime_path)
            paths = list(folder.glob('checkpoints/**/row_validation_provenance.json'))
            require(len(paths) == 1, f'{label}/{fold}: ambiguous validation provenance')
            provenance = read_json(paths[0])
            require(provenance['mode'] == 'full_original_training_spectra_by_internal_val_sequence'
                    and provenance['outer_test_used_for_internal_validation'] is False
                    and provenance['split_seed'] == info['fold_seeds'][fold]['split'],
                    f'{label}/{fold}: incorrect internal validation source')
            require(file_sha(provenance['source_path']) == provenance['source_sha256'] == proof['folds'][fold]['source_sha256']
                    and file_sha(provenance['split_val_csv']) == proof['folds'][fold]['val_sha256']
                    and provenance['val_rows'] == proof['folds'][fold]['val_rows']
                    and file_sha(provenance['split_train_csv']) == proof['folds'][fold]['train_sha256'],
                    f'{label}/{fold}: training/validation pairing SHA mismatch')
            state_path = paths[0].with_name('four_state_provenance.json')
            require(read_json(state_path)['manifest_sha256'] == info['target_manifests'][fold],
                    f'{label}/{fold}: four-state target binding mismatch')
            for path in (paths[0], state_path, provenance['split_train_csv'], provenance['split_val_csv']):
                binding[str(path)] = file_sha(path)
            checkpoints = list(folder.glob('checkpoints/**/best_model.pt'))
            require(len(checkpoints) == 1 and checkpoints[0].parent == paths[0].parent,
                    f'{label}/{fold}: ambiguous validation-selected checkpoint')
            checkpoint = checkpoints[0]
            checkpoint_proof = verify_checkpoint(checkpoint, runtime, label, count, provenance, fold)
            pred_path = prediction_path_for_fold(folder, raw_path)
            test_path = Path(runtime['data']['test_csv_path'])
            require(pred_path.is_file() and test_path.is_file(),
                    f'{label}/{fold}: missing per-spectrum prediction or test source')
            prediction_proof = recompute_prediction_metrics(
                pred_path, test_path, checkpoint, {**values,
                    'max_seq_len': runtime['data']['max_seq_len']})
            require(fold not in test_sources or test_sources[fold] == prediction_proof['test_sha256'],
                    f'{label}/{fold}: paired arms used different outer-test source rows')
            test_sources[fold] = prediction_proof['test_sha256']
            binding[str(pred_path)] = prediction_proof['prediction_sha256']
            binding[str(test_path)] = prediction_proof['test_sha256']
            recovery[label][fold] = {
                'gitrev': info['gitrev'], 'config': str(runtime_path), 'config_sha256': binding[str(runtime_path)],
                'checkpoint': str(checkpoint), 'checkpoint_sha256': file_sha(checkpoint),
                'metrics_csv': str(raw_path), 'metrics_sha256': binding[str(raw_path)],
                'prediction_recomputation': prediction_proof,
                'selection': 'internal full-spectrum validation max_f1; outer test not used',
                'checkpoint_verification': checkpoint_proof,
            }
    expected = {(name, metric) for name in names for metric in METRICS}
    require(len(summary['metrics']) == len(expected)
            and {(entry['arm'], entry['metric']) for entry in summary['metrics']} == expected,
            'Missing/duplicate summary metric')
    metrics = {label: {} for label in ARMS}
    for entry in summary['metrics']:
        label, metric = ARMS[names.index(entry['arm'])], entry['metric']
        values = [data[label][f][metric] for f in FOLDS]
        deltas = [data[label][f][metric] - data['single'][f][metric] for f in FOLDS]
        direction = -1 if metric == 'q_brier' else 1
        for key, value in [('mean', statistics.mean(values)), ('std_ddof1', statistics.stdev(values)),
                           ('paired_delta_mean', statistics.mean(deltas)),
                           ('paired_delta_std_ddof1', statistics.stdev(deltas))]:
            require(abs(value - entry[key]) < 1e-12, f'{label}/{metric}: summary {key} mismatch')
        require(sum(direction * d > 1e-12 for d in deltas) == entry['win_splits'],
                f'{label}/{metric}: summary win count mismatch')
        metrics[label][metric] = entry
    contrasts = {}
    for name, (candidate, reference) in CONTRASTS.items():
        contrasts[name] = {}
        for metric in METRICS:
            value = paired_stats({f: data[candidate][f][metric] for f in FOLDS},
                                 {f: data[reference][f][metric] for f in FOLDS})
            direction = -1 if metric == 'q_brier' else 1
            deltas = value['paired_deltas'].values()
            value.update(candidate=candidate, reference=reference,
                         win_splits=sum(direction * d > 1e-12 for d in deltas),
                         tie_splits=sum(abs(d) <= 1e-12 for d in deltas),
                         loss_splits=sum(direction * d < -1e-12 for d in deltas))
            contrasts[name][metric] = value
    decisions = {name: gates(values) for name, values in contrasts.items()}
    out = {'status': 'completed', 'root': info['root'], 'gitrev': info['gitrev'],
           'baseline_ref': BASELINE_REF, 'finished_at': status['finished_at'],
           'virtual_node_modes': list(ARMS), 'parallel_global_slots': 4,
           'node_conv_mode': 'single_k5', 'control_arm': 'single', 'hidden_dim': 256,
           'training_base_seed': BASE_SEED, 'validation_split_base_seed': SPLIT_BASE_SEED,
           'fold_seeds': info['fold_seeds'], 'predecessor': info['predecessor'],
           'metrics': metrics, 'contrasts': contrasts, 'decisions': decisions, 'folds': data,
           'test_source_sha256_by_fold': test_sources,
           'resources': resources, 'recovery': recovery, 'paired_validation_verified': True,
           'scope': 'Five overlapping sequence-level 80/20 splits, one initialization per split/arm. '
                    'All candidates and three contrasts reported. Engineering gates are descriptive, '
                    'not significance tests; no test-selected epoch/threshold/seed or automatic promotion.'}
    dump(OPS / 'result.json', out)
    dump(OPS / 'checkpoint_registry.json', {'baseline_ref': BASELINE_REF, 'gitrev': info['gitrev'],
         'root': info['root'], 'candidate_status': 'All three arms retained; no automatic replacement', 'arms': recovery})
    write_report(out)
    write_visuals(out)
    dump(OPS / 'review_receipt.json', {
        'status': 'passed', 'root': info['root'], 'gitrev': info['gitrev'],
        'checks': ['15 runs completed; source/config/target/source-row hashes bound',
                   'Same k5/256/four-state alpha1/Brier0/late/static/concat/nocharge/masks/RNG protocol verified',
                   'Parameter and virtual-edge counts differ by arm and are reported',
                   '15 validation-selected checkpoints strictly restored on CPU and parallel-slot/readout markers/RNG verified',
                   '15 same-pass per-spectrum prediction CSVs independently rebuilt with explicit valid-bond masks and checked against raw metrics',
                   '24 arm-metric means/SD and three paired contrasts recomputed from raw CSV',
                   'Every checkpoint has real-row graph, parallel-slot packing, k5, and bounded attention diagnostics on first16 internal validation rows'],
        'hashes': binding, 'decisions': decisions,
        'result_sha256': file_sha(OPS / 'result.json'), 'report_sha256': file_sha(OPS / 'report.md'),
        'registry_sha256': file_sha(OPS / 'checkpoint_registry.json'),
        'dashboard_sha256': file_sha(OPS / 'parallel_global_dashboard.html'),
        'svg_sha256': file_sha(OPS / 'parallel_global_deltas.svg'),
        'means_svg_sha256': file_sha(OPS / 'parallel_global_means.svg'),
        'folds_svg_sha256': file_sha(OPS / 'parallel_global_fold_deltas.svg'),
        'structure_svg_sha256': file_sha(OPS / 'parallel_global_structure.svg'),
    })


def write_report(result):
    metrics, contrasts, decisions = result['metrics'], result['contrasts'], result['decisions']
    lines = ['# k=5 四状态模型：并行全局节点及逐肽键读取同轮对照', '',
             f"源码：`{result['gitrev']}`；完成：{result['finished_at']}",
             f'原成功参考 `{BASELINE_REF}` 保留。三臂均从头训练，同轮single为基准。', '',
             '| 去padding外部测试指标 | single 均值 ± SD | parallel4 均值 ± SD | parallel4_cross 均值 ± SD |',
             '|---|---:|---:|---:|']
    for metric in METRICS:
        scale = metric_scale(metric)
        cells = [f"{metrics[a][metric]['mean'] * scale:.6f} ± {metrics[a][metric]['std_ddof1'] * scale:.6f}" for a in ARMS]
        lines.append('| ' + metric + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'F1、accuracy、谱图P/R及其调和F1为百分比，差值为百分点；AUC、q-Brier、q_seq为原始单位。SD是五组描述性样本标准差，不是置信区间。', '',
              '## 接入位置与固定基座', '',
              '三个臂都使用完整NodeEncoder→单尺度k=5 Conv1D残差→图消息传递→晚接入15维理论特征→原concat四状态头。', '',
              '| 臂 | 图与肽键读取 |', '|---|---|',
              '| single | 原1个全局节点，与全部残基双向连接；原concat四状态头 |',
              '| parallel4 | 同一肽链4个并行槽，每个槽与全部残基双向连接；原concat四状态头 |',
              '| parallel4_cross | 与parallel4完全相同的4槽图；相邻残基构成Query，4槽作Key/Value，四头交叉注意力残差后进入原四状态头 |', '',
              'parallel4与cross均以原全局向量为slot0，另3槽是共享条件投影加不同可学习偏置。4槽之间没有互连边；新增虚拟边沿用type4、距离0。cross的读取投影初始为零，所以初始预测与parallel4一致。slot与读取模块增加参数，不能把差异只归于信息容量。',
              '配置中的virtual_group_size=5仅保留旧schema字段，对这三个模式的拓扑无效；这里唯一的5是Conv1D核长k=5，不表示五个虚拟层或分组。',
              '三臂固定H256、4头、3GCN+2GAT、原concat、晚理论、静态边、nocharge、四状态α=1/Brier=0。',
              '四状态顺序none/b_only/y_only/both，q=1−P(none)。训练base45、内部划分base42，group_uniform折叠训练、完整原谱内部验证strict max-F1选epoch、固定0.5阈值与去padding外测。正式评估指标和逐谱图CSV使用同次前向logits。', '',
              '## 归因边界', '',
              'parallel4−single检验扩大全局槽数量，同时也改变节点数、边数和参数量。parallel4_cross−parallel4检验在同一4槽图上加入肽键条件读取，但也增加Query/Key/Value和投影参数；没有等参数共享Q对照，因此不能单独归因逐键选择。parallel4_cross−single是两类改动合并的总效应。槽余弦、有效秩及注意力只是描述量，不是物理接触或因果重要性。',
              '五组序列80/20划分重叠、各一次初始化，且存在多轮探索；工程门槛不等于统计显著。不按外测选epoch/阈值/配置/seed，不修改冻结论文表。', '']
    for name, (candidate, reference) in CONTRASTS.items():
        lines += [f'## {candidate} − {reference}', '',
                  '| 指标 | 配对差均值 ± SD | 改善/持平/下降划分 |', '|---|---:|---:|']
        for metric in METRICS:
            scale, value = metric_scale(metric), contrasts[name][metric]
            lines.append(f"| {metric} | {value['paired_delta_mean'] * scale:+.6f} ± "
                         f"{value['paired_delta_std_ddof1'] * scale:.6f} | "
                         f"{value['win_splits']}/{value['tie_splits']}/{value['loss_splits']} |")
        lines += ['', '预定门槛：' + '；'.join(f"{key}={'通过' if passed else '未通过'}"
                    for key, passed in decisions[name]['gates'].items()), '',
                  '| 划分 | ' + ' | '.join('Δ ' + metric + (' pp' if scale == 100 else '') for metric, _, scale in REPORT_METRICS) + ' |',
                  '|---|' + '---:|' * len(REPORT_METRICS)]
        for fold in FOLDS:
            values = [f"{contrasts[name][metric]['paired_deltas'][fold] * scale:+.6f}" for metric, _, scale in REPORT_METRICS]
            lines.append('| ' + fold + ' | ' + ' | '.join(values) + ' |')
        lines.append('')
    lines += ['## 参数、边与资源', '', '| 臂 | 总参数 | 相对single参数差 | L=32虚拟边 | L=32节点 | 五划分墙钟小时 | 预检train128/val512峰值显存MiB |', '|---|---:|---:|---:|---:|---:|---:|']
    control_count = result['resources']['single']['parameter_count']
    for arm in ARMS:
        value = result['resources'][arm]
        peak = max(value['full_training_batch_peak_allocated_mb'], value['validation_batch_peak_allocated_mb'])
        lines.append(f"| {arm} | {value['parameter_count']:,} | {value['parameter_count'] - control_count:+,} | {value['synthetic_virtual_edges_L32']} | {value['synthetic_total_nodes_L32']} | {value['wall_time_seconds'] / 3600:.4f} | {peak:.1f} |")
    lines += ['', 'L=32是固定长度示例，用于直观比较虚拟边数；训练样本长度各异。墙钟时间受GPU、调度和早停影响。显存是同规格预检批量峰值，不是正式运行每步显存，也不是等步数吞吐量。', '',
              '## 内部实现诊断（固定内部验证前16行）', '',
              '每份权重strict CPU恢复，在所属划分完整原谱内部验证前16行、eval模式前向；独立核验真实边集合、四槽偏置与FiLM路由。4槽余弦/有效秩和逐肽键注意力跨键波动为描述性检查，不据此选择模型。', '',
              '| 臂 | 划分 | 卷积诊断 | 槽与读取诊断 |', '|---|---|---|---|']
    for arm in ARMS:
        for fold in FOLDS:
            checked = result['recovery'][arm][fold]['checkpoint_verification']
            conv = json.dumps(checked['node_conv_diagnostics'], ensure_ascii=False, sort_keys=True)
            slot = json.dumps({'parallel': checked['parallel_global_diagnostics'],
                               'independent': checked['route_proof']}, ensure_ascii=False, sort_keys=True)
            lines.append(f'| {arm} | {fold} | `{conv}` | `{slot}` |')
    lines += ['', '## 恢复记录', '', f"批次：`{result['root']}`", '',
              '| 臂 | 划分 | best_model.pt | SHA256 |', '|---|---|---|---|']
    for arm in ARMS:
        for fold in FOLDS:
            value = result['recovery'][arm][fold]
            lines.append(f"| {arm} | {fold} | `{value['checkpoint']}` | `{value['checkpoint_sha256']}` |")
    (OPS / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def metric_scale(metric):
    return 100 if metric in {'f1', 'accuracy', 'ex_precision', 'ex_recall', 'ex_f1_harmonic'} else 1


def svg_start(title, height=1095):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="1160" height="{height}" viewBox="0 0 1160 {height}">',
            f'<rect width="1160" height="{height}" fill="#f8fafc"/>',
            '<style>text{font-family:Arial,sans-serif;fill:#0f172a}</style>',
            f'<text x="30" y="35" font-size="21">{html.escape(title)}</text>']


def save_svg(name, parts):
    (OPS / name).write_text(''.join([*parts, '</svg>']), encoding='utf-8')


def write_structure():
    parts = svg_start('Four parallel global slots and bond-conditioned readout over the k5 four-state backbone', 670)
    boxes = [(65, 'Shared: NodeEncoder [B,L,256] -> residual Conv1D k5 -> [B,L,256]', '#dbeafe'),
             (140, 'single: each residue <-> one original global node; original concat readout', '#e2e8f0'),
             (215, 'parallel4: each residue <-> four distinct global slots; original concat readout', '#dbeafe'),
             (290, 'parallel4_cross: same four-slot graph; bond endpoints query four slots via four-head attention', '#dcfce7'),
             (365, 'Slot 0 = original global embedding + condition; slots 1-3 add learned offsets; type4/dist0', '#e0e7ff'),
             (440, 'Packed graph + FiLM -> 3 GCN (256->256) -> 2 GAT (4x64->256)', '#f3e8ff'),
             (515, 'Late theory15->256 + original edge/endpoints -> concat1536 ->256 -> four-state outputs', '#fef3c7')]
    for yy, label, color in boxes:
        parts += [f'<rect x="30" y="{yy}" width="1100" height="55" rx="9" fill="{color}"/>',
                  f'<text x="46" y="{yy+33}" font-size="14">{html.escape(label)}</text>']
    parts += ['<text x="30" y="606" font-size="14">The four slots are parallel and have no slot-slot edges; k5 refers only to the local Conv1D kernel.</text>',
              '<text x="30" y="635" font-size="14">Cross adds a zero-initialized residual context to the original bond head; parameters and edges are reported.</text>']
    save_svg('parallel_global_structure.svg', parts)


def write_visuals(result):
    """All plots and numerical aggregations are produced on pod only."""
    palette = {'single': '#64748b', 'parallel4': '#2563eb', 'parallel4_cross': '#16a34a'}
    contrast_colors = dict(zip(CONTRASTS, ('#2563eb', '#16a34a', '#7c3aed')))
    contrast_labels = dict(zip(CONTRASTS, ('parallel4 - single', 'cross - single', 'cross - parallel4')))
    means = svg_start('Parallel globals: five-split mean +/- sample SD (not confidence intervals)')
    deltas = svg_start('Parallel globals: three paired contrasts, mean +/- paired SD')
    folds = svg_start('Parallel globals: every paired split difference')
    for index, (metric, _, scale) in enumerate(REPORT_METRICS):
        xx, yy = 30 + (index % 2) * 565, 70 + (index // 2) * 245
        for parts in (means, deltas, folds):
            parts += [f'<rect x="{xx}" y="{yy}" width="535" height="230" rx="10" fill="white"/>',
                      f'<text x="{xx+14}" y="{yy+23}" font-size="15">{html.escape(metric)} ({"lower better" if metric == "q_brier" else "higher better"})</text>']
        entries = [result['metrics'][arm][metric] for arm in ARMS]
        low = min((entry['mean'] - entry['std_ddof1']) * scale for entry in entries)
        high = max((entry['mean'] + entry['std_ddof1']) * scale for entry in entries)
        span = high - low or 1
        low, high = low - span * .08, high + span * .08
        def mean_x(value):
            return xx + 218 + (value - low) / (high - low) * 290
        for row, arm in enumerate(ARMS):
            value = result['metrics'][arm][metric]
            avg, sd = value['mean'] * scale, value['std_ddof1'] * scale
            cy = yy + 58 + row * 52
            center, left, right = mean_x(avg), mean_x(avg-sd), mean_x(avg+sd)
            means += [f'<text x="{xx+14}" y="{cy+4}" font-size="12">{arm}</text>',
                      f'<text x="{xx+14}" y="{cy+19}" font-size="11">{avg:.5f} +/- {sd:.5f}</text>',
                      f'<line x1="{left:.2f}" x2="{right:.2f}" y1="{cy}" y2="{cy}" stroke="{palette[arm]}" stroke-width="2"/>',
                      f'<circle cx="{center:.2f}" cy="{cy}" r="4" fill="{palette[arm]}"/>']
        means.append(f'<text x="{xx+218}" y="{yy+219}" font-size="10">Zoomed axis: {low:.4f} to {high:.4f}; percent metrics in %</text>')
        bound = max((abs(result['contrasts'][name][metric]['paired_delta_mean'])
                     + result['contrasts'][name][metric]['paired_delta_std_ddof1']) * scale for name in CONTRASTS) or 1
        zero_x = xx + 375
        deltas.append(f'<line x1="{zero_x}" x2="{zero_x}" y1="{yy+37}" y2="{yy+202}" stroke="#94a3b8"/>')
        for row, name in enumerate(CONTRASTS):
            value = result['contrasts'][name][metric]
            avg, sd = value['paired_delta_mean'] * scale, value['paired_delta_std_ddof1'] * scale
            cy = yy + 66 + row * 53
            center, left, right = [zero_x + v / bound * 130 for v in (avg, avg-sd, avg+sd)]
            deltas += [f'<text x="{xx+14}" y="{cy}" font-size="12">{contrast_labels[name]}: {avg:+.5f}</text>',
                       f'<text x="{xx+14}" y="{cy+16}" font-size="11">SD {sd:.5f}; improved {value["win_splits"]}/5</text>',
                       f'<line x1="{left:.2f}" x2="{right:.2f}" y1="{cy}" y2="{cy}" stroke="{contrast_colors[name]}" stroke-width="2"/>',
                       f'<circle cx="{center:.2f}" cy="{cy}" r="4" fill="{contrast_colors[name]}"/>']
        deltas.append(f'<text x="{zero_x-4}" y="{yy+218}" font-size="11">0</text>')
        fold_bound = max(abs(delta * scale) for name in CONTRASTS
                         for delta in result['contrasts'][name][metric]['paired_deltas'].values()) or 1
        zero_y = yy + 117
        folds.append(f'<line x1="{xx+45}" x2="{xx+505}" y1="{zero_y}" y2="{zero_y}" stroke="#94a3b8"/>')
        for name in CONTRASTS:
            points = []
            for col, fold in enumerate(FOLDS):
                delta = result['contrasts'][name][metric]['paired_deltas'][fold] * scale
                px, py = xx + 67 + col * 103, zero_y - delta / fold_bound * 70
                points.append(f'{px:.2f},{py:.2f}')
                folds.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4" fill="{contrast_colors[name]}"><title>{contrast_labels[name]}/{fold}: {delta:+.6f}</title></circle>')
            folds.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{contrast_colors[name]}" stroke-width="1.5"/>')
        for col, fold in enumerate(FOLDS):
            folds.append(f'<text x="{xx+52+col*103}" y="{yy+212}" font-size="11">{fold}</text>')
        folds.append(f'<text x="{xx+14}" y="{yy+40}" font-size="10">Scale +/- {fold_bound:.5f}; 0 = equal within that contrast</text>')
    for parts in (means, deltas, folds):
        parts.append('<text x="30" y="1080" font-size="12">Overlapping sequence splits. Blue four-single; green cross-single; purple cross-four. Percent deltas in pp.</text>')
    save_svg('parallel_global_means.svg', means)
    save_svg('parallel_global_deltas.svg', deltas)
    save_svg('parallel_global_fold_deltas.svg', folds)
    write_structure()
    payload = json.dumps({'metrics': result['metrics'], 'contrasts': result['contrasts'],
                          'decisions': result['decisions'], 'resources': result['resources'],
                          'gitrev': result['gitrev'], 'root': result['root']}, ensure_ascii=False).replace('</', '<\\/')
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DBond k5 并行全局节点与肽键读取对照</title><style>body{font:16px system-ui;margin:30px auto;padding:0 24px;max-width:1160px;background:#f8fafc;color:#172033;line-height:1.6}table{border-collapse:collapse;background:white;width:100%;margin:20px 0}th,td{padding:10px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}select{font:inherit;padding:8px}img{width:100%}small{color:#526074}.tablebox{overflow:auto}.note{padding:15px;background:#e0e7ff;border-radius:10px}</style>
<h1>256维四状态＋k5：四个并行全局槽与肽键按需读取</h1><p class="note">同轮3臂×5划分，共15次从头训练。single为原单全局节点；parallel4让每个残基连接四个并行槽；parallel4_cross保持相同四槽图，再让各肽键通过交叉注意力读取。全部指标为去padding外部测试；完整原谱内部验证选epoch，固定0.5阈值。</p>
<p>三个模式沿用同一k5卷积、四状态头和训练方案。k=5仅表示一维卷积覆盖连续五个残基；它不表示虚拟节点数量或分组。新增槽和注意力增加参数，图节点/边数也改变，不能将效果简单归因于某一种机制。</p>
<img src="parallel_global_structure.svg" alt="三种虚拟节点图结构"><label>指标 <select id="metric"></select></label><div id="tables" class="tablebox"></div>
<h2>均值和划分间波动</h2><p>误差线为样本标准差；横轴按指标放大显示，不是置信区间。</p><img src="parallel_global_means.svg" alt="三臂均值与样本标准差">
<h2>三组配对比较</h2><p>误差线为配对差样本标准差，0表示该比较内无变化。蓝：parallel4−single；绿：cross−single；紫：cross−parallel4。工程门槛不等于统计显著。</p><img src="parallel_global_deltas.svg" alt="全部三比较的配对差均值与标准差">
<h2>逐划分变化</h2><p>五组划分重叠，不能作为五个独立总体样本。注意力权重仅作描述量，不代表实际空间接触或因果贡献。</p><img src="parallel_global_fold_deltas.svg" alt="五划分全部配对差">
<div id="resources" class="tablebox"></div><p><a href="report.md">完整报告和内部诊断</a> · <a href="checkpoint_registry.json">全部权重恢复索引</a> · <a href="result.json">核验结果</a></p><small id="source"></small>
<script>const d=PAYLOAD;const sel=document.querySelector('#metric');Object.keys(d.metrics.single).forEach(k=>{let o=document.createElement('option');o.textContent=k;o.value=k;sel.append(o)});function render(){const k=sel.value;let s='<table><tr><th>臂</th><th>均值</th><th>样本SD</th></tr>';for(const[a,m]of Object.entries(d.metrics)){s+=`<tr><td>${a}</td><td>${m[k].mean.toFixed(6)}</td><td>${m[k].std_ddof1.toFixed(6)}</td></tr>`}s+='</table><table><tr><th>候选−参照</th><th>差值均值</th><th>配对SD</th><th>改善/持平/下降</th><th>完整门槛</th></tr>';for(const[a,m]of Object.entries(d.contrasts)){let v=m[k];s+=`<tr><td>${v.candidate} − ${v.reference}</td><td>${v.paired_delta_mean.toFixed(6)}</td><td>${v.paired_delta_std_ddof1.toFixed(6)}</td><td>${v.win_splits}/${v.tie_splits}/${v.loss_splits}</td><td>${d.decisions[a].improvement_gate?'通过':'未通过'}</td></tr>`}s+='</table><table><tr><th>划分</th><th>parallel4−single Δ</th><th>cross−single Δ</th><th>cross−parallel4 Δ</th></tr>';for(const f of ['1222','2252','3514','6072','9075']){s+=`<tr><td>${f}</td>`;for(const m of Object.values(d.contrasts)){s+=`<td>${m[k].paired_deltas[f].toFixed(6)}</td>`}s+='</tr>'}document.querySelector('#tables').innerHTML=s+'</table><small>交互表为原始单位；百分比指标乘100为百分比或百分点。q-Brier越低越好，其余越高越好。</small>'}sel.onchange=render;render();let r='<h2>参数、边与资源</h2><table><tr><th>臂</th><th>总参数</th><th>相对single差值</th><th>L=32虚拟边</th><th>五划分墙钟小时</th><th>train128/val512预检显存MiB</th></tr>';for(const[a,v]of Object.entries(d.resources)){r+=`<tr><td>${a}</td><td>${v.parameter_count.toLocaleString()}</td><td>${(v.parameter_count-d.resources.single.parameter_count).toLocaleString()}</td><td>${v.synthetic_virtual_edges_L32}</td><td>${(v.wall_time_seconds/3600).toFixed(3)}</td><td>${Math.max(v.full_training_batch_peak_allocated_mb,v.validation_batch_peak_allocated_mb).toFixed(1)}</td></tr>`}document.querySelector('#resources').innerHTML=r+'</table><small>三臂参数和虚拟边数按实际结构列示；墙钟时间受早停、GPU和调度影响，显存为预检批量。</small>';document.querySelector('#source').textContent='源码 '+d.gitrev+'；批次 '+d.root;</script></html>'''.replace('PAYLOAD', payload)
    (OPS / 'parallel_global_dashboard.html').write_text(page, encoding='utf-8')


def worker():
    info = read_json(OPS / 'launch.json')
    try:
        command = [PY, '-u', str(REPO / 'graph_transform/scripts/run_architecture_batch.py'),
                   '--configs', *[str(config(a)) for a in ARMS], '--fold_data_dir', 'dataset/5fold_folded',
                   '--gpus', ','.join(map(str, info['gpus'])), '--run_root', info['root']]
        with (OPS / 'training.log').open('x', encoding='utf-8') as log:
            subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        analyze(info)
        dump(OPS / 'terminal.json', {
            'status': 'completed', 'root': info['root'], 'gitrev': info['gitrev'],
            'result': str(OPS / 'result.json'), 'report': str(OPS / 'report.md'),
            'review_receipt': str(OPS / 'review_receipt.json'),
            'checkpoint_registry': str(OPS / 'checkpoint_registry.json'),
            'dashboard': str(OPS / 'parallel_global_dashboard.html'),
            'svg': str(OPS / 'parallel_global_deltas.svg'),
            'means_svg': str(OPS / 'parallel_global_means.svg'),
            'folds_svg': str(OPS / 'parallel_global_fold_deltas.svg'),
            'structure_svg': str(OPS / 'parallel_global_structure.svg')})
    except BaseException:
        status_path = Path(info['root']) / 'status.json'
        detail = read_json(status_path).get('error') if status_path.exists() else None
        dump(OPS / 'terminal.json', {'status': 'failed', 'root': info['root'], 'gitrev': info['gitrev'],
              'error': traceback.format_exc(), 'batch_error': detail, 'training_log': str(OPS / 'training.log'),
              'retry_policy': 'No automatic rerun, skip, or learned-graph continuation'})
        raise


if __name__ == '__main__':
    os.chdir(REPO)
    OPS.mkdir(exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'smoke', 'launch', 'worker'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output')
    arguments = parser.parse_args()
    if arguments.action == 'preflight':
        try:
            preflight(arguments.gpu, arguments.output)
        except BaseException:
            if arguments.output:
                dump(arguments.output, {'status': 'failed', 'error': traceback.format_exc()})
            raise
    else:
        globals()[arguments.action]()
