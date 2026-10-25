"""Pod-only static/fixed-context/dynamic-context edge-state ablation.

Three contemporaneous 256-d four-state arms, one shared updater in the two
context arms, unchanged topology and original edge representation in the head.
Only terminal receipts are inspected; failures are never automatically retried.
"""
import argparse
import copy
import csv
import datetime
import html
import itertools
import json
import math
import os
from pathlib import Path
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

OPS = Path('/mnt/pvc/graphtrans/four_state_dynamic_edge_ops')
PREVIOUS_OPS = Path('/mnt/pvc/graphtrans/four_state_width_replication_ops')
PREVIOUS_ROOT = str(REPO / 'checkpoints/graph_transform/four_state_width_replication_batch/20261022_034451')
NOTIFIED = Path('/mnt/pvc/graphtrans/dbond_completion_notified.json')
QUEUE = REPO / 'experiments/structural_queue_v1.json'
TARGETS = Path('/mnt/pvc/graphtrans/four_state_targets_v1')
ARMS = ('static', 'fixed_context', 'dynamic_context')
FOLDS = ('1222', '2252', '3514', '6072', '9075')
BASE_SEED = 42
BASELINE_REF = '4da48eeab50decd0a31fbf64f2f179478a6624a7'
METRICS = ('f1', 'accuracy', 'auc', 'ex_precision', 'ex_recall',
           'ex_f1_harmonic', 'q_brier', 'q_seq')
REPORT_METRICS = (
    ('f1', '有效键 micro-F1 %', 100), ('accuracy', '有效键 accuracy %', 100),
    ('auc', 'ROC-AUC', 1), ('ex_f1_harmonic', '谱图 P/R 调和 F1 %', 100),
    ('q_brier', 'q-Brier ↓', 1), ('q_seq', 'q_seq', 1),
)
CONTRASTS = {
    'fixed_minus_static': ('fixed_context', 'static'),
    'dynamic_minus_static': ('dynamic_context', 'static'),
    'dynamic_minus_fixed': ('dynamic_context', 'fixed_context'),
}
PY = str(REPO / '.venv/bin/python')


def require(ok, why):
    if not ok:
        raise ValueError(why)


def config(arm):
    return REPO / f'graph_transform/config/pre_synthesis_four_state_dynamic_edge_{arm}.yaml'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def revision():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()


def sources():
    return {str(p.relative_to(REPO)): file_sha(p)
            for p in (REPO / 'graph_transform').rglob('*.py') if '__pycache__' not in p.parts}


def prior_completion_barrier():
    paths = [PREVIOUS_OPS / name for name in ('launch.json', 'terminal.json', 'review_receipt.json')]
    require(all(p.is_file() for p in (*paths, NOTIFIED)),
            'Width replication must complete, pass review and be acknowledged before dynamic edges')
    previous, terminal, review = map(read_json, paths)
    require(previous['root'] == terminal['root'] == review['root'] == PREVIOUS_ROOT,
            'Preceding replication root mismatch')
    require(terminal['status'] == 'completed' and review['status'] == 'passed'
            and previous['gitrev'] == terminal['gitrev'] == review['gitrev'],
            'Preceding replication failed, incomplete, or unreviewed')
    key = PREVIOUS_ROOT + '|completed'
    reported = read_json(NOTIFIED).get('reported', {})
    require(isinstance(reported, dict) and key in reported, 'Preceding result not acknowledged')
    require(file_sha(PREVIOUS_OPS / 'result.json') == review['result_sha256']
            and file_sha(PREVIOUS_OPS / 'report.md') == review['report_sha256'],
            'Preceding result/report not bound to passing review')
    return {'root': PREVIOUS_ROOT, 'gitrev': previous['gitrev'], 'event_key': key,
            'launch_sha256': file_sha(paths[0]), 'terminal_sha256': file_sha(paths[1]),
            'review_sha256': file_sha(paths[2]), 'reported_receipt': reported[key]}


def authorization_barrier():
    index = read_json(QUEUE)
    stages = {stage['id']: stage for stage in index['stages']}
    require(stages[5]['status'] == 'reviewed' and stages[5]['root'] == PREVIOUS_ROOT,
            'Queue predecessor must be reviewed')
    require(stages[3]['status'] in {'authorized', 'ready_local_unverified', 'ready',
                                  'preflight_passed', 'submitted'},
            'Dynamic-edge stage has not been explicitly authorized')
    require(stages[3]['ops'] == str(OPS) and stages[4]['status'] == 'deferred_by_user',
            'Only dynamic-edge experiment is authorized; learned graph remains deferred')


def validate_runtime(runtime, arm):
    model, loss = runtime['model'], runtime['loss']
    require(model['hidden_dim'] == 256 and model.get('edge_state_mode', 'static') == arm,
            f'{arm}: wrong hidden dimension or edge-state mode')
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
    require(loss.get('four_state_conditional_weight', 1.) == 1.
            and loss.get('four_state_union_brier_weight', 0.) == 0., f'{arm}: alpha1/Brier0 required')
    require(runtime['data']['weighting_scheme'] == 'group_uniform'
            and runtime['data'].get('use_soft_labels') is True
            and runtime['data'].get('validation_source_csv_path')
            and runtime['training']['checkpoint_selection'] == 'max_f1'
            and runtime['evaluation']['threshold'] == .5
            and runtime['evaluation'].get('threshold_strategy', 'fixed') == 'fixed'
            and runtime.get('ablation', {}).get('pre_synthesis') is True,
            f'{arm}: training/validation/mask/evaluation protocol changed')
    seed, training = runtime['experiment']['seed'], runtime['training']
    if 'validation_split_base_seed' in training:
        require(seed == BASE_SEED and training['validation_split_base_seed'] == BASE_SEED
                and 'validation_split_seed' not in training, f'{arm}: base RNG mismatch')
    else:
        require(seed in {_stable_fold_seed(BASE_SEED, fold) for fold in FOLDS}
                and training.get('validation_split_seed') == seed, f'{arm}: actual train/split RNG mismatch')


def validate_proof(proof):
    require(proof.get('status') == 'passed', 'Preflight did not pass')
    require(set(proof['variants']) == set(ARMS) and set(proof['folds']) == set(FOLDS),
            'Missing preflight arm/fold')
    checks = proof.get('dynamic_checks', {})
    require(checks and all(value is True for value in checks.values()),
            'Not all dynamic-edge causal/compatibility checks passed')
    counts = {arm: proof['variants'][arm]['parameter_count'] for arm in ARMS}
    require(all(isinstance(n, int) and n > 0 for n in counts.values())
            and counts['fixed_context'] == counts['dynamic_context'] > counts['static'],
            'Fixed/dynamic capacity must match and updater must be active')
    for arm in ARMS:
        require(proof['variants'][arm]['updater_parameter_count'] == counts[arm] - counts['static'],
                f'{arm}: updater parameter count mismatch')
        validate_runtime(yaml.safe_load(config(arm).read_text(encoding='utf-8')), arm)
        require(file_sha(config(arm)) == proof['configs'][arm], f'{arm}: preflight config changed')


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
        subprocess.run([PY, str(REPO / 'graph_transform/scripts/test_four_state_dynamic_edges.py'),
                        '--gpu', '0', '--output', str(OPS / 'smoke.json')], cwd=REPO, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    proof = read_json(OPS / 'smoke.json')
    validate_proof(proof)
    require(before == sources(), 'Source changed during preflight')
    proof.update(gitrev=revision(), source_sha256=before,
                 target_preparation_sha256=file_sha(TARGETS / 'preparation.json'),
                 target_manifests={f: file_sha(TARGETS / f'fold_{f}/manifest.json') for f in FOLDS})
    dump(OPS / 'smoke.json', proof)
    print(json.dumps({'status': proof['status'], 'gitrev': proof['gitrev'],
                      'checks': proof['dynamic_checks'], 'variants': proof['variants']}))


def launch():
    require(not (OPS / 'launch.json').exists(), 'Launch exists; refuse duplicate')
    predecessor = prior_completion_barrier()
    authorization_barrier()
    proof = read_json(OPS / 'smoke.json')
    validate_proof(proof)
    require(proof['gitrev'] == revision() and proof['source_sha256'] == sources(),
            'Missing/stale source-bound preflight')
    changed = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=REPO, text=True).splitlines()
    require(set(changed) <= {'graph_transform/scripts/ensemble_baseline_preds.py'},
            f'Unexpected source changes: {changed}')
    require(read_json(TARGETS / 'preparation.json')['status'] == 'passed'
            and file_sha(TARGETS / 'preparation.json') == proof['target_preparation_sha256']
            and all(file_sha(TARGETS / f'fold_{f}/manifest.json') == proof['target_manifests'][f] for f in FOLDS),
            'Four-state targets changed or not validated')
    devices, gpu_rows, processes = idle()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
    root = REPO / 'checkpoints/graph_transform/four_state_dynamic_edge_batch' / stamp
    info = {
        'schema_version': 1, 'root': str(root), 'gitrev': revision(), 'predecessor': predecessor,
        'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'baseline_ref': BASELINE_REF, 'control_arm': 'static', 'edge_state_modes': list(ARMS),
        'hidden_dim': 256, 'num_attention_heads': 4, 'concat_heads': True,
        'four_state_conditional_weight': 1., 'four_state_union_brier_weight': 0.,
        'training_base_seed': BASE_SEED, 'validation_split_base_seed': BASE_SEED,
        'fold_seeds': {f: {'training': _stable_fold_seed(BASE_SEED, f),
                           'split': _stable_fold_seed(BASE_SEED, f)} for f in FOLDS},
        'gpus': devices, 'gpu_preflight': gpu_rows, 'compute_processes': processes,
        'configs': proof['configs'], 'smoke_sha256': file_sha(OPS / 'smoke.json'),
        'source_sha256': proof['source_sha256'],
        'target_preparation_sha256': proof['target_preparation_sha256'],
        'target_manifests': proof['target_manifests'],
        'scope': 'Three arms: static e0, shared updater using fixed h0, shared updater using current hl; '
                 'unchanged directed topology and head e0. All 256-d/four-state alpha1/Brier0/late/nocharge, '
                 'folded group_uniform, full-spectrum validation, padding-free fixed threshold 0.5. '
                 'Dynamic-static practical contrast, dynamic-fixed feedback contrast, fixed-static capacity contrast. '
                 'One batch only; no automatic learned-graph continuation.',
    }
    dump(OPS / 'launch.json', info)
    env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    with (OPS / 'worker.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen([PY, __file__, 'worker'], cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    info['worker_pid'] = process.pid
    dump(OPS / 'launch.json', info)
    print(json.dumps({key: value for key, value in info.items() if key != 'source_sha256'}))


def elapsed_seconds(started, finished):
    value = (datetime.datetime.fromisoformat(finished.replace('Z', '+00:00'))
             - datetime.datetime.fromisoformat(started.replace('Z', '+00:00'))).total_seconds()
    require(value >= 0, 'Negative arm wall-clock duration')
    return value


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


def verify_checkpoint(path, runtime, arm, count, provenance, fold):
    import torch
    from models.graph_transformer import GraphTransformer
    from models.dynamic_edges import validate_edge_state_checkpoint
    from models.utils import build_model_config
    saved = torch.load(path, map_location='cpu', weights_only=False)
    saved_config = saved['config']
    require(saved_config['hidden_dim'] == 256 and saved_config['bond_head_type'] == 'four_state_mlp'
            and saved_config.get('theory_fusion_stage', 'late') == 'late'
            and saved_config.get('edge_state_mode', 'static') == arm
            and saved_config['num_attention_heads'] == 4,
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
    model.load_state_dict(saved['model_state_dict'], strict=True)
    require(all(torch.isfinite(p).all().item() for p in model.parameters()),
            f'{arm}/{fold}: nonfinite saved parameters')
    model.eval()
    model.enable_edge_state_diagnostics = True
    batch, sample_proof = diagnostic_batch(effective, provenance, fold)
    with torch.inference_mode():
        output = model(batch)
    require(torch.isfinite(output[batch['label_mask'].bool()]).all().item(),
            f'{arm}/{fold}: nonfinite internal-validation diagnostic output')
    diagnostics = model.last_edge_state_diagnostics
    require(isinstance(diagnostics, list) and len(diagnostics) == 5,
            f'{arm}/{fold}: missing per-layer edge/gate diagnostics')
    # Reject NaN/Inf anywhere in the serializable diagnostics, including nested gate summaries.
    json.dumps(diagnostics, allow_nan=False)
    require([entry['layer'] for entry in diagnostics] == ['gcn_0', 'gcn_1', 'gcn_2', 'gat_0', 'gat_1'],
            f'{arm}/{fold}: diagnostic layer order mismatch')
    for entry in diagnostics:
        require(entry['edge_count'] == int(batch['edge_index'].shape[1]),
                f'{arm}/{fold}: diagnostic edge count changed')
        require(all(entry[key] >= 0 for key in ('step_delta_l2_mean', 'cumulative_delta_l2_mean',
                                                'edge_state_l2_mean', 'initial_edge_l2_mean')),
                f'{arm}/{fold}: negative edge diagnostic norm')
        require(all(entry[key] is not None and 0 <= entry[key] <= 1
                    for key in ('gate_low_fraction', 'gate_high_fraction', 'gate_saturation_fraction')),
                f'{arm}/{fold}: missing/invalid gate fractions')
        if arm == 'static':
            require(entry['step_delta_l2_mean'] == entry['cumulative_delta_l2_mean'] == 0.,
                    f'{arm}/{fold}: static edge state unexpectedly changed')
    return {'strict_cpu_restore': True, 'finite_parameters': True, 'parameter_count': count,
            'edge_state_mode': arm, 'checkpoint_training_rng_protocol': expected_rng,
            'diagnostic_sample': sample_proof, 'edge_state_diagnostics': diagnostics}


def paired_stats(candidate, reference):
    deltas = [candidate[f] - reference[f] for f in FOLDS]
    return {'paired_delta_mean': statistics.mean(deltas), 'paired_delta_std_ddof1': statistics.stdev(deltas),
            'paired_deltas': dict(zip(FOLDS, deltas))}


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
    require(status['status'] == 'completed' and len(status['arms']) == 3
            and all(a['status'] == 'completed' and a['returncode'] == 0 for a in status['arms']),
            'Three-arm batch did not complete successfully')
    require(status['root'] == manifest['root'] == info['root']
            and status['gitrev'] == manifest['gitrev'] == info['gitrev']
            and len(manifest['arms']) == 3 and manifest['folds'] == list(FOLDS),
            'Status/manifest/root/source/split mismatch')
    require(prior_completion_barrier() == info['predecessor'], 'Predecessor receipt changed')
    require(file_sha(OPS / 'smoke.json') == info['smoke_sha256'] and sources() == info['source_sha256'],
            'Preflight or source changed during training')
    require(all(file_sha(config(a)) == info['configs'][a] for a in ARMS), 'Configs changed during training')
    require(file_sha(TARGETS / 'preparation.json') == info['target_preparation_sha256']
            and all(file_sha(TARGETS / f'fold_{f}/manifest.json') == info['target_manifests'][f] for f in FOLDS),
            'Target preparation/manifests changed')
    names = [a['name'] for a in manifest['arms']]
    require(names == [f'{i:02d}_{config(a).stem}' for i, a in enumerate(ARMS, 1)]
            and manifest['control'] == names[0]
            and [a['name'] for a in status['arms']] == names
            and [a['arm'] for a in summary['sources']] == names, 'Arm identity/order mismatch')
    data, binding, recovery, resources = {}, {}, {}, {}
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
            recovery[label][fold] = {
                'gitrev': info['gitrev'], 'config': str(runtime_path), 'config_sha256': binding[str(runtime_path)],
                'checkpoint': str(checkpoint), 'checkpoint_sha256': file_sha(checkpoint),
                'metrics_csv': str(raw_path), 'metrics_sha256': binding[str(raw_path)],
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
        deltas = [data[label][f][metric] - data['static'][f][metric] for f in FOLDS]
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
           'edge_state_modes': list(ARMS), 'control_arm': 'static', 'hidden_dim': 256,
           'training_base_seed': BASE_SEED, 'validation_split_base_seed': BASE_SEED,
           'fold_seeds': info['fold_seeds'], 'predecessor': info['predecessor'],
           'metrics': metrics, 'contrasts': contrasts, 'decisions': decisions, 'folds': data,
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
                   'Same 256/four-state alpha1/Brier0/late/nocharge/masks/graph/RNG protocol verified',
                   '15 validation-selected checkpoints strictly restored on CPU and modes/RNG verified',
                   '24 arm-metric means/SD and three paired contrasts recomputed from raw CSV',
                   'Every checkpoint has five-layer diagnostics on first16 internal validation rows'],
        'hashes': binding, 'decisions': decisions,
        'result_sha256': file_sha(OPS / 'result.json'), 'report_sha256': file_sha(OPS / 'report.md'),
        'registry_sha256': file_sha(OPS / 'checkpoint_registry.json'),
        'dashboard_sha256': file_sha(OPS / 'dynamic_edge_dashboard.html'),
        'svg_sha256': file_sha(OPS / 'dynamic_edge_deltas.svg'),
    })


def write_report(result):
    metrics, contrasts, decisions = result['metrics'], result['contrasts'], result['decisions']
    lines = ['# 四状态头：固定拓扑上的动态边状态实验', '',
             f"源码：`{result['gitrev']}`；完成：{result['finished_at']}",
             f'原成功参考 `{BASELINE_REF}` 保留。三臂均从头训练，使用同轮静态边对照。', '',
             '| 指标 | static 均值 ± SD | fixed_context 均值 ± SD | dynamic_context 均值 ± SD |',
             '|---|---:|---:|---:|']
    for metric, label, scale in REPORT_METRICS:
        cells = [f"{metrics[a][metric]['mean'] * scale:.6f} ± {metrics[a][metric]['std_ddof1'] * scale:.6f}" for a in ARMS]
        lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '百分比指标的差值使用百分点；q-Brier 越低越好。', '',
              'static 每层复用 e0；fixed_context 每层更新边但端点始终为 h0；dynamic_context 使用当前层输入 h_l。',
              'fixed/dynamic 使用同一个结构、同参数量、同初始化的共享更新器；每层前更新一次，共五次；所有预测头继续使用 e0。',
              '边索引、双向主链和全局边、方向顺序、无自环规则均保持；本轮没有学习新拓扑，也不代表三维接触或物理断裂过程。',
              '三臂均256维、4头、3GCN+2GAT、原晚融合、四状态α=1/Brier=0/nocharge；训练和内部划分base42。',
              '折叠group_uniform训练，完整原谱内部验证选max-F1，固定0.5阈值，全部指标去padding。',
              '五组重叠80/20划分、每臂各一次初始化；工程门槛不等于统计显著，测试集不用于选epoch/阈值/配置。', '']
    for name, (candidate, reference) in CONTRASTS.items():
        lines += [f'## {candidate} − {reference}', '',
                  '| 指标 | 配对差均值 ± SD | 改善/持平/下降划分 |', '|---|---:|---:|']
        for metric in METRICS:
            scale = 100 if metric in {'f1', 'accuracy', 'ex_precision', 'ex_recall', 'ex_f1_harmonic'} else 1
            value = contrasts[name][metric]
            lines.append(f"| {metric} | {value['paired_delta_mean'] * scale:+.6f} ± "
                         f"{value['paired_delta_std_ddof1'] * scale:.6f} | "
                         f"{value['win_splits']}/{value['tie_splits']}/{value['loss_splits']} |")
        lines += ['', '预定门槛：' + '；'.join(f"{key}={'通过' if passed else '未通过'}"
                    for key, passed in decisions[name]['gates'].items()), '',
                  '| 划分 | Δ F1 pp | Δ accuracy pp | Δ AUC | Δ 谱图F1 pp | Δ q-Brier | Δ q_seq |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for fold in FOLDS:
            values = [f"{contrasts[name][metric]['paired_deltas'][fold] * scale:+.6f}" for metric, _, scale in REPORT_METRICS]
            lines.append('| ' + fold + ' | ' + ' | '.join(values) + ' |')
        lines.append('')
    lines += ['## 归因边界', '',
              'dynamic−static 衡量整体实用收益；dynamic−fixed 检验当前节点反馈的额外作用；fixed−static 衡量边侧容量/固定上下文重编码。',
              '若dynamic只优于static而未优于fixed，不能把收益归因于动态反馈；负结果也不证明所有动态边方法无效。', '',
              '| 臂 | 总参数 | 更新器参数 | 五划分墙钟小时 |', '|---|---:|---:|---:|']
    for arm in ARMS:
        value = result['resources'][arm]
        lines.append(f"| {arm} | {value['parameter_count']:,} | {value['updater_parameter_count']:,} | {value['wall_time_seconds'] / 3600:.4f} |")
    lines += ['', '耗时受设备、并行调度、早停影响；预检峰值显存详见resources，不能视为等步数吞吐量。', '',
              '## 各层边状态与门控诊断', '',
              '全部15份权重均在同一划分内部验证的前16行做CPU小批前向；只作描述性诊断，不选模型或配置。',
              '下表是五个内部验证小批的描述性均值。范数先对单边取L2再取均值；饱和指实际sigmoid门≤0.01或≥0.99。',
              '逐划分完整诊断在checkpoint_registry.json中；接近零或过大都不能单独构成生物学结论。', '',
              '| 臂 | 层 | 每步Δe L2均值 | 累积e_l−e0 L2均值 | e_l L2均值 | gate饱和比例 |',
              '|---|---|---:|---:|---:|---:|']
    for arm in ARMS:
        for index, layer in enumerate(('gcn_0', 'gcn_1', 'gcn_2', 'gat_0', 'gat_1')):
            entries = [result['recovery'][arm][fold]['checkpoint_verification']['edge_state_diagnostics'][index]
                       for fold in FOLDS]
            values = [statistics.mean(entry[key] for entry in entries) for key in
                      ('step_delta_l2_mean', 'cumulative_delta_l2_mean', 'edge_state_l2_mean', 'gate_saturation_fraction')]
            lines.append('| ' + arm + ' | ' + layer + ' | ' + ' | '.join(f'{v:.6f}' for v in values) + ' |')
    lines += ['', '## 恢复记录', '', f"批次：`{result['root']}`", '',
              '| 臂 | 划分 | best_model.pt | SHA256 |', '|---|---|---|---|']
    for arm in ARMS:
        for fold in FOLDS:
            value = result['recovery'][arm][fold]
            lines.append(f"| {arm} | {fold} | `{value['checkpoint']}` | `{value['checkpoint_sha256']}` |")
    (OPS / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_visuals(result):
    """Small standalone chart and interactive metric table, generated only on pod."""
    colors = {'fixed_minus_static': '#2563eb', 'dynamic_minus_static': '#dc2626', 'dynamic_minus_fixed': '#7c3aed'}
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1160" height="700" viewBox="0 0 1160 700">',
           '<rect width="1160" height="700" fill="#f8fafc"/>',
           '<style>text{font-family:Arial,sans-serif;fill:#0f172a}</style>',
           '<text x="32" y="36" font-size="22">Dynamic edges: paired changes (5 overlapping splits)</text>']
    for i, (metric, label, scale) in enumerate(REPORT_METRICS):
        x, y = 32 + (i % 2) * 565, 70 + (i // 2) * 198
        svg.append(f'<rect x="{x}" y="{y}" width="540" height="180" rx="12" fill="white"/>')
        svg.append(f'<text x="{x+16}" y="{y+25}" font-size="15">{html.escape(metric)} (delta; {"lower better" if metric == "q_brier" else "higher better"})</text>')
        maximum = max(abs(result['contrasts'][name][metric]['paired_delta_mean'] * scale) for name in CONTRASTS) or 1
        center = x + 350
        svg.append(f'<line x1="{center}" y1="{y+40}" x2="{center}" y2="{y+158}" stroke="#94a3b8"/>')
        for j, name in enumerate(CONTRASTS):
            value = result['contrasts'][name][metric]
            delta = value['paired_delta_mean'] * scale
            width = abs(delta) / maximum * 96
            start = center if delta >= 0 else center - width
            yy = y + 50 + j * 36
            svg.append(f'<text x="{x+16}" y="{yy+16}" font-size="12">{name}: {delta:+.5f} ({value["win_splits"]}/5)</text>')
            svg.append(f'<rect x="{start:.2f}" y="{yy}" width="{width:.2f}" height="22" fill="{colors[name]}"/>')
    svg += ['<text x="32" y="686" font-size="12">Percent metrics: percentage points. Engineering gates are not significance tests; see report and recovery registry.</text>', '</svg>']
    (OPS / 'dynamic_edge_deltas.svg').write_text(''.join(svg), encoding='utf-8')
    payload = json.dumps({'metrics': result['metrics'], 'contrasts': result['contrasts'],
                          'decisions': result['decisions']}, ensure_ascii=False).replace('</', '<\\/')
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DBond 动态边同轮对照</title><style>body{font:16px system-ui;margin:32px auto;padding:0 24px;max-width:1160px;background:#f8fafc;color:#172033}table{border-collapse:collapse;background:white;width:100%;margin:20px 0}th,td{padding:12px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}select{font:inherit;padding:8px}img{width:100%}small{color:#526074}</style>
<h1>256维四状态头：动态边状态</h1><p>static：原边e0；fixed_context：固定h0上下文；dynamic_context：当前层节点上下文。固定拓扑、原晚融合、α=1。</p>
<p>同轮15次训练、去padding、固定0.5阈值。五组划分重叠，门槛不代表统计显著。</p>
<label>指标 <select id="metric"></select></label><div id="tables"></div><img src="dynamic_edge_deltas.svg" alt="三组配对差对比图">
<p><a href="report.md">完整报告</a> · <a href="checkpoint_registry.json">权重恢复索引</a> · <a href="result.json">核验结果</a></p>
<script>const d=PAYLOAD;const sel=document.querySelector('#metric');Object.keys(d.metrics.static).forEach(k=>{let o=document.createElement('option');o.textContent=k;o.value=k;sel.append(o)});function render(){const k=sel.value;let s='<table><tr><th>臂</th><th>均值</th><th>样本SD</th></tr>';for(const[a,m]of Object.entries(d.metrics)){s+=`<tr><td>${a}</td><td>${m[k].mean.toFixed(6)}</td><td>${m[k].std_ddof1.toFixed(6)}</td></tr>`}s+='</table><table><tr><th>比较</th><th>差值均值</th><th>配对SD</th><th>改善/持平/下降</th><th>完整门槛</th></tr>';for(const[a,m]of Object.entries(d.contrasts)){let v=m[k];s+=`<tr><td>${a}</td><td>${v.paired_delta_mean.toFixed(6)}</td><td>${v.paired_delta_std_ddof1.toFixed(6)}</td><td>${v.win_splits}/${v.tie_splits}/${v.loss_splits}</td><td>${d.decisions[a].improvement_gate?'通过':'未通过'}</td></tr>`}document.querySelector('#tables').innerHTML=s+'</table><small>表格采用原始0–1单位；q_seq为相关系数。q-Brier越低越好，其余越高越好。完整门槛综合F1增幅、F1胜率、AUC和Brier。</small>'}sel.onchange=render;render();</script></html>'''.replace('PAYLOAD', payload)
    (OPS / 'dynamic_edge_dashboard.html').write_text(page, encoding='utf-8')


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
            'dashboard': str(OPS / 'dynamic_edge_dashboard.html'), 'svg': str(OPS / 'dynamic_edge_deltas.svg')})
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
    parser.add_argument('action', choices=['smoke', 'launch', 'worker'])
    globals()[parser.parse_args().action]()
