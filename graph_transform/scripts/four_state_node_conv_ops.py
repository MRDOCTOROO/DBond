"""Pod-only four-state no-convolution / single-scale / multiscale ablation.

Three contemporaneous 256-d four-state arms retain static edges, late theory,
concat fusion, and one frozen training/validation recipe.
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

OPS = Path('/mnt/pvc/graphtrans/four_state_node_conv_ops')
PREVIOUS_OPS = Path('/mnt/pvc/graphtrans/four_state_head_fusion_ops')
PREVIOUS_ROOT = str(REPO / 'checkpoints/graph_transform/four_state_head_fusion_batch/20261022_214615')
PREVIOUS_REVISION = 'd4a2d1c45221a78d1cedb08f87404adf8ad07533'
NOTIFIED = Path('/mnt/pvc/graphtrans/dbond_completion_notified.json')
QUEUE = REPO / 'experiments/structural_queue_v1.json'
TARGETS = Path('/mnt/pvc/graphtrans/four_state_targets_v1')
ARMS = ('none', 'single_k5', 'multiscale_357')
FOLDS = ('1222', '2252', '3514', '6072', '9075')
BASE_SEED = 42
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
    'single_k5_minus_none': ('single_k5', 'none'),
    'multiscale_357_minus_none': ('multiscale_357', 'none'),
    'multiscale_357_minus_single_k5': ('multiscale_357', 'single_k5'),
}
PY = str(REPO / '.venv/bin/python')


def require(ok, why):
    if not ok:
        raise ValueError(why)


def config(arm):
    return REPO / f'graph_transform/config/pre_synthesis_four_state_node_conv_{arm}.yaml'


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
            'Head-fusion stage must complete, pass review and be acknowledged before node convolution')
    previous, terminal, review = map(read_json, paths)
    require(previous['root'] == terminal['root'] == review['root'] == PREVIOUS_ROOT,
            'Preceding head-fusion root mismatch')
    require(terminal['status'] == 'completed' and review['status'] == 'passed'
            and previous['gitrev'] == terminal['gitrev'] == review['gitrev'] == PREVIOUS_REVISION,
            'Preceding head-fusion stage failed, incomplete, unreviewed, or has wrong source')
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
    require(stages[6]['status'] == 'reviewed' and stages[6]['root'] == PREVIOUS_ROOT,
            'Queue predecessor must be reviewed')
    require(stages[7]['status'] in {'authorized', 'authorized_waiting_predecessor', 'ready_local_unverified', 'ready',
                                  'preflight_passed', 'submitted'},
            'Node-convolution stage has not been explicitly authorized')
    require(stages[7]['ops'] == str(OPS) and stages[7]['requires_reviewed_stage'] == 6
            and stages[4]['status'] == 'deferred_by_user',
            'Node-convolution queue identity incorrect; learned graph remains deferred')


def validate_runtime(runtime, arm):
    model, loss = runtime['model'], runtime['loss']
    require(model['hidden_dim'] == 256 and model.get('node_conv_mode', 'none') == arm
            and model.get('bond_fusion_mode', 'concat') == 'concat'
            and model.get('edge_state_mode', 'static') == 'static',
            f'{arm}: wrong width/node-convolution mode, or fusion/dynamic-edge changes were enabled')
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
    checks = proof.get('conv_checks', {})
    require(checks and all(value is True for value in checks.values()),
            'Not all node-convolution causal/compatibility checks passed')
    counts = {arm: proof['variants'][arm]['parameter_count'] for arm in ARMS}
    require(all(isinstance(n, int) and n > 0 for n in counts.values()), 'Invalid model parameter count')
    require(counts['single_k5'] == counts['multiscale_357'] > counts['none'],
            'Single and multiscale convolution must have equal positive added parameter budgets')
    for arm in ARMS:
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
        subprocess.run([PY, str(REPO / 'graph_transform/scripts/test_four_state_node_conv.py'),
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
                      'checks': proof['conv_checks'], 'variants': proof['variants']}))


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
    root = REPO / 'checkpoints/graph_transform/four_state_node_conv_batch' / stamp
    info = {
        'schema_version': 1, 'root': str(root), 'gitrev': revision(), 'predecessor': predecessor,
        'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'baseline_ref': BASELINE_REF, 'control_arm': 'none', 'node_conv_modes': list(ARMS),
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
        'scope': 'Three residue encoders: original none, Conv1d 256->192 k5, and parallel k3/5/7 each 64. '
                 'SiLU and zero-initialized Linear192->256 residual after NodeEncoder, before global/FiLM. '
                 'All 256-d/four-state alpha1/Brier0/late/static edges/concat fusion/nocharge, '
                 'folded group_uniform, full-spectrum validation, padding-free fixed threshold 0.5. '
                 'Both candidates versus contemporaneous none and multiscale versus equal-parameter single. '
                 'No automatic promotion or stacking; learned graph remains deferred.',
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
    from models.bond_fusion_heads import validate_bond_fusion_checkpoint
    from models.node_context_conv import validate_node_conv_checkpoint
    from models.utils import build_model_config
    saved = torch.load(path, map_location='cpu', weights_only=False)
    saved_config = saved['config']
    require(saved_config['hidden_dim'] == 256 and saved_config['bond_head_type'] == 'four_state_mlp'
            and saved_config.get('theory_fusion_stage', 'late') == 'late'
            and saved_config.get('edge_state_mode', 'static') == 'static'
            and saved_config.get('bond_fusion_mode', 'concat') == 'concat'
            and saved_config.get('node_conv_mode', 'none') == arm
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
    model.load_state_dict(saved['model_state_dict'], strict=True)
    require(all(torch.isfinite(p).all().item() for p in model.parameters()),
            f'{arm}/{fold}: nonfinite saved parameters')
    model.eval()
    model.enable_node_conv_diagnostics = True
    batch, sample_proof = diagnostic_batch(effective, provenance, fold)
    with torch.inference_mode():
        output = model(batch)
    require(torch.isfinite(output[batch['label_mask'].bool()]).all().item(),
            f'{arm}/{fold}: nonfinite internal-validation diagnostic output')
    diagnostics = model.last_node_conv_diagnostics
    require(isinstance(diagnostics, dict) and diagnostics.get('mode') == arm
            and diagnostics.get('num_valid_residues') == int(batch['seq_lens'].sum()),
            f'{arm}/{fold}: missing node-convolution mode/count diagnostics')
    # Reject NaN/Inf anywhere, including per-branch activation summaries.
    json.dumps(diagnostics, allow_nan=False)
    kernels = [] if arm == 'none' else [5] if arm == 'single_k5' else [3, 5, 7]
    channels = [] if arm == 'none' else [192] if arm == 'single_k5' else [64, 64, 64]
    activations = diagnostics.get('branch_activation_l2_mean', [])
    require(diagnostics.get('kernel_sizes') == kernels and diagnostics.get('branch_channels') == channels
            and len(activations) == len(kernels) and all(value >= 0 for value in activations),
            f'{arm}/{fold}: convolution branch identity/activation diagnostics mismatch')
    require(all(diagnostics.get(key, -1) >= 0 for key in
                ('projection_residual_l2_mean', 'input_l2_mean', 'output_l2_mean', 'residual_relative_l2_mean')),
            f'{arm}/{fold}: node residual diagnostics invalid')
    if arm == 'none':
        require(diagnostics['projection_residual_l2_mean'] == 0
                and diagnostics['residual_relative_l2_mean'] == 0,
                f'{arm}/{fold}: baseline must not have a convolution residual')
    return {'strict_cpu_restore': True, 'finite_parameters': True, 'parameter_count': count,
            'node_conv_mode': arm, 'bond_fusion_mode': 'concat', 'edge_state_mode': 'static',
            'checkpoint_training_rng_protocol': expected_rng,
            'diagnostic_sample': sample_proof, 'node_conv_diagnostics': diagnostics}


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
        deltas = [data[label][f][metric] - data['none'][f][metric] for f in FOLDS]
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
           'node_conv_modes': list(ARMS), 'control_arm': 'none', 'hidden_dim': 256,
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
                   'Same 256/four-state alpha1/Brier0/late/static/concat/nocharge/masks/graph/RNG protocol verified',
                   'Single k5 and multiscale k3/5/7 have exactly equal parameter counts',
                   '15 validation-selected checkpoints strictly restored on CPU and node-convolution modes/RNG verified',
                   '24 arm-metric means/SD and three paired contrasts recomputed from raw CSV',
                   'Every checkpoint has local node convolution diagnostics on first16 internal validation rows'],
        'hashes': binding, 'decisions': decisions,
        'result_sha256': file_sha(OPS / 'result.json'), 'report_sha256': file_sha(OPS / 'report.md'),
        'registry_sha256': file_sha(OPS / 'checkpoint_registry.json'),
        'dashboard_sha256': file_sha(OPS / 'node_conv_dashboard.html'),
        'svg_sha256': file_sha(OPS / 'node_conv_deltas.svg'),
        'means_svg_sha256': file_sha(OPS / 'node_conv_means.svg'),
        'folds_svg_sha256': file_sha(OPS / 'node_conv_fold_deltas.svg'),
        'structure_svg_sha256': file_sha(OPS / 'node_conv_structure.svg'),
    })


def write_report(result):
    metrics, contrasts, decisions = result['metrics'], result['contrasts'], result['decisions']
    lines = ['# 四状态基座：图前节点序列卷积同轮对照', '',
             f"源码：`{result['gitrev']}`；完成：{result['finished_at']}",
             f'原成功参考 `{BASELINE_REF}` 保留。三臂均从头训练，同轮none为基准。', '',
             '| 指标 | none 均值 ± SD | single_k5 均值 ± SD | multiscale_357 均值 ± SD |',
             '|---|---:|---:|---:|']
    for metric in METRICS:
        scale = metric_scale(metric)
        cells = [f"{metrics[a][metric]['mean'] * scale:.6f} ± {metrics[a][metric]['std_ddof1'] * scale:.6f}" for a in ARMS]
        lines.append('| ' + metric + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'F1、accuracy、谱图P/R及其调和F1为百分比，差值为百分点；AUC、q-Brier、q_seq为原始单位。SD是五组描述性样本标准差，不是置信区间。', '',
              '## 接入位置与固定基座', '',
              '原NodeEncoder输出[B,L,256] → 本轮序列分支 → 原global打包及FiLM → 3GCN+2GAT → 原concat四状态头；15维理论特征仍在末端投影为256维接入。', '',
              '| 臂 | 图前残基序列分支 |', '|---|---|',
              '| none | 原256维节点编码，不添加卷积 |',
              '| single_k5 | Conv1d(256,192,k5,pad2)→SiLU→Linear192→256，残差加回原编码 |',
              '| multiscale_357 | 并行Conv1d(256,64,k3/k5/k7,pad1/2/3)，各SiLU→拼接192→Linear192→256，残差加回原编码 |', '',
              '布局为[B,256,L]，沿单条肽链序列位置卷积，不卷积特征轴或跨样本打包序列。先以真实长度清除无效输入，输出再次清除padding；global节点完全不进入卷积。',
              '投影weight/bias从零开始，初始候选函数与none一致。卷积正常初始化；不加额外norm/dropout/可调缩放系数。两候选实际参数完全相同，但最大感受野7对5，multi−single不能单独归因分支数。',
              '三臂固定256维、4头、3GCN+2GAT、原concat末端融合、晚理论融合、静态边、FiLM/global、四状态α=1/Brier=0/nocharge。未叠加其他轮候选。',
              '四状态顺序none/b_only/y_only/both，q=1−P(none)。训练和内部划分base42，group_uniform折叠训练、完整原谱内部验证strict max-F1选epoch、固定0.5和去padding指标。', '',
              '## 历史与归因边界', '',
              '历史多尺度卷积并非未测试：770ff82b6fa5bf977c9c5e02b18834efcf5e3566，seq2_batch/20261017_172512。AA embedding64→三个32通道卷积→96→256，二分类头和当时折叠内部验证，F1−0.0448个百分点、AUC−0.000453、q-Brier+0.000491；q_seq+0.002355但仅2/5改善，未通过门槛。',
              '本轮在完整256维节点编码上采用更宽分支，配四状态监督与完整原谱验证，并新增等参数单尺度对照。新旧差异不能单独归因某一改动。',
              'multi−none检查新增序列分支整体收益，single−none检查单尺度分支收益，multi−single检验相同参数预算下当前多尺度设计额外收益。仅胜none而未胜single不能宣称多尺度优越；未提升也不证明所有Conv1D无用。',
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
    lines += ['## 参数与资源', '', '| 臂 | 总参数 | 相对none参数差 | 五划分墙钟小时 |', '|---|---:|---:|---:|']
    control_count = result['resources']['none']['parameter_count']
    for arm in ARMS:
        value = result['resources'][arm]
        lines.append(f"| {arm} | {value['parameter_count']:,} | {value['parameter_count'] - control_count:+,} | {value['wall_time_seconds'] / 3600:.4f} |")
    lines += ['', '两候选总参数相等已核验。耗时受设备、调度、早停影响；预检峰值显存保存在resources，不能视为等步数吞吐量。', '',
              '## 内部实现诊断（固定内部验证前16行）', '',
              '每份权重strict CPU恢复，在所属划分完整原谱内部验证前16行、eval模式前向；不据此选择模型。卷积分支激活和投影残差范数只说明实现行为，不代表特征重要性或物理接触。', '',
              '| 臂 | 划分 | 内部诊断 |', '|---|---|---|']
    for arm in ARMS:
        for fold in FOLDS:
            diag = result['recovery'][arm][fold]['checkpoint_verification']['node_conv_diagnostics']
            lines.append('| ' + arm + ' | ' + fold + ' | `' + json.dumps(diag, ensure_ascii=False, sort_keys=True) + '` |')
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
    parts = svg_start('Node Conv1D: sequence context before graph message passing', 670)
    boxes = [(65, 'NodeEncoder: AA / position / physicochemical / masked state and environment -> [B,L,256]', '#dbeafe'),
             (140, 'none: original residual stream, no convolution', '#e2e8f0'),
             (215, 'single_k5: [B,256,L] -> Conv k5,192 channels -> SiLU -> Linear192->256', '#dbeafe'),
             (290, 'multiscale_357: parallel k3/k5/k7,64 each -> SiLU -> concat192 -> Linear192->256', '#dcfce7'),
             (365, 'Candidates: clean padding -> sequence-only branches -> zero-initialized projection -> residual + H', '#e0e7ff'),
             (440, 'Then original global packing / FiLM -> 3 GCN (256->256) -> 2 GAT (4x64->256)', '#f3e8ff'),
             (515, 'Late theory15->256 + endpoints / original edge / difference / product -> concat1536 ->256 ->4', '#fef3c7')]
    for yy, label, color in boxes:
        parts += [f'<rect x="30" y="{yy}" width="1100" height="55" rx="9" fill="{color}"/>',
                  f'<text x="46" y="{yy+33}" font-size="14">{html.escape(label)}</text>']
    parts += ['<text x="30" y="606" font-size="14">No global node in Conv1D; no mixing peptides; k3/5/7 are sequence windows, not 3D contacts.</text>',
              '<text x="30" y="635" font-size="14">Single and multiscale have equal parameters; receptive field 5 vs 7 remains a design difference.</text>']
    save_svg('node_conv_structure.svg', parts)


def write_visuals(result):
    """All plots and numerical aggregations are produced on pod only."""
    palette = {'none': '#64748b', 'single_k5': '#2563eb', 'multiscale_357': '#16a34a'}
    contrast_colors = dict(zip(CONTRASTS, ('#2563eb', '#16a34a', '#7c3aed')))
    contrast_labels = dict(zip(CONTRASTS, ('single - none', 'multi - none', 'multi - single')))
    means = svg_start('Node Conv1D: five-split mean +/- sample SD (not confidence intervals)')
    deltas = svg_start('Node Conv1D: all three paired contrasts, mean +/- paired SD')
    folds = svg_start('Node Conv1D: every paired split difference for all three contrasts')
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
        parts.append('<text x="30" y="1080" font-size="12">Overlapping sequence splits. Contrasts: blue single-none; green multi-none; purple multi-single. Percent deltas in pp.</text>')
    save_svg('node_conv_means.svg', means)
    save_svg('node_conv_deltas.svg', deltas)
    save_svg('node_conv_fold_deltas.svg', folds)
    write_structure()
    payload = json.dumps({'metrics': result['metrics'], 'contrasts': result['contrasts'],
                          'decisions': result['decisions'], 'resources': result['resources'],
                          'gitrev': result['gitrev'], 'root': result['root']}, ensure_ascii=False).replace('</', '<\\/')
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DBond 四状态节点序列卷积同轮对照</title><style>body{font:16px system-ui;margin:30px auto;padding:0 24px;max-width:1160px;background:#f8fafc;color:#172033;line-height:1.6}table{border-collapse:collapse;background:white;width:100%;margin:20px 0}th,td{padding:10px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}select{font:inherit;padding:8px}img{width:100%}small{color:#526074}.tablebox{overflow:auto}.note{padding:15px;background:#e0e7ff;border-radius:10px}</style>
<h1>256维四状态基座：图前序列卷积</h1><p class="note">同轮3臂×5划分，共15次训练。固定原concat头、静态边、原晚理论融合、α=1。未叠加其他轮候选。全部指标去padding，内部验证选epoch、固定0.5阈值。</p>
<p>在NodeEncoder完整256维输出后、global/FiLM之前，对每条肽链独立卷积。两候选参数完全相同，最大感受野仍为5对7。历史AA64多尺度分支曾失败；本轮是新底座上的复验，不是首次卷积。</p>
<img src="node_conv_structure.svg" alt="三种序列分支及接入位置"><label>指标 <select id="metric"></select></label><div id="tables" class="tablebox"></div>
<h2>均值和划分间波动</h2><p>误差线为样本标准差；横轴按指标放大显示，不是置信区间。</p><img src="node_conv_means.svg" alt="三臂均值与样本标准差">
<h2>三组配对比较</h2><p>误差线为配对差样本标准差，0表示该比较内无变化。蓝：single−none；绿：multi−none；紫：multi−single。工程门槛不等于统计显著。</p><img src="node_conv_deltas.svg" alt="全部三比较的配对差均值与标准差">
<h2>逐划分变化</h2><p>五组划分重叠，不能作为五个独立总体样本。仅胜none而未胜single，不能宣称多尺度优越。</p><img src="node_conv_fold_deltas.svg" alt="五划分全部配对差">
<div id="resources" class="tablebox"></div><p><a href="report.md">完整报告和内部诊断</a> · <a href="checkpoint_registry.json">全部权重恢复索引</a> · <a href="result.json">核验结果</a></p><small id="source"></small>
<script>const d=PAYLOAD;const sel=document.querySelector('#metric');Object.keys(d.metrics.none).forEach(k=>{let o=document.createElement('option');o.textContent=k;o.value=k;sel.append(o)});function render(){const k=sel.value;let s='<table><tr><th>臂</th><th>均值</th><th>样本SD</th></tr>';for(const[a,m]of Object.entries(d.metrics)){s+=`<tr><td>${a}</td><td>${m[k].mean.toFixed(6)}</td><td>${m[k].std_ddof1.toFixed(6)}</td></tr>`}s+='</table><table><tr><th>候选−参照</th><th>差值均值</th><th>配对SD</th><th>改善/持平/下降</th><th>完整门槛</th></tr>';for(const[a,m]of Object.entries(d.contrasts)){let v=m[k];s+=`<tr><td>${v.candidate} − ${v.reference}</td><td>${v.paired_delta_mean.toFixed(6)}</td><td>${v.paired_delta_std_ddof1.toFixed(6)}</td><td>${v.win_splits}/${v.tie_splits}/${v.loss_splits}</td><td>${d.decisions[a].improvement_gate?'通过':'未通过'}</td></tr>`}s+='</table><table><tr><th>划分</th><th>single−none Δ</th><th>multi−none Δ</th><th>multi−single Δ</th></tr>';for(const f of ['1222','2252','3514','6072','9075']){s+=`<tr><td>${f}</td>`;for(const m of Object.values(d.contrasts)){s+=`<td>${m[k].paired_deltas[f].toFixed(6)}</td>`}s+='</tr>'}document.querySelector('#tables').innerHTML=s+'</table><small>交互表为原始单位；百分比指标乘100为百分比或百分点。q-Brier越低越好，其余越高越好。</small>'}sel.onchange=render;render();let r='<h2>参数与耗时</h2><table><tr><th>臂</th><th>总参数</th><th>相对none差值</th><th>五划分墙钟小时</th></tr>';for(const[a,v]of Object.entries(d.resources)){r+=`<tr><td>${a}</td><td>${v.parameter_count.toLocaleString()}</td><td>${(v.parameter_count-d.resources.none.parameter_count).toLocaleString()}</td><td>${(v.wall_time_seconds/3600).toFixed(3)}</td></tr>`}document.querySelector('#resources').innerHTML=r+'</table><small>候选参数相等已核验。墙钟时间受早停、GPU和调度影响，不是等步数吞吐量；预检显存见完整结果。</small>';document.querySelector('#source').textContent='源码 '+d.gitrev+'；批次 '+d.root;</script></html>'''.replace('PAYLOAD', payload)
    (OPS / 'node_conv_dashboard.html').write_text(page, encoding='utf-8')


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
            'dashboard': str(OPS / 'node_conv_dashboard.html'), 'svg': str(OPS / 'node_conv_deltas.svg'),
            'means_svg': str(OPS / 'node_conv_means.svg'), 'folds_svg': str(OPS / 'node_conv_fold_deltas.svg'),
            'structure_svg': str(OPS / 'node_conv_structure.svg')})
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
