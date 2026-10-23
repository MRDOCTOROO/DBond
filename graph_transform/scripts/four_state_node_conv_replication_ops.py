"""Pod-only independent-training-seed replication of single-k5 versus no convolution.

Two contemporaneous 256-d four-state arms retain static edges, late theory,
concat fusion, and one frozen training/validation recipe.
Only terminal receipts are inspected; failures are never automatically retried.
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

OPS = Path('/mnt/pvc/graphtrans/four_state_node_conv_replication_ops')
PREVIOUS_OPS = Path('/mnt/pvc/graphtrans/four_state_node_conv_ops')
PREVIOUS_ROOT = str(REPO / 'checkpoints/graph_transform/four_state_node_conv_batch/20261023_055934')
PREVIOUS_REVISION = 'fba57b5412ba12a96a4bd0588fe688a5ff11d856'
NOTIFIED = Path('/mnt/pvc/graphtrans/dbond_completion_notified.json')
QUEUE = REPO / 'experiments/structural_queue_v1.json'
TARGETS = Path('/mnt/pvc/graphtrans/four_state_targets_v1')
ARMS = ('none', 'single_k5')
FOLDS = ('1222', '2252', '3514', '6072', '9075')
BASE_SEED = 43
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
    'single_k5_minus_none': ('single_k5', 'none'),
}
PY = str(REPO / '.venv/bin/python')


def require(ok, why):
    if not ok:
        raise ValueError(why)


def config(arm):
    return REPO / f'graph_transform/config/pre_synthesis_four_state_node_conv_seed43_{arm}.yaml'


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
            'Discovery node-convolution stage must complete, pass review and be acknowledged before node convolution')
    previous, terminal, review = map(read_json, paths)
    require(previous['root'] == terminal['root'] == review['root'] == PREVIOUS_ROOT,
            'Preceding node-convolution discovery root mismatch')
    require(terminal['status'] == 'completed' and review['status'] == 'passed'
            and previous['gitrev'] == terminal['gitrev'] == review['gitrev'] == PREVIOUS_REVISION,
            'Preceding node-convolution discovery stage failed, incomplete, unreviewed, or has wrong source')
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
    require(stages[7]['status'] == 'reviewed' and stages[7]['root'] == PREVIOUS_ROOT,
            'Queue predecessor must be reviewed')
    require(stages[8]['status'] in {'authorized', 'authorized_waiting_predecessor', 'ready_local_unverified', 'ready',
                                  'preflight_passed', 'submitted'},
            'Node-convolution stage has not been explicitly authorized')
    require(stages[8]['ops'] == str(OPS) and stages[8]['requires_reviewed_stage'] == 7
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
        require(seed == BASE_SEED and training['validation_split_base_seed'] == SPLIT_BASE_SEED
                and 'validation_split_seed' not in training, f'{arm}: base RNG mismatch')
    else:
        pairs = {(_stable_fold_seed(BASE_SEED, fold), _stable_fold_seed(SPLIT_BASE_SEED, fold)) for fold in FOLDS}
        require((seed, training.get('validation_split_seed')) in pairs and seed != training.get('validation_split_seed'),
                f'{arm}: actual train/split RNG mismatch')


def validate_proof(proof):
    require(proof.get('status') == 'passed', 'Preflight did not pass')
    require(set(proof['variants']) == set(ARMS) and set(proof['folds']) == set(FOLDS),
            'Missing preflight arm/fold')
    checks = proof.get('conv_checks', {})
    require(checks and all(value is True for value in checks.values()),
            'Not all node-convolution causal/compatibility checks passed')
    counts = {arm: proof['variants'][arm]['parameter_count'] for arm in ARMS}
    require(all(isinstance(n, int) and n > 0 for n in counts.values()), 'Invalid model parameter count')
    require(counts['single_k5'] == 2002284 and counts['none'] == 1706924,
            'Candidate/reference parameter budgets changed from discovery')
    require(proof.get('training_base_seed') == BASE_SEED and proof.get('validation_split_base_seed') == SPLIT_BASE_SEED,
            'Training/split seed bases changed')
    require(proof.get('discovery') == discovery_binding(), 'Discovery configuration/source binding changed')
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
        subprocess.run([PY, str(REPO / 'graph_transform/scripts/test_four_state_node_conv_replication.py'),
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
    root = REPO / 'checkpoints/graph_transform/four_state_node_conv_replication_batch' / stamp
    info = {
        'schema_version': 1, 'root': str(root), 'gitrev': revision(), 'predecessor': predecessor,
        'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'baseline_ref': BASELINE_REF, 'control_arm': 'none', 'node_conv_modes': list(ARMS),
        'hidden_dim': 256, 'num_attention_heads': 4, 'concat_heads': True,
        'four_state_conditional_weight': 1., 'four_state_union_brier_weight': 0.,
        'training_base_seed': BASE_SEED, 'validation_split_base_seed': SPLIT_BASE_SEED,
        'fold_seeds': {f: {'training': _stable_fold_seed(BASE_SEED, f),
                           'split': _stable_fold_seed(SPLIT_BASE_SEED, f)} for f in FOLDS},
        'gpus': devices, 'gpu_preflight': gpu_rows, 'compute_processes': processes,
        'configs': proof['configs'], 'smoke_sha256': file_sha(OPS / 'smoke.json'),
        'source_sha256': proof['source_sha256'], 'discovery': proof['discovery'],
        'target_preparation_sha256': proof['target_preparation_sha256'],
        'target_manifests': proof['target_manifests'],
        'scope': 'Two residue encoders: original none and Conv1d 256->192 k5 with SiLU and '
                 'zero-initialized Linear192->256 residual after NodeEncoder, before global/FiLM. '
                 'Both 256-d/four-state alpha1/Brier0/late/static/concat/nocharge. Training base43, '
                 'split base42; folded group_uniform, full-spectrum validation, padding-free threshold0.5. '
                 'Report discovery seed42 and replication seed43 separately. No pooling ten independent samples, '
                 'no test-selected epoch/threshold/seed or automatic promotion; learned graph deferred.',
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
            'Two-arm replication batch did not complete successfully')
    require(status['root'] == manifest['root'] == info['root']
            and status['gitrev'] == manifest['gitrev'] == info['gitrev']
            and len(manifest['arms']) == len(ARMS) and manifest['folds'] == list(FOLDS),
            'Status/manifest/root/source/split mismatch')
    require(prior_completion_barrier() == info['predecessor'], 'Predecessor receipt changed')
    require(discovery_binding() == info['discovery'], 'Discovery source/config/result binding changed')
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
           'training_base_seed': BASE_SEED, 'validation_split_base_seed': SPLIT_BASE_SEED,
           'fold_seeds': info['fold_seeds'], 'predecessor': info['predecessor'], 'discovery': info['discovery'],
           'seed_comparison': compare_seeds(contrasts),
           'metrics': metrics, 'contrasts': contrasts, 'decisions': decisions, 'folds': data,
           'resources': resources, 'recovery': recovery, 'paired_validation_verified': True,
           'scope': 'Five overlapping sequence-level 80/20 splits, one initialization per split/arm. '
                    'Both arms and their paired contrast reported; discovery and new seed reported separately. '
                    'not significance tests; no test-selected epoch/threshold/seed or automatic promotion.'}
    dump(OPS / 'result.json', out)
    dump(OPS / 'checkpoint_registry.json', {'baseline_ref': BASELINE_REF, 'gitrev': info['gitrev'],
         'root': info['root'], 'candidate_status': 'Both arms retained; independent seed replication; no automatic replacement', 'arms': recovery})
    write_report(out)
    write_visuals(out)
    dump(OPS / 'review_receipt.json', {
        'status': 'passed', 'root': info['root'], 'gitrev': info['gitrev'],
        'checks': ['10 runs completed; source/config/target/source-row hashes bound',
                   'Same 256/four-state alpha1/Brier0/late/static/concat/nocharge/masks/graph/RNG protocol verified',
                   'Both arms preserve discovery architectures; training base43 independent, split base42 unchanged',
                   '10 validation-selected checkpoints strictly restored on CPU and node-convolution modes/RNG verified',
                   '16 arm-metric means/SD and paired contrast recomputed; old/new seed comparisons separate',
                   'Every checkpoint has local node convolution diagnostics on first16 internal validation rows'],
        'hashes': binding, 'decisions': decisions,
        'result_sha256': file_sha(OPS / 'result.json'), 'report_sha256': file_sha(OPS / 'report.md'),
        'registry_sha256': file_sha(OPS / 'checkpoint_registry.json'),
        'dashboard_sha256': file_sha(OPS / 'node_conv_replication_dashboard.html'),
        'svg_sha256': file_sha(OPS / 'node_conv_replication_deltas.svg'),
        'means_svg_sha256': file_sha(OPS / 'node_conv_replication_means.svg'),
        'folds_svg_sha256': file_sha(OPS / 'node_conv_replication_fold_deltas.svg'),
        'structure_svg_sha256': file_sha(OPS / 'node_conv_replication_structure.svg'),
    })



def discovery_binding():
    """Bind reviewed discovery, immutable implementation and exact config changes."""
    smoke = read_json(PREVIOUS_OPS / 'smoke.json')
    result = read_json(PREVIOUS_OPS / 'result.json')
    discovery_launch = read_json(PREVIOUS_OPS / 'launch.json')
    require(discovery_launch['gitrev'] == PREVIOUS_REVISION
            and discovery_launch['root'] == PREVIOUS_ROOT
            and file_sha(PREVIOUS_OPS / 'smoke.json') == discovery_launch['smoke_sha256'],
            'Discovery preflight is not bound to its launch')
    require(smoke['status'] == 'passed' and result['root'] == PREVIOUS_ROOT
            and result['gitrev'] == smoke['gitrev'] == PREVIOUS_REVISION, 'Discovery binding mismatch')
    require(result['training_base_seed'] == result['validation_split_base_seed'] == SPLIT_BASE_SEED,
            'Discovery training/split seeds must both be42')
    existing = sources()
    protected = {name: sha for name, sha in smoke['source_sha256'].items()
                 if name != 'graph_transform/scripts/ensemble_baseline_preds.py'}
    require(all(existing.get(name) == sha for name, sha in protected.items()),
            'Existing production/helper source changed after discovery')
    tracked = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', PREVIOUS_REVISION,
                                       '--', 'graph_transform'], cwd=REPO, text=True).splitlines()
    git_binding = {}
    for name in tracked:
        if not name.endswith('.py') or name == 'graph_transform/scripts/ensemble_baseline_preds.py':
            continue
        content = subprocess.check_output(['git', 'show', f'{PREVIOUS_REVISION}:{name}'], cwd=REPO)
        git_binding[name] = hashlib.sha256(content).hexdigest()
        require(existing.get(name) == git_binding[name], f'Frozen discovery git source differs: {name}')
    paths = [('data', 'cache_dir'), ('data', 'cache_dir_template'),
             ('evaluation', 'output_metric_dir'), ('evaluation', 'output_pred_dir'),
             ('experiment', 'name'), ('logging', 'log_dir'), ('logging', 'tensorboard_log_dir'),
             ('training', 'checkpoint_dir')]
    bindings = {}
    for arm in ARMS:
        old_path = REPO / f'graph_transform/config/pre_synthesis_four_state_node_conv_{arm}.yaml'
        require(file_sha(old_path) == smoke['configs'][arm], f'{arm}: discovery config hash changed')
        old = yaml.safe_load(old_path.read_text(encoding='utf-8'))
        new = yaml.safe_load(config(arm).read_text(encoding='utf-8'))
        expected = copy.deepcopy(old)
        require(expected['experiment']['seed'] == SPLIT_BASE_SEED
                and expected['training']['validation_split_base_seed'] == SPLIT_BASE_SEED,
                f'{arm}: discovery seed configuration changed')
        expected['experiment']['seed'] = BASE_SEED
        for section, key in paths:
            expected[section][key] = expected[section][key].replace(
                f'four_state_node_conv_{arm}', f'four_state_node_conv_seed43_{arm}')
        require(new == expected, f'{arm}: config changed beyond training seed and output identity')
        bindings[arm] = {'discovery_config': str(old_path), 'discovery_sha256': file_sha(old_path),
                         'replication_config': str(config(arm)), 'replication_sha256': file_sha(config(arm))}
    return {'root': PREVIOUS_ROOT, 'gitrev': PREVIOUS_REVISION,
            'smoke_sha256': file_sha(PREVIOUS_OPS / 'smoke.json'),
            'result_sha256': file_sha(PREVIOUS_OPS / 'result.json'),
            'report_sha256': file_sha(PREVIOUS_OPS / 'report.md'),
            'independent_review_sha256': file_sha(PREVIOUS_OPS / 'heartbeat_review.json'),
            'unchanged_source_sha256': protected, 'discovery_git_source_sha256': git_binding, 'configs': bindings}


def compare_seeds(new_contrasts):
    """Recompute old within-seed deltas; never pool overlapping splits/seeds."""
    discovery = read_json(PREVIOUS_OPS / 'result.json')
    comparison = {}
    for metric in METRICS:
        old = paired_stats({f: discovery['folds']['single_k5'][f][metric] for f in FOLDS},
                           {f: discovery['folds']['none'][f][metric] for f in FOLDS})
        direction = -1 if metric == 'q_brier' else 1
        values = old['paired_deltas'].values()
        old.update(win_splits=sum(direction*d > 1e-12 for d in values),
                   tie_splits=sum(abs(d) <= 1e-12 for d in values),
                   loss_splits=sum(direction*d < -1e-12 for d in values))
        bound = discovery['contrasts']['single_k5_minus_none'][metric]
        for key,value in old.items():
            if isinstance(value,dict):
                require(value == bound[key], f'Discovery paired values changed: {metric}')
            else:
                require(abs(value-bound[key]) < 1e-12, f'Discovery statistic changed: {metric}/{key}')
        new = copy.deepcopy(new_contrasts['single_k5_minus_none'][metric])
        def outcome(value):
            return 'improved' if direction*value > 1e-12 else 'worse' if direction*value < -1e-12 else 'tied'
        old_outcome,new_outcome = map(outcome,(old['paired_delta_mean'],new['paired_delta_mean']))
        comparison[metric] = {'seed42':old,'seed43':new,'seed42_direction':old_outcome,
                              'seed43_direction':new_outcome,
                              'improvement_direction_reproduced':old_outcome == new_outcome == 'improved'}
    return comparison


def metric_scale(metric):
    return 100 if metric in {'f1','accuracy','ex_precision','ex_recall','ex_f1_harmonic'} else 1


def write_report(result):
    contrast = result['contrasts']['single_k5_minus_none']
    lines = ['# 单尺度 k=5 独立训练种子复验','',
             f"源码 {result['gitrev']}；完成 {result['finished_at']}；批次 {result['root']}。",'',
             '两臂256维四状态α1、Brier0、原concat/late/static/nocharge、3GCN+2GAT、4头。'
             '均从头训练，训练base43，内部划分base42保持；折叠group_uniform，完整原谱内部验证'
             'strict max-F1选权重，固定0.5，全部指标去padding。','',
             'k5接在完整NodeEncoder[B,L,256]后、global打包/FiLM前：'
             'Conv1d256→192(k5,pad2)→SiLU→零初始化Linear192→256→残差。显式清padding，'
             '不卷积global/跨肽链；保留原15维理论特征末端接入。','',
             '| 指标 | none均值±SD | single_k5均值±SD | 配对差±SD | 改善/平/降 |',
             '|---|---:|---:|---:|---:|']
    for metric in METRICS:
        s,d = metric_scale(metric),contrast[metric]
        cells = [f"{result['metrics'][a][metric]['mean']*s:.6f} ± {result['metrics'][a][metric]['std_ddof1']*s:.6f}" for a in ARMS]
        lines.append(f"| {metric} | {' | '.join(cells)} | {d['paired_delta_mean']*s:+.6f} ± {d['paired_delta_std_ddof1']*s:.6f} | {d['win_splits']}/{d['tie_splits']}/{d['loss_splits']} |")
    lines += ['', 'F1/accuracy/谱图P/R及其调和F1为百分比，差值为百分点；其余原始单位。SD是描述性样本标准差，非置信区间。','',
              '## 新旧训练种子分开比较','',
              '| 指标 | seed42差值±SD | seed42改善/平/降 | seed43差值±SD | seed43改善/平/降 | 改善方向复现 |',
              '|---|---:|---:|---:|---:|---|']
    for metric in METRICS:
        s,c = metric_scale(metric),result['seed_comparison'][metric]
        cells=[]
        for name in ('seed42','seed43'):
            d=c[name]
            cells += [f"{d['paired_delta_mean']*s:+.6f} ± {d['paired_delta_std_ddof1']*s:.6f}",
                      f"{d['win_splits']}/{d['tie_splits']}/{d['loss_splits']}"]
        lines.append(f"| {metric} | {' | '.join(cells)} | {'是' if c['improvement_direction_reproduced'] else '否'} |")
    lines += ['', '两轮使用同五组重叠序列级80/20划分，不能合并为十个独立样本。'
                  '本次检验训练随机性下方向、幅度和逐划分一致性；不按测试选epoch/阈值/seed，'
                  '不混合权重、不追加搜索、不自动替换主模型或冻结论文表。','',
              '## 预设工程门槛','']
    for key,value in result['decisions']['single_k5_minus_none']['gates'].items():
        lines.append(f"- {key}: {'通过' if value else '未通过'}")
    lines += ['', '四项全部通过才通过本轮工程门槛，不等于统计显著或全局最优。k5比none增加295360参数，不能排除容量作用；不检验多尺度优越性。','',
              '## 逐划分配对差','', '| 种子 | 划分 | '+' | '.join(METRICS)+' |',
              '|---|---|'+'---:|'*len(METRICS)]
    for seed_name in ('seed42','seed43'):
        for fold in FOLDS:
            cells=[f"{result['seed_comparison'][m][seed_name]['paired_deltas'][fold]*metric_scale(m):+.6f}" for m in METRICS]
            lines.append(f"| {seed_name} | {fold} | {' | '.join(cells)} |")
    lines += ['', '## 参数、耗时与显存','',
              '| 臂 | 参数 | 增量 | 墙钟小时 | 预检峰值allocated/reserved MiB |','|---|---:|---:|---:|---:|']
    for arm in ARMS:
        r=result['resources'][arm]
        lines.append(f"| {arm} | {r['parameter_count']} | {r['parameter_count']-result['resources']['none']['parameter_count']:+d} | {r['wall_time_seconds']/3600:.4f} | {r['peak_cuda_allocated_bytes']/1024**2:.2f}/{r['peak_cuda_reserved_bytes']/1024**2:.2f} |")
    lines += ['', '耗时受早停和调度影响，不是等步吞吐；显存是有来源绑定的预检峰值。','',
              '## 内部诊断与恢复','',
              '每份权重strict CPU恢复，实际training_rng_protocol、node_conv模式、四态配置及来源匹配。'
              '所属内部验证前16行的分支/投影残差范数只作实现诊断，不是因果特征重要性。','',
              '| 臂 | 划分 | checkpoint | SHA256 | 内部诊断 |','|---|---|---|---|---|']
    for arm in ARMS:
        for fold in FOLDS:
            r=result['recovery'][arm][fold]
            diag=r['checkpoint_verification']['node_conv_diagnostics']
            lines.append(f"| {arm} | {fold} | {r['checkpoint']} | {r['checkpoint_sha256']} | {json.dumps(diag,ensure_ascii=False,sort_keys=True)} |")
    lines += ['',f"原成功{BASELINE_REF}和发现批次{PREVIOUS_REVISION}保留；完整权重/配置SHA见checkpoint_registry.json。",
              '完成后独立复核、报告并ack，结束自动接续；学习图仍暂缓。']
    (OPS/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def svg_start(title,height=1120):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="1160" height="{height}" viewBox="0 0 1160 {height}">',
            f'<rect width="1160" height="{height}" fill="#f8fafc"/>',
            '<style>text{font-family:Arial,sans-serif;fill:#0f172a}</style>',
            f'<text x="25" y="35" font-size="22">{html.escape(title)}</text>']


def save_svg(name,parts):
    (OPS/name).write_text(''.join([*parts,'</svg>']),encoding='utf-8')


def write_visuals(result):
    colors={'none':'#64748b','single_k5':'#2563eb','seed42':'#94a3b8','seed43':'#2563eb'}
    means=svg_start('Seed43 contemporaneous replication: means +/- sample SD')
    deltas=svg_start('Single k5 - none: discovery seed42 versus replication seed43')
    folds=svg_start('Each within-seed paired difference: no pooling as ten independent splits')
    for i,metric in enumerate(METRICS):
        x,y,scale=25+(i%2)*575,65+(i//2)*250,metric_scale(metric)
        for p in (means,deltas,folds):
            p += [f'<rect x="{x}" y="{y}" width="550" height="235" rx="10" fill="white"/>',
                  f'<text x="{x+12}" y="{y+25}" font-size="15">{metric} / {"lower better" if metric=="q_brier" else "higher better"}</text>']
        entries=[result['metrics'][a][metric] for a in ARMS]
        lo=min((e['mean']-e['std_ddof1'])*scale for e in entries)
        hi=max((e['mean']+e['std_ddof1'])*scale for e in entries)
        span=hi-lo or 1
        lo,hi=lo-span*.1,hi+span*.1
        for j,arm in enumerate(ARMS):
            d=result['metrics'][arm][metric]
            mean,sd,cy=d['mean']*scale,d['std_ddof1']*scale,y+78+j*74
            left,mid,right=[x+220+(v-lo)/(hi-lo)*300 for v in (mean-sd,mean,mean+sd)]
            means += [f'<text x="{x+12}" y="{cy-9}" font-size="13">{arm}</text>',
                      f'<text x="{x+12}" y="{cy+11}" font-size="12">{mean:.5f} +/- {sd:.5f}</text>',
                      f'<line x1="{left:.2f}" x2="{right:.2f}" y1="{cy}" y2="{cy}" stroke="{colors[arm]}" stroke-width="2"/>',
                      f'<circle cx="{mid:.2f}" cy="{cy}" r="5" fill="{colors[arm]}"/>']
        means.append(f'<text x="{x+15}" y="{y+220}" font-size="11">Zoomed axis {lo:.5f} to {hi:.5f}; SD is not CI</text>')
        c=result['seed_comparison'][metric]
        bound=max((abs(c[s]['paired_delta_mean'])+c[s]['paired_delta_std_ddof1'])*scale for s in ('seed42','seed43')) or 1
        zero=x+390
        deltas.append(f'<line x1="{zero}" x2="{zero}" y1="{y+40}" y2="{y+202}" stroke="#94a3b8"/>')
        for j,s in enumerate(('seed42','seed43')):
            d,cy=c[s],y+77+j*79
            mean,sd=d['paired_delta_mean']*scale,d['paired_delta_std_ddof1']*scale
            left,mid,right=[zero+v/bound*130 for v in (mean-sd,mean,mean+sd)]
            deltas += [f'<text x="{x+12}" y="{cy-11}" font-size="12">{s}: {mean:+.5f} +/- {sd:.5f}</text>',
                       f'<text x="{x+12}" y="{cy+10}" font-size="12">improved/tied/worse {d["win_splits"]}/{d["tie_splits"]}/{d["loss_splits"]}</text>',
                       f'<line x1="{left:.2f}" x2="{right:.2f}" y1="{cy}" y2="{cy}" stroke="{colors[s]}" stroke-width="2"/>',
                       f'<circle cx="{mid:.2f}" cy="{cy}" r="5" fill="{colors[s]}"/>']
        deltas.append(f'<text x="{zero-4}" y="{y+220}" font-size="12">0</text>')
        bound=max(abs(d*scale) for s in ('seed42','seed43') for d in c[s]['paired_deltas'].values()) or 1
        zy=y+120
        folds.append(f'<line x1="{x+40}" x2="{x+520}" y1="{zy}" y2="{zy}" stroke="#94a3b8"/>')
        for s in ('seed42','seed43'):
            points=[]
            for j,f in enumerate(FOLDS):
                d=c[s]['paired_deltas'][f]*scale
                px,py=x+65+j*105,zy-d/bound*72
                points.append(f'{px:.2f},{py:.2f}')
                folds.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4" fill="{colors[s]}"><title>{s}/{f}: {d:+.6f}</title></circle>')
            folds.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[s]}" stroke-width="1.5"/>')
        for j,f in enumerate(FOLDS):
            folds.append(f'<text x="{x+49+j*105}" y="{y+220}" font-size="11">{f}</text>')
    for p in (means,deltas,folds):
        p.append('<text x="25" y="1102" font-size="12">Fixed split base42; training base43. Five overlapping sequence splits. Percent differences in pp; others raw.</text>')
    save_svg('node_conv_replication_means.svg',means)
    save_svg('node_conv_replication_deltas.svg',deltas)
    save_svg('node_conv_replication_fold_deltas.svg',folds)
    structure=svg_start('Fixed architecture and independent training RNG',600)
    labels=['Same five source / train / full-spectrum internal-validation SHA: split base42',
            'NodeEncoder [B,L,256] -> none OR Conv1d k5: 256->192 -> SiLU -> Linear192->256 residual',
            'Clean padding; no global node or cross-peptide convolution',
            'Original global packing + FiLM -> 3 GCN -> 2 GAT (4 x 64)',
            'Six-source concat including late theory15->256 -> MLP256->4; q = 1-P(none)',
            'Both arms training base43; same-round reference; discovery base42 reported separately']
    for j,label in enumerate(labels):
        yy=65+j*75
        structure += [f'<rect x="25" y="{yy}" width="1110" height="55" rx="8" fill="#dbeafe"/>',
                      f'<text x="40" y="{yy+32}" font-size="14">{html.escape(label)}</text>']
    save_svg('node_conv_replication_structure.svg',structure)
    page=['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
          '<title>DBond k5独立种子复验</title><style>body{font:16px system-ui;line-height:1.65;max-width:1160px;margin:28px auto;padding:0 24px;background:#f8fafc;color:#172033}img{width:100%}table{border-collapse:collapse;width:100%;background:white}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}.box{overflow:auto}details{margin:20px 0}</style>',
          '<h1>单尺度 k=5：独立训练种子复验</h1><p>同轮none与single_k5，共10次训练；固定256维四状态α1、原concat/late/static/nocharge。训练base43，原内部划分base42保持。全部指标去padding，内部验证选epoch，固定0.5。</p>',
          '<img src="node_conv_replication_structure.svg" alt="结构与固定划分">',
          '<p>seed42为发现批次，seed43为新训练随机性复验。两轮相同五组划分重叠，不能当十个独立样本。方向复现和工程门槛不是统计显著，不自动选seed、叠加模块或替换主模型。百分比指标显示为%和百分点，其余原始单位。</p>']
    for metric in METRICS:
        scale=metric_scale(metric)
        page += [f'<details open><summary>{metric}</summary><div class="box"><table><tr><th>本轮臂</th><th>均值</th><th>样本SD</th></tr>']
        for arm in ARMS:
            d=result['metrics'][arm][metric]
            page.append(f'<tr><td>{arm}</td><td>{d["mean"]*scale:.6f}</td><td>{d["std_ddof1"]*scale:.6f}</td></tr>')
        page.append('</table><table><tr><th>训练种子</th><th>k5−none差值</th><th>配对SD</th><th>改善/平/降</th></tr>')
        for s in ('seed42','seed43'):
            d=result['seed_comparison'][metric][s]
            page.append(f'<tr><td>{s}</td><td>{d["paired_delta_mean"]*scale:+.6f}</td><td>{d["paired_delta_std_ddof1"]*scale:.6f}</td><td>{d["win_splits"]}/{d["tie_splits"]}/{d["loss_splits"]}</td></tr>')
        page.append('</table></div></details>')
    page += ['<h2>本轮均值与样本标准差</h2><img src="node_conv_replication_means.svg" alt="均值SD">',
             '<h2>新旧种子配对差</h2><p>灰色seed42，蓝色seed43。误差线为样本SD，非置信区间。</p><img src="node_conv_replication_deltas.svg" alt="配对差SD">',
             '<h2>逐划分变化</h2><img src="node_conv_replication_fold_deltas.svg" alt="逐划分差">',
             '<h2>参数与资源</h2><table><tr><th>臂</th><th>参数</th><th>墙钟小时</th><th>预检峰值allocated MiB</th></tr>']
    for arm in ARMS:
        r=result['resources'][arm]
        page.append(f'<tr><td>{arm}</td><td>{r["parameter_count"]}</td><td>{r["wall_time_seconds"]/3600:.3f}</td><td>{r["peak_cuda_allocated_bytes"]/1024**2:.2f}</td></tr>')
    page += ['</table><p>墙钟受早停/调度影响，不是等步吞吐；k5增加295360参数，不能排除容量影响。</p>',
             '<p><a href="report.md">完整报告、门槛和诊断</a> · <a href="checkpoint_registry.json">恢复索引</a> · <a href="result.json">结果来源</a></p>',
             f'<small>源码 {html.escape(result["gitrev"])}；批次 {html.escape(result["root"])}</small></html>']
    (OPS/'node_conv_replication_dashboard.html').write_text('\n'.join(page),encoding='utf-8')


def worker():
    info=read_json(OPS/'launch.json')
    try:
        command=[PY,'-u',str(REPO/'graph_transform/scripts/run_architecture_batch.py'),
                 '--configs',*[str(config(a)) for a in ARMS],'--fold_data_dir','dataset/5fold_folded',
                 '--gpus',','.join(map(str,info['gpus'])),'--run_root',info['root']]
        with (OPS/'training.log').open('x',encoding='utf-8') as log:
            subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
        analyze(info)
        dump(OPS/'terminal.json',{'status':'completed','root':info['root'],'gitrev':info['gitrev'],
             'result':str(OPS/'result.json'),'report':str(OPS/'report.md'),
             'review_receipt':str(OPS/'review_receipt.json'),'checkpoint_registry':str(OPS/'checkpoint_registry.json'),
             'dashboard':str(OPS/'node_conv_replication_dashboard.html'),'svg':str(OPS/'node_conv_replication_deltas.svg'),
             'means_svg':str(OPS/'node_conv_replication_means.svg'),'folds_svg':str(OPS/'node_conv_replication_fold_deltas.svg'),
             'structure_svg':str(OPS/'node_conv_replication_structure.svg')})
    except BaseException:
        path=Path(info['root'])/'status.json'
        detail=read_json(path).get('error') if path.exists() else None
        dump(OPS/'terminal.json',{'status':'failed','root':info['root'],'gitrev':info['gitrev'],
             'error':traceback.format_exc(),'batch_error':detail,'training_log':str(OPS/'training.log'),
             'retry_policy':'No automatic rerun, skip or further experiment'})
        raise


if __name__ == '__main__':
    require(sys.platform.startswith('linux') and str(REPO).startswith('/mnt/pvc/'),'Pod execution only')
    os.chdir(REPO)
    OPS.mkdir(exist_ok=True)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['smoke','launch','worker'])
    globals()[parser.parse_args().action]()
