"""Budget-reduced Stage 13: five new 3x64 folds with a preserved k5 control.

Usage is deliberately gated: smoke first, then launch only after an audited
Stage 12 receipt. The worker emits one completed/failed terminal event; callers
should wait for that event rather than repeatedly reading epoch logs.
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import traceback
import uuid
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'graph_transform/scripts'
CONFIGS = {
    'S13-C': ROOT / 'graph_transform/config/stage13_multiscale/s13_c_multiscale_357_c64.yaml',
}
BASELINE_CONFIG = ROOT / 'graph_transform/config/stage13_multiscale/s13_a_single_k5.yaml'
FOLDS = ('1222', '2252', '3514', '6072', '9075')
ALIASES = []
STAGE12_OPS = Path('/mnt/pvc/graphtrans/dbond_s12_gated_fusion_20261026_ops')
STAGE13_OPS = Path('/mnt/pvc/graphtrans/dbond_s13_minimal_c64_20261026_ops')
PRESERVED_OPS = Path('/mnt/pvc/graphtrans/dbond_s13_multiscale_conv_20261026_ops')
PRESERVED_ROOT = Path('/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/dbond_s13_multiscale_conv_20261026_20261026_010518_d4ec97ad')
STAGE13_RUN_BASE = Path('/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def source_path(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def write_new_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        stream.write('\n')


def replace_json(path, value):
    temporary = Path(path).with_name(Path(path).name + f'.{uuid.uuid4().hex}.tmp')
    write_new_json(temporary, value)
    os.replace(temporary, path)


def write_new_or_matching_json(path, value):
    """Allow registration to resume after a partial ops write, never change it."""
    path = Path(path)
    if path.exists():
        canonical = json.loads(json.dumps(value, allow_nan=False))
        require(read_json(path) == canonical,
                f'Existing {path.name} differs from reconstructed output')
    else:
        write_new_json(path, value)


def pod_only(ops_dir):
    require(sys.platform.startswith('linux') and str(ROOT).startswith('/mnt/pvc/'),
            'Stage 13 ops must run in the authorized pod snapshot')
    ops_dir = Path(ops_dir).expanduser().resolve()
    require(ops_dir == STAGE13_OPS,
            f'Use the dedicated Stage 13 ops namespace {STAGE13_OPS}')
    require(ops_dir != ROOT and ROOT not in ops_dir.parents,
            'Ops root must be outside the frozen source snapshot')
    return ops_dir


def source_fingerprint():
    paths = sorted((ROOT / 'graph_transform').rglob('*.py'))
    paths += list(CONFIGS.values()) + [BASELINE_CONFIG]
    paths += [ROOT / 'graph_transform/config/pre_synthesis_four_state_node_conv_single_k5.yaml']
    require(all(path.is_file() for path in paths), 'A Stage 13 source/config file is missing')
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def fingerprint_sha(fingerprint):
    encoded = json.dumps(fingerprint, sort_keys=True,
                         separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def git_revision():
    try:
        top = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=ROOT,
                             capture_output=True, text=True)
        if top.returncode != 0 or Path(top.stdout.strip()).resolve() != ROOT:
            return None
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                capture_output=True, text=True)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def check_fold_sources(fold_sources):
    require(set(fold_sources) == set(FOLDS), 'Smoke lacks one or more fold sources')
    for fold, row in fold_sources.items():
        for path_key, sha_key in (
                ('folded_training_source', 'folded_training_source_sha256'),
                ('test_csv', 'test_csv_sha256'),
                ('validation_source_csv', 'validation_source_sha256'),
                ('target_manifest', 'target_manifest_sha256')):
            path = source_path(row[path_key])
            require(path.is_file() and sha(path) == row[sha_key],
                    f'{fold}: {path_key} changed since Stage 13 smoke')


def check_preserved_baseline():
    """Bind the five completed A folds before adopting prior smoke evidence."""
    old_launch = read_json(PRESERVED_OPS / 'launch.json')
    old_manifest = read_json(PRESERVED_ROOT / 'manifest.json')
    old_status = read_json(PRESERVED_ROOT / 'status.json')
    old_smoke = read_json(PRESERVED_OPS / 'smoke.json')
    require(old_launch['ops'] == str(PRESERVED_OPS)
            and old_launch['root'] == str(PRESERVED_ROOT)
            and old_launch['source_revision'] ==
                '2c90d2cbea0900adf90df6554da6e3155292f91d'
            and old_manifest['root'] == old_status['root'] == str(PRESERVED_ROOT)
            and len(old_manifest['arms']) == len(old_status['arms']) == 3
            and old_status['arms'][0]['status'] == 'completed'
            and old_status['arms'][0]['returncode'] == 0,
            'Preserved single-k5 baseline is not the completed old first arm')
    arm = old_manifest['arms'][0]
    require(Path(arm['config_snapshot']).resolve() ==
            (PRESERVED_ROOT / 'arms' / ('01_' + BASELINE_CONFIG.stem) /
             'config.yaml').resolve()
            and arm['config_sha256'] == old_launch['configs']['S13-A']
            == sha(BASELINE_CONFIG) == sha(arm['config_snapshot'])
            and old_launch['smoke_sha256'] == sha(PRESERVED_OPS / 'smoke.json')
            and old_smoke['status'] == 'passed',
            'Preserved baseline config or smoke proof changed')
    with Path(arm['metrics_csv']).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 5 and {row['fold_id'] for row in rows} == set(FOLDS),
            'Preserved baseline lacks five complete folds')
    for relative, expected in old_launch['source_files_sha256'].items():
        current = ROOT / relative
        require(current.is_file() and sha(current) == expected,
                f'Production source differs from preserved baseline: {relative}')
    check_fold_sources(old_smoke['fold_sources'])
    return {'ops': str(PRESERVED_OPS), 'root': str(PRESERVED_ROOT),
            'source_commit': old_launch['source_revision'],
            'launch_sha256': sha(PRESERVED_OPS / 'launch.json'),
            'batch_manifest_sha256': sha(PRESERVED_ROOT / 'manifest.json'),
            'baseline_config_sha256': sha(BASELINE_CONFIG),
            'fivefold_metrics_sha256': sha(arm['metrics_csv']),
            'smoke_sha256': sha(PRESERVED_OPS / 'smoke.json'),
            'fold_sources': old_smoke['fold_sources']}


def check_idle_gpus(gpus):
    """Refuse devices with existing compute PIDs or substantial used memory."""
    requested = [int(part) for part in gpus.split(',')]
    require(requested and len(set(requested)) == len(requested)
            and all(index >= 0 for index in requested), 'Invalid --gpus list')
    gpu_info = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,uuid,memory.used',
         '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    rows = {}
    for line in gpu_info.stdout.splitlines():
        fields = [field.strip() for field in line.split(',')]
        require(len(fields) == 3, f'Unexpected nvidia-smi GPU row: {line}')
        rows[int(fields[0])] = {'uuid': fields[1], 'memory_used_mb': int(fields[2])}
    processes = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
         '--format=csv,noheader'], capture_output=True, text=True, check=True)
    busy = set()
    for line in processes.stdout.splitlines():
        fields = [field.strip() for field in line.split(',')]
        if len(fields) >= 2 and fields[1].isdigit():
            busy.add(fields[0])
    for index in requested:
        require(index in rows and rows[index]['uuid'] not in busy
                and rows[index]['memory_used_mb'] < 1024,
                f'GPU {index} is occupied or has more than 1 GiB in use')
    return {str(index): rows[index] for index in requested}


def run_logged(command, log_path, env=None):
    with Path(log_path).open('x', encoding='utf-8') as log:
        return subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                              stdout=log, stderr=subprocess.STDOUT).returncode


def queue_state(status, run_root=None, detail=None):
    return {'namespace': 'stage13_minimal_c64_v1', 'stage': 'S13',
            'status': status, 'updated_at': now(), 'root': run_root,
            'run_root': run_root,
            'detail': detail,
            'arms': [{'arm': 'S13-A', 'status': 'reused_completed',
                      'independent_training': False,
                      'source_root': str(PRESERVED_ROOT)},
                     {'arm': 'S13-C', 'status': status,
                      'independent_training': True}]}


def smoke(args):
    ops = pod_only(args.ops_dir)
    ops.mkdir(parents=True, exist_ok=True)
    require(args.gpu is not None, 'smoke requires --gpu')
    require(not (ops / 'launch.json').exists() and not (ops / 'terminal.json').exists(),
            'Stage 13 was submitted; preflight cannot be repeated')
    allowed = {'smoke_attempts', 'smoke.json', 'queue.json'}
    require({path.name for path in ops.iterdir()} <= allowed,
            'This Stage 13 ops namespace contains non-preflight outputs')
    prior_smoke = ops / 'smoke.json'
    if prior_smoke.exists():
        require(read_json(prior_smoke).get('status') == 'failed',
                'Passing Stage 13 preflight cannot be repeated')
    prior_queue = ops / 'queue.json'
    if prior_queue.exists():
        require(read_json(prior_queue).get('status') in ('failed', 'smoke_passed'),
                'Stage 13 queue has advanced past preflight')
    attempts = ops / 'smoke_attempts'
    attempts.mkdir(exist_ok=True)
    previous = sorted(attempts.glob('attempt_*'))
    require(all(path.is_dir() and (path / 'receipt.json').is_file()
                for path in previous), 'Incomplete prior Stage 13 preflight receipt')
    attempt = attempts / f'attempt_{len(previous) + 1:04d}'
    attempt.mkdir()
    try:
        before = source_fingerprint()
        old_smoke_path = PRESERVED_OPS / 'smoke.json'
        old_smoke = read_json(old_smoke_path)
        require(old_smoke['status'] == 'passed'
                and old_smoke['stage'] == 'S13'
                and sha(old_smoke['structural_preflight']) ==
                    old_smoke['structural_preflight_sha256']
                and sha(old_smoke['gpu_smoke']) == old_smoke['gpu_smoke_sha256']
                and old_smoke['config_sha256']['S13-C'] == sha(CONFIGS['S13-C']),
                'Prior Stage13 passing C smoke proof cannot be reused')
        fold_sources = old_smoke['fold_sources']
        check_fold_sources(fold_sources)
        proof = {'status': 'passed', 'stage': 'S13', 'created_at': now(),
                 'attempt': str(attempt),
                 'source_revision': git_revision(),
                 'source_tree_sha256': fingerprint_sha(before),
                 'source_files_sha256': before,
                 'config_sha256': {arm: sha(path) for arm, path in CONFIGS.items()},
                 'structural_preflight': old_smoke['structural_preflight'],
                 'structural_preflight_sha256': old_smoke['structural_preflight_sha256'],
                 'gpu_smoke': old_smoke['gpu_smoke'],
                 'gpu_smoke_sha256': old_smoke['gpu_smoke_sha256'],
                 'reused_prior_smoke': str(old_smoke_path),
                 'reused_prior_smoke_sha256': sha(old_smoke_path),
                 'smoke_gpu': old_smoke['smoke_gpu'],
                 'requested_gpu': args.gpu,
                 'fold_sources': fold_sources,
                 'D_alias_of_A': False,
                 'preserved_baseline': check_preserved_baseline()}
        write_new_json(attempt / 'checks_passed.json', proof)
        write_new_json(attempt / 'receipt.json', proof)
        replace_json(ops / 'queue.json', queue_state('smoke_passed'))
        replace_json(ops / 'smoke.json', proof)
    except BaseException as error:
        failure = {'status': 'failed', 'stage': 'S13', 'at': now(),
                   'attempt': str(attempt),
                   'error': f'{type(error).__name__}: {error}',
                   'traceback': traceback.format_exc()}
        replace_json(attempt / 'receipt.json', failure)
        replace_json(ops / 'smoke.json', failure)
        replace_json(ops / 'queue.json', queue_state('failed', detail=failure['error']))
        raise
    print(json.dumps({'status': 'passed', 'smoke': str(ops / 'smoke.json'),
                      'attempt': str(attempt)}))


def launch(args):
    ops = pod_only(args.ops_dir)
    require(not (ops / 'launch.json').exists() and not (ops / 'terminal.json').exists(),
            'Stage 13 already submitted in this ops namespace')
    proof = read_json(ops / 'smoke.json')
    require(proof['status'] == 'passed' and proof['stage'] == 'S13',
            'Passing Stage 13 pod smoke is required before launch')
    attempt = Path(proof['attempt']).resolve()
    require(attempt.parent == (ops / 'smoke_attempts').resolve()
            and read_json(attempt / 'receipt.json') == proof,
            'Stage 13 passing smoke is not bound to its immutable attempt receipt')
    require(proof['source_files_sha256'] == source_fingerprint()
            and proof['source_tree_sha256'] == fingerprint_sha(source_fingerprint())
            and all(proof['config_sha256'][arm] == sha(path)
                    for arm, path in CONFIGS.items())
            and proof['structural_preflight_sha256'] == sha(proof['structural_preflight'])
            and proof['gpu_smoke_sha256'] == sha(proof['gpu_smoke']),
            'Source, config, or preflight proof changed after smoke')
    check_fold_sources(proof['fold_sources'])
    preserved = check_preserved_baseline()
    require(proof.get('preserved_baseline') == preserved,
            'Preserved completed baseline changed since smoke')
    require(args.stage12_audit is not None, 'Stage 12 audited receipt is required')
    predecessor = Path(args.stage12_audit).expanduser().resolve()
    receipt = read_json(predecessor)
    require(receipt.get('status') == 'audited'
            and receipt.get('stage') in ('S12', 'stage12'),
            'Stage 12 has not been independently audited')
    require(Path(receipt.get('ops', '')).resolve() == STAGE12_OPS
            and predecessor == STAGE12_OPS / 'independent_audit/formal/audit_receipt.json',
            'Stage 12 audit receipt is not bound to the designated independent ops')
    stage12_root = receipt.get('root')
    require(isinstance(stage12_root, str) and Path(stage12_root).is_absolute()
            and Path(stage12_root).is_dir(),
            'Stage 12 receipt has no existing absolute run root')
    stage12_launch_path = STAGE12_OPS / 'launch.json'
    require(stage12_launch_path.is_file(), 'Stage 12 launch record missing')
    stage12_launch = read_json(stage12_launch_path)
    require((stage12_launch.get('root') or stage12_launch.get('run_root')) == stage12_root,
            'Stage 12 audited root differs from its launch record')
    stage12_queue_path = STAGE12_OPS / 'queue.json'
    if stage12_queue_path.is_file():
        stage12_queue = read_json(stage12_queue_path)
        queue_root = stage12_queue.get('root') or stage12_queue.get('run_root')
        require(queue_root in (None, stage12_root),
                'Stage 12 audited root differs from its queue record')
    require(args.gpus is not None, 'launch requires explicitly allocated --gpus')
    gpu_info = check_idle_gpus(args.gpus)
    require(gpu_info, 'No idle GPU is available for C-only folds')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
    run_root = STAGE13_RUN_BASE / f'dbond_s13_minimal_c64_20261026_{stamp}_{uuid.uuid4().hex[:8]}'
    require(STAGE13_RUN_BASE.is_dir(), 'Stage 13 checkpoint base directory is missing')
    require(not run_root.exists(), 'Run root collision')
    current_revision = git_revision()
    source_commit = args.source_commit or current_revision
    require(source_commit is not None and re.fullmatch(r'[0-9a-f]{40}', source_commit),
            'Supply the exact 40-character source commit for a Git-free pod snapshot')
    require(current_revision is None or current_revision == source_commit,
            'Declared source commit differs from pod Git HEAD')
    info = {'status': 'submitted', 'stage': 'S13', 'created_at': now(),
            'budget_profile': 'minimal_s13_c64_5fold_v1',
            'ops': str(ops), 'ops_root': str(ops), 'root': str(run_root),
            'run_root': str(run_root),
            'source_root': str(ROOT.resolve()),
            'source_revision': source_commit,
            'source_tree_sha256': proof['source_tree_sha256'],
            'source_files_sha256': proof['source_files_sha256'],
            'configs': proof['config_sha256'], 'smoke_sha256': sha(ops / 'smoke.json'),
            'stage12_audit': str(predecessor), 'stage12_audit_sha256': sha(predecessor),
            'stage12_root': stage12_root,
            'stage12_launch_sha256': sha(stage12_launch_path),
            'stage12_queue_sha256': sha(stage12_queue_path) if stage12_queue_path.is_file() else None,
            'gpus': args.gpus, 'gpu_idle_check': gpu_info,
            'batch_runner': str(SCRIPTS / 'run_architecture_batch.py'),
            'batch_runner_sha256': sha(SCRIPTS / 'run_architecture_batch.py'),
            'fold_data_dir': str(ROOT / 'dataset/5fold_folded'),
            'D_alias_of_A': False,
            'preserved_baseline': preserved}
    process = None
    try:
        write_new_json(ops / 'launch.json', info)
        replace_json(ops / 'queue.json', queue_state('submitted', str(run_root)))
        environment = dict(os.environ, PYTHONUNBUFFERED='1',
                           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        with (ops / 'worker.log').open('x', encoding='utf-8') as log:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), 'worker', '--ops-dir', str(ops)],
                cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        info['worker_pid'] = process.pid
        replace_json(ops / 'launch.json', info)
    except BaseException as error:
        # Once a worker exists, only that worker may issue the terminal event.
        if process is None and (ops / 'launch.json').is_file():
            failure = {'status': 'failed', 'stage': 'S13', 'finished_at': now(),
                       'ops': str(ops),
                       'root': str(run_root), 'run_root': str(run_root),
                       'source_revision': source_commit,
                       'error': f'Launch initialization failed: {type(error).__name__}: {error}',
                       'traceback': traceback.format_exc()}
            try:
                replace_json(ops / 'queue.json', queue_state('failed', str(run_root),
                             detail=failure['error']))
            except OSError:
                pass
            write_new_json(ops / 'terminal.json', failure)
        raise
    print(json.dumps({'status': 'submitted', 'pid': process.pid,
                      'run_root': str(run_root), 'terminal_event': str(ops / 'terminal.json')}))


def collect_resource_record(history_path, fold_log, best_epoch):
    """Bind formal training resources to the selected fold without inference."""
    with history_path.open(encoding='utf-8-sig', newline='') as stream:
        history = list(csv.DictReader(stream))
    selected = [row for row in history if row.get('epoch') == str(best_epoch)]
    require(len(selected) == 1, f'{history_path}: best epoch missing from training history')
    train_f1, val_f1 = (float(selected[0][key]) for key in ('train_f1', 'val_f1'))
    require(math.isfinite(train_f1) and math.isfinite(val_f1),
            f'{history_path}: nonfinite train/validation F1 at best epoch')
    log_text = fold_log.read_text(encoding='utf-8', errors='replace')
    epoch_seconds = {int(epoch): float(seconds) for epoch, seconds in re.findall(
        r'Epoch\s+(\d+)\s+wall time:\s*([0-9]+(?:\.[0-9]+)?)s', log_text)}
    inference_seconds = re.findall(
        r'Final test evaluation took\s+([0-9]+(?:\.[0-9]+)?)s', log_text)
    training_seconds = re.findall(
        r'Training completed in\s+([0-9]+(?:\.[0-9]+)?)s', log_text)
    gpu_peaks = [float(value) for value in re.findall(
        r'peak_allocated:\s*([0-9]+(?:\.[0-9]+)?)MB', log_text)]
    require(len(epoch_seconds) == len(history) and len(inference_seconds) == 1
            and len(training_seconds) == 1 and gpu_peaks,
            f'{fold_log}: incomplete formal timing or GPU memory record')
    return {
        'training_history': str(history_path),
        'training_history_sha256': sha(history_path),
        'best_epoch_train_f1': train_f1,
        'best_epoch_val_f1': val_f1,
        'best_epoch_train_minus_val_f1': train_f1 - val_f1,
        'epoch_wall_seconds': epoch_seconds,
        'mean_epoch_wall_seconds': sum(epoch_seconds.values()) / len(epoch_seconds),
        'test_inference_seconds': float(inference_seconds[0]),
        'total_training_seconds': float(training_seconds[0]),
        'peak_gpu_allocated_mb': max(gpu_peaks),
    }


def register_outputs(info, ops):
    sys.path.insert(0, str(ROOT / 'graph_transform'))
    import torch
    from models.graph_transformer import GraphTransformer
    from models.utils import build_model_config
    from train_5fold import _stable_fold_seed, make_fold_config

    root = Path(info['run_root'])
    require(info['ops'] == info['ops_root'] == str(ops)
            and info['root'] == info['run_root'] == str(root)
            and Path(info['source_root']).resolve() == ROOT.resolve(),
            'Stage 13 launch/ops/run-root binding mismatch')
    status = read_json(root / 'status.json')
    manifest = read_json(root / 'manifest.json')
    require(status['status'] == 'completed' and len(manifest['arms']) == 1,
            'Budget-reduced batch did not finish the sole C candidate arm')
    require(status['root'] == manifest['root'] == str(root), 'Run-root binding mismatch')
    require(Path(manifest['repo_root']).resolve() == ROOT
            and status.get('gitrev') == manifest.get('gitrev')
            and status.get('gitrev') in (None, info['source_revision'])
            and len(status['arms']) == 1
            and all(item['status'] == 'completed' for item in status['arms']),
            'Batch source snapshot or arm completion record mismatch')
    preserved = check_preserved_baseline()
    require(info['preserved_baseline'] == preserved,
            'Reused single-k5 baseline changed after launch')
    old_launch = read_json(PRESERVED_OPS / 'launch.json')
    old_manifest = read_json(PRESERVED_ROOT / 'manifest.json')
    registry = {'status': 'completed_unreviewed', 'stage': 'S13',
                'ops': str(ops), 'root': str(root),
                'created_at': info['created_at'],
                'source_root': info['source_root'],
                'source_revision': info['source_revision'],
                'source_tree_sha256': info['source_tree_sha256'],
                'source_files_sha256': info['source_files_sha256'],
                'run_root': str(root), 'manifest_sha256': sha(root / 'manifest.json'),
                'status_sha256': sha(root / 'status.json'),
                'summary': str(root / 'summary/summary.json'),
                'summary_sha256': sha(root / 'summary/summary.json'),
                'arms': {}, 'D_alias_of_A': False, 'aliases': ALIASES,
                'preserved_baseline': preserved,
                'required_independent_metrics': ['bond_f1', 'bond_accuracy',
                    'roc_auc', 'pr_auc', 'mcc', 'spectrum_precision',
                    'spectrum_recall', 'spectrum_f1_harmonic', 'q_brier',
                    'ece', 'q_seq', 'top10', 'top20']}
    audit_rows = []
    smoke_folds = read_json(ops / 'smoke.json')['fold_sources']
    training_arms = (
        ('S13-A', old_manifest['arms'][0], PRESERVED_ROOT,
         old_launch['fold_data_dir'], BASELINE_CONFIG, 1),
        ('S13-C', manifest['arms'][0], root,
         info['fold_data_dir'], CONFIGS['S13-C'], 1),
    )
    for arm_id, manifest_arm, arm_batch_root, fold_data_dir, arm_config, index in training_arms:
        arm_root = arm_batch_root / 'arms' / f'{index:02d}_{arm_config.stem}'
        snapshot = Path(manifest_arm['config_snapshot'])
        expected_config_sha = (preserved['baseline_config_sha256']
                               if arm_id == 'S13-A' else info['configs'][arm_id])
        require(manifest_arm['config_sha256'] == expected_config_sha
                and sha(snapshot) == expected_config_sha
                and snapshot.resolve() == (arm_root / 'config.yaml').resolve()
                and Path(manifest_arm['cv_root']).resolve() == (arm_root / 'cv').resolve()
                and Path(manifest_arm['work_root']).resolve() == (arm_root / 'logs').resolve(),
                f'{arm_id}: batch used a different config or output namespace')
        base_config = yaml.safe_load(snapshot.read_text(encoding='utf-8'))
        metrics_path = Path(manifest_arm['metrics_csv'])
        with metrics_path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == 5 and {row['fold_id'] for row in rows} == set(FOLDS),
                f'{arm_id}: missing or duplicate fold')
        records = {}
        for row in rows:
            fold = row['fold_id']
            folder = Path(manifest_arm['cv_root']) / f'fold_{fold}'
            config = folder / 'config.yaml'
            checkpoints = list(folder.glob('checkpoints/**/best_model.pt'))
            predictions = list(folder.glob('preds/**/latest_test.pred.csv'))
            metrics = list(folder.glob('metrics/**/latest_test_metric.csv'))
            provenance_paths = list(folder.glob('checkpoints/**/row_validation_provenance.json'))
            require(len(checkpoints) == len(predictions) == len(metrics) == 1,
                    f'{arm_id}/{fold}: ambiguous checkpoint, prediction, or metric')
            require(len(provenance_paths) == 1,
                    f'{arm_id}/{fold}: ambiguous validation split provenance')
            runtime = yaml.safe_load(config.read_text(encoding='utf-8'))
            expected_runtime, _, _ = make_fold_config(
                base_config, fold, fold_data_dir, manifest_arm['cv_root'],
                _stable_fold_seed(42, fold))
            require(runtime == expected_runtime,
                    f'{arm_id}/{fold}: runtime config differs from the frozen fold recipe')
            provenance = read_json(provenance_paths[0])
            require(int(runtime['data']['max_seq_len']) == 32,
                    f'{arm_id}/{fold}: formal loader max_seq_len changed')
            train_csv = source_path(provenance['split_train_csv'])
            val_csv = source_path(provenance['split_val_csv'])
            test_csv = source_path(runtime['data']['test_csv_path'])
            folded_source = source_path(runtime['data']['train_csv_path'])
            validation_source = source_path(runtime['data']['validation_source_csv_path'])
            target_manifest = source_path(runtime['data']['four_state_targets_manifest'])
            require(sha(train_csv) == smoke_folds[fold]['train_csv_sha256']
                    and sha(val_csv) == smoke_folds[fold]['val_csv_sha256']
                    and sha(test_csv) == smoke_folds[fold]['test_csv_sha256']
                    and sha(folded_source) == smoke_folds[fold]['folded_training_source_sha256']
                    and sha(validation_source) == smoke_folds[fold]['validation_source_sha256']
                    and sha(target_manifest) == smoke_folds[fold]['target_manifest_sha256']
                    and source_path(provenance['source_path']) == validation_source
                    and provenance['source_sha256'] == smoke_folds[fold]['validation_source_sha256']
                    and provenance['folded_source_sha256'] == smoke_folds[fold]['folded_training_source_sha256']
                    and provenance['outer_test_used_for_internal_validation'] is False,
                    f'{arm_id}/{fold}: source or split SHA differs from smoke')
            four_state_paths = list(folder.glob('checkpoints/**/four_state_provenance.json'))
            require(len(four_state_paths) == 1
                    and read_json(four_state_paths[0])['manifest_sha256'] == sha(target_manifest),
                    f'{arm_id}/{fold}: four-state target manifest binding failed')
            saved = torch.load(checkpoints[0], map_location='cpu', weights_only=False)
            expected_mode = {'S13-A': 'single_k5', 'S13-B': 'multiscale_357_c32',
                             'S13-C': 'multiscale_357'}[arm_id]
            require(runtime['model']['node_conv_mode'] == expected_mode
                    and saved['config']['node_conv_mode'] == expected_mode
                    and saved['config']['state_feature_mask'] == [True, True, False]
                    and saved['config']['env_feature_mask'] == [True, False],
                    f'{arm_id}/{fold}: checkpoint architecture or feature mask changed')
            expected_rng = {
                'training_seed': int(row['seed']),
                'validation_split_seed': int(provenance['split_seed']),
                'hidden_dim': 256, 'bond_head_type': 'four_state_mlp',
                'theory_fusion_stage': 'late',
                'four_state_conditional_weight': 1.,
                'four_state_union_brier_weight': 0.}
            require(saved.get('extra_state', {}).get('training_rng_protocol') == expected_rng
                    and int(saved['epoch']) + 1 == int(row['best_epoch']),
                    f'{arm_id}/{fold}: checkpoint seed, split, loss, or selected epoch mismatch')
            effective = dict(runtime, model=saved['config'])
            with torch.random.fork_rng(devices=[]):
                model = GraphTransformer(build_model_config(effective)).cpu()
            model.load_state_dict(saved['model_state_dict'], strict=True)
            require(all(torch.isfinite(parameter).all().item()
                        for parameter in model.parameters()),
                    f'{arm_id}/{fold}: nonfinite selected checkpoint parameter')
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            del saved, model
            prediction = predictions[0]
            with prediction.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                require({'true', 'pred', 'pred_prob', 'evaluation_id',
                         'checkpoint_path', 'prediction_export_source',
                         'prediction_probability_precision'} <= set(reader.fieldnames or []),
                        f'{arm_id}/{fold}: raw prediction fields missing')
                identities = set()
                prediction_rows = 0
                for prediction_row in reader:
                    identities.add((prediction_row['evaluation_id'],
                                    prediction_row['checkpoint_path'],
                                    prediction_row['prediction_export_source'],
                                    prediction_row['prediction_probability_precision']))
                    prediction_rows += 1
            require(prediction_rows > 0, f'{arm_id}/{fold}: empty raw predictions')
            require(len(identities) == 1,
                    f'{arm_id}/{fold}: mixed evaluation IDs or export methods')
            evaluation_id, checkpoint_path, export_source, precision = next(iter(identities))
            require(Path(checkpoint_path).resolve() == checkpoints[0].resolve()
                    and export_source == 'same_pass_evaluate_logits'
                    and precision == 'float32_roundtrip_9_significant_digits',
                    f'{arm_id}/{fold}: prediction CSV is not same-pass float32 output of best checkpoint')
            archived_pred = prediction.with_name(
                f'{prediction.stem}__{evaluation_id}{prediction.suffix}')
            archived_metric = metrics[0].with_name(
                f'{metrics[0].stem}__{evaluation_id}{metrics[0].suffix}')
            require(archived_pred.is_file() and archived_metric.is_file()
                    and sha(archived_pred) == sha(prediction)
                    and sha(archived_metric) == sha(metrics[0]),
                    f'{arm_id}/{fold}: evaluation_id archive binding failed')
            require(Path(row['metric_csv_path']).resolve() == metrics[0].resolve()
                    and (not row.get('pred_csv_path')
                         or Path(row['pred_csv_path']).resolve() == prediction.resolve()),
                    f'{arm_id}/{fold}: summary/raw output path mismatch')
            require(int(row['seed']) == int(runtime['experiment']['seed'])
                    and int(provenance['split_seed']) == int(runtime['training']['validation_split_seed']),
                    f'{arm_id}/{fold}: seed metadata mismatch')
            fold_logs = list(Path(manifest_arm['work_root']).glob(f'fold{fold}_gpu*.log'))
            require(len(fold_logs) == 1, f'{arm_id}/{fold}: missing or ambiguous fold log')
            history_paths = list(folder.glob('checkpoints/**/plots/training_history.csv'))
            require(len(history_paths) == 1, f'{arm_id}/{fold}: missing or ambiguous training history')
            resources = collect_resource_record(history_paths[0], fold_logs[0],
                                                int(row['best_epoch']))
            records[fold] = {
                'training_seed': int(row['seed']),
                'best_epoch': int(row['best_epoch']),
                'runtime_config': str(config), 'runtime_config_sha256': sha(config),
                'checkpoint': str(checkpoints[0]),
                'checkpoint_sha256': sha(checkpoints[0]),
                'parameter_count': parameter_count,
                'prediction_csv': str(prediction),
                'prediction_sha256': sha(prediction),
                'prediction_rows': prediction_rows,
                'metric_csv': str(metrics[0]), 'metric_sha256': sha(metrics[0]),
                'evaluation_id': evaluation_id,
                'prediction_export_source': export_source,
                'prediction_probability_precision': precision,
                'max_seq_len': int(runtime['data']['max_seq_len']),
                'fold_log': str(fold_logs[0]), 'fold_log_sha256': sha(fold_logs[0]),
                'resources': resources,
                'train_csv': str(train_csv), 'train_csv_sha256': sha(train_csv),
                'val_csv': str(val_csv), 'val_csv_sha256': sha(val_csv),
                'test_csv': str(test_csv), 'test_csv_sha256': sha(test_csv),
                'folded_training_source': str(folded_source),
                'folded_training_source_sha256': sha(folded_source),
                'validation_source_csv': str(validation_source),
                'validation_source_sha256': sha(validation_source),
                'target_manifest': str(target_manifest),
                'target_manifest_sha256': sha(target_manifest),
                'four_state_provenance': str(four_state_paths[0]),
                'four_state_provenance_sha256': sha(four_state_paths[0]),
            }
            audit_rows.append({
                'arm': arm_id, 'fold': fold, 'seed': int(row['seed']),
                'validation_split_seed': int(provenance['split_seed']),
                'max_seq_len': int(runtime['data']['max_seq_len']),
                'best_epoch': int(row['best_epoch']),
                'pred_csv': str(prediction), 'pred_csv_sha256': sha(prediction),
                'test_csv': str(test_csv), 'test_csv_sha256': sha(test_csv),
                'metric_csv': str(metrics[0]), 'metric_csv_sha256': sha(metrics[0]),
                'checkpoint': str(checkpoints[0]),
                'checkpoint_sha256': sha(checkpoints[0]),
                'parameter_count': parameter_count,
                'config': str(config), 'config_sha256': sha(config),
                'train_csv': str(train_csv), 'train_csv_sha256': sha(train_csv),
                'val_csv': str(val_csv), 'val_csv_sha256': sha(val_csv),
                'folded_training_source': str(folded_source),
                'folded_training_source_sha256': sha(folded_source),
                'validation_source_csv': str(validation_source),
                'validation_source_sha256': sha(validation_source),
                'target_manifest': str(target_manifest),
                'target_manifest_sha256': sha(target_manifest),
                'four_state_provenance': str(four_state_paths[0]),
                'four_state_provenance_sha256': sha(four_state_paths[0]),
                'evaluation_id': evaluation_id,
                'prediction_export_source': export_source,
                'prediction_probability_precision': precision,
                'fold_log': str(fold_logs[0]), 'fold_log_sha256': sha(fold_logs[0]),
                'resources': resources,
            })
        registry['arms'][arm_id] = {
            'config': manifest_arm['config_snapshot'],
            'config_sha256': manifest_arm['config_sha256'],
            'fivefold_metrics': str(metrics_path),
            'fivefold_metrics_sha256': sha(metrics_path),
            'run_log': manifest_arm['run_log'],
            'folds': records,
        }
    parameter_counts = {arm: {row['parameter_count'] for row in audit_rows
                              if row['arm'] == arm} for arm in ('S13-A', 'S13-C')}
    require(all(len(counts) == 1 for counts in parameter_counts.values())
            and parameter_counts['S13-A'] == parameter_counts['S13-C'],
            'Formal checkpoint parameter counts violate the declared A/C matching')
    audit_manifest = {
        'stage': 'S13', 'ops': str(ops), 'root': str(root),
        'budget_profile': 'minimal_s13_c64_5fold_v1',
        'source_root': info['source_root'],
        'source_commit': info['source_revision'],
        'source_tree_sha256': info['source_tree_sha256'],
        'source_files_sha256': info['source_files_sha256'],
        'arms': audit_rows,
        'preserved_baseline': preserved,
        'stage12_audit': {
            'path': info['stage12_audit'],
            'sha256': info['stage12_audit_sha256'],
            'root': info['stage12_root'],
            'launch_sha256': info['stage12_launch_sha256'],
        },
        'aliases': ALIASES,
        'required_independent_metrics': registry['required_independent_metrics'],
        'audit_status': 'pending',
    }
    write_new_or_matching_json(ops / 'registry.json', registry)
    write_new_or_matching_json(ops / 'audit_manifest.json', audit_manifest)
    return registry


def worker(args):
    ops = pod_only(args.ops_dir)
    info = read_json(ops / 'launch.json')
    phase = 'frozen_launch_check'
    try:
        replace_json(ops / 'queue.json',
                     queue_state('running', info['run_root']))
        run_root = Path(info['run_root']).resolve()
        require(run_root.parent == STAGE13_RUN_BASE
                and run_root.name.startswith('dbond_s13_minimal_c64_20261026_'),
                'Stage 13 run root escaped the dedicated checkpoint namespace')
        require(info['ops'] == info['ops_root'] == str(ops)
                and info['root'] == info['run_root']
                and Path(info['source_root']).resolve() == ROOT.resolve()
                and info['status'] == 'submitted'
                and info['source_files_sha256'] == source_fingerprint()
                and info['source_tree_sha256'] == fingerprint_sha(source_fingerprint())
                and info['smoke_sha256'] == sha(ops / 'smoke.json')
                and info['stage12_audit_sha256'] == sha(info['stage12_audit'])
                and info['stage12_launch_sha256'] == sha(STAGE12_OPS / 'launch.json')
                and (info['stage12_queue_sha256'] is None
                     or info['stage12_queue_sha256'] == sha(STAGE12_OPS / 'queue.json'))
                and info['batch_runner_sha256'] == sha(info['batch_runner']),
                'Frozen launch source, smoke, or predecessor changed')
        check_fold_sources(read_json(ops / 'smoke.json')['fold_sources'])
        require(info['preserved_baseline'] == check_preserved_baseline(),
                'Preserved baseline changed during C-only run')
        phase = 'training'
        command = [sys.executable, str(SCRIPTS / 'run_architecture_batch.py'),
                   '--configs', *(str(path) for path in CONFIGS.values()),
                   '--fold_data_dir', info['fold_data_dir'],
                   '--gpus', info['gpus'], '--run_root', info['run_root']]
        require(run_logged(command, ops / 'training.log') == 0,
                'Stage 13 batch failed; inspect the independent batch status event')
        require(info['source_files_sha256'] == source_fingerprint(),
                'Frozen Stage 13 source changed during training')
        check_fold_sources(read_json(ops / 'smoke.json')['fold_sources'])
        phase = 'registration'
        register_outputs(info, ops)
        phase = 'terminal_event'
        replace_json(ops / 'queue.json',
                     queue_state('completed', info['run_root'],
                                 detail='Independent metric audit pending'))
        write_new_json(ops / 'terminal.json',
                       {'status': 'completed', 'stage': 'S13', 'finished_at': now(),
                        'ops': str(ops),
                        'root': info['root'], 'run_root': info['run_root'],
                        'source_root': info['source_root'],
                        'source_revision': info['source_revision'],
                        'registry': str(ops / 'registry.json'),
                        'registry_sha256': sha(ops / 'registry.json'),
                        'audit_manifest': str(ops / 'audit_manifest.json'),
                        'audit_manifest_sha256': sha(ops / 'audit_manifest.json'),
                        'audit_status': 'pending'})
    except BaseException as error:
        replace_json(ops / 'queue.json',
                     queue_state('failed', info['run_root'],
                                 detail=f'{type(error).__name__}: {error}'))
        write_new_json(ops / 'terminal.json',
                       {'status': 'failed', 'stage': 'S13', 'finished_at': now(),
                        'ops': str(ops),
                        'root': info['root'], 'run_root': info['run_root'],
                        'source_root': info['source_root'],
                        'source_revision': info['source_revision'],
                        'failure_phase': phase,
                        'error': f'{type(error).__name__}: {error}',
                        'traceback': traceback.format_exc(),
                        'training_log': str(ops / 'training.log')})
        raise


def finalize(args):
    """Recover registration after completed training; never rerun or edit the batch."""
    ops = pod_only(args.ops_dir)
    info = read_json(ops / 'launch.json')
    failed = read_json(ops / 'terminal.json')
    root = Path(info['run_root']).resolve()
    require(failed.get('status') == 'failed'
            and failed.get('failure_phase') in ('registration', 'terminal_event')
            and failed.get('stage') == info.get('stage') == 'S13'
            and failed.get('ops') == info.get('ops') == info.get('ops_root') == str(ops)
            and failed.get('root') == failed.get('run_root')
            == info.get('root') == info.get('run_root') == str(root)
            and failed.get('source_revision') == info.get('source_revision')
            and failed.get('source_root') == info.get('source_root')
            and root.parent == STAGE13_RUN_BASE
            and root.name.startswith('dbond_s13_minimal_c64_20261026_'),
            'Finalize requires a completed-training Stage 13 registration failure')
    require(Path(info['source_root']).resolve() == ROOT.resolve()
            and info['source_files_sha256'] == source_fingerprint()
            and info['source_tree_sha256'] == fingerprint_sha(source_fingerprint())
            and info['smoke_sha256'] == sha(ops / 'smoke.json')
            and info['batch_runner_sha256'] == sha(info['batch_runner'])
            and info['stage12_audit_sha256'] == sha(info['stage12_audit'])
            and info['stage12_launch_sha256'] == sha(STAGE12_OPS / 'launch.json')
            and (info['stage12_queue_sha256'] is None
                 or info['stage12_queue_sha256'] == sha(STAGE12_OPS / 'queue.json'))
            and git_revision() in (None, info['source_revision']),
            'Frozen Stage 13 source, smoke, or predecessor changed before finalize')
    require(info['preserved_baseline'] == check_preserved_baseline(),
            'Preserved fivefold baseline changed before registration recovery')
    check_fold_sources(read_json(ops / 'smoke.json')['fold_sources'])
    batch_status = read_json(root / 'status.json')
    require(batch_status.get('status') == 'completed'
            and batch_status.get('root') == str(root),
            'Finalize cannot recover a failed or incomplete training batch')
    attempts = ops / 'finalize_attempts'
    attempts.mkdir(exist_ok=True)
    attempt = attempts / f'attempt_{uuid.uuid4().hex}'
    attempt.mkdir()
    write_new_json(attempt / 'prior_terminal.json', failed)
    if (ops / 'queue.json').is_file():
        write_new_json(attempt / 'prior_queue.json', read_json(ops / 'queue.json'))
    try:
        register_outputs(info, ops)
        checks = {'status': 'passed', 'stage': 'S13', 'at': now(),
                  'mode': 'registration_only_no_retraining',
                  'ops': str(ops), 'root': str(root),
                  'source_root': info['source_root'],
                  'source_revision': info['source_revision'],
                  'batch_status_sha256': sha(root / 'status.json'),
                  'batch_manifest_sha256': sha(root / 'manifest.json'),
                  'prior_terminal': str(attempt / 'prior_terminal.json'),
                  'prior_terminal_sha256': sha(attempt / 'prior_terminal.json'),
                  'registry_sha256': sha(ops / 'registry.json'),
                  'audit_manifest_sha256': sha(ops / 'audit_manifest.json')}
        write_new_json(attempt / 'checks_passed.json', checks)
        write_new_json(attempt / 'receipt.json', checks)
        replace_json(ops / 'queue.json',
                     queue_state('completed', str(root),
                                 detail='Recovered registration; independent metric audit pending'))
        replace_json(ops / 'terminal.json',
                     {'status': 'completed', 'stage': 'S13', 'finished_at': now(),
                      'ops': str(ops), 'root': str(root), 'run_root': str(root),
                      'source_root': info['source_root'],
                      'source_revision': info['source_revision'],
                      'registry': str(ops / 'registry.json'),
                      'registry_sha256': checks['registry_sha256'],
                      'audit_manifest': str(ops / 'audit_manifest.json'),
                      'audit_manifest_sha256': checks['audit_manifest_sha256'],
                      'audit_status': 'pending',
                      'recovered_from': checks['prior_terminal'],
                      'recovered_from_sha256': checks['prior_terminal_sha256'],
                      'finalize_receipt': str(attempt / 'receipt.json'),
                      'finalize_receipt_sha256': sha(attempt / 'receipt.json')})
    except BaseException as error:
        failure = {'status': 'failed', 'stage': 'S13', 'at': now(),
                   'mode': 'registration_only_no_retraining',
                   'root': str(root), 'source_revision': info['source_revision'],
                   'error': f'{type(error).__name__}: {error}',
                   'traceback': traceback.format_exc()}
        replace_json(attempt / 'receipt.json', failure)
        raise
    print(json.dumps({'status': 'completed', 'mode': 'registration_only_no_retraining',
                      'terminal_event': str(ops / 'terminal.json'),
                      'finalize_receipt': str(attempt / 'receipt.json')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('smoke', 'launch', 'worker', 'finalize'))
    parser.add_argument('--ops-dir', required=True)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--gpus')
    parser.add_argument('--stage12-audit')
    parser.add_argument('--source-commit')
    arguments = parser.parse_args()
    {'smoke': smoke, 'launch': launch, 'worker': worker,
     'finalize': finalize}[arguments.action](arguments)
