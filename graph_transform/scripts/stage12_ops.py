"""Stage 12 pod-only queue: preflight, one launch, terminal event, audit manifest.

This namespace is independent of all historical structural_queue/Stage 11 ops.
The worker records completion/failure events; consumers need not poll epochs.
Finalization is read-only with respect to training artifacts and leaves metric
recomputation/paired review to the independent audit_core consumer.
"""

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'graph_transform/scripts'
sys.path[:0] = [str(ROOT / 'graph_transform'), str(SCRIPTS)]
from train_5fold import _stable_fold_seed

ARMS = ('a_legacy', 'b_wide_mlp', 'c_equal', 'd_softmax_gate',
        'e_softmax_gate_residual')
MODES = ('legacy', 'wide_mlp', 'equal', 'softmax_gate',
         'softmax_gate_residual')
FOLDS = ('1222', '2252', '3514', '6072', '9075')


def now():
    return datetime.now(timezone.utc).isoformat()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def atomic(path, value):
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def config(arm):
    return ROOT / f'graph_transform/config/pre_synthesis_stage12_fusion_{arm}.yaml'


def absolute(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def sources():
    names = sorted((ROOT / 'graph_transform').rglob('*.py'))
    names += [config(arm) for arm in ARMS]
    return {str(path.relative_to(ROOT)): sha(path) for path in names
            if '__pycache__' not in path.parts}


def pod_only():
    require(sys.platform.startswith('linux') and str(ROOT).startswith('/mnt/pvc/'),
            'Stage 12 ops run only in the authorized pod source snapshot')


def preflight(args):
    pod_only()
    ops = Path(args.ops).resolve()
    ops.mkdir(parents=True, exist_ok=True)
    require(not (ops / 'smoke.json').exists(), 'Preflight receipt exists; do not overwrite')
    bound = sources()
    command = [sys.executable, str(SCRIPTS / 'stage12_preflight.py'),
               '--gpu', str(args.gpu), '--namespace', str(ops),
               '--output', str(ops / 'smoke.json')]
    preflight_log = ops / f'preflight_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.log'
    with preflight_log.open('x', encoding='utf-8') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    require(result.returncode == 0 and (ops / 'smoke.json').is_file(),
            f'Preflight failed; see {preflight_log}')
    smoke = read(ops / 'smoke.json')
    require(smoke.get('status') == 'passed' and bound == sources(),
            'Source changed during preflight')
    write_new(ops / 'source_manifest.json', bound)
    print(json.dumps({'status': 'smoke_passed', 'ops': str(ops),
                      'smoke_sha256': sha(ops / 'smoke.json')}))


def launch(args):
    pod_only()
    ops = Path(args.ops).resolve()
    root = Path(args.root).resolve()
    require((ops / 'smoke.json').is_file()
            and read(ops / 'smoke.json').get('status') == 'passed',
            'Smoke must pass before launch')
    require(read(ops / 'source_manifest.json') == sources(),
            'Source differs from preflight snapshot')
    require(args.source_commit and re.fullmatch(r'[0-9a-f]{40}', args.source_commit),
            'Pass the frozen source commit SHA')
    actual_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                            cwd=ROOT, text=True).strip()
    require(actual_commit == args.source_commit,
            'Frozen bundle-clone HEAD differs from declared source commit')
    require(not root.exists(), 'Run root exists; refuse repeated submission')
    devices = [int(token) for token in args.gpus.split(',')]
    require(devices and len(devices) == len(set(devices)) and all(0 <= gpu <= 4 for gpu in devices),
            'Invalid GPU allocation')
    write_new(ops / 'launch.claim', {'pid': os.getpid(), 'claimed_at': now(),
                                     'root': str(root)})
    root.mkdir(parents=True, exist_ok=False)
    arm_rows = []
    for arm, mode in zip(ARMS, MODES):
        folder = root / 'arms' / arm
        folder.mkdir(parents=True)
        snapshot = folder / 'config.yaml'
        snapshot.write_bytes(config(arm).read_bytes())
        arm_rows.append({'arm': arm, 'mode': mode, 'config': str(snapshot),
                         'config_sha256': sha(snapshot), 'cv_root': str(folder / 'cv'),
                         'log_root': str(folder / 'logs'), 'status': 'planned'})
    launch_record = {
        'stage': 'S12', 'status': 'submitted', 'created_at': now(),
        'ops': str(ops), 'root': str(root), 'source_root': str(ROOT),
        'source_commit': args.source_commit,
        'source_manifest_sha256': sha(ops / 'source_manifest.json'),
        'smoke_sha256': sha(ops / 'smoke.json'), 'gpus': devices,
        'folds': list(FOLDS), 'base_seed': 42, 'split_base_seed': 42,
        'fold_seeds': {fold: {'training': _stable_fold_seed(42, fold),
                              'split': _stable_fold_seed(42, fold)} for fold in FOLDS},
        'arms': arm_rows,
        'checkpoint_selection': 'internal full-spectrum validation max_f1',
        'prediction_export': 'same_pass_evaluate_logits, float32 roundtrip 9 significant digits',
    }
    write_new(ops / 'launch.json', launch_record)
    write_new(root / 'manifest.json', launch_record)
    try:
        env = dict(os.environ, PYTHONUNBUFFERED='1')
        with (ops / 'worker.log').open('x', encoding='utf-8') as log:
            worker = subprocess.Popen([sys.executable, __file__, 'worker',
                '--ops', str(ops)], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        launch_record['worker_pid'] = worker.pid
        atomic(ops / 'launch.json', launch_record)
        atomic(root / 'manifest.json', launch_record)
    except BaseException:
        write_new(ops / 'terminal.json', {'status': 'failed', 'time': now(),
            'root': str(root), 'error': traceback.format_exc()})
        raise
    print(json.dumps({'status': 'submitted', 'worker_pid': worker.pid,
                      'root': str(root), 'ops': str(ops)}))


def worker(args):
    pod_only()
    ops = Path(args.ops).resolve()
    info = read(ops / 'launch.json')
    root = Path(info['root'])
    status_path = ops / 'status.json'
    status = {'stage': 'S12', 'status': 'running', 'time': now(),
              'root': str(root), 'worker_pid': os.getpid(), 'arms': []}
    atomic(status_path, status)
    failures = []
    try:
        require(read(ops / 'source_manifest.json') == sources(),
                'Source changed before worker start')
        for arm in info['arms']:
            label = arm['arm']
            require(read(ops / 'source_manifest.json') == sources(),
                    f'Source changed before {label}')
            require(sha(arm['config']) == arm['config_sha256'],
                    f'{label}: config snapshot changed')
            cv = Path(arm['cv_root'])
            logs = Path(arm['log_root'])
            require(not cv.exists() and not logs.exists(),
                    f'{label}: existing outputs; refuse duplicate training')
            logs.mkdir(parents=True)
            status['current_arm'] = label
            status['current_started_at'] = now()
            atomic(status_path, status)
            command = [sys.executable, str(SCRIPTS / 'train_5fold_parallel.py'),
                       '--config', arm['config'], '--fold_data_dir',
                       str(ROOT / 'dataset/5fold_folded'), '--gpus',
                       ','.join(map(str, info['gpus'])), '--cv_root', str(cv),
                       '--work_root', str(logs)]
            with (root / 'arms' / label / 'run.log').open('x', encoding='utf-8') as stream:
                process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                           stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                status['child_pid'] = process.pid
                atomic(status_path, status)
                result = process.wait()
            status['child_pid'] = None
            row = {'arm': label, 'status': 'completed' if result == 0 else 'failed',
                   'returncode': result, 'finished_at': now(),
                   'config_sha256': arm['config_sha256'], 'cv_root': str(cv),
                   'run_log': str(root / 'arms' / label / 'run.log')}
            if result == 0 and (cv / '5fold_metrics.csv').is_file():
                row['metrics_sha256'] = sha(cv / '5fold_metrics.csv')
            else:
                row['status'] = 'failed'
                row['error'] = ('Training returned nonzero' if result != 0 else
                                'Five-fold metrics missing after zero exit')
                failures.append(label)
            status['arms'].append(row)
            atomic(status_path, status)
        require(read(ops / 'source_manifest.json') == sources(),
                'Source changed during training')
        terminal = {'stage': 'S12', 'status': 'completed' if not failures else 'failed',
                    'time': now(), 'root': str(root), 'worker_pid': os.getpid(),
                    'arms': status['arms'], 'failed_arms': failures,
                    'source_manifest_sha256': info['source_manifest_sha256']}
    except BaseException:
        terminal = {'stage': 'S12', 'status': 'failed', 'time': now(),
                    'root': str(root), 'worker_pid': os.getpid(),
                    'arms': status['arms'], 'failed_arms': failures,
                    'error': traceback.format_exc()}
    write_new(ops / 'terminal.json', terminal)
    status.update(status=terminal['status'], current_arm=None, child_pid=None,
                  finished_at=terminal['time'])
    atomic(status_path, status)
    return 0 if terminal['status'] == 'completed' else 1


def unique(folder, pattern):
    paths = list(folder.glob(pattern))
    require(len(paths) == 1, f'Expected one {pattern} under {folder}; got {len(paths)}')
    return paths[0].resolve()


def verify_prediction(path, checkpoint):
    require(path.is_file(), f'Missing prediction CSV: {path}')
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        rows = 0
        eval_id = None
        for row in reader:
            rows += 1
            require(row.get('prediction_export_source') == 'same_pass_evaluate_logits'
                    and row.get('prediction_probability_precision') ==
                    'float32_roundtrip_9_significant_digits',
                    'Prediction not exported from formal same-pass logits')
            require(Path(row.get('checkpoint_path', '')).resolve() == checkpoint
                    and float(row.get('threshold', 'nan')) == 0.5,
                    'Prediction checkpoint/threshold binding differs')
            value = row.get('evaluation_id')
            require(value and (eval_id is None or value == eval_id),
                    'Mixed or missing evaluation ID')
            eval_id = value
        require(rows > 0, 'Empty prediction CSV')
    require(re.fullmatch(r'\d{8}_\d{6}_.+_best_model_test', eval_id),
            'Unexpected prediction evaluation ID')
    archive = path.with_name(f'{path.stem}__{eval_id}{path.suffix}')
    require(archive.is_file() and sha(archive) == sha(path),
            'Immutable prediction archive differs from latest CSV')
    return rows, eval_id, str(archive)


def finalize(args):
    pod_only()
    import torch
    import yaml
    from models.graph_transformer import GraphTransformer
    from models.utils import build_model_config

    ops = Path(args.ops).resolve()
    info = read(ops / 'launch.json')
    terminal = read(ops / 'terminal.json')
    root = Path(info['root']).resolve()
    require(terminal['root'] == str(root) and terminal['status'] in ('completed', 'failed'),
            'Missing/mismatched terminal event')
    require(read(ops / 'source_manifest.json') == sources()
            and sha(ops / 'source_manifest.json') == info['source_manifest_sha256'],
            'Frozen source changed')
    require(sha(ops / 'smoke.json') == info['smoke_sha256'], 'Smoke receipt changed')
    require(not (ops / 'audit_manifest.json').exists(),
            'Audit manifest exists; do not overwrite')
    require(not (ops / 'checkpoint_registry.json').exists(),
            'Checkpoint registry exists; do not overwrite')
    entries = []
    failures = []
    for arm in info['arms']:
        label, mode = arm['arm'], arm['mode']
        run_status = next((entry for entry in terminal['arms'] if entry['arm'] == label), None)
        if not run_status or run_status['status'] != 'completed':
            failures.append(label)
            continue
        cv = Path(arm['cv_root']).resolve()
        require(sha(arm['config']) == arm['config_sha256'], f'{label}: config SHA changed')
        for fold in FOLDS:
            folder = cv / f'fold_{fold}'
            runtime_path = folder / 'config.yaml'
            runtime = yaml.safe_load(runtime_path.read_text(encoding='utf-8'))
            require(runtime['model']['stage12_fusion_mode'] == mode
                    and runtime['experiment']['seed'] == info['fold_seeds'][fold]['training']
                    and runtime['training']['validation_split_seed'] == info['fold_seeds'][fold]['split']
                    and runtime['training']['checkpoint_selection'] == 'max_f1'
                    and runtime['evaluation']['save_outputs'] is True
                    and runtime['evaluation']['export_from_evaluate_cache'] is True,
                    f'{label}/{fold}: runtime protocol mismatch')
            provenance_path = unique(folder, 'checkpoints/**/row_validation_provenance.json')
            provenance = read(provenance_path)
            require(provenance['outer_test_used_for_internal_validation'] is False
                    and provenance['split_seed'] == info['fold_seeds'][fold]['split'],
                    f'{label}/{fold}: split/selection provenance mismatch')
            train_csv = absolute(provenance['split_train_csv'])
            val_csv = absolute(provenance['split_val_csv'])
            source_csv = absolute(provenance['source_path'])
            require(sha(source_csv) == provenance['source_sha256'],
                    f'{label}/{fold}: validation source changed')
            checkpoint = unique(folder, 'checkpoints/**/best_model.pt')
            require(checkpoint.parent == provenance_path.parent,
                    f'{label}/{fold}: checkpoint/split provenance mismatch')
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            require(saved['config']['stage12_fusion_mode'] == mode,
                    f'{label}/{fold}: checkpoint model mode mismatch')
            require(saved['config']['state_feature_mask'] == [True, True, False]
                    and saved['config']['env_feature_mask'] == [True, False],
                    f'{label}/{fold}: checkpoint contains post-synthesis features')
            expected_rng = {'training_seed': info['fold_seeds'][fold]['training'],
                            'validation_split_seed': info['fold_seeds'][fold]['split'],
                            'hidden_dim': 256, 'bond_head_type': 'four_state_mlp',
                            'theory_fusion_stage': 'late',
                            'four_state_conditional_weight': 1.,
                            'four_state_union_brier_weight': 0.}
            require(saved.get('extra_state', {}).get('training_rng_protocol') == expected_rng,
                    f'{label}/{fold}: checkpoint RNG/loss protocol mismatch')
            effective = copy.deepcopy(runtime)
            effective['model'] = saved['config']
            with torch.random.fork_rng(devices=[]):
                model = GraphTransformer(build_model_config(effective))
            model.load_state_dict(saved['model_state_dict'], strict=True)
            require(all(torch.isfinite(parameter).all().item()
                        for parameter in model.parameters()),
                    f'{label}/{fold}: nonfinite checkpoint parameter')
            metric_csv = unique(folder, 'metrics/**/latest_test_metric.csv')
            metric_suffix = metric_csv.relative_to(folder / 'metrics').parent
            pred_csv = folder / 'preds' / metric_suffix / 'latest_test.pred.csv'
            require(len(list(folder.glob('preds/**/latest_test.pred.csv'))) == 1,
                    f'{label}/{fold}: ambiguous prediction CSV')
            rows, eval_id, archived_pred = verify_prediction(pred_csv, checkpoint)
            metric_archive = metric_csv.with_name(
                f'{metric_csv.stem}__{eval_id}{metric_csv.suffix}')
            require(metric_archive.is_file() and sha(metric_archive) == sha(metric_csv),
                    f'{label}/{fold}: metric archive differs')
            test_csv = absolute(runtime['data']['test_csv_path'])
            target_manifest = absolute(runtime['data']['four_state_targets_manifest'])
            four_state_provenance = provenance_path.with_name('four_state_provenance.json')
            require(read(four_state_provenance)['manifest_sha256'] == sha(target_manifest),
                    f'{label}/{fold}: target manifest binding mismatch')
            paths = {'pred_csv': pred_csv, 'test_csv': test_csv,
                     'metric_csv': metric_csv, 'checkpoint': checkpoint,
                     'config': runtime_path, 'train_csv': train_csv,
                     'val_csv': val_csv, 'validation_source_csv': source_csv,
                     'four_state_manifest': target_manifest,
                     'row_validation_provenance': provenance_path,
                     'four_state_provenance': four_state_provenance}
            item = {'arm': label, 'mode': mode, 'fold': fold,
                    'seed': info['fold_seeds'][fold]['training'],
                    'split_seed': info['fold_seeds'][fold]['split'],
                    'evaluation_id': eval_id, 'prediction_rows': rows,
                    'max_seq_len': runtime['data']['max_seq_len'],
                    'pred_archive': archived_pred,
                    'checkpoint_epoch': saved.get('epoch'),
                    'parameter_count': sum(p.numel() for p in model.parameters())}
            for key, path in paths.items():
                require(path.is_file(), f'{label}/{fold}: missing {key}: {path}')
                item[key] = str(path)
                item[key + '_sha256'] = sha(path)
            entries.append(item)
    require(entries, 'No completed arms to finalize')
    for fold in FOLDS:
        paired = [item for item in entries if item['fold'] == fold]
        require(len({item['test_csv_sha256'] for item in paired}) == 1
                and len({item['train_csv_sha256'] for item in paired}) == 1
                and len({item['val_csv_sha256'] for item in paired}) == 1
                and len({item['four_state_manifest_sha256'] for item in paired}) == 1,
                f'{fold}: paired data sources differ')
    manifest = {'stage': 'S12', 'schema_version': 1, 'root': str(root),
                'ops': str(ops), 'source_commit': info['source_commit'],
                'source_root': str(ROOT), 'source_manifest': str(ops / 'source_manifest.json'),
                'source_manifest_sha256': info['source_manifest_sha256'],
                'smoke': str(ops / 'smoke.json'), 'smoke_sha256': info['smoke_sha256'],
                'terminal': str(ops / 'terminal.json'),
                'terminal_sha256': sha(ops / 'terminal.json'),
                'status': 'completed_pending_audit' if not failures else 'partial_pending_audit',
                'failed_arms': failures, 'arms': entries,
                'note': 'Raw predictions and metric CSV are immutable training outputs; independent audit must recompute metrics and gates.'}
    write_new(ops / 'audit_manifest.json', manifest)
    write_new(ops / 'checkpoint_registry.json', {
        'stage': 'S12', 'source_commit': info['source_commit'],
        'entries': [{key: item[key] for key in ('arm', 'fold', 'seed', 'split_seed',
                    'checkpoint', 'checkpoint_sha256', 'checkpoint_epoch',
                    'config', 'config_sha256', 'parameter_count')} for item in entries]})
    print(json.dumps({'status': manifest['status'], 'entries': len(entries),
                      'audit_manifest': str(ops / 'audit_manifest.json')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('preflight')
    prep.add_argument('--ops', required=True)
    prep.add_argument('--gpu', type=int, default=0)
    start = sub.add_parser('launch')
    start.add_argument('--ops', required=True)
    start.add_argument('--root', required=True)
    start.add_argument('--gpus', required=True)
    start.add_argument('--source-commit', required=True)
    work = sub.add_parser('worker')
    work.add_argument('--ops', required=True)
    final = sub.add_parser('finalize')
    final.add_argument('--ops', required=True)
    args = parser.parse_args()
    try:
        return {'preflight': preflight, 'launch': launch, 'worker': worker,
                'finalize': finalize}[args.command](args) or 0
    except BaseException as error:
        print(f'Stage 12 {args.command} failed: {error}', file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
