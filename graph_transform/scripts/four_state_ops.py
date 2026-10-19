"""Pod-only paired launch and terminal analysis; never poll training epochs."""
import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'graph_transform'))
from data.row_validation import file_sha

OPS = Path('/mnt/pvc/graphtrans/four_state_ops')
TARGETS = Path('/mnt/pvc/graphtrans/four_state_targets_v1')
ARMS = ('control', 'joint')
PY = str(REPO / '.venv/bin/python')
FOLDS = ('1222', '2252', '3514', '6072', '9075')


def config(arm):
    return REPO / f'graph_transform/config/pre_synthesis_four_state_{arm}.yaml'


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def idle():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    procs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    busy = {r.split(',')[0].strip() for r in procs.splitlines() if ',' in r}
    parsed = [[v.strip() for v in row.split(',')] for row in rows.splitlines()]
    ids = [int(p[0]) for p in parsed if int(p[0]) in range(5) and p[1] not in busy and int(p[2]) < 200 and int(p[3]) < 5]
    if not ids:
        raise RuntimeError('No idle GPU; do not displace existing workloads')
    return ids, rows, procs


def smoke():
    if (OPS / 'smoke.json').exists():
        raise RuntimeError('Successful smoke already exists; do not repeat')
    ids, _, _ = idle()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(ids[0]), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    with (OPS / 'preflight.log').open('a') as log:
        subprocess.run([PY, str(REPO / 'graph_transform/scripts/test_four_state.py'), '--output', str(OPS / 'smoke.json')],
                       cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    proof = json.loads((OPS / 'smoke.json').read_text())
    proof['gitrev'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    dump(OPS / 'smoke.json', proof)
    print(json.dumps(proof))


def launch():
    if (OPS / 'launch.json').exists():
        raise RuntimeError('Batch launch already exists; refuse duplicate')
    proof = json.loads((OPS / 'smoke.json').read_text())
    if proof['status'] != 'passed' or any(file_sha(config(a)) != proof['configs'][a] for a in ARMS):
        raise RuntimeError('Missing or stale preflight proof')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    if revision != proof['gitrev']:
        raise RuntimeError('HEAD differs from tested revision')
    changes = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=REPO, text=True).splitlines()
    if not set(changes) <= {'graph_transform/scripts/ensemble_baseline_preds.py'}:
        raise RuntimeError(f'Unexpected tracked source changes: {changes}')
    preparation = json.loads((TARGETS / 'preparation.json').read_text())
    if preparation['status'] != 'passed':
        raise RuntimeError('Target preparation not passed')
    ids, rows, procs = idle()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
    root = REPO / 'checkpoints/graph_transform/four_state_batch' / stamp
    source = {str(p.relative_to(REPO)): file_sha(p) for p in (REPO / 'graph_transform').rglob('*.py') if '__pycache__' not in p.parts}
    info = {'root': str(root), 'gitrev': revision, 'gpus': ids, 'gpu_preflight': rows,
            'compute_processes': procs, 'source_sha256': source, 'configs': proof['configs'],
            'target_preparation_sha256': file_sha(TARGETS / 'preparation.json'),
            'target_manifests': {f: file_sha(TARGETS / f'fold_{f}/manifest.json') for f in FOLDS},
            'scope': 'Original 3GCN+2GAT theory/FiLM, full row internal validation in BOTH arms; folded group_uniform train; control binary vs joint four-state head'}
    dump(OPS / 'launch.json', info)
    environment = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    with (OPS / 'worker.log').open('x') as log:
        proc = subprocess.Popen([PY, __file__, 'worker'], cwd=REPO, env=environment,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    info['worker_pid'] = proc.pid
    dump(OPS / 'launch.json', info)
    print(json.dumps({k: v for k, v in info.items() if k != 'source_sha256'}))


def analyze(info):
    """Only completed, saved outputs; independent small CSV recalculation."""
    root = Path(info['root'])
    status = json.loads((root / 'status.json').read_text())
    if status['status'] != 'completed' or not all(a['returncode'] == 0 for a in status['arms']):
        raise RuntimeError('Batch is not completely successful')
    manifest = json.loads((root / 'manifest.json').read_text())
    summary = json.loads((root / 'summary/summary.json').read_text())
    if manifest['gitrev'] != info['gitrev'] or len(manifest['arms']) != 2:
        raise RuntimeError('Batch manifest/revision mismatch')
    source_changed = [p for p, h in info['source_sha256'].items() if file_sha(REPO / p) != h]
    if source_changed:
        raise RuntimeError(f'Training source changed: {source_changed}')
    data = {}
    provenance = {}
    for arm in manifest['arms']:
        if file_sha(arm['config_snapshot']) != arm['config_sha256']:
            raise RuntimeError('Config snapshot SHA mismatch')
        bound = next(s for s in summary['sources'] if s['arm'] == arm['name'])
        if file_sha(arm['metrics_csv']) != bound['sha256']:
            raise RuntimeError('Summary raw metric SHA mismatch')
        with Path(arm['metrics_csv']).open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 5 or {r['fold_id'] for r in rows} != set(FOLDS):
            raise RuntimeError('Missing or duplicate split metrics')
        out = {}
        for row in rows:
            p, r = float(row['ex_precision']), float(row['ex_recall'])
            values = {k: float(row[k]) for k in ('f1', 'accuracy', 'auc', 'q_brier')}
            values.update(ex_precision=p, ex_recall=r, ex_f1_harmonic=2*p*r/(p+r) if p+r else 0., q_seq=float(row['q_spearman_pep_seq']))
            if not all(math.isfinite(v) for v in values.values()):
                raise RuntimeError('Nonfinite result metric')
            out[row['fold_id']] = values
        data[arm['name']] = out
        provenance[arm['name']] = {}
        for fold in FOLDS:
            folder = Path(arm['cv_root']) / f'fold_{fold}'
            paths = list(folder.glob('checkpoints/**/row_validation_provenance.json'))
            if len(paths) != 1:
                raise RuntimeError(f'Missing/ambiguous full-row validation provenance for {arm["name"]}/{fold}')
            prov = json.loads(paths[0].read_text())
            if (prov['mode'] != 'full_original_training_spectra_by_internal_val_sequence'
                    or prov['outer_test_used_for_internal_validation'] is not False
                    or file_sha(prov['source_path']) != prov['source_sha256']):
                raise RuntimeError('Invalid full-row validation source proof')
            provenance[arm['name']][fold] = {k: prov[k] for k in ('val_rows', 'val_sequences', 'source_sha256', 'split_seed')}
            if file_sha(prov['split_val_csv']) != json.loads((OPS / 'smoke.json').read_text())['folds'][fold]['val_sha256']:
                raise RuntimeError('Runtime validation rows differ from preflight')
    names = list(data)
    if provenance[names[0]] != provenance[names[1]]:
        raise RuntimeError('Paired validation source/split differs')
    for entry in summary['metrics']:
        metric, arm = entry['metric'], entry['arm']
        values = [data[arm][f][metric] for f in FOLDS]
        delta = [data[arm][f][metric] - data[names[0]][f][metric] for f in FOLDS]
        sign = -1 if metric == 'q_brier' else 1
        if (abs(statistics.mean(values) - entry['mean']) > 1e-12
                or abs(statistics.stdev(values) - entry['std_ddof1']) > 1e-12
                or abs(statistics.mean(delta) - entry['paired_delta_mean']) > 1e-12
                or sum(sign*d > 1e-12 for d in delta) != entry['win_splits']):
            raise RuntimeError(f'Independent summary mismatch: {arm}/{metric}')
    metrics = {a: {r['metric']: r for r in summary['metrics'] if r['arm'] == name} for a, name in zip(ARMS, names)}
    candidate = metrics['joint']
    gates = {'delta_f1_at_least_0.002': candidate['f1']['paired_delta_mean'] >= .002,
             'f1_win_at_least_4_of_5': candidate['f1']['win_splits'] >= 4,
             'delta_auc_nonnegative': candidate['auc']['paired_delta_mean'] >= 0,
             'delta_q_brier_nonpositive': candidate['q_brier']['paired_delta_mean'] <= 0}
    out = {'status': 'completed', 'root': str(root), 'gitrev': info['gitrev'],
           'finished_at': status['finished_at'], 'metrics': metrics, 'gates': gates,
           'improvement_gate': all(gates.values()), 'paired_validation_verified': True,
           'folds': data, 'scope': 'Five overlapping repeated sequence-level 80/20 splits; exploratory paired comparison, not independent k-fold CV. Padding-free, threshold 0.5. No historical-control substitution.'}
    dump(OPS / 'result.json', out)
    lines = ['# 四状态头与同轮原版对照', '', f"源码：{info['gitrev']}；完成时间：{status['finished_at']}", '',
             '| 指标 | 原版 | 四状态头 | 差值 | 改善划分 |', '|---|---:|---:|---:|---:|']
    for metric, label, scale in [('accuracy', '有效键准确率 %', 100), ('f1', '微平均 F1 %', 100),
            ('auc', 'ROC-AUC', 1), ('ex_f1_harmonic', '谱图P/R调和F1 %', 100),
            ('q_brier', 'q-Brier（低为好）', 1), ('q_seq', 'q_seq', 1)]:
        a, b = metrics['control'][metric], metrics['joint'][metric]
        lines.append(f"| {label} | {a['mean']*scale:.6f} ± {a['std_ddof1']*scale:.6f} | {b['mean']*scale:.6f} ± {b['std_ddof1']*scale:.6f} | {b['paired_delta_mean']*scale:+.6f} | {b['win_splits']}/5 |")
    lines += ['', '四项工程筛选门槛：' + ('全部通过；列为待确认候选。' if out['improvement_gate'] else '未全部通过；尚无达到预定标准的提升。'),
              '两臂都恢复完整原始谱图验证，训练保持折叠及group_uniform；原版3GCN+2GAT、FiLM、理论特征保留。',
              '四状态为none/b_only/y_only/both；q=1-P(none)，四状态CE保留原序列化q并学习正状态的条件分配。',
              '该对照衡量四状态头的增益，不能单独量化修复验证相对旧协议的因果效果。冻结论文主表不变。', '',
              '| 划分 | 原版F1 % | 四状态F1 % | 差值（百分点） |', '|---|---:|---:|---:|']
    for fold in FOLDS:
        a, b = data[names[0]][fold]['f1'], data[names[1]][fold]['f1']
        lines.append(f'| {fold} | {a*100:.6f} | {b*100:.6f} | {(b-a)*100:+.6f} |')
    (OPS / 'report.md').write_text('\n'.join(lines) + '\n')
    dump(OPS / 'review_receipt.json', {'status': 'passed', 'root': str(root), 'checks': [
        'All 10 runs completed', 'source/config/raw metric hashes bound', 'paired full-row validation sources and split rows match preflight',
        'all summary means/SD/paired differences/win counts independently recalculated from saved CSV'], 'gates': gates})
    return out


def worker():
    info = json.loads((OPS / 'launch.json').read_text())
    try:
        command = [PY, '-u', str(REPO / 'graph_transform/scripts/run_architecture_batch.py'), '--configs',
                   *[str(config(a)) for a in ARMS], '--fold_data_dir', 'dataset/5fold_folded',
                   '--gpus', ','.join(map(str, info['gpus'])), '--run_root', info['root']]
        with (OPS / 'training.log').open('x') as log:
            subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        analyze(info)
        dump(OPS / 'terminal.json', {'status': 'completed', 'root': info['root'],
             'result': str(OPS / 'result.json'), 'report': str(OPS / 'report.md')})
    except BaseException:
        dump(OPS / 'terminal.json', {'status': 'failed', 'root': info['root'], 'error': traceback.format_exc()})
        raise


if __name__ == '__main__':
    os.chdir(REPO)
    OPS.mkdir(exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['smoke', 'launch', 'worker'])
    args = parser.parse_args()
    globals()[args.action]()
