"""Pod-only four-state hidden-width sweep, reported through terminal events.

Only hidden_dim differs across four predeclared arms. The contemporaneous
256-d control is always first. No outer-test result selects a unique winner.
"""
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
sys.path[:0] = [str(REPO / 'graph_transform'), str(REPO / 'graph_transform/scripts')]
from data.row_validation import file_sha
from four_state_ops import dump, idle
import yaml

OPS = Path('/mnt/pvc/graphtrans/four_state_width_ops')
TARGETS = Path('/mnt/pvc/graphtrans/four_state_targets_v1')
ARMS = {'256': 256, '128': 128, '384': 384, '512': 512}
FOLDS = ('1222', '2252', '3514', '6072', '9075')
BASELINE_REF = '4da48eeab50decd0a31fbf64f2f179478a6624a7'
METRICS = ('f1', 'accuracy', 'auc', 'ex_precision', 'ex_recall',
           'ex_f1_harmonic', 'q_brier', 'q_seq')
REPORT_METRICS = (
    ('f1', '有效键 micro-F1 %', 100),
    ('accuracy', '有效键 accuracy %', 100),
    ('auc', 'ROC-AUC', 1),
    ('ex_f1_harmonic', '谱图 P/R 调和 F1 %', 100),
    ('q_brier', 'q-Brier ↓', 1),
    ('q_seq', 'q_seq', 1),
)
PY = str(REPO / '.venv/bin/python')


def require(ok, why):
    if not ok:
        raise ValueError(why)


def config(arm):
    return REPO / f'graph_transform/config/pre_synthesis_four_state_width_{arm}.yaml'


def revision():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_runtime(runtime, arm):
    model, loss = runtime['model'], runtime['loss']
    require(model['hidden_dim'] == ARMS[arm], f'{arm}: wrong hidden_dim')
    require(model['num_attention_heads'] == 4 and model['concat_heads'] is True,
            f'{arm}: four concatenated GAT heads required')
    require(model['bond_head_type'] == 'four_state_mlp', f'{arm}: four-state head required')
    require(not model.get('use_charge_partition_features', False), f'{arm}: charge readout forbidden')
    require(loss.get('four_state_conditional_weight', 1.) == 1., f'{arm}: alpha must be 1')
    require(loss.get('four_state_union_brier_weight', 0.) == 0., f'{arm}: Brier loss must be off')
    require(runtime['data']['weighting_scheme'] == 'group_uniform'
            and runtime['training']['checkpoint_selection'] == 'max_f1'
            and runtime['evaluation']['threshold'] == .5,
            f'{arm}: training/evaluation protocol changed')


def smoke():
    require(not (OPS / 'smoke.json').exists(), 'Preflight already exists; refuse repeat')
    devices, _, _ = idle()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(devices[0]), OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    with (OPS / 'preflight.log').open('a', encoding='utf-8') as log:
        subprocess.run([PY, str(REPO / 'graph_transform/scripts/test_four_state_width.py'),
                        '--output', str(OPS / 'smoke.json')], cwd=REPO, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    proof = read_json(OPS / 'smoke.json')
    require(proof['status'] == 'passed', 'Preflight did not pass')
    proof['gitrev'] = revision()
    dump(OPS / 'smoke.json', proof)
    print(json.dumps(proof))


def launch():
    require(not (OPS / 'launch.json').exists(), 'Launch exists; refuse duplicate')
    proof = read_json(OPS / 'smoke.json')
    require(proof['status'] == 'passed' and proof['gitrev'] == revision(), 'Missing/stale preflight')
    require(all(file_sha(config(a)) == proof['configs'][a] for a in ARMS),
            'Configs changed since preflight')
    require(set(proof['widths']) == set(ARMS), 'Missing width preflight results')
    for arm in ARMS:
        validate_runtime(yaml.safe_load(config(arm).read_text(encoding='utf-8')), arm)
    changed = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'],
                                      cwd=REPO, text=True).splitlines()
    require(set(changed) <= {'graph_transform/scripts/ensemble_baseline_preds.py'},
            f'Unexpected source changes: {changed}')
    require(read_json(TARGETS / 'preparation.json')['status'] == 'passed',
            'Target preparation did not pass')
    devices, gpu_rows, processes = idle()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
    root = REPO / 'checkpoints/graph_transform/four_state_width_batch' / stamp
    info = {
        'schema_version': 1, 'root': str(root), 'gitrev': revision(),
        'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'baseline_ref': BASELINE_REF, 'control_width': 256, 'widths': ARMS,
        'num_attention_heads': 4, 'concat_heads': True,
        'four_state_conditional_weight': 1., 'four_state_union_brier_weight': 0.,
        'gpus': devices, 'gpu_preflight': gpu_rows, 'compute_processes': processes,
        'configs': proof['configs'], 'smoke_sha256': file_sha(OPS / 'smoke.json'),
        'source_sha256': {str(p.relative_to(REPO)): file_sha(p)
                          for p in (REPO / 'graph_transform').rglob('*.py')
                          if '__pycache__' not in p.parts},
        'target_preparation_sha256': file_sha(TARGETS / 'preparation.json'),
        'target_manifests': {f: file_sha(TARGETS / f'fold_{f}/manifest.json') for f in FOLDS},
        'scope': 'Four-state alpha=1/Brier=0, 3GCN+2GAT with four concatenated heads; '
                 'only hidden_dim 256/128/384/512 differs. Folded group_uniform, '
                 'full-spectrum validation, valid-bond metrics, fixed 0.5 threshold. '
                 'All three predeclared candidates reported; no test-selected unique winner.',
    }
    dump(OPS / 'launch.json', info)
    env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    with (OPS / 'worker.log').open('x', encoding='utf-8') as log:
        proc = subprocess.Popen([PY, __file__, 'worker'], cwd=REPO, env=env,
                                stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
    info['worker_pid'] = proc.pid
    dump(OPS / 'launch.json', info)
    print(json.dumps({k: v for k, v in info.items() if k != 'source_sha256'}))


def elapsed_seconds(started, finished):
    start = datetime.datetime.fromisoformat(started.replace('Z', '+00:00'))
    stop = datetime.datetime.fromisoformat(finished.replace('Z', '+00:00'))
    result = (stop - start).total_seconds()
    require(result >= 0, 'Negative arm wall-clock duration')
    return result


def analyze(info):
    root = Path(info['root'])
    status = read_json(root / 'status.json')
    require(status['status'] == 'completed' and len(status['arms']) == 4,
            'Four-arm batch not complete')
    require(all(a['status'] == 'completed' and a['returncode'] == 0 for a in status['arms']),
            'One or more arms did not finish successfully')
    manifest = read_json(root / 'manifest.json')
    summary = read_json(root / 'summary/summary.json')
    proof = read_json(OPS / 'smoke.json')
    require(manifest['gitrev'] == info['gitrev'] and len(manifest['arms']) == 4,
            'Manifest revision/arm count mismatch')
    require(file_sha(OPS / 'smoke.json') == info['smoke_sha256'], 'Preflight proof changed')
    require(all(file_sha(REPO / p) == h for p, h in info['source_sha256'].items()),
            'Source contents changed after launch')
    require(all(file_sha(config(a)) == info['configs'][a] for a in ARMS),
            'Config contents changed after launch')
    require(file_sha(TARGETS / 'preparation.json') == info['target_preparation_sha256'],
            'Target preparation changed')
    require(all(file_sha(TARGETS / f'fold_{f}/manifest.json') == info['target_manifests'][f]
                for f in FOLDS), 'Target manifest changed')
    require([a['name'] for a in status['arms']] == [a['name'] for a in manifest['arms']],
            'Status/manifest arm ordering differs')
    require([s['arm'] for s in summary['sources']] == [a['name'] for a in manifest['arms']],
            'Summary/manifest arm ordering differs')
    data, binding, recovery, resources = {}, {}, {}, {}
    names = [a['name'] for a in manifest['arms']]
    require(manifest['control'] == names[0], '256-d control must be first')
    for arm, arm_status, (label, width) in zip(manifest['arms'], status['arms'], ARMS.items()):
        require(file_sha(arm['config_snapshot']) == arm['config_sha256'] == info['configs'][label],
                f'{label}: config snapshot mismatch')
        bound = next(s for s in summary['sources'] if s['arm'] == arm['name'])
        require(file_sha(arm['metrics_csv']) == bound['sha256'], f'{label}: metrics source mismatch')
        binding[arm['metrics_csv']] = file_sha(arm['metrics_csv'])
        with Path(arm['metrics_csv']).open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == 5 and {r['fold_id'] for r in rows} == set(FOLDS),
                f'{label}: expected all five unique split rows')
        data[arm['name']], recovery[label] = {}, {}
        width_proof = proof['widths'][label]
        count = width_proof.get('parameter_count', width_proof.get('parameters'))
        require(isinstance(count, int) and count > 0, f'{label}: invalid parameter count')
        resources[label] = {
            **width_proof, 'hidden_dim': width, 'head_dim': width // 4,
            'parameter_count': count, 'gpus': info['gpus'],
            'started_at': arm_status['started_at'], 'finished_at': arm_status['finished_at'],
            'wall_time_seconds': elapsed_seconds(arm_status['started_at'], arm_status['finished_at']),
        }
        for row in rows:
            fold = row['fold_id']
            raw_path = Path(row['metric_csv_path'])
            with raw_path.open(encoding='utf-8-sig', newline='') as stream:
                raw = {r['metric']: float(r['value']) for r in csv.DictReader(stream)}
            binding[str(raw_path)] = file_sha(raw_path)
            vals = {k: float(row[k]) for k in ('f1', 'accuracy', 'auc', 'q_brier',
                                               'ex_precision', 'ex_recall')}
            vals['q_seq'] = float(row['q_spearman_pep_seq'])
            for metric, value in vals.items():
                raw_key = 'q_spearman_pep_seq' if metric == 'q_seq' else metric
                require(abs(value - raw[raw_key]) < 1e-12,
                        f'{label}/{fold}: raw {metric} mismatch')
            p, r = vals['ex_precision'], vals['ex_recall']
            vals['ex_f1_harmonic'] = 2 * p * r / (p + r) if p + r else 0.
            require(all(math.isfinite(v) and (-1 if k == 'q_seq' else 0) <= v <= 1
                        for k, v in vals.items()), f'{label}/{fold}: invalid metrics')
            data[arm['name']][fold] = vals
            folder = Path(arm['cv_root']) / f'fold_{fold}'
            runtime_path = folder / 'config.yaml'
            validate_runtime(yaml.safe_load(runtime_path.read_text(encoding='utf-8')), label)
            binding[str(runtime_path)] = file_sha(runtime_path)
            paths = list(folder.glob('checkpoints/**/row_validation_provenance.json'))
            require(len(paths) == 1, f'{label}/{fold}: ambiguous row-validation provenance')
            prov = read_json(paths[0])
            require(prov['mode'] == 'full_original_training_spectra_by_internal_val_sequence'
                    and prov['outer_test_used_for_internal_validation'] is False,
                    f'{label}/{fold}: invalid internal validation source')
            require(file_sha(prov['source_path']) == prov['source_sha256']
                    == proof['folds'][fold]['source_sha256'], f'{label}/{fold}: source SHA mismatch')
            require(file_sha(prov['split_val_csv']) == proof['folds'][fold]['val_sha256']
                    and prov['val_rows'] == proof['folds'][fold]['val_rows'],
                    f'{label}/{fold}: paired validation rows mismatch')
            require(file_sha(prov['split_train_csv']) == proof['folds'][fold]['train_sha256'],
                    f'{label}/{fold}: paired training split SHA mismatch')
            binding[prov['split_train_csv']] = file_sha(prov['split_train_csv'])
            binding[prov['split_val_csv']] = file_sha(prov['split_val_csv'])
            states_path = paths[0].with_name('four_state_provenance.json')
            require(read_json(states_path)['manifest_sha256'] == info['target_manifests'][fold],
                    f'{label}/{fold}: four-state targets mismatch')
            binding[str(paths[0])] = file_sha(paths[0])
            binding[str(states_path)] = file_sha(states_path)
            checkpoints = list(folder.glob('checkpoints/**/best_model.pt'))
            require(len(checkpoints) == 1 and checkpoints[0].parent == paths[0].parent,
                    f'{label}/{fold}: ambiguous or unbound best_model.pt')
            checkpoint = checkpoints[0]
            recovery[label][fold] = {
                'gitrev': info['gitrev'], 'config': str(runtime_path),
                'config_sha256': binding[str(runtime_path)],
                'checkpoint': str(checkpoint), 'checkpoint_sha256': file_sha(checkpoint),
                'metrics_csv': str(raw_path), 'metrics_sha256': binding[str(raw_path)],
                'selection': 'internal full-spectrum validation max_f1; outer test not used',
            }
    expected = {(name, metric) for name in names for metric in METRICS}
    require(len(summary['metrics']) == len(expected)
            and {(e['arm'], e['metric']) for e in summary['metrics']} == expected,
            'Missing, duplicate, or unexpected summary metric')
    for entry in summary['metrics']:
        arm, metric = entry['arm'], entry['metric']
        values = [data[arm][f][metric] for f in FOLDS]
        deltas = [data[arm][f][metric] - data[names[0]][f][metric] for f in FOLDS]
        sign = -1 if metric == 'q_brier' else 1
        for key, value in (
                ('mean', statistics.mean(values)), ('std_ddof1', statistics.stdev(values)),
                ('paired_delta_mean', statistics.mean(deltas)),
                ('paired_delta_std_ddof1', statistics.stdev(deltas))):
            require(abs(value - entry[key]) < 1e-12, f'{arm}/{metric}: {key} mismatch')
        require(sum(sign * d > 1e-12 for d in deltas) == entry['win_splits'],
                f'{arm}/{metric}: win count mismatch')
    metrics = {label: {r['metric']: r for r in summary['metrics'] if r['arm'] == name}
               for label, name in zip(ARMS, names)}
    decisions = {}
    for label in list(ARMS)[1:]:
        values = metrics[label]
        gates = {
            'delta_f1_at_least_0.002': values['f1']['paired_delta_mean'] >= .002,
            'f1_win_at_least_4_of_5': values['f1']['win_splits'] >= 4,
            'delta_auc_nonnegative': values['auc']['paired_delta_mean'] >= 0,
            'delta_q_brier_nonpositive': values['q_brier']['paired_delta_mean'] <= 0,
        }
        decisions[label] = {'gates': gates, 'improvement_gate': all(gates.values())}
    out = {
        'status': 'completed', 'root': str(root), 'gitrev': info['gitrev'],
        'baseline_ref': BASELINE_REF, 'finished_at': status['finished_at'],
        'widths': ARMS, 'control_width': 256, 'num_attention_heads': 4,
        'metrics': metrics, 'decisions': decisions, 'folds': data,
        'resources': resources, 'recovery': recovery, 'paired_validation_verified': True,
        'scope': 'Exploratory five overlapping sequence-level 80/20 splits, one initialization '
                 'per split/arm; three predeclared candidates all reported, no test-selected '
                 'unique winner. Padding-free, fixed threshold 0.5. Gates are descriptive '
                 'engineering criteria, not statistical significance.',
    }
    dump(OPS / 'result.json', out)
    dump(OPS / 'checkpoint_registry.json', {
        'baseline_ref': BASELINE_REF, 'gitrev': info['gitrev'], 'root': str(root),
        'candidate_status': 'All arms retained; any passing arm remains unconfirmed pending replication',
        'widths': recovery,
    })
    write_report(info, status, metrics, decisions, data, names, resources, recovery)
    dump(OPS / 'review_receipt.json', {
        'status': 'passed', 'root': str(root), 'gitrev': info['gitrev'],
        'checks': [
            '20 runs completed; raw metrics/source/config/target hashes bound',
            'Every runtime width, four concatenated heads, alpha=1/Brier=0, no charge readout verified',
            'All training splits, full-row validation sources and validation split SHAs match paired preflight',
            '32 arm-metrics mean/sampleSD/paired delta/sampleSD/wins independently recalculated',
            'Every validation-selected best_model.pt path and SHA256 retained',
        ],
        'hashes': binding, 'decisions': decisions,
        'result_sha256': file_sha(OPS / 'result.json'),
        'report_sha256': file_sha(OPS / 'report.md'),
        'registry_sha256': file_sha(OPS / 'checkpoint_registry.json'),
    })


def write_report(info, status, metrics, decisions, data, names, resources, recovery):
    lines = [
        '# 四状态头：隐藏维度 256 / 128 / 384 / 512 同轮对照', '',
        f"源码：`{info['gitrev']}`；完成：{status['finished_at']}",
        f'保留的已提升四状态参考提交：`{BASELINE_REF}`。本轮比较使用重新训练的 256 维同轮对照。', '',
        '| 指标 | 256（对照）均值 ± SD | 128 均值 ± SD | Δ ± 配对 SD / 改善划分 | 384 均值 ± SD | Δ ± 配对 SD / 改善划分 | 512 均值 ± SD | Δ ± 配对 SD / 改善划分 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for metric, label, scale in REPORT_METRICS:
        cells = []
        for arm in ARMS:
            value = metrics[arm][metric]
            cells.append(f"{value['mean'] * scale:.6f} ± {value['std_ddof1'] * scale:.6f}")
            if arm != '256':
                cells.append(f"{value['paired_delta_mean'] * scale:+.6f} ± "
                             f"{value['paired_delta_std_ddof1'] * scale:.6f} / {value['win_splits']}/5")
        lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '百分比指标的 Δ 和 SD 使用百分点；Brier 越低越好，改善划分按相应方向计数。', '']
    for arm in list(ARMS)[1:]:
        outcome = '全部预定门槛通过，保留为待复验候选。' if decisions[arm]['improvement_gate'] else '未全部通过预定提升门槛。'
        lines.append(f'{arm} 维：{outcome} 门槛：' + '；'.join(
            f"{key}={'通过' if passed else '未通过'}" for key, passed in decisions[arm]['gates'].items()))
    lines += ['', '三种候选全部报告，不依外层测试成绩选出唯一最佳宽度，也不改动冻结论文主表。',
              '四臂均保持原四状态头、3GCN+2GAT、4 头拼接、α=1、union-Brier=0、无电荷分配分支；仅 hidden_dim 不同。',
              '保持折叠 group_uniform、完整原谱内部验证、有效键 mask 和固定 0.5 阈值。',
              '五组序列级 80/20 划分有重叠，各一次初始化；均值/样本 SD/配对差和胜率为描述性结果，工程门槛不代表统计显著。', '',
              '| 隐藏维度 | GAT 头数 × 每头维度 | 参数量 | 五划分臂耗时（小时） |',
              '|---|---:|---:|---:|']
    for arm, width in ARMS.items():
        value = resources[arm]
        lines.append(f"| {width} | 4 × {width // 4} | {value['parameter_count']:,} | {value['wall_time_seconds'] / 3600:.4f} |")
    lines += ['', '耗时为每臂五划分调度的墙钟时间，受设备、并行调度和早停影响，不是等步数计算效率。',
              '预检峰值显存及完整 512 维训练/验证批量核验保存在 result.json 的 resources 和 smoke.json。', '']
    for arm, name in zip(ARMS, names):
        if arm == '256':
            continue
        lines += [f'## {arm} 维逐划分变化（相对同轮 256 维）', '',
                  '| 划分 | Δ F1 百分点 | Δ accuracy 百分点 | Δ AUC | Δ 谱图调和 F1 百分点 | Δ q-Brier ↓ | Δ q_seq |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for fold in FOLDS:
            values = [f"{(data[name][fold][metric] - data[names[0]][fold][metric]) * scale:+.6f}"
                      for metric, _, scale in REPORT_METRICS]
            lines.append('| ' + fold + ' | ' + ' | '.join(values) + ' |')
        lines.append('')
    lines += ['## 恢复记录', '',
              '每个宽度、每个划分的 validation max-F1 权重路径、SHA256、运行配置和原始指标已写入 `checkpoint_registry.json`；没有删除候选。',
              f"批次：`{info['root']}`", '', '| 宽度 | 划分 | best_model.pt | SHA256 |', '|---|---|---|---|']
    for arm in ARMS:
        for fold in FOLDS:
            item = recovery[arm][fold]
            lines.append(f"| {arm} | {fold} | `{item['checkpoint']}` | `{item['checkpoint_sha256']}` |")
    (OPS / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def worker():
    info = read_json(OPS / 'launch.json')
    try:
        command = [PY, '-u', str(REPO / 'graph_transform/scripts/run_architecture_batch.py'),
                   '--configs', *[str(config(a)) for a in ARMS],
                   '--fold_data_dir', 'dataset/5fold_folded',
                   '--gpus', ','.join(map(str, info['gpus'])), '--run_root', info['root']]
        with (OPS / 'training.log').open('x', encoding='utf-8') as log:
            subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        analyze(info)
        dump(OPS / 'terminal.json', {
            'status': 'completed', 'root': info['root'], 'gitrev': info['gitrev'],
            'result': str(OPS / 'result.json'), 'report': str(OPS / 'report.md'),
            'review_receipt': str(OPS / 'review_receipt.json'),
            'checkpoint_registry': str(OPS / 'checkpoint_registry.json'),
        })
    except BaseException:
        error = traceback.format_exc()
        status_path = Path(info['root']) / 'status.json'
        detail = read_json(status_path).get('error') if status_path.exists() else None
        dump(OPS / 'terminal.json', {
            'status': 'failed', 'root': info['root'], 'gitrev': info['gitrev'],
            'error': error, 'batch_error': detail, 'training_log': str(OPS / 'training.log'),
            'retry_policy': 'No automatic rerun or additional experiment',
        })
        raise


if __name__ == '__main__':
    os.chdir(REPO)
    OPS.mkdir(exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['smoke', 'launch', 'worker'])
    globals()[parser.parse_args().action]()
