"""Pod-only posthoc audit of the completed stage-9 hierarchy training batch.

This does not launch or resume training. It preserves the original failed
terminal/result files, verifies the frozen training sources and all 15 weights,
then writes a separately qualified report under OPS/posthoc_review_v2_20261024.
"""

import argparse
import contextlib
import copy
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'graph_transform'), str(REPO / 'graph_transform/scripts')]
import four_state_virtual_hierarchy_ops as ops
from data.graph_dataset import CachedGraphDataset, GraphDataset, GraphDataLoader
from data.row_validation import file_sha
from evaluation.evaluator import Evaluator
from models.graph_transformer import GraphTransformer
from models.utils import build_model_config

POSTHOC = ops.OPS / 'posthoc_review_v2_20261024'
KNOWN_DIRTY_UNRELATED_SOURCE = {
    'path': 'graph_transform/scripts/ensemble_baseline_preds.py',
    'launch_sha256': '8202bf2b660dfb56fd79ebeae2eda57cd952c57e89cbeb30f30e416db49f4be0',
    'git_blob_sha256': '805d4aaaffacaf8157d62cf532e24255e093e46313e59edb3710f3c5f5c6d4d4',
}
METRIC_NAMES = ('f1', 'accuracy', 'auc', 'ex_precision', 'ex_recall',
                'ex_f1_harmonic', 'q_brier', 'q_seq')
# A separately run full-precision CUDA evaluation may differ at the 0.5
# boundary. These are audit tolerances, not criteria for choosing an arm.
REPLAY_TOLERANCE = {'f1': 5e-5, 'accuracy': 5e-5, 'auc': 5e-5,
                    'ex_precision': 5e-5, 'ex_recall': 5e-5,
                    'ex_f1_harmonic': 5e-5, 'q_brier': 2e-6, 'q_seq': 2e-6}


def demand(condition, message):
    if not condition:
        raise ValueError(message)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def frozen_source_audit(info):
    """Bind each live source to launch SHA and training files to Git blobs.

    The new posthoc script is intentionally absent from the original manifest.
    The one pre-existing unrelated dirty source is bound to its exact launch
    SHA and its distinct Git blob, with both identities explicitly recorded.
    """
    bound = info['source_sha256']
    demand(isinstance(bound, dict) and len(bound) > 20, 'Missing training source manifest')
    checked = {}
    exceptions = []
    for name, expected in sorted(bound.items()):
        path = REPO / name
        demand(path.is_file() and file_sha(path) == expected,
               f'Training source changed since launch: {name}')
        blob = subprocess.check_output(['git', 'show', f"{info['gitrev']}:{name}"], cwd=REPO)
        git_sha = hashlib.sha256(blob).hexdigest()
        if name == KNOWN_DIRTY_UNRELATED_SOURCE['path']:
            demand(expected == KNOWN_DIRTY_UNRELATED_SOURCE['launch_sha256']
                   and git_sha == KNOWN_DIRTY_UNRELATED_SOURCE['git_blob_sha256'],
                   f'Known unrelated dirty source identity changed: {name}')
            exceptions.append({'path': name, 'launch_sha256': expected,
                               'git_blob_sha256': git_sha,
                               'reason': 'Pre-existing unrelated dirty script, not imported by training/evaluation'})
        else:
            demand(git_sha == expected,
                   f'Launch source differs from Git revision: {name}')
        checked[name] = expected
    demand(len(exceptions) == 1 and len(checked) == len(bound),
           'Unexpected source-audit exception count')
    return {'launch_source_sha256': checked, 'known_dirty_unrelated_sources': exceptions}


def original_terminal_barrier(info):
    """This script is only for the known analysis failure after completed training."""
    terminal_path = ops.OPS / 'terminal.json'
    demand(terminal_path.is_file(), 'Original terminal event is missing')
    terminal = ops.read_json(terminal_path)
    demand(terminal.get('status') == 'failed'
           and terminal.get('root') == info['root']
           and terminal.get('gitrev') == info['gitrev'],
           'Original terminal identity/status differs from failed stage-9 event')
    demand('Prediction-row recomputation disagrees' in terminal.get('error', ''),
           'Original failure is not the known exported-prediction metric disagreement')
    root = Path(info['root'])
    status, manifest = (ops.read_json(root / name) for name in ('status.json', 'manifest.json'))
    demand(status.get('status') == 'completed' and manifest.get('root') == info['root']
           and status.get('root') == info['root']
           and manifest.get('gitrev') == status.get('gitrev') == info['gitrev']
           and len(status.get('arms', [])) == len(manifest.get('arms', [])) == len(ops.ARMS)
           and all(arm.get('status') == 'completed' and arm.get('returncode') == 0
                   for arm in status['arms']),
           'Training status/manifest is not a completed three-arm batch')
    return {'original_terminal': str(terminal_path),
            'original_terminal_sha256': file_sha(terminal_path),
            'original_status_sha256': file_sha(root / 'status.json'),
            'original_manifest_sha256': file_sha(root / 'manifest.json'),
            'original_terminal_status': terminal['status'],
            'original_failure_excerpt': next((line for line in terminal['error'].splitlines()
                                              if 'Prediction-row recomputation disagrees' in line), '')}


def analyze_without_hiding_export_drift(info):
    """Reuse strict stage-9 provenance/15-weight checks; capture only known drift.

    All output paths are redirected to POSTHOC. The original ops terminal and
    original batch are never opened for writing. Every other failed assertion
    remains fatal. The provisional receipt is downgraded before delivery.
    """
    shutil.copyfile(ops.OPS / 'smoke.json', POSTHOC / 'smoke.json')
    original_ops, original_require, original_dump, original_sources = (
        ops.OPS, ops.require, ops.dump, ops.sources)
    export_messages = []

    def qualified_require(ok, why):
        if not ok and why.startswith('Prediction-row recomputation disagrees for '):
            export_messages.append(why)
            return None
        return original_require(ok, why)

    def provisional_dump(path, value):
        if Path(path).name == 'review_receipt.json':
            value = {**value, 'status': 'provisional_export_discrepancy',
                     'export_metric_discrepancies': export_messages}
        return original_dump(path, value)

    try:
        ops.OPS = POSTHOC
        ops.require = qualified_require
        ops.dump = provisional_dump
        # frozen_source_audit already checked the entire original training
        # manifest. A newly added audit script may be present in this checkout.
        ops.sources = lambda: info['source_sha256']
        ops.analyze(info)
    finally:
        ops.OPS, ops.require, ops.dump, ops.sources = (
            original_ops, original_require, original_dump, original_sources)
    return export_messages


def full_precision_evaluate(runtime, checkpoint, test_path, prediction_path, gpu,
                            cache_key):
    """Rerun test inference from the saved selected weight, without training."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    effective = copy.deepcopy(runtime)
    effective['model'] = saved['config']
    model = GraphTransformer(build_model_config(effective)).cpu()
    model.load_state_dict(saved['model_state_dict'], strict=True)
    demand(all(torch.isfinite(parameter).all().item() for parameter in model.parameters()),
           'Full-precision replay loaded a nonfinite parameter')
    data_config = effective['data']
    dataset_cls = CachedGraphDataset if data_config.get('cache_graphs', False) else GraphDataset
    dataset_kwargs = {'csv_path': str(test_path), 'config': effective,
                      'max_seq_len': data_config['max_seq_len'],
                      'graph_strategy': data_config['graph_strategy'],
                      'augmentation': False, 'split': 'test'}
    cache_source = 'not_used'
    if dataset_cls is CachedGraphDataset:
        # Never let a posthoc evaluation rebuild the training cache. Reuse its
        # immutable edge cache bytes in a separate audit-owned directory.
        original_cache = Path(data_config.get('cache_dir', 'cache/graph_data')).resolve()
        isolated_cache = POSTHOC / 'cache' / cache_key
        isolated_cache.mkdir(parents=True, exist_ok=False)
        cached_edges = sorted(original_cache.glob('edges_*.pt'))
        for edge_file in cached_edges:
            shutil.copy2(edge_file, isolated_cache / edge_file.name)
        cache_source = {'original_dir': str(original_cache),
                        'isolated_dir': str(isolated_cache),
                        'copied_edge_cache': [str(path) for path in cached_edges],
                        'original_rebuild_cache': data_config.get('rebuild_cache')}
        dataset_kwargs.update(cache_dir=str(isolated_cache),
                              rebuild_cache=False,
                              cache_full_graphs=data_config.get('cache_full_graphs', False))
    dataset = dataset_cls(**dataset_kwargs)
    # Match the original training-script test loader rather than the separate
    # CLI default: evaluation.batch_size can differ from training.batch_size.
    batch_size = effective.get('evaluation', {}).get(
        'batch_size', effective['training']['batch_size'])
    performance = effective.get('performance', {})
    loader = GraphDataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                             num_workers=data_config.get('num_workers', 4),
                             pin_memory=data_config.get('pin_memory', True),
                             persistent_workers=performance.get('persistent_workers', False),
                             prefetch_factor=performance.get('prefetch_factor'),
                             drop_last=False)
    device = torch.device(f'cuda:{gpu}')
    evaluator = Evaluator(model, device=device, config=effective,
                          allow_target_aware_threshold=False)
    with torch.inference_mode():
        computed = evaluator.evaluate(loader, allow_target_aware=False)
    raw_logits = evaluator.metrics_calculator.sample_predictions
    evaluated_rows = len(raw_logits)
    differing_rows = differing_bonds = mismatch_rows_all_logits_01 = 0
    all_logits_01_rows = 0
    with prediction_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        for index, logit_row in enumerate(raw_logits):
            row = next(reader, None)
            demand(row is not None, 'Prediction export ended before full-precision replay')
            exported = ([int(bit) for bit in row['pred'].split(';')]
                        if row['pred'].strip() else [])
            demand(len(exported) == len(logit_row),
                   f'Full-precision replay/export bond count differs at row {index}')
            if len(logit_row) == 0:
                continue
            raw = torch.as_tensor(logit_row, dtype=torch.float32)
            all_in_01 = bool(((raw >= 0) & (raw <= 1)).all().item())
            all_logits_01_rows += int(all_in_01)
            formal_bits = (torch.sigmoid(raw) > .5).to(torch.int32).tolist()
            differences = sum(formal != saved for formal, saved in zip(formal_bits, exported))
            if differences:
                differing_rows += 1
                differing_bonds += differences
                mismatch_rows_all_logits_01 += int(all_in_01)
        demand(next(reader, None) is None,
               'Prediction export has rows beyond full-precision replay')
    del evaluator, model, loader, dataset
    torch.cuda.empty_cache()
    precision, recall = float(computed['ex_precision']), float(computed['ex_recall'])
    harmonic = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return {'f1': float(computed['f1']), 'accuracy': float(computed['accuracy']),
            'auc': float(computed['auc']), 'ex_precision': precision,
            'ex_recall': recall, 'ex_f1_harmonic': harmonic,
            'q_brier': float(computed['q_brier']),
            'q_seq': float(computed['q_spearman_pep_seq']),
            'evaluated_rows': evaluated_rows,
            'replay_export_hard_bit_comparison': {
                'differing_spectra': differing_rows,
                'differing_bonds': differing_bonds,
                'mismatch_spectra_with_all_raw_logits_in_01': mismatch_rows_all_logits_01,
                'all_raw_logits_in_01_spectra': all_logits_01_rows,
                'rule': 'sigmoid(raw formal replay logit) > 0.5 versus archived pred bit; descriptive only'},
            'batch_size': batch_size,
            'num_workers': data_config.get('num_workers', 4),
            'pin_memory': data_config.get('pin_memory', True),
            'persistent_workers': performance.get('persistent_workers', False),
            'prefetch_factor': performance.get('prefetch_factor'),
            'cache_source': cache_source,
            'amp': bool(effective.get('device', {}).get('use_amp', False))}


def replay_all(result, gpu):
    replay = {}
    all_within = True
    rows = []
    for arm in ops.ARMS:
        replay[arm] = {}
        for fold in ops.FOLDS:
            saved = result['recovery'][arm][fold]
            proof = saved['prediction_recomputation']
            checkpoint = Path(saved['checkpoint'])
            runtime = yaml.safe_load(Path(saved['config']).read_text(encoding='utf-8'))
            source = Path(proof['test_csv'])
            demand(file_sha(checkpoint) == saved['checkpoint_sha256']
                   and file_sha(source) == proof['test_sha256']
                   and file_sha(saved['config']) == saved['config_sha256'],
                   f'{arm}/{fold}: weight/config/test source changed before full-precision replay')
            prediction_path = Path(proof['prediction_csv'])
            demand(file_sha(prediction_path) == proof['prediction_sha256'],
                   f'{arm}/{fold}: prediction export changed before full-precision replay')
            computed = full_precision_evaluate(runtime, checkpoint, source, prediction_path, gpu,
                                               f'{arm}_{fold}')
            expected = result['folds'][arm][fold]
            deltas = {name: computed[name] - expected[name] for name in METRIC_NAMES}
            within = {name: abs(delta) <= REPLAY_TOLERANCE[name]
                      for name, delta in deltas.items()}
            # A wide generic tolerance must not validate a replay that merely
            # reproduces a known wrong export. For each material export/formal
            # disagreement the replay must be meaningfully closer to formal.
            export_resolution = {}
            for name, export_difference in proof['absolute_differences'].items():
                export_tolerance = proof['tolerances'][name]
                if export_difference <= export_tolerance:
                    continue
                export_resolution[name] = {
                    'export_abs_difference': export_difference,
                    'replay_abs_difference': abs(deltas[name]),
                    'original_audit_tolerance': export_tolerance,
                    'replay_resolves_export_discrepancy':
                    abs(deltas[name]) <= export_tolerance
                    and abs(deltas[name]) <= .5 * export_difference}
            resolved = all(item['replay_resolves_export_discrepancy']
                           for item in export_resolution.values())
            all_within &= all(within.values()) and resolved
            demand(computed['evaluated_rows'] == proof['rows'],
                   f'{arm}/{fold}: full-precision replay row count differs from prediction export')
            replay[arm][fold] = {'full_precision': computed,
                                 'original_metric_csv': {name: expected[name] for name in METRIC_NAMES},
                                 'delta_from_original': deltas, 'within_tolerance': within,
                                 'all_within_tolerance': all(within.values()),
                                 'export_resolution': export_resolution,
                                 'material_export_discrepancies_resolved': resolved,
                                 'prediction_export_differences': proof['absolute_differences'],
                                 'prediction_export_tolerances': proof['tolerances'],
                                 'rounded_threshold_ambiguities': proof['rounded_threshold_ambiguities']}
            for metric in METRIC_NAMES:
                rows.append({'arm': arm, 'fold': fold, 'metric': metric,
                             'formal_metric_csv': expected[metric],
                             'full_precision_replay': computed[metric],
                             'replay_minus_formal': deltas[metric],
                             'replay_tolerance': REPLAY_TOLERANCE[metric],
                             'within_tolerance': within[metric],
                             'export_abs_difference': proof['absolute_differences'].get(metric, ''),
                             'material_export_discrepancy_resolved':
                             export_resolution.get(metric, {}).get('replay_resolves_export_discrepancy', '')})
    with (POSTHOC / 'full_precision_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return replay, all_within


def finish_report(result, original, source_manifest, export_messages, replay, all_within):
    mismatches = {}
    for arm in ops.ARMS:
        for fold in ops.FOLDS:
            proof = result['recovery'][arm][fold]['prediction_recomputation']
            wrong = {name: difference for name, difference in proof['absolute_differences'].items()
                     if difference > proof['tolerances'][name]}
            if wrong:
                mismatches[f'{arm}/{fold}'] = wrong
    status = ('qualified_export_discrepancy' if all_within else
              'unverified_full_precision_disagreement')
    verdict = ('正式指标与15份全精度复评在明示容差内，且复评比所有超容差导出值更接近正式指标；'
               '配对性能比较仅作描述性结论。' if all_within else
               '全精度复评未充分解释全部导出差异；本批次没有可靠的性能优劣结论。')
    bit_diagnostics = {f'{arm}/{fold}': replay[arm][fold]['full_precision']
                       ['replay_export_hard_bit_comparison']
                       for arm in ops.ARMS for fold in ops.FOLDS}
    metrics_within_tolerance = all(replay[arm][fold]['all_within_tolerance']
                                   for arm in ops.ARMS for fold in ops.FOLDS)
    audit = {'status': status, 'completed_at': utc_now(),
             'posthoc_script': str(Path(__file__).resolve()),
             'posthoc_script_sha256': file_sha(Path(__file__)),
             'original_terminal': original,
             'training_status': 'completed',
             'original_terminal_status': 'failed',
             'original_terminal_remains_failed': True,
             'source_sha256': source_manifest,
             'export_discrepancy_messages': export_messages,
             'export_metric_mismatches': mismatches,
             'full_precision_replay': replay,
             'full_precision_replay_all_within_tolerance': metrics_within_tolerance,
             'all_qualification_checks_pass': all_within,
             'performance_verdict_valid': all_within,
             'performance_verdict': verdict,
             'replay_export_hard_bit_comparison': bit_diagnostics,
             'replay_tolerances': REPLAY_TOLERANCE,
             'interpretation': ('Saved pred.csv classes are produced by a separate forward pass '
                                'and probabilities are printed to six decimals. This audit does '
                                'not rewrite the original failed terminal or silently accept '
                                'export-metric disagreement. Formal raw CSV paired comparisons '
                                'remain descriptive; no automatic model promotion.')}
    result['status'] = status
    result['training_status'] = 'completed'
    result['original_terminal_status'] = 'failed'
    result['performance_verdict_valid'] = all_within
    result['performance_verdict'] = verdict
    result['posthoc_audit'] = audit
    ops.dump(POSTHOC / 'result.json', result)
    lines = [f'# 事后核验：{status}', '',
             '**原自动分析终态：failed；训练状态：completed。** '
             '原批次15次训练均完成，但保存的逐谱图预测与正式指标不一致；'
             '事后审计没有改写原终态或原始产物。', '',
             '**性能判断：** ' + verdict, '',
             '仅从15份已保存权重重新测试推理，无训练、无重新选模型。', '',
             f'逐谱图导出超容差：{json.dumps(mismatches, ensure_ascii=False, sort_keys=True)}', '',
             '已从15份最佳权重逐份完整测试推理；各指标正式 CSV、全精度复评、差值与明示容差见 '
             '`full_precision_metrics.csv`。所有权重、源码、配置、目标和测试来源 SHA 均在本事后索引中。', '',
             '原逐谱图 CSV 的 `pred` 是第二次推理在概率舍入前确定的类别；'
             '全精度复评与导出 hard bit 的逐行差异在结果文件中单列，'
             '其中原始 logit 全落在[0,1]的行数只提示可能的阈值语义问题，不能单独确定原因。'
             '五组划分重叠、每臂每组一次初始化；工程门槛不是显著性检验。', '',
             '---', '',
             '以下是按原正式指标 CSV 计算的全部候选描述性对照。'
             '若上方状态为 unverified，它们不能用来判断模型优劣。', '']
    report_path = POSTHOC / 'report.md'
    original_report = report_path.read_text(encoding='utf-8')
    report_path.write_text('\n'.join(lines) + '\n' + original_report, encoding='utf-8')
    dashboard_path = POSTHOC / 'virtual_hierarchy_dashboard.html'
    original_dashboard = dashboard_path.read_text(encoding='utf-8')
    banner = ('<div role="alert" style="background:#fee2e2;border:2px solid #b91c1c;'
              'border-radius:12px;padding:18px;margin:20px 0;color:#7f1d1d">'
              f'<strong>原分析失败；训练已完成。事后核验：{status}</strong><br>'
              f'{verdict} <a href="full_precision_metrics.csv">查看15份全精度复评差值</a>。'
              '</div>')
    heading = '<h1>256维四状态＋k5：层级虚拟节点</h1>'
    demand(original_dashboard.count(heading) == 1, 'Dashboard heading changed unexpectedly')
    dashboard = original_dashboard.replace(heading, heading + banner, 1)
    if not all_within:
        gate_script = "d.decisions[a].improvement_gate?'通过':'未通过'"
        demand(dashboard.count(gate_script) == 1, 'Dashboard gate renderer changed unexpectedly')
        dashboard = dashboard.replace(gate_script, "'待核验，不能判断性能优劣'", 1)
    dashboard_path.write_text(dashboard, encoding='utf-8')
    registry = ops.read_json(POSTHOC / 'checkpoint_registry.json')
    registry['posthoc_audit_status'] = status
    registry['original_terminal_sha256'] = original['original_terminal_sha256']
    registry['full_precision_replay_csv'] = str(POSTHOC / 'full_precision_metrics.csv')
    registry['full_precision_replay_csv_sha256'] = file_sha(POSTHOC / 'full_precision_metrics.csv')
    ops.dump(POSTHOC / 'checkpoint_registry.json', registry)
    receipt = ops.read_json(POSTHOC / 'review_receipt.json')
    receipt.update(status=status, original_terminal=original,
                   posthoc_script=str(Path(__file__).resolve()),
                   posthoc_script_sha256=file_sha(Path(__file__)),
                   original_terminal_remains_failed=True,
                   training_status='completed', original_terminal_status='failed',
                   performance_verdict_valid=all_within, performance_verdict=verdict,
                   export_discrepancy_messages=export_messages,
                   export_metric_mismatches=mismatches,
                   full_precision_replay_all_within_tolerance=metrics_within_tolerance,
                   all_qualification_checks_pass=all_within,
                   replay_export_hard_bit_comparison=bit_diagnostics,
                   replay_tolerances=REPLAY_TOLERANCE,
                   replay_csv_sha256=file_sha(POSTHOC / 'full_precision_metrics.csv'),
                   result_sha256=file_sha(POSTHOC / 'result.json'),
                   report_sha256=file_sha(POSTHOC / 'report.md'),
                   registry_sha256=file_sha(POSTHOC / 'checkpoint_registry.json'),
                   dashboard_sha256=file_sha(dashboard_path))
    receipt['checks'] = [check for check in receipt['checks']
                         if not check.startswith('15 per-spectrum prediction CSVs')]
    receipt['checks'].append('15 prediction CSV discrepancies recorded, never silently waived')
    receipt['checks'].append('15 saved checkpoints replayed through full-precision Evaluator.evaluate')
    ops.dump(POSTHOC / 'review_receipt.json', receipt)
    ops.dump(POSTHOC / 'audit_status.json', {'status': status, 'completed_at': audit['completed_at'],
             'original_terminal_sha256': original['original_terminal_sha256'],
             'receipt_sha256': file_sha(POSTHOC / 'review_receipt.json')})
    return status


def run(gpu):
    demand(sys.platform.startswith('linux') and str(REPO).startswith('/mnt/pvc/'), 'Pod only')
    demand(torch.cuda.is_available() and isinstance(gpu, int) and not isinstance(gpu, bool),
           'A CUDA pod device is required for full-precision replay')
    torch.cuda.set_device(gpu)
    torch.set_num_threads(2)
    ops.OPS.mkdir(parents=True, exist_ok=True)
    claim = ops.OPS / 'posthoc_audit_v2.claim'
    with claim.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps({'pid': os.getpid(), 'created_at': utc_now()}) + '\n')
    POSTHOC.mkdir(exist_ok=False)
    ops.dump(POSTHOC / 'audit_status.json', {'status': 'in_progress', 'started_at': utc_now()})
    try:
        info = ops.read_json(ops.OPS / 'launch.json')
        original = original_terminal_barrier(info)
        source_manifest = frozen_source_audit(info)
        export_messages = analyze_without_hiding_export_drift(info)
        result = ops.read_json(POSTHOC / 'result.json')
        replay, all_within = replay_all(result, gpu)
        status = finish_report(result, original, source_manifest,
                               export_messages, replay, all_within)
        demand(file_sha(ops.OPS / 'terminal.json') == original['original_terminal_sha256']
               and file_sha(Path(info['root']) / 'status.json') == original['original_status_sha256']
               and file_sha(Path(info['root']) / 'manifest.json') == original['original_manifest_sha256'],
               'Original terminal/training status/manifest changed during posthoc audit')
        print(json.dumps({'status': status, 'root': info['root'],
                          'posthoc': str(POSTHOC), 'export_mismatch_count': len(export_messages),
                          'all_qualification_checks_pass': all_within}))
    except BaseException:
        ops.dump(POSTHOC / 'audit_status.json', {'status': 'failed', 'finished_at': utc_now(),
                 'error': traceback.format_exc(),
                 'original_terminal_preserved': str(ops.OPS / 'terminal.json')})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    os.chdir(REPO)
    run(args.gpu)
