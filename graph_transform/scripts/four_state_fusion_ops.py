"""Pod-only early/late theory-routing ablation, using terminal events only.

Wait for the width sweep to finish, pass review, and be reported before any
preflight or launch. Three predeclared 256-d arms reuse the same parameters;
all are reported against the contemporaneous late-route control.
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

OPS = Path('/mnt/pvc/graphtrans/four_state_fusion_ops')
PREVIOUS_OPS = Path('/mnt/pvc/graphtrans/four_state_width_ops')
NOTIFIED = Path('/mnt/pvc/graphtrans/dbond_completion_notified.json')
TARGETS = Path('/mnt/pvc/graphtrans/four_state_targets_v1')
ARMS = {'late': 'late', 'early_only': 'early_only', 'early_plus_late': 'early_plus_late'}
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
ROUTING_CHECKS = (
    'late_backbone_theory_gradient_absent', 'early_backbone_theory_gradient_present',
    'padding_excluded', 'global_direct_injection_zero', 'initialization_equal',
    'legacy_late_exact', 'checkpoint_cross_mode_rejected_before_copy',
    'causal_routing_checks_on_cpu',
)
PY = str(REPO / '.venv/bin/python')


def require(ok, why):
    if not ok:
        raise ValueError(why)


def config(arm):
    return REPO / f'graph_transform/config/pre_synthesis_four_state_fusion_{arm}.yaml'


def revision():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prior_completion_barrier():
    """Read terminal receipts only; never inspect a running training status."""
    launch_path = PREVIOUS_OPS / 'launch.json'
    terminal_path = PREVIOUS_OPS / 'terminal.json'
    review_path = PREVIOUS_OPS / 'review_receipt.json'
    require(all(p.is_file() for p in (launch_path, terminal_path, review_path, NOTIFIED)),
            'Width sweep has not completed, passed review, and been reported; do not start fusion')
    launch_info, terminal, review = (read_json(p) for p in (launch_path, terminal_path, review_path))
    root = launch_info.get('root')
    require(isinstance(root, str) and root.startswith(str(REPO / 'checkpoints/graph_transform/four_state_width_batch') + '/'),
            'Invalid preceding width-sweep root')
    require(terminal.get('status') == 'completed' and terminal.get('root') == root
            and terminal.get('gitrev') == launch_info.get('gitrev'),
            'Preceding width sweep is incomplete, failed, or has an unbound terminal receipt')
    require(review.get('status') == 'passed' and review.get('root') == root
            and review.get('gitrev') == launch_info.get('gitrev'),
            'Preceding width sweep has not passed its bound numerical review')
    event_key = root + '|completed'
    reported = read_json(NOTIFIED).get('reported', {})
    require(isinstance(reported, dict) and event_key in reported,
            'Preceding width result has not been reported and acknowledged; do not advance')
    return {'root': root, 'event_key': event_key, 'gitrev': launch_info['gitrev'],
            'launch_sha256': file_sha(launch_path), 'terminal_sha256': file_sha(terminal_path),
            'review_sha256': file_sha(review_path), 'reported_receipt': reported[event_key]}


def validate_runtime(runtime, arm):
    model, loss = runtime['model'], runtime['loss']
    require(model['hidden_dim'] == 256, f'{arm}: hidden_dim must remain 256')
    require(model.get('theory_fusion_stage', 'late') == ARMS[arm], f'{arm}: wrong theory routing stage')
    require(model.get('use_theory_features') is True, f'{arm}: theory features must remain enabled')
    require(model['num_gcn_layers'] == 3 and model['num_gat_layers'] == 2,
            f'{arm}: original 3GCN+2GAT backbone required')
    require(model.get('use_condition_film') is True and model.get('use_global_node') is True,
            f'{arm}: original FiLM/global-node branches required')
    require(model.get('position_embedding_mode', 'input_concat') == 'input_concat',
            f'{arm}: original input-concatenated positions required')
    require(model['num_attention_heads'] == 4 and model['concat_heads'] is True,
            f'{arm}: four concatenated GAT heads required')
    require(model['bond_head_type'] == 'four_state_mlp', f'{arm}: four-state head required')
    require(not model.get('use_charge_partition_features', False), f'{arm}: charge readout forbidden')
    require(loss.get('four_state_conditional_weight', 1.) == 1., f'{arm}: alpha must be 1')
    require(loss.get('four_state_union_brier_weight', 0.) == 0., f'{arm}: Brier loss must be off')
    require(runtime['data']['weighting_scheme'] == 'group_uniform'
            and runtime['data'].get('use_soft_labels') is True
            and runtime['training']['checkpoint_selection'] == 'max_f1'
            and runtime['evaluation']['threshold'] == .5
            and runtime['evaluation'].get('threshold_strategy', 'fixed') == 'fixed',
            f'{arm}: training/evaluation protocol changed')


def smoke():
    prior_completion_barrier()
    require(not (OPS / 'smoke.json').exists(), 'Preflight already exists; refuse repeat')
    devices, _, _ = idle()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(devices[0]), OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    with (OPS / 'preflight.log').open('a', encoding='utf-8') as log:
        subprocess.run([PY, str(REPO / 'graph_transform/scripts/test_four_state_fusion_stage.py'),
                        '--output', str(OPS / 'smoke.json')], cwd=REPO, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    proof = read_json(OPS / 'smoke.json')
    require(proof['status'] == 'passed', 'Preflight did not pass')
    require(all(proof.get('routing_checks', {}).get(key) is True for key in ROUTING_CHECKS),
            'Required routing/alignment/legacy/checkpoint checks did not all pass')
    proof['gitrev'] = revision()
    dump(OPS / 'smoke.json', proof)
    print(json.dumps(proof))


def launch():
    require(not (OPS / 'launch.json').exists(), 'Launch exists; refuse duplicate')
    predecessor = prior_completion_barrier()
    proof = read_json(OPS / 'smoke.json')
    require(proof['status'] == 'passed' and proof['gitrev'] == revision(), 'Missing/stale preflight')
    require(all(file_sha(config(a)) == proof['configs'][a] for a in ARMS),
            'Configs changed since preflight')
    require(set(proof['variants']) == set(ARMS), 'Missing theory-routing preflight results')
    require(all(proof.get('routing_checks', {}).get(key) is True for key in ROUTING_CHECKS),
            'Required routing/alignment/legacy/checkpoint checks missing')
    counts = [proof['variants'][arm]['parameter_count'] for arm in ARMS]
    require(all(isinstance(n, int) and n > 0 for n in counts) and len(set(counts)) == 1,
            'All three routes must retain the same positive parameter count')
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
    root = REPO / 'checkpoints/graph_transform/four_state_fusion_batch' / stamp
    info = {
        'schema_version': 1, 'root': str(root), 'gitrev': revision(),
        'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'baseline_ref': BASELINE_REF, 'control_arm': 'late', 'fusion_stages': ARMS,
        'hidden_dim': 256, 'parameter_count': counts[0], 'predecessor': predecessor,
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
                 'theory routes late/early_only/early_plus_late. Same learned tensors and '
                 'parameter count; early uses mean incident bond-vector addition after '
                 'FiLM and before GCN1, with no direct global injection. Folded group_uniform, '
                 'full-spectrum validation, valid-bond metrics, fixed 0.5 threshold. '
                 'Both predeclared early candidates reported; no test-selected unique winner.',
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


def verify_checkpoint(checkpoint_path, runtime, arm, expected_count):
    """After training, verify all saved weights against the requested route on CPU."""
    import torch
    from models.graph_transformer import GraphTransformer
    from models.theory_fusion import validate_theory_fusion_checkpoint
    from models.utils import build_model_config
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    with torch.random.fork_rng(devices=[]):
        model = GraphTransformer(build_model_config(runtime))
    count = sum(parameter.numel() for parameter in model.parameters())
    require(count == expected_count, f'{arm}: runtime/checkpoint parameter count differs from preflight')
    state = checkpoint['model_state_dict']
    validate_theory_fusion_checkpoint(model, state, checkpoint.get('config'))
    model.load_state_dict(state, strict=True)
    require(all(torch.isfinite(parameter).all().item() for parameter in model.parameters()),
            f'{arm}: nonfinite saved parameters')
    require(model.theory_fusion_stage == ARMS[arm], f'{arm}: checkpoint route mismatch')
    return {'parameter_count': count, 'theory_fusion_stage': model.theory_fusion_stage,
            'strict_state_load_verified': True, 'finite_saved_parameters': True}


def analyze(info):
    root = Path(info['root'])
    status = read_json(root / 'status.json')
    require(status['status'] == 'completed' and len(status['arms']) == 3,
            'Three-arm batch not complete')
    require(all(a['status'] == 'completed' and a['returncode'] == 0 for a in status['arms']),
            'One or more arms did not finish successfully')
    manifest = read_json(root / 'manifest.json')
    summary = read_json(root / 'summary/summary.json')
    proof = read_json(OPS / 'smoke.json')
    require(status['root'] == info['root'] == manifest['root']
            and status['gitrev'] == info['gitrev'] == manifest['gitrev']
            and len(manifest['arms']) == 3 and manifest['folds'] == list(FOLDS),
            'Manifest revision/arm count mismatch')
    require(prior_completion_barrier() == info['predecessor'],
            'Preceding width terminal/review/report binding changed after fusion launch')
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
    require(manifest['control'] == names[0], 'Late-fusion control must be first')
    require(names == [f'{index:02d}_{config(label).stem}' for index, label in enumerate(ARMS, 1)],
            'Manifest arm identities do not match the predeclared fusion routes')
    for arm, arm_status, (label, stage) in zip(manifest['arms'], status['arms'], ARMS.items()):
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
        variant_proof = proof['variants'][label]
        count = variant_proof['parameter_count']
        require(isinstance(count, int) and count > 0, f'{label}: invalid parameter count')
        require(count == info['parameter_count'], f'{label}: parameter count differs across routes')
        resources[label] = {
            **variant_proof, 'hidden_dim': 256, 'head_dim': 64, 'theory_fusion_stage': stage,
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
            runtime = yaml.safe_load(runtime_path.read_text(encoding='utf-8'))
            validate_runtime(runtime, label)
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
            checkpoint_proof = verify_checkpoint(checkpoint, runtime, label, count)
            recovery[label][fold] = {
                'gitrev': info['gitrev'], 'config': str(runtime_path),
                'config_sha256': binding[str(runtime_path)],
                'checkpoint': str(checkpoint), 'checkpoint_sha256': file_sha(checkpoint),
                'metrics_csv': str(raw_path), 'metrics_sha256': binding[str(raw_path)],
                'selection': 'internal full-spectrum validation max_f1; outer test not used',
                **checkpoint_proof,
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
        'fusion_stages': ARMS, 'control_arm': 'late', 'hidden_dim': 256, 'num_attention_heads': 4,
        'metrics': metrics, 'decisions': decisions, 'folds': data,
        'resources': resources, 'recovery': recovery, 'paired_validation_verified': True,
        'predecessor': info['predecessor'], 'preflight_checks': proof.get('routing_checks', {}),
        'routing_interpretation': {
            'early_location': 'after FiLM, before GCN1',
            'mapping': 'same 15->256 bond projection; mean incident bond vectors added to each residue',
            'global_node': 'no direct theory addition; theory can arrive through graph propagation',
            'early_only_head_slot': 'zeros; matching total parameters does not match effective function capacity',
            'new_parameters': 0, 'new_scaling_hyperparameters': 0,
        },
        'scope': 'Exploratory five overlapping sequence-level 80/20 splits, one initialization '
                 'per split/arm; both predeclared early candidates all reported, no test-selected '
                 'unique winner. Padding-free, fixed threshold 0.5. Gates are descriptive '
                 'engineering criteria, not statistical significance.',
    }
    dump(OPS / 'result.json', out)
    dump(OPS / 'checkpoint_registry.json', {
        'baseline_ref': BASELINE_REF, 'gitrev': info['gitrev'], 'root': str(root),
        'candidate_status': 'All arms retained; any passing arm remains unconfirmed pending replication',
        'fusion_stages': recovery,
    })
    write_report(info, status, metrics, decisions, data, names, resources, recovery)
    dump(OPS / 'review_receipt.json', {
        'status': 'passed', 'root': str(root), 'gitrev': info['gitrev'],
        'checks': [
            '15 runs completed; raw metrics/source/config/target hashes bound',
            'Width sweep completed, passed review, and was reported before fusion launch',
            'Every runtime route, 256d, four concatenated heads, alpha=1/Brier=0, no charge readout verified',
            'All training splits, full-row validation sources and validation split SHAs match paired preflight',
            '24 arm-metrics mean/sampleSD/paired delta/sampleSD/wins independently recalculated',
            'Every validation-selected best_model.pt strictly loaded on CPU with matching route/count; path and SHA256 retained',
        ],
        'hashes': binding, 'decisions': decisions,
        'result_sha256': file_sha(OPS / 'result.json'),
        'report_sha256': file_sha(OPS / 'report.md'),
        'registry_sha256': file_sha(OPS / 'checkpoint_registry.json'),
    })


def write_report(info, status, metrics, decisions, data, names, resources, recovery):
    lines = [
        '# 四状态头：理论特征晚期 / 仅早期 / 早期加晚期融合', '',
        f"源码：`{info['gitrev']}`；完成：{status['finished_at']}",
        f'已提升四状态参考提交：`{BASELINE_REF}`。本轮使用重新训练的 late 同轮对照，固定 256 维和 4 头。',
        f"前置宽度实验：`{info['predecessor']['root']}`，已完成、复核通过并报告后才提交本轮。", '',
        '| 指标 | late（对照）均值 ± SD | early_only 均值 ± SD | Δ ± 配对 SD / 改善划分 | early_plus_late 均值 ± SD | Δ ± 配对 SD / 改善划分 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    display_metrics = REPORT_METRICS + (('ex_precision', '谱图平均 Precision %', 100),
                                        ('ex_recall', '谱图平均 Recall %', 100))
    for metric, label, scale in display_metrics:
        cells = []
        for arm in ARMS:
            value = metrics[arm][metric]
            cells.append(f"{value['mean'] * scale:.6f} ± {value['std_ddof1'] * scale:.6f}")
            if arm != 'late':
                cells.append(f"{value['paired_delta_mean'] * scale:+.6f} ± "
                             f"{value['paired_delta_std_ddof1'] * scale:.6f} / {value['win_splits']}/5")
        lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '百分比指标的 Δ 和 SD 使用百分点；Brier 越低越好，改善划分按相应方向计数。', '']
    for arm in list(ARMS)[1:]:
        outcome = '全部预定门槛通过，保留为待复验候选。' if decisions[arm]['improvement_gate'] else '未全部通过预定提升门槛。'
        lines.append(f'{arm}：{outcome} 门槛：' + '；'.join(
            f"{key}={'通过' if passed else '未通过'}" for key, passed in decisions[arm]['gates'].items()))
    lines += ['', '两个早期候选和晚期对照全部报告，不依外层测试成绩选出唯一最佳方案，也不改动冻结论文主表。',
              '三臂均保持原四状态头、3GCN+2GAT、256 维、4 头拼接、α=1、union-Brier=0、无电荷分配分支。',
              '保持折叠 group_uniform、完整原谱内部验证、有效键 mask 和固定 0.5 阈值。',
              '五组序列级 80/20 划分有重叠，各一次初始化；均值/样本 SD/配对差和胜率为描述性结果，工程门槛不代表统计显著。', '',
              '## 融合位置及实验能解释的范围', '',
              '当前 late 模式先完成全部 3 层 GCN 与 2 层 GAT，再把 15 维逐键理论特征经既有 Linear(15→256) 投影，拼入最终 6×256 维键向量；因此理论信息没有作为输入参与骨干消息传递与注意力计算。',
              'early 两臂把同一投影的有效键向量按左右端点求和、再按相邻有效键数平均，相加到残基节点。注入位置固定在 FiLM 条件调制之后、第一层 GCN 之前；全局节点不直接获得该向量，可通过后续图传播接收理论信息。',
              'early_only 的最终理论槽恒为零，early_plus_late 保留最终理论槽并复用前面的同次投影。三臂总参数数目相同且没有新增参数或任意缩放系数，但 early_only 零槽对应的预测头列没有数据梯度，因此总参数量相同不等于有效函数容量完全匹配。',
              'early_plus_late 保留直接读取理论的路径，用于检验让上游骨干也获得理论信息是否有益；early_only 则检验通过传播后的节点能否承载这些信息。早期注入还包含端点邻接平均映射，不能将本轮解释为仅把同一操作沿时间轴移动的完全隔离因果实验。',
              '15 维仍仅由序列计算，不增加观测图谱输入，也不代表空间接触、三维结构或真实断裂物理模拟。',
              '预检对 GCN1 输入做 autograd 路径检查：late 对理论输入无直接骨干梯度路径，early 两臂存在该路径且 padding 梯度为零；这验证信息路由实现，不证明训练已学到更准确的机制。完整实证字段保存在 result.json 的 preflight_checks 和 smoke.json。', '',
              '| 模式 | GAT 头数 × 每头维度 | 参数量 | 五划分臂耗时（小时） |',
              '|---|---:|---:|---:|']
    for arm in ARMS:
        value = resources[arm]
        lines.append(f"| {arm} | 4 × 64 | {value['parameter_count']:,} | {value['wall_time_seconds'] / 3600:.4f} |")
    lines += ['', '耗时为每臂五划分调度的墙钟时间，受设备、并行调度和早停影响，不是等步数计算效率。',
              '预检峰值显存、完整训练/验证批量和对齐核验保存在 result.json 的 resources 和 smoke.json。', '']
    for arm, name in zip(ARMS, names):
        if arm == 'late':
            continue
        lines += [f'## {arm} 逐划分变化（相对同轮 late）', '',
                  '| 划分 | Δ F1 百分点 | Δ accuracy 百分点 | Δ AUC | Δ 谱图调和 F1 百分点 | Δ q-Brier ↓ | Δ q_seq |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for fold in FOLDS:
            values = [f"{(data[name][fold][metric] - data[names[0]][fold][metric]) * scale:+.6f}"
                      for metric, _, scale in REPORT_METRICS]
            lines.append('| ' + fold + ' | ' + ' | '.join(values) + ' |')
        lines.append('')
    lines += ['## 恢复记录', '',
              '每个模式、每个划分的 validation max-F1 权重路径、SHA256、运行配置和原始指标已写入 `checkpoint_registry.json`；没有删除候选。每份权重均在 pod 的 CPU 上按相应模式严格载入并核对参数数目。',
              f"批次：`{info['root']}`", '', '| 模式 | 划分 | best_model.pt | SHA256 |', '|---|---|---|---|']
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
        detail = None
        if status_path.exists():
            try:
                detail = read_json(status_path).get('error')
            except (OSError, ValueError):
                detail = 'Batch status receipt could not be read; original failure retained above'
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
