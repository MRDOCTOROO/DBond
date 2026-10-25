"""Audited v2 fixed three-seed ensemble of the single-global-node k=5 model.

Run ONLY on the authorized DBond pod.  This is an inference experiment, not a
new training run: for each of the five original sequence splits, average the
union probabilities from the independently trained seed-42/43/45 checkpoints
of that SAME split.  Never average checkpoints from different splits.

The members, arithmetic mean, threshold and primary comparison are frozen in
this file.  No validation/test labels are used to choose weights or members.
The first frozen batch failed on a q_seq difference exactly consistent with
one near-tie rank swap; this version records historic ranking drift as a diagnostic
while continuing to gate source identity on SHA and smooth/classification metrics.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'graph_transform'), str(REPO / 'graph_transform/scripts')]

import numpy as np
import torch
import yaml
from data.graph_dataset import GraphDataset, GraphDataLoader
from data.row_validation import file_sha
from models.graph_transformer import GraphTransformer
from models.dynamic_edges import validate_edge_state_checkpoint
from models.bond_fusion_heads import validate_bond_fusion_checkpoint
from models.node_context_conv import validate_node_conv_checkpoint
from models.utils import build_model_config, validate_virtual_node_checkpoint
from training.metrics import BinaryBondMetrics, batch_group_keys
from train_5fold import _stable_fold_seed
from train_graph_model import apply_ablation_config

FOLDS = ('1222', '2252', '3514', '6072', '9075')
SOURCES = {
    'seed42': {
        'ops': Path('/mnt/pvc/graphtrans/four_state_parallel_global_replication_ops'),
        'revision': '111c23313da9ae11a7ee33f8357be0a1876086ec',
        'root': '/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/'
                'four_state_parallel_global_replication_batch/20261025_052100',
        'arm': 'single', 'base_seed': 42, 'receipt': 'review_receipt.json',
    },
    'seed43': {
        'ops': Path('/mnt/pvc/graphtrans/four_state_node_conv_replication_ops'),
        'revision': '8ec60bca58b879cba5632379141238d44cc2074e',
        'root': '/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/'
                'four_state_node_conv_replication_batch/20261023_142354',
        'arm': 'single_k5', 'base_seed': 43, 'receipt': 'heartbeat_review.json',
    },
    'seed45': {
        'ops': Path('/mnt/pvc/graphtrans/four_state_parallel_global_ops'),
        'revision': 'ce0185c1c313a9f36410cd33701699fe8df53579',
        'root': '/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/'
                'four_state_parallel_global_batch/20261024_171434',
        'arm': 'single', 'base_seed': 45, 'receipt': 'review_receipt.json',
    },
}
LABELS = tuple(SOURCES) + ('equal_probability_ensemble',)
METRICS = ('f1', 'accuracy', 'auc', 'ex_precision', 'ex_recall',
           'ex_f1_harmonic', 'q_brier', 'q_seq', 'top10')
SOURCE_KEYS = {'q_seq': 'q_spearman_pep_seq',
               'top10': 'q_top10_enrichment_seq'}
PRIMARY_REFERENCE = 'seed42'
PARAMETERS = 2_002_284
EXPECTED_BASELINE = '4da48eeab50decd0a31fbf64f2f179478a6624a7'
PRODUCTION_PATHS = ('graph_transform/models', 'graph_transform/data',
                    'graph_transform/evaluation', 'graph_transform/training',
                    'graph_transform/scripts/train_graph_model.py',
                    'graph_transform/scripts/train_5fold.py')


def require(condition, why):
    if not condition:
        raise ValueError(why)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')


def in_root(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def git_bytes(revision, name):
    return subprocess.check_output(['git', 'show', f'{revision}:{name}'], cwd=REPO)


def verify_live_production():
    """A new report script cannot silently change inference/model semantics."""
    tracked = subprocess.check_output(['git', 'ls-files', *PRODUCTION_PATHS],
                                      cwd=REPO, text=True).splitlines()
    require(tracked, 'No tracked production source files')
    mismatches = []
    for name in tracked:
        path = REPO / name
        if not path.is_file() or file_sha(path) != hashlib.sha256(
                git_bytes(SOURCES['seed42']['revision'], name)).hexdigest():
            mismatches.append(name)
    require(not mismatches, f'Production source differs from audited seed42 source: {mismatches}')
    return {'reference_revision': SOURCES['seed42']['revision'],
            'production_file_count': len(tracked),
            'reference_git_blob_match': True}


def metric_from_dict(values):
    row = {key: float(values[SOURCE_KEYS.get(key, key)])
           for key in METRICS if key != 'ex_f1_harmonic'}
    p, r = row['ex_precision'], row['ex_recall']
    row['ex_f1_harmonic'] = 2 * p * r / (p + r) if p + r else 0.
    require(all(math.isfinite(value) for value in row.values()), 'Nonfinite metric')
    return row


def read_historical_metric(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        raw = {row['metric']: float(row['value']) for row in csv.DictReader(stream)}
    return metric_from_dict(raw)


def load_sources():
    sources, proofs = {}, {}
    for label, spec in SOURCES.items():
        ops = spec['ops']
        registry_path, receipt_path = ops / 'checkpoint_registry.json', ops / spec['receipt']
        registry, receipt = read_json(registry_path), read_json(receipt_path)
        require(registry['root'] == receipt['root'] == spec['root']
                and registry['gitrev'] == receipt['gitrev'] == spec['revision']
                and receipt['status'] == 'passed'
                and receipt['registry_sha256'] == file_sha(registry_path)
                and registry['baseline_ref'] == EXPECTED_BASELINE,
                f'{label}: source batch not tied to passing independent review')
        require(file_sha(ops / 'result.json') == receipt['result_sha256']
                and file_sha(ops / 'report.md') == receipt['report_sha256'],
                f'{label}: original result/report no longer match independent review')
        terminal = read_json(ops / 'terminal.json')
        require(terminal['status'] == 'completed'
                and terminal['root'] == spec['root']
                and terminal['gitrev'] == spec['revision'],
                f'{label}: terminal completion binding failed')
        smoke = read_json(ops / 'smoke.json')
        require(smoke['status'] == 'passed', f'{label}: preflight not passed')
        require(spec['arm'] in registry['arms']
                and set(registry['arms'][spec['arm']]) == set(FOLDS),
                f'{label}: missing or extra fold registry entries')
        sources[label] = registry['arms'][spec['arm']]
        proofs[label] = {'root': spec['root'], 'gitrev': spec['revision'],
                         'registry': str(registry_path),
                         'registry_sha256': file_sha(registry_path),
                         'review_receipt': str(receipt_path),
                         'review_receipt_sha256': file_sha(receipt_path),
                         'terminal_sha256': file_sha(ops / 'terminal.json'),
                         'result_sha256': receipt['result_sha256'],
                         'report_sha256': receipt['report_sha256'],
                         'smoke_sha256': file_sha(ops / 'smoke.json')}
    return sources, proofs


def verify_checkpoint(label, fold, item):
    spec = SOURCES[label]
    reviewed_hashes = read_json(spec['ops'] / spec['receipt'])['hashes']
    for key in ('checkpoint', 'config', 'metrics_csv'):
        require(in_root(item[key], spec['root']) and Path(item[key]).is_file(),
                f'{label}/{fold}: {key} missing or outside frozen batch root')
        require(f'fold_{fold}' in Path(item[key]).parts,
                f'{label}/{fold}: wrong fold path in {key}')
        require(file_sha(item[key]) == item[f'{key}_sha256' if key != 'metrics_csv'
                                                else 'metrics_sha256'],
                f'{label}/{fold}: {key} SHA mismatch')
        if key in ('config', 'metrics_csv'):
            require(reviewed_hashes.get(str(item[key])) == file_sha(item[key]),
                    f'{label}/{fold}: {key} differs from independently reviewed source')
    require(item['gitrev'] == spec['revision']
            and item.get('selection', 'internal full-spectrum validation max_f1; outer test not used')
                == 'internal full-spectrum validation max_f1; outer test not used'
            and Path(item['checkpoint']).name == 'best_model.pt',
            f'{label}/{fold}: revision/selection mismatch')
    # Fold YAML is the immutable *raw* recipe.  The original train/eval entry
    # points apply the ablation before constructing either a model or a graph.
    # In particular, pre_synthesis masks the observed precursor intensity and
    # scan number; replaying the raw YAML would leak those post-observation
    # features into the graph even if checkpoint restoration were strict.
    raw_runtime = yaml.safe_load(Path(item['config']).read_text(encoding='utf-8'))
    require(raw_runtime.get('ablation', {}).get('pre_synthesis') is True,
            f'{label}/{fold}: expected frozen pre_synthesis recipe')
    runtime = apply_ablation_config(copy.deepcopy(raw_runtime))
    saved = torch.load(item['checkpoint'], map_location='cpu', weights_only=False)
    model_config = saved['config']
    require(runtime['experiment']['seed'] != 0
            and runtime['training']['validation_split_seed'] != 0
            and runtime['data']['use_soft_labels'] is True
            and runtime['data']['graph_strategy'] == 'distance'
            and runtime['data']['max_seq_len'] == 32
            and runtime['training']['checkpoint_selection'] == 'max_f1'
            and runtime['evaluation']['threshold'] == .5
            and runtime['evaluation']['threshold_strategy'] == 'fixed',
            f'{label}/{fold}: runtime split/metric configuration mismatch')
    require(model_config['hidden_dim'] == 256
            and model_config['bond_head_type'] == 'four_state_mlp'
            and model_config.get('node_conv_mode') == 'single_k5'
            and model_config.get('virtual_node_mode', 'single') == 'single'
            and model_config.get('edge_state_mode', 'static') == 'static'
            and model_config.get('bond_fusion_mode', 'concat') == 'concat'
            and model_config.get('theory_fusion_stage', 'late') == 'late'
            and model_config['num_attention_heads'] == 4
            and model_config['state_feature_mask'] == [True, True, False]
            and model_config['env_feature_mask'] == [True, False]
            and runtime['loss']['four_state_conditional_weight'] == 1.
            and runtime['loss']['four_state_union_brier_weight'] == 0.,
            f'{label}/{fold}: not the frozen single-global/k5/four-state architecture')
    expected_rng = {'training_seed': runtime['experiment']['seed'],
                    'validation_split_seed': runtime['training']['validation_split_seed'],
                    'hidden_dim': 256, 'bond_head_type': 'four_state_mlp',
                    'theory_fusion_stage': 'late',
                    'four_state_conditional_weight': 1.,
                    'four_state_union_brier_weight': 0.}
    require(saved.get('extra_state', {}).get('training_rng_protocol') == expected_rng
            and item['checkpoint_verification']['checkpoint_training_rng_protocol'] == expected_rng,
            f'{label}/{fold}: actual saved training/split RNG differs from registry')
    require(expected_rng['training_seed'] == _stable_fold_seed(spec['base_seed'], fold)
            and expected_rng['validation_split_seed'] == _stable_fold_seed(42, fold),
            f'{label}/{fold}: historical base training/split seed protocol changed')
    # Compare every effective ModelConfig field.  The saved checkpoint config
    # is effective, while the YAML masks are intentionally still all true.
    normalized_model = build_model_config(runtime).to_dict()
    require(build_model_config({'model': model_config, 'data': runtime['data']}).to_dict()
            == normalized_model,
            f'{label}/{fold}: saved model config differs from its frozen run config')
    with torch.random.fork_rng(devices=[]):
        model = GraphTransformer(build_model_config(runtime))
    require(sum(parameter.numel() for parameter in model.parameters()) == PARAMETERS,
            f'{label}/{fold}: parameter count mismatch')
    validate_edge_state_checkpoint(model, saved['model_state_dict'], model_config)
    validate_bond_fusion_checkpoint(model, saved['model_state_dict'], model_config)
    validate_node_conv_checkpoint(model, saved['model_state_dict'], model_config)
    validate_virtual_node_checkpoint(model, saved['model_state_dict'], model_config)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    require(all(torch.isfinite(value).all().item() for value in model.parameters()),
            f'{label}/{fold}: nonfinite checkpoint weights')
    model.eval()
    del saved
    checkpoint = Path(item['checkpoint'])
    provenance_path = checkpoint.parent / 'row_validation_provenance.json'
    target_path = checkpoint.parent / 'four_state_provenance.json'
    require(provenance_path.is_file() and target_path.is_file(),
            f'{label}/{fold}: source/target provenance missing')
    provenance, target = read_json(provenance_path), read_json(target_path)
    target_manifest = Path('/mnt/pvc/graphtrans/four_state_targets_v1') / f'fold_{fold}' / 'manifest.json'
    require(target_manifest.is_file()
            and Path(target['manifest']).resolve() == target_manifest.resolve()
            and str(target['fold']) == fold
            and file_sha(target_manifest) == target['manifest_sha256'],
            f'{label}/{fold}: frozen four-state target manifest changed')
    manifest = read_json(target_manifest)
    sidecar = target_manifest.parent / manifest['sidecar']['path']
    require(str(manifest['fold']) == fold
            and sidecar.is_file()
            and sidecar.resolve().is_relative_to(target_manifest.parent.resolve())
            and file_sha(sidecar) == manifest['sidecar']['sha256'] == target['sidecar_sha256'],
            f'{label}/{fold}: four-state target sidecar or fold binding changed')
    require(provenance['mode'] == 'full_original_training_spectra_by_internal_val_sequence'
            and provenance['outer_test_used_for_internal_validation'] is False
            and provenance['split_seed'] == expected_rng['validation_split_seed'],
            f'{label}/{fold}: internal validation source mismatch')
    hashes = {name: file_sha(provenance[key]) for name, key in
              (('train', 'split_train_csv'), ('val', 'split_val_csv'),
               ('source', 'source_path'))}
    require(hashes['source'] == provenance['source_sha256'],
            f'{label}/{fold}: original training-spectrum source SHA mismatch')
    smoke_fold = read_json(spec['ops'] / 'smoke.json')['folds'][fold]
    require(all(hashes[key] == smoke_fold[f'{key}_sha256'] for key in ('train', 'val', 'source'))
            and expected_rng['training_seed'] == smoke_fold['training_seed']
            and expected_rng['validation_split_seed'] == smoke_fold['split_seed'],
            f'{label}/{fold}: archived preflight train/val/source/seed binding differs')
    for path in (provenance_path, target_path, provenance['split_train_csv'],
                 provenance['split_val_csv'], provenance['source_path']):
        if str(path) in reviewed_hashes:
            require(file_sha(path) == reviewed_hashes[str(path)],
                    f'{label}/{fold}: reviewed train/validation/target SHA changed: {path}')
    test_path = Path(runtime['data']['test_csv_path'])
    test_path = (test_path if test_path.is_absolute() else REPO / test_path).resolve()
    require(test_path.is_file(), f'{label}/{fold}: outer-test CSV missing')
    historical = read_historical_metric(item['metrics_csv'])
    proof = {'config': item['config'], 'config_sha256': item['config_sha256'],
             'checkpoint': item['checkpoint'], 'checkpoint_sha256': item['checkpoint_sha256'],
             'metrics_csv': item['metrics_csv'], 'metrics_sha256': item['metrics_sha256'],
             'training_seed': expected_rng['training_seed'],
             'validation_split_seed': expected_rng['validation_split_seed'],
             'train_val_source_sha256': hashes,
             'provenance_path': str(provenance_path),
             'provenance_sha256': file_sha(provenance_path),
             'target_provenance_sha256': file_sha(target_path),
             'target_manifest': str(target_manifest),
             'target_manifest_sha256': target['manifest_sha256'],
             'normalized_full_model_config': normalized_model,
             'test_path': str(test_path), 'test_sha256': file_sha(test_path),
             'historical_metrics': historical,
             'parameter_count': PARAMETERS,
             'strict_cpu_restore': True}
    prediction = item.get('prediction_recomputation')
    if prediction:
        require(proof['test_sha256'] == prediction['test_sha256']
                and file_sha(prediction['prediction_csv']) == prediction['prediction_sha256'],
                f'{label}/{fold}: independently audited same-pass source changed')
    return runtime, model, proof


def verify_fold_sources(fold, sources):
    runtimes, models, proofs = {}, {}, {}
    for label in SOURCES:
        runtimes[label], models[label], proofs[label] = verify_checkpoint(label, fold,
                                                                        sources[label][fold])
    keys = ('train', 'val', 'source')
    for key in keys:
        require(len({proofs[label]['train_val_source_sha256'][key] for label in SOURCES}) == 1,
                f'{fold}: {key} split/source SHA differs across seeds')
    for key in ('validation_split_seed', 'test_sha256', 'target_manifest_sha256'):
        require(len({proofs[label][key] for label in SOURCES}) == 1,
                f'{fold}: {key} differs across seeds')
    canonical = proofs[PRIMARY_REFERENCE]['normalized_full_model_config']
    for label in SOURCES:
        require(proofs[label]['normalized_full_model_config'] == canonical,
                f'{fold}: full effective model config differs for {label} vs {PRIMARY_REFERENCE}')
    require(len({proofs[label]['training_seed'] for label in SOURCES}) == 3,
            f'{fold}: three members do not have independent training seeds')
    require(len({runtime['data']['max_seq_len'] for runtime in runtimes.values()}) == 1
            and len({runtime['data']['graph_strategy'] for runtime in runtimes.values()}) == 1
            and len({runtime['device']['use_amp'] for runtime in runtimes.values()}) == 1,
            f'{fold}: inference settings differ')
    return runtimes, models, proofs


def summarize(metric_rows):
    results = {}
    for label in LABELS:
        results[label] = {}
        for metric in METRICS:
            values = [metric_rows[fold][label][metric] for fold in FOLDS]
            result = {'mean': statistics.mean(values),
                      'std_ddof1': statistics.stdev(values),
                      'by_fold': dict(zip(FOLDS, values))}
            if label != PRIMARY_REFERENCE:
                deltas = [metric_rows[fold][label][metric]
                          - metric_rows[fold][PRIMARY_REFERENCE][metric] for fold in FOLDS]
                direction = -1 if metric == 'q_brier' else 1
                result.update(paired_delta_mean=statistics.mean(deltas),
                              paired_delta_std_ddof1=statistics.stdev(deltas),
                              paired_deltas=dict(zip(FOLDS, deltas)),
                              wins=sum(direction * d > 1e-12 for d in deltas),
                              ties=sum(abs(d) <= 1e-12 for d in deltas),
                              losses=sum(direction * d < -1e-12 for d in deltas))
            results[label][metric] = result
    # The second reference was frozen before any ensemble result was read:
    # arithmetic mean of the *three individual metric values within each fold*.
    # It is a descriptive expectation for a random member, not another model
    # or a test-selected combination of members.
    results['constituent_metric_mean'] = {}
    results['ensemble_minus_constituent_metric_mean'] = {}
    results['ensemble_minus_each_member'] = {member: {} for member in SOURCES}
    for metric in METRICS:
        member_mean_by_fold = {fold: statistics.mean(
            metric_rows[fold][member][metric] for member in SOURCES) for fold in FOLDS}
        member_values = [member_mean_by_fold[fold] for fold in FOLDS]
        deltas = [metric_rows[fold]['equal_probability_ensemble'][metric]
                  - member_mean_by_fold[fold] for fold in FOLDS]
        direction = -1 if metric == 'q_brier' else 1
        results['constituent_metric_mean'][metric] = {
            'mean': statistics.mean(member_values),
            'std_ddof1': statistics.stdev(member_values),
            'by_fold': member_mean_by_fold,
            'definition': 'Arithmetic mean of seed42/43/45 metric values within each fixed fold'}
        results['ensemble_minus_constituent_metric_mean'][metric] = {
            'paired_delta_mean': statistics.mean(deltas),
            'paired_delta_std_ddof1': statistics.stdev(deltas),
            'paired_deltas': dict(zip(FOLDS, deltas)),
            'wins': sum(direction * delta > 1e-12 for delta in deltas),
            'ties': sum(abs(delta) <= 1e-12 for delta in deltas),
            'losses': sum(direction * delta < -1e-12 for delta in deltas),
            'reference': 'constituent_metric_mean',
            'prespecified': True}
        for member in SOURCES:
            member_deltas = [metric_rows[fold]['equal_probability_ensemble'][metric]
                             - metric_rows[fold][member][metric] for fold in FOLDS]
            results['ensemble_minus_each_member'][member][metric] = {
                'paired_delta_mean': statistics.mean(member_deltas),
                'paired_delta_std_ddof1': statistics.stdev(member_deltas),
                'paired_deltas': dict(zip(FOLDS, member_deltas)),
                'wins': sum(direction * delta > 1e-12 for delta in member_deltas),
                'ties': sum(abs(delta) <= 1e-12 for delta in member_deltas),
                'losses': sum(direction * delta < -1e-12 for delta in member_deltas),
                'reference': member,
                'prespecified': True}
    return results


def update_metric(calculator, logits, batch):
    calculator.update(logits.detach().cpu(), batch['labels'].detach().cpu(),
                      label_mask=batch['label_mask'].detach().cpu(),
                      soft_targets=batch['soft_labels'].detach().cpu(),
                      sequences=batch['sequences'], group_keys=batch_group_keys(batch))


def compare_historical(observed, historic, label, fold):
    # The v1 batch stopped at seed42/fold3514: a single near-tied sequence
    # swap would change Spearman by exactly 0.0002601059, the observed drift,
    # despite identical source hashes,
    # strict checkpoint restoration and tiny smooth/classification differences.
    # Rank metrics are discontinuous under such GPU replay perturbations, so
    # report their archived drift without making exact historical rank equality
    # a source-identity gate. The FINAL comparison always uses same-pass outputs.
    tolerances = {'f1': 2e-5, 'accuracy': 2e-5, 'auc': 2e-4,
                  'ex_precision': 2e-5, 'ex_recall': 2e-5,
                  'ex_f1_harmonic': 2e-5, 'q_brier': 2e-5,
                  'q_seq': None, 'top10': None}
    differences = {key: abs(observed[key] - historic[key]) for key in METRICS}
    identity_metrics = tuple(key for key in METRICS if key not in ('q_seq', 'top10'))
    require(all(differences[key] <= tolerances[key] for key in identity_metrics),
            f'{label}/{fold}: checkpoint replay disagrees with archived smooth/classification metrics: {differences}')
    return {'absolute_differences': differences, 'tolerances': tolerances,
            'replayed_from_checkpoint': True,
            'source_identity_gate_metrics': identity_metrics,
            'historical_q_seq_top10_diagnostic_only': True}


def verify_graph_input_parity(dataset, runtimes):
    """Reject silent changes in graph construction between old configurations."""
    selected = sorted({0, 1, len(dataset) // 2, len(dataset) - 1})
    keys = ('edge_index', 'edge_attr', 'edge_types', 'edge_distances',
            'bond_edge_map', 'labels', 'soft_labels', 'node_len', 'seq_len')
    for label in SOURCES:
        if label == PRIMARY_REFERENCE:
            continue
        other = GraphDataset(str(dataset.csv_path), runtimes[label],
                             max_seq_len=runtimes[label]['data']['max_seq_len'],
                             graph_strategy=runtimes[label]['data']['graph_strategy'],
                             augmentation=False, split='test')
        require(len(other) == len(dataset), f'{label}: test row count changed')
        for row in selected:
            a, b = dataset[row], other[row]
            for key in keys:
                left, right = a[key], b[key]
                same = torch.equal(left, right) if isinstance(left, torch.Tensor) else left == right
                require(same, f'{label}/{row}: {key} differs from shared inference graph')
    return {'sample_rows': selected, 'checked_fields': keys,
            'same_test_dataset_graph_and_labels': True}


def infer_fold(fold, runtimes, models, proofs, out, device, batch_size):
    runtime = runtimes[PRIMARY_REFERENCE]
    dataset = GraphDataset(proofs[PRIMARY_REFERENCE]['test_path'], runtime,
                           max_seq_len=runtime['data']['max_seq_len'],
                           graph_strategy=runtime['data']['graph_strategy'],
                           augmentation=False, split='test')
    require(len(dataset) > 0 and dataset.data['seq'].nunique() > 1,
            f'{fold}: no usable sequence ranking set')
    parity = verify_graph_input_parity(dataset, runtimes)
    loader = GraphDataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=device.type == 'cuda', drop_last=False)
    width = runtime['data']['max_seq_len'] - 1
    n_rows = len(dataset)
    q_rows = {label: np.zeros((n_rows, width), dtype=np.float32) for label in LABELS}
    targets = np.zeros((n_rows, width), dtype=np.uint8)
    soft = np.zeros((n_rows, width), dtype=np.float32)
    valid = np.zeros((n_rows, width), dtype=np.bool_)
    charges, nces = np.zeros(n_rows, dtype=np.float32), np.zeros(n_rows, dtype=np.float32)
    sequences = []
    calculators = {label: BinaryBondMetrics({'threshold': .5, 'threshold_strategy': 'fixed'},
                   allow_target_aware_threshold=False) for label in LABELS}
    for model in models.values():
        model.to(device).eval()
    cursor = 0
    first_batch_union_checked = {label: False for label in SOURCES}
    use_amp = bool(runtime['device']['use_amp'] and device.type == 'cuda')
    with torch.inference_mode():
        for batch in loader:
            moved = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor)
                     else value for key, value in batch.items()}
            labels, mask, soft_target = moved['labels'], moved['label_mask'].bool(), moved['soft_labels']
            require(labels.shape == mask.shape == soft_target.shape,
                    f'{fold}: label/soft/mask shape mismatch')
            require(torch.isfinite(soft_target[mask]).all().item(),
                    f'{fold}: nonfinite q targets')
            batch_len, batch_width = labels.shape
            require(batch_width <= width and cursor + batch_len <= n_rows,
                    f'{fold}: invalid batch width/row count')
            target_slice = slice(cursor, cursor + batch_len)
            targets[target_slice, :batch_width] = labels.detach().cpu().numpy().astype(np.uint8)
            soft[target_slice, :batch_width] = soft_target.detach().cpu().numpy()
            valid[target_slice, :batch_width] = mask.detach().cpu().numpy()
            charges[target_slice] = moved['charges'].detach().cpu().numpy()
            nces[target_slice] = moved['nces'].detach().cpu().numpy()
            sequences.extend(moved['sequences'])
            probabilities = []
            for label in SOURCES:
                model = models[label]
                captured = []
                hook = None
                if not first_batch_union_checked[label]:
                    hook = model.bond_head[3].register_forward_hook(
                        lambda module, args, output: captured.append(output.detach().float()))
                try:
                    with torch.amp.autocast('cuda', enabled=use_amp) if device.type == 'cuda' else contextlib.nullcontext():
                        logits = model(moved)
                finally:
                    if hook is not None:
                        hook.remove()
                require(logits.shape == mask.shape and torch.isfinite(logits[mask]).all().item(),
                        f'{label}/{fold}: invalid union logits')
                if not first_batch_union_checked[label]:
                    require(len(captured) == 1 and captured[0].shape == (int(mask.sum()), 4),
                            f'{label}/{fold}: four-state head did not produce four logits per valid bond')
                    from models.four_state import union_logit
                    require(torch.allclose(logits[mask].float(), union_logit(captured[0]),
                                           atol=3e-5, rtol=3e-5),
                            f'{label}/{fold}: q is not the none-vs-observed four-state union')
                    first_batch_union_checked[label] = True
                update_metric(calculators[label], logits, moved)
                q = torch.sigmoid(logits.float())
                q_rows[label][target_slice, :batch_width] = q.detach().cpu().numpy()
                probabilities.append(q)
            avg_q = torch.stack(probabilities, dim=0).mean(dim=0)
            avg_logit = torch.logit(avg_q.clamp(1e-7, 1 - 1e-7))
            require(torch.isfinite(avg_logit[mask]).all().item(),
                    f'{fold}: nonfinite ensemble union logits')
            update_metric(calculators['equal_probability_ensemble'], avg_logit, moved)
            q_rows['equal_probability_ensemble'][target_slice, :batch_width] = avg_q.cpu().numpy()
            cursor += batch_len
    require(cursor == n_rows and sequences == dataset.data['seq'].astype(str).tolist(),
            f'{fold}: test rows not in source order')
    require(np.array_equal(valid.sum(axis=1), np.asarray([max(len(seq) - 1, 0)
                                                         for seq in sequences])),
            f'{fold}: padding mask does not match sequence bond counts')
    require(np.allclose(q_rows['equal_probability_ensemble'],
                        np.mean(np.stack([q_rows[label] for label in SOURCES]), axis=0),
                        rtol=0, atol=1.2e-7),
            f'{fold}: ensemble is not the fixed three-seed arithmetic probability mean')
    computed = {label: metric_from_dict(calculator.compute())
                for label, calculator in calculators.items()}
    historical_audits = {label: compare_historical(computed[label],
                         proofs[label]['historical_metrics'], label, fold) for label in SOURCES}
    archive = out / f'fold_{fold}_same_pass_q.npz'
    np.savez_compressed(archive, **q_rows, labels=targets, soft_q=soft, valid=valid,
                        sequences=np.asarray(sequences), charges=charges, nces=nces)
    return computed, {'rows': n_rows, 'valid_bonds': int(valid.sum()),
                      'input_parity': parity, 'historical_replay': historical_audits,
                      'same_pass_probability_archive': str(archive),
                      'same_pass_probability_archive_sha256': file_sha(archive),
                      'union_formula_verified': first_batch_union_checked,
                      'fixed_mean_verified': True}


def write_report(out, result):
    summary = result['summary']
    lines = ['# 单全局节点＋k5：固定三训练种子同折概率集成', '',
             '这是探索性、无新增训练的推理对照。每个外部划分只使用该划分的 seed42/43/45 '
             '检查点；绝不跨划分集成。三模型对四状态概率求 q=1−P(none)，然后以 '
             '1/3 固定权重平均 q。阈值固定 0.5，所有分类和排序指标只统计有效肽键。', '',
             '旧批次的严格历史 q_seq 重放门曾因与近并列换位一致的差异而失败；本批次另起根目录，'
             '统一对全部五折从头同精度推理，历史 q_seq/Top10 差值保留为诊断，不据此挑选运行结果。',
             '训练种子和五个外部划分并非十五个独立数据集。之前多轮使用过这些划分进行模型探索，'
             '因此本次结果不能替代未触及的序列隔离确认集。', '',
             '| 指标 | seed42 | seed43 | seed45 | 固定集成 | 集成−seed42 | 对seed42胜出 | 集成−三单模指标均值 | 对均值胜出 |',
             '|:--|--:|--:|--:|--:|--:|--:|--:|--:|']
    for key in METRICS:
        row = [summary[label][key]['mean'] for label in LABELS]
        contrast = summary['equal_probability_ensemble'][key]
        member_mean_contrast = summary['ensemble_minus_constituent_metric_mean'][key]
        scale = 100 if key in ('f1', 'accuracy', 'ex_precision', 'ex_recall', 'ex_f1_harmonic') else 1
        lines.append('| ' + key + ' | ' + ' | '.join(f'{number*scale:.6f}' for number in row)
                     + f' | {contrast["paired_delta_mean"]*scale:+.6f}'
                     + f' | {contrast["wins"]}/5'
                     + f' | {member_mean_contrast["paired_delta_mean"]*scale:+.6f}'
                     + f' | {member_mean_contrast["wins"]}/5 |')
    lines.extend(['', '所有均值由五个固定外部划分计算；每个划分的完整数值、样本SD、'
                  '配对差SD、权重/config/source SHA 和原始同次前向概率归档均在 result.json。',
                  '分类 F1 采用有效键 micro-F1；谱图 F1 是谱图平均 P/R 的调和，'
                  '排序 q_seq 是先按 (序列、charge、NCE) 去重，再对同序列条件均匀平均的 Spearman；'
                  'Top10 是最高预测序列前10%的平均真实 q 与全集平均真实 q 之比。', '',
                  '不能只凭集成相对较弱种子的平均优势宣称改进；报告同时保留三个单模型及固定 seed42 参照。'])
    lines.append('“三单模指标均值”在每一划分内先取 seed42/43/45 指标的算术平均，'
                 '再与该划分的集成指标配对；这一对照在查看集成结果前已固定，不涉及挑选种子。')
    lines.extend(['', '## 集成与三个固定单模型的逐一配对比较', '',
                  '下表完整保留所有种子比较，供判断收益是否仅来自较弱成员；'
                  '这些差值不参与成员或权重选择。F1、accuracy、谱图P/R/F1差值以百分点表示。', '',
                  '| 指标 | 参照单模型 | 集成−参照，五折均值 | 配对SD | 胜/平/负 |',
                  '|:--|:--|--:|--:|:--:|'])
    for key in METRICS:
        scale = 100 if key in ('f1', 'accuracy', 'ex_precision', 'ex_recall', 'ex_f1_harmonic') else 1
        for member in SOURCES:
            contrast = summary['ensemble_minus_each_member'][member][key]
            lines.append(f'| {key} | {member} | {contrast["paired_delta_mean"]*scale:+.6f} | '
                         f'{contrast["paired_delta_std_ddof1"]*scale:.6f} | '
                         f'{contrast["wins"]}/{contrast["ties"]}/{contrast["losses"]} |')
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    rows = []
    for key in METRICS:
        values = [summary[label][key]['mean'] for label in LABELS]
        contrast = summary['equal_probability_ensemble'][key]
        member_mean_contrast = summary['ensemble_minus_constituent_metric_mean'][key]
        rows.append('<tr><th>' + html.escape(key) + '</th>'
                    + ''.join(f'<td>{value:.6f}</td>' for value in values)
                    + f'<td>{contrast["paired_delta_mean"]:+.6f}</td>'
                    + f'<td>{contrast["wins"]}/5</td>'
                    + f'<td>{member_mean_contrast["paired_delta_mean"]:+.6f}</td>'
                    + f'<td>{member_mean_contrast["wins"]}/5</td></tr>')
    paired_rows = []
    for key in METRICS:
        scale = 100 if key in ('f1', 'accuracy', 'ex_precision', 'ex_recall', 'ex_f1_harmonic') else 1
        for member in SOURCES:
            contrast = summary['ensemble_minus_each_member'][member][key]
            paired_rows.append(f'<tr><th>{html.escape(key)}</th><td>{member}</td>'
                               f'<td>{contrast["paired_delta_mean"]*scale:+.6f}</td>'
                               f'<td>{contrast["paired_delta_std_ddof1"]*scale:.6f}</td>'
                               f'<td>{contrast["wins"]}/{contrast["ties"]}/{contrast["losses"]}</td></tr>')
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8">'
            '<title>DBond k5 same-fold seed ensemble</title>'
            '<style>body{font:16px system-ui;max-width:1120px;margin:3rem auto;color:#143047}'
            'table{border-collapse:collapse;width:100%}th,td{padding:.6rem;border-bottom:1px solid #cad4de}'
            'td{text-align:right}thead{background:#dfeef6}.note{background:#edf6f3;padding:1rem}</style>'
            '<h1>单全局节点＋k5：固定三种子集成</h1>'
            '<p class="note">同一外部划分内平均 seed42、43、45 的四状态可观测断裂概率；'
            '没有新增训练、权重搜索、阈值选择或跨划分混合。结果是探索性的。</p>'
            '<table><thead><tr><th>指标</th><th>seed42</th><th>seed43</th><th>seed45</th>'
            '<th>固定集成</th><th>Δ vs seed42</th><th>胜出</th>'
            '<th>Δ vs 三单模指标均值</th><th>胜出</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>'
            '<h2>与每个固定种子逐一配对</h2><p>全部比较在实验前固定，未据此挑选成员或权重。'
            '百分比指标的差值单位是百分点。</p>'
            '<table><thead><tr><th>指标</th><th>参照</th><th>均值差</th><th>配对SD</th>'
            '<th>胜/平/负</th></tr></thead><tbody>' + ''.join(paired_rows)
            + '</tbody></table><p><a href="report.md">完整说明</a> · '
            '<a href="result.json">逐划分、来源与恢复信息</a></p></html>')
    (out / 'dashboard.html').write_text(page, encoding='utf-8')


def finalize(out, source_proof, batch_proofs):
    fold_proofs, fold_metrics = {}, {}
    for fold in FOLDS:
        path = out / f'fold_{fold}_result.json'
        require(path.is_file(), f'{fold}: missing same-pass fold result')
        row = read_json(path)
        require(row['fold'] == fold
                and row['source_batches'] == batch_proofs
                and row['live_production'] == source_proof
                and set(row['metrics']) == set(LABELS),
                f'{fold}: resumable result source/member identity changed')
        proof = row['proof']
        archive = proof['inference']['same_pass_probability_archive']
        require(file_sha(archive) == proof['inference']['same_pass_probability_archive_sha256'],
                f'{fold}: same-pass probability archive SHA changed')
        require(all(proof[label]['checkpoint_sha256']
                    == read_json(SOURCES[label]['ops'] / 'checkpoint_registry.json')
                        ['arms'][SOURCES[label]['arm']][fold]['checkpoint_sha256']
                    for label in SOURCES),
                f'{fold}: member checkpoint identity changed')
        fold_metrics[fold], fold_proofs[fold] = row['metrics'], proof
    result = {'status': 'completed',
              'purpose': 'Fixed same-fold equal-probability inference; exploratory ranking analysis',
              'member_order': list(SOURCES), 'primary_reference': PRIMARY_REFERENCE,
              'member_weight': 1 / len(SOURCES), 'threshold': .5,
              'source_batches': batch_proofs, 'live_production': source_proof,
              'folds': fold_proofs, 'metrics_by_fold': fold_metrics,
              'summary': summarize(fold_metrics),
              'scope': 'Five overlapping sequence-isolated outer resplits, three independent training '
                       'seeds per split, fixed probability average. Repeated model exploration on these '
                       'outer splits prevents confirmatory paper claims.'}
    dump(out / 'result.json', result)
    write_report(out, result)
    dump(out / 'audit_receipt.json', {'status': 'passed',
                                     'source_batches': batch_proofs,
                                     'result_sha256': file_sha(out / 'result.json'),
                                     'report_sha256': file_sha(out / 'report.md'),
                                     'dashboard_sha256': file_sha(out / 'dashboard.html'),
                                     'fold_result_sha256': {
                                         fold: file_sha(out / f'fold_{fold}_result.json') for fold in FOLDS},
                                     'same_pass_npz_sha256_by_fold': {
                                         fold: fold_proofs[fold]['inference'][
                                             'same_pass_probability_archive_sha256'] for fold in FOLDS}})
    print(f'Complete report: {out / "report.md"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('verify', 'run', 'finalize'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--fold', choices=FOLDS,
                        help='Run one complete fold for resumability; omit to run all and finalize')
    args = parser.parse_args()
    require(sys.platform.startswith('linux') and str(REPO).startswith('/mnt/pvc/graphtrans/'),
            'This script must run on the authorized graphtrans pod')
    out = args.out.resolve()
    require(str(out).startswith('/mnt/pvc/graphtrans/') and not in_root(out, REPO),
            'Output must be a separate pod ops directory')
    require(args.batch_size == 512,
            'This frozen study requires the original evaluation batch size 512')
    out.mkdir(parents=True, exist_ok=True)
    require(not (out / 'result.json').exists(), 'Result exists: refusing an ambiguous rerun')
    source_proof = verify_live_production()
    sources, batch_proofs = load_sources()
    device = torch.device(args.device)
    if args.mode == 'run':
        require(device.type == 'cuda' and torch.cuda.is_available(),
                'Full inference must use an available pod GPU')
        torch.cuda.set_device(device)
    if args.mode == 'finalize':
        require(args.fold is None, '--fold has no meaning for finalize')
        finalize(out, source_proof, batch_proofs)
        return
    folds = (args.fold,) if args.fold else FOLDS
    verified = {}
    for fold in folds:
        cached = out / f'fold_{fold}_result.json'
        if args.mode == 'run' and cached.exists():
            row = read_json(cached)
            require(row['fold'] == fold and row['source_batches'] == batch_proofs
                    and row['live_production'] == source_proof
                    and file_sha(row['proof']['inference']['same_pass_probability_archive'])
                        == row['proof']['inference']['same_pass_probability_archive_sha256'],
                    f'{fold}: existing resumable output is incomplete or changed')
            print(f'{fold}: existing verified fold result retained', flush=True)
            continue
        runtimes, models, proof = verify_fold_sources(fold, sources)
        verified[fold] = proof
        if args.mode == 'run':
            metrics, run_proof = infer_fold(fold, runtimes, models, proof, out,
                                            device, args.batch_size)
            proof['inference'] = run_proof
            dump(cached, {'fold': fold, 'source_batches': batch_proofs,
                          'live_production': source_proof, 'metrics': metrics, 'proof': proof})
            print(f'{fold}: q_seq ensemble={metrics["equal_probability_ensemble"]["q_seq"]:.6f}',
                  flush=True)
        del models
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if args.mode == 'verify':
        provenance = {'status': 'verified', 'source_batches': batch_proofs,
                      'live_production': source_proof,
                      'folds_checked': list(folds),
                      'checkpoints_and_splits': verified,
                      'scope': 'source/config/split/target/checkpoint strict CPU restoration only'}
        dump(out / 'preflight.json', provenance)
        print(f'Preflight verified {len(SOURCES)*len(folds)} archived checkpoints.')
        return
    if args.fold is None:
        finalize(out, source_proof, batch_proofs)


if __name__ == '__main__':
    main()
