"""Pod-only Stage14 same-round pairwise-ranking sweep (four lambda arms).

Only `smoke` and `launch` are operator actions.  `worker` runs one immutable
batch and writes a single terminal event; it never polls training progress.
Independent metric recomputation and reporting belong to the review/audit
process, after the terminal event.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'graph_transform'), str(REPO / 'graph_transform/scripts')]

import yaml
from data.row_validation import file_sha
from train_5fold import _stable_fold_seed, make_fold_config

OPS = Path('/mnt/pvc/graphtrans/dbond_s14_pairwise_ranking_20261026_ops')
CHECKPOINT_BASE = Path('/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/stage14_pairwise_batch')
PRIOR = Path('/mnt/pvc/graphtrans/four_state_parallel_global_replication_ops')
TARGETS = Path('/mnt/pvc/graphtrans/four_state_targets_v1')
ARMS = {'lambda0': 0.0, 'lambda001': 0.01, 'lambda005': 0.05, 'lambda010': 0.10}
FOLDS = ('1222', '2252', '3514', '6072', '9075')
PY = str(REPO / '.venv/bin/python')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)


def config_path(arm: str) -> Path:
    return REPO / f'graph_transform/config/pre_synthesis_stage14_pairwise_{arm}.yaml'


def revision() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()


def source_manifest() -> dict[str, str]:
    return {str(p.relative_to(REPO)): file_sha(p)
            for p in (REPO / 'graph_transform').rglob('*.py') if '__pycache__' not in p.parts}


def committed_source_provenance(bound: dict[str, str], gitrev: str) -> dict:
    """Verify the actual training code is Git-bound, save one old unrelated exception."""
    exceptions = []
    for relative, actual_sha in sorted(bound.items()):
        try:
            blob = subprocess.check_output(['git', 'show', f'{gitrev}:{relative}'], cwd=REPO)
        except subprocess.CalledProcessError as error:
            raise ValueError(f'Uncommitted source file: {relative}') from error
        blob_sha = hashlib.sha256(blob).hexdigest()
        if blob_sha != actual_sha:
            require(relative == 'graph_transform/scripts/ensemble_baseline_preds.py',
                    f'Uncommitted production source: {relative}')
            exceptions.append({'path': relative, 'live_sha256': actual_sha, 'commit_sha256': blob_sha})
    require(len(exceptions) <= 1, 'Unexpected dirty Python source')
    for arm in ARMS:
        relative = str(config_path(arm).relative_to(REPO)).replace('\\', '/')
        blob = subprocess.check_output(['git', 'show', f'{gitrev}:{relative}'], cwd=REPO)
        require(hashlib.sha256(blob).hexdigest() == file_sha(config_path(arm)),
                f'{arm}: config differs from committed source')
    return {'gitrev': gitrev, 'known_unrelated_source_exception': exceptions}


def pod_guard() -> None:
    require(sys.platform.startswith('linux')
            and REPO.resolve() == Path('/mnt/pvc/graphtrans/DBond').resolve(),
            'Stage14 numerical work and training must run on the authorized pod')


def bound_run_root(info: dict) -> Path:
    root = Path(info['root']).resolve()
    require(root.parent == CHECKPOINT_BASE.resolve() and root.name,
            'Stage14 run root escaped the dedicated graph_transform checkpoint namespace')
    return root


def canonical_config(value: dict, arm: str) -> dict:
    value = copy.deepcopy(value)
    value['loss']['ranking_loss_weight'] = 'ARM_LAMBDA'
    value['experiment'].pop('description', None)
    text = json.dumps(value, sort_keys=True).replace(f'stage14_pairwise_{arm}', 'stage14_pairwise_ARM')
    return json.loads(text)


def check_config(arm: str, value: dict) -> None:
    model, data, loss, train = value['model'], value['data'], value['loss'], value['training']
    require(model['hidden_dim'] == 256 and model['node_conv_mode'] == 'single_k5'
            and model['virtual_node_mode'] == 'single'
            and model['bond_head_type'] == 'four_state_mlp'
            and model['bond_fusion_mode'] == 'concat'
            and model['edge_state_mode'] == 'static'
            and model['theory_fusion_stage'] == 'late'
            and model['num_gcn_layers'] == 3 and model['num_gat_layers'] == 2
            and model['num_attention_heads'] == 4
            and not model.get('use_charge_partition_features', False),
            f'{arm}: architecture drift from original single-global k5 four-state model')
    require(loss['four_state_conditional_weight'] == 1
            and loss['four_state_union_brier_weight'] == 0
            and loss['ranking_loss_mode'] == 'pairwise_logistic'
            and loss['ranking_loss_weight'] == ARMS[arm]
            and loss['ranking_temperature'] == .1
            and loss['ranking_min_target_delta'] == .05,
            f'{arm}: loss/temperature/delta mismatch')
    require(value['experiment']['seed'] == 42 and train['validation_split_base_seed'] == 42
            and train['checkpoint_selection'] == 'max_f1'
            and train['batch_size'] == 128
            and not train.get('pretrained_backbone_path'),
            f'{arm}: training/split/checkpoint budget mismatch')
    require(data['use_soft_labels'] is True and data['weighting_scheme'] == 'group_uniform'
            and data['augmentation'] is False and data['drop_last'] is True
            and data['validation_source_csv_template'] ==
                'dataset/5fold_soft/{fold}.train.fbr.shuffle.multi.csv'
            and data['four_state_targets_manifest_template'] ==
                '/mnt/pvc/graphtrans/four_state_targets_v1/fold_{fold}/manifest.json'
            and value['evaluation']['threshold'] == .5
            and value['evaluation']['threshold_strategy'] == 'fixed',
            f'{arm}: data/source/evaluation protocol drift')


def prior_audit() -> dict:
    """Bind to reviewed Stage11 data split, source and target lineage."""
    paths = {name: PRIOR / name for name in ('launch.json', 'terminal.json', 'review_receipt.json',
                                             'smoke.json', 'result.json')}
    require(all(path.is_file() for path in paths.values()), 'Reviewed Stage11 provenance missing')
    items = {name: read_json(path) for name, path in paths.items()}
    launch, terminal, review, smoke = (items[name] for name in
                                       ('launch.json', 'terminal.json', 'review_receipt.json', 'smoke.json'))
    require(launch['root'] == terminal['root'] == review['root']
            and launch['gitrev'] == terminal['gitrev'] == review['gitrev']
            and terminal['status'] == 'completed' and review['status'] == 'passed'
            and smoke['status'] == 'passed', 'Stage11 provenance/review mismatch')
    require(file_sha(paths['result.json']) == review['result_sha256'],
            'Stage11 reviewed result changed')
    return {'root': launch['root'], 'gitrev': launch['gitrev'],
            'review_sha256': file_sha(paths['review_receipt.json']),
            'smoke_sha256': file_sha(paths['smoke.json']),
            'folds': smoke['folds'], 'target_manifests': smoke['target_manifests'],
            'test_sha256': items['result.json']['test_source_sha256_by_fold']}


def verify_training_inputs(proof: dict) -> None:
    """Reject changed raw train/test, full-spectrum source, or target manifests."""
    for fold in FOLDS:
        row = proof['folds'][fold]
        folded = REPO / 'dataset/5fold_folded'
        full_source = REPO / f'dataset/5fold_soft/{fold}.train.fbr.shuffle.multi.csv'
        target = TARGETS / f'fold_{fold}/manifest.json'
        require(file_sha(folded / f'{fold}.train.fbr.shuffle.multi.csv') ==
                row['folded_train_sha256']
                and file_sha(folded / f'{fold}.test.fbr.multi.csv') ==
                row['test_sha256']
                and file_sha(full_source) == row['source_sha256']
                and file_sha(target) == row['target_manifest_sha256'],
                f'{fold}: train/test/full-spectrum source/target changed after smoke')


def smoke(gpu: int) -> None:
    pod_guard()
    require(not (OPS / 'launch.json').exists() and not (OPS / 'smoke.json').exists(),
            'Stage14 smoke/launch already exists; refuse duplicate')
    import torch
    from data.graph_dataset import GraphDataLoader
    from models.graph_transformer import GraphTransformer
    from models.utils import build_model_config
    from training.loss_functions import BinaryBondLoss
    from training.four_state_loss import four_state_cross_entropy
    from training.trainer import Trainer
    from training.pairwise_sequence_ranking import pairwise_sequence_logistic_loss
    from train_graph_model import create_datasets, create_optimizer
    from test_four_state_dynamic_edges import seed as seed_everything

    require(torch.cuda.is_available(), 'Pod CUDA required for smoke')
    torch.cuda.set_device(gpu)
    old = prior_audit()
    configs = {arm: yaml.safe_load(config_path(arm).read_text(encoding='utf-8')) for arm in ARMS}
    canonical = None
    for arm, value in configs.items():
        check_config(arm, value)
        normalized = canonical_config(value, arm)
        if canonical is None:
            canonical = normalized
        require(normalized == canonical, f'{arm}: config differs beyond lambda and artifact names')

    # This folded training source has one q row per (seq,charge,NCE), computed
    # exclusively from the outer training spectra.  Internal split is by seq.
    evidence = {}
    with tempfile.TemporaryDirectory(prefix='stage14_', dir=OPS) as temporary:
        for fold in FOLDS:
            runtime, _, _ = make_fold_config(configs['lambda0'], fold,
                str(REPO / 'dataset/5fold_folded'), str(Path(temporary) / fold),
                _stable_fold_seed(42, fold))
            runtime['data']['cache_graphs'] = False
            runtime['data']['cache_dir'] = str(Path(temporary) / fold / 'cache')
            train_ds, val_ds, test_ds = create_datasets(runtime)
            seq_sets = [set(ds.data['seq'].astype(str)) for ds in (train_ds, val_ds, test_ds)]
            require(not (seq_sets[0] & seq_sets[1] or seq_sets[0] & seq_sets[2]
                         or seq_sets[1] & seq_sets[2]), f'{fold}: sequence leakage')
            require(train_ds.data[['seq', 'charge', 'nce']].duplicated().sum() == 0,
                    f'{fold}: folded train contains duplicate condition rows')
            require(train_ds.data['soft_multi'].notna().all(), f'{fold}: missing train soft q')
            source = val_ds.row_validation_provenance
            row = {'train_sha256': file_sha(train_ds.csv_path),
                   'val_sha256': file_sha(val_ds.csv_path),
                   'source_sha256': source['source_sha256'],
                   'folded_train_sha256': file_sha(runtime['data']['train_csv_path']),
                   'test_sha256': file_sha(runtime['data']['test_csv_path']),
                   'target_manifest_sha256': file_sha(Path(runtime['data']['four_state_targets_manifest'])),
                   'training_seed': runtime['experiment']['seed'],
                   'split_seed': runtime['training']['validation_split_seed']}
            historical = old['folds'][fold]
            require(all(row[key] == historical[key] for key in ('train_sha256', 'val_sha256', 'source_sha256'))
                    and row['test_sha256'] == old['test_sha256'][fold]
                    and row['target_manifest_sha256'] == old['target_manifests'][fold]
                    and row['training_seed'] == _stable_fold_seed(42, fold)
                    and row['split_seed'] == _stable_fold_seed(42, fold)
                    and file_sha(REPO / f'dataset/5fold_soft/{fold}.train.fbr.shuffle.multi.csv')
                    == row['source_sha256']
                    and source['outer_test_used_for_internal_validation'] is False,
                    f'{fold}: Stage11 train/val/test/target binding changed')
            evidence[fold] = row
            if fold != FOLDS[0]:
                continue

            # Real folded training rows; only pod-side smoke does numerical work.
            indices = list(range(min(128, len(train_ds))))
            batch_cpu = GraphDataLoader._default_collate_fn(None, [train_ds[i] for i in indices])
            batch = {k: v.cuda(gpu) if torch.is_tensor(v) else v for k, v in batch_cpu.items()}
            require('soft_labels' in batch and 'four_state_targets' in batch,
                    'Train batch lacks joint four-state or soft-q source')
            losses, checks = {}, {}
            model_states = {}
            for arm, value in configs.items():
                arm_runtime, _, _ = make_fold_config(value, fold,
                    str(REPO / 'dataset/5fold_folded'), str(Path(temporary) / fold / arm),
                    _stable_fold_seed(42, fold))
                seed_everything(_stable_fold_seed(42, fold))
                model = GraphTransformer(build_model_config(arm_runtime)).cuda(gpu)
                model.train()
                model_states[arm] = {name: weight.detach().cpu().clone()
                                     for name, weight in model.state_dict().items()}
                trainer = Trainer(model, create_optimizer(model, arm_runtime),
                                  BinaryBondLoss(arm_runtime['loss']), device=torch.device(f'cuda:{gpu}'),
                                  config=arm_runtime)
                loss, stats = trainer._forward_pass(batch)
                require(torch.isfinite(loss) and stats['valid_bond_count'] > 0,
                        f'{arm}: non-finite/no-bond real forward')
                losses[arm] = float(loss.detach().item())
                checks[arm] = {k: stats[k] for k in ('ranking_pairs', 'ranking_zero_pair_batch',
                                                   'ranking_loss', 'ranking_output_gradient_abs',
                                                   'ranking_grad_probe_seconds',
                                                   'ranking_compute_seconds')}
                if arm == 'lambda0':
                    base_loss = loss.detach().clone()
                    manual_ce = four_state_cross_entropy(
                        model.last_four_state_logits, batch['four_state_targets'],
                        batch['label_mask'], trainer._masked_row_weights(batch),
                        conditional_weight=1., union_brier_weight=0.)
                    require(torch.equal(loss, manual_ce),
                            'lambda0 does not equal the unmodified four-state CE exactly')
                    production_grad = torch.autograd.grad(
                        loss, model.last_four_state_logits, retain_graph=True)[0]
                    manual_grad = torch.autograd.grad(
                        manual_ce, model.last_four_state_logits, retain_graph=True)[0]
                    require(torch.allclose(production_grad, manual_grad, rtol=0, atol=1e-8),
                            'lambda0 changed four-state loss gradient')
                else:
                    require(stats['ranking_pairs'] > 0 and stats['ranking_output_gradient_abs'] > 0,
                            f'{arm}: ranking did not provide valid pair/gradient')
                    require(torch.allclose(loss - base_loss,
                                           value['loss']['ranking_loss_weight'] *
                                           torch.as_tensor(stats['ranking_loss'], device=loss.device),
                                           rtol=1e-4, atol=1e-5),
                            f'{arm}: loss differs beyond lambda times pair term')
                model.zero_grad(set_to_none=True)
                loss.backward()
                require(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in model.parameters()), f'{arm}: nonfinite task gradient')
                del model, trainer, loss
            require(all(model_states[arm].keys() == model_states['lambda0'].keys()
                        and all(torch.equal(model_states[arm][key], model_states['lambda0'][key])
                                for key in model_states['lambda0']) for arm in ARMS),
                    'Four arms do not share identical initial parameters')
            require(checks['lambda0']['ranking_pairs'] == checks['lambda001']['ranking_pairs']
                    == checks['lambda005']['ranking_pairs'] == checks['lambda010']['ranking_pairs'],
                    'Four arms see different sequence-pair counts')
            # Permuting labels across two sequences must change direction; no
            # valid pair and same-sequence-only examples must return exact zero.
            synthetic = {'soft_labels': torch.tensor([[.8], [.1]], device=f'cuda:{gpu}'),
                         'label_mask': torch.ones(2, 1, device=f'cuda:{gpu}'),
                         'sequences': ['AAA', 'BBB'], 'charges': torch.ones(2, device=f'cuda:{gpu}'),
                         'nces': torch.zeros(2, device=f'cuda:{gpu}')}
            logits = torch.tensor([[2.], [-2.]], device=f'cuda:{gpu}', requires_grad=True)
            correct, meta = pairwise_sequence_logistic_loss(logits, synthetic,
                temperature=.1, min_target_delta=.05)
            wrong, _ = pairwise_sequence_logistic_loss(-logits, synthetic,
                temperature=.1, min_target_delta=.05)
            require(meta['eligible_pairs'] == 1 and correct < wrong,
                    'Pairwise logistic sign or selection reversed')
            synthetic['sequences'] = ['AAA', 'AAA']
            same, meta = pairwise_sequence_logistic_loss(logits, synthetic,
                temperature=.1, min_target_delta=.05)
            require(meta['eligible_pairs'] == 0 and same == 0,
                    'Same-sequence rows formed an illegal pair')
            # A repeated identical condition must not gain double weight.
            duplicate = {'soft_labels': torch.tensor([[.8], [.8], [.1]], device=f'cuda:{gpu}'),
                         'label_mask': torch.ones(3, 1, device=f'cuda:{gpu}'),
                         'sequences': ['AAA', 'AAA', 'BBB'],
                         'charges': torch.ones(3, device=f'cuda:{gpu}'),
                         'nces': torch.zeros(3, device=f'cuda:{gpu}')}
            duplicate_logits = torch.tensor([[2.], [2.], [-2.]], device=f'cuda:{gpu}')
            duplicated, dupe_meta = pairwise_sequence_logistic_loss(
                duplicate_logits, duplicate, temperature=.1, min_target_delta=.05)
            require(dupe_meta['eligible_pairs'] == 1
                    and torch.allclose(duplicated, correct, rtol=0, atol=1e-7),
                    'Duplicate condition changed the sequence-level pair')
            # Invalid padding may contain NaN/Inf and must be entirely inert,
            # including in the backward path; masking after multiplication is unsafe.
            padded = {'soft_labels': torch.tensor([[.8, float('nan')],
                                                   [.1, float('-inf')]], device=f'cuda:{gpu}'),
                      'label_mask': torch.tensor([[1., 0.], [1., 0.]], device=f'cuda:{gpu}'),
                      'sequences': ['AAA', 'BBB'], 'charges': torch.ones(2, device=f'cuda:{gpu}'),
                      'nces': torch.zeros(2, device=f'cuda:{gpu}')}
            padded_logits = torch.tensor([[2., float('nan')], [-2., float('inf')]],
                                         device=f'cuda:{gpu}', requires_grad=True)
            cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(gpu).clone()
            padded_loss, padded_meta = pairwise_sequence_logistic_loss(
                padded_logits, padded, temperature=.1, min_target_delta=.05)
            require(padded_meta['eligible_pairs'] == 1 and torch.isfinite(padded_loss)
                    and torch.allclose(padded_loss, correct, rtol=0, atol=1e-6),
                    'Invalid padding polluted pairwise forward')
            padded_gradient = torch.autograd.grad(padded_loss, padded_logits)[0]
            require(torch.isfinite(padded_gradient).all()
                    and torch.equal(padded_gradient[:, 1], torch.zeros_like(padded_gradient[:, 1]))
                    and torch.equal(cpu_rng, torch.get_rng_state())
                    and torch.equal(cuda_rng, torch.cuda.get_rng_state(gpu)),
                    'Invalid padding polluted gradients or ranking computation consumed RNG')

    bound_source, gitrev = source_manifest(), revision()
    verify_training_inputs({'folds': evidence})
    proof = {'status': 'passed', 'created_at': utc_now(), 'gitrev': gitrev,
             'source_sha256': bound_source,
             'source_provenance': committed_source_provenance(bound_source, gitrev),
             'configs': {arm: file_sha(config_path(arm)) for arm in ARMS},
             'prior': old, 'folds': evidence, 'smoke_losses': losses,
             'smoke_ranking_stats': checks,
             'definition': 'train-only folded soft-q; condition-first/sequence-second aggregation; '
                           'distinct-sequence unordered pairs with |target gap|>0.05; '
                           'softplus(-signed score difference/0.1); original row shuffle in all arms',
             'checks': {'lambda0_same_initialization': True,
                        'four_arm_same_sampler_config': True,
                        'historical_split_and_target_bindings': True,
                        'loss_additivity': True, 'ranking_gradient': True,
                        'same_sequence_exclusion': True, 'duplicate_condition_invariance': True,
                        'nan_inf_padding_inert': True, 'rng_inert': True,
                        'lambda0_exact_four_state_ce_and_gradient': True, 'sign': True}}
    write_json(OPS / 'smoke.json', proof)
    print(json.dumps({'status': proof['status'], 'gitrev': proof['gitrev'], 'checks': proof['checks']}))


def launch() -> None:
    pod_guard()
    claim = OPS / 'launch.claim'
    worker_started = False
    try:
        with claim.open('x', encoding='utf-8') as stream:
            stream.write(utc_now() + '\n')
    except FileExistsError as error:
        raise ValueError('Stage14 already claimed; inspect launch/terminal, never relaunch blindly') from error
    try:
        require(not (OPS / 'launch.json').exists(), 'Stage14 launch already exists')
        proof = read_json(OPS / 'smoke.json')
        require(proof['status'] == 'passed' and proof['gitrev'] == revision()
                and proof['source_sha256'] == source_manifest()
                and proof['source_provenance'] == committed_source_provenance(proof['source_sha256'], revision())
                and all(proof['configs'][arm] == file_sha(config_path(arm)) for arm in ARMS),
                'Stale or incomplete source/config-bound pod smoke')
        require(proof['prior'] == prior_audit(), 'Reviewed predecessor changed')
        verify_training_inputs(proof)
        from four_state_ops import idle
        devices, gpu_rows, processes = idle()
        root = CHECKPOINT_BASE / dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')
        info = {'schema_version': 1, 'root': str(root), 'gitrev': revision(),
                'created_at': utc_now(), 'arms': ARMS, 'folds': FOLDS,
                'training_base_seed': 42, 'validation_split_base_seed': 42,
                'fold_seeds': {fold: _stable_fold_seed(42, fold) for fold in FOLDS},
                'configs': proof['configs'], 'smoke_sha256': file_sha(OPS / 'smoke.json'),
                'source_sha256': proof['source_sha256'], 'prior': proof['prior'],
                'source_provenance': proof['source_provenance'],
                'gpus': devices, 'gpu_preflight': gpu_rows,
                'compute_processes': processes,
                'selection_rule': 'internal valid-bond F1 only; no external test selection'}
        write_json(OPS / 'launch.json', info)
        env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        with (OPS / 'worker.log').open('x', encoding='utf-8') as log:
            worker = subprocess.Popen([PY, __file__, 'worker'], cwd=REPO, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        worker_started = True
        info['worker_pid'] = worker.pid
        write_json(OPS / 'launch.json', info)
        print(json.dumps({'root': info['root'], 'gitrev': info['gitrev'], 'worker_pid': worker.pid}))
    except BaseException:
        if not (OPS / 'launch.json').exists():
            claim.unlink(missing_ok=True)
        elif not worker_started and not (OPS / 'terminal.json').exists():
            bound = read_json(OPS / 'launch.json')
            write_json(OPS / 'terminal.json', {'status': 'failed', 'root': bound['root'],
                'gitrev': bound['gitrev'], 'finished_at': utc_now(),
                'error': traceback.format_exc(), 'retry_policy': 'No automatic relaunch'})
        raise


def make_audit_manifest(info: dict) -> dict:
    root = bound_run_root(info)
    status, batch = (read_json(root / name) for name in ('status.json', 'manifest.json'))
    require(status['status'] == 'completed' and status['root'] == batch['root'] == info['root']
            and status['gitrev'] == batch['gitrev'] == info['gitrev'],
            'Batch status/source/root mismatch')
    require(batch['folds'] == list(FOLDS) and len(batch['arms']) == len(ARMS),
            'Batch arm/fold count mismatch')
    entries = []
    for arm_name, bound in zip(ARMS, batch['arms']):
        require(bound['name'].endswith(f'pre_synthesis_stage14_pairwise_{arm_name}')
                and file_sha(bound['config_snapshot']) == info['configs'][arm_name],
                f'{arm_name}: configuration snapshot differs from frozen launch')
        with Path(bound['metrics_csv']).open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == 5 and {row['fold_id'] for row in rows} == set(FOLDS),
                f'{arm_name}: missing or duplicate fold metric rows')
        for row in rows:
            fold = row['fold_id']
            fold_root = Path(bound['cv_root']) / f'fold_{fold}'
            runtime_path = fold_root / 'config.yaml'
            runtime = yaml.safe_load(runtime_path.read_text(encoding='utf-8'))
            require(runtime['experiment']['seed'] == info['fold_seeds'][fold]
                    and runtime['training']['validation_split_seed'] == info['fold_seeds'][fold]
                    and runtime['loss']['ranking_loss_weight'] == ARMS[arm_name]
                    and runtime['loss']['ranking_temperature'] == .1
                    and runtime['loss']['ranking_min_target_delta'] == .05
                    and runtime['loss']['ranking_loss_mode'] == 'pairwise_logistic',
                    f'{arm_name}/{fold}: actual seed or loss config mismatch')
            provenance_files = list(fold_root.glob('checkpoints/**/row_validation_provenance.json'))
            require(len(provenance_files) == 1, f'{arm_name}/{fold}: ambiguous train/val provenance')
            provenance = read_json(provenance_files[0])
            train_csv = Path(provenance['split_train_csv'])
            val_csv = Path(provenance['split_val_csv'])
            pred_csv = Path(row['pred_csv_path'])
            metric_csv = Path(row['metric_csv_path'])
            checkpoint = Path(row['checkpoint_dir']) / 'best_model.pt'
            test_csv = Path(runtime['data']['test_csv_path'])
            try:
                metric_relative = metric_csv.resolve().relative_to((fold_root / 'metrics').resolve())
                checkpoint.resolve().relative_to((fold_root / 'checkpoints').resolve())
            except ValueError as error:
                raise ValueError(f'{arm_name}/{fold}: metric/checkpoint escaped this fold') from error
            expected_prediction = fold_root / 'preds' / metric_relative.parent / 'latest_test.pred.csv'
            require(metric_csv.name == 'latest_test_metric.csv'
                    and pred_csv.resolve() == expected_prediction.resolve(),
                    f'{arm_name}/{fold}: prediction CSV is not paired to metric CSV')
            expected = info['prior']['folds'][fold]
            require(file_sha(train_csv) == expected['train_sha256']
                    and file_sha(val_csv) == expected['val_sha256']
                    and provenance['source_sha256'] == expected['source_sha256']
                    and file_sha(Path(provenance['source_path'])) == expected['source_sha256']
                    and file_sha(test_csv) == info['prior']['test_sha256'][fold]
                    and file_sha(Path(runtime['data']['four_state_targets_manifest']))
                    == info['prior']['target_manifests'][fold]
                    and pred_csv.is_file() and metric_csv.is_file() and checkpoint.is_file(),
                    f'{arm_name}/{fold}: split, test or outputs differ from reviewed source')
            with pred_csv.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                require(reader.fieldnames is not None and
                        {'checkpoint_path', 'evaluation_id', 'threshold', 'pred_prob'} <= set(reader.fieldnames),
                        f'{arm_name}/{fold}: prediction source columns missing')
                evaluation_ids, prediction_rows = set(), 0
                for prediction in reader:
                    require(Path(prediction['checkpoint_path']).resolve() == checkpoint.resolve()
                            and float(prediction['threshold']) == .5,
                            f'{arm_name}/{fold}: same-pass checkpoint or threshold mismatch')
                    evaluation_ids.add(prediction['evaluation_id'])
                    prediction_rows += 1
            require(prediction_rows > 0 and len(evaluation_ids) == 1
                    and bool(next(iter(evaluation_ids))),
                    f'{arm_name}/{fold}: empty/mixed prediction evaluation')
            files = {'pred_csv': pred_csv, 'test_csv': test_csv,
                     'metric_csv': metric_csv, 'checkpoint': checkpoint,
                     'config': runtime_path, 'train_csv': train_csv, 'val_csv': val_csv}
            entries.append({'arm': arm_name, 'fold': fold, 'seed': runtime['experiment']['seed'],
                            'split_seed': runtime['training']['validation_split_seed'],
                            'lambda': ARMS[arm_name], 'tau': .1, 'delta': .05,
                            'max_seq_len': runtime['data']['max_seq_len'],
                            'source_commit': info['gitrev'],
                            'evaluation_id': next(iter(evaluation_ids)),
                            'prediction_rows': prediction_rows,
                            **{name: str(path) for name, path in files.items()},
                            **{name + '_sha256': file_sha(path) for name, path in files.items()}})
    return {'stage': 14, 'root': info['root'], 'source_commit': info['gitrev'],
            'arms': entries, 'same_round_control': 'lambda0',
            'pairwise_mode': 'train-only logistic, tau0.1, delta0.05; no sequence-aware sampler',
            'scope': 'Manifest binds training fold, internal validation and same-pass external prediction; '
                     'independent metric audit must still verify CSV row provenance and recompute metrics.'}


def worker() -> None:
    pod_guard()
    info = read_json(OPS / 'launch.json')
    try:
        bound_run_root(info)
        require(revision() == info['gitrev'] and source_manifest() == info['source_sha256']
                and all(file_sha(config_path(arm)) == info['configs'][arm] for arm in ARMS),
                'Source/config changed after launch')
        verify_training_inputs(read_json(OPS / 'smoke.json'))
        command = [PY, '-u', str(REPO / 'graph_transform/scripts/run_architecture_batch.py'),
                   '--configs', *[str(config_path(arm)) for arm in ARMS],
                   '--fold_data_dir', 'dataset/5fold_folded',
                   '--gpus', ','.join(map(str, info['gpus'])), '--run_root', info['root']]
        with (OPS / 'training.log').open('x', encoding='utf-8') as log:
            subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        verify_training_inputs(read_json(OPS / 'smoke.json'))
        manifest = make_audit_manifest(info)
        write_json(OPS / 'audit_manifest.json', manifest)
        write_json(OPS / 'terminal.json', {'status': 'completed', 'root': info['root'],
            'gitrev': info['gitrev'], 'finished_at': utc_now(),
            'audit_manifest': str(OPS / 'audit_manifest.json'),
            'audit_manifest_sha256': file_sha(OPS / 'audit_manifest.json'),
            'review_state': 'pending_independent_pod_audit'})
    except BaseException:
        write_json(OPS / 'terminal.json', {'status': 'failed', 'root': info['root'],
            'gitrev': info['gitrev'], 'finished_at': utc_now(),
            'error': traceback.format_exc(), 'retry_policy': 'No automatic rerun'})
        raise


if __name__ == '__main__':
    pod_guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['smoke', 'launch', 'worker'])
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    OPS.mkdir(parents=True, exist_ok=True)
    if args.action == 'smoke':
        smoke(args.gpu)
    elif args.action == 'launch':
        launch()
    else:
        worker()
