"""Pod-only Stage14 same-round pairwise-ranking sweep (four lambda arms).

`smoke` and `launch` are the training operator actions. `worker` runs one
immutable batch. `finalize` only registers an already completed batch after
strict source/output checks; it never invokes training or polls progress.
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
import re
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
STAGE13_OPS = Path('/mnt/pvc/graphtrans/dbond_s13_multiscale_conv_20261026_ops')
STAGE13_AUDIT = STAGE13_OPS / 'independent_audit/audit_receipt.json'
PRIOR = Path('/mnt/pvc/graphtrans/four_state_parallel_global_replication_ops')
TARGETS = Path('/mnt/pvc/graphtrans/four_state_targets_v1')
ARMS = {'lambda0': 0.0, 'lambda001': 0.01, 'lambda005': 0.05, 'lambda010': 0.10}
FOLDS = ('1222', '2252', '3514', '6072', '9075')
# The shared venv supplies only the interpreter and installed packages.  Every
# script, config, and Python import used by this batch comes from REPO.
PY = '/mnt/pvc/graphtrans/DBond/.venv/bin/python'


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


def write_new_or_matching_json(path: Path, value: dict) -> None:
    """Never silently replace an independently consumed audit artifact."""
    if path.exists():
        require(read_json(path) == value, f'Existing {path.name} differs from verified batch')
    else:
        write_json(path, value)


def config_path(arm: str) -> Path:
    return REPO / f'graph_transform/config/pre_synthesis_stage14_pairwise_{arm}.yaml'


def git_revision() -> str | None:
    """Return HEAD only when this snapshot itself is a Git checkout."""
    try:
        top = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=REPO,
                             capture_output=True, text=True)
        if top.returncode != 0 or Path(top.stdout.strip()).resolve() != REPO.resolve():
            return None
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                                capture_output=True, text=True)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def revision(declared: str | None = None) -> str:
    """Bind a Git-free source snapshot to the operator-declared full commit."""
    actual = git_revision()
    require(declared is not None and re.fullmatch(r'[0-9a-f]{40}', declared),
            'Supply the exact 40-character Stage14 source commit')
    require(actual is None or actual == declared,
            'Declared Stage14 source commit differs from snapshot Git HEAD')
    pod_guard(declared)
    return declared


def source_manifest() -> dict[str, str]:
    return {str(p.relative_to(REPO)): file_sha(p)
            for p in (REPO / 'graph_transform').rglob('*.py') if '__pycache__' not in p.parts}


def committed_source_provenance(bound: dict[str, str], gitrev: str) -> dict:
    """Bind the frozen source fingerprint to a checkout or declared snapshot."""
    current = git_revision()
    if current is None:
        require(REPO.name == f'dbond_s14_source_{gitrev[:7]}',
                'Git-free source snapshot directory does not match source commit')
        digest = hashlib.sha256(json.dumps(bound, sort_keys=True,
                                  separators=(',', ':')).encode('utf-8')).hexdigest()
        return {'gitrev': gitrev, 'source_root': str(REPO),
                'source_tree_sha256': digest, 'method': 'declared_git_free_snapshot'}
    require(current == gitrev, 'Source snapshot HEAD changed')
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
    return {'gitrev': gitrev, 'source_root': str(REPO),
            'method': 'git_checkout', 'known_unrelated_source_exception': exceptions}


def pod_guard(source_commit: str | None = None) -> None:
    require(sys.platform.startswith('linux')
            and REPO.resolve().parent == Path('/mnt/pvc/graphtrans').resolve()
            and re.fullmatch(r'dbond_s14_source_[0-9a-f]{7}', REPO.name),
            'Stage14 must run from its isolated source snapshot on the authorized pod')
    if source_commit is not None:
        require(REPO.name == f'dbond_s14_source_{source_commit[:7]}',
                'Stage14 source snapshot name and declared commit mismatch')


def bound_run_root(info: dict) -> Path:
    root = Path(info['root']).resolve()
    require(root.parent == CHECKPOINT_BASE.resolve() and root.name,
            'Stage14 run root escaped the dedicated graph_transform checkpoint namespace')
    return root


def stage13_audit(receipt_path: Path = STAGE13_AUDIT) -> dict:
    """Stage14 may start only after the immediately preceding round is audited."""
    require(receipt_path.resolve() == STAGE13_AUDIT.resolve(),
            'Stage13 audit receipt must be in its dedicated independent ops namespace')
    files = {'receipt': STAGE13_AUDIT,
             'launch': STAGE13_OPS / 'launch.json',
             'terminal': STAGE13_OPS / 'terminal.json',
             'manifest': STAGE13_OPS / 'audit_manifest.json'}
    require(all(path.is_file() for path in files.values()),
            'Stage13 launch, terminal, audit manifest or independent audit receipt missing')
    objects = {name: read_json(path) for name, path in files.items()}
    receipt, launch, terminal, manifest = (objects[name] for name in
                                           ('receipt', 'launch', 'terminal', 'manifest'))
    root = launch.get('root')
    source = launch.get('source_revision')
    require(receipt.get('status') == 'audited' and receipt.get('stage') == 'S13'
            and Path(receipt.get('ops', '')).resolve() == STAGE13_OPS.resolve()
            and launch.get('stage') == terminal.get('stage') == manifest.get('stage') == 'S13'
            and launch.get('ops_root') == str(STAGE13_OPS)
            and root == launch.get('run_root') == terminal.get('root')
            == terminal.get('run_root') == manifest.get('root') == receipt.get('root')
            and source == terminal.get('source_revision')
            == manifest.get('source_commit') == receipt.get('source_commit')
            and terminal.get('status') == 'completed'
            and receipt.get('manifest_sha256') == file_sha(files['manifest'])
            and len(manifest.get('arms', [])) == 15,
            'Stage13 independent audit, run root, source or manifest binding failed')
    require(Path(root).resolve().parent == Path('/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform').resolve(),
            'Stage13 audited run root escaped graph_transform checkpoints')
    return {'path': str(STAGE13_AUDIT), 'receipt_sha256': file_sha(files['receipt']),
            'launch_sha256': file_sha(files['launch']),
            'terminal_sha256': file_sha(files['terminal']),
            'manifest_sha256': file_sha(files['manifest']),
            'root': root, 'source_commit': source}


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


def smoke(gpu: int, source_commit: str | None) -> None:
    pod_guard(source_commit)
    gitrev = revision(source_commit)
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
    from train_graph_model import (create_datasets, create_optimizer,
                                   ranking_loss_checkpoint_protocol)
    from test_four_state_dynamic_edges import seed as seed_everything

    require(torch.cuda.is_available(), 'Pod CUDA required for smoke')
    torch.cuda.set_device(gpu)
    predecessor = stage13_audit()
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
                ranking_protocol = ranking_loss_checkpoint_protocol(trainer, config_path(arm))
                require(ranking_protocol is not None
                        and ranking_protocol['mode'] == 'pairwise_logistic'
                        and ranking_protocol['weight'] == ARMS[arm]
                        and ranking_protocol['temperature'] == .1
                        and ranking_protocol['min_target_delta'] == .05
                        and ranking_protocol['target_source'] ==
                        'folded_outer_training_soft_multi_only'
                        and ranking_protocol['source_config_sha256'] == file_sha(config_path(arm)),
                        f'{arm}: actual trainer ranking settings cannot be checkpoint-bound')
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

    bound_source = source_manifest()
    verify_training_inputs({'folds': evidence})
    proof = {'status': 'passed', 'created_at': utc_now(), 'gitrev': gitrev,
             'source_root': str(REPO),
             'source_sha256': bound_source,
             'source_provenance': committed_source_provenance(bound_source, gitrev),
             'configs': {arm: file_sha(config_path(arm)) for arm in ARMS},
             'stage13_audit': predecessor, 'prior': old,
             'folds': evidence, 'smoke_losses': losses,
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


def launch(receipt_path: Path | None, source_commit: str | None) -> None:
    pod_guard(source_commit)
    gitrev = revision(source_commit)
    require(receipt_path is not None,
            'Launch requires --stage13-audit pointing to the audited Stage13 receipt')
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
        require(proof['status'] == 'passed' and proof['gitrev'] == gitrev
                and proof['source_root'] == str(REPO)
                and proof['source_sha256'] == source_manifest()
                and proof['source_provenance'] == committed_source_provenance(proof['source_sha256'], revision())
                and all(proof['configs'][arm] == file_sha(config_path(arm)) for arm in ARMS),
                'Stale or incomplete source/config-bound pod smoke')
        require(proof['stage13_audit'] == stage13_audit(receipt_path),
                'Stage13 audit receipt changed or was not independently passed')
        require(proof['prior'] == prior_audit(), 'Reviewed historical baseline changed')
        verify_training_inputs(proof)
        from four_state_ops import idle
        devices, gpu_rows, processes = idle()
        root = CHECKPOINT_BASE / dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')
        info = {'schema_version': 1, 'root': str(root), 'gitrev': gitrev,
                'source_root': str(REPO),
                'created_at': utc_now(), 'arms': ARMS, 'folds': FOLDS,
                'training_base_seed': 42, 'validation_split_base_seed': 42,
                'fold_seeds': {fold: _stable_fold_seed(42, fold) for fold in FOLDS},
                'configs': proof['configs'], 'smoke_sha256': file_sha(OPS / 'smoke.json'),
                'source_sha256': proof['source_sha256'], 'prior': proof['prior'],
                'stage13_audit': proof['stage13_audit'],
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
                'phase': 'preworker_launch',
                'error': traceback.format_exc(), 'retry_policy': 'No automatic relaunch'})
        raise


def make_audit_manifest(info: dict) -> dict:
    import torch
    from models.graph_transformer import GraphTransformer
    from models.utils import build_model_config

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
            metric_csv = Path(row['metric_csv_path'])
            checkpoint = Path(row['checkpoint_dir']) / 'best_model.pt'
            test_csv = Path(runtime['data']['test_csv_path'])
            training_log = Path(runtime['logging']['log_dir']) / runtime['logging'].get(
                'log_file', 'training.log')
            try:
                metric_relative = metric_csv.resolve().relative_to((fold_root / 'metrics').resolve())
                checkpoint.resolve().relative_to((fold_root / 'checkpoints').resolve())
                training_log.resolve().relative_to((fold_root / 'logs').resolve())
            except ValueError as error:
                raise ValueError(f'{arm_name}/{fold}: metric/checkpoint/log escaped this fold') from error
            expected_prediction = fold_root / 'preds' / metric_relative.parent / 'latest_test.pred.csv'
            pred_csv = expected_prediction
            require(metric_csv.name == 'latest_test_metric.csv'
                    and len(list(fold_root.glob('preds/**/latest_test.pred.csv'))) == 1,
                    f'{arm_name}/{fold}: prediction CSV is missing or ambiguous for metric CSV')
            expected = info['prior']['folds'][fold]
            require(file_sha(train_csv) == expected['train_sha256']
                    and file_sha(val_csv) == expected['val_sha256']
                    and provenance['source_sha256'] == expected['source_sha256']
                    and file_sha(Path(provenance['source_path'])) == expected['source_sha256']
                    and file_sha(test_csv) == info['prior']['test_sha256'][fold]
                    and file_sha(Path(runtime['data']['four_state_targets_manifest']))
                    == info['prior']['target_manifests'][fold]
                    and pred_csv.is_file() and metric_csv.is_file() and checkpoint.is_file()
                    and training_log.is_file(),
                    f'{arm_name}/{fold}: split, test or outputs differ from reviewed source')
            # These figures are copied from this fold's completed, SHA-bound log.
            # The final test timer includes CSV export; it is not pure inference.
            log_text = training_log.read_text(encoding='utf-8', errors='replace')
            total_times = re.findall(r'Training completed in ([0-9]+(?:\.[0-9]+)?)s', log_text)
            test_times = re.findall(r'Final test evaluation took ([0-9]+(?:\.[0-9]+)?)s', log_text)
            epoch_peaks = re.findall(r'Epoch [0-9]+ GPU Memory - [^\n]*?peak_allocated: '
                                     r'([0-9]+(?:\.[0-9]+)?)MB', log_text)
            require(len(total_times) == len(test_times) == 1 and bool(epoch_peaks),
                    f'{arm_name}/{fold}: completed timing or epoch GPU-memory source missing')
            require(all(math.isfinite(float(value)) and float(value) >= 0
                        for value in total_times + test_times + epoch_peaks),
                    f'{arm_name}/{fold}: non-finite completed resource observation')
            resource_observation = {
                'source': 'per_fold_completed_training_log',
                'training_plus_final_evaluation_seconds': float(total_times[0]),
                'test_evaluation_and_export_seconds': float(test_times[0]),
                'test_inference_only_seconds': None,
                'epoch_peak_gpu_allocated_mb_max': max(map(float, epoch_peaks)),
                'epoch_peak_count': len(epoch_peaks),
                'limitations': 'Training timer includes validation and final test; test timer includes CSV '
                               'export; epoch GPU peaks exclude final test and do not measure process RSS.'}
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            actual_rank = saved.get('extra_state', {}).get('ranking_loss_protocol', {})
            actual_rng = saved.get('extra_state', {}).get('training_rng_protocol', {})
            require(actual_rank.get('name') == 'train_only_sequence_pairwise_logistic_v1'
                    and actual_rank.get('mode') == 'pairwise_logistic'
                    and actual_rank.get('weight') == ARMS[arm_name]
                    and actual_rank.get('temperature') == .1
                    and actual_rank.get('min_target_delta') == .05
                    and actual_rank.get('target_source') ==
                    'folded_outer_training_soft_multi_only'
                    and actual_rank.get('source_config_sha256') == file_sha(runtime_path)
                    and actual_rng.get('training_seed') == info['fold_seeds'][fold]
                    and actual_rng.get('validation_split_seed') == info['fold_seeds'][fold],
                    f'{arm_name}/{fold}: checkpoint actual ranking loss or RNG protocol differs')
            model_config = saved.get('config', {})
            require(model_config.get('hidden_dim') == 256
                    and model_config.get('node_conv_mode') == 'single_k5'
                    and model_config.get('virtual_node_mode') == 'single'
                    and model_config.get('bond_head_type') == 'four_state_mlp'
                    and model_config.get('bond_fusion_mode') == 'concat',
                    f'{arm_name}/{fold}: checkpoint architecture differs')
            strict_config = dict(runtime, model=model_config)
            with torch.random.fork_rng(devices=[]):
                model = GraphTransformer(build_model_config(strict_config)).cpu()
            model.load_state_dict(saved['model_state_dict'], strict=True)
            require(int(model.four_state_mode_code.item()) == 4
                    and int(model.node_conv_mode_code.item()) == 1
                    and int(model.node_conv_schema_code.item()) == 1
                    and 'virtual_node_mode_code' not in saved['model_state_dict']
                    and all(torch.isfinite(parameter).all().item()
                            for parameter in model.parameters()),
                    f'{arm_name}/{fold}: checkpoint strict restore or mode marker failed')
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            del model, saved
            with pred_csv.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                require(reader.fieldnames is not None and
                        {'checkpoint_path', 'evaluation_id', 'threshold', 'pred_prob',
                         'true', 'pred', 'soft_multi', 'seq', 'charge', 'nce',
                         'prediction_export_source',
                         'prediction_probability_precision'} <= set(reader.fieldnames),
                        f'{arm_name}/{fold}: prediction source columns missing')
                evaluation_ids, prediction_rows = set(), 0
                for prediction in reader:
                    require(Path(prediction['checkpoint_path']).resolve() == checkpoint.resolve()
                            and float(prediction['threshold']) == .5
                            and prediction['prediction_export_source'] == 'same_pass_evaluate_logits'
                            and prediction['prediction_probability_precision']
                            == 'float32_roundtrip_9_significant_digits',
                            f'{arm_name}/{fold}: same-pass source, checkpoint or threshold mismatch')
                    evaluation_ids.add(prediction['evaluation_id'])
                    prediction_rows += 1
            require(prediction_rows > 0 and len(evaluation_ids) == 1
                    and bool(next(iter(evaluation_ids))),
                    f'{arm_name}/{fold}: empty/mixed prediction evaluation')
            evaluation_id = next(iter(evaluation_ids))
            pred_archive = pred_csv.with_name(
                f'{pred_csv.stem}__{evaluation_id}{pred_csv.suffix}')
            metric_archive = metric_csv.with_name(
                f'{metric_csv.stem}__{evaluation_id}{metric_csv.suffix}')
            require(pred_archive.is_file() and metric_archive.is_file()
                    and file_sha(pred_archive) == file_sha(pred_csv)
                    and file_sha(metric_archive) == file_sha(metric_csv),
                    f'{arm_name}/{fold}: same-pass prediction/metric archive missing or changed')
            files = {'pred_csv': pred_csv, 'pred_archive': pred_archive,
                     'metric_archive': metric_archive, 'test_csv': test_csv,
                     'metric_csv': metric_csv, 'checkpoint': checkpoint,
                     'config': runtime_path, 'train_csv': train_csv, 'val_csv': val_csv,
                     'validation_source_csv': Path(provenance['source_path']),
                     'four_state_manifest': Path(runtime['data']['four_state_targets_manifest']),
                     'training_log': training_log}
            entries.append({'arm': arm_name, 'fold': fold, 'seed': runtime['experiment']['seed'],
                            'split_seed': runtime['training']['validation_split_seed'],
                            'lambda': ARMS[arm_name], 'tau': .1, 'delta': .05,
                            'max_seq_len': runtime['data']['max_seq_len'],
                            'parameter_count': parameter_count,
                             'resource_observation': resource_observation,
                            'ranking_loss_protocol': actual_rank,
                            'source_commit': info['gitrev'],
                            'evaluation_id': evaluation_id,
                            'prediction_rows': prediction_rows,
                            **{name: str(path) for name, path in files.items()},
                            **{name + '_sha256': file_sha(path) for name, path in files.items()}})
    require(len(entries) == len(ARMS) * len(FOLDS)
            and {(entry['arm'], entry['fold']) for entry in entries}
            == {(arm, fold) for arm in ARMS for fold in FOLDS},
            'Stage14 audit manifest is not a complete unique four-arm/five-fold matrix')
    for fold in FOLDS:
        paired = [entry for entry in entries if entry['fold'] == fold]
        require(all(len({entry[key] for entry in paired}) == 1
                    for key in ('train_csv_sha256', 'val_csv_sha256', 'test_csv_sha256',
                                'validation_source_csv_sha256', 'four_state_manifest_sha256')),
                f'{fold}: same-round arms do not share train/validation/test/target source SHA')
    return {'stage': 'S14', 'schema_version': 1, 'status': 'completed_pending_audit',
            'ops': str(OPS), 'root': info['root'],
            'source_commit': info['gitrev'], 'source_root': info['source_root'],
            'stage13_audit': info['stage13_audit'],
            'arms': entries, 'baseline_arm': 'lambda0',
            'same_round_control': 'lambda0',
            'pairwise_mode': 'train-only logistic, tau0.1, delta0.05; no sequence-aware sampler',
            'scope': 'Manifest binds training fold, internal validation and same-pass external prediction; '
                     'independent metric audit must still verify CSV row provenance and recompute metrics.'}


def completed_terminal(info: dict, *, recovery: dict | None = None) -> dict:
    result = {'status': 'completed', 'root': info['root'], 'gitrev': info['gitrev'],
              'finished_at': utc_now(), 'audit_manifest': str(OPS / 'audit_manifest.json'),
              'audit_manifest_sha256': file_sha(OPS / 'audit_manifest.json'),
              'review_state': 'pending_independent_pod_audit'}
    if recovery is not None:
        result['recovery'] = recovery
    return result


def worker_still_active(info: dict) -> bool:
    """A one-shot concurrency guard; this does not poll training progress."""
    pid = info.get('worker_pid')
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        arguments = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except OSError:
        return False
    decoded = [argument.decode('utf-8', errors='replace') for argument in arguments]
    return str(Path(__file__).resolve()) in decoded and 'worker' in decoded


def finalize_completed(receipt_path: Path | None, source_commit: str | None,
                       recover_failed_terminal: bool) -> None:
    """Register only a completed frozen batch; never launch or rerun training."""
    pod_guard(source_commit)
    gitrev = revision(source_commit)
    require(receipt_path is not None,
            'Finalize requires --stage13-audit pointing to the original audited receipt')
    info = read_json(OPS / 'launch.json')
    require(not worker_still_active(info),
            'Stage14 worker is still active; explicit finalize must wait for worker exit')
    proof_path = OPS / 'smoke.json'
    proof = read_json(proof_path)
    require(info['gitrev'] == gitrev and info['source_root'] == str(REPO)
            and bound_run_root(info).is_dir()
            and info['smoke_sha256'] == file_sha(proof_path)
            and proof['status'] == 'passed' and proof['gitrev'] == gitrev
            and proof['source_root'] == str(REPO)
            and info['source_sha256'] == proof['source_sha256'] == source_manifest()
            and info['source_provenance'] == proof['source_provenance']
            == committed_source_provenance(proof['source_sha256'], gitrev)
            and info['configs'] == proof['configs']
            and all(info['configs'][arm] == file_sha(config_path(arm)) for arm in ARMS)
            and info['prior'] == proof['prior'] == prior_audit()
            and info['stage13_audit'] == proof['stage13_audit'] == stage13_audit(receipt_path)
            and info['arms'] == ARMS and tuple(info['folds']) == FOLDS
            and info['training_base_seed'] == info['validation_split_base_seed'] == 42
            and info['fold_seeds'] == {fold: _stable_fold_seed(42, fold) for fold in FOLDS},
            'Finalize source, audit barrier, smoke, launch, config or seed binding failed')
    verify_training_inputs(proof)
    # make_audit_manifest requires the batch status to be completed and strictly
    # restores all 20 checkpoint models before any recovery artifact is written.
    manifest = make_audit_manifest(info)
    manifest_path = OPS / 'audit_manifest.json'
    terminal_path = OPS / 'terminal.json'
    old_terminal = read_json(terminal_path) if terminal_path.exists() else None
    if old_terminal is not None:
        require(old_terminal.get('root') == info['root']
                and old_terminal.get('gitrev') == info['gitrev'],
                'Existing terminal does not belong to this frozen batch')
        if old_terminal.get('status') == 'completed':
            require(manifest_path.is_file() and read_json(manifest_path) == manifest
                    and old_terminal.get('audit_manifest_sha256') == file_sha(manifest_path),
                    'Existing completed terminal/audit manifest differs from verified batch')
            print(json.dumps({'status': 'already_completed', 'root': info['root'],
                              'audit_manifest_sha256': file_sha(manifest_path)}))
            return
        require(old_terminal.get('status') == 'failed' and recover_failed_terminal,
                'Failed terminal requires explicit --recover-failed-terminal; never overwrite silently')
        require(old_terminal.get('phase') in (None, 'post_training_finalize'),
                'Only post-training registration failure can replace a failed terminal')
    write_new_or_matching_json(manifest_path, manifest)
    recovery = {'method': 'explicit_read_only_finalize', 'recovered_at': utc_now(),
                'training_restarted': False}
    if old_terminal is not None:
        archive = OPS / 'terminal.failed-before-finalize.json'
        old_bytes = terminal_path.read_bytes()
        if archive.exists():
            require(archive.read_bytes() == old_bytes,
                    'Earlier failed-terminal archive differs; manual investigation required')
        else:
            with archive.open('xb') as stream:
                stream.write(old_bytes)
        recovery['prior_failed_terminal'] = str(archive)
        recovery['prior_failed_terminal_sha256'] = file_sha(archive)
    write_json(terminal_path, completed_terminal(info, recovery=recovery))
    print(json.dumps({'status': 'completed', 'root': info['root'],
                      'audit_manifest_sha256': file_sha(manifest_path),
                      'training_restarted': False}))


def worker() -> None:
    pod_guard()
    info = read_json(OPS / 'launch.json')
    training_completed = False
    try:
        require(info['source_root'] == str(REPO), 'Stage14 source snapshot changed')
        revision(info['gitrev'])
        bound_run_root(info)
        require(info['stage13_audit'] == stage13_audit(),
                'Audited Stage13 predecessor changed after Stage14 launch')
        require(source_manifest() == info['source_sha256']
                and all(file_sha(config_path(arm)) == info['configs'][arm] for arm in ARMS),
                'Source/config changed after launch')
        verify_training_inputs(read_json(OPS / 'smoke.json'))
        command = [PY, '-u', str(REPO / 'graph_transform/scripts/run_architecture_batch.py'),
                   '--configs', *[str(config_path(arm)) for arm in ARMS],
                   '--fold_data_dir', 'dataset/5fold_folded',
                   '--gpus', ','.join(map(str, info['gpus'])), '--run_root', info['root']]
        with (OPS / 'training.log').open('x', encoding='utf-8') as log:
            subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        training_completed = True
        require(info['stage13_audit'] == stage13_audit(),
                'Audited Stage13 predecessor changed during Stage14 training')
        verify_training_inputs(read_json(OPS / 'smoke.json'))
        manifest = make_audit_manifest(info)
        write_new_or_matching_json(OPS / 'audit_manifest.json', manifest)
        write_json(OPS / 'terminal.json', completed_terminal(info))
    except BaseException:
        write_json(OPS / 'terminal.json', {'status': 'failed', 'root': info['root'],
            'gitrev': info['gitrev'], 'finished_at': utc_now(),
            'phase': 'post_training_finalize' if training_completed else 'batch_training',
            'error': traceback.format_exc(), 'retry_policy': 'No automatic rerun'})
        raise


if __name__ == '__main__':
    pod_guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['smoke', 'launch', 'worker', 'finalize'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--stage13-audit', type=Path,
                        help='Exact independent Stage13 audit receipt, required for launch')
    parser.add_argument('--source-commit',
                        help='Exact Stage14 source commit; required for smoke, launch and finalize')
    parser.add_argument('--recover-failed-terminal', action='store_true',
                        help='Explicitly archive and replace a failed post-training terminal')
    args = parser.parse_args()
    require(not args.recover_failed_terminal or args.action == 'finalize',
            '--recover-failed-terminal is only valid for finalize')
    OPS.mkdir(parents=True, exist_ok=True)
    if args.action == 'smoke':
        smoke(args.gpu, args.source_commit)
    elif args.action == 'launch':
        launch(args.stage13_audit, args.source_commit)
    elif args.action == 'finalize':
        finalize_completed(args.stage13_audit, args.source_commit,
                           args.recover_failed_terminal)
    else:
        worker()
