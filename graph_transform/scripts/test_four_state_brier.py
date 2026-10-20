"""Pod-only preflight for CE(alpha=1) plus valid-bond union Brier."""
import argparse
import builtins
import copy
from contextlib import contextmanager
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'graph_transform'), str(ROOT/'graph_transform/scripts')]
from data.graph_dataset import GraphDataLoader
from data.row_validation import file_sha
from models.graph_transformer import GraphTransformer
from models.utils import build_model_config
from training.four_state_loss import four_state_cross_entropy
from training.loss_functions import BinaryBondLoss
from training.trainer import Trainer
from test_four_state import seed
from train_graph_model import (load_config, create_datasets, create_optimizer, create_scheduler,
                               _validate_four_state_union_brier_protocol)
import train_graph_model as train_script
from train_5fold import make_fold_config, _stable_fold_seed
from four_state_brier_ops import choose_weights, forbid_outer_test_reads, LAMBDAS

FOLDS = ('1222', '2252', '3514', '6072', '9075')
CONFIG = ROOT/'graph_transform/config/pre_synthesis_four_state_brier.yaml'
ARMS = {'control': 0.0, 'brier': 1.0}
train_script.logger = logging.getLogger('four_state_brier_test')


def expect_value_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Invalid parameter accepted')


def check_loss():
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool)
    targets = torch.full((2, 3, 4), float('nan'))
    targets[mask] = torch.tensor([[1., 0., 0., 0.], [0., .2, .3, .5],
                                  [.4, .3, .1, .2], [0., 1., 0., 0.]])
    logits = torch.tensor([[80., -80., -60., -70.], [-60., 70., -80., 40.],
                           [.2, 1., -.4, .8], [1., .3, .5, .9]], requires_grad=True)
    valid = targets[mask]
    weights = torch.tensor([1., 2., 0., 4.])
    grad = lambda loss: torch.autograd.grad(loss, logits, retain_graph=True)[0]
    ce_rows = -(valid * logits.log_softmax(-1)).sum(-1)
    # Independent union reference, without calling production union helpers.
    p = torch.sigmoid(torch.logsumexp(logits[:, 1:], -1) - logits[:, 0])
    q = 1-valid[:, 0]
    assert torch.allclose(p, 1-logits.softmax(-1)[:, 0], atol=1e-7, rtol=0)
    mse_rows = (p-q).square()
    for w in (None, weights):
        reduce = (lambda rows: rows.mean()) if w is None else (lambda rows: (rows*w).sum()/w.sum())
        old = reduce(ce_rows)
        default = four_state_cross_entropy(logits, targets, mask, w)
        zero = four_state_cross_entropy(logits, targets, mask, w, conditional_weight=1., union_brier_weight=0.)
        assert torch.equal(old, default) and torch.equal(default, zero)
        assert torch.equal(grad(old), grad(default)) and torch.equal(grad(old), grad(zero))
        mse = reduce(mse_rows)
        for value in (.25, 1., 4.):
            actual = four_state_cross_entropy(logits, targets, mask, w, conditional_weight=1., union_brier_weight=value)
            expected = old+value*mse
            assert torch.allclose(actual, expected, atol=3e-6, rtol=1e-6)
            assert torch.allclose(grad(actual), grad(old)+value*grad(mse), atol=1e-6, rtol=1e-5)
            assert torch.isfinite(actual) and torch.isfinite(grad(actual)).all()
            poisoned = targets.clone()
            poisoned[~mask] = 1e9
            assert torch.equal(actual, four_state_cross_entropy(logits, poisoned, mask, w, 1., value))
    a = four_state_cross_entropy(logits, targets, mask, weights, 1., 1.)
    b = four_state_cross_entropy(logits, targets, mask, weights*11, 1., 1.)
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)
    for invalid in (True, '1', None, -.1, float('nan'), float('inf')):
        expect_value_error(lambda: four_state_cross_entropy(logits, targets, mask, weights, 1., invalid))
    for invalid in (torch.zeros(4), torch.tensor([1., -1., 1., 1.]), torch.full((4,), float('nan'))):
        expect_value_error(lambda: four_state_cross_entropy(logits, targets, mask, invalid, 1., 1.))
    expect_value_error(lambda: four_state_cross_entropy(logits, torch.full_like(targets, .2), mask, weights, 1., 1.))
    bad_mask = mask.float(); bad_mask[0, 0] = .5
    expect_value_error(lambda: four_state_cross_entropy(logits, targets, bad_mask, weights, 1., 1.))
    # Trainer must reject malformed lambda and positive lambda on binary heads.
    tiny = torch.nn.Linear(1, 1)
    tiny.bond_head_type = 'four_state_mlp'
    optimizer = torch.optim.SGD(tiny.parameters(), lr=.1)
    for invalid in (True, '1', None, -.1, float('nan'), float('inf')):
        expect_value_error(lambda: Trainer(tiny, optimizer, None, config={
            'loss': {'main_loss': 'binary_cross_entropy', 'four_state_union_brier_weight': invalid}}))
    tiny.bond_head_type = 'concat_mlp'
    expect_value_error(lambda: Trainer(tiny, optimizer, None, config={
        'loss': {'main_loss': 'binary_cross_entropy', 'four_state_union_brier_weight': 1.}}))


@contextmanager
def forbid_test_csv():
    """Fail before test CSV content access; this preflight cannot tune on test."""
    attempts = []
    def checked(original):
        def call(path, *args, **kwargs):
            name = str(getattr(path, 'name', path)).replace('\\', '/').lower()
            if name.endswith('.test.fbr.multi.csv'):
                attempts.append(name)
                raise AssertionError('External test CSV access forbidden during Brier preflight')
            return original(path, *args, **kwargs)
        return call
    with patch('builtins.open', checked(builtins.open)), patch('io.open', checked(io.open)), \
         patch('pandas.read_csv', checked(pd.read_csv)):
        # Test the barrier without opening or creating a test file.
        try:
            pd.read_csv('/deliberate_guard_probe/1222.test.fbr.multi.csv')
        except AssertionError:
            pass
        assert len(attempts) == 1
        attempts.clear()
        yield attempts


def normalized(cfg, identity):
    cfg = copy.deepcopy(cfg)
    cfg['loss'].pop('four_state_union_brier_weight', None)
    cfg['experiment'].pop('description', None)
    cfg['experiment'].pop('loss_protocol', None)
    cfg['data']['test_csv_path'] = None
    return json.dumps(cfg, sort_keys=True).replace(identity, 'four_state_VARIANT')


def check_protocol(cfg):
    assert _validate_four_state_union_brier_protocol(cfg, None) is True
    invalid = [('data', 'test_csv_path', 'unopened.test.fbr.multi.csv'),
               ('loss', 'four_state_conditional_weight', .5),
               ('loss', 'four_state_conditional_weight', True),
               ('loss', 'four_state_conditional_weight', '1'),
               ('model', 'bond_head_type', 'concat_mlp'),
               ('model', 'use_charge_partition_features', True),
               ('model', 'use_charge_partition_features', 0)]
    for section, name, value in invalid:
        bad = copy.deepcopy(cfg)
        bad[section][name] = value
        expect_value_error(lambda: _validate_four_state_union_brier_protocol(bad, None))
    expect_value_error(lambda: _validate_four_state_union_brier_protocol(cfg, 'unused_checkpoint.pt'))
    legacy = copy.deepcopy(cfg)
    legacy['experiment'].pop('loss_protocol')
    legacy['data']['test_csv_path'] = 'legacy_unopened.test.fbr.multi.csv'
    assert _validate_four_state_union_brier_protocol(legacy, 'unused_checkpoint.pt') is False


def check_selection():
    expected = dict(zip(FOLDS, (0., .25, 1., 4., .25)))
    rows = [dict(fold=f, weight=w, val_q_brier=.1 if w == expected[f] else .4, val_f1=.8)
            for f in FOLDS for w in LAMBDAS]
    selected = choose_weights(rows)
    assert {f: row['weight'] for f, row in selected.items()} == expected
    # Unrelated outer-test columns must not influence validation-only selection.
    poisoned = [dict(row, test_q_brier=1-row['val_q_brier'], test_f1=row['weight']/4.,
                     external_test_score=-row['weight']) for row in rows]
    assert {f: row['weight'] for f, row in choose_weights(poisoned).items()} == expected
    expect_value_error(lambda: choose_weights(rows[:-1]))
    for key in ('val_q_brier', 'val_f1'):
        bad = copy.deepcopy(rows)
        bad[0][key] = float('nan')
        expect_value_error(lambda: choose_weights(bad))
    # Exact ties select the smaller lambda, including the unmodified CE fallback.
    ties = copy.deepcopy(rows)
    for row in ties:
        if row['fold'] == FOLDS[0]:
            row['val_q_brier'] = .1 if row['weight'] in (.25, 1.) else .4
        if row['fold'] == FOLDS[-1]:
            row['val_q_brier'] = .2
    tied = choose_weights(ties)
    assert tied[FOLDS[0]]['weight'] == .25 and tied[FOLDS[-1]]['weight'] == 0.
    for fold in FOLDS[1:-1]:
        assert tied[fold]['weight'] == selected[fold]['weight']


def check_outer_test_audit_guard(temp):
    # Audit hooks are process-lifetime state. Probe production guard only in a
    # disposable subprocess, using nonexistent names rather than actual data.
    code = '''
import io, json, os, pathlib, sys
sys.path[:0] = json.loads(sys.argv[1])
from four_state_brier_ops import forbid_outer_test_reads
forbid_outer_test_reads()
path = sys.argv[2]
count = 0
for opener in (open, io.open, pathlib.Path.open, lambda p: os.open(p, os.O_RDONLY)):
    try:
        opener(pathlib.Path(path))
    except PermissionError:
        count += 1
    else:
        raise AssertionError('Outer-test guard did not block before opening')
assert count == 4
print(json.dumps({'blocked_before_open': count}))
'''
    assert callable(forbid_outer_test_reads)
    paths = [str(ROOT/'graph_transform'), str(ROOT/'graph_transform/scripts')]
    probe = temp/'nonexistent_guard_probe.test.fbr.multi.csv'
    assert not probe.exists()
    done = subprocess.run([sys.executable, '-c', code, json.dumps(paths), str(probe)],
                          cwd=ROOT, text=True, capture_output=True, check=True)
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result == {'blocked_before_open': 4} and not probe.exists()
    return result


def run(output):
    assert torch.cuda.is_available(), 'Run only on an available pod GPU'
    torch.set_num_threads(2)
    check_loss()
    args = argparse.Namespace(epochs=None, batch_size=None, learning_rate=None, device=None)
    cfg = load_config(CONFIG, args)
    old = load_config(ROOT/'graph_transform/config/pre_synthesis_four_state_charge_control.yaml', args)
    assert normalized(cfg, 'four_state_brier') == normalized(old, 'four_state_charge_control')
    assert cfg['data']['test_csv_path'] is None
    assert cfg['model']['bond_head_type'] == 'four_state_mlp'
    assert cfg['model']['use_charge_partition_features'] is False
    assert cfg['loss']['four_state_conditional_weight'] == 1.
    assert cfg['loss']['four_state_union_brier_weight'] == 0.
    assert cfg['experiment']['loss_protocol'] == 'four_state_union_brier_v1'
    check_protocol(cfg)
    check_selection()
    prior = json.loads(Path('/mnt/pvc/graphtrans/four_state_ops/smoke.json').read_text())
    report = {'status': 'running', 'config_sha256': file_sha(CONFIG), 'folds': {}, 'arms': {},
              'checks': ['default and explicit lambda=0 equal old CE loss AND gradient bit for bit',
                         'lambda=.25/1/4 matches CE plus union MSE and additive gradients',
                         'q endpoints, extreme logits, poisoned padding, weight normalization and invalid values checked',
                         'marked protocol rejects test access/resume/changed alpha/head/charge; unmarked legacy unaffected',
                         'selection rejects missing/nonfinite scores; exact ties choose smaller lambda; lambda=0 eligible; per-fold independent',
                         'fake outer-test score fields cannot change validation-only choices',
                         'original alpha=1 model/config retained; charge feature branch disabled']}
    with tempfile.TemporaryDirectory(prefix='four_state_brier_') as folder, forbid_test_csv() as attempts:
        temp = Path(folder)
        report['audit_guard_probe'] = check_outer_test_audit_guard(temp)
        for fold in FOLDS:
            c, _, _ = make_fold_config(cfg, fold, str(ROOT/'dataset/5fold_folded'), str(temp/'control'), _stable_fold_seed(42, fold))
            # make_fold_config normally reinstates test path; remove BEFORE loading.
            c['data']['test_csv_path'] = None
            c['data']['cache_dir'] = str(temp/fold/'cache')
            c['data']['cache_graphs'] = False
            train, val, test = create_datasets(c)
            assert test is None
            train_seqs, val_seqs = set(train.data['seq']), set(val.data['seq'])
            assert not train_seqs & val_seqs
            proof = {'train_rows': len(train), 'val_rows': len(val), 'train_sequences': len(train_seqs),
                     'val_sequences': len(val_seqs), 'train_sha256': file_sha(train.csv_path),
                     'val_sha256': file_sha(val.csv_path), 'source_sha256': val.row_validation_provenance['source_sha256']}
            assert proof == prior['folds'][fold], fold
            assert 'four_state_targets' not in val[0]
            report['folds'][fold] = proof
            if fold == FOLDS[0]:
                first = (c, train)
        c, train = first
        batch = GraphDataLoader._default_collate_fn(None, [train[i] for i in range(16)])
        assert len(batch['sequences']) == 16 and int(batch['label_mask'].sum()) == 427
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        initial, rng = None, None
        for arm, weight in ARMS.items():
            arm_cfg = copy.deepcopy(c)
            arm_cfg['loss']['four_state_union_brier_weight'] = weight
            arm_cfg['debug']['diagnostic_dir'] = str(temp/arm/'diagnostics')
            seed()
            model = GraphTransformer(build_model_config(arm_cfg)).cuda()
            assert not model.use_charge_partition_features
            if initial is None:
                initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
                rng = (torch.get_rng_state().clone(), [x.clone() for x in torch.cuda.get_rng_state_all()])
            else:
                assert initial.keys() == model.state_dict().keys()
                assert all(torch.equal(v, initial[k]) for k, v in model.state_dict().items())
                assert torch.equal(torch.get_rng_state(), rng[0])
                assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), rng[1]))
            opt = create_optimizer(model, arm_cfg)
            trainer = Trainer(model, opt, BinaryBondLoss(arm_cfg['loss']), create_scheduler(opt, arm_cfg),
                              torch.device('cuda:0'), arm_cfg, train_script.logger)
            assert trainer.use_amp and trainer.four_state_conditional_weight == 1.
            assert trainer.four_state_union_brier_weight == weight
            model.train()
            before = model.bond_head[3].weight.detach().clone()
            seed()
            loss, stats = trainer._forward_pass(batch)
            logits = model.last_four_state_logits
            valid = batch['four_state_targets'][batch['label_mask'].bool()].float()
            ce_rows = -(valid*logits.float().log_softmax(-1)).sum(-1)
            p = torch.sigmoid(logits.float()[:, 1:].logsumexp(-1)-logits.float()[:, 0])
            brier_rows = (p-(1-valid[:, 0])).square()
            row_weights = trainer._masked_row_weights(batch)
            reduce = (lambda rows: rows.mean()) if row_weights is None else (lambda rows: (rows*row_weights).sum()/row_weights.sum())
            expected = reduce(ce_rows+weight*brier_rows)
            assert torch.allclose(loss, expected, atol=1e-6, rtol=1e-6)
            if weight == 0:
                assert torch.equal(loss, reduce(ce_rows))
            trainer._backward_pass(loss)
            assert not torch.equal(before, model.bond_head[3].weight)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            assert (model.bond_head[3].weight.grad.abs().sum(-1)>0).all()
            assert stats['valid_bond_count'] == 427
            report['arms'][arm] = {'conditional_weight': trainer.four_state_conditional_weight,
                                   'union_brier_weight': weight, 'parameters': sum(p.numel() for p in model.parameters()),
                                   'amp': trainer.use_amp, 'optimizer_updates': 1, 'valid_bonds': stats['valid_bond_count'],
                                   'loss': float(loss.detach()), 'ce': float(reduce(ce_rows).detach()),
                                   'union_brier': float(reduce(brier_rows).detach())}
            del model, trainer, opt
        assert not attempts
    report['checks'] += ['five train/validation split SHAs match prior proof; train and val sequences disjoint',
                         'test path disabled and test CSV access barrier active; no external test CSV opened',
                         'lambda=0/1 have identical full model initial state and CPU/CUDA RNG',
                         '16 real folded groups / 427 valid bonds: both production AMP optimizer steps finite and loss matches formula']
    report['external_test_csv_reads'] = 0
    report['test_isolation_basis'] = 'unchanged train/validation hashes versus prior audited smoke; no new test access'
    report['status'] = 'passed'
    Path(output).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    os.chdir(ROOT)
    run(parser.parse_args().output)
