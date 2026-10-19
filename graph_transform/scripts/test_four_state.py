"""Run on pod: loss/mask, initialization, validation isolation and CUDA update checks."""
import argparse
import copy
import json
import logging
import os
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'graph_transform'), str(ROOT / 'graph_transform/scripts')]
from data.row_validation import restore_row_validation, file_sha
from data.graph_dataset import GraphDataLoader
from models.graph_transformer import GraphTransformer
from models.utils import build_model_config
from models.four_state import union_logit
from training.four_state_loss import four_state_cross_entropy
from training.loss_functions import BinaryBondLoss
from training.trainer import Trainer
from train_graph_model import load_config, create_datasets, create_optimizer, create_scheduler
import train_graph_model as training_script
training_script.logger = logging.getLogger('four_state_test')
from train_5fold import make_fold_config, _stable_fold_seed

FOLDS = ('1222', '2252', '3514', '6072', '9075')


def seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def check_loss():
    logits = torch.tensor([[.2, 1., -.4, .8], [-1., .3, .5, .9]], requires_grad=True)
    target = torch.tensor([[[.2, .3, .1, .4], [float('nan')] * 4], [[.7, .1, .1, .1], [float('nan')] * 4]])
    mask = torch.tensor([[1., 0.], [1., 0.]])
    weights = torch.tensor([1., 2.])
    value = four_state_cross_entropy(logits, target, mask, weights)
    valid = target[mask.bool()]
    expected = -(valid * logits.log_softmax(-1)).sum(-1)
    assert torch.allclose(value, (expected * weights).sum() / 3)
    q = 1 - valid[:, 0]
    ce_union = F.binary_cross_entropy_with_logits(union_logit(logits), q, reduction='none')
    cond = -(valid[:, 1:] * logits[:, 1:].log_softmax(-1)).sum(-1)
    assert torch.allclose(expected, ce_union + cond, atol=1e-6)
    assert torch.allclose(torch.sigmoid(union_logit(logits)), 1 - logits.softmax(-1)[:, 0], atol=1e-7)
    value.backward()
    assert torch.isfinite(logits.grad).all() and (logits.grad.abs().sum(0) > 0).all()
    for bad in (torch.full_like(target, .2), torch.full_like(target, -1)):
        try:
            four_state_cross_entropy(logits, bad, mask)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid state distribution accepted')


def check_synthetic_validation(temp):
    frame = pd.DataFrame([
        {'seq': 'AC', 'charge': 2, 'nce': 30, 'true_multi': '0', 'soft_multi': '0.5', 'pep_mass': 100.1},
        {'seq': 'AC', 'charge': 2, 'nce': 30, 'true_multi': '1', 'soft_multi': '0.5', 'pep_mass': 100.2},
        {'seq': 'AD', 'charge': 2, 'nce': 30, 'true_multi': '1', 'soft_multi': '1.0', 'pep_mass': 110.1}])
    folded = frame.iloc[[0, 2]].copy()
    folded['group_n'] = [2, 1]
    path = temp / 'original.csv'
    frame.to_csv(path, index=False)
    # pandas serializes/parses these one-bond q columns as floats; string
    # equivalence in real multi-bond inputs is tested by the production fixtures.
    folded['soft_multi'] = folded['soft_multi'].astype(float)
    result, _ = restore_row_validation(folded, {'AC'}, path)
    assert list(result['true_multi']) == [0, 1] and list(result['pep_mass']) == [100.1, 100.2]
    broken = folded.copy()
    broken['group_n'] = [1, 1]
    try:
        restore_row_validation(broken, {'AC'}, path)
    except ValueError:
        pass
    else:
        raise AssertionError('Mismatched validation source count accepted')


def run(output):
    assert torch.cuda.is_available(), 'Run this test on an available pod GPU'
    torch.set_num_threads(2)
    check_loss()
    report = {'status': 'running', 'folds': {}, 'arms': {}, 'checks': [
        'joint CE decomposes into union BCE plus positive-state conditional CE',
        'padding excluded even with poisoned padding targets', 'joint q identity and finite four-state gradients']}
    args = argparse.Namespace(epochs=None, batch_size=None, learning_rate=None, device=None)
    configs = {a: load_config(ROOT / f'graph_transform/config/pre_synthesis_four_state_{a}.yaml', args)
               for a in ('control', 'joint')}
    original = load_config(ROOT / 'graph_transform/config/pre_synthesis_fold1222_theory_film.yaml', args)
    assert build_model_config(original).to_dict() == build_model_config(configs['control']).to_dict()
    expected = build_model_config(configs['control']).to_dict()
    expected['bond_head_type'] = 'four_state_mlp'
    assert build_model_config(configs['joint']).to_dict() == expected
    for section in ('training', 'optimizer', 'loss'):
        a, b = copy.deepcopy(configs['control'][section]), copy.deepcopy(configs['joint'][section])
        a.pop('checkpoint_dir', None)
        b.pop('checkpoint_dir', None)
        assert a == b
    with tempfile.TemporaryDirectory(prefix='four_state_preflight_') as folder:
        temp = Path(folder)
        check_synthetic_validation(temp)
        first = {}
        for fold in FOLDS:
            pair = {}
            for arm, cfg in configs.items():
                c, _, _ = make_fold_config(cfg, fold, str(ROOT / 'dataset/5fold_folded'), str(temp / arm), _stable_fold_seed(42, fold))
                c['data']['cache_dir'] = str(temp / arm / fold / 'cache')
                c['data']['cache_graphs'] = False
                train, val, test = create_datasets(c)
                seqs = [set(ds.data['seq']) for ds in (train, val, test)]
                assert not (seqs[0] & seqs[1] or seqs[0] & seqs[2] or seqs[1] & seqs[2])
                prov = val.row_validation_provenance
                assert len(val) == prov['val_rows'] and len(val) > len(prov['val_sequences'])
                pair[arm] = {'train_rows': len(train), 'val_rows': len(val),
                             'train_sequences': len(seqs[0]), 'val_sequences': len(seqs[1]),
                             'train_sha256': file_sha(train.csv_path), 'val_sha256': file_sha(val.csv_path),
                             'source_sha256': prov['source_sha256']}
                assert 'four_state_targets' not in val[0] and 'four_state_targets' not in test[0]
                if fold == FOLDS[0]:
                    first[arm] = (c, train, val)
            assert pair['control'] == pair['joint'], 'Paired train/validation rows or split SHA differs'
            report['folds'][fold] = pair['control']
        c, ds, val = first['control']
        samples = [ds[i] for i in range(16)]
        batch = GraphDataLoader._default_collate_fn(None, samples)
        joint_batch = GraphDataLoader._default_collate_fn(None, [first['joint'][1][i] for i in range(16)])
        for key in ('edge_index', 'edge_attr', 'bond_edge_map', 'labels', 'label_mask', 'soft_labels', 'sample_weights'):
            assert torch.equal(batch[key], joint_batch[key])
        mask = joint_batch['label_mask'].bool()
        assert torch.allclose(1 - joint_batch['four_state_targets'][..., 0][mask], joint_batch['soft_labels'][mask], atol=1e-7)
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        joint_batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in joint_batch.items()}
        seed()
        control = GraphTransformer(build_model_config(c)).cuda()
        seed()
        joint = GraphTransformer(build_model_config(first['joint'][0])).cuda()
        for key, value in control.state_dict().items():
            if not key.startswith('bond_head.3.'):
                assert torch.equal(value, joint.state_dict()[key]), key
        control.eval()
        joint.eval()
        with torch.no_grad():
            p, q = control(batch), joint(joint_batch)
        assert torch.allclose(p, q, atol=3e-6, rtol=1e-5), 'Initial union q differs from original'
        changed = dict(joint_batch)
        changed['four_state_targets'] = torch.rand_like(joint_batch['four_state_targets'])
        with torch.no_grad():
            assert torch.equal(q, joint(changed)), 'Targets enter inference inputs'
        for model, other in ((control, joint), (joint, control)):
            before = {k: v.detach().clone() for k, v in model.state_dict().items()}
            try:
                model.load_state_dict(other.state_dict(), strict=False)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Cross-mode checkpoint accepted')
            assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
        for arm, model, b in (('control', control, batch), ('joint', joint, joint_batch)):
            cfg = copy.deepcopy(first[arm][0])
            cfg['debug']['diagnostic_dir'] = str(temp / arm / 'diagnostics')
            opt = create_optimizer(model, cfg)
            trainer = Trainer(model, opt, BinaryBondLoss(cfg['loss']), create_scheduler(opt, cfg), torch.device('cuda:0'), cfg, logging.getLogger('four_state_test'))
            model.train()
            before = model.bond_head[3].weight.detach().clone()
            loss, stats = trainer._forward_pass(b)
            assert torch.isfinite(loss)
            if arm == 'joint':
                logits = model.last_four_state_logits
                logits.retain_grad()
            trainer._backward_pass(loss)
            assert not torch.equal(before, model.bond_head[3].weight)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            if arm == 'joint':
                assert (logits.grad.abs().sum(0) > 0).all()
            report['arms'][arm] = {'parameters': sum(p.numel() for p in model.parameters()),
                                   'amp': trainer.use_amp, 'optimizer_updates': 1, 'valid_bonds': stats['valid_bond_count']}
            model.eval()
        # One full-size validation batch: preserve real row inputs, exercise the
        # production non-AMP eval path, and check a safe memory budget.
        eval_batch = GraphDataLoader._default_collate_fn(None, [val[i] for i in range(512)])
        eval_batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in eval_batch.items()}
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for model in (control, joint):
                pred = model(eval_batch)
                assert pred.shape == eval_batch['labels'].shape and torch.isfinite(pred).all()
        torch.cuda.synchronize()
        report['validation_batch_size'] = 512
        report['validation_batch_peak_mb'] = torch.cuda.max_memory_allocated() / 1024**2
        assert report['validation_batch_peak_mb'] < 18000
    report['status'] = 'passed'
    report['configs'] = {a: file_sha(ROOT / f'graph_transform/config/pre_synthesis_four_state_{a}.yaml') for a in configs}
    Path(output).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    os.chdir(ROOT)
    run(a.output)
