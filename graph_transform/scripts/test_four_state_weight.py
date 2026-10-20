"""Pod-only preflight for the four-state conditional-loss weight experiment."""
import argparse
import copy
import json
import logging
import os
from pathlib import Path
import sys
import tempfile

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'graph_transform'), str(ROOT / 'graph_transform/scripts')]
from data.graph_dataset import GraphDataLoader
from data.row_validation import file_sha
from models.graph_transformer import GraphTransformer
from models.utils import build_model_config
from training.four_state_loss import four_state_cross_entropy
from training.loss_functions import BinaryBondLoss
from training.trainer import Trainer
from test_four_state import seed
from train_graph_model import load_config, create_datasets, create_optimizer, create_scheduler
import train_graph_model as train_script
from train_5fold import make_fold_config, _stable_fold_seed

ARMS = {'control': 1.0, 'half': .5, 'double': 2.0}
FOLDS = ('1222', '2252', '3514', '6072', '9075')
train_script.logger = logging.getLogger('four_state_weight_test')


def check_loss():
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool)
    targets = torch.full((2, 3, 4), float('nan'))
    targets[mask] = torch.tensor([[1., 0., 0., 0.], [0., .2, .3, .5], [.4, .3, .1, .2], [0., 1., 0., 0.]])
    logits = torch.tensor([[80., -80., -60., -70.], [-60., 70., -80., 40.], [.2, 1., -.4, .8], [1., .3, .5, .9]], requires_grad=True)
    weights = torch.tensor([1., 2., 3., 4.])
    t = targets[mask]
    old = (-(t * logits.log_softmax(-1)).sum(-1) * weights).sum() / weights.sum()
    default = four_state_cross_entropy(logits, targets, mask, weights)
    explicit = four_state_cross_entropy(logits, targets, mask, weights, 1.)
    assert torch.equal(old, default) and torch.equal(default, explicit)
    assert torch.equal(torch.autograd.grad(old, logits, retain_graph=True)[0],
                       torch.autograd.grad(default, logits, retain_graph=True)[0])
    z = logits[:, 1:].logsumexp(-1) - logits[:, 0]
    union = F.binary_cross_entropy_with_logits(z, 1-t[:, 0], reduction='none')
    conditional = -(t[:, 1:] * logits[:, 1:].log_softmax(-1)).sum(-1)
    grads = {}
    for alpha in (0., .5, 1., 2.):
        value = four_state_cross_entropy(logits, targets, mask, weights, alpha)
        expected = ((union + alpha * conditional) * weights).sum() / weights.sum()
        assert torch.allclose(value, expected, atol=3e-6, rtol=1e-6)
        grads[alpha] = torch.autograd.grad(value, logits, retain_graph=True)[0]
        assert torch.isfinite(value) and torch.isfinite(grads[alpha]).all()
    # Conditional CE cannot change the gradient of a common shift of all three
    # positive logits: that shift controls union q while leaving type ratios fixed.
    for alpha in (.5, 1., 2.):
        assert torch.allclose(grads[alpha][:, 1:].sum(-1), grads[0.][:, 1:].sum(-1), atol=1e-6)
    assert torch.allclose(grads[2.]-grads[0.], 4*(grads[.5]-grads[0.]), atol=1e-6)
    for invalid in (True, '0.5', -.1, float('nan'), float('inf')):
        try:
            four_state_cross_entropy(logits, targets, mask, weights, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid weight accepted')


def normalized(cfg, arm):
    cfg = copy.deepcopy(cfg)
    cfg['loss'].pop('four_state_conditional_weight', None)
    cfg['experiment'].pop('description', None)
    return json.dumps(cfg, sort_keys=True).replace(f'four_state_weight_{arm}', 'four_state_VARIANT')


def run(output):
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    check_loss()
    args = argparse.Namespace(epochs=None, batch_size=None, learning_rate=None, device=None)
    configs = {arm: load_config(ROOT / f'graph_transform/config/pre_synthesis_four_state_weight_{arm}.yaml', args) for arm in ARMS}
    canonical = normalized(configs['control'], 'control')
    old = load_config(ROOT / 'graph_transform/config/pre_synthesis_four_state_joint.yaml', args)
    old['experiment'].pop('description', None)
    assert canonical == json.dumps(old, sort_keys=True).replace('four_state_joint', 'four_state_VARIANT')
    for arm, cfg in configs.items():
        assert normalized(cfg, arm) == canonical
        assert cfg['loss']['four_state_conditional_weight'] == ARMS[arm]
        assert cfg['model']['bond_head_type'] == 'four_state_mlp'
    prior = json.loads(Path('/mnt/pvc/graphtrans/four_state_ops/smoke.json').read_text())
    report = {'status': 'running', 'folds': {}, 'arms': {}, 'checks': [
        'default and alpha=1 preserve old CE value AND gradient exactly on synthetic masked fixtures',
        'alpha=0/0.5/1/2 loss decomposition and conditional-gradient scaling',
        'q=0/1, extreme logits, poisoned padding, sample weights and invalid configuration rejection',
        'only declared alpha and output/cache identity paths differ; old joint control preserved']}
    with tempfile.TemporaryDirectory(prefix='four_state_weight_') as folder:
        temp = Path(folder)
        for fold in FOLDS:
            cfg, _, _ = make_fold_config(configs['control'], fold, str(ROOT/'dataset/5fold_folded'), str(temp/'control'), _stable_fold_seed(42, fold))
            cfg['data']['cache_dir'] = str(temp/fold/'cache')
            cfg['data']['cache_graphs'] = False
            train, val, test = create_datasets(cfg)
            seqs = [set(ds.data['seq']) for ds in (train, val, test)]
            assert not (seqs[0]&seqs[1] or seqs[0]&seqs[2] or seqs[1]&seqs[2])
            proof = {'train_rows': len(train), 'val_rows': len(val), 'train_sequences': len(seqs[0]),
                     'val_sequences': len(seqs[1]), 'train_sha256': file_sha(train.csv_path),
                     'val_sha256': file_sha(val.csv_path), 'source_sha256': val.row_validation_provenance['source_sha256']}
            assert proof == prior['folds'][fold], fold
            assert 'four_state_targets' not in val[0] and 'four_state_targets' not in test[0]
            report['folds'][fold] = proof
            if fold == FOLDS[0]:
                first = (cfg, train)
        cfg, train = first
        batch = GraphDataLoader._default_collate_fn(None, [train[i] for i in range(16)])
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        initial = None
        for arm, alpha in ARMS.items():
            c = copy.deepcopy(cfg)
            c['loss']['four_state_conditional_weight'] = alpha
            c['debug']['diagnostic_dir'] = str(temp/arm/'diagnostics')
            seed()
            model = GraphTransformer(build_model_config(c)).cuda()
            if initial is None:
                initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                assert all(torch.equal(v, initial[k]) for k, v in model.state_dict().items())
            opt = create_optimizer(model, c)
            trainer = Trainer(model, opt, BinaryBondLoss(c['loss']), create_scheduler(opt, c), torch.device('cuda:0'), c, train_script.logger)
            model.train()
            before = model.bond_head[3].weight.detach().clone()
            seed()
            loss, stats = trainer._forward_pass(batch)
            logits = model.last_four_state_logits
            expected = four_state_cross_entropy(logits, batch['four_state_targets'], batch['label_mask'], trainer._masked_row_weights(batch), alpha)
            assert torch.equal(loss, expected) and trainer.four_state_conditional_weight == alpha
            logits.retain_grad()
            trainer._backward_pass(loss)
            assert not torch.equal(before, model.bond_head[3].weight)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            assert (logits.grad.abs().sum(0)>0).all()
            report['arms'][arm] = {'conditional_weight': alpha, 'parameters': sum(p.numel() for p in model.parameters()),
                                   'amp': trainer.use_amp, 'optimizer_updates': 1, 'valid_bonds': stats['valid_bond_count']}
            del model, trainer, opt
    report['configs'] = {a: file_sha(ROOT/f'graph_transform/config/pre_synthesis_four_state_weight_{a}.yaml') for a in ARMS}
    report['status'] = 'passed'
    Path(output).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    os.chdir(ROOT)
    run(p.parse_args().output)
