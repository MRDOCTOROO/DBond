"""Pod-only preflight: charge descriptors, isolation, checkpoint safety and AMP."""
import argparse
import copy
import json
import logging
import os
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'graph_transform'), str(ROOT/'graph_transform/scripts')]
from data.graph_dataset import GraphDataLoader
from data.row_validation import file_sha
from models.graph_transformer import GraphTransformer
from models.utils import build_model_config
from models.charge_partition import ChargePartitionReadout, FEATURE_DIM
from models.four_state import union_logit
from training.four_state_loss import four_state_cross_entropy
from training.loss_functions import BinaryBondLoss
from training.trainer import Trainer
from test_four_state import seed
from train_graph_model import load_config, create_datasets, create_optimizer, create_scheduler
import train_graph_model as train_script
from train_5fold import make_fold_config, _stable_fold_seed

ARMS = {'control': False, 'features': True}
FOLDS = ('1222', '2252', '3514', '6072', '9075')
train_script.logger = logging.getLogger('four_state_charge_test')


def check_features():
    alphabet = 'ACDEFGHIKLMNPQRSTVWYBOXZ'
    mapping = {c: i+1 for i,c in enumerate(alphabet)}
    readout = ChargePartitionReadout(mapping, len(alphabet)+1)
    sequences = ['RBOKHXDE', 'AR', 'A', '']
    tokens = torch.zeros(4,8,dtype=torch.long)
    mask = torch.zeros(4,7,dtype=torch.bool)
    charge = torch.tensor([3., 1., 1., 1.])
    expected = []
    for row,seq in enumerate(sequences):
        tokens[row,:len(seq)] = torch.tensor([mapping[c] for c in seq], dtype=torch.long)
        mask[row,:max(len(seq)-1,0)] = True
        for cut in range(1,len(seq)):
            # Independent character-count reference, including B/O/X and acid groups.
            counts = [sum(c in group for c in side)/len(seq)
                      for side in (seq[:cut],seq[cut:]) for group in ('R','KBO','H','X','D','E')]
            expected.append(counts + [cut/32.,(len(seq)-cut)/32.,float(charge[row])/10.,
                (float(charge[row])-seq.count('R'))/len(seq),
                (float(charge[row])-sum(c in 'RKBOHX' for c in seq))/len(seq)])
    actual = readout.features(tokens,charge,mask)
    assert FEATURE_DIM == 17 and actual.shape == (8,17)
    assert torch.allclose(actual, torch.tensor(expected), atol=1e-7, rtol=0)
    assert torch.equal(readout(tokens,charge,mask), torch.zeros(8,4))
    changed = readout.features(tokens,charge+1,mask)
    assert torch.equal(actual[:,:14], changed[:,:14])
    assert torch.all(changed[:,14:] > actual[:,14:])
    # Padding width and batch neighbors cannot affect any valid bond descriptor.
    for row in (0,1):
        n = len(sequences[row])
        individual = readout.features(tokens[row:row+1,:n], charge[row:row+1], mask[row:row+1,:n-1])
        start = 0 if row == 0 else 7
        assert torch.equal(individual,actual[start:start+n-1])
    assert readout.features(torch.ones(2,1,dtype=torch.long),torch.ones(2),torch.zeros(2,0,dtype=torch.bool)).shape == (0,17)


def normalized(cfg, arm):
    cfg = copy.deepcopy(cfg)
    cfg['model'].pop('use_charge_partition_features', None)
    cfg['experiment'].pop('description', None)
    return json.dumps(cfg,sort_keys=True).replace(f'four_state_charge_{arm}','four_state_VARIANT')


def run(output):
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    check_features()
    args = argparse.Namespace(epochs=None,batch_size=None,learning_rate=None,device=None)
    configs = {a:load_config(ROOT/f'graph_transform/config/pre_synthesis_four_state_charge_{a}.yaml',args) for a in ARMS}
    canonical = normalized(configs['control'],'control')
    old = load_config(ROOT/'graph_transform/config/pre_synthesis_four_state_weight_control.yaml',args)
    old['experiment'].pop('description',None)
    assert canonical == json.dumps(old,sort_keys=True).replace('four_state_weight_control','four_state_VARIANT')
    for arm,c in configs.items():
        assert normalized(c,arm)==canonical
        assert c['model']['use_charge_partition_features'] == ARMS[arm]
        assert c['loss']['four_state_conditional_weight']==1.
    prior = json.loads(Path('/mnt/pvc/graphtrans/four_state_ops/smoke.json').read_text())
    report = {'status':'running','folds':{},'arms':{},'checks':[
        '17 descriptors equal an independent count reference; B/O/X/D/E handled explicitly',
        'valid descriptors invariant to batch padding; lengths 0/1 safe; charge changes only charge features',
        'config parity with previous alpha=1 control; only feature flag and output/cache identities differ']}
    with tempfile.TemporaryDirectory(prefix='four_state_charge_') as folder:
        temp=Path(folder)
        for fold in FOLDS:
            c,_,_=make_fold_config(configs['control'],fold,str(ROOT/'dataset/5fold_folded'),str(temp/'control'),_stable_fold_seed(42,fold))
            c['data']['cache_dir']=str(temp/fold/'cache')
            c['data']['cache_graphs']=False
            train,val,test=create_datasets(c)
            seqs=[set(ds.data['seq']) for ds in (train,val,test)]
            assert not (seqs[0]&seqs[1] or seqs[0]&seqs[2] or seqs[1]&seqs[2])
            proof={'train_rows':len(train),'val_rows':len(val),'train_sequences':len(seqs[0]),
                   'val_sequences':len(seqs[1]),'train_sha256':file_sha(train.csv_path),
                   'val_sha256':file_sha(val.csv_path),'source_sha256':val.row_validation_provenance['source_sha256']}
            assert proof==prior['folds'][fold]
            assert 'four_state_targets' not in val[0] and 'four_state_targets' not in test[0]
            report['folds'][fold]=proof
            if fold==FOLDS[0]: first=(c,train)
        c,train=first
        batch=GraphDataLoader._default_collate_fn(None,[train[i] for i in range(16)])
        batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
        seed()
        control=GraphTransformer(build_model_config(c)).cuda()
        rng=torch.get_rng_state().clone()
        candidate_cfg=copy.deepcopy(c)
        candidate_cfg['model']['use_charge_partition_features']=True
        seed()
        candidate=GraphTransformer(build_model_config(candidate_cfg)).cuda()
        assert torch.equal(rng,torch.get_rng_state())
        assert all(torch.equal(v,candidate.state_dict()[k]) for k,v in control.state_dict().items())
        # Cross-mode / corrupt-marker loads must reject BEFORE copying any tensor,
        # including strict=False, which otherwise tolerates missing readout weights.
        for destination,source in ((control,candidate),(candidate,control)):
            before={k:v.clone() for k,v in destination.state_dict().items()}
            try: destination.load_state_dict(source.state_dict(),strict=False)
            except RuntimeError: pass
            else: raise AssertionError('Cross-mode checkpoint accepted')
            assert all(torch.equal(v,before[k]) for k,v in destination.state_dict().items())
        corrupt={k:v.clone() for k,v in candidate.state_dict().items()}
        corrupt['charge_partition.feature_version'].fill_(2)
        try: candidate.load_state_dict(corrupt,strict=False)
        except RuntimeError: pass
        else: raise AssertionError('Corrupt feature version accepted')
        candidate.load_state_dict(candidate.state_dict())
        # CUDA scatter/index_add aggregation uses atomic reductions in the
        # production nondeterministic setting. Compare algebraic initialization
        # on pod CPU, then verify the exact zero correction WITHIN an AMP forward.
        control.cpu(); candidate.cpu()
        cpu_batch={k:v.cpu() if torch.is_tensor(v) else v for k,v in batch.items()}
        control.eval(); candidate.eval()
        with torch.no_grad():
            assert torch.equal(control(cpu_batch),candidate(cpu_batch))
        control.train(); candidate.train()
        with torch.no_grad():
            seed(); p0=control(cpu_batch)
            seed(); p1=candidate(cpu_batch)
            assert torch.equal(p0,p1)
        control.cuda(); candidate.cuda()
        control.eval(); candidate.eval()
        captured={}
        head_hook=candidate.bond_head.register_forward_hook(lambda module,args,out:captured.update(base=out.float()))
        delta_hook=candidate.charge_partition.register_forward_hook(lambda module,args,out:captured.update(delta=out))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            p0=control(batch)
            p0_repeat=control(batch)
            p1=candidate(batch)
        head_hook.remove(); delta_hook.remove()
        assert torch.equal(captured['delta'],torch.zeros_like(captured['delta']))
        assert torch.equal(p1[batch['label_mask'].bool()],union_logit(captured['base']))
        assert torch.isfinite(p0).all() and torch.isfinite(p1).all()
        report['initial_amp_diagnostic']={
            'paired_max_abs_logit_difference':float((p0-p1).abs().max()),
            'same_control_repeat_max_abs_logit_difference':float((p0-p0_repeat).abs().max()),
            'feature_correction_max_abs':float(captured['delta'].abs().max()),
            'shared_parameters_exact':True,'pod_cpu_eval_and_train_predictions_exact':True,
            'candidate_amp_union_exactly_matches_its_captured_base_logits':True,
            'production_deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}
        control.train(); candidate.train()
        report['checks'] += ['shared initialization/RNG and initial pod CPU eval/train predictions match exactly',
                             'zero residual and unchanged union logits verified inside a production AMP forward; CUDA reduction variability recorded',
                             'cross-mode and corrupted checkpoint markers rejected before copying weights']
        for arm,model in (('control',control),('features',candidate)):
            cfg=copy.deepcopy(c if arm=='control' else candidate_cfg)
            cfg['debug']['diagnostic_dir']=str(temp/arm/'diagnostics')
            optimizer=create_optimizer(model,cfg)
            trainer=Trainer(model,optimizer,BinaryBondLoss(cfg['loss']),create_scheduler(optimizer,cfg),torch.device('cuda:0'),cfg,train_script.logger)
            updates=2 if arm=='features' else 1
            for step in range(updates):
                seed()
                loss,stats=trainer._forward_pass(batch)
                logits=model.last_four_state_logits
                assert torch.equal(loss,four_state_cross_entropy(logits,batch['four_state_targets'],batch['label_mask'],trainer._masked_row_weights(batch),1.))
                if arm=='features':
                    tokens=model.node_encoder._encode_sequences(batch['sequences'],torch.device('cuda:0'))
                    with torch.autocast('cuda',dtype=torch.float16):
                        correction=model.charge_partition(tokens,batch['charges'],batch['label_mask'].bool())
                    assert torch.equal(correction[:,0],torch.zeros_like(correction[:,0]))
                    assert (correction[:,1:].abs().sum()>0) == (step>0)
                trainer._backward_pass(loss)
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                if arm=='features':
                    assert model.charge_partition.mlp[2].weight.grad.abs().sum()>0
                    if step: assert model.charge_partition.mlp[0].weight.grad.abs().sum()>0
            report['arms'][arm]={'features':ARMS[arm],'conditional_weight':trainer.four_state_conditional_weight,
                                 'parameters':sum(p.numel() for p in model.parameters()),'amp':trainer.use_amp,
                                 'optimizer_updates':updates,'valid_bonds':stats['valid_bond_count']}
    report['configs']={a:file_sha(ROOT/f'graph_transform/config/pre_synthesis_four_state_charge_{a}.yaml') for a in ARMS}
    report['checks'] += ['production CE alpha=1 and AMP optimizer updates finite; new output then hidden weights receive gradients']
    report['status']='passed'
    Path(output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    os.chdir(ROOT)
    run(p.parse_args().output)
