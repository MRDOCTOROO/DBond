"""One candidate, five fixed seed42 splits; pod-only preflight and event worker."""
import argparse
import copy
import csv
import datetime
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import traceback
import types

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'graph_transform'), str(REPO / 'graph_transform/scripts')]
PY = '/mnt/pvc/graphtrans/DBond/.venv/bin/python'
OPS = Path('/mnt/pvc/graphtrans/dbond_neighbor_qkv8_20261027_ops')
BASE_OPS = Path('/mnt/pvc/graphtrans/four_state_parallel_global_replication_ops')
BASE_REV = '111c23313da9ae11a7ee33f8357be0a1876086ec'
CONFIG = REPO / 'graph_transform/config/pre_synthesis_neighbor_qkv8_seed42.yaml'
FOLDS = ('1222', '2252', '3514', '6072', '9075')
from data.row_validation import file_sha
from four_state_ops import dump, idle
from train_5fold import _stable_fold_seed, make_fold_config
import yaml


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def revision():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()


def seed(value=42):
    import numpy as np
    import torch
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def sources():
    return {str(p.relative_to(REPO)): file_sha(p) for p in (REPO / 'graph_transform').rglob('*.py')}


def legacy_class():
    text = subprocess.check_output(['git', 'show', f'{BASE_REV}:graph_transform/models/graph_transformer.py'],
                                   cwd=REPO, text=True)
    module = types.ModuleType('models._frozen_neighbor_baseline')
    module.__package__ = 'models'
    exec(compile(text, '<frozen-baseline-graph-transformer>', 'exec'), module.__dict__)
    return module.GraphTransformer


def formula_check(config):
    """Independent per-target/head loop, including gradients, not scatter reuse."""
    import math
    import torch
    import torch.nn.functional as F
    from models.neighbor_qkv_attention import NeighborQKVAttention
    layer = NeighborQKVAttention(config).double().eval()
    reference = copy.deepcopy(layer)
    x = torch.randn(8, 256, dtype=torch.float64, requires_grad=True)
    edge = torch.tensor([[0, 1, 2, 4, 5, 6], [2, 2, 0, 6, 6, 4]])
    attr = torch.randn(6, 256, dtype=torch.float64, requires_grad=True)
    other_x = x.detach().clone().requires_grad_()
    other_e = attr.detach().clone().requires_grad_()
    observed, attention = layer(x, edge, attr, True)
    q = F.linear(other_x, reference.q_proj.weight).reshape(8, 8, 32)
    k = F.linear(other_x, reference.k_proj.weight).reshape(8, 8, 32)
    v = F.linear(other_x, reference.v_proj.weight).reshape(8, 8, 32)
    bias = F.linear(other_e, reference.edge_attention_bias.weight, reference.edge_attention_bias.bias)
    gates = F.linear(other_e, reference.edge_message_gate.weight, reference.edge_message_gate.bias).sigmoid().reshape(6, 8, 32)
    rows = []; weights = torch.zeros_like(attention)
    for center in range(8):
        ids = (edge[1] == center).nonzero().flatten()
        heads = []
        for head in range(8):
            if ids.numel():
                logits = torch.stack([(q[center, head] * k[edge[0, j], head]).sum() / math.sqrt(32)
                                      + bias[j, head] for j in ids])
                alpha = logits.softmax(0)
                weights[ids, head] = alpha
                heads.append(sum(alpha[t] * gates[j, head] * v[edge[0, j], head] for t, j in enumerate(ids)))
            else:
                heads.append(v.new_zeros(32))
        update = F.linear(torch.cat(heads), reference.out_proj.weight, reference.out_proj.bias) if ids.numel() else other_x.new_zeros(256)
        rows.append(F.layer_norm(other_x[center] + update, (256,), reference.layer_norm.weight,
                                 reference.layer_norm.bias, reference.layer_norm.eps))
    expected = torch.stack(rows)
    torch.testing.assert_close(observed, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(attention, weights, atol=1e-12, rtol=1e-12)
    probe = torch.randn_like(observed)
    (observed * probe).sum().backward(); (expected * probe).sum().backward()
    torch.testing.assert_close(x.grad, other_x.grad, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(attr.grad, other_e.grad, atol=1e-9, rtol=1e-9)
    for (name, p), (other, r) in zip(layer.named_parameters(), reference.named_parameters()):
        require(name == other and p.grad is not None and r.grad is not None, 'Missing formula gradient')
        torch.testing.assert_close(p.grad, r.grad, atol=1e-9, rtol=1e-9)
    perm = torch.tensor([5, 2, 0, 3, 1, 4])
    torch.testing.assert_close(layer(x, edge[:, perm], attr[perm]), observed, atol=1e-10, rtol=1e-10)
    changed = x.detach().clone(); changed[4:] += 100
    torch.testing.assert_close(layer(changed, edge, attr)[:4], observed[:4], atol=1e-10, rtol=1e-10)
    # Empty graph and isolated/empty nodes have finite residual-only outputs.
    for n in (0, 1):
        y = layer(x.detach()[:n], torch.empty(2, 0, dtype=torch.long), attr.detach()[:0])
        require(y.shape == (n, 256) and torch.isfinite(y).all(), 'Empty/isolated graph failure')
    return {'destination_query_source_key_value': True, 'neighbor_softmax_per_head': True,
            'eight_complete_input_projections_32d': True, 'loop_forward_and_all_gradients': True,
            'edge_permutation_invariant': True, 'disconnected_graph_isolation': True,
            'empty_and_isolated_nodes': True}


def preflight(gpu, output):
    import gc
    import torch
    from models.graph_transformer import GraphTransformer
    from models.utils import build_model_config
    from train_graph_model import load_config, create_datasets, create_optimizer, create_scheduler
    import train_graph_model as train_script
    from training.trainer import Trainer
    from training.loss_functions import BinaryBondLoss
    from test_four_state_width import collate, longest_indices, to_cuda, topology_proof
    train_script.logger = logging.getLogger('neighbor_qkv8_preflight')
    require(sys.platform.startswith('linux') and str(REPO).startswith('/mnt/pvc/'), 'Pod only')
    torch.set_num_threads(2); torch.cuda.set_device(gpu)
    args = argparse.Namespace(epochs=None, batch_size=None, learning_rate=None, device=None)
    runtime = load_config(str(CONFIG), args)
    base_path = REPO / 'graph_transform/config/pre_synthesis_four_state_parallel_global_replication_single.yaml'
    baseline_runtime = load_config(str(base_path), args)
    normalized = copy.deepcopy(runtime)
    for key in ('neighbor_attention_mode', 'neighbor_qkv_heads'):
        normalized['model'].pop(key)
    require(json.dumps(normalized, sort_keys=True).replace('neighbor_qkv8_seed42',
            'four_state_parallel_global_replication_single') == json.dumps(baseline_runtime, sort_keys=True),
            'Protocol differs beyond declared attention mode and artifact identities')
    require(runtime['model']['state_feature_mask'] == [True, True, False]
            and runtime['model']['env_feature_mask'] == [True, False], 'Pre-synthesis masks wrong')
    prior = read(BASE_OPS / 'smoke.json'); old = read(BASE_OPS / 'result.json')
    launch = read(BASE_OPS / 'launch.json')
    require(launch['gitrev'] == old['gitrev'] == BASE_REV and old['status'] == 'completed', 'Baseline revision/status')
    changed = {'graph_transform/models/graph_transformer.py', 'graph_transform/models/attention_v2.py',
               'graph_transform/models/utils.py',
               # Historical unrelated dirty script; never imported by training.
               # Keep that original file untouched and exclude only this script.
               'graph_transform/scripts/ensemble_baseline_preds.py'}
    for name, digest in launch['source_sha256'].items():
        if name not in changed:
            require(file_sha(REPO / name) == digest, f'Unexpected production difference: {name}')
    proof = {'status': 'running', 'gitrev': revision(), 'source_sha256': sources(),
             'config_sha256': file_sha(CONFIG), 'folds': {}, 'baseline': {}, 'checks': {},
             'baseline_result_sha256': file_sha(BASE_OPS / 'result.json'),
             'baseline_registry_sha256': file_sha(BASE_OPS / 'checkpoint_registry.json')}
    proof['checks']['independent_formula'] = formula_check(build_model_config(runtime))
    Legacy = legacy_class()
    seed(42); legacy = Legacy(build_model_config(baseline_runtime)); rng = torch.get_rng_state()
    seed(42); control = GraphTransformer(build_model_config(baseline_runtime))
    require(torch.equal(rng, torch.get_rng_state()), 'Legacy initialization RNG changed')
    require(legacy.state_dict().keys() == control.state_dict().keys()
            and all(torch.equal(v, control.state_dict()[k]) for k, v in legacy.state_dict().items()),
            'Legacy off-mode state differs')
    seed(42); candidate = GraphTransformer(build_model_config(runtime))
    require(torch.equal(rng, torch.get_rng_state()), 'Candidate changed following RNG stream')
    require(all(torch.equal(v, candidate.state_dict()[k]) for k, v in control.state_dict().items()
                if not k.startswith('gat_layers.')), 'Candidate changed shared initialization')
    candidate.load_state_dict(candidate.state_dict(), strict=True)
    for receiver, state in ((candidate, control.state_dict()), (control, candidate.state_dict())):
        try:
            receiver.load_state_dict(state, strict=False)
        except RuntimeError:
            pass
        else:
            raise ValueError('Cross-mode checkpoint load did not fail')
    proof['checks']['legacy_schema_rng_and_shared_initialization'] = True
    proof['checks']['cross_mode_checkpoint_rejection'] = True
    first = None
    for fold in FOLDS:
        actual_seed = _stable_fold_seed(42, fold)
        resolved, _, _ = make_fold_config(runtime, fold, str(REPO / 'dataset/5fold_folded'),
                                         str(OPS / 'preflight_folds'), actual_seed)
        resolved['data']['cache_graphs'] = False
        train, val, test = create_datasets(resolved)
        sets = [set(ds.data['seq']) for ds in (train, val, test)]
        require(not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), 'Sequence leakage')
        provenance = val.row_validation_provenance
        bound = old['recovery']['single'][fold]
        for path_key, hash_key in (('checkpoint','checkpoint_sha256'), ('config','config_sha256'),
                                   ('metrics_csv','metrics_sha256')):
            require(file_sha(bound[path_key]) == bound[hash_key], f'Baseline {fold} {path_key} SHA changed')
        prediction = bound['prediction_recomputation']
        require(file_sha(prediction['prediction_csv']) == prediction['prediction_sha256'], 'Baseline prediction SHA')
        row = {'train_rows': len(train), 'val_rows': len(val), 'train_sequences': len(sets[0]),
               'val_sequences': len(sets[1]), 'train_sha256': file_sha(train.csv_path),
               'val_sha256': file_sha(val.csv_path), 'source_sha256': provenance['source_sha256']}
        require(all(row[k] == prior['folds'][fold][k] for k in row), f'{fold}: train/val/source mismatch')
        require(file_sha(resolved['data']['test_csv_path']) == prediction['test_sha256'], 'Test SHA mismatch')
        target = Path(resolved['data']['four_state_targets_manifest'])
        require(file_sha(target) == launch['target_manifests'][fold], 'Target manifest SHA mismatch')
        saved = torch.load(bound['checkpoint'], map_location='cpu', weights_only=False)
        config_saved = copy.deepcopy(saved.get('config') or yaml.safe_load(Path(bound['config']).read_text()))
        restore = GraphTransformer(build_model_config(config_saved))
        restore.load_state_dict(saved['model_state_dict'], strict=True)
        require(saved['extra_state']['training_rng_protocol']['training_seed'] == actual_seed
                and saved['extra_state']['training_rng_protocol']['validation_split_seed'] == actual_seed, 'Baseline actual seeds')
        proof['baseline'][fold] = {k: bound[k] for k in ('checkpoint', 'checkpoint_sha256', 'config', 'config_sha256', 'metrics_csv', 'metrics_sha256')}
        proof['baseline'][fold]['prediction'] = prediction
        row.update(training_seed=actual_seed, split_seed=actual_seed,
                   test_sha256=prediction['test_sha256'], target_manifest_sha256=file_sha(target))
        proof['folds'][fold] = row
        del saved, restore
        if first is None:
            first = (resolved, train, val)
        else:
            del train, val, test
        gc.collect()
    resolved, train, val = first
    small = collate(train, list(range(16)))
    proof['checks']['real_graph_packing_and_effective_bond_mask'] = topology_proof(small)
    # Exact legacy off-mode forward, four-state loss, gradients and AdamW update.
    from training.four_state_loss import four_state_cross_entropy
    snapshots = []
    for model in (legacy, control):
        seed(123); model.train()
        optim = create_optimizer(model, resolved)
        logits = model(small)
        loss = four_state_cross_entropy(model.last_four_state_logits, small['four_state_targets'],
                                       small['label_mask'], small.get('weights'))
        loss.backward(); optim.step()
        snapshots.append((logits.detach(), {k: v.detach().clone() for k, v in model.state_dict().items()}, torch.get_rng_state()))
    require(torch.equal(snapshots[0][0], snapshots[1][0])
            and torch.equal(snapshots[0][2], snapshots[1][2])
            and all(torch.equal(v, snapshots[1][1][k]) for k, v in snapshots[0][1].items()), 'Legacy one-step equivalence')
    proof['checks']['legacy_real_task_one_step_equivalence'] = True
    del legacy, control, snapshots, optim, candidate
    seed(_stable_fold_seed(42, FOLDS[0]))
    model = GraphTransformer(build_model_config(resolved)).cuda()
    optimizer = create_optimizer(model, resolved)
    trainer = Trainer(model, optimizer, BinaryBondLoss(resolved['loss']), create_scheduler(optimizer, resolved),
                      torch.device(f'cuda:{gpu}'), resolved, train_script.logger)
    cuda_small = to_cuda(small)
    gradients = []
    for _ in range(2):
        model.train(); loss, stats = trainer._forward_pass(cuda_small)
        require(torch.isfinite(loss) and stats['valid_bond_count'] > 0, 'Invalid task loss')
        trainer._backward_pass(loss)
        entry = {n: float(p.grad.float().norm()) for n,p in model.named_parameters()
                 if n.startswith('gat_layers.') and any(part in n for part in ('q_proj','k_proj','v_proj','out_proj')) and p.grad is not None}
        require(len(entry) == 10 and all(v > 0 for v in entry.values()), 'Missing real-task QKV gradient')
        gradients.append(entry)
    proof['checks']['actual_four_state_task_qkv_gradients'] = gradients
    torch.cuda.reset_peak_memory_stats()
    full = to_cuda(collate(train, longest_indices(train, 128)))
    model.train(); loss, _ = trainer._forward_pass(full); trainer._backward_pass(loss)
    proof['train128_peak_allocated_mb'] = torch.cuda.max_memory_allocated()/2**20
    del full
    torch.cuda.reset_peak_memory_stats()
    full_val = to_cuda(collate(val, longest_indices(val, 512)))
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        output_val = model(full_val)
    require(torch.isfinite(output_val[full_val['label_mask'].bool()]).all(), 'Nonfinite validation batch')
    proof['val512_peak_allocated_mb'] = torch.cuda.max_memory_allocated()/2**20
    proof['parameter_count'] = sum(p.numel() for p in model.parameters())
    proof['attention_parameter_count'] = sum(p.numel() for p in model.gat_layers.parameters())
    proof['baseline_parameter_count'] = old['resources']['single']['parameter_count']
    proof['checks']['candidate_strict_cpu_roundtrip'] = True
    cpu = GraphTransformer(build_model_config(resolved)); cpu.load_state_dict({k:v.cpu() for k,v in model.state_dict().items()}, strict=True)
    proof['status'] = 'passed'
    proof['baseline_reuse'] = 'passed_five_checkpoint_sha_strict_cpu_data_protocol_and_legacy_one_step_equivalence'
    dump(output, proof)
    print(json.dumps({'status':'passed', 'checks':list(proof['checks']), 'parameter_count':proof['parameter_count'],
                      'train128_peak_allocated_mb':proof['train128_peak_allocated_mb'], 'val512_peak_allocated_mb':proof['val512_peak_allocated_mb']}))


def smoke():
    require(not (OPS/'smoke.json').exists() and not (OPS/'launch.json').exists(), 'Existing successful smoke/launch')
    ids, _, _ = idle()
    attempt=1
    while (OPS/f'preflight.attempt_{attempt:02d}.log').exists():
        attempt+=1
    with (OPS/f'preflight.attempt_{attempt:02d}.log').open('x') as log:
        subprocess.run([PY, __file__, 'preflight', '--gpu', str(ids[0]), '--output', str(OPS/'smoke.json')],
                       cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({'status':'passed','smoke_sha256':file_sha(OPS/'smoke.json')}))


def launch():
    require(not (OPS/'launch.json').exists() and not (OPS/'launch.claim').exists(), 'Duplicate launch forbidden')
    proof = read(OPS/'smoke.json')
    require(proof['status']=='passed' and proof['gitrev']==revision() and proof['config_sha256']==file_sha(CONFIG)
            and proof['source_sha256']==sources(), 'Stale smoke/source')
    require(not subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=REPO,text=True).strip(), 'Dirty frozen source')
    ids, rows, processes = idle()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
    root = Path('/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/neighbor_qkv8_batch')/stamp
    with (OPS/'launch.claim').open('x') as stream:
        stream.write(str(root))
    info = {'root':str(root),'gitrev':revision(),'source_snapshot':str(REPO),
            'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpus':ids,
            'gpu_preflight':rows,'compute_processes':processes, 'config':str(CONFIG),'config_sha256':file_sha(CONFIG),
            'smoke_sha256':file_sha(OPS/'smoke.json'),'source_sha256':sources(),
            'fold_seeds':{f:{'training':_stable_fold_seed(42,f),'split':_stable_fold_seed(42,f)} for f in FOLDS},
            'new_trainings':5, 'baseline_reuse':proof['baseline_reuse'], 'baseline_ops':str(BASE_OPS),
            'scope':'Replace two legacy GAT4 blocks with two neighbor QKV8 blocks; retain edge bias/value gate and fixed topology; no extra branches, seeds or search.'}
    dump(OPS/'launch.json',info)
    env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    with (OPS/'worker.log').open('x') as log:
        process = subprocess.Popen([PY,__file__,'worker'],cwd=REPO,env=env,stdin=subprocess.DEVNULL,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    info['worker_pid']=process.pid
    dump(OPS/'launch.json',info)
    print(json.dumps({k:v for k,v in info.items() if k!='source_sha256'}))


def worker():
    info=read(OPS/'launch.json')
    try:
        with (OPS/'training.log').open('x') as log:
            subprocess.run([PY,'-u',str(REPO/'graph_transform/scripts/run_architecture_batch.py'),
                '--configs',str(CONFIG),'--fold_data_dir',str(REPO/'dataset/5fold_folded'),
                '--gpus',','.join(map(str,info['gpus'])),'--run_root',info['root']],
                cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
        dump(OPS/'terminal.json',{'status':'completed','root':info['root'],'gitrev':info['gitrev'],
              'audit_pending':True,'manifest':str(Path(info['root'])/'manifest.json'),
              'scope':'Training complete; do not claim performance until independent terminal audit.'})
    except BaseException:
        dump(OPS/'terminal.json',{'status':'failed','root':info['root'],'gitrev':info['gitrev'],
              'error':traceback.format_exc(),'no_automatic_retry':True})
        raise


if __name__=='__main__':
    os.chdir(REPO); OPS.mkdir(exist_ok=True)
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=('preflight','smoke','launch','worker'))
    parser.add_argument('--gpu',type=int,default=0); parser.add_argument('--output')
    args=parser.parse_args()
    if args.action=='preflight':
        try:
            preflight(args.gpu,args.output)
        except BaseException:
            dump(OPS/'preflight.failed.json',{'status':'failed','error':traceback.format_exc()})
            raise
    else:
        globals()[args.action]()
