"""Pod-only factorial composition checks using two already frozen mechanisms."""
import copy,importlib.util,json,logging,sys
from pathlib import Path
import torch
import adaptive_research_ops as core
import adaptive_regularization_wave2 as reuse
from adaptive_baseline_audit import dump,sha
W8=Path('/mnt/pvc/graphtrans/dbond_adaptive_nce_head_wave8_20261104_ops')
W10=Path('/mnt/pvc/graphtrans/dbond_adaptive_cardinality_wave10_20261104_ops')
NUM=W8/'wave8_fp64_audit.py'
NUM_SHA='57858db1b648f5eddfcc4417659ce0e1d8031014da3e4f813e37bb63ba5afe04'
LEGACY=Path('/mnt/pvc/graphtrans/dbond_adaptive_research_source_addf42c')

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def references():
 assert sha(W8/'final_review_fp64_v1/result.json')=='1abfdfbefd7c45e2e121995007015ae08d569f72bf67848abf55e1b1975edd6d'
 assert sha(W8/'independent_audit/result.json')=='2313eacffb2366425c6de3945822216d8a2cdffd700b1337299995257ddb325c'
 assert sha(W8/'independent_audit_fp64_v1/result.json')=='e8ce28a1a0febdb3297feb76e6b6ec57a93a300682a27333726a3f65d4eb8768'
 assert sha(W8/'independent_audit/state.json')=='b6fed560746c72e2b8cc0a0ad8ea116a3644c06d52fea34e0e6136b1e5de97cd'
 assert sha(W10/'independent_audit/result.json')=='16948af0239df1a4d748fb5e7392360d263b6bb565373710503316129c4357b6'
 assert core.read(W10/'independent_audit/state.json')['status']=='passed' and sha(NUM)==NUM_SHA
 assert sha(W10/'wave10_terminal_review.py')=='4eb58c5fcc78b94e3f0837ed9814b01992514869d7d6e35eb0ca6178386741c2'
 return dict(nce_result_sha256=sha(W8/'independent_audit/result.json'),nce_final_sha256=sha(W8/'final_review_fp64_v1/result.json'),nce_fp64_sha256=sha(W8/'independent_audit_fp64_v1/result.json'),old_nce_failed_state_sha256=sha(W8/'independent_audit/state.json'),count_result_sha256=sha(W10/'independent_audit/result.json'),numeric_helper_sha256=sha(NUM))

def smoke(repo,ops,patches,gpu):
 from models.graph_transformer import GraphTransformer
 from models.utils import build_model_config
 from train_graph_model import create_datasets
 import train_graph_model as ts
 from training.trainer import Trainer
 from training.loss_functions import BinaryBondLoss
 from training.cardinality_loss import cardinality_cross_entropy
 from test_four_state_width import collate,to_cuda,longest_indices
 assert gpu in core.free_gpus(),'Preflight GPU must be idle before CUDA context'
 torch.set_num_threads(1);torch.cuda.set_device(gpu);ops.mkdir(exist_ok=True);assert not (ops/'smoke.json').exists()
 ts.logger=logging.getLogger('nce_count_preflight');bound=reuse.previous_binding();factorial=references()
 ncel=core.read(W8/'launch.json');countl=core.read(W10/'launch.json');ncefiles={'graph_transform/models/graph_transformer.py','graph_transform/models/utils.py','graph_transform/models/nce_state_residual.py'}
 for n,h in countl['source_sha256'].items():
  if n.startswith(('graph_transform/models/','graph_transform/training/','graph_transform/data/','graph_transform/evaluation/')) or n=='graph_transform/scripts/train_graph_model.py':assert sha(repo/n)==(ncel['source_sha256'][n] if n in ncefiles else h),n
 for n in ncefiles:assert sha(repo/n)==ncel['source_sha256'][n],n
 oldgraph=load('models.original_graph',LEGACY/'graph_transform/models/graph_transformer.py')
 oldtrainer=load('training.original_trainer',LEGACY/'graph_transform/training/trainer.py')
 num=load('fixed_fp64_formulas',NUM);num.fixtures(torch,repo)
 report=dict(status='running',gitrev=core.rev(),source_sha256=core.source_hashes(),folds={},checks={},parameters={},factorial_reference=factorial)
 refproof={}
 def trainer(m,cfg,cls=Trainer):return cls(model=m,optimizer=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4),criterion=BinaryBondLoss(cfg['loss']),device=next(m.parameters()).device,config=cfg)
 for fold in core.FOLDS:
  cfg,_=core.fold_config('nce_count',fold,ops/'preflight');cfg=core.load_runtime(cfg);cfg['data']['cache_graphs']=False
  train,val,test=create_datasets(cfg);assert test is None and val.sample_weights is None and not set(train.data.seq)&set(val.data.seq)
  oldproof=core.read(W10/'cardinality_smoke.json')['folds'][fold]
  assert sha(train.csv_path)==oldproof['train_sha256'] and sha(val.csv_path)==oldproof['val_sha256']
  assert train.four_state_provenance['cardinality']==oldproof['histogram_provenance'] and 'cardinality_histogram' not in val[0]
  configs={}
  for arm,mode,weight in [('full','none',0.),('nce_only','stratified4',0.),('count_only','none',.1),('nce_count','stratified4',.1)]:
   cc=copy.deepcopy(cfg);cc['model']['nce_head_mode']=mode;cc['loss']['cardinality_weight']=weight;configs[arm]=cc
  refproof[fold]={}
  for arm,oldops,oldarm in [('nce_only',W8,'nce_head'),('count_only',W10,'cardinality_01')]:
   oldroot=Path(core.read(oldops/'launch.json')['root']);ap=oldroot/oldarm/f'fold_{fold}/trial_audit.json';a=core.read(ap);assert a['status']=='passed'
   recs={}
   for p,r in a['policies'].items():
    assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['prediction'])==r['prediction_sha256']
    saved=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);m=GraphTransformer(build_model_config(configs[arm]));m.load_state_dict(saved['model_state_dict'],strict=True)
    assert saved['extra_state']['training_rng_protocol']==r['actual_seed'] and r['actual_seed']['training_seed']==cfg['experiment']['seed']
    assert sha(saved['extra_state']['row_validation_provenance']['split_val_csv'])==sha(val.csv_path)
    recs[p]=dict(checkpoint_sha256=r['checkpoint_sha256'],prediction_sha256=r['prediction_sha256'],strict_cpu=True)
   refproof[fold][arm]=dict(root=str(oldroot),original_arm=oldarm,trial_receipt_sha256=sha(ap),policies=recs)
  report['folds'][fold]=dict(seed=cfg['experiment']['seed'],train_sha256=sha(train.csv_path),val_sha256=sha(val.csv_path),four_state_manifest_sha256=sha(cfg['data']['four_state_targets_manifest']),histogram_provenance=train.four_state_provenance['cardinality'])
  if fold!=core.FOLDS[0]:continue
  models={};rng=[]
  for arm in configs:
   core.seed(cfg['experiment']['seed']);models[arm]=GraphTransformer(build_model_config(configs[arm]));rng.append(torch.get_rng_state().clone())
  core.seed(cfg['experiment']['seed']);original=oldgraph.GraphTransformer(build_model_config(configs['full']));rng.append(torch.get_rng_state().clone())
  assert all(torch.equal(rng[0],v) for v in rng[1:])
  for k,v in original.state_dict().items():assert all(torch.equal(v,m.state_dict()[k]) for m in models.values()),k
  report['parameters']={a:sum(p.numel() for p in m.parameters()) for a,m in models.items()};assert report['parameters']==dict(full=2002284,nce_only=2006396,count_only=2002284,nce_count=2006396)
  for target,state in [(models['full'],models['nce_only'].state_dict()),(models['nce_count'],models['count_only'].state_dict())]:
   before={k:v.clone() for k,v in target.state_dict().items()}
   try:target.load_state_dict(state,strict=False)
   except RuntimeError:pass
   else:raise AssertionError('Cross-mode checkpoint accepted')
   assert all(torch.equal(v,target.state_dict()[k]) for k,v in before.items())
  bt=collate(train,list(range(16)));losses={};grads={};updates={};logits={}
  for arm,m in dict(legacy=original,**models).items():
   cc=configs['full'] if arm=='legacy' else configs[arm];t=trainer(m,cc,oldtrainer.Trainer if arm=='legacy' else Trainer);core.seed(60123);loss,_=t._forward_pass(bt);loss.backward()
   losses[arm]=loss.detach();logits[arm]=m.last_four_state_logits.detach().clone();grads[arm]={k:p.grad.clone() if p.grad is not None else None for k,p in m.named_parameters()};t.optimizer.step();updates[arm]={k:v.clone() for k,v in m.state_dict().items()}
  for a,b in [('legacy','full'),('full','nce_only'),('count_only','nce_count')]:
   assert torch.equal(losses[a],losses[b]) and torch.equal(logits[a],logits[b])
   for k,g in grads[a].items():assert ((g is None and grads[b][k] is None) or torch.equal(g,grads[b][k])) and torch.equal(updates[a][k],updates[b][k]),k
  assert any(g is not None and g.abs().sum()>0 for k,g in grads['nce_count'].items() if k.startswith('nce_state_residual.'))
  # Fresh candidate: current fixed count loss value/gradient against independent FP64 probability DP.
  core.seed(cfg['experiment']['seed']);candidate=GraphTransformer(build_model_config(cfg));candidate.train();t=trainer(candidate,cfg);loss,_=t._forward_pass(bt);x=candidate.last_four_state_logits
  ref=load('fixed_count_formulas',W10/'wave10_terminal_review.py').reference(x,bt['label_mask'].bool(),bt['cardinality_histogram']);actual=cardinality_cross_entropy(x,bt['label_mask'],bt['cardinality_histogram'])
  ga=torch.autograd.grad(actual,x,retain_graph=True)[0];gb=torch.autograd.grad(ref,x,retain_graph=True)[0];assert abs(float(actual-ref))<3e-6 and float((ga-gb).abs().max())<3e-6
  # Nonzero all-NCE fixture and learned full-route formula use the fixed FP64 engineering budget, never an absolute gate relaxed after results.
  candidate.nce_state_residual.weight.data.copy_(torch.arange(4096).reshape(4,4,256)*1e-6);candidate.nce_state_residual.bias.data.copy_(torch.arange(16).reshape(4,4)*.01)
  candidate.eval();seen,out,z,index,calls=num.exact_route(candidate,bt,torch);num.numerical_rows(seen['hidden'],seen['base'],seen['delta'],out,z,index,candidate.nce_state_residual.weight.detach(),candidate.nce_state_residual.bias.detach(),torch)
  for nce in (torch.tensor([25.]),torch.tensor([float('nan')])):
   try:candidate.nce_state_residual(torch.zeros(1,256),nce,torch.tensor([0]))
   except ValueError:pass
   else:raise AssertionError('Unknown NCE accepted')
  del models,original,candidate,t;torch.cuda.empty_cache()
  core.seed(cfg['experiment']['seed']);candidate=GraphTransformer(build_model_config(cfg)).cuda().train();t=trainer(candidate,cfg);big=to_cuda(collate(train,longest_indices(train,128)))
  for _ in range(2):
   t.optimizer.zero_grad();loss,_=t._forward_pass(big);assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in candidate.parameters());t.optimizer.step()
  candidate.eval()
  with torch.no_grad():assert torch.isfinite(candidate(to_cuda(collate(val,longest_indices(val,512))))).all()
  cpu=GraphTransformer(build_model_config(cfg));cpu.load_state_dict({k:v.cpu() for k,v in candidate.state_dict().items()},strict=True)
  report['checks']=dict(disabled_original_exact=True,zero_nce_same_count_loss_gradients_first_update=True,initial_rng_state_exact=True,all_reference_cpu_strict=True,count_formula_value_gradient=True,nce_route_fp64_components=True,cross_mode_no_partial_copy=True,unknown_nce_rejected=True,longest_train128_val512=True,two_step_finite=True,cpu_strict_roundtrip=True)
  del candidate,cpu,t,big;torch.cuda.empty_cache()
 report['status']='passed';dump(ops/'smoke.json',report)
 dump(ops/'cardinality_smoke.json',dict(status='passed',folds={f:dict(histogram_provenance=report['folds'][f]['histogram_provenance'],train_sha256=report['folds'][f]['train_sha256'],val_sha256=report['folds'][f]['val_sha256']) for f in core.FOLDS},coefficient=.1,architecture_smoke_sha256=sha(ops/'smoke.json'),reused_target_audit_sha256=factorial['count_result_sha256']))
 dump(ops/'factorial_reference_receipt.json',dict(status='passed',bindings=factorial,folds=refproof,new_reference_trainings=0))
 dump(ops/'reference_reuse_receipt.json',dict(status='passed',**bound,new_reference_trainings=0,patches=patches,cardinality_smoke_sha256=sha(ops/'cardinality_smoke.json'),factorial_reference_sha256=sha(ops/'factorial_reference_receipt.json')))
 print(json.dumps(dict(status='passed',checks=report['checks'],parameters=report['parameters'])))
