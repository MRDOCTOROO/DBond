"""One-factor late NCE head trial. Pod-only, grouped inner validation only."""
import argparse,copy,datetime,importlib.util,json,logging,os,subprocess,sys,time,traceback
from pathlib import Path
import yaml
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'graph_transform/scripts'),str(REPO/'graph_transform')]
import adaptive_research_ops as core
import adaptive_regularization_wave2 as reuse
from adaptive_baseline_audit import dump,sha
OPS=Path('/mnt/pvc/graphtrans/dbond_adaptive_nce_head_wave8_20261104_ops')
PATCHES={'full':{},'nce_head':{'model.nce_head_mode':'stratified4'}}
core.OPS=OPS;core.ARMS=PATCHES
LEGACY=Path('/mnt/pvc/graphtrans/dbond_adaptive_research_source_addf42c')
PRIOR_ROOT=reuse.PRIOR_ROOT

def smoke(gpu):
 import torch,numpy as np
 from models.graph_transformer import GraphTransformer
 from models.utils import build_model_config
 from train_graph_model import create_datasets
 import train_graph_model as ts
 from test_four_state_width import collate,to_cuda,longest_indices,topology_proof
 from training.trainer import Trainer
 from training.loss_functions import BinaryBondLoss
 ts.logger=logging.getLogger('nce_head_preflight')
 torch.set_num_threads(1);torch.cuda.set_device(gpu)
 core.require(gpu in core.free_gpus(),'Preflight requires an idle GPU')
 OPS.mkdir(exist_ok=True);core.require(not(OPS/'smoke.json').exists(),'Smoke already committed')
 bound=reuse.previous_binding()
 allowed={'graph_transform/models/graph_transformer.py','graph_transform/models/utils.py'}
 prior=core.read(reuse.PREV/'launch.json')['source_sha256']
 for name,h in prior.items():
  if name not in allowed:core.require(sha(REPO/name)==h,'Undeclared source change '+name)
 spec=importlib.util.spec_from_file_location('models.frozen_wave1_graph',LEGACY/'graph_transform/models/graph_transformer.py')
 oldmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldmod)
 report=dict(status='running',gitrev=core.rev(),source_sha256=core.source_hashes(),folds={},checks={},parameters={})
 def trainer(m,cfg):return Trainer(model=m,optimizer=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4),criterion=BinaryBondLoss(cfg['loss']),device=next(m.parameters()).device,config=cfg)
 for fold in core.FOLDS:
  cfg,_=core.fold_config('nce_head',fold,OPS/'preflight');cfg=core.load_runtime(cfg);cfg['data']['cache_graphs']=False
  train,val,test=create_datasets(cfg);core.require(test is None,'Test must be deferred')
  ss=[set(d.data.seq) for d in (train,val)];core.require(not ss[0]&ss[1],'Sequence leakage')
  oldproof=core.read(reuse.PREV/'smoke.json')['folds'][fold]
  core.require(sha(train.csv_path)==oldproof['train_sha256'] and sha(val.csv_path)==oldproof['val_sha256'],'Reference split changed')
  ref=core.read(PRIOR_ROOT/'full'/f'fold_{fold}/trial_audit.json')
  no=copy.deepcopy(cfg);no['model']['nce_head_mode']='none'
  for rec in ref['policies'].values():
   saved=torch.load(rec['checkpoint'],map_location='cpu',weights_only=False)
   m=GraphTransformer(build_model_config(no));m.load_state_dict(saved['model_state_dict'],strict=True)
   core.require(sha(rec['checkpoint'])==rec['checkpoint_sha256'],'Reference checkpoint changed')
  core.require(cfg['model']['state_feature_mask']==[True,True,False] and cfg['model']['env_feature_mask']==[True,False],'Masks changed')
  report['folds'][fold]=dict(seed=cfg['experiment']['seed'],train_sha256=sha(train.csv_path),val_sha256=sha(val.csv_path),four_state_manifest_sha256=sha(cfg['data']['four_state_targets_manifest']),reference_strict_cpu=True,train_sequences=len(ss[0]),val_sequences=len(ss[1]))
  if fold!=core.FOLDS[0]:continue
  models=[];rngs=[]
  for cls,cc in [(oldmod.GraphTransformer,no),(GraphTransformer,no),(GraphTransformer,cfg)]:
   core.seed(cfg['experiment']['seed']);mm=cls(build_model_config(cc));models.append(mm);rngs.append(torch.get_rng_state().clone())
  old,disabled,candidate=models
  core.require(torch.equal(rngs[0],rngs[1]) and torch.equal(rngs[0],rngs[2]),'Initialization RNG changed')
  for key,value in old.state_dict().items():
   core.require(torch.equal(value,disabled.state_dict()[key]) and torch.equal(value,candidate.state_dict()[key]),'Shared initial tensor changed '+key)
  report['parameters']={n:sum(p.numel() for p in m.parameters()) for n,m in zip(['legacy','disabled','candidate'],models)}
  core.require(report['parameters']['candidate']-report['parameters']['legacy']==4112,'Unexpected capacity')
  # Reject either cross-mode restore before copying any shared parameters.
  for target,state in [(disabled,candidate.state_dict()),(candidate,disabled.state_dict())]:
   before={k:v.clone() for k,v in target.state_dict().items()}
   try:target.load_state_dict(state,strict=False)
   except RuntimeError:pass
   else:raise AssertionError('Cross-mode checkpoint accepted')
   core.require(all(torch.equal(v,target.state_dict()[k]) for k,v in before.items()),'Rejected restore partially copied')
  bt=collate(train,list(range(16)));topology_proof(bt)
  losses=[];states=[];gradients=[];after=[]
  for i,m in enumerate(models):
   m.train();t=trainer(m,no if i<2 else cfg);core.seed(60123)
   loss,_=t._forward_pass(bt);loss.backward()
   losses.append(loss.detach());states.append(m.last_four_state_logits.detach().clone())
   gradients.append({k:p.grad.clone() if p.grad is not None else None for k,p in m.named_parameters()})
   t.optimizer.step();after.append({k:v.clone() for k,v in m.state_dict().items()})
  core.require(all(torch.equal(losses[0],x) for x in losses[1:]),'Zero/disabled initial loss differs')
  core.require(all(torch.equal(states[0],x) for x in states[1:]),'Zero/disabled four-state logits differ')
  for key,g in gradients[0].items():
   for other in gradients[1:]:core.require((g is None and other[key] is None) or (g is not None and other[key] is not None and torch.equal(g,other[key])),'Shared gradient differs '+key)
   core.require(torch.equal(after[0][key],after[1][key]) and torch.equal(after[0][key],after[2][key]),'First shared AdamW update differs '+key)
  core.require(any(g is not None and g.abs().sum()>0 for k,g in gradients[2].items() if k.startswith('nce_state_residual.')),'No new adapter gradient')
  # Nonzero independent per-sample routing fixture, all four NCEs.
  adapter=candidate.nce_state_residual
  with torch.no_grad():adapter.weight.copy_(torch.arange(adapter.weight.numel()).reshape_as(adapter.weight)*1e-6);adapter.bias.copy_(torch.arange(16).reshape(4,4)*.01)
  h=torch.linspace(-1,1,8*256).reshape(8,256);nce=torch.tensor([20.,30.,40.,50.]);route=torch.tensor([0,1,2,3,3,2,1,0])
  got=adapter(h,nce,route);expected=torch.stack([torch.mv(adapter.weight[int(route[i])],h[i])+adapter.bias[int(route[i])] for i in range(8)])
  core.require(torch.allclose(got,expected,rtol=1e-6,atol=1e-6),'Independent NCE affine formula')
  for invalid in (torch.tensor([25.]),torch.tensor([float('nan')])):
   try:adapter(h[:1],invalid,route[:1])
   except ValueError:pass
   else:raise AssertionError('Unknown NCE accepted')
  # Head route uses exactly the same post-dropout hidden and valid-bond order.
  seen={};handle=candidate.bond_head[3].register_forward_pre_hook(lambda mod,args:seen.update(hidden=args[0].detach().clone()))
  candidate.eval()
  with torch.no_grad():candidate.train();core.seed(889);candidate(bt);candidate.eval()
  handle.remove();hs=seen['hidden'];route_real=torch.arange(len(bt['sequences']))[:,None].expand_as(bt['label_mask'])[bt['label_mask'].bool()]
  independent=candidate.bond_head[3](hs).float()+adapter(hs,bt['nces'],route_real).float()
  core.require(torch.equal(independent,candidate.last_four_state_logits),'Hidden-to-logit fusion route')
  # Real longest batches, finite updates and CPU strict save/restore.
  core.seed(cfg['experiment']['seed']);candidate=GraphTransformer(build_model_config(cfg)).cuda();candidate.train();t=trainer(candidate,cfg)
  batch=to_cuda(collate(train,longest_indices(train,128)));topology_proof(batch)
  for _ in range(2):
   t.optimizer.zero_grad();loss,_=t._forward_pass(batch);core.require(torch.isfinite(loss),'Nonfinite real loss');loss.backward()
   core.require(all(torch.isfinite(p.grad).all() for p in candidate.parameters() if p.grad is not None),'Nonfinite real gradients');t.optimizer.step()
  candidate.eval()
  with torch.no_grad():out=candidate(to_cuda(collate(val,longest_indices(val,512))))
  core.require(torch.isfinite(out).all(),'Longest validation output')
  cpu=GraphTransformer(build_model_config(cfg));cpu.load_state_dict({k:v.cpu() for k,v in candidate.state_dict().items()},strict=True)
  report['checks']=dict(disabled_legacy_equivalent=True,initial_shared_rng_and_state_exact=True,zero_logits_loss_gradients_and_first_update_exact=True,cross_mode_strict_false_rejected_before_copy=True,independent_nonzero_nce_formula=True,real_hidden_fusion_route=True,longest_train128_val512=True,real_two_step_finite=True,cpu_strict_roundtrip=True,unknown_nce_rejected=True)
  del models,old,disabled,candidate,t,batch,cpu;torch.cuda.empty_cache()
 report['status']='passed';dump(OPS/'reference_reuse_receipt.json',dict(status='passed',**bound,new_reference_trainings=0,patches=PATCHES,declared_production_changes=sorted(allowed)))
 dump(OPS/'smoke.json',report);print(json.dumps(dict(status='passed',parameters=report['parameters'],checks=report['checks'])))

def launch():
 OPS.mkdir(exist_ok=True);core.require(not(OPS/'launch.json').exists(),'Unique launch exists')
 s=core.read(OPS/'smoke.json');core.require(s['status']=='passed' and s['gitrev']==core.rev() and s['source_sha256']==core.source_hashes(),'Smoke/source binding')
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S');root=Path('/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/adaptive_nce_head_batch')/stamp
 root.mkdir(parents=True,exist_ok=False);(root/'full').symlink_to(PRIOR_ROOT/'full',target_is_directory=True)
 info=dict(status='submitted',root=str(root),source=str(REPO),gitrev=core.rev(),created_at=core.utc(),source_sha256=core.source_hashes(),smoke_sha256=sha(OPS/'smoke.json'),reference_reuse_sha256=sha(OPS/'reference_reuse_receipt.json'),arms=PATCHES,new_tasks=5,reused_tasks=5,training_cap=None,reference_root=str(PRIOR_ROOT),outer_test_deferred=True,selection='per-fold inner validation only',gpus_available_at_launch=core.free_gpus())
 with(OPS/'launch.claim').open('x') as f:f.write(core.utc())
 dump(OPS/'launch.json',info)
 with(OPS/'worker.log').open('x') as log:p=subprocess.Popen([core.PY,'-u',str(Path(__file__)),'worker'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 info['worker_pid']=p.pid;dump(OPS/'launch.json',info);print(json.dumps({k:v for k,v in info.items() if k!='source_sha256'}))

def worker():
 info=core.read(OPS/'launch.json');root=Path(info['root']);failures=[];tasks=[];active={}
 with(OPS/'worker.claim').open('x') as f:f.write(str(os.getpid()))
 try:
  (OPS/'baseline_audit').symlink_to(reuse.PREV/'baseline_audit',target_is_directory=True)
  for fold in core.FOLDS:tasks.append(dict(arm='full',fold=fold,status='reused_audited',run_root=str(PRIOR_ROOT/'full'/f'fold_{fold}')))
  for fold in core.FOLDS:
   cfg,rr=core.fold_config('nce_head',fold,root);rr.mkdir(parents=True,exist_ok=False);path=rr/'config.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False));tasks.append(dict(arm='nce_head',fold=fold,status='queued',config=str(path),run_root=str(rr)))
  pending=[t for t in tasks if t['arm']=='nce_head']
  while pending or active:
   if not failures:
    for gpu in core.free_gpus():
     if not pending:break
     if gpu in active:continue
     t=pending.pop(0);t.update(status='running',gpu=gpu,started_at=core.utc());env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
     log=open(Path(t['run_root'])/'train_and_audit.log','x');p=subprocess.Popen([core.PY,'-u',str(Path(__file__)),'trial','--config',t['config']],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);t['pid']=p.pid;active[gpu]=(p,t,log)
   else:pending.clear()
   for gpu,(p,t,log) in list(active.items()):
    if p.poll() is None:continue
    log.close();t['ended_at']=core.utc()
    if p.returncode:t['status']='failed';failures.append(dict(fold=t['fold'],exit_code=p.returncode,automatic_retry=False))
    else:
     a=core.read(Path(t['run_root'])/'trial_audit.json');core.require(a['status']=='passed','Trial audit failed');t.update(status='audited',receipt_sha256=sha(Path(t['run_root'])/'trial_audit.json'))
    del active[gpu]
   dump(OPS/'wave_state.json',dict(status='running',root=str(root),tasks=tasks))
   if pending or active:time.sleep(15)
  core.require(not failures,'Trial failures, artifacts retained '+json.dumps(failures))
  dump(OPS/'nce_head_completed.json',dict(status='completed',arm='nce_head',root=str(root),ended_at=core.utc()))
  result=core.present(root,tasks);result.update(reference_reused=True,reference_root=str(PRIOR_ROOT),reference_new_trainings=0,declared_change='late NCE-routed four-state linear residual only')
  (OPS/'analysis.md').write_text((OPS/'analysis.md').read_text().replace('第一轮简化与选权重对照','第八轮NCE末端条件残差单因素对照'))
  dump(OPS/'result.json',result);dump(OPS/'terminal.json',dict(status='completed',root=str(root),gitrev=core.rev(),completed_at=core.utc(),result_sha256=sha(OPS/'result.json'),review_required=True,external_test_deferred=True))
 except BaseException:
  # Leave other users untouched and let any already running owned tasks finish.
  for p,t,log in list(active.values()):p.wait();log.close()
  dump(OPS/'terminal.json',dict(status='failed',root=str(root),gitrev=core.rev(),error=traceback.format_exc(),failures=failures,automatic_retry=False,completed_weights_retained=True));raise

if __name__=='__main__':
 core.require(sys.platform.startswith('linux') and str(REPO).startswith('/mnt/pvc/'),'Pod only');os.chdir(REPO)
 p=argparse.ArgumentParser();p.add_argument('action',choices=['smoke','launch','worker','trial']);p.add_argument('--gpu',type=int,default=0);p.add_argument('--config');a=p.parse_args()
 if a.action=='smoke':smoke(a.gpu)
 elif a.action=='trial':core.trial(a.config)
 else:globals()[a.action]()
