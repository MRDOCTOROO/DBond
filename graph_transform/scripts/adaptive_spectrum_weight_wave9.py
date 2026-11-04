"""Single-factor spectrum-count loss weighting, original four-state model."""
import argparse,datetime,json,logging,os,subprocess,sys,time,traceback
from pathlib import Path
import yaml
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'graph_transform/scripts'),str(REPO/'graph_transform')]
import adaptive_research_ops as core
import adaptive_regularization_wave2 as reuse
from adaptive_baseline_audit import dump,sha
OPS=Path('/mnt/pvc/graphtrans/dbond_adaptive_spectrum_weight_wave9_20261104_ops')
PATCHES={'full':{},'spectrum_count':{'data.weighting_scheme':'spectrum'}}
core.OPS=OPS;core.ARMS=PATCHES
PRIOR_ROOT=reuse.PRIOR_ROOT

def smoke(gpu):
 # Reuse every original architecture/longest-batch/source/seed check unchanged.
 reuse.OPS=OPS;reuse.PATCHES=PATCHES;reuse.smoke(gpu)
 import torch,numpy as np
 from train_graph_model import create_datasets
 import train_graph_model as ts
 from models.graph_transformer import GraphTransformer
 from models.utils import build_model_config
 from training.trainer import Trainer
 from training.loss_functions import BinaryBondLoss
 from test_four_state_width import collate
 ts.logger=logging.getLogger('spectrum_weight_preflight');torch.set_num_threads(1)
 proof={}
 for fold in core.FOLDS:
  cfg,_=core.fold_config('spectrum_count',fold,OPS/'weighting_preflight');cfg=core.load_runtime(cfg);cfg['data']['cache_graphs']=False
  train,val,test=create_datasets(cfg);core.require(test is None,'Test exposed')
  weights=np.asarray(train.sample_weights);counts=train.data.group_n.to_numpy(dtype=np.float32)
  core.require(np.array_equal(weights,counts) and np.isfinite(weights).all() and (weights>0).all(),'Weights must equal source-bound group_n exactly')
  core.require(val.sample_weights is None and not set(train.data.seq)&set(val.data.seq),'Validation weights or sequence leakage')
  core.require(train.four_state_provenance['weighting_scheme']=='spectrum','Target provenance must name actual weighting')
  batch=collate(train,list(range(16)));mask=batch['label_mask'].bool();expected=torch.tensor(counts[:16])[:,None].expand_as(mask)[mask]
  core.seed(cfg['experiment']['seed']);model=GraphTransformer(build_model_config(cfg));model.train()
  trainer=Trainer(model=model,optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4),criterion=BinaryBondLoss(cfg['loss']),device=torch.device('cpu'),config=cfg)
  core.require(torch.equal(trainer._masked_row_weights(batch),expected),'Valid-bond weight routing')
  loss,_=trainer._forward_pass(batch);logits=model.last_four_state_logits;logits.retain_grad();loss.backward()
  x=logits.detach().double();target=batch['four_state_targets'][mask].double();w=expected.double();logprob=x-torch.logsumexp(x,dim=-1,keepdim=True)
  manual=(-(target*logprob).sum(-1)*w).sum()/w.sum();gradient=(torch.softmax(x,-1)*target.sum(-1,keepdim=True)-target)*w[:,None]/w.sum()
  loss_error=abs(float(loss)-float(manual));grad_error=float((logits.grad.double()-gradient).abs().max())
  core.require(loss_error<3e-6 and grad_error<3e-6,'Independent weighted CE loss/gradient mismatch')
  without=dict(batch);without['sample_weights']=torch.ones_like(batch['sample_weights'])
  core.seed(337);a=model(batch).detach().clone();core.seed(337);b=model(without).detach().clone()
  core.require(torch.equal(a,b),'Observation weights leaked into model inputs')
  target_manifest=cfg['data']['four_state_targets_manifest'];m=core.read(target_manifest);sidecar=Path(target_manifest).parent/m['sidecar']['path']
  core.require(sha(sidecar)==m['sidecar']['sha256'],'Target sidecar binding')
  proof[fold]=dict(weights_equal_group_n=True,valid_bond_weight_route_exact=True,weights_not_model_inputs=True,validation_unweighted=True,weighted_loss_error=loss_error,weighted_gradient_error=grad_error,weight_vector_sha256=__import__('hashlib').sha256(weights.astype('<f4').tobytes()).hexdigest(),min_weight=float(weights.min()),max_weight=float(weights.max()),mean_weight=float(weights.mean()),train_rows=len(weights),train_sha256=sha(train.csv_path),val_sha256=sha(val.csv_path),target_sidecar_sha256=sha(sidecar),target_provenance=train.four_state_provenance)
  del model,trainer,train,val,batch,logits,loss
 dump(OPS/'weighting_smoke.json',dict(status='passed',folds=proof,loss_gradient_tolerance=3e-6,architecture_smoke_sha256=sha(OPS/'smoke.json')))
 ref=core.read(OPS/'reference_reuse_receipt.json');ref.update(weighting_smoke_sha256=sha(OPS/'weighting_smoke.json'),declared_metadata_change='FourStateDataset provenance describes actual weighting; group_uniform path and every numeric model/training/loss file unchanged')
 dump(OPS/'reference_reuse_receipt.json',ref);print(json.dumps(dict(status='passed',parameters=2002284,folds=proof)))

def launch():
 OPS.mkdir(exist_ok=True);core.require(not(OPS/'launch.json').exists(),'Unique launch exists')
 s=core.read(OPS/'smoke.json');w=core.read(OPS/'weighting_smoke.json');r=core.read(OPS/'reference_reuse_receipt.json')
 core.require(s['status']==w['status']==r['status']=='passed' and s['gitrev']==core.rev() and s['source_sha256']==core.source_hashes(),'Smoke/source mismatch')
 core.require(r['weighting_smoke_sha256']==sha(OPS/'weighting_smoke.json') and w['architecture_smoke_sha256']==sha(OPS/'smoke.json'),'Weighting proof binding')
 core.require(r['review_sha256']==reuse.previous_binding()['review_sha256'],'Reference reuse changed')
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S');root=Path('/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/adaptive_spectrum_weight_batch')/stamp
 root.mkdir(parents=True,exist_ok=False);(root/'full').symlink_to(PRIOR_ROOT/'full',target_is_directory=True)
 info=dict(status='submitted',root=str(root),source=str(REPO),gitrev=core.rev(),created_at=core.utc(),source_sha256=core.source_hashes(),smoke_sha256=sha(OPS/'smoke.json'),weighting_smoke_sha256=sha(OPS/'weighting_smoke.json'),reference_reuse_sha256=sha(OPS/'reference_reuse_receipt.json'),arms=PATCHES,new_tasks=5,reused_tasks=5,reference_root=str(PRIOR_ROOT),training_cap=None,outer_test_deferred=True,gpus_available_at_launch=core.free_gpus())
 with(OPS/'launch.claim').open('x') as f:f.write(core.utc())
 dump(OPS/'launch.json',info)
 with(OPS/'worker.log').open('x') as log:p=subprocess.Popen([core.PY,'-u',str(Path(__file__)),'worker'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 info['worker_pid']=p.pid;dump(OPS/'launch.json',info);print(json.dumps({k:v for k,v in info.items() if k!='source_sha256'}))

def worker():
 info=core.read(OPS/'launch.json');root=Path(info['root']);tasks=[];active={};failures=[]
 with(OPS/'worker.claim').open('x') as f:f.write(str(os.getpid()))
 try:
  (OPS/'baseline_audit').symlink_to(reuse.PREV/'baseline_audit',target_is_directory=True)
  for f in core.FOLDS:tasks.append(dict(arm='full',fold=f,status='reused_audited',run_root=str(PRIOR_ROOT/'full'/f'fold_{f}')))
  for f in core.FOLDS:
   cfg,rr=core.fold_config('spectrum_count',f,root);rr.mkdir(parents=True,exist_ok=False);path=rr/'config.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False));tasks.append(dict(arm='spectrum_count',fold=f,status='queued',config=str(path),run_root=str(rr)))
  pending=[t for t in tasks if t['arm']=='spectrum_count']
  while pending or active:
   if not failures:
    for gpu in core.free_gpus():
     if not pending:break
     if gpu in active:continue
     t=pending.pop(0);t.update(status='running',gpu=gpu,started_at=core.utc());env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1');log=open(Path(t['run_root'])/'train_and_audit.log','x')
     p=subprocess.Popen([core.PY,'-u',str(Path(__file__)),'trial','--config',t['config']],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);t['pid']=p.pid;active[gpu]=(p,t,log)
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
  core.require(not failures,'Trial failures; weights retained '+json.dumps(failures))
  dump(OPS/'spectrum_count_completed.json',dict(status='completed',arm='spectrum_count',root=str(root),ended_at=core.utc()))
  result=core.present(root,tasks);result.update(reference_reused=True,reference_root=str(PRIOR_ROOT),reference_new_trainings=0,declared_factor='data.weighting_scheme: group_uniform to spectrum; numeric production files unchanged')
  (OPS/'analysis.md').write_text((OPS/'analysis.md').read_text().replace('第一轮简化与选权重对照','第九轮训练观测数损失加权单因素对照')+'\n模型仍为原full，未叠加NCE残差；仅训练有效键CE按所属条件组group_n加权并按batch有效键权重和归一化。验证和全部指标不加权。折叠输入和序列化目标不改，不能称完整原谱硬标签训练等价。\n')
  dump(OPS/'result.json',result);dump(OPS/'terminal.json',dict(status='completed',root=str(root),gitrev=core.rev(),completed_at=core.utc(),result_sha256=sha(OPS/'result.json'),review_required=True,external_test_deferred=True))
 except BaseException:
  for p,t,log in list(active.values()):p.wait();log.close()
  dump(OPS/'terminal.json',dict(status='failed',root=str(root),gitrev=core.rev(),error=traceback.format_exc(),failures=failures,automatic_retry=False,completed_weights_retained=True));raise

if __name__=='__main__':
 core.require(sys.platform.startswith('linux') and str(REPO).startswith('/mnt/pvc/'),'Pod only');os.chdir(REPO)
 p=argparse.ArgumentParser();p.add_argument('action',choices=['smoke','launch','worker','trial']);p.add_argument('--gpu',type=int,default=0);p.add_argument('--config');a=p.parse_args()
 if a.action=='smoke':smoke(a.gpu)
 elif a.action=='trial':core.trial(a.config)
 else:globals()[a.action]()
