"""One-factor empirical cardinality supervision; original inference unchanged."""
import argparse, copy, datetime, importlib.util, json, logging, os, subprocess, sys, time, traceback
from pathlib import Path
import yaml
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'graph_transform/scripts'),str(REPO/'graph_transform')]
import adaptive_research_ops as core
import adaptive_regularization_wave2 as reuse
from adaptive_baseline_audit import dump,sha
OPS=Path('/mnt/pvc/graphtrans/dbond_adaptive_cardinality_wave10_20261104_ops')
PATCHES={'full':{},'cardinality_01':{'loss.cardinality_weight':.1}}
core.OPS=OPS;core.ARMS=PATCHES
PRIOR_ROOT=reuse.PRIOR_ROOT
PRIOR=Path('/mnt/pvc/graphtrans/dbond_adaptive_research_source_addf42c')


def smoke(gpu):
    import torch,numpy as np,pandas as pd,collections,itertools
    from train_graph_model import create_datasets
    import train_graph_model as ts
    from models.graph_transformer import GraphTransformer
    from models.utils import build_model_config
    from training.trainer import Trainer
    from training.cardinality_loss import cardinality_cross_entropy
    from training.loss_functions import BinaryBondLoss
    from test_four_state_width import collate,to_cuda,longest_indices
    core.require(gpu in core.free_gpus(),'Preflight GPU must be idle before context creation')
    bound=reuse.previous_binding();core.preflight(gpu)
    ts.logger=logging.getLogger('count_loss_preflight');torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('training.original_trainer',PRIOR/'graph_transform/training/trainer.py')
    orig=importlib.util.module_from_spec(spec);spec.loader.exec_module(orig)
    # Independent probability-domain recurrence, used only as a reference.
    def reference(x,mask,h):
        lp=torch.softmax(x.double(),-1);rows=[];offset=0
        for i,n in enumerate(mask.sum(1).tolist()):
            p=1-lp[offset:offset+n,0];offset+=n;d=x.new_ones(1,dtype=torch.float64)
            for v in p:
                d=torch.cat((d*(1-v),d.new_zeros(1)))+torch.cat((d.new_zeros(1),d*v))
            rows.append(-(h[i,:n+1].double()*d.log()).sum()/n)
        return torch.stack(rows).mean()
    toy=torch.tensor([[.1,-.7,.8,.4],[-.3,.5,.2,-.1],[.8,-.5,.1,.2]],requires_grad=True)
    mm=torch.tensor([[1,1],[1,0]],dtype=torch.bool);hh=torch.tensor([[.2,.5,.3],[.7,.3,0.]])
    a=cardinality_cross_entropy(toy,mm,hh);ga=torch.autograd.grad(a,toy)[0]
    b=reference(toy,mm,hh);gb=torch.autograd.grad(b,toy)[0]
    core.require(abs(float(a-b))<3e-6 and float((ga-gb).abs().max())<3e-6,'Count toy independent value/gradient')
    # Exhaustive two-bond enumeration checks the reference distribution itself.
    p=(1-torch.softmax(toy.double(),-1)[:2,0]).detach().numpy();dist=np.zeros(3)
    for bits in itertools.product((0,1),repeat=2):dist[sum(bits)]+=np.prod([p[j] if bit else 1-p[j] for j,bit in enumerate(bits)])
    core.require(abs(float(b)-(-np.dot(hh[0].double().numpy(),np.log(dist))/2-float(hh[1,0])*np.log(float(torch.softmax(toy.double(),-1)[2,0]))-float(hh[1,1])*np.log(float(1-torch.softmax(toy.double(),-1)[2,0])))/2)<3e-6,'Enumeration mismatch')
    proof={}
    for fold in core.FOLDS:
        cfg,_=core.fold_config('cardinality_01',fold,OPS/'count_preflight');cfg=core.load_runtime(cfg);cfg['data']['cache_graphs']=False
        train,val,test=create_datasets(cfg);core.require(test is None and val.sample_weights is None,'Validation/test contract')
        core.require(not set(train.data.seq)&set(val.data.seq),'Sequence leakage')
        # Separate source: already audited W3 inner training rows, not a new split.
        prepared=Path('/mnt/pvc/graphtrans/dbond_adaptive_grouped_baselines_wave3_20261103_ops/prepared')/fold
        receipt=core.read(prepared/'split_receipt.json');tp=prepared/'train_multi.csv'
        core.require(sha(tp)==receipt['prepared_sha256']['train_multi.csv'],'Prepared source binding')
        raw=pd.read_csv(tp);hist={}
        for key,g in raw.groupby(['seq','charge','nce'],sort=False):
            c=collections.Counter(sum(map(int,str(z).split(';'))) for z in g.true_multi)
            hist[(str(key[0]),int(key[1]),float(key[2]))]=c
        core.require(set(raw.seq)==set(train.data.seq),'Prepared train sequence set')
        for row,h in zip(train.data.itertuples(),train.histograms):
            c=hist[(str(row.seq),int(row.charge),float(row.nce))]
            expected=torch.tensor([c[k]/int(row.group_n) for k in range(len(row.seq))],dtype=torch.float32)
            core.require(sum(c.values())==row.group_n and torch.equal(h,expected),'Histogram differs from independent training source')
        core.require('cardinality_histogram' not in val[0],'Training targets exposed to validation')
        if fold==core.FOLDS[0]:
            validation_model=GraphTransformer(build_model_config(cfg)).cuda().eval()
            with torch.no_grad():
                vb=to_cuda(collate(val,longest_indices(val,512)))
                vo=validation_model(vb)
                core.require(torch.isfinite(vo).all(),'Longest validation batch must remain finite')
            del validation_model,vb,vo;torch.cuda.empty_cache()
        bt=collate(train,list(range(16)));mask=bt['label_mask'].bool()
        core.seed(cfg['experiment']['seed']);m=GraphTransformer(build_model_config(cfg));m.train()
        core.require(sum(p.numel() for p in m.parameters())==2002284,'Architecture changed')
        opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4)
        trainer=Trainer(model=m,optimizer=opt,criterion=BinaryBondLoss(cfg['loss']),device=torch.device('cpu'),config=cfg)
        total,_=trainer._forward_pass(bt);x=m.last_four_state_logits
        part=cardinality_cross_entropy(x,mask,bt['cardinality_histogram']);gx=torch.autograd.grad(part,x,retain_graph=True)[0]
        ref=reference(x,mask,bt['cardinality_histogram']);gr=torch.autograd.grad(ref,x,retain_graph=True)[0]
        err=abs(float(part-ref));ge=float((gx.double()-gr.double()).abs().max())
        core.require(err<3e-6 and ge<3e-6,'Real count value/gradient mismatch')
        # Disabled path preserves original Trainer loss, all gradients and first update.
        off=copy.deepcopy(cfg);off['loss'].pop('cardinality_weight');clean=dict(bt);clean.pop('cardinality_histogram')
        m0=GraphTransformer(build_model_config(off));m0.load_state_dict(m.state_dict(),strict=True)
        m1=GraphTransformer(build_model_config(off));m1.load_state_dict(m.state_dict(),strict=True)
        o0=torch.optim.AdamW(m0.parameters(),lr=1e-4,weight_decay=1e-4);o1=torch.optim.AdamW(m1.parameters(),lr=1e-4,weight_decay=1e-4)
        t0=orig.Trainer(model=m0,optimizer=o0,criterion=BinaryBondLoss(off['loss']),device=torch.device('cpu'),config=off)
        t1=Trainer(model=m1,optimizer=o1,criterion=BinaryBondLoss(off['loss']),device=torch.device('cpu'),config=off)
        core.seed(901);l0,_=t0._forward_pass(clean);l0.backward();o0.step()
        core.seed(901);l1,_=t1._forward_pass(clean);l1.backward();o1.step()
        core.require(torch.equal(l0,l1) and all(torch.equal(v,m1.state_dict()[k]) for k,v in m0.state_dict().items()),'Disabled original loss/update/RNG mismatch')
        for (k,p0),(_,p1) in zip(m0.named_parameters(),m1.named_parameters()):
            core.require((p0.grad is None and p1.grad is None) or torch.equal(p0.grad,p1.grad),'Disabled gradient mismatch '+k)
        core.seed(110);u=m(bt).detach();changed=dict(bt);changed['cardinality_histogram']=torch.flip(bt['cardinality_histogram'],[-1]);core.seed(110);v=m(changed).detach()
        core.require(torch.equal(u,v),'Count labels leaked into model input')
        for _ in range(2):
            opt.zero_grad();z,_=trainer._forward_pass(bt);z.backward();core.require(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()),'Nonfinite real update');opt.step()
        proof[fold]=dict(histogram_provenance=train.four_state_provenance['cardinality'],prepared_train_sha256=sha(tp),train_sha256=sha(train.csv_path),val_sha256=sha(val.csv_path),count_loss_error=err,count_gradient_error=ge,disabled_loss_update_gradient_exact=True,targets_not_inputs=True,validation_has_no_count_targets=True)
    dump(OPS/'cardinality_smoke.json',dict(status='passed',folds=proof,tolerance=3e-6,coefficient=.1,architecture_smoke_sha256=sha(OPS/'smoke.json')))
    dump(OPS/'reference_reuse_receipt.json',dict(status='passed',**bound,new_reference_trainings=0,patches=PATCHES,cardinality_smoke_sha256=sha(OPS/'cardinality_smoke.json'),model_inference_unchanged=True,declared_changes=['training/cardinality_loss.py','data/cardinality_dataset.py','training/trainer.py','data/graph_dataset.py','scripts/train_graph_model.py']))
    print(json.dumps(dict(status='passed',folds=proof)))


def launch():
    OPS.mkdir(exist_ok=True);core.require(not(OPS/'launch.json').exists(),'Unique launch exists')
    s=core.read(OPS/'smoke.json');c=core.read(OPS/'cardinality_smoke.json');r=core.read(OPS/'reference_reuse_receipt.json')
    core.require(s['status']==c['status']==r['status']=='passed' and s['gitrev']==core.rev() and s['source_sha256']==core.source_hashes(),'Smoke/source mismatch')
    core.require(r['cardinality_smoke_sha256']==sha(OPS/'cardinality_smoke.json') and c['architecture_smoke_sha256']==sha(OPS/'smoke.json'),'Count proof binding')
    core.require(r['review_sha256']==reuse.previous_binding()['review_sha256'],'Reference changed')
    root=Path('/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/adaptive_cardinality_batch')/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
    root.mkdir(parents=True,exist_ok=False);(root/'full').symlink_to(PRIOR_ROOT/'full',target_is_directory=True)
    info=dict(status='submitted',root=str(root),source=str(REPO),gitrev=core.rev(),created_at=core.utc(),source_sha256=core.source_hashes(),smoke_sha256=sha(OPS/'smoke.json'),cardinality_smoke_sha256=sha(OPS/'cardinality_smoke.json'),reference_reuse_sha256=sha(OPS/'reference_reuse_receipt.json'),arms=PATCHES,new_tasks=5,reused_tasks=5,reference_root=str(PRIOR_ROOT),training_cap=None,outer_test_deferred=True,gpus_available_at_launch=core.free_gpus())
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
            cfg,rr=core.fold_config('cardinality_01',f,root);rr.mkdir(parents=True,exist_ok=False);path=rr/'config.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False));tasks.append(dict(arm='cardinality_01',fold=f,status='queued',config=str(path),run_root=str(rr)))
        pending=[t for t in tasks if t['arm']=='cardinality_01']
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
        core.require(not failures,'Trial failures; retained '+json.dumps(failures))
        dump(OPS/'cardinality_01_completed.json',dict(status='completed',arm='cardinality_01',root=str(root),ended_at=core.utc()))
        r=core.present(root,tasks);r.update(reference_reused=True,reference_root=str(PRIOR_ROOT),reference_new_trainings=0,declared_factor='four-state CE + .1 mean(normalized empirical count-distribution CE)')
        dump(OPS/'result.json',r);dump(OPS/'terminal.json',dict(status='completed',root=str(root),gitrev=core.rev(),completed_at=core.utc(),result_sha256=sha(OPS/'result.json'),review_required=True,external_test_deferred=True))
    except BaseException:
        for p,t,log in list(active.values()):p.wait();log.close()
        dump(OPS/'terminal.json',dict(status='failed',root=str(root),gitrev=core.rev(),error=traceback.format_exc(),failures=failures,automatic_retry=False,completed_weights_retained=True));raise


if __name__=='__main__':
    core.require(sys.platform.startswith('linux') and str(REPO).startswith('/mnt/pvc/'),'Pod only');os.chdir(REPO)
    p=argparse.ArgumentParser();p.add_argument('action',choices=['smoke','launch','worker','trial']);p.add_argument('--gpu',type=int,default=0);p.add_argument('--config');a=p.parse_args()
    if a.action=='smoke':smoke(a.gpu)
    elif a.action=='trial':core.trial(a.config)
    else:globals()[a.action]()
