"""Pod-only grouped internal-validation baseline retraining; no external scoring."""
import argparse,copy,datetime,importlib.util,json,os,random,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'graph_transform/scripts'))
from adaptive_baseline_audit import sha,dump,independent_metrics
OPS=Path('/mnt/pvc/graphtrans/dbond_adaptive_grouped_baselines_wave3_20261103_ops')
OLD=Path('/mnt/pvc/graphtrans/dbond_adaptive_research_20261102_ops')
PRIOR=Path('/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/adaptive_research_batch/20261102_155846')
DATA=Path('/mnt/pvc/graphtrans/DBond/dataset')
PY='/mnt/pvc/graphtrans/DBond/.venv/bin/python'
FOLDS=('1222','2252','3514','6072','9075')
SEEDS=(949822381,2775888976,1365222152,612878555,1785808201)
SPECS={
 'dbond_s':('ludbond','train.dbond_s.py','dbond_s_config/pre.yaml'),
 'dbond_m':('ludbond','train.dbond_m.py','dbond_m_config/pre.yaml'),
 'dbond_af':('ludbondaf','train.dbond_m.exp_af.py','dbond_m_exp_af_config/af_pre.yaml'),
 'dbond_af_opt':('ludbondaf','train.dbond_m.exp_af.py','dbond_m_exp_af_config/af_opt_pre.yaml'),
}
def read(p):return json.loads(Path(p).read_text())
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def rev():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
def sources():
    paths=[*REPO.joinpath('ludbond').rglob('*.py'),*REPO.joinpath('ludbondaf').rglob('*.py'),
           *REPO.joinpath('ludbond').rglob('*.yaml'),*REPO.joinpath('ludbondaf').rglob('*.yaml'),
           Path(__file__),REPO/'graph_transform/scripts/adaptive_baseline_audit.py']
    return {str(p.relative_to(REPO)):sha(p) for p in sorted(set(paths))}
def seed(n):
    random.seed(n);np.random.seed(n);torch.manual_seed(n)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(n)
def free_gpus():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return [int(a[0]) for s in out.splitlines() if (a:=[int(x.strip()) for x in s.split(',')]) and a[1]<64 and a[2]==0]
def module(arm):
    folder,name,_=SPECS[arm];sys.path.insert(0,str(REPO/folder))
    spec=importlib.util.spec_from_file_location('grouped_'+arm,REPO/folder/name)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def keys(df):
    return list(zip(df.seq.astype(str),df.charge.astype(float),df.pep_mass.astype(float),df.nce.astype(float),df.scan_num.astype(float)))
def mapping(single,frame):
    kk=keys(frame);assert len(set(kk))==len(kk),'Duplicate spectrum identities require explicit resolution'
    index={k:i for i,k in enumerate(kk)};ids=np.array([index[k] for k in keys(single)],dtype=np.int64)
    bi=single.bond_pos.to_numpy(dtype=np.int64);width=35
    assert len(np.unique(ids*width+bi))==len(single),'Duplicate single-bond source'
    mask=np.zeros((len(frame),width),bool);mask[ids,bi]=True
    expected=np.arange(width)[None,:]<(frame.seq.str.len().to_numpy()-1)[:,None]
    assert np.array_equal(mask,expected),'Single-bond source missing or adding bonds'
    labels=np.zeros(mask.shape,np.int8)
    for i,s in enumerate(frame.true_multi):
        v=np.fromstring(str(s),sep=';',dtype=np.int8);labels[i,:len(v)]=v
    assert np.array_equal(labels[ids,bi],single.bond_label.to_numpy(dtype=np.int8))
    return ids,bi
def prepare():
    """Reuse exactly the GT inner sequence sets, not a newly drawn split."""
    out={}
    for fold,n in zip(FOLDS,SEEDS):
        dst=OPS/'prepared'/fold;dst.mkdir(parents=True,exist_ok=False)
        a=read(PRIOR/'full'/f'fold_{fold}/trial_audit.json');assert a['status']=='passed' and a['external_test_deferred']
        c=Path(a['policies']['max_f1']['checkpoint']);saved=torch.load(c,map_location='cpu',weights_only=False)
        assert saved['extra_state']['training_rng_protocol']['training_seed']==n
        p=saved['extra_state']['row_validation_provenance'];tp=Path(p['split_train_csv']);vp=Path(p['split_val_csv'])
        trainseq=set(pd.read_csv(tp,usecols=['seq']).seq);val=pd.read_csv(vp,na_filter=False);valseq=set(val.seq)
        assert not trainseq&valseq and trainseq and valseq
        assert sha(vp)==a['policies']['max_f1']['validation_source_sha256']
        raw=DATA/'5fold'/f'{fold}.train.fbr.shuffle.multi.csv';multi=pd.read_csv(raw,na_filter=False)
        assert set(multi.seq)==trainseq|valseq and val.seq.str.len().max()<=36
        original=multi[multi.seq.isin(valseq)].copy()
        assert len(original)==len(val) and set(keys(original))==set(keys(val))
        truth={k:s for k,s in zip(keys(original),original.true_multi)}
        for k,s in zip(keys(val),val.true_multi):
            assert np.array_equal(np.fromstring(str(truth[k]),sep=';',dtype=np.int8)[:len(k[0])-1],np.fromstring(str(s),sep=';',dtype=np.int8)[:len(k[0])-1])
        val.to_csv(dst/'val_multi.csv',index=False)
        multi[multi.seq.isin(trainseq)].to_csv(dst/'train_multi.csv',index=False)
        raw_s=DATA/'5fold'/f'{fold}.train.shuffle.csv';single=pd.read_csv(raw_s,na_filter=False)
        assert set(single.seq)==trainseq|valseq
        sv=single[single.seq.isin(valseq)].copy();mapping(sv,val)
        single[single.seq.isin(trainseq)].to_csv(dst/'train_single.csv',index=False)
        sv.to_csv(dst/'val_single.csv',index=False)
        proof=dict(training_seed=n,validation_split_seed=n,split_base_seed=42,
            train_sequences=sorted(trainseq),validation_sequences=sorted(valseq),sequence_overlap=0,
            gt_checkpoint_sha256=sha(c),gt_trial_receipt_sha256=sha(PRIOR/'full'/f'fold_{fold}/trial_audit.json'),
            gt_train_sha256=sha(tp),gt_val_sha256=sha(vp),raw_multi_sha256=sha(raw),raw_single_sha256=sha(raw_s),
            prepared_sha256={p.name:sha(p) for p in dst.glob('*.csv')})
        dump(dst/'split_receipt.json',proof);out[fold]=dict(split_receipt_sha256=sha(dst/'split_receipt.json'))
        del saved,multi,single,val,original,sv
    return out
def config(arm,fold):
    folder,_,name=SPECS[arm];cfg=yaml.safe_load((REPO/folder/name).read_text())
    dst=OPS/'prepared'/fold;suffix='single' if arm=='dbond_s' else 'multi'
    cfg['csv'].update(train_dataset_path=str(dst/f'train_{suffix}.csv'),validation_dataset_path=str(dst/f'val_{suffix}.csv'),test_dataset_path=None)
    cfg['train_args']['seed']=SEEDS[FOLDS.index(fold)]
    cfg['_research']=dict(arm=arm,fold=fold,external_test_deferred=True,validation='Same GT inner sequences, no row random_split',
        source_recipe=str(REPO/folder/name),source_recipe_sha256=sha(REPO/folder/name),split_receipt_sha256=sha(dst/'split_receipt.json'),
        preprocessing='Original hard-label raw spectra and input masking; no folded soft-target replacement',
        optimizer_loss_schedule='Original model recipe; padded loss unchanged; validation uses length-1 bonds')
    return cfg
def checked_data(cfg,split):
    a=cfg['_research'];dst=OPS/'prepared'/a['fold'];receipt=read(dst/'split_receipt.json')
    assert sha(dst/'split_receipt.json')==a['split_receipt_sha256']
    p=Path(cfg['csv'][split+'_dataset_path']);assert sha(p)==receipt['prepared_sha256'][p.name]
    assert cfg['csv']['test_dataset_path'] is None
    assert cfg['csv']['zero_fill_col_name']==['intensity','scan_num']
    m=module(a['arm']);return m,m.PepDataset(cfg,split)
def inputs(batch,arm,device):
    x=[v.to(device) for v in batch]
    if arm=='dbond_s':names=('seq_index_batch','seq_padding_mask_batch','bond_index_batch','bond_vec_batch','state_vec_batch','env_vec_batch')
    else:names=('seq_index_batch','seq_padding_mask_batch','state_vec_batch','env_vec_batch')
    return dict(zip(names,x[:-1])),x[-1]
def metrics(y,p,mask,frame):
    m=independent_metrics(y,p,mask);lengths=frame.seq.str.len().to_numpy()-1
    soft=[np.fromstring(str(s),sep=';',dtype=np.float32)[:n] for s,n in zip(frame.soft_multi,lengths)]
    # Match the prior independent same-pass GT report: probability means in float64,
    # soft-target row means in their archived float32 representation.
    pred=np.array([float(p[i,:n].astype(float).mean()) for i,n in enumerate(lengths)])
    real=np.array([float(s.mean()) for s in soft]);first={}
    for i,k in enumerate(zip(frame.seq,frame.charge.astype(float),frame.nce.astype(float))):first.setdefault(k,i)
    seq={}
    for k,i in first.items():seq.setdefault(k[0],[]).append(i)
    sp=np.array([pred[z].mean() for z in seq.values()]);sq=np.array([real[z].mean() for z in seq.values()]);top=np.argsort(-sp,kind='mergesort')[:max(1,round(len(sp)*.1))]
    m.update(q_brier=float(np.concatenate([(p[i,:n].astype(float)-s)**2 for i,(n,s) in enumerate(zip(lengths,soft))]).mean()),
        q_seq=float(spearmanr(sp,sq).statistic),top10=float(sq[top].mean()/sq.mean()))
    assert all(np.isfinite(v) for v in m.values());return m
def predict(net,dataset,cfg,device):
    arm=cfg['_research']['arm'];net.eval();pp=[];yy=[]
    with torch.inference_mode():
        for start in range(0,len(dataset),cfg['train_args']['batch_size']):
            batch=dataset.__getitem__(list(range(start,min(start+cfg['train_args']['batch_size'],len(dataset)))))
            kw,y=inputs(batch,arm,device);logits=net(**kw)
            p=torch.softmax(logits,dim=1)[:,1] if arm=='dbond_s' else torch.sigmoid(logits)
            assert torch.isfinite(p).all();pp.append(p.cpu().numpy());yy.append(y.cpu().numpy())
    pp=np.concatenate(pp);yy=np.concatenate(yy)
    frame=pd.read_csv(OPS/'prepared'/cfg['_research']['fold']/'val_multi.csv',na_filter=False)
    mask=np.arange(35)[None,:]<(frame.seq.str.len().to_numpy()-1)[:,None]
    if arm=='dbond_s':
        ids,bi=mapping(dataset.df,frame);p=np.zeros(mask.shape,np.float32);y=np.zeros(mask.shape,np.int8)
        p[ids,bi]=pp;y[ids,bi]=yy
    else:p=pp.astype(np.float32);y=yy.astype(np.int8)
    assert p.shape==y.shape==mask.shape
    for i,s in enumerate(frame.true_multi):
        assert np.array_equal(y[i,mask[i]],np.fromstring(str(s),sep=';',dtype=np.int8)[:mask[i].sum()])
    return y,p,mask,metrics(y,p,mask,frame)
def validate_fixture():
    y=np.array([[1,0,0,0],[0,1,0,0]],np.int8);p=np.array([[.8,.4,1.,1.],[.6,.7,1.,1.]])
    mask=np.array([[1,1,0,0],[1,1,0,0]],bool);a=independent_metrics(y,p,mask)
    assert a['bond_f1']==.8 and a['subset_acc']==.5
    p[:,2:]=0;assert independent_metrics(y,p,mask)==a
def smoke(gpu):
    assert gpu in free_gpus();OPS.mkdir(exist_ok=True)
    with (OPS/'smoke.claim').open('x') as f:f.write(utc())
    torch.set_num_threads(1);torch.cuda.set_device(gpu);validate_fixture();splits=prepare();proof={}
    old=read(OLD/'baseline_audit/result.json');assert old['status']=='completed' and len(old['receipts'])==20
    for arm in SPECS:
        cfg=config(arm,'1222');m,ds=checked_data(cfg,'train');seed(SEEDS[0])
        net=m.Net(copy.deepcopy(cfg));opt=m.build_optimizer(net,cfg);lossfn=m.build_loss_func(cfg)
        assert cfg['train_args']['optimizer']=='adam' and cfg['train_args']['loss_type']=='ce'
        # Exact current-model versus archived checkpoint structural compatibility, all five folds.
        for fold in FOLDS:
            archived=next(r for r in old['receipts'] if r['model']==arm+'_pre' and r['fold']==fold)
            sourcefile=REPO/('ludbond/dbond_s.py' if arm=='dbond_s' else 'ludbond/dbond_m.py' if arm=='dbond_m' else 'ludbondaf/dbond_m_exp_af.py')
            assert sha(sourcefile)==archived['model_source_sha256']
            saved=torch.load(archived['checkpoint'],map_location='cpu',weights_only=False);clone=m.Net(copy.deepcopy(saved['config'] if 'config' in saved else {k:saved[k] for k in ('csv','seq','model','train_args')}))
            clone.load_state_dict(saved['model_state_dict'],strict=True);del clone,saved
        batch=ds.__getitem__(list(range(16)));kw,y=inputs(batch,arm,'cpu');net.eval();out=net(**kw)
        expected=torch.nn.functional.cross_entropy(out,y,reduction='mean',**cfg['train_args']['loss_args']) if arm=='dbond_s' else torch.nn.functional.multilabel_soft_margin_loss(out,y)
        torch.testing.assert_close(lossfn(out,y),expected,rtol=0,atol=0)
        assert np.all(batch[-3 if arm=='dbond_s' else -3].numpy()[:,-1]==0) # state intensity is masked
        assert np.all(batch[-2].numpy()[:,-1]==0) # scan_num is masked
        net=net.cuda();kw,y=inputs(batch,arm,'cuda');net.train();opt=m.build_optimizer(net,cfg)
        value=lossfn(net(**kw),y);assert torch.isfinite(value);value.backward()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None);opt.step()
        _,val=checked_data(cfg,'validation');net.eval()
        longest=val.df.seq.str.len().sort_values().index[-16:].tolist();b=val.__getitem__(longest);kw,y=inputs(b,arm,'cuda')
        assert torch.isfinite(lossfn(net(**kw),y))
        proof[arm]=dict(archived_strict_cpu=5,loss_exact=True,pre_synthesis_mask=True,real_step_finite=True,longest_validation_finite=True,
            parameters=sum(p.numel() for p in net.parameters()),recipe_sha256=cfg['_research']['source_recipe_sha256'])
        del net,opt,ds,val;torch.cuda.empty_cache()
    dump(OPS/'smoke.json',dict(status='passed',source=str(REPO),gitrev=rev(),source_sha256=sources(),splits=splits,models=proof,baseline_audit_sha256=sha(OLD/'baseline_audit/result.json'),outer_test_deferred=True))
    print(json.dumps(dict(status='passed',gitrev=rev(),models=proof)))
def trial(cfgpath):
    torch.set_num_threads(1);cfg=yaml.safe_load(Path(cfgpath).read_text());a=cfg['_research'];rr=Path(cfgpath).parent
    torch.cuda.set_device(0);seed(cfg['train_args']['seed']);m,train=checked_data(cfg,'train');_,val=checked_data(cfg,'validation')
    net=m.Net(copy.deepcopy(cfg)).cuda();opt=m.build_optimizer(net,cfg);lossfn=m.build_loss_func(cfg);started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    best={'max_f1':-1.,'composite':-1.};stopbest=-1.;bad=0
    for epoch in range(1,cfg['train_args']['epoch']+1):
        net.train()
        for start in range(0,len(train),cfg['train_args']['batch_size']):
            batch=train.__getitem__(list(range(start,min(start+cfg['train_args']['batch_size'],len(train)))))
            kw,y=inputs(batch,a['arm'],'cuda');net.zero_grad();loss=lossfn(net(**kw),y)
            assert torch.isfinite(loss);loss.backward();opt.step()
        y,p,mask,ms=predict(net,val,cfg,'cuda');scores={'max_f1':ms['bond_f1'],'composite':.4*ms['bond_f1']+.4*ms['example_f1_harmonic']+.2*ms['subset_acc']}
        for policy,value in scores.items():
            if value>best[policy]:
                best[policy]=value
                torch.save(dict(model_state_dict=net.state_dict(),optimizer_state_dict=opt.state_dict(),config=cfg,epoch=epoch,
                    metrics=ms,selector=policy,selector_score=value,gitrev=rev(),source_sha256=sources(),split_receipt_sha256=a['split_receipt_sha256']),rr/f'{policy}.pt')
                np.savez_compressed(rr/f'{policy}.npz',y=y,p=p,mask=mask)
        if ms['bond_f1']>stopbest+cfg['train_args']['early_stopping']['delta']:stopbest=ms['bond_f1'];bad=0
        else:bad+=1
        print(json.dumps(dict(epoch=epoch,internal_f1=ms['bond_f1'],subset_acc=ms['subset_acc'])),flush=True)
        if bad>=cfg['train_args']['early_stopping']['patience']:break
    elapsed=time.monotonic()-started;resource=dict(training_wall_seconds=elapsed,training_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,training_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    del net,opt,train;torch.cuda.empty_cache();policies={};states={}
    for policy in best:
        ck=rr/f'{policy}.pt';saved=torch.load(ck,map_location='cpu',weights_only=False)
        clone=m.Net(copy.deepcopy(cfg));clone.load_state_dict(saved['model_state_dict'],strict=True);clone.cuda()
        yy,pp,mm,replay=predict(clone,val,cfg,'cuda')
        with np.load(rr/f'{policy}.npz') as z:oy=z['y'];op=z['p'];om=z['mask']
        assert np.array_equal(yy,oy) and np.array_equal(mm,om)
        error=float(np.max(np.abs(pp-op)));assert np.allclose(pp[mm],op[mm],rtol=1e-5,atol=1e-5),'Replay probability mismatch'
        frame=pd.read_csv(OPS/'prepared'/a['fold']/'val_multi.csv',na_filter=False);computed=metrics(oy,op,om,frame)
        assert max(abs(computed[k]-saved['metrics'][k]) for k in computed)<1e-12
        policies[policy]=dict(checkpoint=str(ck),checkpoint_sha256=sha(ck),prediction=str(rr/f'{policy}.npz'),prediction_sha256=sha(rr/f'{policy}.npz'),strict_cpu=True,epoch=saved['epoch'],metrics=computed,replay_metrics=replay,max_probability_replay_error=error)
        states[policy]=saved['model_state_dict'];del clone;torch.cuda.empty_cache()
    identical=states['max_f1'].keys()==states['composite'].keys() and all(torch.equal(v,states['composite'][k]) for k,v in states['max_f1'].items())
    splitproof=read(OPS/'prepared'/a['fold']/'split_receipt.json')
    assert not set(splitproof['train_sequences'])&set(splitproof['validation_sequences'])
    dump(rr/'trial_audit.json',dict(status='passed',arm=a['arm'],fold=a['fold'],gitrev=rev(),config_sha256=sha(cfgpath),split_receipt_sha256=a['split_receipt_sha256'],actual_seed=cfg['train_args']['seed'],outer_test_deferred=True,policies=policies,identical_model_states=identical,resources=resource))
def launch():
    s=read(OPS/'smoke.json');assert s['status']=='passed' and s['gitrev']==rev() and s['source_sha256']==sources()
    # Prior wave must be audited and acknowledged by the caller before this unique launch.
    prior=read(OPS/'previous_wave_review.json');assert prior['status']=='audited_and_acknowledged'
    with (OPS/'launch.claim').open('x') as f:f.write(utc())
    root=Path('/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/adaptive_grouped_baselines_batch')/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S');root.mkdir(parents=True,exist_ok=False)
    info=dict(status='submitted',root=str(root),source=str(REPO),gitrev=rev(),created_at=utc(),smoke_sha256=sha(OPS/'smoke.json'),source_sha256=sources(),arms=list(SPECS),folds=FOLDS,new_trainings=20,external_test_deferred=True,gpus_available=free_gpus(),prior_review_sha256=sha(OPS/'previous_wave_review.json'))
    dump(OPS/'launch.json',info)
    with (OPS/'worker.log').open('x') as f:job=subprocess.Popen([PY,'-u',str(Path(__file__)),'worker'],cwd=REPO,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    info['worker_pid']=job.pid;dump(OPS/'launch.json',info);print(json.dumps({k:v for k,v in info.items() if k!='source_sha256'}))
def worker():
    info=read(OPS/'launch.json');root=Path(info['root']);tasks=[];failures=[]
    with (OPS/'worker.claim').open('x') as f:f.write(str(os.getpid()))
    try:
        for arm in SPECS:
            pending=[];active={}
            for fold in FOLDS:
                rr=root/arm/f'fold_{fold}';rr.mkdir(parents=True,exist_ok=False);cfg=config(arm,fold);cp=rr/'config.yaml';cp.write_text(yaml.safe_dump(cfg,sort_keys=False))
                t=dict(arm=arm,fold=fold,config=str(cp),run_root=str(rr),status='queued');pending.append(t);tasks.append(t)
            while pending or active:
                if not failures:
                    for gpu in free_gpus():
                        if not pending:break
                        if gpu in active:continue
                        t=pending.pop(0);t.update(status='running',gpu=gpu);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
                        log=open(Path(t['run_root'])/'train_and_audit.log','x');job=subprocess.Popen([PY,'-u',str(Path(__file__)),'trial','--config',t['config']],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[gpu]=(job,t,log)
                else:pending.clear()
                for gpu,(job,t,log) in list(active.items()):
                    if job.poll() is None:continue
                    log.close();t['ended_at']=utc()
                    if job.returncode:t['status']='failed';failures.append(dict(arm=t['arm'],fold=t['fold'],exit_code=job.returncode))
                    else:
                        receipt=Path(t['run_root'])/'trial_audit.json';assert read(receipt)['status']=='passed';t.update(status='audited',receipt_sha256=sha(receipt))
                    del active[gpu]
                dump(OPS/'wave_state.json',dict(status='running',root=str(root),tasks=tasks,failures=failures))
                if pending or active:time.sleep(20)
            assert not failures,str(failures)
            dump(OPS/f'{arm}_completed.json',dict(status='completed',arm=arm,root=str(root),ended_at=utc()))
        rows=[read(Path(t['run_root'])/'trial_audit.json') for t in tasks]
        result=dict(status='completed',scope='Grouped internal validation only; original model recipes retained',root=str(root),gitrev=rev(),trials=rows,external_test_deferred=True,review_required=True)
        dump(OPS/'result.json',result);dump(OPS/'terminal.json',dict(status='completed',root=str(root),gitrev=rev(),completed_at=utc(),result_sha256=sha(OPS/'result.json'),review_required=True))
    except BaseException:
        dump(OPS/'terminal.json',dict(status='failed',root=str(root),gitrev=rev(),error=traceback.format_exc(),failures=failures,automatic_retry=False,weights_retained=True));raise
if __name__=='__main__':
    assert sys.platform.startswith('linux') and str(REPO).startswith('/mnt/pvc/')
    os.chdir(REPO);p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','launch','worker','trial']);p.add_argument('--gpu',type=int,default=0);p.add_argument('--config');args=p.parse_args()
    if args.mode=='smoke':smoke(args.gpu)
    elif args.mode=='trial':trial(args.config)
    else:globals()[args.mode]()
