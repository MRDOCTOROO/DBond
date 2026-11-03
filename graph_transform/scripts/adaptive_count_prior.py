"""Train-only conditional cardinality model plus fixed n-choose-k MAP decoding.

The backbone probabilities remain unchanged and are reported separately from
joint MAP decisions. No validation labels enter fit(), no validation-based early
stopping or hyperparameter search. One count model per fold shared by 5 backbones.
"""
import argparse,datetime,hashlib,importlib.util,itertools,json,os,subprocess,sys,time,traceback
from pathlib import Path
import joblib,numpy as np,pandas as pd,sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef
REPO=Path(__file__).resolve().parents[2]
FOLDS=('1222','2252','3514','6072','9075');SEEDS=(949822381,2775888976,1365222152,612878555,1785808201)
MODELS=('gt_full','dbond_s','dbond_m','dbond_af','dbond_af_opt');AA='ACDEFGHIKLMNPQRSTVWY?'
CONFIG=dict(max_iter=60,max_leaf_nodes=7,learning_rate=.1,l2_regularization=1.,min_samples_leaf=20,early_stopping=False)
FEATURES=['length','charge','nce','pep_mass']+['fraction_'+a for a in AA]+[f'{pos}_{a}' for pos in ('N1','N2','C2','C1') for a in AA]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
    p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False));t.replace(p)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def rev():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def runtime(plan):
    ref=Path(plan['input_reference_ops']);info=read(ref/'launch.json');a=read(ref/'independent_audit/audit_receipt.json')
    assert a['status']=='passed' and sha(ref/'independent_audit/result.json')==a['result_sha256']==plan['input_reference_result_sha256']
    for p,h in info['source_sha256'].items():assert sha(Path(info['source'])/p)==h
    prior=Path(plan['previous_ops']);assert read(prior/'independent_audit/state.json')['status']=='passed'
    assert sha(prior/'independent_audit/result.json')==plan['previous_result_sha256']
    return load('audited_cardinality_inputs',Path(info['source'])/'graph_transform/scripts/adaptive_crossfit_calibration.py')
def seq_feature(seq):
    ids=[AA.index(c) if c in AA else 20 for c in seq];n=len(ids);assert 2<=n<=36
    count=np.bincount(ids,minlength=21)/n;terminal=np.zeros((4,21));terminal[np.arange(4),np.array([ids[0],ids[1],ids[-2],ids[-1]])]=1
    return np.r_[n,count,terminal.ravel()]
def features(frame):
    unique={s:seq_feature(str(s)) for s in frame.seq.unique()};x=np.vstack([unique[s] for s in frame.seq]);c=frame[['charge','nce','pep_mass']].to_numpy(dtype=float)
    x=np.c_[x[:,:1],c,x[:,1:]];assert x.shape==(len(frame),len(FEATURES)) and np.isfinite(x).all();return x
def labels(frame):
    counts=[]
    for seq,s in zip(frame.seq,frame.true_multi):
        v=np.fromstring(str(s),sep=';',dtype=np.int8);n=len(seq)-1;assert len(v)>=n and np.isin(v[:n],[0,1]).all();counts.append(int(v[:n].sum()))
    return np.array(counts,dtype=np.int64)
def training(rt,fold):
    directory=rt.BASE/'prepared'/fold;split=read(directory/'split_receipt.json');path=directory/'train_multi.csv';assert sha(path)==split['prepared_sha256']['train_multi.csv']
    frame=pd.read_csv(path,na_filter=False);assert set(frame.seq)==set(split['train_sequences']) and not set(frame.seq)&set(split['validation_sequences'])
    assert sha(directory/'val_multi.csv')==split['prepared_sha256']['val_multi.csv']
    x=features(frame);y=labels(frame);sizes=frame.groupby(['seq','charge','nce','pep_mass'],dropna=False).seq.transform('size').to_numpy();w=1/sizes.astype(float)
    proof=dict(split_receipt_sha256=sha(directory/'split_receipt.json'),train_frame_sha256=sha(path),validation_frame_sha256=sha(directory/'val_multi.csv'),train_sequence_sha256=hashlib.sha256('\n'.join(sorted(set(frame.seq))).encode()).hexdigest(),n_train_rows=len(frame),n_train_sequences=frame.seq.nunique(),sample_weight='inverse number of original spectra per seq/charge/nce/pep_mass',feature_names=FEATURES,target='valid L-1 hard-label cleavage count',training_seed=split['training_seed'],sequence_overlap=0)
    return x,y,w,proof
def estimator(seed,iterations=None):
    cfg=CONFIG.copy()
    if iterations is not None:cfg['max_iter']=iterations
    return HistGradientBoostingClassifier(**cfg,random_state=seed)
def prior_probability(model,x,n):
    raw=model.predict_proba(x);out=np.full((len(x),36),1e-6,dtype=np.float64);out[:,model.classes_.astype(int)]+=raw
    out[np.arange(36)[None,:]>n[:,None]]=0;out/=out.sum(1,keepdims=True);assert np.isfinite(out).all();return out
def joint_map(p,mask,prior):
    """log pi(k|x) + log P_ind(y_topk|p) - log P_ind(K=k|p)."""
    out=np.zeros_like(mask,bool);counts=np.zeros(len(p),int)
    for n in np.unique(mask.sum(1)):
        idx=np.flatnonzero(mask.sum(1)==n);v=np.clip(p[idx,:n],1e-6,1-1e-6);lp=np.log(v);lq=np.log1p(-v)
        table=np.full((len(idx),n+1),-np.inf);table[:,0]=0
        for j in range(n):
            prev=table.copy();table[:,0]=prev[:,0]+lq[:,j]
            table[:,1:j+2]=np.logaddexp(prev[:,1:j+2]+lq[:,j,None],prev[:,:j+1]+lp[:,j,None])
        theta=lp-lq;order=np.argsort(-theta,axis=1,kind='stable');prefix=np.c_[np.zeros(len(idx)),np.cumsum(np.take_along_axis(theta,order,axis=1),axis=1)]
        scores=np.log(prior[idx,:n+1])+lq.sum(1)[:,None]+prefix-table
        k=np.argmax(scores,axis=1);selected=np.arange(n)[None,:]<k[:,None];rows=np.broadcast_to(idx[:,None],order.shape);out[rows,order]=selected;counts[idx]=k
        assert np.max(np.abs(np.exp(table).sum(1)-1))<1e-12
    return out,counts
def brute_fixture():
    p=np.array([[.13,.48,.72,.91]]);m=np.ones_like(p,bool);prior=np.array([[.12,.26,.34,.18,.1]+[0.]*31]);d,k=joint_map(p,m,prior)
    patterns=np.array(list(itertools.product([0,1],repeat=4)));bernoulli=np.prod(np.where(patterns,p,1-p),axis=1);counts=patterns.sum(1);normal=np.bincount(counts,weights=bernoulli,minlength=5);score=bernoulli*prior[0,counts]/normal[counts]
    assert abs(score.max()-score[np.flatnonzero(np.all(patterns==d[0],axis=1))[0]])<1e-12
    b=np.zeros((1,36));b[0,:5]=normal;assert np.array_equal(joint_map(p,m,b)[0],p>.5)
    uniform=np.ones((1,36))/36;bad=np.c_[p,np.ones((1,3))];mm=np.c_[m,np.zeros((1,3),bool)]
    assert np.array_equal(joint_map(bad,mm,uniform)[0][:,:4],joint_map(p,m,uniform)[0]) and not joint_map(bad,mm,uniform)[0][:,4:].any()
def smoke(plan):
    ops=Path(plan['ops']);ops.mkdir(exist_ok=True);assert not (ops/'smoke.json').exists();brute_fixture();rt=runtime(plan);bindings={}
    for fold,seed in zip(FOLDS,SEEDS):
        x,y,w,proof=training(rt,fold);assert proof['training_seed']==seed
        for model in MODELS:
            yy,p,mask,frame,_,binding=rt.input_data(model,fold);assert np.array_equal(labels(frame),(yy*mask).sum(1));assert features(frame).shape[1]==x.shape[1];bindings[model+'_'+fold]=binding
        del x,y,w
    # Synthetic estimator/serialization only; no production count model is fit in smoke.
    toy=pd.DataFrame(dict(seq=['ACDEFG','KLMNPQ','RSTVWY','AAAAAA']*12,charge=[2,3,4,5]*12,nce=[20,30,40,50]*12,pep_mass=[600.,700.,800.,900.]*12))
    x=features(toy);model=estimator(42,2).fit(x,np.tile([1,2,3,4],12));path=ops/'synthetic_roundtrip.joblib';joblib.dump(model,path);assert np.array_equal(model.predict_proba(x),joblib.load(path).predict_proba(x))
    dump(ops/'smoke.json',dict(status='passed',gitrev=rev(),source_sha256=sha(__file__),plan_sha256=sha(ARGS.plan),bound_inputs=bindings,independent_bruteforce_MAP=True,poisson_binomial_recovers_raw=True,padding_invariant=True,synthetic_estimator_CPU_roundtrip=True,training_validation_sequence_overlap=0,production_count_models_trained=0))
    print(json.dumps(dict(status='passed',bound_inputs=len(bindings))))
def fold_worker(rt,fold,seed,root):
    rr=root/f'fold_{fold}';rr.mkdir(exist_ok=False);x,y,w,proof=training(rt,fold);t=time.monotonic();model=estimator(seed).fit(x,y,sample_weight=w);fit_seconds=time.monotonic()-t
    cp=rr/'count_model.joblib';joblib.dump(model,cp);restored=joblib.load(cp);assert np.array_equal(model.predict_proba(x[:256]),restored.predict_proba(x[:256]));del model,x,y,w
    records={}
    for name in MODELS:
        yy,p,mask,frame,_,binding=rt.input_data(name,fold);n=mask.sum(1);xx=features(frame);prior=prior_probability(restored,xx,n);decision,k=joint_map(p,mask,prior)
        raw=(p>.5)&mask;prob=rt.score(yy,p,mask,frame,raw);metrics={}
        for key,d in [('raw',raw),('count_prior_joint_MAP',decision)]:
            v=prob.copy();v.update(rt.hard(yy,d,mask));v['mcc']=float(matthews_corrcoef(yy[mask],d[mask]));metrics[key]=v
        path=rr/f'{name}.npz';np.savez_compressed(path,y=yy,p=p,mask=mask,raw=raw,decision=decision,prior=prior,chosen_count=k)
        true=(yy*mask).sum(1).astype(int);modal=np.argmax(prior,axis=1)
        records[name]=dict(source=binding,prediction=str(path),prediction_sha256=sha(path),metrics=metrics,count_mode_accuracy=float(np.mean(modal==true)),count_nll=float(-np.log(prior[np.arange(len(true)),true]).mean()),chosen_count_accuracy=float(np.mean(k==true)),changed_spectrum_fraction=float(np.any(decision!=raw,axis=1).mean()))
    dump(rr/'trial_result.json',dict(status='completed',fold=fold,training=proof,configuration=CONFIG,feature_names=FEATURES,count_checkpoint=str(cp),count_checkpoint_sha256=sha(cp),classes=restored.classes_.tolist(),sklearn_version=sklearn.__version__,numpy_version=np.__version__,fit_seconds=fit_seconds,backbone_training=False,new_count_model_trainings=1,records=records))
    return dict(fold=fold,trial_result=str(rr/'trial_result.json'),sha256=sha(rr/'trial_result.json'))
def launch(plan):
    ops=Path(plan['ops']);sm=read(ops/'smoke.json');assert sm['status']=='passed' and sm['gitrev']==rev() and sm['source_sha256']==sha(__file__) and sm['plan_sha256']==sha(ARGS.plan)
    with (ops/'launch.claim').open('x') as f:f.write(utc())
    root=Path(plan['output_parent'])/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S');root.mkdir(parents=True,exist_ok=False)
    info=dict(status='submitted',root=str(root),source=str(REPO),gitrev=rev(),created_at=utc(),source_sha256=sha(__file__),plan_sha256=sha(ARGS.plan),smoke_sha256=sha(ops/'smoke.json'),new_backbone_trainings=0,new_count_model_trainings=5,models=MODELS,folds=FOLDS,configuration=CONFIG,selection='fixed recipe; no validation early stopping, thresholds or hyperparameter selection')
    dump(ops/'launch.json',info);env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (ops/'worker.log').open('x') as f:p=subprocess.Popen([sys.executable,'-u',__file__,'worker','--plan',str(Path(ARGS.plan).resolve())],cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    info['worker_pid']=p.pid;dump(ops/'launch.json',info);print(json.dumps(info))
def worker(plan):
    os.nice(5);ops=Path(plan['ops']);launch=read(ops/'launch.json');root=Path(launch['root']);t=time.monotonic()
    with (ops/'worker.claim').open('x') as f:f.write(str(os.getpid()))
    try:
        assert launch['gitrev']==rev() and launch['source_sha256']==sha(__file__) and launch['plan_sha256']==sha(ARGS.plan)
        rt=runtime(plan);records=[fold_worker(rt,fold,seed,root) for fold,seed in zip(FOLDS,SEEDS)]
        dump(ops/'result.json',dict(status='completed',root=str(root),gitrev=rev(),records=records,wall_seconds=time.monotonic()-t,external_test_evaluated=False,review_required=True,new_backbone_trainings=0,new_count_model_trainings=5))
        dump(ops/'terminal.json',dict(status='completed',root=str(root),gitrev=rev(),completed_at=utc(),result_sha256=sha(ops/'result.json'),review_required=True))
    except BaseException:
        dump(ops/'terminal.json',dict(status='failed',root=str(root),gitrev=rev(),error=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
    assert sys.platform.startswith('linux') and str(REPO).startswith('/mnt/pvc/')
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['smoke','launch','worker']);parser.add_argument('--plan',required=True);ARGS=parser.parse_args();os.chdir(REPO);globals()[ARGS.mode](read(ARGS.plan))
