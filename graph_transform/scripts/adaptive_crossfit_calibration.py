"""Pod-only sequence-cross-fitted calibration diagnostics, no new backbone training."""
import argparse,datetime,hashlib,importlib.util,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'graph_transform/scripts'))
from adaptive_baseline_audit import sha,dump,independent_metrics
from adaptive_grouped_baselines import metrics as probability_metrics
PY='/mnt/pvc/graphtrans/DBond/.venv/bin/python'
OPS=Path('/mnt/pvc/graphtrans/dbond_adaptive_crossfit_calibration_wave4_20261103_ops')
BASE=Path('/mnt/pvc/graphtrans/dbond_adaptive_grouped_baselines_wave3_20261103_ops')
GT=Path('/mnt/pvc/graphtrans/dbond_adaptive_regularization_wave2_20261103_ops/completed_reviews/complete_wave2')
GTROOT=Path('/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/adaptive_research_batch/20261102_155846')
FOLDS=('1222','2252','3514','6072','9075')
MODELS=('gt_full','dbond_s','dbond_m','dbond_af','dbond_af_opt')
FAMILIES=('raw','binary_temperature','global_affine','condition_affine','global_affine_subset','condition_affine_subset')
THRESHOLDS=np.linspace(.30,.70,21)
PERFORMANCE=('bond_f1','bond_accuracy','bond_precision','bond_recall','example_precision','example_recall','example_f1_harmonic','example_f1_samples','example_jaccard','subset_acc','auc','ap','mcc','hard_brier','q_brier','q_seq','top10')
def read(p):return json.loads(Path(p).read_text())
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def rev():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
def source_sha():return {str(p.relative_to(REPO)):sha(p) for p in (Path(__file__),REPO/'graph_transform/scripts/adaptive_baseline_audit.py',REPO/'graph_transform/scripts/adaptive_grouped_baselines.py')}
def stats(x):return dict(mean=float(np.mean(x)),sample_sd=float(np.std(x,ddof=1)))
def difference(a,b):
    v=np.asarray(a)-np.asarray(b);return dict(**stats(v),per_fold=dict(zip(FOLDS,v.tolist())),wins=int((v>0).sum()),ties=int((v==0).sum()),losses=int((v<0).sum()))
def memberships(frame,fold):
    byseq={s:int(hashlib.sha256(('calibration42|'+fold+'|'+s).encode()).hexdigest()[:16],16)%5 for s in frame.seq.unique()}
    return np.array([byseq[s] for s in frame.seq],dtype=np.int8)
def hard(y,decision,mask):
    tp=((y==1)&decision&mask).sum(1);fp=((y==0)&decision&mask).sum(1);fn=((y==1)&~decision&mask).sum(1);tn=((y==0)&~decision&mask).sum(1)
    div=lambda a,b:np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b!=0)
    ep=float(div(tp,tp+fp).mean());er=float(div(tp,tp+fn).mean());t,f,n=int(tp.sum()),int(fp.sum()),int(fn.sum())
    return dict(bond_f1=2*t/(2*t+f+n) if 2*t+f+n else 0.,bond_accuracy=float((tp.sum()+tn.sum())/mask.sum()),bond_precision=t/(t+f) if t+f else 0.,bond_recall=t/(t+n) if t+n else 0.,example_precision=ep,example_recall=er,example_f1_harmonic=2*ep*er/(ep+er) if ep+er else 0.,example_f1_samples=float(div(2*tp,2*tp+fp+fn).mean()),example_jaccard=float(div(tp,tp+fp+fn).mean()),subset_acc=float(((fp+fn)==0).mean()))
def score(y,p,mask,frame,decision):
    m=probability_metrics(y,p,mask,frame);m.update(hard(y,decision,mask))
    # MCC is a hard-decision statistic, not a probability statistic.
    from sklearn.metrics import matthews_corrcoef
    m['mcc']=float(matthews_corrcoef(y[mask],decision[mask]));return {k:m[k] for k in PERFORMANCE}
def objective(w,x,y,penalty):
    z=x@w;value=float((np.logaddexp(0,z)-y*z).mean()+.5*np.sum(penalty*w*w))
    gradient=x.T@(expit(z)-y)/len(y)+penalty*w
    return value,gradient
def fit(p,y,mask,condition,kind):
    center=condition.mean(0);scale=condition.std(0);scale=np.where(scale>1e-12,scale,1.)
    z=np.log(np.clip(p,1e-6,1-1e-6))-np.log1p(-np.clip(p,1e-6,1-1e-6))
    parts=[z[mask]]
    if kind!='binary_temperature':parts.append(np.ones(int(mask.sum())))
    if kind=='condition_affine':
        normalized=(condition-center)/scale
        parts.extend([np.broadcast_to(normalized[:,j,None],p.shape)[mask] for j in range(2)])
    x=np.column_stack(parts);target=y[mask].astype(float);w=np.zeros(x.shape[1]);w[0]=1
    penalty=np.zeros_like(w)
    if kind=='condition_affine':penalty[2:]=.001
    bounds=[(.25,4.)]+[(-6.,6.)]*(len(w)-1)
    result=minimize(objective,w,args=(x,target,penalty),jac=True,method='L-BFGS-B',bounds=bounds,options={'maxiter':200,'ftol':1e-12,'gtol':1e-7})
    assert result.success and np.isfinite(result.fun) and np.isfinite(result.x).all(),str(result.message)
    projected=result.jac.copy()
    for j,(lo,hi) in enumerate(bounds):
        if result.x[j]<=lo+1e-10 and projected[j]>0:projected[j]=0
        if result.x[j]>=hi-1e-10 and projected[j]<0:projected[j]=0
    assert np.max(np.abs(projected))<1e-5,'Calibration optimizer KKT residual'
    return dict(kind=kind,coefficients=result.x.tolist(),condition_center=center.tolist(),condition_scale=scale.tolist(),objective=float(result.fun),converged=True,n_real_bonds=int(mask.sum()),optimizer_iterations=int(result.nit),projected_gradient_max=float(np.max(np.abs(projected))),regularization=penalty.tolist(),bounds=bounds)
def apply(p,condition,parameters):
    v=np.clip(p,1e-6,1-1e-6);z=np.log(v)-np.log1p(-v);w=parameters['coefficients'];out=w[0]*z
    if parameters['kind']!='binary_temperature':out=out+w[1]
    if parameters['kind']=='condition_affine':
        c=(condition-np.array(parameters['condition_center']))/np.array(parameters['condition_scale']);out=out+(c@np.array(w[2:]))[:,None]
    return expit(out)
def threshold(y,raw,calibrated,mask):
    control=hard(y,raw>.5,mask);options=[];table=[]
    for t in THRESHOLDS:
        m=hard(y,calibrated>t,mask);feasible=m['bond_f1']>=control['bond_f1']-.002 and m['example_f1_harmonic']>=control['example_f1_harmonic']-.002
        s=.4*m['bond_f1']+.4*m['example_f1_harmonic']+.2*m['subset_acc'];table.append(dict(threshold=float(t),feasible=bool(feasible),metrics=m))
        if feasible:options.append((m['subset_acc'],s,-abs(float(t)-.5),-float(t),float(t)))
    if not options:return dict(mode='raw_fallback',threshold=.5,training_control=control,threshold_table=table)
    return dict(mode='calibrated',threshold=max(options)[-1],training_control=control,threshold_table=table)
def input_data(model,fold):
    gt=read(GT/'result.json');base=read(BASE/'independent_audit/result.json')
    assert sha(GT/'result.json')=='12ff9ed645c62155e78c3d3a1663dde0f9c785c02209f7e682901c1df961c855'
    assert sha(BASE/'independent_audit/result.json')=='375b691786dc3eb0e88457a17d66d80ac456d413de48b16711954c8c776dda67'
    if model=='gt_full':
        proof=gt['provenance']['full']['folds'][fold];trial=GTROOT/'full'/f'fold_{fold}';assert sha(trial/'trial_audit.json')==proof['receipt_sha256'];trusted=proof['max_f1']
    else:
        proof=base['provenance'][model][fold];trial=Path(base['root'])/model/f'fold_{fold}';assert sha(trial/'trial_audit.json')==proof['trial_receipt_sha256'];trusted=proof['policies']['max_f1']
    a=read(trial/'trial_audit.json');r=a['policies']['max_f1'];assert sha(r['checkpoint'])==trusted['checkpoint_sha256']==r['checkpoint_sha256'];assert sha(r['prediction'])==trusted['prediction_sha256']==r['prediction_sha256']
    prepared=BASE/'prepared'/fold;sp=read(prepared/'split_receipt.json');framepath=prepared/'val_multi.csv';assert sha(framepath)==sp['prepared_sha256']['val_multi.csv']
    frame=pd.read_csv(framepath,na_filter=False)
    with np.load(r['prediction']) as z:y=z['y'];p=z['p'].astype(np.float64);mask=z['mask']
    assert y.shape==p.shape==mask.shape and len(y)==len(frame) and mask.dtype==bool
    assert np.array_equal(mask,np.arange(p.shape[1])[None,:]<(frame.seq.str.len().to_numpy()-1)[:,None])
    for i,s in enumerate(frame.true_multi):assert np.array_equal(y[i,mask[i]],np.fromstring(str(s),sep=';',dtype=np.int8)[:mask[i].sum()])
    condition=frame[['nce','charge']].to_numpy(dtype=float);assert np.isfinite(condition).all() and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return y,p,mask,frame,condition,dict(checkpoint_sha256=r['checkpoint_sha256'],prediction_sha256=r['prediction_sha256'],prediction=r['prediction'],checkpoint=r['checkpoint'],frame_sha256=sha(framepath),trial_receipt_sha256=sha(trial/'trial_audit.json'),split_receipt_sha256=sha(prepared/'split_receipt.json'))
def fixture():
    x=np.array([[1.,1.,.2],[2.,1.,-.3],[-1.,1.,.8]]);y=np.array([1.,0.,1.]);w=np.array([.9,.1,-.1]);penalty=np.array([0.,0.,.001]);_,g=objective(w,x,y,penalty)
    h=1e-6;numerical=np.array([(objective(w+np.eye(3)[j]*h,x,y,penalty)[0]-objective(w-np.eye(3)[j]*h,x,y,penalty)[0])/(2*h) for j in range(3)])
    assert np.max(np.abs(g-numerical))<1e-7
    yy=np.array([[1,0,0],[0,1,0]],np.int8);pp=np.array([[.8,.4,1],[.6,.7,1.]]) ;mm=np.array([[1,1,0],[1,1,0]],bool)
    a=hard(yy,pp>.5,mm);assert a['bond_f1']==.8 and a['subset_acc']==.5
    pp[:,2]=0;assert hard(yy,pp>.5,mm)==a
    par=dict(kind='binary_temperature',coefficients=[.6],condition_center=[0.,0.],condition_scale=[1.,1.]);assert np.array_equal(apply(pp,np.ones((2,2)),par)>.5,pp>.5)
def smoke():
    OPS.mkdir(exist_ok=True)
    with (OPS/'smoke.claim').open('x') as f:f.write(utc())
    fixture();bindings={}
    for fold in FOLDS:
        for model in MODELS:
            y,p,mask,frame,c,b=input_data(model,fold);g=memberships(frame,fold)
            assert len(set(g))==5
            for k in range(5):assert not set(frame.seq[g!=k])&set(frame.seq[g==k])
            m=score(y,p,mask,frame,p>.5)
            source=read(GT/'result.json')['per_fold']['full'][fold]['max_f1'] if model=='gt_full' else read(BASE/'independent_audit/result.json')['per_fold'][model][fold]['max_f1']
            assert max(abs(m[z]-source[z]) for z in PERFORMANCE)<3e-6
            bindings[model+'_'+fold]=b
    # A real bounded calibration optimization and full transform, with no assessment labels used.
    y,p,mask,frame,c,_=input_data('gt_full','1222');g=memberships(frame,'1222');train=g!=0;assess=g==0
    params=fit(p[train],y[train],mask[train],c[train],'condition_affine');pp=apply(p[assess],c[assess],params);assert pp.shape==p[assess].shape and np.isfinite(pp).all()
    dump(OPS/'smoke.json',dict(status='passed',gitrev=rev(),source_sha256=source_sha(),bindings=bindings,synthetic_gradient_and_padding=True,sequence_groups_disjoint=True,real_optimization_passed=True,external_test_deferred=True))
    print(json.dumps(dict(status='passed',gitrev=rev(),bound_predictions=len(bindings))))
def evaluate_one(model,fold,root):
    y,p,mask,frame,condition,binding=input_data(model,fold);groups=memberships(frame,fold);rr=root/model/f'fold_{fold}';rr.mkdir(parents=True,exist_ok=False)
    output={name:dict(p=np.zeros_like(p),decision=np.zeros_like(mask),threshold=np.zeros(len(y))) for name in FAMILIES}
    output['raw'].update(p=p.copy(),decision=(p>.5)&mask,threshold=np.full(len(y),.5));fits=[]
    for k in range(5):
        train=groups!=k;assess=groups==k;assert train.any() and assess.any()
        ts=sorted(set(frame.seq[train]));vs=sorted(set(frame.seq[assess]));assert not set(ts)&set(vs)
        item=dict(group=k,train_sequences_sha256=hashlib.sha256('\n'.join(ts).encode()).hexdigest(),assessment_sequences_sha256=hashlib.sha256('\n'.join(vs).encode()).hexdigest(),n_train_sequences=len(ts),n_assessment_sequences=len(vs),parameters={},thresholds={})
        for family in ('binary_temperature','global_affine','condition_affine'):
            parameters=fit(p[train],y[train],mask[train],condition[train],family);caltrain=apply(p[train],condition[train],parameters);calassess=apply(p[assess],condition[assess],parameters)
            output[family]['p'][assess]=calassess;output[family]['decision'][assess]=(calassess>.5)&mask[assess];output[family]['threshold'][assess]=.5;item['parameters'][family]=parameters
            if family!='binary_temperature':
                name=family+'_subset';rule=threshold(y[train],p[train],caltrain,mask[train]);item['thresholds'][name]=rule
                use=p[assess] if rule['mode']=='raw_fallback' else calassess
                output[name]['p'][assess]=use;output[name]['decision'][assess]=(use>rule['threshold'])&mask[assess];output[name]['threshold'][assess]=rule['threshold']
        fits.append(item)
    assert np.array_equal(output['binary_temperature']['decision'][mask],output['raw']['decision'][mask]),'Positive temperature must preserve hard decisions'
    records={}
    for family,item in output.items():
        path=rr/f'{family}.npz';np.savez_compressed(path,y=y,p=item['p'],mask=mask,decision=item['decision'],threshold=item['threshold'],groups=groups)
        records[family]=dict(prediction=str(path),prediction_sha256=sha(path),metrics=score(y,item['p'],mask,frame,item['decision']))
    dump(rr/'crossfit_parameters.json',dict(model=model,fold=fold,source=binding,group_count=5,fit_records=fits))
    dump(rr/'trial_result.json',dict(status='completed',model=model,fold=fold,source=binding,families=records,parameters_sha256=sha(rr/'crossfit_parameters.json'),limitations='Backbone epoch already selected on all inner validation sequences; crossfitting isolates calibration fitting only. Exploratory, not unbiased full-pipeline assessment.'))
    return dict(model=model,fold=fold,root=str(rr),receipt_sha256=sha(rr/'trial_result.json'))
def launch():
    smoke=read(OPS/'smoke.json');assert smoke['status']=='passed' and smoke['gitrev']==rev() and smoke['source_sha256']==source_sha()
    prior=read(OPS/'previous_wave_review.json');assert prior['status']=='audited_and_acknowledged' and prior['result_sha256']=='375b691786dc3eb0e88457a17d66d80ac456d413de48b16711954c8c776dda67'
    with (OPS/'launch.claim').open('x') as f:f.write(utc())
    root=Path('/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/adaptive_crossfit_calibration_batch')/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S');root.mkdir(parents=True,exist_ok=False)
    info=dict(status='submitted',root=str(root),source=str(REPO),gitrev=rev(),created_at=utc(),smoke_sha256=sha(OPS/'smoke.json'),source_sha256=source_sha(),models=MODELS,folds=FOLDS,families=FAMILIES,new_backbone_trainings=0,calibration_crossfit_groups=5,external_test_deferred=True,previous_review_sha256=sha(OPS/'previous_wave_review.json'))
    dump(OPS/'launch.json',info)
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (OPS/'worker.log').open('x') as f:job=subprocess.Popen([PY,'-u',str(Path(__file__)),'worker'],cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    info['worker_pid']=job.pid;dump(OPS/'launch.json',info);print(json.dumps({k:v for k,v in info.items() if k!='source_sha256'}))
def worker():
    os.nice(5);launch=read(OPS/'launch.json');root=Path(launch['root']);records=[];start=time.monotonic()
    with (OPS/'worker.claim').open('x') as f:f.write(str(os.getpid()))
    try:
        for fold in FOLDS:
            for model in MODELS:records.append(evaluate_one(model,fold,root))
            dump(OPS/f'fold_{fold}_completed.json',dict(status='completed',root=str(root),fold=fold))
        result=dict(status='completed',root=str(root),gitrev=rev(),records=records,wall_seconds=time.monotonic()-start,external_test_deferred=True,review_required=True,new_backbone_trainings=0)
        dump(OPS/'result.json',result);dump(OPS/'terminal.json',dict(status='completed',root=str(root),gitrev=rev(),completed_at=utc(),result_sha256=sha(OPS/'result.json'),review_required=True))
    except BaseException:
        dump(OPS/'terminal.json',dict(status='failed',root=str(root),gitrev=rev(),error=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
    assert sys.platform.startswith('linux') and str(REPO).startswith('/mnt/pvc/');os.chdir(REPO)
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['smoke','launch','worker']);args=parser.parse_args();globals()[args.mode]()
