"""Pod-only fixed count/location diagnostics on audited internal predictions.

No training, threshold fitting, model selection or external evaluation. The true-
count arm deliberately uses assessment labels and is diagnostic only.
"""
import argparse,datetime,hashlib,importlib.util,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[2]
FOLDS=('1222','2252','3514','6072','9075')
MODELS=('gt_full','dbond_s','dbond_m','dbond_af','dbond_af_opt')
METHODS=('raw','rounded_expected_count','oracle_true_count_NOT_DEPLOYABLE')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(p)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def rev():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def runtime(plan):
    ref=Path(plan['reference_ops']);info=read(ref/'launch.json');receipt=read(ref/'independent_audit/audit_receipt.json')
    assert receipt['status']=='passed' and sha(ref/'independent_audit/result.json')==receipt['result_sha256']==plan['reference_audit_result_sha256']
    assert sha(ref/'launch.json')==receipt['launch_sha256']==plan['reference_launch_sha256']
    source=Path(info['source'])
    for p,h in info['source_sha256'].items():assert sha(source/p)==h
    return load('audited_input_runtime',source/'graph_transform/scripts/adaptive_crossfit_calibration.py')
def select_count(p,mask,count):
    out=np.zeros_like(mask,dtype=bool)
    for row in range(len(p)):
        indices=np.flatnonzero(mask[row]);k=int(count[row]);assert 0<=k<=len(indices)
        order=np.argsort(-p[row,indices],kind='stable');out[row,indices[order[:k]]]=True
    return out
def decisions(y,p,mask):
    assert p.shape==y.shape==mask.shape and mask.dtype==bool and np.isfinite(p).all()
    n=mask.sum(1);assert (n>0).all()
    expected=np.floor(np.where(mask,p,0).sum(1)+.5).astype(int)
    expected=np.minimum(np.maximum(expected,0),n)
    truth=np.where(mask,y,0).sum(1).astype(int)
    return dict(raw=(p>.5)&mask,rounded_expected_count=select_count(p,mask,expected),oracle_true_count_NOT_DEPLOYABLE=select_count(p,mask,truth))
def stats(v):return dict(mean=float(np.mean(v)),sample_sd=float(np.std(v,ddof=1)))
def summarize_errors(y,p,mask,raw,oracle):
    true=np.where(mask,y,0).sum(1).astype(int);pred=raw.sum(1);n=mask.sum(1)
    hamming=((y!=raw)&mask).sum(1);exact=hamming==0;good_count=true==pred
    assert np.array_equal(hamming%2,np.abs(true-pred)%2)
    hist={str(int(k)):int(v) for k,v in zip(*np.unique(hamming,return_counts=True))}
    return dict(n_spectra=len(y),n_valid_bonds=int(mask.sum()),exact=int(exact.sum()),
      count_correct_location_wrong=int((good_count&~exact).sum()),over_count=int((pred>true).sum()),under_count=int((pred<true).sum()),
      raw_count_accuracy=float(good_count.mean()),mean_absolute_count_error=float(np.abs(pred-true).mean()),
      expected_count_mae=float(np.abs(np.where(mask,p,0).sum(1)-true).mean()),
      errors_per_spectrum=hist,one_error_fraction=float((hamming==1).mean()),two_error_fraction=float((hamming==2).mean()),
      oracle_rank_recoverable_fraction=float(np.all((y==oracle)|~mask,axis=1).mean()),
      mean_true_positive_fraction=float((true/n).mean()))
def grouped_diagnostics(y,p,mask,frame,ds):
    n=mask.sum(1);length=n+1
    keys={'length_bin':pd.cut(length,[0,10,15,20,30,np.inf],right=True).astype(str),
          'charge':frame.charge.astype(str).to_numpy(),'nce':frame.nce.astype(str).to_numpy()}
    result={}
    for name,values in keys.items():
        result[name]={}
        for val in sorted(set(values)):
            idx=np.asarray(values)==val
            result[name][str(val)]=summarize_errors(y[idx],p[idx],mask[idx],ds['raw'][idx],ds['oracle_true_count_NOT_DEPLOYABLE'][idx])
    return result
def repeated_labels(y,mask,frame):
    # Grouping is descriptive; it is not proof that all physical inputs match.
    columns=[k for k in ('seq','charge','nce','pep_mass') if k in frame.columns]
    patterns=[''.join(str(int(z)) for z in y[i,mask[i]]) for i in range(len(y))]
    tab=frame[columns].copy();tab['pattern']=patterns
    repeated_groups=0;rows=0;conflicting=0;modal_correct=0;pair_total=0;pair_same=0
    for _,g in tab.groupby(columns,dropna=False,sort=False):
        if len(g)<2:continue
        counts=g.pattern.value_counts().to_numpy();repeated_groups+=1;rows+=len(g);conflicting+=int(len(counts)>1);modal_correct+=int(counts.max())
        pair_total+=len(g)*(len(g)-1)//2;pair_same+=int(sum(int(c)*(int(c)-1)//2 for c in counts))
    return dict(grouping_columns=columns,repeated_groups=repeated_groups,repeated_rows=rows,conflicting_groups=conflicting,
      empirical_modal_pattern_agreement=float(modal_correct/rows) if rows else None,
      pairwise_pattern_agreement=float(pair_same/pair_total) if pair_total else None,
      interpretation='Descriptive repeated-label variability, NOT a universal performance ceiling; unobserved acquisition factors may differ.')
def one(rt,model,fold,root):
    y,p,mask,frame,condition,binding=rt.input_data(model,fold);ds=decisions(y,p,mask)
    rr=root/model/f'fold_{fold}';rr.mkdir(parents=True,exist_ok=False)
    path=rr/'decisions.npz';np.savez_compressed(path,y=y,p=p,mask=mask,**ds)
    metrics={};raw=rt.score(y,p,mask,frame,ds['raw'])
    from sklearn.metrics import matthews_corrcoef
    for name,d in ds.items():
        m=raw.copy();m.update(rt.hard(y,d,mask));m['mcc']=float(matthews_corrcoef(y[mask],d[mask]));metrics[name]=m
    out=dict(status='completed',model=model,fold=fold,source=binding,metrics=metrics,
      errors=summarize_errors(y,p,mask,ds['raw'],ds[METHODS[2]]),groups=grouped_diagnostics(y,p,mask,frame,ds),
      repeat_label_diagnostic=repeated_labels(y,mask,frame),prediction=str(path),prediction_sha256=sha(path),
      no_fit=True,oracle_label_access=True,external_test_evaluated=False)
    dump(rr/'trial_result.json',out)
    return dict(model=model,fold=fold,trial_result=str(rr/'trial_result.json'),sha256=sha(rr/'trial_result.json'))
def smoke(plan):
    ops=Path(plan['ops']);ops.mkdir(parents=True,exist_ok=True);assert not (ops/'smoke.json').exists()
    rt=runtime(plan)
    y=np.array([[0,1,0,1],[1,1,0,0],[0,0,0,0]]);m=np.array([[1,1,1,1],[1,1,1,0],[1,1,0,0]],bool)
    p=np.array([[.7,.8,.1,.9],[.9,.9,.1,.99],[.1,.1,.99,.99]])
    ds=decisions(y,p,m);assert np.array_equal(ds[METHODS[2]].sum(1),(y*m).sum(1));assert not ds[METHODS[1]][~m].any()
    tie=select_count(np.ones((1,3)),np.ones((1,3),bool),np.array([1]));assert np.array_equal(tie,[[True,False,False]])
    assert not select_count(p,m,np.zeros(3,int)).any() and np.array_equal(select_count(p,m,m.sum(1)),m)
    bindings={}
    for model in MODELS:
        for fold in FOLDS:
            yy,pp,mm,fr,_,b=rt.input_data(model,fold);d=decisions(yy,pp,mm)
            assert np.array_equal(mm,np.arange(mm.shape[1])[None,:]<(fr.seq.str.len().to_numpy()-1)[:,None])
            assert np.array_equal(d[METHODS[2]].sum(1),(yy*mm).sum(1))
            bindings[model+'_'+fold]=b
    dump(ops/'smoke.json',dict(status='passed',gitrev=rev(),source_sha256=sha(__file__),plan_sha256=sha(ARGS.plan),bindings=bindings,synthetic_tie_padding_zero_all_counts=True,new_trainings=0))
    print(json.dumps(dict(status='passed',bound_inputs=len(bindings))))
def launch(plan):
    ops=Path(plan['ops']);sm=read(ops/'smoke.json');assert sm['status']=='passed' and sm['gitrev']==rev() and sm['source_sha256']==sha(__file__) and sm['plan_sha256']==sha(ARGS.plan)
    with (ops/'launch.claim').open('x') as f:f.write(utc())
    root=Path(plan['output_parent'])/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S');root.mkdir(parents=True,exist_ok=False)
    info=dict(status='submitted',root=str(root),source=str(REPO),gitrev=rev(),created_at=utc(),smoke_sha256=sha(ops/'smoke.json'),source_sha256=sha(__file__),plan_sha256=sha(ARGS.plan),new_trainings=0,new_backbone_inferences=0,methods=METHODS)
    dump(ops/'launch.json',info);env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (ops/'worker.log').open('x') as f:
        job=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'worker','--plan',str(Path(ARGS.plan).resolve())],cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    info['worker_pid']=job.pid;dump(ops/'launch.json',info);print(json.dumps(info))
def worker(plan):
    os.nice(5);ops=Path(plan['ops']);info=read(ops/'launch.json');root=Path(info['root']);start=time.monotonic()
    with (ops/'worker.claim').open('x') as f:f.write(str(os.getpid()))
    try:
        assert info['gitrev']==rev() and info['source_sha256']==sha(__file__) and info['plan_sha256']==sha(ARGS.plan)
        rt=runtime(plan);records=[one(rt,model,fold,root) for fold in FOLDS for model in MODELS]
        dump(ops/'result.json',dict(status='completed',root=str(root),gitrev=rev(),records=records,wall_seconds=time.monotonic()-start,new_trainings=0,review_required=True))
        dump(ops/'terminal.json',dict(status='completed',root=str(root),gitrev=rev(),completed_at=utc(),result_sha256=sha(ops/'result.json'),review_required=True))
    except BaseException:
        dump(ops/'terminal.json',dict(status='failed',root=str(root),gitrev=rev(),error=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
    assert sys.platform.startswith('linux') and str(REPO).startswith('/mnt/pvc/')
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['smoke','launch','worker']);parser.add_argument('--plan',required=True);ARGS=parser.parse_args();os.chdir(REPO);globals()[ARGS.mode](read(ARGS.plan))
