"""Pod-only descriptive condition/label/target diagnostics; never fits or selects."""
import argparse,collections,datetime,hashlib,importlib.util,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np,pandas as pd
REPO=Path(__file__).resolve().parents[2]
FOLDS=('1222','2252','3514','6072','9075');MODELS=('gt_full','dbond_s','dbond_m','dbond_af','dbond_af_opt')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
    p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False));t.replace(p)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def rev():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def runtime(plan):
    ref=Path(plan['input_reference_ops']);info=read(ref/'launch.json')
    assert sha(ref/'independent_audit/result.json')==plan['input_reference_result_sha256']
    for path,h in info['source_sha256'].items():assert sha(Path(info['source'])/path)==h
    prev=Path(plan['previous_ops']);assert read(prev/'independent_audit/state.json')['status']=='passed' and sha(prev/'independent_audit/result.json')==plan['previous_result_sha256']
    return load('audited_condition_inputs',Path(info['source'])/'graph_transform/scripts/adaptive_crossfit_calibration.py')
def matrix(frame):
    y=np.zeros((len(frame),35),np.int8);mask=np.arange(35)[None,:]<(frame.seq.str.len().to_numpy()-1)[:,None]
    assert (frame.seq.str.len().between(2,36)).all()
    for i,(seq,s) in enumerate(zip(frame.seq,frame.true_multi)):
        n=len(seq)-1;v=np.fromstring(str(s),sep=';',dtype=np.int8);assert len(v)>=n and np.isin(v[:n],[0,1]).all();y[i,:n]=v[:n]
    return y,mask
def key(seq,charge,nce):return str(seq),int(charge),float(nce)
def groups(frame,y):
    result=[]
    for k,indices in frame.groupby(['seq','charge','nce'],dropna=False,sort=True).indices.items():
        kk=key(*k);idx=np.asarray(indices);n=len(kk[0])-1;values=y[idx,:n];hist=collections.Counter(''.join(map(str,row)) for row in values.tolist());count_hist=collections.Counter(map(int,values.sum(1)))
        result.append(dict(key=kk,indices=idx,n_bonds=n,rows=len(idx),sum_y=values.sum(0,dtype=np.int64),q=values.mean(0),pattern_mode_count=max(hist.values()),count_mode_count=max(count_hist.values()),zero_rows=int((values.sum(1)==0).sum()),mass_range=float(np.ptp(frame.pep_mass.to_numpy()[idx]))))
    return result
def summarize(gs):
    rows=sum(g['rows'] for g in gs);bonds=sum(g['rows']*g['n_bonds'] for g in gs)
    return dict(rows=rows,condition_groups=len(gs),sequences=len({g['key'][0] for g in gs}),valid_bonds=bonds,
      empirical_label_variance=float(sum(g['rows']*np.sum(g['q']*(1-g['q'])) for g in gs)/bonds),
      empirical_minority_bond_fraction=float(sum(g['rows']*np.minimum(g['q'],1-g['q']).sum() for g in gs)/bonds),
      empirical_modal_pattern_agreement=float(sum(g['pattern_mode_count'] for g in gs)/rows),empirical_modal_count_agreement=float(sum(g['count_mode_count'] for g in gs)/rows),
      zero_spectrum_fraction=float(sum(g['zero_rows'] for g in gs)/rows),single_observation_group_fraction=float(np.mean([g['rows']==1 for g in gs])),
      groups_with_mass_variation=sum(g['mass_range']>1e-6 for g in gs),max_mass_range=max(g['mass_range'] for g in gs))
def partition(frame,y):
    gs=groups(frame,y)
    return gs,dict(all=summarize(gs),nce={str(v):summarize([g for g in gs if g['key'][2]==v]) for v in sorted({g['key'][2] for g in gs})},charge={str(v):summarize([g for g in gs if g['key'][1]==v]) for v in sorted({g['key'][1] for g in gs})})
def moments(y,p,gs):
    def one(selected):
        sums=np.zeros(6);bonds=0;maxspread=0.
        for g in selected:
            idx=g['indices'];n=g['n_bonds'];v=p[idx,:n];truth=y[idx,:n];avg=v.mean(0);q=g['q'];m=g['rows'];bonds+=m*n
            sums+=np.array([np.square(v-truth).sum(),m*np.sum(q*(1-q)),m*np.square(avg-q).sum(),np.square(v-avg).sum(),-2*np.sum((v-avg)*(truth-q)),np.sum(v-truth)])
            maxspread=max(maxspread,float(np.max(np.ptp(v,axis=0))))
        terms=sums/bonds;assert abs(terms[0]-sum(terms[1:5]))<2e-12
        return dict(zip(('hard_brier','empirical_label_variance','group_mean_error','within_group_prediction_variance','within_group_cross_term','mean_probability_bias'),map(float,terms)),valid_bonds=bonds,max_duplicate_probability_range=maxspread)
    return dict(all=one(gs),nce={str(v):one([g for g in gs if g['key'][2]==v]) for v in sorted({g['key'][2] for g in gs})})
def inputs(rt,fold):
    directory=rt.BASE/'prepared'/fold;sp=read(directory/'split_receipt.json');tp=directory/'train_multi.csv';vp=directory/'val_multi.csv'
    assert sha(tp)==sp['prepared_sha256']['train_multi.csv'] and sha(vp)==sp['prepared_sha256']['val_multi.csv']
    train=pd.read_csv(tp,na_filter=False);val=pd.read_csv(vp,na_filter=False)
    assert set(train.seq)==set(sp['train_sequences']) and set(val.seq)==set(sp['validation_sequences']) and not set(train.seq)&set(val.seq)
    return train,val,dict(train_sha256=sha(tp),val_sha256=sha(vp),split_receipt_sha256=sha(directory/'split_receipt.json'),sequence_overlap=0)
def check_targets(plan,fold,train_groups,val_groups,val):
    path=Path(plan['manifest_template'].format(fold=fold));manifest=read(path);assert manifest['status']=='passed' and manifest['states']==['none','b_only','y_only','both']
    side=path.parent/manifest['sidecar']['path'];assert sha(side)==manifest['sidecar']['sha256']
    foldedpath=Path(manifest['source_folded_train']['path']);original=Path(manifest['source_original_train']['path']);assert sha(foldedpath)==manifest['source_folded_train']['sha256'] and sha(original)==manifest['source_original_train']['sha256']
    records={key(r['seq'],r['charge'],r['nce']):r for r in read(side)['records']};folded=pd.read_csv(foldedpath,na_filter=False)
    target={key(r.seq,r.charge,r.nce):r for r in folded.itertuples()};allgroups=train_groups+val_groups
    assert len(records)==len(target)==len(allgroups) and set(records)==set(target)=={g['key'] for g in allgroups}
    round_error=0.;soft_error=0.
    for g in allgroups:
        record=records[g['key']];counts=np.asarray(record['counts']);n=g['rows'];q=g['q'];t=target[g['key']];sq=np.fromstring(str(t.soft_multi),sep=';')
        assert record['n']==int(t.group_n)==n and counts.shape==(g['n_bonds'],4) and (counts.sum(1)==n).all()
        assert np.array_equal(n-counts[:,0],g['sum_y'])
        err=float(np.max(np.abs(q-sq)));round_error=max(round_error,err);assert err<=5.0001e-5
    if 'soft_multi' in val:
        for row in val.itertuples():
            sq=np.fromstring(str(row.soft_multi),sep=';')[:len(row.seq)-1];tq=np.fromstring(str(target[key(row.seq,row.charge,row.nce)].soft_multi),sep=';');soft_error=max(soft_error,float(np.max(np.abs(sq-tq))))
        assert soft_error<=5.0001e-5
    return dict(status='passed',manifest_sha256=sha(path),sidecar_sha256=sha(side),source_folded_sha256=sha(foldedpath),source_original_sha256=sha(original),groups=len(allgroups),max_raw_frequency_vs_serialized_q_error=round_error,max_val_soft_vs_folded_q_error=soft_error,existing_rounding_tolerance=5.0001e-5)
def one(rt,plan,fold,root):
    train,val,binding=inputs(rt,fold);ty,tm=matrix(train);vy,vm=matrix(val);tg,ts=partition(train,ty);vg,vs=partition(val,vy);target=check_targets(plan,fold,tg,vg,val)
    models={};proof={}
    for model in MODELS:
        y,p,mask,frame,_,b=rt.input_data(model,fold)
        expected=np.arange(mask.shape[1])[None,:]<(frame.seq.str.len().to_numpy()-1)[:,None]
        assert np.array_equal(mask,expected) and np.array_equal(mask.sum(1),vm.sum(1)) and np.array_equal(y[mask],vy[vm]) and b['frame_sha256']==binding['val_sha256']
        models[model]=moments(y,p,vg);proof[model]=b
    path=root/f'fold_{fold}.json';dump(path,dict(status='completed',fold=fold,input=binding,target_consistency=target,train=ts,validation=vs,model_moments=models,model_sources=proof,interpretation='Group frequencies and modal agreements are descriptive in-sample label statistics, NOT universal Bayes ceilings or deployable scores; grouping omits some physical acquisition factors.'))
    return dict(fold=fold,result=str(path),sha256=sha(path))
def smoke(plan):
    ops=Path(plan['ops']);ops.mkdir(exist_ok=True);assert not (ops/'smoke.json').exists();rt=runtime(plan);bindings={}
    toy=pd.DataFrame(dict(seq=['ACD']*3,charge=[2]*3,nce=[20]*3,pep_mass=[300.]*3,true_multi=['1;0','0;1','1;1']))
    y,m=matrix(toy);gs,s=partition(toy,y);p=np.zeros_like(y,dtype=float);p[:,:2]=[[.6,.4],[.61,.39],[.59,.41]];mom=moments(y,p,gs)
    assert abs(s['all']['empirical_modal_pattern_agreement']-1/3)<1e-12 and abs(mom['all']['hard_brier']-sum(mom['all'][k] for k in ('empirical_label_variance','group_mean_error','within_group_prediction_variance','within_group_cross_term')))<1e-12
    for fold in FOLDS:
        train,val,b=inputs(rt,fold);bindings[fold]=b
        assert set(train.nce)<=set((20,30,40,50)) and set(val.nce)<=set((20,30,40,50))
        truth,valid=matrix(val)
        for name in MODELS:
            yy,pp,mm,fr,_,proof=rt.input_data(name,fold)
            assert np.array_equal(mm,np.arange(mm.shape[1])[None,:]<(fr.seq.str.len().to_numpy()-1)[:,None]) and np.array_equal(mm.sum(1),valid.sum(1)) and np.array_equal(yy[mm],truth[valid]) and proof['frame_sha256']==b['val_sha256']
    dump(ops/'smoke.json',dict(status='passed',gitrev=rev(),source_sha256=sha(__file__),plan_sha256=sha(ARGS.plan),bindings=bindings,sequence_isolation=True,variance_decomposition_fixture=True,no_model_fitting=True))
    print(json.dumps(dict(status='passed',bound_folds=len(bindings))))
def launch(plan):
    ops=Path(plan['ops']);s=read(ops/'smoke.json');assert s['status']=='passed' and s['gitrev']==rev() and s['source_sha256']==sha(__file__) and s['plan_sha256']==sha(ARGS.plan)
    with (ops/'launch.claim').open('x') as f:f.write(utc())
    root=Path(plan['output_parent'])/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S');root.mkdir(parents=True,exist_ok=False)
    info=dict(status='submitted',root=str(root),source=str(REPO),gitrev=rev(),created_at=utc(),source_sha256=sha(__file__),plan_sha256=sha(ARGS.plan),smoke_sha256=sha(ops/'smoke.json'),new_trainings=0,new_backbone_inferences=0,scope='descriptive condition/label/target audit only')
    dump(ops/'launch.json',info);env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (ops/'worker.log').open('x') as f:p=subprocess.Popen([sys.executable,'-u',__file__,'worker','--plan',str(Path(ARGS.plan).resolve())],cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    info['worker_pid']=p.pid;dump(ops/'launch.json',info);print(json.dumps(info))
def worker(plan):
    os.nice(5);ops=Path(plan['ops']);info=read(ops/'launch.json');root=Path(info['root']);t=time.monotonic()
    with (ops/'worker.claim').open('x') as f:f.write(str(os.getpid()))
    try:
        assert info['gitrev']==rev() and info['source_sha256']==sha(__file__) and info['plan_sha256']==sha(ARGS.plan)
        rt=runtime(plan);records=[one(rt,plan,f,root) for f in FOLDS]
        dump(ops/'result.json',dict(status='completed',root=str(root),gitrev=rev(),records=records,new_trainings=0,external_test_evaluated=False,wall_seconds=time.monotonic()-t,review_required=True))
        dump(ops/'terminal.json',dict(status='completed',root=str(root),gitrev=rev(),completed_at=utc(),result_sha256=sha(ops/'result.json'),review_required=True))
    except BaseException:
        dump(ops/'terminal.json',dict(status='failed',root=str(root),gitrev=rev(),error=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
    assert sys.platform.startswith('linux') and str(REPO).startswith('/mnt/pvc/')
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','launch','worker']);p.add_argument('--plan',required=True);ARGS=p.parse_args();os.chdir(REPO);globals()[ARGS.mode](read(ARGS.plan))
