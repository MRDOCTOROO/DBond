"""Read-only archival baseline audit. All numerical work runs on the Pod."""
import argparse, hashlib, importlib.util, json, sys, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef

REPO = Path(__file__).resolve().parents[2]
HIST = Path('/mnt/pvc/graphtrans/DBond')
FOLDS = ('1222','2252','3514','6072','9075')
RUNS = {'dbond_s_pre':'20260902_200428', 'dbond_m_pre':'20260902_200428',
        'dbond_af_pre':'20260907_230122', 'dbond_af_opt_pre':'20260907_230122'}

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def dump(p,d):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)); t.replace(p)

def path(p):
    p=Path(p)
    return p if p.is_absolute() else HIST/p

def divide(a,b): return np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b!=0)

def independent_metrics(y,p,mask):
    assert y.shape==p.shape==mask.shape and mask.any(axis=1).all()
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    v=y[mask]; prob=p[mask]; pred=p>.5
    tp=((y==1)&pred&mask).sum(1); fp=((y==0)&pred&mask).sum(1)
    fn=((y==1)&~pred&mask).sum(1); tn=((y==0)&~pred&mask).sum(1)
    ep=float(divide(tp,tp+fp).mean()); er=float(divide(tp,tp+fn).mean())
    T,F,N=int(tp.sum()),int(fp.sum()),int(fn.sum())
    out=dict(bond_accuracy=float((tp.sum()+tn.sum())/mask.sum()),
        bond_precision=T/(T+F) if T+F else 0., bond_recall=T/(T+N) if T+N else 0.,
        bond_f1=2*T/(2*T+F+N) if 2*T+F+N else 0., example_precision=ep, example_recall=er,
        example_f1_harmonic=2*ep*er/(ep+er) if ep+er else 0.,
        example_f1_samples=float(divide(2*tp,2*tp+fp+fn).mean()),
        example_jaccard=float(divide(tp,tp+fp+fn).mean()),subset_acc=float(((fp+fn)==0).mean()),
        auc=float(roc_auc_score(v,prob)),ap=float(average_precision_score(v,prob)),
        mcc=float(matthews_corrcoef(v,prob>.5)),hard_brier=float(np.mean((prob-v)**2)),
        tp=T,fp=F,fn=N,tn=int(tn.sum()),rows=int(len(y)),valid_bonds=int(mask.sum()))
    # Soft frequency metrics must not be confused with hard Brier.
    return out

def audit_one(name,run,fold,out):
    root=HIST/'result/cv'/name/run
    index=pd.read_csv(root/'5fold_metrics.csv'); row=index[index.fold_id.astype(str)==fold].iloc[0]
    ck=path(row.best_model_path)
    saved=torch.load(ck,map_location='cpu',weights_only=False)
    cfg={k:saved[k] for k in ('csv','seq','model','train_args')}
    model_path=REPO/('ludbond/dbond_s.py' if name=='dbond_s_pre' else
        'ludbond/dbond_m.py' if name=='dbond_m_pre' else 'ludbondaf/dbond_m_exp_af.py')
    sys.path.insert(0,str(model_path.parent))
    spec=importlib.util.spec_from_file_location('audit_'+name,model_path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    net=mod.Model(cfg); net.load_state_dict(saved['model_state_dict'],strict=True); del net
    testpath=path(cfg['csv']['test_dataset_path']); trainpath=path(cfg['csv']['train_dataset_path'])
    source=pd.read_csv(testpath,na_filter=False)
    predpath=root/f'fold_{fold}/pred/test.pred.csv'
    pred=pd.read_csv(predpath)
    prob=pred.pred_prob.to_numpy(dtype=np.float64); true=pred.true.to_numpy(dtype=np.int8)
    assert np.array_equal(pred.pred.to_numpy(dtype=np.int8),(prob>.5).astype(np.int8)), 'Threshold/pred mismatch'
    seqcol=cfg['csv']['seq_col_name']; labelcol=cfg['csv']['label_col_name']
    if name=='dbond_s_pre':
        assert len(source)==len(pred)
        assert np.array_equal(true,source[labelcol].to_numpy(dtype=np.int8))
        bi=source[cfg['csv']['bond_index_col_name']].to_numpy(dtype=int)
        assert np.array_equal(bi,pred.bond_index.to_numpy(dtype=int))
        # Reconstruct whole spectra from the frozen test single-bond source.
        cols=[seqcol]+[c for c in ('charge','pep_mass','nce','scan_num') if c in source]
        ids=pd.factorize(pd.MultiIndex.from_frame(source[cols]),sort=False)[0]
        width=int(bi.max())+1; rows=int(ids.max())+1
        composite=ids*width+bi
        assert len(np.unique(composite))==len(composite), 'Duplicate precursor/bond collapsed by historical example metrics'
        y=np.zeros((rows,width),dtype=np.int8); p=np.zeros_like(y,dtype=float); mask=np.zeros_like(y,dtype=bool)
        y[ids,bi]=true; p[ids,bi]=prob; mask[ids,bi]=True
        lengths=source.groupby(ids,sort=True)[seqcol].first().str.len().to_numpy()-1
        assert np.array_equal(mask,np.arange(width)[None,:]<lengths[:,None]),'Missing/extra real bonds'
    else:
        assert len(pred)%len(source)==0
        width=len(pred)//len(source); y=true.reshape(len(source),width); p=prob.reshape(len(source),width)
        lengths=source[seqcol].str.len().to_numpy()-1
        mask=np.arange(width)[None,:]<lengths[:,None]
        labels=[np.fromstring(s,sep=';',dtype=np.int8) for s in source[labelcol].astype(str)]
        assert all(len(z)>=int(n) and np.array_equal(z[:n],a[:n]) for z,a,n in zip(labels,y,lengths)), 'Source labels/row order mismatch'
    metrics=independent_metrics(y,p,mask)
    train_seq=pd.read_csv(trainpath,usecols=[seqcol])[seqcol].astype(str)
    test_seq=source[seqcol].astype(str)
    overlap=sorted(set(train_seq)&set(test_seq))
    assert not overlap,'Outer train/test sequence leakage'
    # Recover the actual historical row-level random split and quantify sequence leakage.
    n=len(train_seq); nt=int((1-float(cfg['train_args'].get('validation_split',.2)))*n)
    perm=torch.randperm(n,generator=torch.Generator().manual_seed(int(cfg['train_args']['seed']))).numpy()
    ts=set(train_seq.iloc[perm[:nt]]); vs=set(train_seq.iloc[perm[nt:]])
    leakage=len(ts&vs)
    compare={'lab_acc_mi':'bond_accuracy','lab_precision_mi':'bond_precision','lab_recall_mi':'bond_recall',
             'lab_f1_mi':'bond_f1','subset_acc':'subset_acc','ex_acc':'example_jaccard',
             'ex_precision':'example_precision','ex_recall':'example_recall','AUC':'auc'}
    deltas={k:metrics[v]-float(row[k]) for k,v in compare.items() if k in row and pd.notna(row[k])}
    receipt=dict(model=name,run=run,fold=fold,strict_cpu=True,checkpoint=str(ck),checkpoint_sha256=sha(ck),
        prediction=str(predpath),prediction_sha256=sha(predpath),test=str(testpath),test_sha256=sha(testpath),
        train=str(trainpath),train_sha256=sha(trainpath),model_source_sha256=sha(model_path),config=cfg,
        seed=int(cfg['train_args']['seed']),metrics=metrics,delta_vs_original_index=deltas,
        outer_sequence_overlap=0,internal_row_split_sequence_overlap=leakage,
        internal_train_sequence_count=len(ts),internal_validation_sequence_count=len(vs),
        protocol_comparable_to_current_gt=False,
        caveat='Historical training-source identity is not proven by strict restore. Internal split differs; corrected grouped baseline training required.')
    dump(out/f'{name}_{fold}.json',receipt)
    return receipt

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',required=True); args=parser.parse_args()
    assert sys.platform.startswith('linux') and str(REPO).startswith('/mnt/pvc/')
    torch.set_num_threads(1); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    receipts=[]; failures=[]
    for name,run in RUNS.items():
        for fold in FOLDS:
            try: receipts.append(audit_one(name,run,fold,out))
            except Exception as e:
                failures.append(dict(model=name,fold=fold,error=str(e),traceback=traceback.format_exc()))
                dump(out/f'{name}_{fold}_failure.json',failures[-1])
    summary={}
    for name in RUNS:
        rr=[r for r in receipts if r['model']==name]
        if len(rr)==5:
            summary[name]={k:dict(mean=float(np.mean([r['metrics'][k] for r in rr])),
                sample_sd=float(np.std([r['metrics'][k] for r in rr],ddof=1))) for k in rr[0]['metrics']}
    result=dict(status='completed' if not failures else 'partial_requires_review',
        statistical_unit='Five overlapping sequence splits; ddof=1, no independent n=5 assumption',
        summary=summary,receipts=receipts,failures=failures,auto_correct_paper=False,
        fairness_action='Retrain s/m/AF/AF-opt with group-by-sequence internal validation before a fair final comparison')
    dump(out/'result.json',result)
    lines=['# 复现基线原始预测独立审计','',f"已完成 {len(receipts)}/20；来源或数值待处理 {len(failures)} 项。",'',
        '不预设基线虚高；区分指标字段错误、padding、聚合、外层泄漏与内部验证污染。原历史文件不覆盖。','',
        '|模型|Bond F1|Example F1（P/R调和）|Subset accuracy|内部序列泄漏|','|---|---:|---:|---:|---|']
    for name,ms in summary.items():
        f=lambda k:f"{ms[k]['mean']*100:.4f} ± {ms[k]['sample_sd']*100:.4f}%"
        rr=[r for r in receipts if r['model']==name]
        lines.append(f"|{name}|{f('bond_f1')}|{f('example_f1_harmonic')}|{f('subset_acc')}|{[r['internal_row_split_sequence_overlap'] for r in rr]}|")
    lines += ['', '严格 CPU 恢复只证明当前源与权重结构相容，不能证明历史训练源码未改。内部行级拆分的泄漏会影响选权重；不能用重算测试指标消除此问题，也不能据此按比例扣减基线。需要序列隔离的重训对照。', '', '失败详情与所有 SHA、实际种子保存在 result.json。']
    (out/'analysis.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status=result['status'],audited=len(receipts),failures=len(failures),summary=summary)))

if __name__=='__main__': main()
