"""Independent read-only review of completed arms; Pod CPU only."""
import argparse,copy,datetime,hashlib,json,os,re,subprocess,sys,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,roc_auc_score,average_precision_score,matthews_corrcoef

OPS=Path('/mnt/pvc/graphtrans/dbond_adaptive_regularization_wave2_20261103_ops')
ROOT=Path('/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/adaptive_regularization_wave2_batch/20261102_205543')
REFERENCE_ROOT=Path('/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/adaptive_research_batch/20261102_155846')
REFERENCE_REV='addf42cae71f19c4bd1894249da3623b5f00034c'
SOURCE=Path('/mnt/pvc/graphtrans/dbond_adaptive_wave2_source_3566c14')
REV='3566c14d461dd5e9d4d2a51a53929dc13da52efd'
FOLDS=('1222','2252','3514','6072','9075')

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for x in iter(lambda:f.read(2**20),b''):h.update(x)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False));t.replace(p)
def check(x,msg):
    if not x:raise AssertionError(msg)
def stats(a):return dict(mean=float(np.mean(a)),sample_sd=float(np.std(a,ddof=1)))
def difference(a,b):
    d=np.asarray(a)-np.asarray(b)
    return dict(**stats(d),per_fold=dict(zip(FOLDS,d.tolist())),wins=int((d>0).sum()),ties=int((d==0).sum()),losses=int((d<0).sum()))
def div(a,b):return np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b!=0)

def recompute(npz,frame):
    with np.load(npz) as z:y=z['y'];p=z['p'];mask=z['mask']
    check(y.shape==p.shape==mask.shape,'NPZ shapes')
    lengths=frame.seq.str.len().to_numpy()-1
    check(len(frame)==len(y) and np.array_equal(mask,np.arange(y.shape[1])[None,:]<lengths[:,None]),'Row identity/mask')
    labels=[np.fromstring(str(s),sep=';',dtype=np.int8) for s in frame.true_multi]
    check(all(np.array_equal(a[:n],b[:n]) for a,b,n in zip(labels,y,lengths)),'Validation label order')
    pred=p>.5;v=y[mask];vp=p[mask];vb=pred[mask]
    tp=((y==1)&pred&mask).sum(1);fp=((y==0)&pred&mask).sum(1);fn=((y==1)&~pred&mask).sum(1)
    ep=float(div(tp,tp+fp).mean());er=float(div(tp,tp+fn).mean())
    result=dict(bond_f1=float(f1_score(v,vb,zero_division=0)),bond_accuracy=float(accuracy_score(v,vb)),
        bond_precision=float(precision_score(v,vb,zero_division=0)),bond_recall=float(recall_score(v,vb,zero_division=0)),
        example_precision=ep,example_recall=er,example_f1_harmonic=2*ep*er/(ep+er) if ep+er else 0.,
        example_f1_samples=float(div(2*tp,2*tp+fp+fn).mean()),example_jaccard=float(div(tp,tp+fp+fn).mean()),
        subset_acc=float(((fp+fn)==0).mean()),auc=float(roc_auc_score(v,vp)),ap=float(average_precision_score(v,vp)),
        mcc=float(matthews_corrcoef(v,vb)),hard_brier=float(np.mean((vp.astype(float)-v)**2)))
    soft=[np.fromstring(str(s),sep=';',dtype=np.float32)[:n] for s,n in zip(frame.soft_multi,lengths)]
    rows_p=np.asarray([float(p[i,:n].mean()) for i,n in enumerate(lengths)])
    rows_q=np.asarray([float(s.mean()) for s in soft])
    keys=list(zip(frame.seq,frame.charge.astype(float),frame.nce.astype(float)))
    first={}
    for i,k in enumerate(keys):first.setdefault(k,i)
    byseq={}
    for k,i in first.items():byseq.setdefault(k[0],[]).append(i)
    sp=np.array([rows_p[idx].mean() for idx in byseq.values()]);sq=np.array([rows_q[idx].mean() for idx in byseq.values()])
    nq=max(1,round(len(sp)*.1));top=np.argsort(-sp,kind='mergesort')[:nq]
    result.update(q_brier=float(np.mean(np.concatenate([(p[i,:n].astype(float)-soft[i])**2 for i,n in enumerate(lengths)]))),
        q_seq=float(spearmanr(sp,sq).statistic),top10=float(sq[top].mean()/sq.mean()))
    return result

def worker(arms,out):
    sys.path[:0]=[str(SOURCE/'graph_transform'),str(SOURCE/'graph_transform/scripts')]
    from models.graph_transformer import GraphTransformer
    from models.utils import build_model_config
    torch.set_num_threads(1)
    launch=read(OPS/'launch.json');smoke=read(OPS/'smoke.json')
    reuse=read(OPS/'reference_reuse_receipt.json')
    check(reuse['status']=='passed' and sha(OPS/'reference_reuse_receipt.json')==launch['reference_reuse_sha256'],'Reference reuse receipt')
    previous_review=Path('/mnt/pvc/graphtrans/dbond_adaptive_research_20261102_ops/completed_reviews/complete_five_arms')
    check(sha(previous_review/'result.json')==reuse['review_sha256'] and sha(previous_review/'receipt.json')==reuse['receipt_sha256'],'Prior independent review changed')
    check(launch['root']==str(ROOT) and launch['gitrev']==REV and smoke['gitrev']==REV,'Launch/root/revision')
    check(sha(OPS/'smoke.json')==launch['smoke_sha256'],'Smoke binding')
    for name,h in launch['source_sha256'].items():check(sha(SOURCE/name)==h,'Frozen source changed '+name)
    rows={};proof={};same=[]
    for arm in arms:
        marker=OPS/f'{arm}_completed.json' if arm!='full' else Path('/mnt/pvc/graphtrans/dbond_adaptive_research_20261102_ops/full_completed.json');d=read(marker)
        check(d['status']=='completed' and d['root']==str(ROOT if arm!='full' else REFERENCE_ROOT) and d['arm']==arm,'Incomplete arm')
        rows[arm]={};proof[arm]=dict(marker_sha256=sha(marker),folds={})
        for fold in FOLDS:
            trial=(ROOT if arm!='full' else REFERENCE_ROOT)/arm/f'fold_{fold}';auditpath=trial/'trial_audit.json';a=read(auditpath)
            if arm=='full':check(sha(auditpath)==reuse['folds'][fold]['audit_sha256'],'Reused reference audit changed')
            check(a['status']=='passed' and a['gitrev']==(REV if arm!='full' else REFERENCE_REV) and a['external_test_deferred'],'Trial audit')
            check(Path(a['config'])==trial/'config.yaml' and sha(a['config'])==a['config_sha256'],'Config binding')
            cfg=yaml.safe_load(Path(a['config']).read_text());check(cfg['data']['test_csv_path'] is None,'Test exposure')
            target=cfg['data'].get('four_state_targets_manifest')
            if target:check(sha(target)==smoke['folds'][fold]['four_state_manifest_sha256'],'Target binding')
            states={};vals={};foldproof={}
            for policy in ('max_f1','composite'):
                r=a['policies'][policy];ck=Path(r['checkpoint']);pred=Path(r['prediction'])
                check(ck.is_relative_to(trial) and pred.is_relative_to(trial),'Artifact escaped trial')
                check(sha(ck)==r['checkpoint_sha256'] and sha(pred)==r['prediction_sha256'],'Checkpoint/prediction binding')
                check(sha(ck.parent/'resolved_training_config.yaml')==r['effective_config_sha256'],'Resolved config')
                saved=torch.load(ck,map_location='cpu',weights_only=False)
                model=GraphTransformer(build_model_config(cfg));model.load_state_dict(saved['model_state_dict'],strict=True)
                check(int(saved['epoch'])+1==r['epoch'],'Epoch')
                rng=saved['extra_state']['training_rng_protocol'];check(rng==r['actual_seed'],'RNG')
                check(rng['training_seed']==smoke['folds'][fold]['seed'] and rng['validation_split_seed']==rng['training_seed'],'Actual seed')
                check(model.config.state_feature_mask==[True,True,False] and model.config.env_feature_mask==[True,False],'Masks')
                vp=saved['extra_state']['row_validation_provenance']['split_val_csv']
                check(sha(vp)==r['validation_source_sha256']==smoke['folds'][fold]['val_sha256'],'Validation source')
                frame=pd.read_csv(vp,usecols=['seq','charge','nce','true_multi','soft_multi'],na_filter=False)
                computed=recompute(pred,frame)
                errors={k:computed[k]-r['metrics'][k] for k in computed}
                check(all(abs(v)<3e-6 for v in errors.values()),'Independent NPZ replay differs '+str(errors))
                vals[policy]=computed;states[policy]=saved['model_state_dict']
                # Decode saved checkpoint metrics as a separate, raw validation view.
                m=saved['metrics'];p0=float(m['ex_precision']);r0=float(m['ex_recall'])
                raw_score=.4*float(m['f1'])+.4*(2*p0*r0/(p0+r0) if p0+r0 else 0.)+.2*float(m['subset_acc'])
                foldproof[policy]=dict(checkpoint=str(ck),checkpoint_sha256=r['checkpoint_sha256'],prediction_sha256=r['prediction_sha256'],
                    strict_cpu=True,epoch=r['epoch'],parameters=sum(p.numel() for p in model.parameters()),recompute_errors=errors,
                    raw_selection_score=raw_score,raw_validation_f1=float(m['f1']),raw_validation_subset=float(m['subset_acc']),
                    audit_replay_resource=r['resource_metrics'])
                del model,saved,frame
            identical=states['max_f1'].keys()==states['composite'].keys() and all(torch.equal(v,states['composite'][k]) for k,v in states['max_f1'].items())
            if identical:
                same.append(dict(arm=arm,fold=fold,epoch=foldproof['max_f1']['epoch'],
                    replay_metric_noise={k:vals['composite'][k]-vals['max_f1'][k] for k in vals['max_f1']}))
                # Identical model states cannot constitute a selection improvement.
                vals['composite']=copy.deepcopy(vals['max_f1'])
            rows[arm][fold]=vals
            finished_log=(trial/'train_and_audit.log').read_text(errors='replace')
            matches=re.findall(r'Training completed in ([0-9.]+)s',finished_log)
            foldproof.update(receipt_sha256=sha(auditpath),identical_model_state=identical,
                training_wall_seconds=float(matches[-1]) if matches else None)
            proof[arm]['folds'][fold]=foldproof
            del states,finished_log
    summary={arm:{pol:{k:stats([rows[arm][f][pol][k] for f in FOLDS]) for k in rows[arm][FOLDS[0]][pol]}
        for pol in ('max_f1','composite')} for arm in arms}
    pairs={arm:{pol:{k:difference([rows[arm][f][pol][k] for f in FOLDS],[rows['full'][f][pol][k] for f in FOLDS])
        for k in rows[arm][FOLDS[0]][pol]} for pol in ('max_f1','composite')} for arm in arms if arm!='full'}
    policy_pairs={arm:{k:difference([rows[arm][f]['composite'][k] for f in FOLDS],[rows[arm][f]['max_f1'][k] for f in FOLDS])
        for k in rows[arm][FOLDS[0]]['max_f1']} for arm in arms}
    result=dict(status='passed',root=str(ROOT),gitrev=REV,arms=arms,scope='Internal validation only, not paper external performance',
        summary=summary,paired_vs_full=pairs,composite_minus_max_f1=policy_pairs,per_fold=rows,provenance=proof,
        identical_weight_pairs=same,source_bound=True,outer_test_evaluated=False)
    dump(out/'result.json',result)
    lines=['# 第二轮单因素调参：内部验证配对复核','',
        '只展示已完成方案；full严格复用首轮权重，不是本轮新训练。尚非外测结果。每组为五个重叠外层划分各自的内部验证均值 ± 样本SD。',
        '同一权重两次推理的微小差异不计作选权重收益；原回执与原始预测保留。','',
        '|方案|选权重|Bond F1 %|谱图F1 %|subAcc %|q_seq|Top10|','|---|---|---:|---:|---:|---:|---:|']
    for arm in arms:
        for pol in ('max_f1','composite'):
            m=summary[arm][pol]
            ff=lambda k,s=100:f"{m[k]['mean']*s:.4f} ± {m[k]['sample_sd']*s:.4f}"
            lines.append(f"|{arm}|{pol}|{ff('bond_f1')}|{ff('example_f1_harmonic')}|{ff('subset_acc')}|{ff('q_seq',1)}|{ff('top10',1)}|")
    lines+=['','## 相对同规则full的逐折配对差','', '|方案|规则|Bond F1差 pp|谱图F1差 pp|subAcc差 pp|q_seq差|','|---|---|---:|---:|---:|---:|']
    for arm,pols in pairs.items():
        for pol,m in pols.items():
            ff=lambda k,s=100:f"{m[k]['mean']*s:+.4f} ± {m[k]['sample_sd']*s:.4f} ({m[k]['wins']}/{m[k]['ties']}/{m[k]['losses']})"
            lines.append(f"|{arm}|{pol}|{ff('bond_f1')}|{ff('example_f1_harmonic')}|{ff('subset_acc')}|{ff('q_seq',1)}|")
    lines+=['','括号为原始数值差的胜/平/负，q-Brier应反向阅读。全部指标及配对SD在result.json，不能只报有利指标。',
        f'','完全相同权重的策略对数量：'+str(len(same))+'；这些策略对统一使用max-F1同次预测作为对照。',
        '', '参数量：'+str({arm:proof[arm]['folds'][FOLDS[0]]['max_f1']['parameters'] for arm in arms}),
        '', '审计显存是验证重推过程的PyTorch分配/保留峰值，不等于训练峰值。训练耗时取已完成任务的结束记录，未读取运行中epoch。',
        '', '此为已完成候选的审计报告；不能据此按外测选模型。最终选择仅依据各外层划分自己的内部验证；此跨折汇总不能作为无偏全局最优或显著性证据。']
    (out/'analysis.md').write_text('\n'.join(lines))
    dump(out/'receipt.json',dict(status='passed',root=str(ROOT),arms=arms,script_sha256=sha(Path(__file__)),result_sha256=sha(out/'result.json'),report_sha256=sha(out/'analysis.md')))
    print(json.dumps(dict(status='passed',summary=summary,paired_vs_full=pairs),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['start','worker']);p.add_argument('--arms',nargs='+',required=True);p.add_argument('--label',required=True);a=p.parse_args()
    check(sys.platform.startswith('linux'),'Pod only');check(re.fullmatch(r'[a-z0-9_]+',a.label),'Label')
    check(a.arms[0]=='full' and set(a.arms)<=set(['full','lr_3e4','dropout_03','weight_decay_1e3','hidden128_k5']),'Arms')
    out=OPS/'completed_reviews'/a.label;out.mkdir(parents=True,exist_ok=True)
    if a.mode=='start':
        with(out/'claim').open('x') as f:f.write(datetime.datetime.now(datetime.timezone.utc).isoformat())
        with(out/'review.log').open('x') as log:
            job=subprocess.Popen([sys.executable,str(Path(__file__)),'worker','--arms',*a.arms,'--label',a.label],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        dump(out/'state.json',dict(status='running',pid=job.pid));print(json.dumps(dict(status='started',pid=job.pid,out=str(out))))
    else:
        try:
            worker(a.arms,out);dump(out/'state.json',dict(status='passed'))
        except BaseException:
            dump(out/'state.json',dict(status='failed',error=traceback.format_exc()));raise

