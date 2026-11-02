"""Pod-only, event-driven research wave. No external-test based decisions."""
import argparse, copy, datetime, hashlib, json, logging, math, os, random
from pathlib import Path
import subprocess, sys, time, traceback

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'graph_transform'),str(REPO/'graph_transform/scripts')]
import yaml
from adaptive_baseline_audit import dump, sha, independent_metrics
from train_5fold import make_fold_config, _stable_fold_seed

PY='/mnt/pvc/graphtrans/DBond/.venv/bin/python'
OPS=Path('/mnt/pvc/graphtrans/dbond_adaptive_research_20261102_ops')
BASE=Path('/mnt/pvc/graphtrans/four_state_parallel_global_replication_ops')
DATA=Path('/mnt/pvc/graphtrans/DBond/dataset')
FOLDS=('1222','2252','3514','6072','9075')
BASE_REV='111c23313da9ae11a7ee33f8357be0a1876086ec'
SOURCE_CONFIG=REPO/'graph_transform/config/pre_synthesis_four_state_parallel_global_replication_single.yaml'
ARMS={
    'full':{},
    'no_theory':{'model.use_theory_features':False},
    'no_message_passing':{'model.num_gcn_layers':0,'model.num_gat_layers':0},
    'shallow_1gcn_1gat':{'model.num_gcn_layers':1,'model.num_gat_layers':1},
    'binary_k5':{'model.bond_head_type':'concat_mlp','data.four_state_targets_manifest':None,
                 'data.four_state_targets_manifest_template':None},
}

def read(p):return json.loads(Path(p).read_text())
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def rev():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
def source_hashes():return {str(p.relative_to(REPO)):sha(p) for p in (REPO/'graph_transform').rglob('*.py')}
def require(v,msg):
    if not v:raise ValueError(msg)
def seed(v):
    import numpy as np,torch
    random.seed(v);np.random.seed(v);torch.manual_seed(v)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(v)

def base_config():
    cfg=yaml.safe_load(SOURCE_CONFIG.read_text())
    # Preserve the pre-synthesis mask and actual folded training/row validation protocol.
    cfg['model']['state_feature_mask']=[True,True,False]
    cfg['model']['env_feature_mask']=[True,False]
    cfg['training'].pop('validation_split_seed',None)
    cfg['training']['validation_split_base_seed']=42
    cfg['training']['research_selection']={'schema':1,'weights':[.4,.4,.2]}
    cfg['training']['checkpoint_selection']='max_f1'
    return cfg

def fold_config(arm,fold,root,base_seed=42,patch=None):
    cfg=base_config()
    for dotted,value in (patch if patch is not None else ARMS[arm]).items():
        section,key=dotted.split('.'); cfg[section][key]=value
    cfg['experiment']['name']='adaptive_'+arm
    cfg['ablation']['tag']='gt_pre'
    cfg['data']['cache_dir_template']=str(root/'cache'/arm/'fold_{fold}')
    cfg['data']['validation_source_csv_template']=str(DATA/'5fold_soft/{fold}.train.fbr.shuffle.multi.csv')
    if cfg['model']['bond_head_type']=='four_state_mlp':
        cfg['data']['four_state_targets_manifest_template']='/mnt/pvc/graphtrans/four_state_targets_v1/fold_{fold}/manifest.json'
    cfg,runroot,_=make_fold_config(cfg,fold,str(DATA/'5fold_folded'),str(root/arm),_stable_fold_seed(base_seed,fold))
    cfg['data']['test_csv_path']=None
    cfg['evaluation']['export_from_evaluate_cache']=True
    return cfg,Path(runroot)

def load_runtime(cfg):
    from train_graph_model import apply_ablation_config
    return apply_ablation_config(copy.deepcopy(cfg))

def preflight(gpu):
    import numpy as np,torch
    from models.graph_transformer import GraphTransformer
    from models.utils import build_model_config
    from training.research_selection import ResearchSelection,composite_score
    from train_graph_model import create_datasets
    import train_graph_model as ts
    from test_four_state_width import collate,to_cuda,longest_indices
    from training.four_state_loss import four_state_cross_entropy
    from training.loss_functions import BinaryBondLoss
    from training.trainer import Trainer
    ts.logger=logging.getLogger('research_preflight')
    torch.set_num_threads(1);torch.cuda.set_device(gpu)
    OPS.mkdir(exist_ok=True)
    require(not (OPS/'smoke.json').exists(),'Smoke already committed')
    bound=read(BASE/'checkpoint_registry.json')['arms']['single']
    require(sha(BASE/'checkpoint_registry.json')=='4f4b7d476d94fe25126d4ca8d4cf85ab2e1285c5d910cff624e98fd62bb61254','Baseline registry changed')
    report=dict(status='running',gitrev=rev(),source_sha256=source_hashes(),folds={},arms={},checks={})
    toy=dict(f1=.8,ex_precision=.75,ex_recall=.75,subset_acc=.1)
    require(abs(composite_score(toy)-.64)<1e-12,'Composite formula')
    sel=ResearchSelection(dict(schema=1,weights=[.4,.4,.2]));require(sel.update(toy,1)[0],'Initial selector')
    require(not sel.update(toy,2)[0],'Ties must retain earlier checkpoint')
    for bad in (float('nan'),float('inf'),-.1,1.1):
        try:composite_score(dict(toy,f1=bad))
        except ValueError:pass
        else:raise ValueError('Nonfinite/range selector accepted')
    from evaluation.metrics import BinaryBondMetrics
    yy=np.array([[1,0,1,0],[0,1,0,0]],dtype=np.int8)
    pp=np.array([[.9,.2,.4,.99],[.6,.7,.8,.8]],dtype=np.float32)
    mm=np.array([[1,1,1,0],[1,1,0,0]],dtype=bool)
    fixture=independent_metrics(yy,pp,mm)
    calc=BinaryBondMetrics({'threshold':.5,'threshold_strategy':'fixed'})
    calc.update(torch.tensor(pp),torch.tensor(yy),label_mask=torch.tensor(mm),from_logits=False)
    fixture_prod=calc.compute()
    for key,target in [('bond_f1','f1'),('bond_accuracy','accuracy'),('subset_acc','subset_acc'),('example_jaccard','ex_acc')]:
        require(abs(fixture[key]-fixture_prod[target])<1e-7,'Independent padding-free fixture mismatch')
    for fold in FOLDS:
        cfg,_=fold_config('full',fold,OPS/'preflight')
        runtime=load_runtime(cfg);runtime['data']['cache_graphs']=False
        train,val,test=create_datasets(runtime)
        require(test is None,'External test must be deferred')
        ss=[set(ds.data.seq) for ds in (train,val)]
        # Reading test sequence IDs for exclusion is not evaluating labels or choosing a model.
        test_path=DATA/f'5fold_folded/{fold}.test.fbr.multi.csv'
        import pandas as pd
        outer=set(pd.read_csv(test_path,usecols=['seq']).seq)
        require(not(ss[0]&ss[1] or ss[0]&outer or ss[1]&outer),'Sequence isolation failed')
        prior=read(BASE/'smoke.json')['folds'][fold]
        require(sha(train.csv_path)==prior['train_sha256'] and sha(val.csv_path)==prior['val_sha256'],'Baseline internal split changed')
        saved=torch.load(bound[fold]['checkpoint'],map_location='cpu',weights_only=False)
        require(sha(bound[fold]['checkpoint'])==bound[fold]['checkpoint_sha256'],'Weight binding')
        old=GraphTransformer(build_model_config(runtime));old.load_state_dict(saved['model_state_dict'],strict=True)
        require(runtime['model']['state_feature_mask']==[True,True,False] and runtime['model']['env_feature_mask']==[True,False],'Effective masks')
        report['folds'][fold]=dict(seed=_stable_fold_seed(42,fold),train_sha256=sha(train.csv_path),val_sha256=sha(val.csv_path),
            outer_test_sha256=sha(test_path),four_state_manifest_sha256=sha(runtime['data']['four_state_targets_manifest']),
            train_sequences=len(ss[0]),val_sequences=len(ss[1]),strict_cpu=True)
        if fold==FOLDS[0]:
            for arm in ARMS:
                acfg,_=fold_config(arm,fold,OPS/'preflight'); acfg=load_runtime(acfg);acfg['data']['cache_graphs']=False
                at,av,_=create_datasets(acfg)
                require(sha(at.csv_path)==prior['train_sha256'] and sha(av.csv_path)==prior['val_sha256'],'Ablation split changed')
                seed(_stable_fold_seed(42,fold));m=GraphTransformer(build_model_config(acfg)).cuda();m.train()
                bt=to_cuda(collate(at,longest_indices(at,128)))
                opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4)
                if arm=='binary_k5':require('four_state_targets' not in bt,'Binary target source not removed')
                trainer=Trainer(model=m,optimizer=opt,criterion=BinaryBondLoss(acfg['loss']),device=torch.device('cuda'),config=acfg)
                loss,_=trainer._forward_pass(bt)
                require(torch.isfinite(loss).item(),'Nonfinite preflight loss')
                trainer._backward_pass(loss)
                grads=[p.grad for p in m.parameters() if p.grad is not None]
                require(grads and all(torch.isfinite(g).all().item() for g in grads),'Invalid gradients')
                cpu=GraphTransformer(build_model_config(acfg));cpu.load_state_dict({k:v.cpu() for k,v in m.state_dict().items()},strict=True)
                report['arms'][arm]=dict(parameters=sum(p.numel() for p in m.parameters()),loss=float(loss.detach()),strict_cpu_roundtrip=True,
                    gcn_layers=acfg['model']['num_gcn_layers'],gat_layers=acfg['model']['num_gat_layers'],head=acfg['model']['bond_head_type'])
                del m,cpu,opt,bt,loss,trainer;torch.cuda.empty_cache()
        del old,saved,train,val
    # The model and loss production files must be exactly the retained baseline's.
    prior=read(BASE/'launch.json')['source_sha256']
    for name in ('graph_transform/models/graph_transformer.py','graph_transform/models/utils.py','graph_transform/training/four_state_loss.py'):
        require(sha(REPO/name)==prior[name],f'Unexpected architecture implementation change {name}')
    report['status']='passed';report['checks']=dict(validation_only=True,selector_formula_and_ties=True,production_model_unchanged=True)
    dump(OPS/'smoke.json',report);print(json.dumps(dict(status='passed',arms=report['arms'])))

def independent_q(mc,prob_rows):
    import numpy as np
    from scipy.stats import spearmanr
    valid=[i for i,z in enumerate(mc.sample_soft_targets) if z is not None and len(z)]
    ps=np.array([float(np.mean(prob_rows[i],dtype=np.float32)) for i in valid],dtype=float)
    qs=np.array([np.mean(mc.sample_soft_targets[i]) for i in valid],dtype=float)
    first={}
    for j,i in enumerate(valid):first.setdefault(mc.row_group_keys[i],j)
    byseq={}
    for j in first.values():byseq.setdefault(mc.row_seq_keys[valid[j]],[]).append(j)
    pp=np.array([ps[idx].mean() for idx in byseq.values()]);qq=np.array([qs[idx].mean() for idx in byseq.values()])
    n=max(1,int(round(len(pp)*.1)))
    top=np.argsort(-pp,kind='stable')[:n]
    brier=float(np.mean(np.concatenate([(prob_rows[i]-mc.sample_soft_targets[i])**2 for i in valid])))
    return dict(q_brier=brier,q_seq=float(spearmanr(pp,qq).statistic),top10=float(qq[top].mean()/qq.mean()))

def audit_trial(cfg_path,gpu):
    import numpy as np,torch
    from models.graph_transformer import GraphTransformer
    from models.utils import build_model_config
    from train_graph_model import create_datasets,create_data_loaders
    from evaluation import Evaluator
    from training.research_selection import composite_score
    import train_graph_model as ts
    ts.logger=logging.getLogger('independent_trial_audit');torch.set_num_threads(1);torch.cuda.set_device(gpu)
    cfg=yaml.safe_load(Path(cfg_path).read_text()); runtime=load_runtime(cfg)
    roots=list(Path(runtime['training']['checkpoint_dir']).glob('*/best_model.pt'))
    require(len(roots)==1,'Expected exactly one unique training run')
    cp_root=roots[0].parent; resolved=yaml.safe_load((cp_root/'resolved_training_config.yaml').read_text())
    require(resolved['data']['test_csv_path'] is None,'Outer test evaluated during search')
    _,val,_=create_datasets(runtime)
    from data import GraphDataLoader
    vl=GraphDataLoader(val,batch_size=runtime['evaluation']['batch_size'],shuffle=False,num_workers=0)
    records={}
    for policy,name in (('max_f1','best_model.pt'),('composite','best_model_composite.pt')):
        ck=cp_root/name;saved=torch.load(ck,map_location='cpu',weights_only=False)
        model=GraphTransformer(build_model_config(runtime));model.load_state_dict(saved['model_state_dict'],strict=True)
        rng=saved['extra_state']['training_rng_protocol']
        require(rng['training_seed']==cfg['experiment']['seed'] and rng['validation_split_seed']==runtime['training']['validation_split_seed'],'Seed protocol mismatch')
        require(model.config.state_feature_mask==[True,True,False] and model.config.env_feature_mask==[True,False],'Checkpoint masks')
        model.cuda();e=Evaluator(model,torch.device('cuda'),runtime,logging.getLogger('validation_audit'))
        metrics=e.evaluate(vl,allow_target_aware=False);mc=e.metrics_calculator
        probabilities=[torch.sigmoid(torch.from_numpy(np.array(z,dtype=np.float32))).numpy() for z in mc.sample_predictions]
        width=max(len(z) for z in probabilities);n=len(probabilities)
        y=np.zeros((n,width),np.int8);p=np.zeros((n,width));mask=np.zeros((n,width),bool)
        for i,(a,b) in enumerate(zip(probabilities,mc.sample_targets)):
            p[i,:len(a)]=a;y[i,:len(a)]=b;mask[i,:len(a)]=True
        independent=independent_metrics(y,p,mask);independent.update(independent_q(mc,probabilities))
        mapping={'bond_f1':'f1','bond_accuracy':'accuracy','example_precision':'ex_precision','example_recall':'ex_recall',
                 'subset_acc':'subset_acc','auc':'auc','q_brier':'q_brier','q_seq':'q_spearman_pep_seq','top10':'q_top10_enrichment_seq'}
        diffs={a:independent[a]-float(metrics[b]) for a,b in mapping.items()}
        require(all(abs(z)<3e-6 for z in diffs.values()),f'Independent metric mismatch {diffs}')
        require(abs(float(saved['metrics']['f1'])-float(metrics['f1']))<3e-5,'Saved/replayed validation F1 mismatch')
        predfile=cp_root/f'validation_{policy}.npz'
        np.savez_compressed(predfile,y=y,p=p.astype(np.float32),mask=mask)
        records[policy]=dict(checkpoint=str(ck),checkpoint_sha256=sha(ck),prediction=str(predfile),prediction_sha256=sha(predfile),
            epoch=int(saved['epoch'])+1,metrics=independent,selector_score=composite_score(metrics),
            independent_metric_differences=diffs,strict_cpu=True,actual_seed=rng,
            validation_source_sha256=sha(val.csv_path),effective_config_sha256=sha(cp_root/'resolved_training_config.yaml'),
            parameters=sum(v.numel() for v in model.parameters()),resource_metrics={k:float(v) for k,v in metrics.items() if k.startswith('gpu_mem_')})
        del model,e,mc;torch.cuda.empty_cache()
    receipt=dict(status='passed',config=str(cfg_path),config_sha256=sha(cfg_path),
        gitrev=rev(),external_test_deferred=True,policies=records)
    dump(Path(cfg_path).parent/'trial_audit.json',receipt)
    return receipt

def free_gpus():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return [int(v[0]) for line in text.strip().splitlines() if (v:=[int(x.strip()) for x in line.split(',')]) and v[1]<64 and v[2]==0]

def launch():
    OPS.mkdir(exist_ok=True);require(not(OPS/'launch.json').exists(),'Unique launch already exists')
    smoke=read(OPS/'smoke.json');require(smoke['status']=='passed' and smoke['gitrev']==rev() and smoke['source_sha256']==source_hashes(),'Smoke/source mismatch')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
    root=Path('/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/adaptive_research_batch')/stamp
    root.mkdir(parents=True,exist_ok=False)
    info=dict(status='submitted',root=str(root),gitrev=rev(),source=str(REPO),created_at=utc(),
        source_sha256=source_hashes(),smoke_sha256=sha(OPS/'smoke.json'),arms=ARMS,
        initial_training_tasks=25,training_cap=None,automatic_stage='validation_only_single_factor_wave',
        baseline_audit='CPU independent; does not overwrite historical results',outer_test_deferred=True)
    with(OPS/'launch.claim').open('x') as f:f.write(utc())
    dump(OPS/'launch.json',info)
    with(OPS/'worker.log').open('x') as log:
        proc=subprocess.Popen([PY,'-u',str(Path(__file__)),'worker'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    info['worker_pid']=proc.pid;dump(OPS/'launch.json',info);print(json.dumps(info))

def worker():
    info=read(OPS/'launch.json');root=Path(info['root'])
    with(OPS/'worker.claim').open('x') as f:f.write(str(os.getpid()))
    try:
        # Durable independent archival audit starts once on CPU, alongside reference training.
        with(OPS/'baseline_audit.log').open('x') as alog:
            audit=subprocess.Popen([PY,str(REPO/'graph_transform/scripts/adaptive_baseline_audit.py'),'--out',str(OPS/'baseline_audit')],
                cwd=REPO,stdout=alog,stderr=subprocess.STDOUT,start_new_session=True)
        dump(OPS/'baseline_audit_process.json',dict(pid=audit.pid,started_at=utc()))
        tasks=[]
        for arm in ARMS:
            for fold in FOLDS:
                cfg,rr=fold_config(arm,fold,root);rr.mkdir(parents=True,exist_ok=True)
                cfgpath=rr/'config.yaml';cfgpath.write_text(yaml.safe_dump(cfg,sort_keys=False))
                tasks.append(dict(arm=arm,fold=fold,config=str(cfgpath),run_root=str(rr),status='queued'))
        state=dict(status='running',root=str(root),gitrev=rev(),tasks=tasks,external_test_deferred=True,updated_at=utc())
        dump(OPS/'wave_state.json',state)
        # Serial batches: every single-factor arm has one fixed split per available GPU.
        for arm in ARMS:
            pending=[t for t in tasks if t['arm']==arm];active={}
            while pending or active:
                for gpu in free_gpus():
                    if not pending:break
                    if gpu in active:continue
                    task=pending.pop(0);task['status']='running';task['gpu']=gpu;task['started_at']=utc()
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
                    log=open(Path(task['run_root'])/'train_and_audit.log','x')
                    command=[PY,'-u',str(Path(__file__)),'trial','--config',task['config']]
                    proc=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    task['pid']=proc.pid;active[gpu]=(proc,task,log)
                for gpu,(proc,task,log) in list(active.items()):
                    code=proc.poll()
                    if code is None:continue
                    log.close();require(code==0,f"Trial failed {task['arm']} {task['fold']}: no automatic retry")
                    receipt=read(Path(task['run_root'])/'trial_audit.json');require(receipt['status']=='passed','Trial audit not passed')
                    task['status']='audited';task['ended_at']=utc();task['receipt_sha256']=sha(Path(task['run_root'])/'trial_audit.json')
                    del active[gpu]
                state['updated_at']=utc();dump(OPS/'wave_state.json',state)
                if pending or active:time.sleep(15)
            dump(OPS/f'{arm}_completed.json',dict(status='completed',arm=arm,root=str(root),ended_at=utc()))
        audit.wait()
        state['status']='completed';state['updated_at']=utc();dump(OPS/'wave_state.json',state)
        result=present(root,tasks);dump(OPS/'result.json',result)
        dump(OPS/'terminal.json',dict(status='completed',root=str(root),gitrev=rev(),completed_at=utc(),
            result_sha256=sha(OPS/'result.json'),review_required=True,external_test_deferred=True))
    except BaseException:
        # Stop only this controller's recorded children; never unrelated GPU users.
        for proc,task,log in list(locals().get('active',{}).values()):
            if proc.poll() is None:
                os.killpg(proc.pid,15)
            log.close()
        dump(OPS/'terminal.json',dict(status='failed',root=str(root),gitrev=rev(),error=traceback.format_exc(),automatic_retry=False))
        raise

def present(root,tasks):
    import numpy as np
    byarm={}
    for arm in ARMS:
        rr=[read(Path(t['run_root'])/'trial_audit.json') for t in tasks if t['arm']==arm]
        byarm[arm]={pol:{k:dict(mean=float(np.mean([r['policies'][pol]['metrics'][k] for r in rr])),
            sample_sd=float(np.std([r['policies'][pol]['metrics'][k] for r in rr],ddof=1)))
            for k in rr[0]['policies'][pol]['metrics']} for pol in ('max_f1','composite')}
    # Results are development diagnostics. No external-test ranking or global winning claim.
    result=dict(status='completed',root=str(root),gitrev=rev(),summary=byarm,
        baseline_archival_audit=read(OPS/'baseline_audit/result.json'),external_test_evaluated=False,
        checkpoint_weights=[.4,.4,.2],all_negative_results_retained=True,
        next_step='Independent review; per-outer-fold validation-only selection. No test-based auto promotion.')
    lines=['# 自适应研究：第一轮简化与选权重对照','',
        '此表仅为内部序列隔离验证集结果。外测未推理，不能替换论文主结果。','',
        '|候选|权重规则|Bond F1|Example F1|SubAcc|q_seq|','|---|---|---:|---:|---:|---:|']
    for arm,pols in byarm.items():
        for pol,ms in pols.items():
            f=lambda k:f"{ms[k]['mean']:.6f} ± {ms[k]['sample_sd']:.6f}"
            lines.append(f"|{arm}|{pol}|{f('bond_f1')}|{f('example_f1_harmonic')}|{f('subset_acc')}|{f('q_seq')}|")
    (OPS/'analysis.md').write_text('\n'.join(lines))
    return result

def trial(config):
    cfg=yaml.safe_load(Path(config).read_text());require(cfg['data']['test_csv_path'] is None,'Test forbidden')
    subprocess.run([PY,'-u',str(REPO/'graph_transform/scripts/train_graph_model.py'),'--config',config],cwd=REPO,check=True)
    audit_trial(config,0)

if __name__=='__main__':
    require(sys.platform.startswith('linux') and str(REPO).startswith('/mnt/pvc/'),'Pod only')
    os.chdir(REPO);parser=argparse.ArgumentParser();parser.add_argument('action',choices=['smoke','launch','worker','trial','audit-trial'])
    parser.add_argument('--gpu',type=int,default=0);parser.add_argument('--config');a=parser.parse_args()
    if a.action=='smoke':preflight(a.gpu)
    elif a.action=='trial':trial(a.config)
    elif a.action=='audit-trial':audit_trial(a.config,a.gpu)
    else:globals()[a.action]()
