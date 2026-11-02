"""Second wave: four single-factor trials, audited first-wave full reused."""
import argparse,copy,json,os,subprocess,sys,time,traceback
from pathlib import Path
import yaml
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'graph_transform/scripts'),str(REPO/'graph_transform')]
import adaptive_research_ops as core
from adaptive_baseline_audit import dump,sha

OPS=Path('/mnt/pvc/graphtrans/dbond_adaptive_regularization_wave2_20261103_ops')
PREV=Path('/mnt/pvc/graphtrans/dbond_adaptive_research_20261102_ops')
PRIOR_ROOT=Path('/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/adaptive_research_batch/20261102_155846')
PATCHES={
    'full':{},
    'lr_3e4':{'training.learning_rate':.0003},
    'dropout_03':{'model.dropout':.3},
    'weight_decay_1e3':{'training.weight_decay':.001},
    'hidden128_k5':{'model.hidden_dim':128},
}
core.OPS=OPS;core.ARMS=PATCHES

def previous_binding():
    review=PREV/'completed_reviews/complete_five_arms'
    r=core.read(review/'receipt.json');s=core.read(review/'source_binding_receipt.json')
    core.require(r['status']==s['status']=='passed','Previous wave not independently audited')
    core.require(r['result_sha256']==s['result_sha256']==sha(review/'result.json'),'Review binding changed')
    result=core.read(review/'result.json');core.require(result['root']==str(PRIOR_ROOT),'Prior root mismatch')
    core.require(sha(PREV/'result.json')==s['original_result_sha256'] and sha(PREV/'terminal.json')==s['original_terminal_sha256'],'Original terminal/result changed')
    proof={}
    for fold in core.FOLDS:
        audit=PRIOR_ROOT/'full'/f'fold_{fold}/trial_audit.json';a=core.read(audit)
        core.require(a['status']=='passed' and a['external_test_deferred'],'Reference not valid')
        for p in a['policies'].values():
            core.require(sha(p['checkpoint'])==p['checkpoint_sha256'] and sha(p['prediction'])==p['prediction_sha256'],'Reference artifact changed')
        proof[fold]=dict(audit_sha256=sha(audit),reference_gitrev=a['gitrev'],reference=str(audit))
    return dict(review_sha256=r['result_sha256'],receipt_sha256=sha(review/'receipt.json'),source_binding_sha256=sha(review/'source_binding_receipt.json'),folds=proof)

def smoke(gpu):
    OPS.mkdir(exist_ok=True)
    core.require(gpu in core.free_gpus(),'Preflight GPU is not idle')
    bound=previous_binding()
    # Shared production training/model/loss paths are exact copies of wave1.
    old=core.read(PREV/'launch.json')['source_sha256']
    for name in ('graph_transform/models/graph_transformer.py','graph_transform/models/utils.py',
                 'graph_transform/training/trainer.py','graph_transform/training/metrics.py',
                 'graph_transform/scripts/train_graph_model.py','graph_transform/training/research_selection.py'):
        core.require(sha(REPO/name)==old[name],'Undeclared production change '+name)
    core.preflight(gpu)
    # Existing assertions remain unmodified; an additional bound receipt records reuse.
    dump(OPS/'reference_reuse_receipt.json',dict(status='passed',**bound,
        production_path_exact=True,new_reference_trainings=0,patches=PATCHES))

def launch():
    OPS.mkdir(exist_ok=True);core.require(not(OPS/'launch.json').exists(),'Unique launch exists')
    smoke=core.read(OPS/'smoke.json');reference=core.read(OPS/'reference_reuse_receipt.json')
    core.require(smoke['status']==reference['status']=='passed' and smoke['gitrev']==core.rev(),'Preflight not passed')
    core.require(smoke['source_sha256']==core.source_hashes(),'Frozen source differs')
    core.require(reference['review_sha256']==previous_binding()['review_sha256'],'Reference binding differs')
    import datetime
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
    root=Path('/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/adaptive_regularization_wave2_batch')/stamp
    root.mkdir(parents=True,exist_ok=False)
    (root/'full').symlink_to(PRIOR_ROOT/'full',target_is_directory=True)
    info=dict(status='submitted',root=str(root),source=str(REPO),gitrev=core.rev(),created_at=core.utc(),
        source_sha256=core.source_hashes(),smoke_sha256=sha(OPS/'smoke.json'),reference_reuse_sha256=sha(OPS/'reference_reuse_receipt.json'),
        arms=PATCHES,new_tasks=20,reused_tasks=5,training_cap=None,reference_root=str(PRIOR_ROOT),
        outer_test_deferred=True,selection='per-fold internal validation only',gpus_available_at_launch=core.free_gpus())
    with(OPS/'launch.claim').open('x') as f:f.write(core.utc())
    dump(OPS/'launch.json',info)
    with(OPS/'worker.log').open('x') as log:
        p=subprocess.Popen([core.PY,'-u',str(Path(__file__)),'worker'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    info['worker_pid']=p.pid;dump(OPS/'launch.json',info)
    print(json.dumps({k:v for k,v in info.items() if k!='source_sha256'}))

def worker():
    info=core.read(OPS/'launch.json');root=Path(info['root']);tasks=[];failures=[]
    with(OPS/'worker.claim').open('x') as f:f.write(str(os.getpid()))
    try:
        # Reference is read-only, reused by exact artifact SHA. No baseline audit rerun.
        (OPS/'baseline_audit').symlink_to(PREV/'baseline_audit',target_is_directory=True)
        for fold in core.FOLDS:
            tasks.append(dict(arm='full',fold=fold,status='reused_audited',run_root=str(PRIOR_ROOT/'full'/f'fold_{fold}')))
        for arm in PATCHES:
            if arm=='full':continue
            for fold in core.FOLDS:
                cfg,rr=core.fold_config(arm,fold,root);rr.mkdir(parents=True,exist_ok=True)
                path=rr/'config.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
                tasks.append(dict(arm=arm,fold=fold,status='queued',config=str(path),run_root=str(rr)))
        state=dict(status='running',root=str(root),gitrev=core.rev(),tasks=tasks,external_test_deferred=True)
        dump(OPS/'wave_state.json',state)
        for arm in PATCHES:
            if arm=='full':continue
            pending=[t for t in tasks if t['arm']==arm];active={}
            while pending or active:
                if not failures:
                    for gpu in core.free_gpus():
                        if not pending:break
                        if gpu in active:continue
                        task=pending.pop(0);task.update(status='running',gpu=gpu,started_at=core.utc())
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
                        log=open(Path(task['run_root'])/'train_and_audit.log','x')
                        p=subprocess.Popen([core.PY,'-u',str(Path(__file__)),'trial','--config',task['config']],
                            cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        task['pid']=p.pid;active[gpu]=(p,task,log)
                else:pending.clear()
                for gpu,(p,t,log) in list(active.items()):
                    if p.poll() is None:continue
                    log.close();t['ended_at']=core.utc()
                    if p.returncode:
                        t['status']='failed';failures.append(dict(arm=t['arm'],fold=t['fold'],exit_code=p.returncode,
                            completed_weights_retained=True,automatic_retry=False))
                    else:
                        receipt=Path(t['run_root'])/'trial_audit.json';a=core.read(receipt)
                        core.require(a['status']=='passed','Independent trial receipt failure')
                        t.update(status='audited',receipt_sha256=sha(receipt))
                    del active[gpu]
                state['updated_at']=core.utc();dump(OPS/'wave_state.json',state)
                if pending or active:time.sleep(15)
            # Never terminate another active split merely because one post-training audit failed.
            core.require(not failures,'Trial failures; finished artifacts kept: '+json.dumps(failures))
            dump(OPS/f'{arm}_completed.json',dict(status='completed',arm=arm,root=str(root),ended_at=core.utc()))
        state['status']='completed';dump(OPS/'wave_state.json',state)
        result=core.present(root,tasks);result.update(reference_reused=True,reference_root=str(PRIOR_ROOT),reference_new_trainings=0)
        text=(OPS/'analysis.md').read_text().replace('第一轮简化与选权重对照','第二轮正则化、学习率与紧凑模型单因素对照')
        (OPS/'analysis.md').write_text(text+'\n\nfull 是第一轮通过独立审计的原权重及预测严格复用，不是本轮新训练。参考训练源码提交与当前候选提交不同，但生产模型/训练/指标路径逐文件SHA完全相同；差别仅在冻结的配置单因素和控制脚本。\n')
        dump(OPS/'result.json',result)
        dump(OPS/'terminal.json',dict(status='completed',root=str(root),gitrev=core.rev(),completed_at=core.utc(),
            result_sha256=sha(OPS/'result.json'),review_required=True,external_test_deferred=True))
    except BaseException:
        dump(OPS/'terminal.json',dict(status='failed',root=str(root),gitrev=core.rev(),error=traceback.format_exc(),
            failures=failures,automatic_retry=False,completed_weights_retained=True));raise

if __name__=='__main__':
    core.require(sys.platform.startswith('linux') and str(REPO).startswith('/mnt/pvc/'),'Pod only')
    os.chdir(REPO);p=argparse.ArgumentParser();p.add_argument('action',choices=['smoke','launch','worker','trial']);p.add_argument('--gpu',type=int,default=0);p.add_argument('--config');a=p.parse_args()
    if a.action=='smoke':smoke(a.gpu)
    elif a.action=='trial':core.trial(a.config)
    else:globals()[a.action]()
