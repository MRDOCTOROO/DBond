"""Fixed NCE-head plus cardinality composition, with three audited controls reused."""
import argparse, copy, datetime, importlib.util, json, logging, os, subprocess, sys, time, traceback
from pathlib import Path
import yaml
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'graph_transform/scripts'),str(REPO/'graph_transform')]
import adaptive_research_ops as core
import adaptive_regularization_wave2 as reuse
from adaptive_baseline_audit import dump,sha
OPS=Path('/mnt/pvc/graphtrans/dbond_adaptive_nce_cardinality_wave11_20261104_ops')
PATCHES={'full':{},'nce_count':{'loss.cardinality_weight':.1,'model.nce_head_mode':'stratified4'}}
core.OPS=OPS;core.ARMS=PATCHES
PRIOR_ROOT=reuse.PRIOR_ROOT
PRIOR=Path('/mnt/pvc/graphtrans/dbond_adaptive_research_source_addf42c')


def smoke(gpu):
    from nce_cardinality_preflight import smoke as checked_smoke
    checked_smoke(REPO,OPS,PATCHES,gpu)


def launch():
    OPS.mkdir(exist_ok=True);core.require(not(OPS/'launch.json').exists(),'Unique launch exists')
    s=core.read(OPS/'smoke.json');c=core.read(OPS/'cardinality_smoke.json');r=core.read(OPS/'reference_reuse_receipt.json')
    core.require(s['status']==c['status']==r['status']=='passed' and s['gitrev']==core.rev() and s['source_sha256']==core.source_hashes(),'Smoke/source mismatch')
    core.require(r['cardinality_smoke_sha256']==sha(OPS/'cardinality_smoke.json') and c['architecture_smoke_sha256']==sha(OPS/'smoke.json'),'Count proof binding')
    core.require(r['review_sha256']==reuse.previous_binding()['review_sha256'],'Reference changed')
    root=Path('/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/adaptive_nce_cardinality_batch')/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
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
            cfg,rr=core.fold_config('nce_count',f,root);rr.mkdir(parents=True,exist_ok=False);path=rr/'config.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False));tasks.append(dict(arm='nce_count',fold=f,status='queued',config=str(path),run_root=str(rr)))
        pending=[t for t in tasks if t['arm']=='nce_count']
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
        dump(OPS/'nce_count_completed.json',dict(status='completed',arm='nce_count',root=str(root),ended_at=core.utc()))
        r=core.present(root,tasks);r.update(reference_reused=True,reference_root=str(PRIOR_ROOT),reference_new_trainings=0,declared_factor='stratified4 NCE head + CE + .1 normalized count-distribution CE')
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


