"""Pod-only CE4 + union-Brier trials, validation-only selection, then locked tests.

No outer-test CSV is opened while training or selecting lambda. All five
selections are persisted before either comparison endpoint is evaluated.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
import datetime
import json
import logging
import math
import os
from pathlib import Path
import queue
import statistics
import subprocess
import sys
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'graph_transform'), str(REPO/'graph_transform/scripts')]
import yaml
from data.row_validation import file_sha
from four_state_ops import dump, idle

OPS = Path('/mnt/pvc/graphtrans/four_state_brier_ops')
TARGETS = Path('/mnt/pvc/graphtrans/four_state_targets_v1')
CONFIG = REPO/'graph_transform/config/pre_synthesis_four_state_brier.yaml'
PY = str(REPO/'.venv/bin/python')
FOLDS = ('1222', '2252', '3514', '6072', '9075')
LAMBDAS = (0.0, 0.25, 1.0, 4.0)
PROTOCOL = 'four_state_union_brier_v1'
FIELDS = ('f1', 'accuracy', 'auc', 'ex_precision', 'ex_recall',
          'ex_f1_harmonic', 'q_brier', 'q_seq')


def revision():
    return subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip()


def require(ok, why):
    if not ok:
        raise ValueError(why)


def forbid_outer_test_reads():
    """Process-local guard covers open(), pandas and other Python file opens."""
    def guard(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            name = os.fsdecode(args[0]).replace('\\', '/').lower()
            if '.test.fbr.' in name:
                raise PermissionError('Outer-test CSV access forbidden before lambda selection: '+name)
    sys.addaudithook(guard)


def validate_config(cfg, weight):
    require(cfg['experiment']['loss_protocol'] == PROTOCOL, 'Protocol marker missing')
    require(cfg['data']['test_csv_path'] is None, 'Training must not construct a test dataset')
    require(cfg['model']['bond_head_type'] == 'four_state_mlp'
            and not cfg['model']['use_charge_partition_features'], 'Original four-state architecture required')
    require(cfg['loss']['four_state_conditional_weight'] == 1.0
            and cfg['loss']['four_state_union_brier_weight'] == weight, 'Unexpected loss weights')
    require(cfg['data']['weighting_scheme'] == 'group_uniform'
            and cfg['training']['checkpoint_selection'] == 'max_f1'
            and cfg['evaluation']['threshold'] == 0.5, 'Training/evaluation protocol changed')


def verify_sources(info):
    require(revision()==info['gitrev'],'Source revision changed')
    require(file_sha(CONFIG)==info['config_sha256'],'Base configuration changed')
    require(all(file_sha(REPO/p)==h for p,h in info['source_sha256'].items()),'Source content changed')
    require(all(file_sha(TARGETS/f'fold_{f}/manifest.json')==h for f,h in info['target_manifests'].items()),'Targets changed')


def smoke():
    require(not (OPS/'smoke.json').exists(), 'Preflight already exists')
    devices, _, _ = idle()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(devices[0]), OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    with (OPS/'preflight.log').open('a') as log:
        subprocess.run([PY,str(REPO/'graph_transform/scripts/test_four_state_brier.py'),
                        '--output',str(OPS/'smoke.json')], cwd=REPO, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    proof = json.loads((OPS/'smoke.json').read_text())
    proof['gitrev'] = revision()
    proof['config_sha256'] = file_sha(CONFIG)
    dump(OPS/'smoke.json', proof)
    print(json.dumps(proof))


def launch():
    require(not (OPS/'launch.json').exists(), 'Refuse duplicate batch')
    proof = json.loads((OPS/'smoke.json').read_text())
    require(proof['status']=='passed' and proof['gitrev']==revision()
            and proof['config_sha256']==file_sha(CONFIG), 'Missing or stale preflight')
    changes = subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=REPO,text=True).splitlines()
    require(set(changes)<={'graph_transform/scripts/ensemble_baseline_preds.py'}, 'Unexpected source changes')
    devices, rows, procs = idle()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
    root = REPO/'checkpoints/graph_transform/four_state_brier_batch'/stamp
    info = {'root':str(root), 'gitrev':revision(), 'gpus':devices,
            'gpu_preflight':rows, 'compute_processes':procs,
            'conditional_weight':1.0, 'lambda_grid':LAMBDAS,
            'config_sha256':file_sha(CONFIG),
            'source_sha256':{str(p.relative_to(REPO)):file_sha(p)
                             for p in (REPO/'graph_transform').rglob('*.py')
                             if '__pycache__' not in p.parts},
            'target_manifests':{f:file_sha(TARGETS/f'fold_{f}/manifest.json') for f in FOLDS},
            'selection_rule':'Per outer split: minimum INTERNAL validation q_brier at each trial max-F1 checkpoint; exact ties smaller lambda; lambda=0 eligible. Freeze all five choices before reading any outer-test CSV.',
            'scope':'CE4 alpha=1, no charge residual, folded group_uniform train, full original-spectrum internal validation, padding-free metrics. 20 training trials; only control and validation-selected policy are tested.'}
    dump(OPS/'launch.json',info)
    try:
        with (OPS/'worker.log').open('x') as log:
            proc = subprocess.Popen([PY,__file__,'worker'],cwd=REPO,
                env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        info['worker_pid'] = proc.pid
        dump(OPS/'launch.json',info)
    except BaseException:
        dump(OPS/'terminal.json',{'status':'failed','root':str(root),'error':traceback.format_exc()})
        raise
    print(json.dumps({k:v for k,v in info.items() if k!='source_sha256'}))


def prepare_jobs(info):
    from train_5fold import make_fold_config, _stable_fold_seed
    base = yaml.safe_load(CONFIG.read_text())
    root = Path(info['root'])
    require(not root.exists(), 'Refuse to overwrite existing batch')
    root.mkdir(parents=True)
    jobs = []
    for weight in LAMBDAS:
        tag = 'lambda_'+str(weight).replace('.','p')
        for fold in FOLDS:
            seed = _stable_fold_seed(42,fold)
            cfg, folder, checkpoint_root = make_fold_config(base,fold,str(REPO/'dataset/5fold_folded'),str(root/'trials'/tag),seed)
            cfg['data']['test_csv_path'] = None
            cfg['data']['cache_dir'] = str(root/'cache'/tag/('fold_'+fold))
            cfg['loss']['four_state_union_brier_weight'] = weight
            validate_config(cfg,weight)
            folder = Path(folder)
            folder.mkdir(parents=True)
            path = folder/'config.yaml'
            path.write_text(yaml.safe_dump(cfg,sort_keys=False))
            jobs.append({'fold':fold,'weight':weight,'seed':seed,'folder':str(folder),
                         'checkpoint_root':checkpoint_root,'config':str(path),'config_sha256':file_sha(path)})
    dump(root/'jobs.json',{'jobs':jobs,'source_config_sha256':file_sha(CONFIG)})
    return jobs


def task_train(path):
    forbid_outer_test_reads()
    import torch
    import train_graph_model as training
    job = json.loads(Path(path).read_text())
    require(file_sha(job['config'])==job['config_sha256'],'Trial config changed')
    cfg = yaml.safe_load(Path(job['config']).read_text())
    validate_config(cfg,job['weight'])
    sys.argv = ['train_graph_model.py','--config',job['config'],'--seed',str(job['seed'])]
    training.main()
    paths = list(Path(job['checkpoint_root']).rglob('best_model.pt'))
    require(len(paths)==1,'Expected exactly one explicit max-F1 checkpoint')
    checkpoint = torch.load(paths[0],map_location='cpu',weights_only=False)
    marker = checkpoint.get('extra_state',{}).get('loss_protocol',{})
    require(marker.get('name')==PROTOCOL and marker.get('four_state_conditional_weight')==1.0
            and marker.get('four_state_union_brier_weight')==job['weight']
            and marker.get('outer_test_deferred') is True
            and marker.get('checkpoint_selection')=='max_f1','Checkpoint training-loss provenance missing')
    require(marker['source_config_sha256']==job['config_sha256']
            and file_sha(marker['resolved_config_path'])==marker['resolved_config_sha256'],
            'Full training config provenance mismatch')
    actual = yaml.safe_load(Path(marker['resolved_config_path']).read_text())
    validate_config(actual,job['weight'])
    require(actual['experiment']['seed']==job['seed'],'Runtime seed mismatch')
    metrics = checkpoint['metrics']
    q_brier, f1 = float(metrics['q_brier']),float(metrics['f1'])
    require(math.isfinite(q_brier) and 0<=q_brier<=1 and math.isfinite(f1) and 0<=f1<=1,
            'Invalid internal validation metrics')
    selection = checkpoint['checkpoint_selection_state']
    require(selection['best_epoch']==checkpoint['epoch']+1 and selection['best_f1']==f1
            and selection['mode']=='max_f1','Checkpoint is not its recorded validation maximum')
    provenance = json.loads(paths[0].with_name('row_validation_provenance.json').read_text())
    prior = json.loads((OPS/'smoke.json').read_text())['folds'][job['fold']]
    require(file_sha(provenance['split_val_csv'])==prior['val_sha256']
            and file_sha(provenance['split_train_csv'])==prior['train_sha256'], 'Paired train/val split changed')
    require(provenance['outer_test_used_for_internal_validation'] is False,'Invalid validation scope')
    require(not list(Path(job['folder']).rglob('*test*metric*.csv')), 'Unexpected test result during tuning')
    out = dict(job, status='completed',checkpoint=str(paths[0]),checkpoint_sha256=file_sha(paths[0]),
               val_q_brier=q_brier,val_f1=f1,best_epoch=checkpoint['epoch']+1,
               loss_protocol=marker,outer_test_csv_reads=0,
               validation_provenance=str(paths[0].with_name('row_validation_provenance.json')))
    dump(Path(job['folder'])/'train_receipt.json',out)


def run_parallel(jobs,devices,action):
    available = queue.Queue()
    for device in devices:
        available.put(device)
    def run(job):
        device = available.get()
        try:
            folder = Path(job['folder'])
            folder.mkdir(parents=True,exist_ok=True)
            job_path = folder/(action+'_job.json')
            dump(job_path,job)
            env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(device),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
            with (folder/(action+'.log')).open('x') as log:
                try:
                    subprocess.run([PY,__file__,action,'--job',str(job_path)],cwd=REPO,env=env,
                                   stdout=log,stderr=subprocess.STDOUT,check=True)
                except subprocess.CalledProcessError as error:
                    log.flush()
                    tail = (folder/(action+'.log')).read_text(errors='replace').splitlines()[-35:]
                    raise RuntimeError(f'{action} failed for {job_path}; failure tail:\n'+ '\n'.join(tail)) from error
        finally:
            available.put(device)
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(run,job) for job in jobs]
        for future in futures:
            future.result()


def choose_weights(records):
    """Pure selection helper: only internal validation fields influence choices."""
    selected = {}
    for fold in FOLDS:
        rows = [r for r in records if r['fold']==fold]
        require(len(rows)==len(LAMBDAS) and {r['weight'] for r in rows}==set(LAMBDAS),'Incomplete lambda grid')
        for row in rows:
            q,f = row['val_q_brier'],row['val_f1']
            require(isinstance(q,(int,float)) and not isinstance(q,bool) and math.isfinite(q) and 0<=q<=1
                    and isinstance(f,(int,float)) and not isinstance(f,bool) and math.isfinite(f) and 0<=f<=1,'Invalid validation score')
        selected[fold] = min(rows,key=lambda r:(r['val_q_brier'],r['weight']))
    return selected


def lock_selection(info,jobs):
    forbid_outer_test_reads()
    verify_sources(info)
    root = Path(info['root'])
    records = []
    for job in jobs:
        path = Path(job['folder'])/'train_receipt.json'
        row = json.loads(path.read_text())
        require(row['status']=='completed' and row['outer_test_csv_reads']==0,'Incomplete train-only trial')
        require(file_sha(row['config'])==row['config_sha256']
                and file_sha(row['checkpoint'])==row['checkpoint_sha256'],'Trial artifact changed')
        records.append(dict(row,receipt_sha256=file_sha(path)))
    selected = choose_weights(records)
    for fold in FOLDS:
        initializations = [r['loss_protocol']['initialization'] for r in records if r['fold']==fold]
        require(all(value==initializations[0] for value in initializations), 'Production initial state/RNG differ within split '+fold)
    control = {r['fold']:r for r in records if r['weight']==0.0}
    require(not (root/'selection.json').exists(),'Selection already locked')
    result = {'status':'locked','root':str(root),'gitrev':info['gitrev'],
              'locked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'rule':info['selection_rule'],'lambda_grid':list(LAMBDAS),
              'outer_test_csv_reads':0,'trials':records,'control':control,'selected':selected,
              'preflight_sha256':file_sha(OPS/'smoke.json'),'target_manifests':info['target_manifests']}
    dump(root/'selection.json',result)
    return result


def evaluation_jobs(info,selection):
    root = Path(info['root'])
    jobs = []
    for label in ('control','selected'):
        for fold in FOLDS:
            if label=='selected' and selection['selected'][fold]['weight']==0.0:
                continue  # Exact same checkpoint: reuse control metric CSV, no extra forward.
            row = selection[label][fold]
            jobs.append({'label':label,'fold':fold,'folder':str(root/'evaluation'/label/('fold_'+fold)),
                         'trial':row,'selection':str(root/'selection.json'),
                         'selection_sha256':file_sha(root/'selection.json')})
    return jobs


def task_evaluate(path):
    import torch
    from models import GraphTransformer
    from models.utils import build_model_config, CheckpointManager
    from data import CachedGraphDataset, GraphDataLoader
    from evaluation import Evaluator
    from evaluation.metrics import metric_rows
    from train_graph_model import apply_ablation_config
    job = json.loads(Path(path).read_text())
    require(file_sha(job['selection'])==job['selection_sha256'],'Selection changed before test')
    locked = json.loads(Path(job['selection']).read_text())
    require(locked['status']=='locked' and len(locked['selected'])==5,'All choices must precede every test')
    row = locked[job['label']][job['fold']]
    require(row==job['trial'],'Evaluation checkpoint differs from locked choice')
    require(file_sha(row['checkpoint'])==row['checkpoint_sha256']
            and file_sha(row['config'])==row['config_sha256'],'Chosen artifacts changed')
    cfg = yaml.safe_load(Path(row['config']).read_text())
    validate_config(cfg,row['weight'])
    cfg['data']['test_csv_path'] = str(REPO/'dataset/5fold_folded'/f"{job['fold']}.test.fbr.multi.csv")
    cfg = apply_ablation_config(cfg)
    torch.manual_seed(row['seed'])
    torch.cuda.manual_seed_all(row['seed'])
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    model = GraphTransformer(build_model_config(cfg)).to(device)
    CheckpointManager.load_checkpoint(row['checkpoint'],model=model,device=device)
    d = cfg['data']
    dataset = CachedGraphDataset(csv_path=d['test_csv_path'],config=cfg,
        max_seq_len=d['max_seq_len'],graph_strategy=d['graph_strategy'],augmentation=False,split='test',
        cache_dir=d['cache_dir'],rebuild_cache=False,cache_full_graphs=False)
    # This isolation check is intentionally after the immutable selection lock.
    import pandas as pd
    outer_train = pd.read_csv(d['train_csv_path'],usecols=['seq'])
    require(not set(outer_train['seq'].astype(str)) & set(dataset.data['seq'].astype(str)), 'Outer sequence overlap')
    loader = GraphDataLoader(dataset=dataset,batch_size=cfg['evaluation']['batch_size'],shuffle=False,
        num_workers=d['num_workers'],pin_memory=d['pin_memory'],drop_last=False,
        persistent_workers=cfg['performance']['persistent_workers'])
    metrics = Evaluator(model,device,cfg,logging.getLogger('brier_test'),allow_target_aware_threshold=False).evaluate(loader,allow_target_aware=False)
    out = Path(job['folder'])/'test_metric.csv'
    with out.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=['metric','value'])
        writer.writeheader()
        writer.writerows(metric_rows(metrics))
    dump(Path(job['folder'])/'evaluation_receipt.json',dict(job,status='completed',metric_csv=str(out),
        metric_sha256=file_sha(out),test_csv_sha256=file_sha(d['test_csv_path']),
        threshold=0.5,padding_free=True,batch_size=cfg['evaluation']['batch_size']))


def analyze(info,selection):
    verify_sources(info)
    root = Path(info['root'])
    require(file_sha(root/'selection.json')==info['selection_sha256'],'Frozen selection changed')
    data,bindings = {},{}
    for label in ('control','selected'):
        data[label] = {}
        for fold in FOLDS:
            owner = 'control' if label=='selected' and selection['selected'][fold]['weight']==0 else label
            folder = root/'evaluation'/owner/('fold_'+fold)
            receipt = json.loads((folder/'evaluation_receipt.json').read_text())
            require(receipt['selection_sha256']==info['selection_sha256'] and receipt['padding_free']
                    and file_sha(receipt['metric_csv'])==receipt['metric_sha256'],'Evaluation provenance mismatch')
            with Path(receipt['metric_csv']).open() as f:
                raw = {r['metric']:float(r['value']) for r in csv.DictReader(f)}
            p,r = raw['ex_precision'],raw['ex_recall']
            values = {k:raw[k] for k in ('f1','accuracy','auc','ex_precision','ex_recall','q_brier')}
            values.update(q_seq=raw['q_spearman_pep_seq'],ex_f1_harmonic=2*p*r/(p+r) if p+r else 0.0)
            require(all(math.isfinite(v) for v in values.values()),'Nonfinite test metric')
            data[label][fold] = values
            bindings[receipt['metric_csv']] = receipt['metric_sha256']
        rows = [dict(fold_id=f,**data[label][f]) for f in FOLDS]
        with (root/f'{label}_metrics.csv').open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=['fold_id',*FIELDS]);writer.writeheader();writer.writerows(rows)
    metrics = {}
    for label in data:
        metrics[label] = {}
        for metric in FIELDS:
            values = [data[label][f][metric] for f in FOLDS]
            delta = [data[label][f][metric]-data['control'][f][metric] for f in FOLDS]
            sign = -1 if metric=='q_brier' else 1
            metrics[label][metric] = {'mean':statistics.mean(values),'std_ddof1':statistics.stdev(values),
                'paired_delta_mean':statistics.mean(delta),'paired_delta_std_ddof1':statistics.stdev(delta),
                'win_splits':sum(sign*d>1e-12 for d in delta),'tie_splits':sum(abs(d)<=1e-12 for d in delta),
                'loss_splits':sum(sign*d < -1e-12 for d in delta)}
    v = metrics['selected']
    gates = {'delta_f1_at_least_0.002':v['f1']['paired_delta_mean']>=0.002,
             'f1_win_at_least_4_of_5':v['f1']['win_splits']>=4,
             'delta_auc_nonnegative':v['auc']['paired_delta_mean']>=0,
             'delta_q_brier_nonpositive':v['q_brier']['paired_delta_mean']<=0}
    out = {'status':'completed','root':str(root),'gitrev':info['gitrev'],
           'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'conditional_weight':1.0,'lambda_grid':list(LAMBDAS),
           'selected_weights':{f:selection['selected'][f]['weight'] for f in FOLDS},
           'selection_sha256':info['selection_sha256'],'metrics':metrics,'folds':data,
           'gates':gates,'improvement_gate':all(gates.values()),
           'scope':info['scope'],'selection_rule':info['selection_rule']}
    dump(OPS/'result.json',out)
    lines = ['# 四状态 CE + union Brier：仅内部验证选择权重','',
             f"源码 {info['gitrev']}；完成 {out['finished_at']}",'',
             '| 指标 | 同轮 CE4 λ=0 | 验证集选权重 | Δ / 改善划分 |','|---|---:|---:|---:|']
    for key,title,scale in [('f1','有效键micro-F1 %',100),('accuracy','有效键accuracy %',100),('auc','ROC-AUC',1),
                           ('ex_f1_harmonic','谱图P/R调和F1 %',100),('q_brier','q-Brier ↓',1),('q_seq','q_seq',1)]:
        c,x = metrics['control'][key],metrics['selected'][key]
        lines.append(f"| {title} | {c['mean']*scale:.6f} ± {c['std_ddof1']*scale:.6f} | {x['mean']*scale:.6f} ± {x['std_ddof1']*scale:.6f} | {x['paired_delta_mean']*scale:+.6f} / {x['win_splits']}/5 |")
    lines += ['', '全部采用门槛通过，列待复验候选。' if out['improvement_gate'] else '未全部通过预定采用门槛。',
              '两臂α=1，无电荷残差；折叠group_uniform训练、完整原谱内部验证、去padding和固定0.5阈值。',
              '每个划分按候选max-F1 checkpoint的内部val q-Brier选λ，全部五项锁定后才读取外部test；λ0被选中则直接复用对照。',
              '这是验证集选参策略的比较，不能按测试结果再挑其他λ；五组重叠80/20划分，每候选各一次初始化，多轮探索不等于统计显著。','',
              '| 划分 | 选中λ | 内部val q-Brier | 对照F1 % | 选中F1 % | Δ pp |','|---|---:|---:|---:|---:|---:|']
    for f in FOLDS:
        row = selection['selected'][f]
        c,x = data['control'][f]['f1'],data['selected'][f]['f1']
        lines.append(f"| {f} | {row['weight']} | {row['val_q_brier']:.8f} | {c*100:.6f} | {x*100:.6f} | {(x-c)*100:+.6f} |")
    (OPS/'report.md').write_text('\n'.join(lines)+'\n')
    dump(OPS/'review_receipt.json',{'status':'passed','root':str(root),'metric_hashes':bindings,
        'selection_sha256':info['selection_sha256'],'checks':['20 train-only trial receipts and max-F1 validation metrics locked before tests',
        'actual alpha/lambda in checkpoint extra_state; same paired train/val SHAs',
        'only control and per-split validation-selected checkpoints tested',
        'per-split P/R harmonic F1; mean/sampleSD/paired differences/wins recomputed'],
        'gates':gates,'improvement_gate':all(gates.values())})


def worker():
    info = json.loads((OPS/'launch.json').read_text())
    try:
        verify_sources(info)
        jobs = prepare_jobs(info)
        run_parallel(jobs,info['gpus'],'train')
        # Selection process has an audit guard; evaluation runs in new processes.
        selection = lock_selection(info,jobs)
        info['selection_sha256'] = file_sha(Path(info['root'])/'selection.json')
        dump(OPS/'launch.json',info)
        run_parallel(evaluation_jobs(info,selection),info['gpus'],'evaluate')
        analyze(info,selection)
        dump(OPS/'terminal.json',{'status':'completed','root':info['root'],'result':str(OPS/'result.json'),'report':str(OPS/'report.md')})
    except BaseException:
        dump(OPS/'terminal.json',{'status':'failed','root':info['root'],'error':traceback.format_exc()})
        raise


if __name__=='__main__':
    os.chdir(REPO)
    OPS.mkdir(exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['smoke','launch','worker','train','evaluate'])
    parser.add_argument('--job')
    args = parser.parse_args()
    if args.action in ('train','evaluate'):
        require(bool(args.job),'--job is required')
        (task_train if args.action=='train' else task_evaluate)(args.job)
    else:
        globals()[args.action]()
