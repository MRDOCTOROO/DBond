"""Pod-only alpha=1 charge-composition paired experiment and terminal receipts."""
import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'graph_transform'),str(REPO/'graph_transform/scripts')]
from data.row_validation import file_sha
from four_state_ops import dump,idle
import yaml

OPS=Path('/mnt/pvc/graphtrans/four_state_charge_ops')
TARGETS=Path('/mnt/pvc/graphtrans/four_state_targets_v1')
ARMS={'control':False,'features':True}
FOLDS=('1222','2252','3514','6072','9075')
PY=str(REPO/'.venv/bin/python')


def config(arm):
    return REPO/f'graph_transform/config/pre_synthesis_four_state_charge_{arm}.yaml'


def revision():
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()


def smoke():
    if (OPS/'smoke.json').exists(): raise RuntimeError('Preflight exists; refuse repeat')
    gpus,_,_=idle()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpus[0]),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    with (OPS/'preflight.log').open('a') as log:
        subprocess.run([PY,str(REPO/'graph_transform/scripts/test_four_state_charge.py'),'--output',str(OPS/'smoke.json')],
                       cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    proof=json.loads((OPS/'smoke.json').read_text())
    proof['gitrev']=revision()
    dump(OPS/'smoke.json',proof)
    print(json.dumps(proof))


def launch():
    if (OPS/'launch.json').exists(): raise RuntimeError('Launch exists; refuse duplicate')
    proof=json.loads((OPS/'smoke.json').read_text())
    if proof['status']!='passed' or proof['gitrev']!=revision(): raise RuntimeError('Missing/stale preflight')
    if any(file_sha(config(a))!=proof['configs'][a] for a in ARMS): raise RuntimeError('Stale configs')
    changed=subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=REPO,text=True).splitlines()
    if not set(changed)<={'graph_transform/scripts/ensemble_baseline_preds.py'}: raise RuntimeError(f'Unexpected changes: {changed}')
    if json.loads((TARGETS/'preparation.json').read_text())['status']!='passed': raise RuntimeError('Targets not prepared')
    evidence=OPS/'evidence/res_000140.pdf'
    assert file_sha(evidence)=='814b51192664603b3b764766b436451b6117c041f84ef619f679e70da25a9e9c'
    ids,rows,processes=idle()
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
    root=REPO/'checkpoints/graph_transform/four_state_charge_batch'/stamp
    info={'root':str(root),'gitrev':revision(),'gpus':ids,'gpu_preflight':rows,'compute_processes':processes,
          'feature_arms':ARMS,'conditional_weight':1.,'configs':proof['configs'],
          'source_sha256':{str(p.relative_to(REPO)):file_sha(p) for p in (REPO/'graph_transform').rglob('*.py') if '__pycache__' not in p.parts},
          'target_preparation_sha256':file_sha(TARGETS/'preparation.json'),
          'target_manifests':{f:file_sha(TARGETS/f'fold_{f}/manifest.json') for f in FOLDS},
          'evidence':{str(evidence):file_sha(evidence),str(OPS/'evidence/source_records.json'):file_sha(OPS/'evidence/source_records.json')},
          'scope':'Both four-state alpha=1; only charge-aware fragment-composition residual differs; folded group_uniform/full-row validation/valid-bond metrics.'}
    dump(OPS/'launch.json',info)
    env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    try:
        with (OPS/'worker.log').open('x') as log:
            proc=subprocess.Popen([PY,__file__,'worker'],cwd=REPO,env=env,stdin=subprocess.DEVNULL,
                                  stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        info['worker_pid']=proc.pid
        dump(OPS/'launch.json',info)
    except BaseException:
        dump(OPS/'terminal.json',{'status':'failed','root':str(root),'error':traceback.format_exc()})
        raise
    print(json.dumps({k:v for k,v in info.items() if k!='source_sha256'}))


def analyze(info):
    root=Path(info['root'])
    status=json.loads((root/'status.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    summary=json.loads((root/'summary/summary.json').read_text())
    proof=json.loads((OPS/'smoke.json').read_text())
    assert status['status']=='completed' and len(status['arms'])==2
    assert all(a['returncode']==0 for a in status['arms'])
    assert manifest['gitrev']==info['gitrev'] and len(manifest['arms'])==2
    assert all(file_sha(REPO/p)==h for p,h in info['source_sha256'].items())
    assert all(file_sha(config(a))==info['configs'][a] for a in ARMS)
    assert all(file_sha(p)==h for p,h in info['evidence'].items())
    assert file_sha(TARGETS/'preparation.json')==info['target_preparation_sha256']
    assert all(file_sha(TARGETS/f'fold_{f}/manifest.json')==info['target_manifests'][f] for f in FOLDS)
    data,binding={},{}
    for arm,(label,enabled) in zip(manifest['arms'],ARMS.items()):
        assert file_sha(arm['config_snapshot'])==arm['config_sha256']==info['configs'][label]
        bound=next(s for s in summary['sources'] if s['arm']==arm['name'])
        assert file_sha(arm['metrics_csv'])==bound['sha256']
        binding[arm['metrics_csv']]=file_sha(arm['metrics_csv'])
        with Path(arm['metrics_csv']).open() as stream: rows=list(csv.DictReader(stream))
        assert len(rows)==5 and {r['fold_id'] for r in rows}==set(FOLDS)
        data[arm['name']]={}
        for row in rows:
            fold=row['fold_id']
            raw_path=Path(row['metric_csv_path'])
            with raw_path.open() as stream: raw={r['metric']:float(r['value']) for r in csv.DictReader(stream)}
            binding[str(raw_path)]=file_sha(raw_path)
            vals={k:float(row[k]) for k in ('f1','accuracy','auc','q_brier','ex_precision','ex_recall')}
            vals['q_seq']=float(row['q_spearman_pep_seq'])
            for k,v in vals.items(): assert abs(v-raw[{'q_seq':'q_spearman_pep_seq'}.get(k,k)])<1e-12
            p,r=vals['ex_precision'],vals['ex_recall']
            vals['ex_f1_harmonic']=2*p*r/(p+r) if p+r else 0.
            assert all(math.isfinite(v) for v in vals.values())
            data[arm['name']][fold]=vals
            folder=Path(arm['cv_root'])/f'fold_{fold}'
            runtime=yaml.safe_load((folder/'config.yaml').read_text())
            assert runtime['loss']['four_state_conditional_weight']==1.
            assert runtime['model']['bond_head_type']=='four_state_mlp'
            assert runtime['model']['use_charge_partition_features']==enabled
            assert runtime['data']['weighting_scheme']=='group_uniform' and runtime['evaluation']['threshold']==.5
            paths=list(folder.glob('checkpoints/**/row_validation_provenance.json'))
            assert len(paths)==1
            prov=json.loads(paths[0].read_text())
            assert prov['mode']=='full_original_training_spectra_by_internal_val_sequence'
            assert prov['outer_test_used_for_internal_validation'] is False
            assert file_sha(prov['source_path'])==prov['source_sha256']==proof['folds'][fold]['source_sha256']
            assert file_sha(prov['split_val_csv'])==proof['folds'][fold]['val_sha256']
            assert file_sha(Path(prov['split_val_csv']).with_name('train.csv'))==proof['folds'][fold]['train_sha256']
            assert prov['val_rows']==proof['folds'][fold]['val_rows']
            states=json.loads(paths[0].with_name('four_state_provenance.json').read_text())
            assert states['manifest_sha256']==info['target_manifests'][fold]
    names=list(data)
    for entry in summary['metrics']:
        arm,metric=entry['arm'],entry['metric']
        values=[data[arm][f][metric] for f in FOLDS]
        delta=[data[arm][f][metric]-data[names[0]][f][metric] for f in FOLDS]
        sign=-1 if metric=='q_brier' else 1
        for key,v in [('mean',statistics.mean(values)),('std_ddof1',statistics.stdev(values)),
                      ('paired_delta_mean',statistics.mean(delta)),('paired_delta_std_ddof1',statistics.stdev(delta))]:
            assert abs(v-entry[key])<1e-12,(arm,metric,key)
        assert sum(sign*d>1e-12 for d in delta)==entry['win_splits']
    metrics={a:{r['metric']:r for r in summary['metrics'] if r['arm']==n} for a,n in zip(ARMS,names)}
    v=metrics['features']
    gates={'delta_f1_at_least_0.002':v['f1']['paired_delta_mean']>=.002,
           'f1_win_at_least_4_of_5':v['f1']['win_splits']>=4,
           'delta_auc_nonnegative':v['auc']['paired_delta_mean']>=0,
           'delta_q_brier_nonpositive':v['q_brier']['paired_delta_mean']<=0}
    out={'status':'completed','root':str(root),'gitrev':info['gitrev'],'finished_at':status['finished_at'],
         'conditional_weight':1.,'feature_arms':ARMS,'metrics':metrics,'gates':gates,'improvement_gate':all(gates.values()),
         'folds':data,'paired_validation_verified':True,
         'scope':'Five overlapping sequence-level random 80/20 splits, one seed per split. Paired alpha=1 four-state controls. Padding-free, fixed threshold .5. Engineering gates are not statistical significance.'}
    dump(OPS/'result.json',out)
    lines=['# 四状态 α=1：电荷感知的两侧碎片组成特征','',f"源码：{info['gitrev']}；完成：{status['finished_at']}",'',
           '| 指标 | 同轮四状态对照 | +组成特征 | Δ / 改善划分 |','|---|---:|---:|---:|']
    for m,label,scale in [('f1','有效键micro-F1 %',100),('accuracy','有效键accuracy %',100),('auc','ROC-AUC',1),
                          ('ex_f1_harmonic','谱图P/R调和F1 %',100),('q_brier','q-Brier ↓',1),('q_seq','q_seq',1)]:
        c,v=metrics['control'][m],metrics['features'][m]
        lines.append(f"| {label} | {c['mean']*scale:.6f} ± {c['std_ddof1']*scale:.6f} | {v['mean']*scale:.6f} ± {v['std_ddof1']*scale:.6f} | {v['paired_delta_mean']*scale:+.6f} / {v['win_splits']}/5 |")
    lines += ['',('全部采用门槛通过，列待确认候选。' if out['improvement_gate'] else '未全部通过预定提升门槛。'),
              '两臂α=1，折叠group_uniform训练、完整原谱验证、有效键mask和固定0.5阈值保持一致。',
              '仅新增17→32→3零初始化残差，修正b-only/y-only/both相对none的logits；新增675参数。',
              'Paizs & Suhai (2005)支持关注电荷、碱性组成与质子保留，未验证本项目特征或D肽收益。',
              '五组划分重叠、各一次初始化；不替换冻结主表。','',
              '| 划分 | 四状态对照F1% | +组成特征F1% | Δ pp |','|---|---:|---:|---:|']
    for f in FOLDS:
        c,v=data[names[0]][f]['f1'],data[names[1]][f]['f1']
        lines.append(f'| {f} | {c*100:.6f} | {v*100:.6f} | {(v-c)*100:+.6f} |')
    (OPS/'report.md').write_text('\n'.join(lines)+'\n')
    dump(OPS/'review_receipt.json',{'status':'passed','root':str(root),'checks':[
        '10 runs completed; source/config/target/evidence/metric hashes bound',
        'runtime feature flags and alpha=1 verified for each run',
        'train/validation row SHAs match paired preflight on all splits',
        '16 arm-metrics mean/sampleSD/paired delta/SD/win counts independently recalculated'],
        'hashes':binding,'gates':gates,'improvement_gate':out['improvement_gate']})


def worker():
    info=json.loads((OPS/'launch.json').read_text())
    try:
        command=[PY,'-u',str(REPO/'graph_transform/scripts/run_architecture_batch.py'),'--configs',
                 *[str(config(a)) for a in ARMS],'--fold_data_dir','dataset/5fold_folded',
                 '--gpus',','.join(map(str,info['gpus'])),'--run_root',info['root']]
        with (OPS/'training.log').open('x') as log:
            subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
        analyze(info)
        dump(OPS/'terminal.json',{'status':'completed','root':info['root'],'result':str(OPS/'result.json'),'report':str(OPS/'report.md')})
    except BaseException:
        dump(OPS/'terminal.json',{'status':'failed','root':info['root'],'error':traceback.format_exc()})
        raise


if __name__=='__main__':
    os.chdir(REPO)
    OPS.mkdir(exist_ok=True)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['smoke','launch','worker'])
    globals()[p.parse_args().action]()
