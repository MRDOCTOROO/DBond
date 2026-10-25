"""Pod-only text export of all saved metrics from a completed four-state batch."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(ops):
    ops = Path(ops)
    terminal = json.loads((ops / 'terminal.json').read_text())
    assert terminal['status'] == 'completed'
    result = json.loads((ops / 'result.json').read_text())
    root = Path(result['root'])
    assert str(root) == terminal['root']
    manifest = json.loads((root / 'manifest.json').read_text())
    smoke = json.loads((ops / 'smoke.json').read_text())
    verified = json.loads((ops / 'heartbeat_review_20261020.json').read_text())
    for p, expected in verified['hashes'].items():
        assert sha(p) == expected, p
    assert sha(ops / 'result.json') == verified['result_sha256']
    data = {}
    for arm in manifest['arms']:
        with Path(arm['metrics_csv']).open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 5 and len({r['fold_id'] for r in rows}) == 5
        fields = list(rows[0])[6:]
        data[arm['name']] = {f: [float(r[f]) for r in rows] for f in fields}
        p, r = data[arm['name']]['ex_precision'], data[arm['name']]['ex_recall']
        data[arm['name']]['ex_f1_harmonic'] = [2*a*b/(a+b) if a+b else 0. for a, b in zip(p, r)]
        assert all(math.isfinite(v) for vs in data[arm['name']].values() for v in vs)
    names = list(data)
    def fmt(key, scale=1):
        return ' | '.join(f'{statistics.mean(data[n][key])*scale:.6f} +/- {statistics.stdev(data[n][key])*scale:.6f}' for n in names)
    lines = ['Four-state batch: full saved metrics (mean +/- sample SD over 5 overlapping sequence splits)',
             f'Source revision: {result["gitrev"]}', f'Root: {root}',
             'Columns: metric | original binary control | four-state head',
             'Valid-bond masking; fixed threshold 0.5. Both arms: folded train + full-spectrum internal validation.',
             'Spectrum F1 (primary) = harmonic of mean P/R within EACH split, then mean over splits.',
             'No four-state classification accuracy is available on outer test: it is evaluated on the historical union task.', '',
             'Parameters: control=' + str(smoke['arms']['control']['parameters']) + '; four_state=' + str(smoke['arms']['joint']['parameters']),
             'Structure: 3 residual GCN + 2 edge-gated GAT; hidden=256; heads=4; dropout=0.1; FiLM; global node; theory=15.',
             'Head: concat1536 -> Linear256 -> ReLU -> Dropout0.1 -> Linear4; q=1-P(none).',
             'Training: AdamW lr=1e-4 wd=1e-4; batch=128; max_epochs=300; warmup=40; cosine T=300; patience=25.', '',
             'PRIMARY METRICS (percent where marked; other metrics on native scale)']
    primary = [('subset_acc','Subset accuracy %',100), ('ex_acc','Spectrum Jaccard accuracy %',100),
               ('ex_precision','Spectrum precision %',100), ('ex_recall','Spectrum recall %',100),
               ('ex_f1_harmonic','Spectrum harmonic F1 %',100), ('accuracy','Bond accuracy %',100),
               ('precision','Bond precision %',100), ('recall','Bond recall %',100), ('f1','Bond micro F1 %',100),
               ('auc','ROC-AUC',1), ('pr_auc','PR-AUC',1), ('mcc','MCC',1), ('hamming_loss','Hamming loss',1),
               ('brier_score','Realized-label Brier',1), ('ece','ECE',1), ('q_brier','q-Brier',1),
               ('q_mae','q-MAE',1), ('q_rmse','q-RMSE',1), ('q_pearson','q-Pearson',1),
               ('q_spearman','q-Spearman key',1), ('q_spearman_pep','q-Spearman spectrum',1),
               ('q_spearman_pep_cond','q-Spearman condition',1), ('q_spearman_pep_seq','q-Spearman sequence',1),
               ('q_top10_enrichment_seq','q-Top10 enrichment sequence',1)]
    for key, label, scale in primary:
        lines.append(f'{label:34s} | {fmt(key, scale)}')
    lines += ['', 'ALL SAVED NUMERIC COLUMNS (native scale, including aliases and GPU-memory diagnostics)',
              'ex_f1 = mean of per-spectrum F1, a DIFFERENT definition from primary harmonic F1.',
              'class_0_* are legacy aliases of positive-class metrics in this source, NOT independent negative-class metrics.',
              'lab_acc_ma is masked macro by bond position; only accuracy/lab_acc_mi is the main bondacc.',
              'loss is the unweighted HARD union BCE used in evaluation, not the four-state training CE.',
              'GPU-memory columns are saved TEST-evaluation measurements, not training peak or throughput.']
    for key in data[names[0]]:
        lines.append(f'{key:34s} | {fmt(key)}')
    lines += ['', 'PER-SPLIT F1 (%): control -> four-state']
    for fold in ('1222','2252','3514','6072','9075'):
        a, b = [result['folds'][name][fold]['f1'] * 100 for name in names]
        lines.append(f'{fold}: {a:.6f} -> {b:.6f} ({b-a:+.6f} percentage points)')
    text = '\n'.join(lines) + '\n'
    (ops / 'all_metrics.txt').write_text(text, encoding='utf-8')
    (ops / 'all_metrics.json').write_text(json.dumps(data, indent=2) + '\n')
    print('\n'.join(lines[:len(primary)+16]))
    print(f'Full text: {ops / "all_metrics.txt"}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ops', default='/mnt/pvc/graphtrans/four_state_ops')
    export(p.parse_args().ops)
