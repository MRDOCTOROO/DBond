"""Pod-only: derive joint b/y state counts from verified per-spectrum ion matches.

No rematching, peak filtering or test CSV reads. The union cache is read-only;
each exported sidecar is restricted to one outer training set and independently
checked against that set's original hard labels, group sizes and folded q.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.row_validation import file_sha
from data.four_state_dataset import STATES
from data.ion12_dataset import CHANNELS, condition_key

FOLDS = (1222, 2252, 3514, 6072, 9075)


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, separators=(',', ':'), allow_nan=False)
        stream.write('\n')


def prepare(source_manifest, output, repo):
    source_manifest, output, repo = map(Path, (source_manifest, output, repo))
    source = json.loads(source_manifest.read_text())
    require(source['status'] == 'passed' and source['channels'] == CHANNELS
            and source['original_union_all_rows_exact'] and source['same_sequence_exact_record_sets']
            and source['all_fold_original_counts_and_n_exact'], 'Source match QA is incomplete')
    cache = Path(source['cache']['path'])
    require(file_sha(cache) == source['cache']['sha256'], 'Source cache SHA mismatch')
    output.mkdir(parents=True, exist_ok=False)
    db = sqlite3.connect(f'file:{cache.resolve()}?mode=ro', uri=True)
    groups = list(db.execute('SELECT group_id,seq,charge,nce,n,valid_bonds,union_hit_counts FROM condition_groups ORDER BY group_id'))
    require([r[0] for r in groups] == list(range(len(groups))), 'Non-dense group IDs')
    counts = np.zeros((len(groups), 31, 4), dtype=np.int64)
    seen = np.zeros(len(groups), dtype=np.int64)
    rows = collisions = 0
    for gid, bonds, slots_blob, union_bits, collision_n in db.execute(
            'SELECT group_id,valid_bonds,slot_ids,union_bits,collision_peak_count FROM spectra'):
        require(bonds == groups[gid][5] and 1 <= bonds <= 31, 'Spectrum/group length mismatch')
        slots = np.frombuffer(slots_blob, dtype='<u2')
        require(len(np.unique(slots)) == len(slots) and (slots < bonds * 12).all(), 'Invalid matched ion slots')
        hit = np.zeros((bonds, 12), dtype=bool)
        hit.reshape(-1)[slots] = True
        b = hit[:, [0, 1, 4, 5, 8, 9]].any(-1)
        y = hit[:, [2, 3, 6, 7, 10, 11]].any(-1)
        state = b.astype(np.int64) + 2 * y.astype(np.int64)
        union = np.array([(union_bits >> i) & 1 for i in range(bonds)], dtype=bool)
        require(np.array_equal(state > 0, union), 'Joint state OR differs from original spectrum union')
        counts[gid, np.arange(bonds), state] += 1
        seen[gid] += 1
        rows += 1
        collisions += collision_n
    db.close()
    require(rows == source['union_rows'], 'Source row coverage mismatch')
    group_map = {}
    for gid, seq, charge, nce, n, bonds, union_blob in groups:
        c = counts[gid, :bonds]
        require(seen[gid] == n and (c.sum(-1) == n).all(), 'Joint counts do not sum to group n')
        require(np.array_equal(c[:, 1:].sum(-1), np.frombuffer(union_blob, dtype='<u4')), 'Joint group OR mismatch')
        group_map[condition_key(seq, charge, nce)] = (n, c)
    reports = {}
    for fold in FOLDS:
        original = repo / f'dataset/5fold/{fold}.train.fbr.shuffle.multi.csv'
        folded = repo / f'dataset/5fold_folded/{fold}.train.fbr.shuffle.multi.csv'
        expected = source['train_sources'][str(fold)]
        require(file_sha(original) == expected['original']['sha256']
                and file_sha(folded) == expected['folded']['sha256'], 'Training sources differ from original matching manifest')
        raw = pd.read_csv(original, usecols=['seq', 'charge', 'nce', 'true_multi'])
        grouped = raw.groupby(['seq', 'charge', 'nce'], sort=False)
        frame = pd.read_csv(folded)
        require(len(frame) == len(grouped) and not frame.duplicated(['seq', 'charge', 'nce']).any(), 'Fold group keys mismatch')
        records = []
        max_error = 0.0
        for row in frame.itertuples():
            key = condition_key(str(row.seq), row.charge, row.nce)
            n, c = group_map[key]
            labels = np.array([str(v).split(';') for v in grouped.get_group(key)['true_multi']], dtype=int)
            q = np.asarray(str(row.soft_multi).split(';'), dtype=float)
            require(n == row.group_n == len(labels) and np.isin(labels, [0, 1]).all()
                    and np.array_equal(labels.sum(0), c[:, 1:].sum(-1)), 'Fold original labels/group counts mismatch')
            error = float(np.abs(q - c[:, 1:].sum(-1) / n).max())
            require(error <= 5.0001e-5, 'Folded union q mismatch')
            max_error = max(error, max_error)
            records.append({'seq': key[0], 'charge': key[1], 'nce': key[2], 'n': n, 'counts': c.tolist()})
        folder = output / f'fold_{fold}'
        folder.mkdir()
        target = folder / 'targets.json'
        write(target, {'schema_version': 1, 'fold': fold, 'states': STATES, 'records': records})
        manifest = {'schema_version': 1, 'status': 'passed', 'fold': fold, 'states': STATES,
                    'source_original_train': {'path': str(original), 'sha256': file_sha(original)},
                    'source_folded_train': {'path': str(folded), 'sha256': file_sha(folded)},
                    'source_union_manifest': {'path': str(source_manifest), 'sha256': file_sha(source_manifest)},
                    'sidecar': {'path': target.name, 'sha256': file_sha(target), 'records': len(records)},
                    'qa': {'spectrum_OR_exact': True, 'group_counts_exact': True,
                           'fold_original_union_exact': True, 'folded_q_max_abs_error': max_error},
                    'scope': 'Only this outer train whitelist; internal val/test never attach joint targets; no peak filtering'}
        write(folder / 'manifest.json', manifest)
        reports[str(fold)] = {'records': len(records), 'original_train_rows': len(raw),
                              'manifest_sha256': file_sha(folder / 'manifest.json'), 'folded_q_max_abs_error': max_error}
    require(file_sha(cache) == source['cache']['sha256'], 'Read-only source cache changed')
    report = {'status': 'passed', 'states': STATES, 'source_rows_verified': rows,
              'source_condition_groups': len(groups), 'source_collision_peaks_retained': collisions,
              'test_csv_read': False, 'original_labels_unchanged': True, 'folds': reports,
              'source_cache_sha256': source['cache']['sha256'], 'helper_sha256': file_sha(__file__)}
    write(output / 'preparation.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source_manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--repo', default=str(Path(__file__).resolve().parents[2]))
    a = p.parse_args()
    prepare(a.source_manifest, a.output, a.repo)
