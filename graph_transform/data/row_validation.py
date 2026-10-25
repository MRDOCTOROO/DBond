"""Restore full-spectrum validation without changing the folded training split."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def restore_row_validation(folded, val_sequences, source_path):
    """The source must be this outer split's original *training* CSV.

    Validate all group sizes and label means before selecting validation rows.
    Never obtain internal validation from the outer test file. Preserve original
    row-specific precursor m/z and every spectrum's hard labels.
    """
    before = file_sha(source_path)
    raw = pd.read_csv(source_path)
    keys = ['seq', 'charge', 'nce']
    required = {*keys, 'true_multi', 'soft_multi'}
    if not required <= set(raw) or not (required | {'group_n'}) <= set(folded):
        raise ValueError('Full-row validation requires original soft-label and folded training CSVs')
    if set(raw['seq']) != set(folded['seq']):
        raise ValueError('Validation source sequence set differs from outer folded train')
    if folded.duplicated(keys).any():
        raise ValueError('Duplicate folded condition key')
    grouped = raw.groupby(keys, sort=False, dropna=False)
    reference = folded.set_index(keys)
    if set(grouped.groups) != set(reference.index):
        raise ValueError('Validation source condition keys differ from folded train')
    for key, frame in grouped:
        row = reference.loc[key]
        labels = np.array([str(v).split(';') for v in frame['true_multi']], dtype=float)
        q = np.array(str(row['soft_multi']).split(';'), dtype=float)
        if (labels.shape != (len(frame), len(str(key[0])) - 1)
                or not np.isin(labels, [0, 1]).all()
                or int(row['group_n']) != row['group_n'] or int(row['group_n']) != len(frame)
                or not np.isfinite(q).all() or q.shape != labels.shape[1:]
                or not np.allclose(labels.mean(axis=0), q, rtol=0, atol=5.0001e-5)
                or set(frame['soft_multi'].astype(str)) != {str(row['soft_multi'])}):
            raise ValueError(f'Validation source count/label/q mismatch: {key}')
    val_sequences = set(val_sequences)
    if not val_sequences or not val_sequences < set(raw['seq']):
        raise ValueError('Invalid internal validation sequence subset')
    result = raw[raw['seq'].isin(val_sequences)].reset_index(drop=True)
    if file_sha(source_path) != before:
        raise ValueError('Validation source changed while reading')
    provenance = {'mode': 'full_original_training_spectra_by_internal_val_sequence',
                  'source_path': str(Path(source_path).resolve()), 'source_sha256': before,
                  'outer_train_rows': len(raw), 'val_rows': len(result),
                  'val_sequences': sorted(val_sequences), 'folded_train_rows': len(folded),
                  'outer_test_used_for_internal_validation': False,
                  'preserve_row_inputs_and_hard_labels': True}
    return result, provenance
