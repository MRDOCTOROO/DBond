"""Empirical training-only count histograms from source-bound original rows."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .ion12_dataset import condition_key
from .row_validation import file_sha


class CardinalityDataset(Dataset):
    def __init__(self, parent):
        self.parent = parent
        source = parent.four_state_provenance['source_original_train']
        path = Path(source['path'])
        if file_sha(path) != source['sha256']:
            raise ValueError('Count target original source changed')
        frame = pd.read_csv(path)
        wanted = {condition_key(str(r.seq), r.charge, r.nce): int(r.group_n) for r in parent.data.itertuples()}
        if len(wanted) != len(parent):
            raise ValueError('Duplicate training count conditions')
        counts = {k: np.zeros(len(k[0]), dtype=np.int64) for k in wanted}
        for r in frame.itertuples():
            key = condition_key(str(r.seq), r.charge, r.nce)
            if key not in wanted:
                continue  # Inner validation conditions are never used as targets.
            labels = np.array(str(r.true_multi).split(';'), dtype=np.int64)
            if labels.shape != (len(key[0])-1,) or not np.isin(labels, [0, 1]).all():
                raise ValueError('Invalid original union labels')
            counts[key][int(labels.sum())] += 1
        self.histograms = []
        canonical = []
        for r in parent.data.itertuples():
            key = condition_key(str(r.seq), r.charge, r.nce)
            hist = counts[key]
            if int(hist.sum()) != wanted[key]:
                raise ValueError('Original row count differs from group_n')
            self.histograms.append(torch.tensor(hist / wanted[key], dtype=torch.float32))
            canonical.append([list(key), hist.tolist()])
        target_sha = hashlib.sha256(json.dumps(canonical, separators=(',', ':')).encode()).hexdigest()
        self.four_state_provenance = dict(parent.four_state_provenance)
        self.four_state_provenance['cardinality'] = dict(schema=1, original_source=source,
            target_sha256=target_sha, train_conditions=len(parent),
            target='empirical hard union count histogram; only inner training conditions',
            objective='CE(Poisson-binomial(existing union p), histogram)/valid_length, uniform condition mean')

    def __getattr__(self, name):
        if name == 'parent':
            raise AttributeError(name)
        return getattr(self.parent, name)

    def __len__(self):
        return len(self.parent)

    def __getitem__(self, i):
        sample = dict(self.parent[i])
        sample['cardinality_histogram'] = self.histograms[i].clone()
        return sample

