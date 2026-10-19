"""Manifest-bound four-state count targets; training targets never model inputs."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .row_validation import file_sha
from .ion12_dataset import condition_key

STATES = ['none', 'b_only', 'y_only', 'both']


class FourStateStore:
    def __init__(self, manifest_path, full_train_csv):
        path = Path(manifest_path).resolve()
        manifest = json.loads(path.read_text())
        if (manifest.get('schema_version') != 1 or manifest.get('status') != 'passed'
                or manifest.get('states') != STATES):
            raise ValueError('Invalid four-state manifest')
        qa = manifest.get('qa', {})
        if not all(qa.get(k) is True for k in ('spectrum_OR_exact', 'group_counts_exact', 'fold_original_union_exact')):
            raise ValueError('Incomplete four-state target QA')
        source = manifest['source_folded_train']
        if (Path(full_train_csv).resolve() != Path(source['path']).resolve()
                or file_sha(full_train_csv) != source['sha256']):
            raise ValueError('four-state outer folded train source mismatch')
        sidecar = path.parent / manifest['sidecar']['path']
        if file_sha(sidecar) != manifest['sidecar']['sha256']:
            raise ValueError('four-state sidecar SHA mismatch')
        payload = json.loads(sidecar.read_text())
        if (payload.get('schema_version') != 1 or payload.get('states') != STATES
                or payload.get('fold') != manifest['fold']):
            raise ValueError('four-state sidecar semantics mismatch')
        self.records = {}
        for record in payload['records']:
            key = condition_key(record['seq'], record['charge'], record['nce'])
            counts = np.asarray(record['counts'], dtype=np.float64)
            n = record['n']
            if (key in self.records or type(n) is not int or n <= 0
                    or counts.shape != (len(key[0]) - 1, 4) or not np.isfinite(counts).all()
                    or (counts < 0).any() or not np.equal(counts, np.floor(counts)).all()
                    or not (counts.sum(-1) == n).all()):
                raise ValueError('Invalid four-state integer counts')
            self.records[key] = (n, counts)
        if len(self.records) != manifest['sidecar']['records']:
            raise ValueError('four-state record count mismatch')
        self.provenance = {'manifest': str(path), 'manifest_sha256': file_sha(path),
                           'sidecar_sha256': file_sha(sidecar), 'states': STATES,
                           'fold': manifest['fold'], 'source_folded_train': source,
                           'source_original_train': manifest['source_original_train'],
                           'target_rounding': 'Preserve legacy serialized q; retain observed positive-state proportions',
                           'loss': 'soft four-state CE, valid-bond weighted mean, group_uniform'}

    def validate_frame(self, frame, require_all=False):
        keys = [condition_key(str(r.seq), r.charge, r.nce) for r in frame.itertuples()]
        if len(set(keys)) != len(keys) or not set(keys) <= self.records.keys():
            raise ValueError('four-state training frame duplicate/unknown conditions')
        if require_all and set(keys) != self.records.keys():
            raise ValueError('four-state sidecar must match exactly the outer training conditions')
        targets = []
        for row, key in zip(frame.itertuples(), keys):
            n, counts = self.records[key]
            q = np.asarray(str(row.soft_multi).split(';'), dtype=float)
            if (row.group_n != n or q.shape != (len(key[0]) - 1,) or not np.isfinite(q).all()
                    or not np.allclose(1 - counts[:, 0] / n, q, atol=5.0001e-5, rtol=0)):
                raise ValueError('four-state group_n or union-q mismatch')
            # Current folded q is serialized to 4 decimals. Preserve it exactly
            # for both arms; only the allocation among positive states is new.
            positive = counts[:, 1:]
            hit_n = positive.sum(-1, keepdims=True)
            conditional = np.divide(positive, hit_n, out=np.zeros_like(positive), where=hit_n > 0)
            if ((q < 0) | (q > 1)).any() or ((hit_n[:, 0] == 0) & (q != 0)).any():
                raise ValueError('Inconsistent positive-state counts/q')
            targets.append(torch.tensor(np.column_stack([1 - q, q[:, None] * conditional]), dtype=torch.float32))
        return targets


class FourStateDataset(Dataset):
    def __init__(self, parent, store, require_all=False):
        if parent.split != 'train' or parent.augmentation:
            raise ValueError('four-state targets attach only to unaugmented folded training')
        self.parent = parent
        self.four_state_provenance = store.provenance
        self.targets = store.validate_frame(parent.data, require_all)

    def __getattr__(self, name):
        if name == 'parent':
            raise AttributeError(name)
        return getattr(self.parent, name)

    def __len__(self):
        return len(self.parent)

    def __getitem__(self, index):
        sample = dict(self.parent[index])
        row = self.parent.data.iloc[index]
        if condition_key(sample['sequence'], sample['charge'], sample['nce']) != condition_key(str(row.seq), row.charge, row.nce):
            raise ValueError('four-state parent sample condition identity changed')
        sample['four_state_targets'] = self.targets[index].clone()
        return sample
