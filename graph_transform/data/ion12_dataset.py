"""Manifest-bound training-only operational ion-channel frequency targets.

Targets are attached after ordinary graph/cache retrieval. They never enter a
graph cache or model input feature; the legacy sample fields stay untouched.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import pandas as pd
import torch
from torch.utils.data import Dataset

CHANNELS = [f'{series}{charge}' + ('' if loss == 'none' else '-' + loss)
            for loss in ('none', 'H2O', 'NH3') for series in ('b', 'y') for charge in (1, 2)]
FOLDS = ('1222', '2252', '3514', '6072', '9075')
FOLDED_Q_ATOL = 5.0001e-5


def _require(value, message):
    if not value:
        raise ValueError('ion12: ' + message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'duplicate JSON key ' + key)
            result[key] = value
        return result
    def reject(value):
        raise ValueError('ion12: nonfinite JSON constant ' + value)
    raw = Path(path).read_bytes()
    value = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=pairs, parse_constant=reject)
    _require(isinstance(value, dict), 'expected JSON object')
    _require('error' not in value and not value.get('errors'), 'error-bearing artifact')
    return value, hashlib.sha256(raw).hexdigest()


def _integer(value, context, minimum=0):
    _require(type(value) is int and value >= minimum, 'invalid integer ' + context)
    return value


def _finite(value, context):
    _require(not isinstance(value, bool), 'boolean ' + context)
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError('ion12: invalid number ' + context) from None
    _require(math.isfinite(result), 'nonfinite ' + context)
    return result


def condition_key(seq, charge, nce):
    _require(isinstance(seq, str) and len(seq) >= 2, 'invalid sequence')
    charge = _finite(charge, 'charge')
    _require(charge.is_integer() and charge > 0, 'invalid charge')
    return seq, int(charge), _finite(nce, 'nce')


def resolve_ion12_manifest(template, full_train_csv):
    """Resolve the fold from the original full train name, before internal split."""
    match = re.fullmatch(r'(' + '|'.join(FOLDS) + r')\.train\.fbr\.shuffle\.multi\.csv',
                         Path(full_train_csv).name)
    _require(match is not None, 'cannot identify canonical train fold from CSV filename')
    _require(isinstance(template, str) and bool(template.strip()), 'manifest path must be nonempty text')
    fold = match.group(1)
    value = template.replace('{fold}', fold)
    _require('{' not in value and '}' not in value, 'unsupported manifest path template')
    return Path(value).expanduser().resolve(), int(fold)


class Ion12TargetStore:
    """Load and validate one fold's immutable sidecar once, before data splitting."""

    def __init__(self, manifest_path, fold, full_train_csv):
        self.manifest_path = Path(manifest_path).resolve()
        self.full_train_csv = Path(full_train_csv).resolve()
        self.fold = int(fold)
        manifest, self.manifest_sha256 = _read_json(self.manifest_path)
        self.manifest = manifest
        _require(type(manifest.get('schema_version')) is int and manifest['schema_version'] == 1 and manifest.get('status') == 'passed',
                 'manifest schema/status is not passed v1')
        _require(type(manifest.get('fold')) is int and manifest['fold'] == self.fold
                 and str(self.fold) in FOLDS, 'manifest fold mismatch')
        _require(manifest.get('channels') == CHANNELS, 'channel order mismatch')
        qa = manifest.get('qa', {})
        _require(all(qa.get(key) is True for key in ('exact_train_group_keys', 'group_n_exact',
                                                    'original_union_counts_exact')), 'incomplete sidecar QA')
        tolerance = _finite(qa.get('folded_q_atol'), 'QA q tolerance')
        error = _finite(qa.get('folded_q_max_abs_error'), 'QA q error')
        _require(0 <= error <= tolerance <= FOLDED_Q_ATOL, 'invalid folded q QA bound')
        self.q_atol = FOLDED_Q_ATOL
        for name in ('source_original_train', 'source_folded_train'):
            source = manifest.get(name, {})
            path = Path(source.get('path', ''))
            if not path.is_absolute():
                path = self.manifest_path.parent / path
            _require(path.name == f'{fold}.train.fbr.shuffle.multi.csv', 'source train filename/fold mismatch')
            _require(sha256_file(path) == source.get('sha256'), name + ' file SHA mismatch')
            _integer(source.get('rows'), name + '.rows', 1)
        folded = manifest['source_folded_train']
        _require(sha256_file(self.full_train_csv) == folded['sha256'], 'configured folded train SHA mismatch')
        sidecar = manifest.get('sidecar', {})
        _require(sidecar.get('path') == 'targets.json', 'unexpected sidecar path')
        payload, self.sidecar_sha256 = _read_json(self.manifest_path.parent / 'targets.json')
        _require(self.sidecar_sha256 == sidecar.get('sha256'), 'sidecar SHA mismatch')
        _require(type(payload.get('schema_version')) is int and payload['schema_version'] == 1 and type(payload.get('fold')) is int
                 and payload['fold'] == self.fold and payload.get('channels') == CHANNELS,
                 'sidecar schema/fold/channels mismatch')
        records = payload.get('records')
        _require(isinstance(records, list) and len(records) > 0
                 and len(records) == sidecar.get('records') == folded['rows'], 'record count mismatch')
        self.records = {}
        self.targets = {}
        for record in records:
            _require(isinstance(record, dict) and 'error' not in record, 'invalid target record')
            _integer(record.get('charge'), 'record charge', 1)
            key = condition_key(record.get('seq'), record.get('charge'), record.get('nce'))
            _require(key not in self.records, 'duplicate condition group')
            n = _integer(record.get('n'), 'group n', 1)
            bonds = _integer(record.get('valid_bonds'), 'valid_bonds', 1)
            _require(bonds == len(key[0]) - 1, 'wrong bond alignment/length')
            slots, union = record.get('slot_hit_counts'), record.get('union_hit_counts')
            _require(isinstance(slots, list) and len(slots) == bonds
                     and isinstance(union, list) and len(union) == bonds, 'count shape mismatch')
            for counts, union_count in zip(slots, union):
                _require(isinstance(counts, list) and len(counts) == 12, 'wrong channel count')
                for value in [*counts, union_count]:
                    _require(_integer(value, 'ion count') <= n, 'count exceeds group n')
                _require(max(counts) <= union_count <= min(n, sum(counts)), 'impossible marginal/union counts')
            self.records[key] = record
            self.targets[key] = torch.tensor(slots, dtype=torch.float32) / n
        original = manifest['source_original_train']
        sequences = sorted({key[0] for key in self.records})
        seq_hash = hashlib.sha256(('\n'.join(sequences) + '\n').encode('utf-8')).hexdigest()
        _require(sum(record['n'] for record in records) == original['rows'], 'original row total mismatch')
        _require(original.get('unique_sequences') == len(sequences)
                 and original.get('sequence_set_sha256') == seq_hash, 'original sequence identity mismatch')
        frame = pd.read_csv(self.full_train_csv, usecols=['seq', 'charge', 'nce', 'group_n', 'soft_multi'])
        self.validate_frame(frame, require_full=True)
        self.provenance = {'manifest_path': str(self.manifest_path), 'manifest_sha256': self.manifest_sha256,
                           'sidecar_sha256': self.sidecar_sha256, 'fold': self.fold,
                           'source_original_train_sha256': original['sha256'],
                           'source_folded_train_sha256': folded['sha256'], 'channels': CHANNELS.copy()}

    def validate_frame(self, frame, require_full=False):
        _require({'seq', 'charge', 'nce', 'group_n', 'soft_multi'} <= set(frame.columns),
                 'requires condition-folded CSV with group_n and soft_multi')
        keys = []
        for row in frame[['seq', 'charge', 'nce', 'group_n', 'soft_multi']].itertuples(index=False, name=None):
            seq, charge, nce, group_n, soft = row
            key = condition_key(seq, charge, nce)
            _require(key in self.records, 'condition missing from this fold sidecar')
            record = self.records[key]
            count = _finite(group_n, 'CSV group_n')
            _require(count.is_integer() and count == record['n'], 'CSV group_n mismatch')
            q = [_finite(value, 'CSV soft_multi') for value in str(soft).split(';')]
            _require(len(q) == record['valid_bonds'] and all(0 <= v <= 1 for v in q), 'invalid folded q shape/range')
            _require(all(abs(value - hits / record['n']) <= self.q_atol
                         for value, hits in zip(q, record['union_hit_counts'])), 'folded q differs from original union frequency')
            keys.append(key)
        _require(len(keys) > 0 and len(set(keys)) == len(keys), 'empty or duplicate folded condition rows')
        if require_full:
            _require(set(keys) == set(self.records), 'full folded train group set mismatch')
        return keys


class Ion12GraphDataset(Dataset):
    """Delegate all legacy sample construction, then append a separate target."""

    def __init__(self, parent, targets):
        from .graph_dataset import GraphDataset
        _require(isinstance(parent, GraphDataset), 'only ordinary/cached GraphDataset is supported')
        _require(parent.split in ('train', 'val'), 'test data must never attach ion12 targets')
        _require(not parent.augmentation and parent.augmentor is None, 'augmentation is unsupported')
        _require(parent.use_soft_labels, 'condition-group soft labels must be enabled')
        self.parent = parent
        self.ion12_store = targets
        self.ion12_provenance = targets.provenance
        self._ion12_keys = targets.validate_frame(parent.data,
            require_full=Path(parent.csv_path).resolve() == targets.full_train_csv)

    def __getattr__(self, name):
        if name == 'parent':
            raise AttributeError(name)
        return getattr(self.parent, name)

    def __len__(self):
        return len(self.parent)

    def __getitem__(self, index):
        sample = dict(self.parent[index])
        key = self._ion12_keys[index]
        _require(condition_key(sample['sequence'], sample['charge'], sample['nce']) == key,
                 'parent sample changed condition identity')
        _require('ion12_targets' not in sample, 'parent/cache already contains ion12 targets')
        sample['ion12_targets'] = self.ion12_store.targets[key].clone()
        return sample
