"""Synthetic CPU-only tests; run on the isolated pod scratch repository."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import logging
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.graph_dataset import GraphDataset, CachedGraphDataset, GraphDataLoader
from data.ion12_dataset import CHANNELS, Ion12TargetStore, Ion12GraphDataset, resolve_ion12_manifest, sha256_file
from models.utils import ModelConfig
import train_graph_model as training_script


class Ion12DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        training_script.logger = logging.getLogger('ion12_dataset_test')

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='ion12_dataset_SYNTHETIC_')
        self.directory = Path(self.temporary.name)
        for name in ('original', 'folded', 'sidecar'):
            (self.directory / name).mkdir()
        filename = '1222.train.fbr.shuffle.multi.csv'
        self.original = self.directory / 'original' / filename
        self.folded = self.directory / 'folded' / filename
        self.manifest_path = self.directory / 'sidecar' / 'manifest.json'
        self.payload_path = self.manifest_path.with_name('targets.json')
        rows, records = [], []
        for seq in ('AGK', 'PEPTIDE'):
            bonds = len(seq) - 1
            rows.append({'name': seq, 'seq': seq, 'charge': 2, 'pep_mass': 500.0,
                         'intensity': 100.0, 'nce': 30.0, 'scan_num': 12, 'rt': 1.0,
                         'true_multi': ';'.join(['1'] * bonds), 'soft_multi': ';'.join(['0.3333'] * bonds),
                         'group_n': 3})
            slots = [[0] * 12 for _ in range(bonds)]
            for bond in range(bonds):
                slots[bond][bond % 12] = 1
            slots[-1][11] = 1  # overlapping channel events are intentionally legal
            records.append({'seq': seq, 'charge': 2, 'nce': 30.0, 'n': 3, 'valid_bonds': bonds,
                            'slot_hit_counts': slots, 'union_hit_counts': [1] * bonds})
        pd.DataFrame(rows).to_csv(self.folded, index=False)
        pd.DataFrame([row for row in rows for _ in range(3)]).to_csv(self.original, index=False)
        self.payload = {'schema_version': 1, 'fold': 1222, 'channels': CHANNELS.copy(), 'records': records}
        seq_hash = hashlib.sha256(b'AGK\nPEPTIDE\n').hexdigest()
        self.manifest = {'schema_version': 1, 'status': 'passed', 'fold': 1222, 'channels': CHANNELS.copy(),
            'source_original_train': {'path': str(self.original), 'sha256': sha256_file(self.original),
                                      'rows': 6, 'unique_sequences': 2, 'sequence_set_sha256': seq_hash},
            'source_folded_train': {'path': str(self.folded), 'sha256': sha256_file(self.folded), 'rows': 2},
            'sidecar': {'path': 'targets.json', 'records': 2},
            'union_manifest': {'path': 'unused_union_manifest.json', 'sha256': '0' * 64},
            'qa': {'exact_train_group_keys': True, 'group_n_exact': True,
                   'original_union_counts_exact': True, 'folded_q_max_abs_error': 1/3 - .3333,
                   'folded_q_atol': 5.0001e-5}}
        self.save()
        self.config = {'model': vars(ModelConfig()), 'data': {
            'max_seq_len': 8, 'graph_strategy': 'distance', 'use_soft_labels': True,
            'weighting_scheme': 'group_uniform', 'strict_label_mode': 'raise',
            'augmentation': False, 'cache_graphs': False,
            'cache_dir': str(self.directory / 'graph_cache'), 'train_csv_path': str(self.folded),
            'val_csv_path': None, 'test_csv_path': None,
            'ion12_targets_manifest': str(self.manifest_path)},
            'training': {'validation_split': 0.5}, 'experiment': {'seed': 42}}
        self.config['model'].update({'max_distance': 3, 'use_global_node': True})

    def tearDown(self):
        self.temporary.cleanup()

    def save(self):
        self.payload_path.write_text(json.dumps(self.payload), encoding='utf-8')
        self.manifest['sidecar']['sha256'] = sha256_file(self.payload_path)
        self.manifest_path.write_text(json.dumps(self.manifest), encoding='utf-8')

    def store(self):
        return Ion12TargetStore(self.manifest_path, 1222, self.folded)

    def parent(self, kind='ordinary', split='train'):
        kwargs = dict(csv_path=str(self.folded), config=self.config, max_seq_len=8,
                      graph_strategy='distance', augmentation=False, split=split)
        if kind == 'ordinary':
            return GraphDataset(**kwargs)
        return CachedGraphDataset(**kwargs, cache_dir=str(self.directory / kind),
                                  cache_full_graphs=(kind == 'full'), rebuild_cache=False)

    def assert_sample_equal(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left:
            if isinstance(left[key], torch.Tensor):
                self.assertTrue(torch.equal(left[key], right[key]), key)
            else:
                self.assertEqual(left[key], right[key], key)

    def test_targets_use_counts_and_fixed_channels(self):
        store = self.store()
        self.assertEqual(CHANNELS[11], 'y2-NH3')
        target = store.targets['AGK', 2, 30.0]
        self.assertEqual(tuple(target.shape), (2, 12))
        self.assertAlmostEqual(target[-1, 11].item(), 1/3, places=7)
        self.assertAlmostEqual(target[0, 0].item(), 1/3, places=7)

    def test_ordinary_and_cache_samples_unchanged(self):
        for kind in ('ordinary', 'edge', 'full'):
            with self.subTest(kind=kind):
                parent = self.parent(kind)
                wrapped = Ion12GraphDataset(parent, self.store())
                self.assertIs(wrapped.data, parent.data)
                for i in range(len(parent)):
                    item = wrapped[i]
                    target = item.pop('ion12_targets')
                    self.assert_sample_equal(parent[i], item)
                    target.zero_()
                    self.assertGreater(wrapped[i]['ion12_targets'].sum().item(), 0)
                if kind == 'full':
                    self.assertTrue(all('ion12_targets' not in x for x in parent.cached_data.values()))

    def test_collator_padding_and_legacy_mask(self):
        parent = self.parent()
        wrapped = Ion12GraphDataset(parent, self.store())
        collate = GraphDataLoader(parent, batch_size=2, num_workers=0)._default_collate_fn
        legacy = collate([parent[0], parent[1]])
        combined = collate([wrapped[0], wrapped[1]])
        targets = combined.pop('ion12_targets')
        self.assertEqual(tuple(targets.shape), (2, 6, 12))
        self.assertEqual(targets[0, 2:].count_nonzero().item(), 0)
        self.assert_sample_equal(legacy, combined)

    def test_collator_rejects_mixed_and_invalid_targets(self):
        parent = self.parent()
        wrapped = Ion12GraphDataset(parent, self.store())
        collate = GraphDataLoader(parent, batch_size=2, num_workers=0)._default_collate_fn
        with self.assertRaises(ValueError):
            collate([parent[0], wrapped[1]])
        for value in (torch.full((2, 12), float('nan')), torch.zeros((2, 11)), torch.full((2, 12), 1.01)):
            with self.subTest(shape=value.shape):
                item = wrapped[0]
                item['ion12_targets'] = value
                with self.assertRaises(ValueError):
                    collate([item, wrapped[1]])

    def test_manifest_bad_status_error_fold_schema_and_channels(self):
        original = copy.deepcopy(self.manifest)
        variants = [('status', 'building'), ('error', None), ('fold', 2252),
                    ('schema_version', True), ('channels', list(reversed(CHANNELS)))]
        for key, value in variants:
            with self.subTest(key=key):
                self.manifest = copy.deepcopy(original)
                self.manifest[key] = value
                self.save()
                with self.assertRaises(ValueError):
                    self.store()

    def test_source_and_sidecar_bytes_are_bound(self):
        for path in (self.original, self.folded, self.payload_path):
            with self.subTest(path=path.name):
                raw = path.read_bytes()
                path.write_bytes(raw + b'\n')
                with self.assertRaises(ValueError):
                    self.store()
                path.write_bytes(raw)

    def test_bad_counts_duplicate_keys_shape_and_nonfinite(self):
        original = copy.deepcopy(self.payload)
        mutations = [lambda p: p['records'][0]['slot_hit_counts'][0].__setitem__(0, -1),
                     lambda p: p['records'][0]['slot_hit_counts'][0].__setitem__(0, 4),
                     lambda p: p['records'][0]['slot_hit_counts'][0].__setitem__(0, True),
                     lambda p: p['records'][0]['slot_hit_counts'][0].__setitem__(0, float('nan')),
                     lambda p: p['records'][0]['union_hit_counts'].__setitem__(0, 0),
                     lambda p: p['records'][0].__setitem__('valid_bonds', 7),
                     lambda p: p['records'].__setitem__(1, copy.deepcopy(p['records'][0]))]
        for mutate in mutations:
            self.payload = copy.deepcopy(original)
            mutate(self.payload)
            self.save()
            with self.assertRaises(ValueError):
                self.store()

    def test_folded_subset_count_q_and_membership_guards(self):
        store = self.store()
        frame = self.parent().data.copy()
        store.validate_frame(frame.iloc[:1])
        with self.assertRaises(ValueError):
            store.validate_frame(frame.iloc[:1], require_full=True)
        for column, value in (('group_n', 4), ('seq', 'VVK'), ('soft_multi', '0.4;0.4')):
            bad = frame.copy()
            bad.loc[0, column] = value
            with self.assertRaises(ValueError):
                store.validate_frame(bad)
        with self.assertRaises(ValueError):
            store.validate_frame(frame.drop(columns=['group_n']))
        with self.assertRaises(ValueError):
            store.validate_frame(pd.concat([frame, frame.iloc[:1]]))

    def test_folded_q_rounding_tolerance_is_bounded(self):
        store = self.store()
        frame = self.parent().data.copy()
        frame.loc[0, 'soft_multi'] = '0.33338;0.33338'
        store.validate_frame(frame)
        frame.loc[0, 'soft_multi'] = '0.33339;0.33339'
        with self.assertRaises(ValueError):
            store.validate_frame(frame)

    def test_test_split_and_augmentation_cannot_attach(self):
        store = self.store()
        with self.assertRaises(ValueError):
            Ion12GraphDataset(self.parent(split='test'), store)
        parent = self.parent()
        parent.augmentation = True
        with self.assertRaises(ValueError):
            Ion12GraphDataset(parent, store)

    def test_canonical_fold_template(self):
        for fold in ('1222', '2252', '3514', '6072', '9075'):
            path, value = resolve_ion12_manifest(str(self.directory / 'fold_{fold}/manifest.json'),
                                               fold + '.train.fbr.shuffle.multi.csv')
            self.assertEqual(value, int(fold))
            self.assertEqual(path.parent.name, 'fold_' + fold)
        for name in ('train.csv', '1222.test.fbr.multi.csv', '9999.train.fbr.shuffle.multi.csv'):
            with self.assertRaises(ValueError):
                resolve_ion12_manifest('fold_{fold}/manifest.json', name)
        with self.assertRaises(ValueError):
            resolve_ion12_manifest('fold_{unknown}/manifest.json', self.folded)
        with self.assertRaises(ValueError):
            Ion12TargetStore(self.manifest_path, 2252, self.folded)

    def test_production_factory_internal_split_and_test_exclusion(self):
        test_path = self.directory / '1222.test.fbr.multi.csv'
        frame = pd.read_csv(self.folded).iloc[:1].copy()
        frame['seq'] = 'VVK'
        frame.to_csv(test_path, index=False)
        self.config['data']['test_csv_path'] = str(test_path)
        on_train, on_val, on_test = training_script.create_datasets(copy.deepcopy(self.config))
        off_config = copy.deepcopy(self.config)
        del off_config['data']['ion12_targets_manifest']
        off_train, off_val, off_test = training_script.create_datasets(off_config)
        self.assertIsInstance(on_train, Ion12GraphDataset)
        self.assertIsInstance(on_val, Ion12GraphDataset)
        self.assertNotIsInstance(on_test, Ion12GraphDataset)
        self.assertFalse(set(on_train.data.seq) & set(on_val.data.seq))
        for on, off in ((on_train, off_train), (on_val, off_val)):
            for i in range(len(on)):
                item = on[i]
                del item['ion12_targets']
                self.assert_sample_equal(item, off[i])
        self.assert_sample_equal(on_test[0], off_test[0])

    def test_factory_rejects_unsupported_data_modes(self):
        for key, value in (('preprocessed', True), ('augmentation', True), ('use_soft_labels', False)):
            config = copy.deepcopy(self.config)
            config['data'][key] = value
            with self.assertRaises(ValueError):
                training_script.create_datasets(config)

    def test_explicit_validation_cannot_overlap_training_sequences(self):
        val_path = self.directory / 'val.csv'
        pd.read_csv(self.folded).iloc[:1].to_csv(val_path, index=False)
        self.config['data']['val_csv_path'] = str(val_path)
        with self.assertRaisesRegex(ValueError, 'sequences overlap'):
            training_script.create_datasets(self.config)

    def test_provenance_file_checkpoint_and_resume_guard(self):
        dataset = Ion12GraphDataset(self.parent(), self.store())
        config = copy.deepcopy(self.config)
        config['model']['use_ion12_aux_head'] = True
        config['loss'] = {'ion12_aux_weight': 0.1}
        folder = self.directory / 'checkpoints'
        folder.mkdir()
        provenance = training_script._write_ion12_targets_provenance(dataset, config, str(folder))
        self.assertEqual(provenance['ion12_aux_weight'], 0.1)
        self.assertNotIn('ion12_aux_weight', dataset.ion12_provenance)
        self.assertEqual(json.loads((folder / 'ion12_targets_provenance.json').read_text()), provenance)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        for name in ('best_model.pt', 'best_model_q.pt', 'checkpoint_epoch_1.pt'):
            path = folder / name
            training_script.save_checkpoint_with_selection(
                model, optimizer, 0, 0.5, {}, str(path), None,
                extra_state={'ion12_targets_provenance': provenance})
            self.assertEqual(torch.load(path, weights_only=True)['extra_state']['ion12_targets_provenance'], provenance)
        with self.assertRaisesRegex(ValueError, 'resume is unsupported'):
            training_script._reject_ion12_resume(config, 'checkpoint.pt')
        training_script._reject_ion12_resume({}, 'legacy_checkpoint.pt')
        self.assertIsNone(training_script._write_ion12_targets_provenance(None, {}, str(folder)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
