"""Pod-only real-batch GPU smoke for the Stage 13 A/B/C training arms.

Reads each internal sequence split, but computes no outer-test metrics and
runs only bounded optimizer steps. It is a gate, not an experiment result.
"""

import argparse
import hashlib
import json
import logging
import sys
import tempfile
import traceback
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'graph_transform'), str(ROOT / 'graph_transform/scripts')]

from models.graph_transformer import GraphTransformer
from models.utils import build_model_config
from training.loss_functions import BinaryBondLoss
from training.trainer import Trainer
from test_four_state_width import collate, longest_indices, memory, smoke_update, to_cuda
from train_5fold import _stable_fold_seed, make_fold_config
from train_graph_model import create_datasets, create_optimizer, create_scheduler
from stage13_multiscale_preflight import CONFIGS, FOLDS, check_configs, require


LOGGER = logging.getLogger('stage13_multiscale_gpu_smoke')


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def run(gpu):
    require(sys.platform.startswith('linux') and str(ROOT).startswith('/mnt/pvc/'),
            'Stage 13 data/GPU smoke must run on the authorized pod')
    require(torch.cuda.is_available(), 'CUDA unavailable on pod')
    torch.cuda.set_device(gpu)
    torch.set_num_threads(2)
    configs = check_configs()
    splits = {}
    resources = {}
    with tempfile.TemporaryDirectory(prefix='stage13_multiscale_smoke_') as tmp:
        first_train = first_val = None
        for fold in FOLDS:
            runtime, _, _ = make_fold_config(
                configs['S13-A'], fold, str(ROOT / 'dataset/5fold_folded'),
                str(Path(tmp) / fold), _stable_fold_seed(42, fold))
            require(runtime['training']['validation_split_seed'] == _stable_fold_seed(42, fold)
                    and runtime['experiment']['seed'] == _stable_fold_seed(42, fold),
                    f'{fold}: train/split seed mismatch')
            # Bounded smoke uses disposable cache settings. Formal runs use the
            # unchanged frozen A/B/C configuration and their own cache roots.
            runtime['data']['cache_graphs'] = False
            runtime['data']['rebuild_cache'] = False
            runtime['data']['cache_dir'] = str(Path(tmp) / fold / 'cache')
            train, val, test = create_datasets(runtime)
            sequences = [set(dataset.data['seq']) for dataset in (train, val, test)]
            require(not (sequences[0] & sequences[1]
                         or sequences[0] & sequences[2]
                         or sequences[1] & sequences[2]),
                    f'{fold}: sequence overlap among train, validation, and test')
            provenance = val.row_validation_provenance
            require(provenance['outer_test_used_for_internal_validation'] is False
                    and provenance['split_seed'] == _stable_fold_seed(42, fold),
                    f'{fold}: invalid internal validation provenance')
            splits[fold] = {
                'training_seed': runtime['experiment']['seed'],
                'validation_split_seed': runtime['training']['validation_split_seed'],
                'train_rows': len(train), 'val_rows': len(val), 'test_rows': len(test),
                'train_sequences': len(sequences[0]),
                'val_sequences': len(sequences[1]),
                'test_sequences': len(sequences[2]),
                'train_csv_sha256': file_sha(train.csv_path),
                'val_csv_sha256': file_sha(val.csv_path),
                'test_csv_sha256': file_sha(test.csv_path),
                'folded_training_source': runtime['data']['train_csv_path'],
                'folded_training_source_sha256':
                    file_sha(runtime['data']['train_csv_path']),
                'test_csv': runtime['data']['test_csv_path'],
                'validation_source_csv': runtime['data']['validation_source_csv_path'],
                'validation_source_sha256': provenance['source_sha256'],
                'target_manifest': runtime['data']['four_state_targets_manifest'],
                'target_manifest_sha256': file_sha(runtime['data']['four_state_targets_manifest']),
            }
            if first_train is None:
                first_train, first_val = train, val
        require(len(first_train) >= 128 and len(first_val) >= 512,
                'First fold lacks a complete train128/validation512 smoke batch')
        train_batch = to_cuda(collate(first_train, longest_indices(first_train, 128)))
        val_batch = to_cuda(collate(first_val, longest_indices(first_val, 512)))
        require('four_state_targets' not in val_batch,
                'Validation batch unexpectedly contains training-only four-state targets')
        for arm, config in configs.items():
            torch.cuda.empty_cache()
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            model = GraphTransformer(build_model_config(config)).cuda(gpu)
            optimizer = create_optimizer(model, config)
            trainer = Trainer(model, optimizer, BinaryBondLoss(config['loss']),
                              create_scheduler(optimizer, config),
                              torch.device(f'cuda:{gpu}'), config, LOGGER)
            require(trainer.use_amp
                    and trainer.four_state_conditional_weight == 1.
                    and trainer.four_state_union_brier_weight == 0.,
                    f'{arm}: original AMP/four-state loss route changed')
            torch.cuda.reset_peak_memory_stats(gpu)
            updates = [smoke_update(model, trainer, train_batch),
                       smoke_update(model, trainer, train_batch)]
            train_memory = memory()
            model.eval()
            torch.cuda.reset_peak_memory_stats(gpu)
            with torch.no_grad():
                prediction = model(val_batch)
            require(prediction.shape == val_batch['label_mask'].shape
                    and torch.isfinite(prediction).all()
                    and model.bond_head_type == 'four_state_mlp'
                    and model.last_four_state_logits is None,
                    f'{arm}: validation512 four-state forward failed')
            validation_memory = memory()
            resources[arm] = {
                'config_sha256': file_sha(ROOT / CONFIGS[arm]),
                'parameters': sum(parameter.numel() for parameter in model.parameters()),
                'train128_optimizer_steps': 2,
                'train128_losses': [update['loss'] for update in updates],
                'train_peak_allocated_bytes': train_memory['allocated_bytes'],
                'train_peak_reserved_bytes': train_memory['reserved_bytes'],
                'validation512_peak_allocated_bytes': validation_memory['allocated_bytes'],
                'validation512_peak_reserved_bytes': validation_memory['reserved_bytes'],
                'validation_finite': True,
            }
            del model, optimizer, trainer, prediction
    require(resources['S13-A']['parameters'] == resources['S13-C']['parameters']
            and resources['S13-B']['parameters'] < resources['S13-C']['parameters'],
            'Real model parameter count mismatch')
    return {'status': 'passed', 'gpu': gpu,
            'gpu_name': torch.cuda.get_device_name(gpu),
            'folds': splits, 'resources': resources,
            'outer_test_metrics_computed': False,
            'D_alias_of_A': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        result = run(args.gpu)
    except Exception as error:
        result = {'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                  'traceback': traceback.format_exc()}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        raise
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n',
                                 encoding='utf-8')
    print(json.dumps({'status': result['status'], 'output': args.output}))
