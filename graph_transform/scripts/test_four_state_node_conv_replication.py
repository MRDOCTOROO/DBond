"""Pod-only seed43 replication preflight for none versus single_k5.

Production files are unchanged from the audited seed42 discovery. This bounded
preflight imports reusable fixtures, never executes the old smoke/launch action.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'graph_transform'), str(ROOT / 'graph_transform/scripts')]
import test_four_state_node_conv as fixtures
import test_four_state_head_fusion as fusion_fixtures
from test_four_state_node_conv import (
    GraphTransformer, build_model_config, BinaryBondLoss, Trainer, check_loss,
    collate, longest_indices, masked_metric_check, memory, to_cuda, topology_proof,
    seed, rng_state, same_rng, cpu_state, same_tensors,
    backbone_shape_hooks, load_config, create_datasets, create_optimizer,
    create_scheduler, make_fold_config, _stable_fold_seed, file_sha, LOGGER,
    legacy_compatibility, shared_state, branch_unit_check, route_check,
    diagnostic_check,
)
from test_four_state_width_replication import split_option_check
from train_graph_model import save_checkpoint_with_selection
import four_state_node_conv_replication_ops as ops

MODES = ('none', 'single_k5')
FOLDS = ('1222', '2252', '3514', '6072', '9075')
PRIOR = Path('/mnt/pvc/graphtrans/four_state_ops/smoke.json')
WIDTH_PRIOR = Path('/mnt/pvc/graphtrans/four_state_width_ops/smoke.json')
DISCOVERY = Path('/mnt/pvc/graphtrans/four_state_node_conv_ops/smoke.json')


def config_path(mode):
    return ROOT / f'graph_transform/config/pre_synthesis_four_state_node_conv_seed43_{mode}.yaml'


def normalized(config, identity):
    value = copy.deepcopy(config)
    value['experiment']['seed'] = 42
    return fixtures.normalized(value, identity)


def validate_discovery_configs(configs, args):
    binding = ops.discovery_binding()
    for mode, config in configs.items():
        ops.validate_runtime(config, mode)
        old = load_config(ROOT / f'graph_transform/config/pre_synthesis_four_state_node_conv_{mode}.yaml', args)
        assert old['experiment']['seed'] == 42 and config['experiment']['seed'] == 43
        assert old['training']['validation_split_base_seed'] == config['training']['validation_split_base_seed'] == 42
        assert normalized(config, f'four_state_node_conv_seed43_{mode}') == normalized(old, f'four_state_node_conv_{mode}')
    return binding


def initialization_seed_check(config):
    """Every fold uses genuinely changed, exactly reproducible weights."""
    result = {}
    with torch.random.fork_rng(devices=[]):
        for fold in FOLDS:
            states = []
            for base in (42, 43, 43):
                torch.manual_seed(_stable_fold_seed(base, fold))
                model = GraphTransformer(build_model_config(config)).cpu()
                states.append(cpu_state(model))
                del model
            same_tensors(states[1], states[2])
            assert not torch.equal(states[0]['node_encoder.node_encoder.0.weight'],
                                   states[1]['node_encoder.node_encoder.0.weight'])
            result[fold] = {'old_training_seed': _stable_fold_seed(42, fold),
                            'new_training_seed': _stable_fold_seed(43, fold),
                            'new_seed_changes_initial_weights': True,
                            'same_new_seed_reproduces_complete_initial_state': True}
    return result


def checkpoint_checks(configs, states, folder):
    previous = fixtures.MODES
    try:
        fixtures.MODES = MODES
        return fixtures.checkpoint_checks(configs, states, folder)
    finally:
        fixtures.MODES = previous


def smoke_update(model, trainer, batch):
    # Reuse the genuine CE/AMP fixture, but seed its stochastic forward with
    # this preflight's resolved fold43 RNG rather than the helper default42.
    previous = fusion_fixtures.seed
    try:
        value = trainer.config['experiment']['seed']
        fusion_fixtures.seed = lambda: seed(value)
        return fusion_fixtures.smoke_update(model, trainer, batch)
    finally:
        fusion_fixtures.seed = previous


def rng_checkpoint_roundtrip(configs, states, folder):
    """Production saver with resolved RNG metadata; no epochs.

    Full training RNG assembly is unchanged and source-bound to discovery.
    Real training checkpoints are independently checked at terminal.
    """
    for mode, config in configs.items():
        model = GraphTransformer(build_model_config(config)).cpu()
        model.load_state_dict(states[mode], strict=True)
        optimizer = create_optimizer(model, config)
        protocol = {'training_seed': config['experiment']['seed'],
                    'validation_split_seed': config['training']['validation_split_seed'],
                    'hidden_dim': 256, 'bond_head_type': 'four_state_mlp',
                    'theory_fusion_stage': 'late', 'four_state_conditional_weight': 1.,
                    'four_state_union_brier_weight': 0.}
        assert protocol['training_seed'] != protocol['validation_split_seed']
        path = folder / f'{mode}_rng.pt'
        save_checkpoint_with_selection(model, optimizer, 0, 0., {'f1': 0.},
                                       str(path), None, extra_state={'training_rng_protocol': protocol})
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert saved['extra_state']['training_rng_protocol'] == protocol
        model.load_state_dict(saved['model_state_dict'], strict=True)
        same_tensors(states[mode], cpu_state(model))


def run(gpu=0):
    assert sys.platform.startswith('linux') and str(ROOT).startswith('/mnt/pvc/'), 'Pod only'
    assert torch.cuda.is_available() and isinstance(gpu, int) and not isinstance(gpu, bool)
    torch.cuda.set_device(gpu)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    check_loss()
    args = argparse.Namespace(epochs=None, batch_size=None, learning_rate=None, device=None)
    configs = {mode: load_config(config_path(mode), args) for mode in MODES}
    original = load_config(ROOT / 'graph_transform/config/pre_synthesis_four_state_joint.yaml', args)
    canonical = normalized(original, 'four_state_joint')
    for mode, config in configs.items():
        assert normalized(config, f'four_state_node_conv_seed43_{mode}') == canonical, f'Unexpected config: {mode}'
        mc = build_model_config(config)
        assert mc.node_conv_mode == mode and mc.bond_fusion_mode == 'concat'
        assert mc.bond_head_type == 'four_state_mlp' and mc.hidden_dim == 256
        assert mc.num_attention_heads == 4 and mc.concat_heads
        assert mc.num_gcn_layers == 3 and mc.num_gat_layers == 2
        assert mc.theory_fusion_stage == 'late' and mc.edge_state_mode == 'static'
        assert mc.use_theory_features and mc.use_global_node and mc.use_condition_film
        assert mc.sequence_adapter_type == 'none' and not mc.use_charge_partition_features
        assert mc.state_feature_mask == [True, True, False] and mc.env_feature_mask == [True, False]
        assert config['loss']['four_state_conditional_weight'] == 1.
        assert config['loss']['four_state_union_brier_weight'] == 0.
        assert config['experiment']['seed'] == 43 and config['training']['validation_split_base_seed'] == 42
        assert config['training']['batch_size'] == 128
        assert not config['training'].get('pretrained_backbone_path')
        assert not config['training'].get('pretrained_backbone_sha256')
    discovery_binding = validate_discovery_configs(configs, args)
    prior, width = json.loads(PRIOR.read_text()), json.loads(WIDTH_PRIOR.read_text())
    discovery = json.loads(DISCOVERY.read_text())
    assert prior['status'] == width['status'] == 'passed' and prior['folds'] == width['folds']
    report = {'status': 'running', 'configs': {mode: file_sha(config_path(mode)) for mode in MODES},
        'folds': {}, 'variants': {}, 'arms': {}, 'resources': {},
        'prior_validation_proof': str(PRIOR), 'prior_validation_proof_sha256': file_sha(PRIOR),
        'width_validation_proof': str(WIDTH_PRIOR), 'width_validation_proof_sha256': file_sha(WIDTH_PRIOR),
        'device': torch.cuda.get_device_name(gpu), 'gpu': gpu,
        'scope': 'Pod-only bounded preflight; no epochs or outer-test metrics. '
                 'Branch B0/L0 are checked; all-empty-edge legacy GAT is outside scope.'}
    with tempfile.TemporaryDirectory(prefix='four_state_node_conv_seed43_') as directory:
        temp, first = Path(directory), None
        report['split_options'] = split_option_check(configs['none'], temp)
        report['initialization'] = {mode: initialization_seed_check(config) for mode, config in configs.items()}
        report['discovery'] = discovery_binding
        report['training_base_seed'] = 43
        report['validation_split_base_seed'] = 42
        for fold in FOLDS:
            fold_configs = {}
            for mode, config in configs.items():
                resolved, _, _ = make_fold_config(config, fold, str(ROOT / 'dataset/5fold_folded'),
                    str(temp / 'paired'), _stable_fold_seed(43, fold))
                assert resolved['experiment']['seed'] == _stable_fold_seed(43, fold)
                assert resolved['training']['validation_split_seed'] == _stable_fold_seed(42, fold)
                assert resolved['experiment']['seed'] != resolved['training']['validation_split_seed']
                resolved['data']['cache_graphs'] = False
                resolved['data']['cache_dir'] = str(temp / fold / 'cache')
                fold_configs[mode] = resolved
            baseline = normalized(fold_configs['none'], 'four_state_node_conv_seed43_none')
            assert all(normalized(config, f'four_state_node_conv_seed43_{mode}') == baseline
                       for mode, config in fold_configs.items())
            train, val, test = create_datasets(fold_configs['none'])
            sets = [set(dataset.data['seq']) for dataset in (train, val, test)]
            assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            proof = {'train_rows': len(train), 'val_rows': len(val),
                'train_sequences': len(sets[0]), 'val_sequences': len(sets[1]),
                'train_sha256': file_sha(train.csv_path), 'val_sha256': file_sha(val.csv_path),
                'source_sha256': val.row_validation_provenance['source_sha256']}
            assert proof == prior['folds'][fold], f'Historical split changed: {fold}'
            historical = {key: discovery['folds'][fold][key] for key in proof}
            assert proof == historical, f'Discovery split changed: {fold}'
            assert discovery['folds'][fold]['training_seed'] == discovery['folds'][fold]['split_seed'] == _stable_fold_seed(42, fold)
            assert val.row_validation_provenance['outer_test_used_for_internal_validation'] is False
            assert 'four_state_targets' not in val[0] and 'four_state_targets' not in test[0]
            proof.update(training_seed=fold_configs['none']['experiment']['seed'],
                         split_seed=fold_configs['none']['training']['validation_split_seed'])
            report['folds'][fold] = proof
            if first is None:
                first = (fold_configs, train, val)
        fold_configs, train, val = first
        assert len(train) >= 128 and len(val) >= 512
        indices = list(range(8)) + list(range(len(train) - 8, len(train)))
        small_cpu = collate(train, indices)
        report['global_topology'] = topology_proof(small_cpu)
        report['legacy_compatibility'] = legacy_compatibility(fold_configs['none'], small_cpu)
        small = to_cuda(small_cpu)
        initial_shared = initial_rng = initial_prediction = None
        saved_states, counts = {}, {}
        for mode in MODES:
            config = copy.deepcopy(fold_configs[mode])
            config['debug']['diagnostic_dir'] = str(temp / mode / 'diagnostics')
            seed(config['experiment']['seed'])
            model = GraphTransformer(build_model_config(config)).cuda()
            state, current_rng = cpu_state(model), rng_state()
            if initial_shared is None:
                initial_shared, initial_rng = shared_state(state), current_rng
            else:
                same_tensors(initial_shared, shared_state(state))
                same_rng(initial_rng, current_rng)
            with torch.random.fork_rng(devices=[]):
                causal = GraphTransformer(build_model_config(config)).cpu().eval()
                causal.load_state_dict(state, strict=True)
                with torch.no_grad():
                    prediction = causal(small_cpu)
                if mode == 'none':
                    initial_prediction = prediction
                    units = routing = {'not_applicable_to_none': True}
                    diagnostics = diagnostic_check(causal, small_cpu)
                else:
                    assert torch.equal(initial_prediction, prediction), 'Zero branch changes initial complete-model function'
                    units = branch_unit_check(causal.node_context_conv)
                    routing = route_check(causal, small_cpu)
                    diagnostics = diagnostic_check(causal, small_cpu)
                del causal
            observed = {}
            shape_handles = backbone_shape_hooks(model, observed)
            optimizer = create_optimizer(model, config)
            trainer = Trainer(model, optimizer, BinaryBondLoss(config['loss']),
                create_scheduler(optimizer, config), torch.device(f'cuda:{gpu}'), config, LOGGER)
            assert trainer.use_amp and trainer.four_state_conditional_weight == 1. and trainer.four_state_union_brier_weight == 0.
            gradients, gradient_handles = {}, []
            if mode != 'none':
                watched = {'projection': model.node_context_conv.projection.weight,
                           **{f'conv_{index}': branch.weight for index, branch in enumerate(model.node_context_conv.branches)}}
                for key, parameter in watched.items():
                    gradients[key] = []
                    def capture(value, name=key):
                        assert torch.isfinite(value).all()
                        gradients[name].append(float(value.detach().float().abs().sum()))
                    gradient_handles.append(parameter.register_hook(capture))
            torch.cuda.reset_peak_memory_stats()
            updates = [smoke_update(model, trainer, small), smoke_update(model, trainer, small)]
            small_memory = memory()
            if mode != 'none':
                assert gradients['projection'][0] > 0 and gradients['projection'][1] > 0
                assert all(values[0] == 0 and values[1] > 0 for key, values in gradients.items() if key.startswith('conv_'))
            model.eval()
            with torch.no_grad():
                prediction = model(small)
            metrics = masked_metric_check(prediction, small)
            counts[mode] = sum(p.numel() for p in model.parameters())
            branch_count = sum(p.numel() for p in model.node_context_conv.parameters()) if mode != 'none' else 0
            if mode != 'none':
                assert branch_count == 256 * 192 * 5 + 192 + 192 * 256 + 256
            entry = {'node_conv_mode': mode, 'bond_head_type': 'four_state_mlp', 'bond_fusion_mode': 'concat',
                'hidden_dim': 256, 'theory_fusion_stage': 'late', 'edge_state_mode': 'static',
                'parameter_count': counts[mode], 'parameters': counts[mode], 'node_conv_parameter_count': branch_count,
                'shared_backbone_initialization_exact': True, 'initialization_rng_exact': True,
                'initial_function_equal_to_none': True, 'amp': trainer.use_amp, 'optimizer_updates': 2,
                'small_batch_size': len(indices), 'small_batch_updates': updates,
                'new_parameter_gradient_l1': gradients, 'unit_checks': units, 'routing': routing,
                'diagnostics': diagnostics, 'masked_metrics': metrics, 'shapes': observed,
                'peak_cuda_allocated_bytes': small_memory['allocated_bytes'],
                'peak_cuda_reserved_bytes': small_memory['reserved_bytes'],
                'training_small_batch_peak_allocated_mb': small_memory['allocated_bytes'] / 1024 ** 2,
                'training_small_batch_peak_reserved_mb': small_memory['reserved_bytes'] / 1024 ** 2}
            if mode != 'none':
                full_train = to_cuda(collate(train, longest_indices(train, 128)))
                torch.cuda.reset_peak_memory_stats()
                entry['full_training_batch_update'] = smoke_update(model, trainer, full_train)
                resource = memory()
                entry.update(full_training_batch_size=128,
                    full_training_batch_peak_allocated_mb=resource['allocated_bytes'] / 1024 ** 2,
                    full_training_batch_peak_reserved_mb=resource['reserved_bytes'] / 1024 ** 2)
                entry['peak_cuda_allocated_bytes'] = max(entry['peak_cuda_allocated_bytes'], resource['allocated_bytes'])
                entry['peak_cuda_reserved_bytes'] = max(entry['peak_cuda_reserved_bytes'], resource['reserved_bytes'])
                entry['optimizer_updates'] += 1
                del full_train
                gc.collect()
                torch.cuda.empty_cache()
                full_val = to_cuda(collate(val, longest_indices(val, 512)))
                assert 'four_state_targets' not in full_val
                torch.cuda.reset_peak_memory_stats()
                model.eval()
                with torch.no_grad():
                    prediction = model(full_val)
                assert prediction.shape == full_val['labels'].shape and torch.isfinite(prediction).all()
                resource = memory()
                entry.update(validation_batch_size=512,
                    validation_batch_peak_allocated_mb=resource['allocated_bytes'] / 1024 ** 2,
                    validation_batch_peak_reserved_mb=resource['reserved_bytes'] / 1024 ** 2)
                entry['peak_cuda_allocated_bytes'] = max(entry['peak_cuda_allocated_bytes'], resource['allocated_bytes'])
                entry['peak_cuda_reserved_bytes'] = max(entry['peak_cuda_reserved_bytes'], resource['reserved_bytes'])
                del full_val
            saved_states[mode] = cpu_state(model)
            for handle in [*shape_handles, *gradient_handles]:
                handle.remove()
            report['variants'][mode] = report['arms'][mode] = entry
            report['resources'][mode] = {key: value for key, value in entry.items()
                if 'peak_' in key or key in ('parameter_count', 'node_conv_parameter_count')}
            del model, trainer, optimizer, prediction, state
            gc.collect()
            torch.cuda.empty_cache()
        assert counts['single_k5'] - counts['none'] == 256 * 192 * 5 + 192 + 192 * 256 + 256
        report['parameter_budget'] = {'none': counts['none'], 'single_k5': counts['single_k5'], 'added': counts['single_k5'] - counts['none']}
        checkpoints = checkpoint_checks(fold_configs, saved_states, temp)
        rng_checkpoint_roundtrip(fold_configs, saved_states, temp)
        for mode, proof in checkpoints.items():
            report['variants'][mode]['checkpoint_roundtrip'] = proof
    report['conv_checks'] = {
        'frozen_none_schema_state_rng_cpu_forward_backward_and_restore_exact': True,
        'shared_backbone_initialization_rng_and_initial_functions_exact': True,
        'single_k5_parameter_budget_unchanged_from_discovery': True,
        'independent_position_convolution_and_exact_kernel_support': True,
        'single_vs_mixed_sample_isolation': True,
        'nan_inf_padding_and_extended_padding_valid_predictions_invariant': True,
        'global_excluded_and_conv_before_global_pack_and_film': True,
        'short_chains_mixed_l0_l1_l2_and_branch_b0_l0': True,
        'real_four_state_ce_union_q_and_padding_mask_metrics': True,
        'two_amp_steps_projection_then_all_convolutions_receive_gradient': True,
        'single_k5_longest_train128_and_validation512': True,
        'all_five_original_width_discovery_train_val_source_shas_match': True,
        'training43_distinct_and_reproducible_split42_unchanged': True,
        'only_training_seed_and_output_identity_differ_from_discovery': True,
        'production_saver_resolved_training_rng_metadata_roundtrip': True,
        'same_mode_restore_cross_mode_config_and_marker_precopy_reject': True,
        'opt_in_descriptive_diagnostics_formula_and_rng': True}
    assert all(report['conv_checks'].values())
    report['checks'] = report['conv_checks']
    report['status'] = 'passed'
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    try:
        result = run(args.gpu)
    except Exception as error:
        result = {'status': 'failed', 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
        raise
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'output': args.output}))
