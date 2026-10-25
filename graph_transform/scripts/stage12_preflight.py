"""Pod-only Stage 12 smoke: five matched configs, real folds, one CUDA update.

Writes a new immutable JSON receipt. It does not train epochs or read test
labels for model selection; create_datasets is used only to audit split/source
identity and to obtain a small training batch for forward/backward checks.
"""

import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
import types

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'graph_transform'), str(ROOT / 'graph_transform/scripts')]
from models.graph_transformer import GraphTransformer
from models.four_state import union_logit
from models.stage12_fusion import stage12_head_parameters
from models.utils import build_model_config
from test_four_state_width import collate, longest_indices, to_cuda
from train_5fold import make_fold_config, _stable_fold_seed
from train_graph_model import create_datasets, load_config
from training.four_state_loss import four_state_cross_entropy
import train_graph_model as train_script

ARMS = ('a_legacy', 'b_wide_mlp', 'c_equal', 'd_softmax_gate',
        'e_softmax_gate_residual')
MODES = ('legacy', 'wide_mlp', 'equal', 'softmax_gate',
         'softmax_gate_residual')
FOLDS = ('1222', '2252', '3514', '6072', '9075')
LOGGER = logging.getLogger('stage12_preflight')
train_script.logger = LOGGER


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


def config_path(arm):
    return ROOT / f'graph_transform/config/pre_synthesis_stage12_fusion_{arm}.yaml'


def comparable(config, arm):
    value = copy.deepcopy(config)
    value['experiment'].pop('description', None)
    value['model'].pop('stage12_fusion_mode', None)
    text = json.dumps(value, sort_keys=True)
    return text.replace(f'stage12_fusion_{arm}', 'stage12_fusion_ARM')


def shared_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            if not name.startswith(('bond_head.', 'stage12_fusion_'))}


def baseline_class():
    root_commit = subprocess.check_output(
        ['git', 'rev-list', '--max-parents=0', 'HEAD'], cwd=ROOT,
        text=True).strip().splitlines()
    require(len(root_commit) == 1, 'Expected one imported source root commit')
    source = subprocess.check_output(['git', 'show',
        f'{root_commit[0]}:graph_transform/models/graph_transformer.py'], cwd=ROOT)
    module = types.ModuleType('models._stage12_imported_reference')
    module.__package__ = 'models'
    module.__file__ = f'git:{root_commit[0]}:graph_transform/models/graph_transformer.py'
    sys.modules[module.__name__] = module
    exec(compile(source.decode('utf-8'), module.__file__, 'exec'), module.__dict__)
    return module.GraphTransformer, {'import_commit': root_commit[0],
        'graph_transformer_sha256': hashlib.sha256(source).hexdigest()}


def legacy_compatibility(runtime, small_cpu):
    Legacy, proof = baseline_class()
    torch.manual_seed(runtime['experiment']['seed'])
    original = Legacy(build_model_config(runtime)).cpu().eval()
    old_state = {name: value.detach().clone() for name, value in original.state_dict().items()}
    torch.manual_seed(runtime['experiment']['seed'])
    current = GraphTransformer(build_model_config(runtime)).cpu().eval()
    new_state = current.state_dict()
    require(old_state.keys() == new_state.keys() and all(
        torch.equal(value, new_state[name]) for name, value in old_state.items()),
        'Stage 12 legacy state/init differs from imported original source')
    current.load_state_dict(old_state, strict=True)
    with torch.no_grad():
        require(torch.equal(original(small_cpu), current(small_cpu)),
                'Stage 12 legacy forward differs from imported original source')
    proof['fresh_legacy_state_and_forward_exact'] = True
    registry = Path('/mnt/pvc/graphtrans/four_state_parallel_global_replication_ops/checkpoint_registry.json')
    if not registry.is_file():
        proof['historical_seed42_checkpoint'] = {'status': 'unavailable', 'registry': str(registry)}
        return proof
    try:
        row = json.loads(registry.read_text(encoding='utf-8'))['arms']['single']['1222']
        checkpoint = Path(row['checkpoint'])
        require(checkpoint.is_file() and sha(checkpoint) == row['checkpoint_sha256'],
                'Historical checkpoint SHA/path changed')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        historical_config = build_model_config(saved['config'])
        old = Legacy(historical_config).cpu().eval()
        new = GraphTransformer(historical_config).cpu().eval()
        old.load_state_dict(saved['model_state_dict'], strict=True)
        new.load_state_dict(saved['model_state_dict'], strict=True)
        with torch.no_grad():
            require(torch.equal(old(small_cpu), new(small_cpu)),
                    'Historical checkpoint forward differs on same training batch')
        proof['historical_seed42_checkpoint'] = {
            'status': 'strict_loaded_forward_exact', 'checkpoint': str(checkpoint),
            'checkpoint_sha256': row['checkpoint_sha256'],
            'registry_sha256': sha(registry)}
    except Exception as error:
        # This historical artifact is never reused as S12-A's paired control.
        # A provenance/version mismatch is reported and A is still retrained.
        proof['historical_seed42_checkpoint'] = {'status': 'incompatible',
            'registry': str(registry), 'error': f'{type(error).__name__}: {error}'}
    return proof


def mask_check(head, device):
    head.eval()
    head.enable_diagnostics = True
    gate_before = head.gate.weight.detach().clone() if hasattr(head, 'gate') else None
    residual_before = (head.raw_residual.weight.detach().clone()
                       if hasattr(head, 'raw_residual') else None)
    parts = [torch.randn(3, 256, device=device) for _ in range(6)]
    mask = torch.tensor([[True, False, True], [True, True, False],
                         [False, True, True]], device=device)
    with torch.no_grad():
        if gate_before is not None:
            head.gate.weight.copy_(torch.arange(head.gate.weight.numel(), device=device)
                                   .reshape_as(head.gate.weight) * 1e-6)
        if residual_before is not None:
            head.raw_residual.weight.fill_(1e-4)
        original = head(parts, mask)
        weights = head.last_source_weights
        require(original.shape == (3, 4) and torch.isfinite(original).all(),
                'Masked branch output is invalid')
        require(torch.isfinite(weights).all() and torch.allclose(
            weights.sum(dim=-1), torch.ones(3, device=device), atol=1e-6),
            'Masked gate weights are not finite/normalized')
        require(torch.equal(weights[~mask], torch.zeros_like(weights[~mask])),
                'Unavailable source received nonzero weight')
        changed = [part.clone() for part in parts]
        changed[2][0] = float('nan')  # edge unavailable for bond 0
        changed[5][1] = float('inf')  # theory unavailable for bond 1
        for index in (0, 1, 3, 4):
            changed[index][2] = float('-inf')  # residue branch unavailable for bond 2
        altered = head(changed, mask)
        require(torch.allclose(original, altered, atol=1e-6, rtol=1e-6),
                'Masked source changed prediction')
        if gate_before is not None:
            head.gate.weight.copy_(gate_before)
        if residual_before is not None:
            head.raw_residual.weight.copy_(residual_before)
    head.enable_diagnostics = False
    return {'weights_normalized': True, 'masked_sources_excluded': True}


def run(args):
    require(sys.platform.startswith('linux') and str(ROOT).startswith('/mnt/pvc/'),
            'Stage 12 numerical preflight must run on the authorized pod')
    require(torch.cuda.is_available(), 'CUDA is unavailable')
    require((ROOT / 'dataset/5fold_folded').is_dir()
            and (ROOT / 'dataset/5fold_soft').is_dir(),
            'Independent snapshot must resolve folded and soft dataset directories')
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    out = Path(args.output).resolve()
    require(not out.exists(), f'Refusing to overwrite preflight receipt: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    namespace = Path(args.namespace).resolve()
    require(str(out).startswith(str(namespace) + '/'), 'Receipt must stay in Stage 12 namespace')
    configs = {}
    for arm, mode in zip(ARMS, MODES):
        config = load_config(config_path(arm), argparse.Namespace(
            epochs=None, batch_size=None, learning_rate=None, device=None))
        model = build_model_config(config)
        require(model.stage12_fusion_mode == mode and model.hidden_dim == 256
                and model.bond_head_type == 'four_state_mlp'
                and model.bond_fusion_mode == 'concat' and model.node_conv_mode == 'single_k5'
                and model.virtual_node_mode == 'single' and model.use_global_node
                and model.use_condition_film and model.theory_fusion_stage == 'late'
                and model.edge_state_mode == 'static' and model.use_theory_features
                and model.num_gcn_layers == 3 and model.num_gat_layers == 2
                and model.num_attention_heads == 4
                and model.state_feature_mask == [True, True, False]
                and model.env_feature_mask == [True, False]
                and not model.use_charge_partition_features
                and model.sequence_adapter_type == 'none',
                f'{arm}: architecture changed')
        require(config['experiment']['seed'] == 42
                and config['training']['validation_split_base_seed'] == 42
                and config['loss']['four_state_conditional_weight'] == 1
                and config['loss']['four_state_union_brier_weight'] == 0
                and config['evaluation']['threshold'] == 0.5
                and config['evaluation']['save_outputs'] is True,
                f'{arm}: training/evaluation policy changed')
        configs[arm] = config
    base = comparable(configs[ARMS[0]], ARMS[0])
    require(all(comparable(configs[arm], arm) == base for arm in ARMS),
            'Arm configs differ beyond mode and artifact namespace')

    receipt = {'status': 'running', 'source_root': str(ROOT), 'gpu': args.gpu,
               'config_sha256': {arm: sha(config_path(arm)) for arm in ARMS},
               'data_roots': {'folded': str((ROOT / 'dataset/5fold_folded').resolve()),
                              'soft': str((ROOT / 'dataset/5fold_soft').resolve())},
               'model_source_sha256': {name: sha(ROOT / name) for name in (
                   'graph_transform/models/graph_transformer.py',
                   'graph_transform/models/stage12_fusion.py',
                   'graph_transform/models/utils.py',
                   'graph_transform/models/four_state.py')},
               'folds': {}, 'arms': {}, 'parameter_protocol': stage12_head_parameters()}
    with tempfile.TemporaryDirectory(prefix='dbond_stage12_smoke_') as temp:
        first_train = None
        first_val = None
        first_runtime = None
        for fold in FOLDS:
            fold_configs = {}
            seed = _stable_fold_seed(42, fold)
            for arm in ARMS:
                cfg, _, _ = make_fold_config(configs[arm], fold,
                    str(ROOT / 'dataset/5fold_folded'), str(Path(temp) / arm), seed)
                cfg['data']['cache_graphs'] = False
                cfg['data']['cache_dir'] = str(Path(temp) / 'cache' / fold)
                fold_configs[arm] = cfg
            base_fold = comparable(fold_configs[ARMS[0]], ARMS[0])
            require(all(comparable(fold_configs[arm], arm) == base_fold for arm in ARMS),
                    f'{fold}: resolved configs are not paired')
            train, val, test = create_datasets(fold_configs[ARMS[0]])
            sets = [set(dataset.data['seq']) for dataset in (train, val, test)]
            require(not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]),
                    f'{fold}: sequence split overlap')
            require(val.row_validation_provenance['outer_test_used_for_internal_validation'] is False,
                    f'{fold}: outer test used for checkpoint selection')
            require('four_state_targets' not in val[0] and 'four_state_targets' not in test[0],
                    f'{fold}: validation/test target leakage')
            receipt['folds'][fold] = {
                'training_seed': seed, 'internal_split_seed': cfg['training']['validation_split_seed'],
                'train_sequences': len(sets[0]), 'val_sequences': len(sets[1]),
                'test_sequences': len(sets[2]),
                'train_split_sha256': sha(train.csv_path),
                'val_split_sha256': sha(val.csv_path),
                'outer_train_sha256': sha(cfg['data']['train_csv_path']),
                'outer_test_sha256': sha(cfg['data']['test_csv_path']),
                'validation_source_sha256': sha(cfg['data']['validation_source_csv_path']),
                'four_state_manifest_sha256': sha(cfg['data']['four_state_targets_manifest'])}
            if first_train is None:
                first_train = train
                first_val = val
                first_runtime = fold_configs

        count = min(8, len(first_train))
        require(count > 0, 'No training rows for smoke')
        small_cpu = collate(first_train, list(range(count)))
        receipt['legacy_compatibility'] = legacy_compatibility(
            first_runtime[ARMS[0]], small_cpu)
        batch = to_cuda(small_cpu)
        mask = batch['label_mask'].bool()
        require(bool(mask.any()), 'Smoke batch has no valid bonds')
        poisoned = dict(batch)
        poisoned['bond_theory'] = batch['bond_theory'].clone()
        poisoned['bond_theory'][~mask] = float('nan')
        require(len(first_train) >= 128 and len(first_val) >= 512,
                'Longest train128/val512 resource checks require full batches')
        train128_cpu = collate(first_train, longest_indices(first_train, 128))
        val512_cpu = collate(first_val, longest_indices(first_val, 512))
        initial_shared = None
        initial_fused = {}
        expected_heads = stage12_head_parameters()
        for arm, mode in zip(ARMS, MODES):
            torch.manual_seed(first_runtime[arm]['experiment']['seed'])
            model = GraphTransformer(build_model_config(first_runtime[arm])).to(device)
            shared = shared_state(model)
            if initial_shared is None:
                initial_shared = shared
            else:
                require(shared.keys() == initial_shared.keys() and all(
                    torch.equal(value, initial_shared[name]) for name, value in shared.items()),
                    f'{arm}: shared backbone initialization changed')
            head_count = sum(p.numel() for p in model.bond_head.parameters())
            require(head_count == expected_heads[mode], f'{arm}: parameter count mismatch')
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            restored = GraphTransformer(build_model_config(first_runtime[arm])).cpu()
            restored.load_state_dict(state, strict=True)
            model.eval()
            with torch.no_grad():
                initial_fused[mode] = model(batch)[mask].detach().clone()
                poisoned_output = model(poisoned)[mask]
                require(torch.equal(initial_fused[mode], poisoned_output),
                        f'{arm}: padded theory changed valid-bond prediction')
            if mode in ('equal', 'softmax_gate', 'softmax_gate_residual'):
                receipt['arms'][arm] = mask_check(model.bond_head, device)
            else:
                receipt['arms'][arm] = {}
            model.train()
            model.zero_grad(set_to_none=True)
            model.enable_stage12_diagnostics = mode in ('softmax_gate', 'softmax_gate_residual')
            logits = model(batch)
            states = model.last_four_state_logits
            require(logits.shape == mask.shape and states.shape == (int(mask.sum()), 4)
                    and torch.isfinite(logits).all() and torch.isfinite(states).all(),
                    f'{arm}: invalid four-state output')
            require(torch.allclose(logits[mask].float(), union_logit(states), atol=1e-6),
                    f'{arm}: four-state union logit changed')
            require(torch.allclose(torch.sigmoid(logits[mask].float()),
                                   1 - torch.softmax(states.float(), dim=-1)[:, 0], atol=1e-6),
                    f'{arm}: four-state probability mapping changed')
            loss = four_state_cross_entropy(states, batch['four_state_targets'], mask,
                                            conditional_weight=1., union_brier_weight=0.)
            require(torch.isfinite(loss), f'{arm}: invalid loss')
            loss.backward()
            require(all(p.grad is None or torch.isfinite(p.grad).all()
                        for p in model.parameters()), f'{arm}: nonfinite gradient')
            if mode in ('softmax_gate', 'softmax_gate_residual'):
                gradient = model.bond_head.gate.weight.grad
                require(gradient is not None and gradient.abs().sum() > 0,
                        f'{arm}: gate did not receive gradient')
                weights = model.bond_head.last_source_weights
                require(weights is not None and torch.isfinite(weights).all()
                        and torch.allclose(weights.sum(-1), torch.ones_like(weights[:, 0]),
                                           atol=1e-6), f'{arm}: gate not normalized')
                receipt['arms'][arm]['gate_gradient_l1'] = float(gradient.abs().sum())
            if mode == 'softmax_gate_residual':
                gradient = model.bond_head.raw_residual.weight.grad
                require(gradient is not None and gradient.abs().sum() > 0,
                        'Raw concat residual did not receive gradient')
                receipt['arms'][arm]['residual_gradient_l1'] = float(gradient.abs().sum())
            receipt['arms'][arm].update(head_parameters=head_count,
                total_parameters=sum(p.numel() for p in model.parameters()),
                output_shape=list(logits.shape), valid_bonds=int(mask.sum()),
                forward_backward_passed=True, strict_checkpoint_restore=True)
            del logits, states, loss, restored
            model.last_four_state_logits = None
            model.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            train128 = to_cuda(train128_cpu)
            long_output = model(train128)
            long_mask = train128['label_mask'].bool()
            long_loss = four_state_cross_entropy(model.last_four_state_logits,
                train128['four_state_targets'], long_mask,
                conditional_weight=1., union_brier_weight=0.)
            require(torch.isfinite(long_output).all() and torch.isfinite(long_loss),
                    f'{arm}: longest train128 output/loss invalid')
            long_loss.backward()
            torch.cuda.synchronize(device)
            train_peak = torch.cuda.max_memory_allocated(device)
            receipt['arms'][arm]['longest_train128_peak_gpu_bytes'] = train_peak
            del train128, long_output, long_mask, long_loss
            model.last_four_state_logits = None
            model.zero_grad(set_to_none=True)
            model.eval()
            torch.cuda.reset_peak_memory_stats(device)
            val512 = to_cuda(val512_cpu)
            with torch.inference_mode():
                val_output = model(val512)
            require(torch.isfinite(val_output).all() and val_output.shape == val512['label_mask'].shape,
                    f'{arm}: longest val512 output invalid')
            torch.cuda.synchronize(device)
            receipt['arms'][arm]['longest_val512_peak_gpu_bytes'] = (
                torch.cuda.max_memory_allocated(device))
            del val512, val_output, model
            torch.cuda.empty_cache()
        require(torch.allclose(initial_fused['equal'], initial_fused['softmax_gate'], atol=1e-6)
                and torch.allclose(initial_fused['softmax_gate'],
                                   initial_fused['softmax_gate_residual'], atol=1e-6),
                'Equal/gated/residual initial functions differ')
    receipt['status'] = 'passed'
    out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--namespace', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as error:
        print(f'Stage 12 preflight failed: {error}', file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
