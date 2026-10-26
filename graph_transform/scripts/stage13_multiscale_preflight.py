"""Pod-only structural preflight for the independent Stage 13 Conv1D matrix.

This bounded check runs no training epoch and reads no outer-test labels. A
separate real-batch GPU smoke and source audit are required before submission.
"""

import argparse
import copy
import hashlib
import json
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'graph_transform'), str(ROOT / 'graph_transform/scripts')]

from models.graph_transformer import GraphTransformer
from models.node_context_conv import NodeContextConv, node_conv_spec
from models.utils import build_model_config
from test_four_state_dynamic_edges import toy_batch
from test_four_state_node_conv import (activate_nonzero_projection,
                                       position_reference, receptive_field_check,
                                       route_check)
from train_graph_model import load_config


CONFIGS = {
    'S13-A': 'graph_transform/config/stage13_multiscale/s13_a_single_k5.yaml',
    'S13-B': 'graph_transform/config/stage13_multiscale/s13_b_multiscale_357_c32.yaml',
    'S13-C': 'graph_transform/config/stage13_multiscale/s13_c_multiscale_357_c64.yaml',
}
MODES = {'S13-A': 'single_k5', 'S13-B': 'multiscale_357_c32',
         'S13-C': 'multiscale_357'}
FOLDS = ('1222', '2252', '3514', '6072', '9075')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_protocol(config):
    """Remove only run identity and the one randomized architecture variable."""
    value = copy.deepcopy(config)
    value['experiment'].pop('name', None)
    value['experiment'].pop('description', None)
    value['experiment'].pop('base_name', None)
    value['model'].pop('node_conv_mode', None)
    for key in ('cache_dir', 'cache_dir_template'):
        value['data'].pop(key, None)
    value['debug'].pop('diagnostic_dir', None)
    for key in ('output_metric_dir', 'output_pred_dir'):
        value['evaluation'].pop(key, None)
    # The Stage 13 snapshot exports the very logits used to compute metrics.
    # This changes CSV provenance only, not model, loss, or metric definitions.
    value['evaluation'].pop('export_from_evaluate_cache', None)
    for key in ('log_dir', 'tensorboard_log_dir'):
        value['logging'].pop(key, None)
    value['training'].pop('checkpoint_dir', None)
    return value


def check_configs():
    overrides = argparse.Namespace(epochs=None, batch_size=None,
                                   learning_rate=None, device=None)
    configs = {arm: load_config(ROOT / path, overrides)
               for arm, path in CONFIGS.items()}
    historical = load_config(
        ROOT / 'graph_transform/config/pre_synthesis_four_state_node_conv_single_k5.yaml',
        overrides)
    canonical = config_protocol(historical)
    for arm, config in configs.items():
        require(config_protocol(config) == canonical,
                f'{arm}: a setting besides run identity or node_conv_mode changed')
        model, loss = config['model'], config['loss']
        require(model['node_conv_mode'] == MODES[arm], f'{arm}: wrong node-conv mode')
        require(model['hidden_dim'] == 256 and model['num_gcn_layers'] == 3
                and model['num_gat_layers'] == 2 and model['num_attention_heads'] == 4,
                f'{arm}: original graph backbone changed')
        require(model.get('virtual_node_mode', 'single') == 'single'
                and model['use_global_node'] and model['use_condition_film'],
                f'{arm}: original single-global/FiLM path changed')
        require(model['bond_head_type'] == 'four_state_mlp'
                and model['bond_fusion_mode'] == 'concat'
                and model['edge_state_mode'] == 'static'
                and model['theory_fusion_stage'] == 'late'
                and not model['use_charge_partition_features'],
                f'{arm}: original four-state concat/static/late/nocharge path changed')
        require(model['state_feature_mask'] == [True, True, False]
                and model['env_feature_mask'] == [True, False],
                f'{arm}: post-acquisition feature was not masked')
        require(loss['four_state_conditional_weight'] == 1.
                and loss['four_state_union_brier_weight'] == 0.,
                f'{arm}: original four-state loss changed')
        require(config['experiment']['seed'] == 42
                and config['training']['validation_split_base_seed'] == 42
                and config['training']['checkpoint_selection'] == 'max_f1'
                and config['evaluation']['threshold'] == .5
                and config['evaluation'].get('export_from_evaluate_cache') is True
                and config['data']['weighting_scheme'] == 'group_uniform'
                and config['ablation']['pre_synthesis'],
                f'{arm}: seed, split, training, or evaluation protocol changed')
    return configs


def check_conv(arm, mode):
    torch.manual_seed(13)
    module = NodeContextConv(256, mode).cpu().eval()
    kernels, channels = node_conv_spec(mode)
    require(kernels == (5,) if arm == 'S13-A' else kernels == (3, 5, 7),
            f'{arm}: wrong kernels')
    require(channels == ((192,) if arm == 'S13-A' else
                         (32, 32, 32) if arm == 'S13-B' else (64, 64, 64)),
            f'{arm}: wrong branch width')
    require(all(branch.padding == (kernel // 2,) and branch.stride == (1,)
                and branch.dilation == (1,) and branch.groups == 1
                for branch, kernel in zip(module.branches, kernels)),
            f'{arm}: same-padding convolution changed')
    lengths = torch.tensor([11, 5, 2, 1, 0])
    features = torch.linspace(-.8, .7, 5 * 11 * 256).reshape(5, 11, 256)
    valid = torch.arange(11)[None, :] < lengths[:, None]
    clean = features.masked_fill(~valid[..., None], 0.)
    require(torch.equal(module(features, lengths), clean),
            f'{arm}: zero-projection residual is not initially the identity')
    activate_nonzero_projection(module)
    with torch.no_grad():
        output = module(features, lengths)
        reference, _, _ = position_reference(module, features, lengths)
        require(torch.allclose(output, reference, atol=5e-6, rtol=1e-5),
                f'{arm}: convolution differs from independent position formula')
        for poison in (float('nan'), float('inf')):
            altered = features.masked_fill(~valid[..., None], poison)
            require(torch.equal(output, module(altered, lengths)),
                    f'{arm}: invalid residue contaminates a valid window')
        extended = F.pad(features, (0, 0, 0, 7), value=float('nan'))
        longer = module(extended, lengths)
        require(torch.allclose(output, longer[:, :11], atol=5e-6, rtol=1e-5)
                and torch.count_nonzero(longer[:, 11:]) == 0,
                f'{arm}: extra padding contaminates a valid tail window')
        for row, length in enumerate(lengths.tolist()):
            single = module(features[row:row + 1, :length], lengths[row:row + 1])
            require(torch.allclose(output[row:row + 1, :length], single,
                                   atol=5e-6, rtol=1e-5),
                    f'{arm}: mixed batch changes an individual sequence')
        require(module(torch.empty(0, 7, 256), torch.empty(0, dtype=torch.long)).shape
                == (0, 7, 256), f'{arm}: empty batch failed')
        require(module(torch.empty(3, 0, 256), torch.zeros(3, dtype=torch.long)).shape
                == (3, 0, 256), f'{arm}: empty sequence length failed')
    receptive_field_check(module)

    # Verify a real optimizer update activates every branch after the initially
    # zero projection. These disposable module weights never enter training.
    torch.manual_seed(13)
    updating = NodeContextConv(256, mode).cpu().train()
    optimizer = torch.optim.SGD(updating.parameters(), lr=.1)
    gradient_rows = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction = updating(features, lengths)
        loss = prediction[valid].square().mean()
        require(torch.isfinite(loss), f'{arm}: nonfinite loss')
        loss.backward()
        projection_grad = updating.projection.weight.grad
        branch_grads = [branch.weight.grad for branch in updating.branches]
        require(torch.isfinite(projection_grad).all()
                and all(torch.isfinite(gradient).all() for gradient in branch_grads),
                f'{arm}: nonfinite gradient')
        gradient_rows.append((float(projection_grad.abs().sum()),
                              [float(gradient.abs().sum()) for gradient in branch_grads]))
        optimizer.step()
    require(gradient_rows[0][0] > 0 and all(value == 0 for value in gradient_rows[0][1])
            and gradient_rows[1][0] > 0 and all(value > 0 for value in gradient_rows[1][1]),
            f'{arm}: expected two-step projection/branch gradient path failed')
    return {'kernels': kernels, 'branch_channels': channels,
            'parameters': sum(parameter.numel() for parameter in module.parameters()),
            'same_padding_and_poison_invariance': True,
            'independent_position_reference': True,
            'two_step_gradients': gradient_rows}


def check_full_models(configs):
    batch = toy_batch()
    models, states, outputs = {}, {}, {}
    for arm, config in configs.items():
        torch.manual_seed(42)
        model = GraphTransformer(build_model_config(config)).cpu().eval()
        with torch.no_grad():
            output = model(batch)
        require(output.shape == batch['label_mask'].shape
                and torch.isfinite(output).all()
                and model.bond_head_type == 'four_state_mlp'
                and model.last_four_state_logits is None,
                f'{arm}: full four-state forward failed')
        models[arm] = model
        states[arm] = {key: value.detach().cpu().clone()
                       for key, value in model.state_dict().items()}
        outputs[arm] = output.detach().clone()
    shared = lambda state: {key: value for key, value in state.items()
                            if not key.startswith(('node_context_conv.', 'node_conv_'))}
    for arm in ('S13-B', 'S13-C'):
        require(shared(states['S13-A']).keys() == shared(states[arm]).keys()
                and all(torch.equal(value, shared(states[arm])[key])
                        for key, value in shared(states['S13-A']).items()),
                f'{arm}: shared backbone initialization changed')
        require(torch.equal(outputs['S13-A'], outputs[arm]),
                f'{arm}: initial four-state prediction differs despite zero residual')
        route_check(models[arm], batch)

    counts = {arm: sum(parameter.numel() for parameter in model.parameters())
              for arm, model in models.items()}
    require(counts['S13-A'] == counts['S13-C'] > counts['S13-B'],
            'Original k5 and three-by-64 must match parameter count')
    for target_arm, target in models.items():
        target.load_state_dict(states[target_arm], strict=True)
        for donor_arm, donor in states.items():
            if donor_arm == target_arm:
                continue
            before = {key: value.detach().clone() for key, value in target.state_dict().items()}
            try:
                target.load_state_dict(donor, strict=False)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f'{donor_arm}->{target_arm}: cross-mode checkpoint accepted')
            require(all(torch.equal(value, target.state_dict()[key])
                        for key, value in before.items()),
                    f'{donor_arm}->{target_arm}: rejected checkpoint modified parameters')
    return {'parameter_counts': counts, 'original_concat_four_state_forward': True,
            'shared_backbone_and_initial_function_exact': True,
            'cross_mode_checkpoint_rejection': True,
            'global_and_film_route_checked': True}


def run():
    require(sys.platform.startswith('linux') and str(ROOT).startswith('/mnt/pvc/'),
            'Stage 13 numerical preflight must run on the authorized pod')
    configs = check_configs()
    conv = {arm: check_conv(arm, mode) for arm, mode in MODES.items()}
    require(conv['S13-A']['parameters'] == conv['S13-C']['parameters']
            and conv['S13-B']['parameters'] < conv['S13-C']['parameters'],
            'Node-conv parameter budget mismatch')
    full = check_full_models(configs)
    return {'status': 'passed', 'folds': FOLDS, 'arms': conv, 'full_model': full,
            'config_sha256': {arm: file_sha(ROOT / path) for arm, path in CONFIGS.items()},
            'S13-D': {'alias_of': 'S13-A', 'reason':
                      'Original k5 has 192 channels and exactly matches three-by-64; '
                      'a distinct wider k5 cannot retain that parameter budget.'},
            'real_batch_gpu_smoke': 'required_separately_before_training'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        report = run()
    except Exception as error:
        report = {'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                  'traceback': traceback.format_exc()}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        raise
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n',
                                 encoding='utf-8')
    print(json.dumps({'status': report['status'], 'output': args.output}))
