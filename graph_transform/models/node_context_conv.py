"""Optional masked sequence context on full-width encoded residue nodes.

Convolutions run independently along each peptide sequence, before global
nodes, graph packing and condition FiLM. They never consume bond targets.
"""

import torch
from torch import nn


_MODE_CODES = {'single_k5': 1, 'multiscale_357': 2}
_CODE_TO_MODE = {code: mode for mode, code in _MODE_CODES.items()}
_MARKERS = ('node_conv_mode_code', 'node_conv_schema_code',
            'node_conv_kernel_sizes', 'node_conv_branch_channels')
_BRANCH_PREFIX = 'node_context_conv.'


def validate_node_conv_mode(value):
    if value not in ('none', 'single_k5', 'multiscale_357'):
        raise ValueError(f'Unknown node_conv_mode: {value!r}')
    return value


def node_conv_spec(mode):
    mode = validate_node_conv_mode(mode)
    if mode == 'single_k5':
        return (5,), (192,)
    if mode == 'multiscale_357':
        return (3, 5, 7), (64, 64, 64)
    return (), ()


def node_conv_mode_code(mode):
    return _MODE_CODES.get(validate_node_conv_mode(mode))


def validate_node_conv_checkpoint(model, state_dict, checkpoint_config=None):
    """Reject missing branches or changed semantics before any tensor copy."""
    requested = validate_node_conv_mode(model.node_conv_mode)
    mode_key = _MARKERS[0]
    if mode_key in state_dict:
        marker = state_dict[mode_key]
        if (not torch.is_tensor(marker) or marker.dtype != torch.int64
                or marker.shape != torch.Size([]) or marker.item() not in _CODE_TO_MODE):
            raise ValueError('Invalid checkpoint node_conv_mode_code')
        saved = _CODE_TO_MODE[int(marker.item())]
    else:
        saved = 'none'
    if saved != requested:
        raise ValueError(f'Checkpoint node_conv_mode mismatch: {saved!r} != {requested!r}')

    expected_state = model.state_dict()
    actual_keys = {key for key in state_dict
                   if key.startswith(_BRANCH_PREFIX) or key.startswith('node_conv_')}
    expected_keys = {key for key in expected_state
                     if key.startswith(_BRANCH_PREFIX) or key.startswith('node_conv_')}
    if actual_keys != expected_keys:
        raise ValueError('Checkpoint node-conv marker/branch keys mismatch')
    for key in expected_keys:
        value = state_dict[key]
        if not torch.is_tensor(value) or value.shape != expected_state[key].shape:
            raise ValueError(f'Checkpoint node-conv tensor shape mismatch: {key}')
        if key in _MARKERS and (value.dtype != torch.int64 or not torch.equal(
                value.detach().cpu(), expected_state[key].detach().cpu())):
            raise ValueError(f'Checkpoint node-conv semantics mismatch: {key}')

    if checkpoint_config is not None:
        if not isinstance(checkpoint_config, dict):
            raise ValueError('Checkpoint config must be a dictionary')
        declared = checkpoint_config.get('model', checkpoint_config)
        if not isinstance(declared, dict):
            raise ValueError('Checkpoint model config must be a dictionary')
        if validate_node_conv_mode(declared.get('node_conv_mode', 'none')) != saved:
            raise ValueError('Checkpoint node_conv_mode config/state mismatch')
        if saved != 'none' and (declared.get('hidden_dim', 256) != model.hidden_dim
                or declared.get('sequence_adapter_type', 'none') != 'none'):
            raise ValueError('Checkpoint node-conv input/legacy-adapter config mismatch')


class NodeContextConv(nn.Module):
    """A zero-output residual branch on padded [batch, residues, hidden] nodes."""

    def __init__(self, hidden_dim, mode):
        super().__init__()
        if not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool) or hidden_dim < 1:
            raise ValueError('Node-context convolution requires a positive hidden_dim')
        self.mode = validate_node_conv_mode(mode)
        if self.mode == 'none':
            raise ValueError('The none route must not construct a node-conv module')
        self.hidden_dim = hidden_dim
        self.kernel_sizes, self.branch_channels = node_conv_spec(self.mode)
        self.branches = nn.ModuleList([
            nn.Conv1d(hidden_dim, channels, kernel, padding=kernel // 2, bias=True)
            for kernel, channels in zip(self.kernel_sizes, self.branch_channels)])
        self.activation = nn.SiLU()
        self.projection = nn.Linear(sum(self.branch_channels), hidden_dim)
        # Construct after the original model initialization and four-state head
        # replacement: a later global Xavier pass would destroy this identity.
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.enable_node_conv_diagnostics = False
        self.last_node_conv_diagnostics = {}

    def forward(self, features, seq_lens):
        if (features.ndim != 3 or features.size(-1) != self.hidden_dim
                or seq_lens.ndim != 1 or seq_lens.numel() != features.size(0)):
            raise ValueError('Node-context convolution requires [B,L,H] features and [B] lengths')
        self.last_node_conv_diagnostics = {}
        lengths = seq_lens.to(device=features.device, dtype=torch.long)
        valid = torch.arange(features.size(1), device=features.device)[None, :] < lengths[:, None]
        invalid = ~valid.unsqueeze(-1)
        # masked_fill, rather than multiplication, keeps NaN padding out of all
        # convolutions and the residual; invalid positions also get zero grad.
        clean = features.masked_fill(invalid, 0.)
        if features.size(0) == 0 or features.size(1) == 0:
            if self.enable_node_conv_diagnostics:
                self.last_node_conv_diagnostics = node_conv_diagnostics(
                    self.mode, clean, clean, valid, [], clean)
            return clean
        channels_first = clean.transpose(1, 2)
        activations = [self.activation(branch(channels_first)).transpose(1, 2).masked_fill(invalid, 0.)
                       for branch in self.branches]
        projected = self.projection(torch.cat(activations, dim=-1)).masked_fill(invalid, 0.)
        result = (clean + projected.to(dtype=clean.dtype)).masked_fill(invalid, 0.)
        if self.enable_node_conv_diagnostics:
            self.last_node_conv_diagnostics = node_conv_diagnostics(
                self.mode, clean, result, valid, activations, projected)
        return result


@torch.no_grad()
def node_conv_diagnostics(mode, before, after, valid, branch_activations=None, projected=None):
    """Opt-in detached means over true residues; no physical-importance claim."""
    kernels, channels = node_conv_spec(mode)
    count = int(valid.sum().item())

    def values(tensor):
        return tensor.detach().float()[valid]

    def mean_l2(tensor):
        return float(values(tensor).norm(dim=-1).mean().item()) if count else 0.

    if count:
        initial_norm = values(before).norm(dim=-1)
        delta_norm = (values(after) - values(before)).norm(dim=-1)
        relative = float((delta_norm / initial_norm.clamp_min(torch.finfo(initial_norm.dtype).eps)).mean().item())
    else:
        relative = 0.
    activations = branch_activations or []
    return {
        'mode': mode,
        'num_valid_residues': count,
        'kernel_sizes': list(kernels),
        'branch_channels': list(channels),
        'branch_activation_l2_mean': [mean_l2(value) for value in activations] if activations else [0.] * len(kernels),
        'projection_residual_l2_mean': mean_l2(projected) if projected is not None else 0.,
        'input_l2_mean': mean_l2(before),
        'output_l2_mean': mean_l2(after),
        'residual_relative_l2_mean': relative,
    }
