"""Optional recurrent edge states on an unchanged, directed graph.

The fixed-context and dynamic-context experiments share this exact updater.
Only the node context passed by the production forward differs. Neither mode
uses bond targets, observed peaks, or the theoretical bond readout features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


EDGE_STATE_MODE_KEY = 'edge_state_mode_code'
_MODE_CODES = {'fixed_context': 1, 'dynamic_context': 2}
_CODE_TO_MODE = {code: mode for mode, code in _MODE_CODES.items()}
_UPDATER_PREFIX = 'edge_state_updater.'


def validate_edge_state_mode(value):
    if value not in ('static', 'fixed_context', 'dynamic_context'):
        raise ValueError(f'Unknown edge_state_mode: {value!r}')
    return value


def edge_state_mode_code(mode):
    """Static retains the legacy state-dict schema without an extra marker."""
    return _MODE_CODES.get(validate_edge_state_mode(mode))


def validate_edge_state_checkpoint(model, state_dict, checkpoint_config=None):
    """Reject mode or updater mismatches before any parameter is copied.

    Fixed/dynamic modes have identical tensor shapes. A mode marker is
    therefore mandatory, including for a direct ``strict=False`` load. The
    exact updater key set and shapes are checked here as strict=False would
    otherwise silently leave a partially missing updater randomly initialized.
    """
    requested = validate_edge_state_mode(model.edge_state_mode)
    if EDGE_STATE_MODE_KEY in state_dict:
        marker = state_dict[EDGE_STATE_MODE_KEY]
        if (not torch.is_tensor(marker) or marker.dtype != torch.int64
                or marker.shape != torch.Size([]) or marker.item() not in _CODE_TO_MODE):
            raise ValueError('Invalid checkpoint edge_state_mode_code')
        saved = _CODE_TO_MODE[int(marker.item())]
    else:
        saved = 'static'
    if saved != requested:
        raise ValueError(f'Checkpoint edge_state_mode mismatch: {saved!r} != {requested!r}')

    expected_state = model.state_dict()
    saved_keys = {key for key in state_dict if key.startswith(_UPDATER_PREFIX)}
    expected_keys = {key for key in expected_state if key.startswith(_UPDATER_PREFIX)}
    if saved_keys != expected_keys:
        raise ValueError('Checkpoint edge-state updater keys mismatch')
    for key in expected_keys:
        tensor = state_dict[key]
        if not torch.is_tensor(tensor) or tensor.shape != expected_state[key].shape:
            raise ValueError(f'Checkpoint edge-state updater shape mismatch: {key}')

    if checkpoint_config is not None:
        if not isinstance(checkpoint_config, dict):
            raise ValueError('Checkpoint config must be a dictionary')
        saved_model_config = checkpoint_config.get('model', checkpoint_config)
        if not isinstance(saved_model_config, dict):
            raise ValueError('Checkpoint model config must be a dictionary')
        declared = validate_edge_state_mode(saved_model_config.get('edge_state_mode', 'static'))
        if declared != saved:
            raise ValueError('Checkpoint edge_state_mode config/state mismatch')


class SharedEdgeStateUpdater(nn.Module):
    """One directed residual updater, reused before every GCN and GAT layer."""

    def __init__(self, hidden_dim):
        super().__init__()
        if not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool) or hidden_dim < 2:
            raise ValueError('Edge-state updating requires hidden_dim >= 2')
        self.hidden_dim = hidden_dim
        self.input_norm = nn.LayerNorm(3 * hidden_dim)
        self.down = nn.Linear(3 * hidden_dim, hidden_dim // 2)
        self.activation = nn.SiLU()
        self.up = nn.Linear(hidden_dim // 2, hidden_dim)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        # Construct AFTER GraphTransformer._init_weights, which would overwrite
        # these zeros. Identity at initialization matches the static reference.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, edge_state, src_h, dst_h):
        """Return updated [E,H] states, preserving row order and both directions."""
        if (edge_state.ndim != 2 or edge_state.size(-1) != self.hidden_dim
                or src_h.shape != edge_state.shape or dst_h.shape != edge_state.shape):
            raise ValueError('Edge-state updater requires aligned [num_edges, hidden_dim] inputs')
        if edge_state.size(0) == 0:
            return edge_state
        inputs = torch.cat([src_h, dst_h, edge_state], dim=-1)
        delta = self.up(self.activation(self.down(self.input_norm(inputs))))
        # No in-place mutation: the bond head intentionally retains e0, and
        # autograd must keep each recurrent state and fixed h0 context intact.
        return edge_state + delta.to(dtype=edge_state.dtype)


@torch.no_grad()
def edge_state_diagnostic(layer_kind, layer_index, initial, previous, current, layer):
    """Small-batch opt-in summaries; never called in normal training.

    Norms are means of per-edge L2 norms. Saturation is the fraction of actual
    sigmoid message-gate values <= .01 or >= .99, before GCN's safety clamp.
    Explicit CPU scalars synchronize the device only when diagnostics are on.
    """
    initial = initial.detach().float()
    previous = previous.detach().float()
    current = current.detach().float()
    count = int(current.size(0))

    def mean_l2(values):
        return float(values.norm(dim=-1).mean().item()) if count else 0.0

    result = {
        'layer': f'{layer_kind}_{layer_index}',
        'edge_count': count,
        'step_delta_l2_mean': mean_l2(current - previous),
        'cumulative_delta_l2_mean': mean_l2(current - initial),
        'edge_state_l2_mean': mean_l2(current),
        'initial_edge_l2_mean': mean_l2(initial),
        'gate_saturation_lower': 0.01,
        'gate_saturation_upper': 0.99,
        'gate_low_fraction': None,
        'gate_high_fraction': None,
        'gate_saturation_fraction': None,
    }
    projection = None
    if layer_kind == 'gcn' and getattr(layer.gcn, 'use_edge_features', False):
        projection = layer.gcn.edge_transform
    elif layer_kind == 'gat' and getattr(layer, 'use_edge_gate', False):
        projection = layer.edge_message_gate
    if projection is not None:
        if count:
            with torch.autocast(device_type=current.device.type, enabled=False):
                logits = F.linear(current, projection.weight.detach().float(),
                                  None if projection.bias is None else projection.bias.detach().float())
                gate = torch.sigmoid(logits)
            low = float((gate <= 0.01).float().mean().item())
            high = float((gate >= 0.99).float().mean().item())
        else:
            low = high = 0.0
        result.update(gate_low_fraction=low, gate_high_fraction=high,
                      gate_saturation_fraction=low + high)
    return result
