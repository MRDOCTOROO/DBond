"""Stage 12 three-branch bond fusion, separate from older six-slot ablations.

The source order is fixed: left, right, edge, difference, product, theory.
All inputs contain valid bonds only; padding is removed by GraphTransformer.
The optional three-column mask also supports explicit missing-source checks.
"""

import torch
from torch import nn
from torch.nn import functional as F
from .four_state import replace_binary_output


STAGE12_MODES = ('legacy', 'wide_mlp', 'equal', 'softmax_gate',
                 'softmax_gate_residual')
_MODE_CODES = {mode: index for index, mode in enumerate(STAGE12_MODES) if index}
_CODE_MODES = {code: mode for mode, code in _MODE_CODES.items()}
_MARKERS = ('stage12_fusion_mode_code', 'stage12_fusion_schema_code',
            'stage12_fusion_source_roles')
WIDE_MLP_WIDTH_256 = 258


def validate_stage12_mode(mode):
    if mode not in STAGE12_MODES:
        raise ValueError(f'Unknown stage12_fusion_mode: {mode!r}')
    return mode


def validate_stage12_checkpoint(model, state_dict):
    """Reject cross-arm or partial checkpoints before any tensor is copied."""
    requested = validate_stage12_mode(model.stage12_fusion_mode)
    marker = state_dict.get(_MARKERS[0])
    if marker is None:
        saved = 'legacy'
    elif (not torch.is_tensor(marker) or marker.dtype != torch.int64
          or marker.shape != torch.Size([]) or int(marker.item()) not in _CODE_MODES):
        raise ValueError('Invalid Stage 12 checkpoint mode marker')
    else:
        saved = _CODE_MODES[int(marker.item())]
    if saved != requested:
        raise ValueError(f'Stage 12 checkpoint mode mismatch: {saved} != {requested}')
    expected = model.state_dict()
    actual_keys = {key for key in state_dict if key.startswith('stage12_fusion_')}
    expected_keys = {key for key in expected if key.startswith('stage12_fusion_')}
    if actual_keys != expected_keys:
        raise ValueError('Stage 12 checkpoint marker keys mismatch')
    for key in expected_keys:
        value = state_dict[key]
        if (not torch.is_tensor(value) or value.dtype != torch.int64
                or value.shape != expected[key].shape
                or not torch.equal(value.detach().cpu(), expected[key].detach().cpu())):
            raise ValueError(f'Stage 12 checkpoint semantic marker mismatch: {key}')
    if requested != 'legacy':
        expected_head = {key for key in expected if key.startswith('bond_head.')}
        saved_head = {key for key in state_dict if key.startswith('bond_head.')}
        if expected_head != saved_head:
            raise ValueError('Stage 12 checkpoint head keys mismatch')
        for key in expected_head:
            value = state_dict[key]
            if not torch.is_tensor(value) or value.shape != expected[key].shape:
                raise ValueError(f'Stage 12 checkpoint head shape mismatch: {key}')


def make_wide_mlp(hidden_dim, dropout):
    if hidden_dim != 256:
        raise ValueError('Stage 12 matched-width control is specified for H=256')
    first = nn.Linear(6 * hidden_dim, WIDE_MLP_WIDTH_256)
    binary = nn.Linear(WIDE_MLP_WIDTH_256, 1)
    for layer in (first, binary):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
    return nn.Sequential(first, nn.ReLU(), nn.Dropout(dropout),
                         replace_binary_output(binary))


class ThreeBranchFusionHead(nn.Module):
    """Project residue interactions, edge, and theory to H, then fuse.

    The residual arm reads every original concat coordinate directly through
    a zero-initialized 6H->4 linear path. Its extra 6H*4 parameters are small
    relative to the matched control and it starts at the gated arm's output.
    """

    def __init__(self, hidden_dim, dropout, mode):
        super().__init__()
        if hidden_dim != 256 or mode not in ('equal', 'softmax_gate',
                                              'softmax_gate_residual'):
            raise ValueError('Stage 12 requires H=256 and a three-branch mode')
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.projections = nn.ModuleList((
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        ))
        self.dropout = nn.Dropout(dropout)
        binary = nn.Linear(hidden_dim, 1)
        for layer in (*self.projections, binary):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.output = replace_binary_output(binary)
        if mode != 'equal':
            self.gate = nn.Linear(3 * hidden_dim, 3)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        if mode == 'softmax_gate_residual':
            self.raw_residual = nn.Linear(6 * hidden_dim, 4, bias=False)
            nn.init.zeros_(self.raw_residual.weight)
        self.enable_diagnostics = False
        self.last_source_weights = None

    def forward(self, parts, source_mask=None):
        if len(parts) != 6 or any(part.ndim != 2 or part.shape[-1] != self.hidden_dim
                                  for part in parts):
            raise ValueError('Stage 12 expects six [valid_bonds, H] sources')
        n = parts[0].shape[0]
        if any(part.shape[0] != n for part in parts):
            raise ValueError('Stage 12 source bond counts differ')
        explicit_mask = source_mask is not None
        if source_mask is None:
            source_mask = torch.ones((n, 3), dtype=torch.bool, device=parts[0].device)
        if (source_mask.shape != (n, 3) or source_mask.dtype != torch.bool
                or source_mask.device != parts[0].device):
            raise ValueError('Stage 12 source mask must be [valid_bonds, 3] bool')
        if explicit_mask and n and not bool(source_mask.any(dim=1).all()):
            raise ValueError('Stage 12 requires at least one source per valid bond')

        source_roles = (0, 0, 1, 0, 0, 2)
        masked_parts = [part.masked_fill(~source_mask[:, role:role + 1], 0)
                        for part, role in zip(parts, source_roles)]
        raw = torch.cat(masked_parts, dim=-1)
        groups = (torch.cat((masked_parts[0], masked_parts[1], masked_parts[3],
                             masked_parts[4]), dim=-1),
                  masked_parts[2], masked_parts[5])
        projected = torch.stack([F.relu(layer(group)) for layer, group in
                                 zip(self.projections, groups)], dim=1)
        # A missing branch must not influence active gate logits either.
        projected = projected.masked_fill(~source_mask.unsqueeze(-1), 0)
        if self.mode == 'equal':
            weights = source_mask.to(projected.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1)
        else:
            gate_logits = self.gate(projected.flatten(start_dim=1))
            # Every valid bond has an active source, so -inf is safe for n>0.
            weights = torch.softmax(gate_logits.masked_fill(~source_mask, float('-inf')),
                                    dim=-1) if n else gate_logits
            if not n:
                weights = gate_logits.new_empty((0, 3))
        fused = (projected * weights.unsqueeze(-1)).sum(dim=1)
        logits = self.output(self.dropout(fused))
        if self.mode == 'softmax_gate_residual':
            # Keep unavailable groups out of the direct raw-concat skip.
            raw_mask = torch.stack((source_mask[:, 0], source_mask[:, 0],
                                    source_mask[:, 1], source_mask[:, 0],
                                    source_mask[:, 0], source_mask[:, 2]), dim=-1)
            masked_raw = raw.reshape(n, 6, self.hidden_dim).masked_fill(
                ~raw_mask.unsqueeze(-1), 0)
            logits = logits + self.raw_residual(masked_raw.flatten(start_dim=1))
        self.last_source_weights = weights.detach() if self.enable_diagnostics else None
        return logits


def stage12_head_parameters(hidden_dim=256):
    """Exact head-only counts at H=256, independent of shared backbone size."""
    if hidden_dim != 256:
        raise ValueError('Stage 12 parameter protocol is fixed to H=256')
    baseline = (6 * hidden_dim + 1) * hidden_dim + (hidden_dim + 1) * 4
    wide = (6 * hidden_dim + 1) * WIDE_MLP_WIDTH_256 + (WIDE_MLP_WIDTH_256 + 1) * 4
    projection = ((4 * hidden_dim + 1) + (hidden_dim + 1) * 2) * hidden_dim
    equal = projection + (hidden_dim + 1) * 4
    gate = (3 * hidden_dim + 1) * 3
    return {'legacy': baseline, 'wide_mlp': wide, 'equal': equal,
            'softmax_gate': equal + gate,
            'softmax_gate_residual': equal + gate + 6 * hidden_dim * 4}
