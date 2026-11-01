"""One independent, masked sequence-Transformer route beside the legacy GNN.

The input is the full NodeEncoder output, before k5, packing and FiLM. Each
peptide is a separate batch item: this route has no graph/global tokens and
cannot attend to another peptide. Only its zero-output residual is fused into
the true residue rows after the original two GAT layers.
"""

import torch
from torch import nn


_MODE_CODES = {'transformer1': 1}
_MARKER_VALUES = {
    'parallel_sequence_mode_code': 1,
    'parallel_sequence_schema_code': 1,
    'parallel_sequence_depth_code': 1,
    'parallel_sequence_heads_code': 8,
    'parallel_sequence_ffn_code': 512,
    'parallel_sequence_dropout_code': 0,
    'parallel_sequence_input_code': 1,  # complete pre-k5 NodeEncoder output
    'parallel_sequence_fusion_code': 1,  # post-GAT, residue-only additive residual
}
_BRANCH_PREFIX = 'parallel_sequence_context.'


def validate_parallel_sequence_encoder_mode(value):
    if value not in ('none', 'transformer1'):
        raise ValueError(f'Unknown parallel_sequence_encoder_mode: {value!r}')
    return value


def parallel_sequence_marker_values():
    """Return a copy so callers cannot mutate the checkpoint schema."""
    return dict(_MARKER_VALUES)


def validate_parallel_sequence_checkpoint(model, state_dict, checkpoint_config=None):
    """Reject cross-route or incomplete branch loads before any tensor is copied.

    This guard deliberately applies to strict=False as well. Tensor shape alone
    does not identify the input location, padding convention or fusion route.
    """
    requested = validate_parallel_sequence_encoder_mode(
        model.parallel_sequence_encoder_mode)
    expected_state = model.state_dict()

    def route_keys(mapping):
        return {key for key in mapping if key.startswith(_BRANCH_PREFIX)
                or key.startswith('parallel_sequence_')}

    actual_keys = route_keys(state_dict)
    expected_keys = route_keys(expected_state)
    if actual_keys != expected_keys:
        raise ValueError(
            'Checkpoint parallel sequence route/marker keys mismatch: '
            f'missing={sorted(expected_keys - actual_keys)}, '
            f'unexpected={sorted(actual_keys - expected_keys)}')
    for key in expected_keys:
        actual = state_dict[key]
        expected = expected_state[key]
        if not torch.is_tensor(actual) or actual.shape != expected.shape:
            raise ValueError(f'Checkpoint parallel sequence tensor shape mismatch: {key}')
        if key in _MARKER_VALUES and (actual.dtype != torch.int64 or not torch.equal(
                actual.detach().cpu(), expected.detach().cpu())):
            raise ValueError(f'Checkpoint parallel sequence semantics mismatch: {key}')
    if requested == 'none' and actual_keys:
        raise ValueError('The none sequence route must not contain branch state')

    if checkpoint_config is not None:
        if not isinstance(checkpoint_config, dict):
            raise ValueError('Checkpoint config must be a dictionary')
        declared = checkpoint_config.get('model', checkpoint_config)
        if not isinstance(declared, dict):
            raise ValueError('Checkpoint model config must be a dictionary')
        saved = validate_parallel_sequence_encoder_mode(
            declared.get('parallel_sequence_encoder_mode', 'none'))
        if saved != requested:
            raise ValueError('Checkpoint parallel_sequence_encoder_mode config/state mismatch')


class ParallelSequenceTransformer1(nn.Module):
    """Return a padded [B,L,256] residual from one fixed Transformer layer.

    Every head learns projections of the complete 256-dimensional vector. The
    existing NodeEncoder already supplies position and allowed conditions; no
    extra position encoding, FiLM, tokens, dropout, layer or learned fusion gate
    is introduced here. The enclosing model constructs this module in fork_rng
    after all legacy initialization.
    """

    def __init__(self, hidden_dim=256):
        super().__init__()
        if hidden_dim != 256:
            raise ValueError('The transformer1 sequence route requires hidden_dim=256')
        self.hidden_dim = hidden_dim
        self.encoder = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512, dropout=0.0,
            activation='gelu', layer_norm_eps=1e-5, batch_first=True,
            norm_first=True, bias=True)
        self.out_proj = nn.Linear(256, 256, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.enable_diagnostics = False
        self.last_diagnostics = {}

    def encode(self, inputs, seq_lens):
        """Return (clean input, valid mask, Transformer context), padding-safe."""
        if inputs.ndim != 3 or inputs.size(-1) != self.hidden_dim:
            raise ValueError('Parallel sequence input must have shape [B,L,256]')
        if not torch.is_tensor(seq_lens) or seq_lens.ndim != 1:
            raise ValueError('Parallel sequence lengths must be a one-dimensional tensor')
        if seq_lens.numel() != inputs.size(0):
            raise ValueError('Parallel sequence lengths disagree with batch size')
        if seq_lens.dtype == torch.bool or seq_lens.is_floating_point():
            raise ValueError('Parallel sequence lengths must be integer-valued tensors')
        lengths = seq_lens.to(device=inputs.device, dtype=torch.long)
        if bool(((lengths < 0) | (lengths > inputs.size(1))).any()):
            raise ValueError('Parallel sequence lengths are outside the padded input')
        positions = torch.arange(inputs.size(1), device=inputs.device)
        valid = positions[None, :] < lengths[:, None]
        # masked_fill, not multiplication: NaN or Inf in padding must not leak.
        clean = inputs.masked_fill(~valid.unsqueeze(-1), 0.0)
        context = clean.new_zeros(clean.shape)
        nonempty_rows = torch.nonzero(lengths > 0, as_tuple=False).flatten()
        # Do not pass an all-masked row into attention. Empty peptides/empty
        # batches have an exactly zero context and never create a dummy token.
        if nonempty_rows.numel() > 0 and inputs.size(1) > 0:
            selected = clean.index_select(0, nonempty_rows)
            padding = ~valid.index_select(0, nonempty_rows)
            encoded = self.encoder(selected, src_key_padding_mask=padding)
            encoded = encoded.masked_fill(padding.unsqueeze(-1), 0.0)
            context = context.index_copy(0, nonempty_rows, encoded.to(context.dtype))
        return clean, valid, context

    def forward(self, inputs, seq_lens):
        clean, valid, context = self.encode(inputs, seq_lens)
        residual = self.out_proj(context).masked_fill(~valid.unsqueeze(-1), 0.0)
        self.last_diagnostics = {}
        if self.enable_diagnostics:
            self.last_diagnostics = {
                'input': clean.detach(), 'valid_mask': valid.detach(),
                'context': context.detach(), 'residual': residual.detach(),
            }
        return residual
