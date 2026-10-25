"""Position routing and packed residue alignment for the optional GAT input."""
import torch


def validate_position_mode(value):
    if value not in ('input_concat', 'pre_attention_add'):
        raise ValueError(f'Unknown position_embedding_mode: {value!r}')
    return value


def validate_position_checkpoint(model, state_dict):
    # Legacy models retain their unmarked state schema. Check before copying
    # tensors, including strict=False loads used by the production evaluator.
    key = 'position_embedding_mode_code'
    has_marker = key in state_dict
    if has_marker:
        marker = state_dict[key]
        if (not torch.is_tensor(marker) or marker.dtype != torch.int64
                or marker.shape != torch.Size([]) or marker.item() != 1):
            raise ValueError('Invalid checkpoint position_embedding_mode_code')
    requested = model.position_embedding_mode == 'pre_attention_add'
    if has_marker != requested:
        raise ValueError('Checkpoint position_embedding_mode mismatch')
    expected = {key for key in model.state_dict() if key.startswith('pre_attention_position_proj.')}
    actual = {key for key in state_dict if key.startswith('pre_attention_position_proj.')}
    if actual != expected:
        raise ValueError('Checkpoint pre-attention position projection mismatch')


def add_packed_residue_positions(node_features, position_vectors, seq_lengths, node_lengths):
    """Add absolute positions to true residues only; globals receive zero.

    Nodes are packed in sample order, each with an optional trailing global.
    The original sequence-relative graph edges are unaffected by this routing.
    """
    positions = torch.arange(position_vectors.size(0), device=node_features.device)
    valid = positions.unsqueeze(0) < seq_lengths.unsqueeze(1)
    offsets = torch.cumsum(torch.cat([node_lengths.new_zeros(1), node_lengths[:-1]]), dim=0)
    indices = (offsets.unsqueeze(1) + positions.unsqueeze(0))[valid]
    vectors = position_vectors.unsqueeze(0).expand(seq_lengths.numel(), -1, -1)[valid]
    additions = torch.zeros_like(node_features)
    additions[indices] = vectors.to(node_features.dtype)
    return node_features + additions
