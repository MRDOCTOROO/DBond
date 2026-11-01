"""Opt-in bond-query attention over a union of true-residue graph neighbors.

This readout does not update graph nodes or create graph edges. A bond query
reads each real residue in N_in(left) union N_in(right) union {left,right}
once. Global nodes, padding and other peptides never become keys or values.
"""

import math

import torch
from torch import nn


BOND_NEIGHBOR_MARKER_VALUES = {
    'bond_neighbor_context_mode_code': 1,
    'bond_neighbor_context_schema_code': 1,
    'bond_neighbor_context_heads_code': 8,
    'bond_neighbor_context_query_code': 1,       # [H5_left,H5_right,e_ij]
    'bond_neighbor_context_neighborhood_code': 1,  # unique real incoming union
    'bond_neighbor_context_input_code': 1,       # original post-two-GAT H5
    'bond_neighbor_context_fusion_code': 1,      # add after head Linear+ReLU
    'bond_neighbor_context_projection_code': 1,  # affine projections, zero Wo
}


def validate_bond_neighbor_context_mode(value):
    if value not in ('none', 'bond_query_qkv8'):
        raise ValueError(f'Unknown bond_neighbor_context_mode: {value!r}')
    return value


def validate_bond_neighbor_checkpoint(model, state_dict, checkpoint_config=None):
    """Reject branch/semantic mismatch before copying any shared weights."""
    requested = validate_bond_neighbor_context_mode(model.bond_neighbor_context_mode)
    expected = model.state_dict()
    prefix = 'bond_neighbor_context'
    actual_keys = {key for key in state_dict if key.startswith(prefix)}
    expected_keys = {key for key in expected if key.startswith(prefix)}
    if actual_keys != expected_keys:
        raise ValueError('Checkpoint bond-neighbor marker/branch keys mismatch')
    for key in expected_keys:
        actual = state_dict[key]
        reference = expected[key]
        if (not torch.is_tensor(actual) or actual.shape != reference.shape
                or actual.dtype != reference.dtype):
            raise ValueError(f'Checkpoint bond-neighbor tensor schema mismatch: {key}')
        if key in BOND_NEIGHBOR_MARKER_VALUES and not torch.equal(
                actual.detach().cpu(), reference.detach().cpu()):
            raise ValueError(f'Checkpoint bond-neighbor semantics mismatch: {key}')
    # Do not accidentally load another experimental residue-context branch
    # with strict=False into the none route of this isolated implementation.
    for key in state_dict:
        if key.startswith(('parallel_neighbor_context.', 'parallel_neighbor_attention_',
                           'parallel_sequence_context.', 'parallel_sequence_encoder_')):
            raise ValueError('Checkpoint contains a different experimental context branch')
    if checkpoint_config is not None:
        if not isinstance(checkpoint_config, dict):
            raise ValueError('Checkpoint config must be a dictionary')
        declared = checkpoint_config.get('model', checkpoint_config)
        if not isinstance(declared, dict):
            raise ValueError('Checkpoint model config must be a dictionary')
        if validate_bond_neighbor_context_mode(
                declared.get('bond_neighbor_context_mode', 'none')) != requested:
            raise ValueError('Checkpoint bond-neighbor config/state mismatch')


def build_bond_neighbor_pairs(edge_index, bond_src, bond_dst,
                              seq_lens, node_lens, num_nodes):
    """Return ([bond index, packed residue index], real-node mask).

    Packed nodes are contiguous by peptide, true residues first and optional
    virtual nodes afterwards. Candidate bonds are the unique ordered adjacent
    residue pairs of each peptide. The resulting pair order is canonical and
    independent of graph-edge ordering or duplicate edges.
    """
    device = bond_src.device
    lengths = seq_lens.to(device=device, dtype=torch.long)
    packed_lengths = node_lens.to(device=device, dtype=torch.long)
    if (bond_src.ndim != 1 or bond_dst.shape != bond_src.shape
            or edge_index.ndim != 2 or edge_index.size(0) != 2
            or lengths.ndim != 1 or packed_lengths.shape != lengths.shape):
        raise ValueError('Invalid packed bond-neighbor index shapes')
    if (torch.any(lengths < 0) or torch.any(packed_lengths < lengths)
            or int(packed_lengths.sum().item()) != num_nodes):
        raise ValueError('Invalid packed bond-neighbor lengths')
    ends = torch.cumsum(packed_lengths, dim=0)
    offsets = ends - packed_lengths
    node_ids = torch.arange(num_nodes, device=device)
    sample_ids = torch.bucketize(node_ids, ends, right=True)
    real_mask = node_ids - offsets[sample_ids] < lengths[sample_ids]
    num_bonds = bond_src.numel()
    if num_bonds == 0:
        return torch.empty((2, 0), device=device, dtype=torch.long), real_mask
    if (torch.any(bond_src < 0) or torch.any(bond_dst >= num_nodes)
            or torch.any(bond_dst != bond_src + 1)):
        raise ValueError('Bond endpoints must be valid ordered adjacent residues')
    if (not bool(torch.all(real_mask[bond_src] & real_mask[bond_dst]))
            or not bool(torch.all(sample_ids[bond_src] == sample_ids[bond_dst]))
            or torch.unique(bond_src).numel() != num_bonds
            or torch.unique(bond_dst).numel() != num_bonds):
        raise ValueError('Bonds must be unique real-residue pairs within one peptide')

    # Each residue occurs at most once as a left endpoint and once as a right
    # endpoint. Map each incoming edge to zero, one or two neighboring bonds.
    left_lookup = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    right_lookup = torch.full_like(left_lookup, -1)
    bond_ids = torch.arange(num_bonds, device=device)
    left_lookup[bond_src] = bond_ids
    right_lookup[bond_dst] = bond_ids
    edge_src, edge_dst = edge_index.to(device=device, dtype=torch.long)
    in_range = ((edge_src >= 0) & (edge_src < num_nodes)
                & (edge_dst >= 0) & (edge_dst < num_nodes))
    edge_src, edge_dst = edge_src[in_range], edge_dst[in_range]
    admissible = (real_mask[edge_src] & real_mask[edge_dst]
                  & (sample_ids[edge_src] == sample_ids[edge_dst]))
    edge_src, edge_dst = edge_src[admissible], edge_dst[admissible]
    left_bonds, right_bonds = left_lookup[edge_dst], right_lookup[edge_dst]
    left_valid, right_valid = left_bonds >= 0, right_bonds >= 0
    pairs_bond = torch.cat((left_bonds[left_valid], right_bonds[right_valid],
                            bond_ids, bond_ids))
    pairs_node = torch.cat((edge_src[left_valid], edge_src[right_valid],
                            bond_src, bond_dst))
    encoded = torch.unique(pairs_bond * num_nodes + pairs_node, sorted=True)
    pairs = torch.stack((torch.div(encoded, num_nodes, rounding_mode='floor'),
                         encoded.remainder(num_nodes)))
    return pairs, real_mask


class BondNeighborQKV8(nn.Module):
    """Eight full-vector heads, producing only a zero-initialized bond delta."""

    def __init__(self, hidden_dim=256):
        super().__init__()
        if hidden_dim != 256:
            raise ValueError('The isolated bond-query experiment requires H256')
        self.hidden_dim = hidden_dim
        self.num_heads = 8
        self.head_dim = hidden_dim // self.num_heads
        self.query_proj = nn.Linear(3 * hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        for projection in (self.query_proj, self.key_proj, self.value_proj):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        self.enable_diagnostics = False
        self.last_diagnostics = {}

    def forward(self, node_features, edge_index, bond_src, bond_dst,
                e_ij, seq_lens, node_lens):
        self.last_diagnostics = {}
        if (node_features.ndim != 2 or node_features.size(-1) != self.hidden_dim
                or e_ij.shape != (bond_src.numel(), self.hidden_dim)):
            raise ValueError('Bond-query features require packed [N,256] and [Nb,256] edge states')
        pair_index, real_mask = build_bond_neighbor_pairs(
            edge_index, bond_src, bond_dst, seq_lens, node_lens,
            node_features.size(0))
        # Clearing invalid nodes before either affine projection prevents NaN
        # virtual-node values from contaminating the weight gradient (0*NaN).
        clean = node_features.masked_fill(~real_mask[:, None], 0.)
        query_input = torch.cat((clean[bond_src], clean[bond_dst], e_ij), dim=-1)
        q = self.query_proj(query_input).reshape(-1, self.num_heads, self.head_dim)
        k = self.key_proj(clean).reshape(-1, self.num_heads, self.head_dim)
        v = self.value_proj(clean).reshape(-1, self.num_heads, self.head_dim)
        compute_dtype = (torch.float32 if q.dtype in (torch.float16, torch.bfloat16)
                         else q.dtype)
        pair_bond, pair_node = pair_index
        scores = (q[pair_bond].to(compute_dtype) * k[pair_node].to(compute_dtype)).sum(-1)
        scores = scores / math.sqrt(self.head_dim)
        expanded = pair_bond[:, None].expand(-1, self.num_heads)
        maxima = scores.new_full((bond_src.numel(), self.num_heads), float('-inf'))
        maxima.scatter_reduce_(0, expanded, scores, reduce='amax', include_self=True)
        exponentials = (scores - maxima[pair_bond]).exp()
        denominators = scores.new_zeros((bond_src.numel(), self.num_heads))
        denominators.scatter_add_(0, expanded, exponentials)
        attention = exponentials / denominators[pair_bond].clamp_min(1e-12)
        messages = attention[:, :, None] * v[pair_node].to(compute_dtype)
        aggregated = scores.new_zeros((bond_src.numel(), self.num_heads, self.head_dim))
        aggregated.index_add_(0, pair_bond, messages)
        context = aggregated.reshape(-1, self.hidden_dim).to(dtype=q.dtype)
        delta = self.output_proj(context)
        if self.enable_diagnostics:
            self.last_diagnostics = {
                'mode': 'bond_query_qkv8',
                'node_features': node_features.detach(),
                'query_input': query_input.detach(),
                'pair_index': pair_index.detach(),
                'real_mask': real_mask.detach(),
                'q': q.detach(), 'k': k.detach(), 'v': v.detach(),
                'scores': scores.detach(),
                'attention': attention.detach(),
                'context': context.detach(), 'delta': delta.detach(),
            }
        return delta
