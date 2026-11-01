"""Independent bond-query formula and graph-routing checks; execute only on pod."""
import copy
import math

import torch
from torch.nn import functional as F

from models.bond_neighbor_context import BondNeighborQKV8, build_bond_neighbor_pairs


def independent_neighbor_sets(edge_index, bond_src, bond_dst, seq_lens, node_lens, num_nodes):
    """Python-set oracle, independently derived from packed graph ownership."""
    owner = [-1] * num_nodes
    real = [False] * num_nodes
    offset = 0
    for graph, (length, count) in enumerate(zip(seq_lens.tolist(), node_lens.tolist())):
        for node in range(offset, offset + count):
            owner[node] = graph
            real[node] = node < offset + length
        offset += count
    assert offset == num_nodes
    edges = edge_index.detach().cpu().t().tolist()
    sets = []
    for left, right in zip(bond_src.tolist(), bond_dst.tolist()):
        assert real[left] and real[right] and owner[left] == owner[right]
        neighbors = {left, right}
        for source, target in edges:
            if target in (left, right) and real[source] and owner[source] == owner[left]:
                neighbors.add(source)
        sets.append(sorted(neighbors))
    return sets, real


def explicit_reference(layer, h, edges, left, right, edge_repr, seq_lens, node_lens):
    """Expand each bond/head without invoking module forward or pair builder."""
    sets, _ = independent_neighbor_sets(edges, left, right, seq_lens, node_lens, h.size(0))
    results = []
    for index, neighbors in enumerate(sets):
        query_input = torch.cat((h[left[index]], h[right[index]], edge_repr[index]))
        query = F.linear(query_input, layer.query_proj.weight, layer.query_proj.bias)
        keys = F.linear(h[neighbors], layer.key_proj.weight, layer.key_proj.bias)
        values = F.linear(h[neighbors], layer.value_proj.weight, layer.value_proj.bias)
        heads = []
        for head in range(8):
            start, stop = 32 * head, 32 * (head + 1)
            score = (keys[:, start:stop] * query[start:stop]).sum(-1) / math.sqrt(32)
            alpha = score.softmax(0)
            heads.append((alpha[:, None] * values[:, start:stop]).sum(0))
        results.append(F.linear(torch.cat(heads), layer.output_proj.weight, layer.output_proj.bias))
    return torch.stack(results) if results else h.new_empty((0, 256))


def formula_check(config):
    layer = BondNeighborQKV8(config.hidden_dim).double().eval()
    assert not layer.output_proj.weight.any() and not layer.output_proj.bias.any()
    assert sum(p.numel() for p in layer.parameters()) == 394240
    assert not any(isinstance(m, (torch.nn.LayerNorm, torch.nn.Dropout)) for m in layer.modules())
    # Activate only this disposable fixture, never the actual initialization.
    with torch.no_grad():
        layer.output_proj.weight.copy_(torch.randn_like(layer.output_proj.weight) * .02)
        layer.output_proj.bias.copy_(torch.randn_like(layer.output_proj.bias) * .01)
    layer.enable_diagnostics = True
    reference = copy.deepcopy(layer)
    lengths = torch.tensor([4, 3, 0])
    node_lens = torch.tensor([5, 4, 1])
    left = torch.tensor([0, 1, 2, 5, 6])
    right = left + 1
    # Deliberately includes duplicates, directed edges, globals and cross-graph edges.
    edges = torch.tensor([[0, 1, 2, 3, 4, 0, 2, 2, 5, 6, 7, 8, 9, 5, 0, 7],
                          [1, 0, 1, 2, 1, 1, 1, 1, 6, 7, 6, 6, 5, 1, 6, 0]])
    h = torch.randn(10, 256, dtype=torch.float64, requires_grad=True)
    e = torch.randn(5, 256, dtype=torch.float64, requires_grad=True)
    other_h = h.detach().clone().requires_grad_()
    other_e = e.detach().clone().requires_grad_()
    sets, real = independent_neighbor_sets(edges, left, right, lengths, node_lens, 10)
    pairs, mask = build_bond_neighbor_pairs(edges, left, right, lengths, node_lens, 10)
    expected_pairs = [(bond, node) for bond, nodes in enumerate(sets) for node in nodes]
    assert sorted(map(tuple, pairs.t().tolist())) == expected_pairs
    assert len(set(map(tuple, pairs.t().tolist()))) == pairs.size(1)
    assert mask.tolist() == real
    rng = torch.get_rng_state().clone()
    observed = layer(h, edges, left, right, e, lengths, node_lens)
    assert torch.equal(rng, torch.get_rng_state()), 'Attention branch consumed RNG'
    expected = explicit_reference(reference, other_h, edges, left, right, other_e, lengths, node_lens)
    torch.testing.assert_close(observed, expected, atol=1e-10, rtol=1e-10)
    diagnostic = layer.last_diagnostics
    assert torch.equal(diagnostic['query_input'], torch.cat((h[left], h[right], e), -1))
    assert torch.equal(diagnostic['pair_index'], pairs)
    for bond in range(left.numel()):
        selected = pairs[0] == bond
        torch.testing.assert_close(diagnostic['attention'][selected].sum(0),
                                   torch.ones(8, dtype=torch.float64), atol=1e-12, rtol=1e-12)
    probe = torch.randn_like(observed)
    (observed * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(h.grad, other_h.grad, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(e.grad, other_e.grad, atol=1e-9, rtol=1e-9)
    for (name, p), (other_name, other) in zip(layer.named_parameters(), reference.named_parameters()):
        assert name == other_name and p.grad is not None and other.grad is not None, name
        torch.testing.assert_close(p.grad, other.grad, atol=1e-9, rtol=1e-9, msg=name)
    assert not h.grad[~mask].any(), 'Excluded global node received branch gradient'
    # Edge order/duplicates do not reweight repeated residues in the union.
    permutation = torch.arange(edges.size(1) - 1, -1, -1)
    noisy_edges = torch.cat((edges[:, permutation], edges[:, :4], edges[:, :4]), dim=1)
    torch.testing.assert_close(layer(h.detach(), noisy_edges, left, right, e.detach(), lengths, node_lens),
                               observed.detach(), atol=1e-10, rtol=1e-10)
    bond_order = torch.tensor([4, 0, 3, 1, 2])
    torch.testing.assert_close(layer(h.detach(), edges, left[bond_order], right[bond_order],
                                     e.detach()[bond_order], lengths, node_lens),
                               observed.detach()[bond_order], atol=1e-10, rtol=1e-10)
    # A second peptide's values cannot affect the first peptide's bond context.
    changed = h.detach().clone(); changed[5:8] += 100
    changed_e = e.detach().clone(); changed_e[3:] -= 100
    torch.testing.assert_close(layer(changed, edges, left, right, changed_e, lengths, node_lens)[:3],
                               observed.detach()[:3], atol=1e-10, rtol=1e-10)
    poisoned = h.detach().clone(); poisoned[~mask] = float('nan'); poisoned.requires_grad_()
    layer.zero_grad(set_to_none=True)
    poisoned_output = layer(poisoned, edges, left, right, e.detach(), lengths, node_lens)
    torch.testing.assert_close(poisoned_output, observed.detach(), atol=1e-10, rtol=1e-10)
    poisoned_output.sum().backward()
    assert torch.isfinite(poisoned.grad).all() and not poisoned.grad[~mask].any()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
    # Empty neighbor edge list still reads the two endpoints.
    empty_edges = edges[:, :0]
    endpoint_output = layer(h.detach(), empty_edges, left, right, e.detach(), lengths, node_lens)
    endpoint_reference = explicit_reference(reference, h.detach(), empty_edges, left, right,
                                             e.detach(), lengths, node_lens)
    torch.testing.assert_close(endpoint_output, endpoint_reference, atol=1e-10, rtol=1e-10)
    empty = layer(h.detach(), edges, left[:0], right[:0], e.detach()[:0], lengths, node_lens)
    assert empty.shape == (0, 256) and torch.isfinite(empty).all()
    return {'zero_initialized_output': True, 'parameter_count': 394240,
            'independent_python_set_union_matches_deduplicated_KV': True,
            'independent_per_bond_per_head_forward_and_all_gradients': True,
            'each_head_projects_complete_256_features_to_32': True,
            'query_uses_complete_left_right_and_original_edge_representation': True,
            'neighbor_softmax_sums_one_and_contains_endpoints': True,
            'no_global_or_cross_graph_KV_and_NaN_global_forward_backward_isolated': True,
            'edge_order_and_duplicate_edge_invariance': True,
            'bond_permutation_equivariance': True, 'cross_peptide_isolation': True,
            'empty_edges_endpoint_fallback_and_empty_bonds_finite': True,
            'no_new_LN_dropout_FFN_or_rng_consumption': True}


def initial_equivalence_and_routing(control, candidate, batch):
    control.train(); candidate.train()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        before = control(batch).detach()
        old_states = control.last_four_state_logits.detach().clone()
        old_rng = torch.get_rng_state().clone()
        torch.manual_seed(123)
        after = candidate(batch).detach()
        new_states = candidate.last_four_state_logits.detach().clone()
        new_rng = torch.get_rng_state().clone()
    assert torch.equal(before, after) and torch.equal(old_states, new_states)
    assert torch.equal(old_rng, new_rng)
    control.last_four_state_logits = None; candidate.last_four_state_logits = None
    fixture = copy.deepcopy(candidate).eval()
    fixture.enable_bond_neighbor_diagnostics = True
    for name, value in (('mode', 1), ('schema', 1), ('heads', 8), ('query', 1),
                        ('neighborhood', 1), ('input', 1), ('fusion', 1), ('projection', 1)):
        assert int(getattr(fixture, 'bond_neighbor_context_' + name + '_code')) == value
    with torch.no_grad():
        fixture.bond_neighbor_context.output_proj.weight.copy_(
            torch.randn_like(fixture.bond_neighbor_context.output_proj.weight) * .01)
        fixture.bond_neighbor_context.output_proj.bias.fill_(.005)
    captured = {'gcn_calls': [0, 0, 0], 'gat_calls': [0, 0]}
    def count(kind, index):
        def hook(module, args, output):
            captured[kind][index] += 1
            if kind == 'gat_calls' and index == 1:
                captured['H5'] = output.detach().clone()
        return hook
    def branch(module, args, output):
        captured['branch_args'] = [value.detach().clone() for value in args]
        captured['delta'] = output.detach().clone()
    def head_input(module, args):
        captured['bond_features'] = args[0].detach().clone()
    def hidden(module, args, output):
        captured['base_hidden'] = output.detach().clone()
    def dropout_input(module, args):
        captured['fused_hidden'] = args[0].detach().clone()
    def final_state(module, args, output):
        captured['state_logits'] = output.detach().float().clone()
    handles = [fixture.bond_neighbor_context.register_forward_hook(branch),
               fixture.bond_head[0].register_forward_pre_hook(head_input),
               fixture.bond_head[1].register_forward_hook(hidden),
               fixture.bond_head[2].register_forward_pre_hook(dropout_input),
               fixture.bond_head[3].register_forward_hook(final_state)]
    handles += [layer.register_forward_hook(count('gcn_calls', i)) for i, layer in enumerate(fixture.gcn_layers)]
    handles += [layer.register_forward_hook(count('gat_calls', i)) for i, layer in enumerate(fixture.gat_layers)]
    with torch.no_grad():
        output = fixture(batch)
    for handle in handles:
        handle.remove()
    assert captured['gcn_calls'] == [1, 1, 1] and captured['gat_calls'] == [1, 1]
    diag = fixture.last_bond_neighbor_diagnostics
    h, edges, left, right, eij, lengths, node_lens = captured['branch_args']
    assert torch.equal(h, captured['H5']) and torch.equal(diag['H5'], captured['H5'])
    assert torch.equal(edges, batch['edge_index'])
    assert torch.equal(lengths, batch['seq_lens']) and torch.equal(node_lens, batch['node_lens'])
    # Independently reconstruct valid bond endpoints from sequence lengths and packing.
    expected_left = []; offset = 0
    for length, count_nodes in zip(lengths.tolist(), node_lens.tolist()):
        expected_left.extend(range(offset, offset + max(length - 1, 0)))
        offset += count_nodes
    assert left.tolist() == expected_left and torch.equal(right, left + 1)
    assert len(expected_left) == int(batch['label_mask'].sum())
    torch.testing.assert_close(captured['bond_features'][:, :256], h[left], atol=0, rtol=0)
    torch.testing.assert_close(captured['bond_features'][:, 256:512], h[right], atol=0, rtol=0)
    torch.testing.assert_close(captured['bond_features'][:, 512:768], eij, atol=0, rtol=0)
    assert captured['bond_features'].shape[1] == 6 * 256
    query_input = torch.cat((h[left], h[right], eij), -1)
    assert torch.equal(fixture.bond_neighbor_context.last_diagnostics['query_input'], query_input)
    sets, real = independent_neighbor_sets(edges, left, right, lengths, node_lens, h.size(0))
    expected_pairs = [(bond, node) for bond, nodes in enumerate(sets) for node in nodes]
    pair_index = fixture.bond_neighbor_context.last_diagnostics['pair_index']
    assert sorted(map(tuple, pair_index.t().tolist())) == expected_pairs
    reference = explicit_reference(fixture.bond_neighbor_context, h, edges, left, right, eij, lengths, node_lens)
    torch.testing.assert_close(captured['delta'], reference, atol=2e-6, rtol=2e-6)
    expected_hidden = captured['base_hidden'] + captured['delta']
    assert captured['delta'].abs().sum() > 0
    torch.testing.assert_close(captured['fused_hidden'], expected_hidden, atol=0, rtol=0)
    assert torch.equal(diag['base_hidden'], captured['base_hidden'])
    assert torch.equal(diag['fused_hidden'], captured['fused_hidden'])
    expected_states = fixture.bond_head[3](fixture.bond_head[2](expected_hidden)).float()
    torch.testing.assert_close(captured['state_logits'], expected_states, atol=0, rtol=0)
    assert torch.isfinite(output[batch['label_mask'].bool()]).all()
    return {'initial_logits_four_states_and_forward_RNG_bitwise_legacy': True,
            'original_3GCN_and_2GAT_each_executed_once': True,
            'Q_and_KV_read_unchanged_post_second_GAT_H5': True,
            'query_reads_left_right_and_original_edge_feature': True,
            'real_neighbor_union_and_endpoints_independently_reconstructed': True,
            'all_original_six_H_bond_features_unchanged': True,
            'nonzero_fixture_delta_adds_to_first_Linear_ReLU_hidden': True,
            'original_dropout_and_four_state_final_Linear_reused': True,
            'valid_bond_count': len(expected_left), 'packed_node_count': offset}
