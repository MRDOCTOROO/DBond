"""Independent scientific checks for two full feature routes; execute only on pod."""
import copy
import math

import torch
from torch.nn import functional as F

from models.parallel_sequence_encoder import ParallelSequenceTransformer1


def _explicit_reference(layer, x, lengths):
    """Expand LN, per-head attention and FFN without calling any encoder forward."""
    block = layer.encoder
    width = x.size(1)
    valid = torch.arange(width, device=x.device)[None, :] < lengths[:, None]
    clean = x.masked_fill(~valid[..., None], 0)
    rows = []
    for row, length in enumerate(lengths.tolist()):
        if length == 0:
            rows.append(clean[row] * 0)
            continue
        # Only true residues participate in the independently expanded formula.
        base = clean[row, :length]
        first = F.layer_norm(base, (256,), block.norm1.weight,
                             block.norm1.bias, block.norm1.eps)
        qkv = F.linear(first, block.self_attn.in_proj_weight,
                       block.self_attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        heads = []
        for head in range(8):
            qh = q[:, 32 * head:32 * (head + 1)]
            kh = k[:, 32 * head:32 * (head + 1)]
            vh = v[:, 32 * head:32 * (head + 1)]
            alpha = (qh @ kh.transpose(0, 1) / math.sqrt(32)).softmax(-1)
            heads.append(alpha @ vh)
        attention = F.linear(torch.cat(heads, dim=-1), block.self_attn.out_proj.weight,
                             block.self_attn.out_proj.bias)
        middle = base + attention
        second = F.layer_norm(middle, (256,), block.norm2.weight,
                              block.norm2.bias, block.norm2.eps)
        feedforward = F.linear(F.gelu(F.linear(second, block.linear1.weight,
                                               block.linear1.bias)),
                               block.linear2.weight, block.linear2.bias)
        context = middle + feedforward
        projected = F.linear(context, layer.out_proj.weight, layer.out_proj.bias)
        rows.append(F.pad(projected, (0, 0, 0, width - length)))
    return torch.stack(rows) if rows else x.new_zeros(x.shape)


def formula_check(config):
    layer = ParallelSequenceTransformer1(config.hidden_dim).double().eval()
    assert not layer.out_proj.weight.any() and not layer.out_proj.bias.any()
    assert layer.encoder.norm_first and layer.encoder.self_attn.batch_first
    assert layer.encoder.self_attn.num_heads == 8
    assert layer.encoder.linear1.out_features == 512
    assert all(module.p == 0 for module in layer.modules()
               if isinstance(module, torch.nn.Dropout))
    # Nonzero output projection only in a disposable verification fixture.
    with torch.no_grad():
        layer.out_proj.weight.copy_(torch.randn_like(layer.out_proj.weight) * .02)
        layer.out_proj.bias.copy_(torch.randn_like(layer.out_proj.bias) * .01)
    reference = copy.deepcopy(layer)
    x = torch.randn(3, 5, 256, dtype=torch.float64, requires_grad=True)
    lengths = torch.tensor([4, 2, 0], dtype=torch.long)
    other = x.detach().clone().requires_grad_()
    rng = torch.get_rng_state().clone()
    observed = layer(x, lengths)
    assert torch.equal(rng, torch.get_rng_state()), 'Dropout-free branch changed RNG'
    expected = _explicit_reference(reference, other, lengths)
    torch.testing.assert_close(observed, expected, atol=1e-10, rtol=1e-10)
    probe = torch.randn_like(observed)
    (observed * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(x.grad, other.grad, atol=1e-9, rtol=1e-9)
    for (name, parameter), (other_name, other_parameter) in zip(
            layer.named_parameters(), reference.named_parameters()):
        assert name == other_name
        assert parameter.grad is not None and other_parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, other_parameter.grad,
                                   atol=1e-9, rtol=1e-9, msg=name)
    valid = torch.arange(5)[None, :] < lengths[:, None]
    assert not observed[~valid].any() and not x.grad[~valid].any()
    changed = x.detach().clone()
    changed[~valid] = float('nan')
    clean_output = layer(changed, lengths)
    torch.testing.assert_close(clean_output, observed, atol=1e-10, rtol=1e-10)
    assert torch.isfinite(clean_output).all()
    changed = x.detach().clone()
    changed[1] = changed[1] + 100
    torch.testing.assert_close(layer(changed, lengths)[0], observed[0], atol=1e-10, rtol=1e-10)
    permutation = torch.tensor([2, 0, 1])
    torch.testing.assert_close(layer(x.detach()[permutation], lengths[permutation]),
                               observed[permutation], atol=1e-10, rtol=1e-10)
    extended = F.pad(x.detach(), (0, 0, 0, 3), value=float('nan'))
    torch.testing.assert_close(layer(extended, lengths)[:, :5], observed, atol=1e-10, rtol=1e-10)
    for shape, lens in (((0, 5, 256), []), ((2, 0, 256), [0, 0])):
        result = layer(torch.empty(shape, dtype=torch.float64), torch.tensor(lens, dtype=torch.long))
        assert result.shape == shape and torch.isfinite(result).all() and not result.any()
    return {'zero_initialized_output': True,
            'independent_preLN_MHA_FFN_forward_and_all_parameter_gradients': True,
            'each_head_projects_complete_256_dimensions_to_32': True,
            'padding_outputs_and_input_gradients_zero': True,
            'NaN_padding_and_padding_width_invariance': True,
            'cross_peptide_isolation_and_batch_permutation': True,
            'empty_sequences_finite_zero': True,
            'dropout_zero_forward_RNG_unchanged': True}


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
    control.last_four_state_logits = None
    candidate.last_four_state_logits = None
    fixture = copy.deepcopy(candidate).eval()
    fixture.enable_parallel_sequence_diagnostics = True
    with torch.no_grad():
        # k5 itself starts as an identity residual. Activate only this disposable
        # fixture so pre-k5 and post-k5 locations cannot pass the same assertion.
        fixture.node_context_conv.projection.weight.copy_(
            torch.randn_like(fixture.node_context_conv.projection.weight) * .01)
        fixture.parallel_sequence_context.out_proj.weight.copy_(
            torch.randn_like(fixture.parallel_sequence_context.out_proj.weight) * .01)
        fixture.parallel_sequence_context.out_proj.bias.fill_(.005)
    captured = {'gcn_calls': [0, 0, 0], 'gat_calls': [0, 0]}
    def node_encoder(module, args, output):
        captured['S'] = output.detach().clone()
    def convolution(module, args, output):
        captured['conv_input'] = args[0].detach().clone()
        captured['conv_output'] = output.detach().clone()
    def branch(module, args, output):
        captured['branch_input'] = args[0].detach().clone()
        captured['branch_output'] = output.detach().clone()
    def count(kind, index):
        def hook(module, args, output):
            captured[kind][index] += 1
            if kind == 'gat_calls' and index == 1:
                captured['graph_output'] = output.detach().clone()
        return hook
    def head(module, args):
        captured['bond_features'] = args[0].detach().clone()
    handles = [fixture.node_encoder.register_forward_hook(node_encoder),
               fixture.node_context_conv.register_forward_hook(convolution),
               fixture.parallel_sequence_context.register_forward_hook(branch),
               fixture.bond_head[0].register_forward_pre_hook(head)]
    handles += [layer.register_forward_hook(count('gcn_calls', i))
                for i, layer in enumerate(fixture.gcn_layers)]
    handles += [layer.register_forward_hook(count('gat_calls', i))
                for i, layer in enumerate(fixture.gat_layers)]
    with torch.no_grad():
        output = fixture(batch)
    for handle in handles:
        handle.remove()
    assert fixture.parallel_sequence_encoder_mode == 'transformer1'
    for field, value in (('mode', 1), ('schema', 1), ('depth', 1), ('heads', 8),
                         ('ffn', 512), ('dropout', 0), ('input', 1), ('fusion', 1)):
        assert int(getattr(fixture, 'parallel_sequence_' + field + '_code')) == value
    assert captured['gcn_calls'] == [1, 1, 1] and captured['gat_calls'] == [1, 1]
    assert torch.equal(captured['S'], captured['conv_input'])
    assert torch.equal(captured['S'], captured['branch_input'])
    assert not torch.equal(captured['conv_output'], captured['branch_input'])
    diag = fixture.last_parallel_sequence_diagnostics
    assert torch.equal(captured['S'], diag['branch_input'])
    assert torch.equal(captured['graph_output'], diag['graph_output'])
    graph = captured['graph_output']
    src = []; dst = []; residue_ids = []; global_ids = []; offset = 0
    lengths = batch['seq_lens'].tolist()
    for length, nodes in zip(lengths, batch['node_lens'].tolist()):
        residue_ids.extend(range(offset, offset + length))
        global_ids.extend(range(offset + length, offset + nodes))
        src.extend(range(offset, offset + max(length - 1, 0)))
        dst.extend(range(offset + 1, offset + max(length - 1, 0) + 1))
        offset += nodes
    valid = torch.arange(captured['branch_output'].size(1))[None, :] < torch.tensor(lengths)[:, None]
    expected_delta = torch.zeros_like(graph)
    expected_delta[residue_ids] = captured['branch_output'][valid]
    assert torch.equal(diag['packed_residue_indices'], torch.tensor(residue_ids, device=graph.device))
    torch.testing.assert_close(diag['packed_residual'], expected_delta, atol=1e-7, rtol=1e-7)
    assert not expected_delta[global_ids].any() and expected_delta[residue_ids].abs().sum() > 0
    fused = graph + expected_delta
    torch.testing.assert_close(diag['fused_output'], fused, atol=1e-7, rtol=1e-7)
    assert torch.equal(diag['fused_output'][global_ids], graph[global_ids])
    assert offset == graph.shape[0] and len(src) == int(batch['label_mask'].sum())
    torch.testing.assert_close(captured['bond_features'][:, :256], fused[src], atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(captured['bond_features'][:, 256:512], fused[dst], atol=1e-7, rtol=1e-7)
    assert torch.isfinite(output[batch['label_mask'].bool()]).all()
    return {'initial_logits_and_four_states_exact': True, 'initial_forward_RNG_exact': True,
            'original_3GCN_and_2GAT_executed_once': True,
            'independent_sequence_branch_reads_pre_k5_NodeEncoder_S': True,
            'both_routes_share_same_S_not_post_conv': True,
            'nonzero_fixture_residue_alignment_and_global_unchanged': True,
            'fused_residues_reach_both_bond_endpoints': True,
            'packed_nodes': offset, 'valid_bonds': len(src)}
