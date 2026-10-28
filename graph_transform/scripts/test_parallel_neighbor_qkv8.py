"""Bounded scientific preflight checks; numerical execution only on pod."""
import copy
import math
import torch
from torch.nn import functional as F
from models.parallel_neighbor_attention import ParallelNeighborQKV8


def formula_check(config):
    layer=ParallelNeighborQKV8(config).double().eval()
    assert not layer.out_proj.weight.any() and not layer.out_proj.bias.any()
    # Activate only a disposable test fixture, never a formal model/checkpoint.
    with torch.no_grad():
        layer.out_proj.weight.copy_(torch.randn_like(layer.out_proj.weight)*.02)
        layer.out_proj.bias.copy_(torch.randn_like(layer.out_proj.bias)*.01)
    reference=copy.deepcopy(layer)
    x=torch.randn(8,256,dtype=torch.float64,requires_grad=True)
    edges=torch.tensor([[0,1,2,4,5,6],[2,2,0,6,6,4]])
    attr=torch.randn(6,256,dtype=torch.float64,requires_grad=True)
    other_x=x.detach().clone().requires_grad_(); other_e=attr.detach().clone().requires_grad_()
    observed=layer(x,edges,attr)
    q=F.linear(other_x,reference.q_proj.weight).reshape(8,8,32)
    k=F.linear(other_x,reference.k_proj.weight).reshape(8,8,32)
    v=F.linear(other_x,reference.v_proj.weight).reshape(8,8,32)
    bias=F.linear(other_e,reference.edge_attention_bias.weight,reference.edge_attention_bias.bias)
    gate=F.linear(other_e,reference.edge_message_gate.weight,reference.edge_message_gate.bias).sigmoid().reshape(6,8,32)
    rows=[]
    for center in range(8):
        ids=(edges[1]==center).nonzero().flatten()
        heads=[]
        for head in range(8):
            if ids.numel():
                score=torch.stack([(q[center,head]*k[edges[0,j],head]).sum()/math.sqrt(32)+bias[j,head] for j in ids])
                alpha=score.softmax(0)
                heads.append(sum(alpha[t]*gate[j,head]*v[edges[0,j],head] for t,j in enumerate(ids)))
            else:
                heads.append(v.new_zeros(32))
        rows.append(F.linear(torch.cat(heads),reference.out_proj.weight,reference.out_proj.bias)
                    if ids.numel() else other_x.new_zeros(256))
    expected=torch.stack(rows)
    torch.testing.assert_close(observed,expected,atol=1e-10,rtol=1e-10)
    probe=torch.randn_like(observed)
    (observed*probe).sum().backward(); (expected*probe).sum().backward()
    torch.testing.assert_close(x.grad,other_x.grad,atol=1e-9,rtol=1e-9)
    torch.testing.assert_close(attr.grad,other_e.grad,atol=1e-9,rtol=1e-9)
    for (name,p),(other,r) in zip(layer.named_parameters(),reference.named_parameters()):
        assert name==other and p.grad is not None and r.grad is not None
        torch.testing.assert_close(p.grad,r.grad,atol=1e-9,rtol=1e-9)
    order=torch.tensor([5,2,0,3,1,4])
    torch.testing.assert_close(layer(x,edges[:,order],attr[order]),observed,atol=1e-10,rtol=1e-10)
    changed=x.detach().clone(); changed[4:]+=100
    torch.testing.assert_close(layer(changed,edges,attr)[:4],observed[:4],atol=1e-10,rtol=1e-10)
    for n in (0,1):
        result=layer(x.detach()[:n],torch.empty(2,0,dtype=torch.long),attr.detach()[:0])
        assert result.shape==(n,256) and torch.isfinite(result).all() and not result.any()
    return {'zero_initialized_output':True,'independent_neighbor_QKV_forward_and_all_gradients':True,
            'edge_order_invariance':True,'disconnected_sample_isolation':True,'empty_isolated_nodes_zero_context':True}


def initial_equivalence_and_routing(control,candidate,batch):
    control.train(); candidate.train()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        before=control(batch).detach(); old_states=control.last_four_state_logits.detach().clone()
        old_rng=torch.get_rng_state().clone()
        torch.manual_seed(123)
        after=candidate(batch).detach(); new_states=candidate.last_four_state_logits.detach().clone()
        new_rng=torch.get_rng_state().clone()
    assert torch.equal(before,after) and torch.equal(old_states,new_states) and torch.equal(old_rng,new_rng)
    control.last_four_state_logits = None
    candidate.last_four_state_logits = None
    fixture=copy.deepcopy(candidate).eval()
    with torch.no_grad():
        fixture.parallel_neighbor_context.out_proj.weight.copy_(
            torch.randn_like(fixture.parallel_neighbor_context.out_proj.weight)*.01)
        fixture.parallel_neighbor_context.out_proj.bias.fill_(.005)
    captured={}
    def first_gat(module,args): captured['shared_input']=args[0].detach().clone()
    def last_gat(module,args,output): captured['original_output']=output.detach().clone()
    def branch(module,args,output):
        captured['branch_input']=args[0].detach().clone(); captured['residual']=output.detach().clone()
    def head(module,args): captured['bond_features']=args[0].detach().clone()
    handles=[fixture.gat_layers[0].register_forward_pre_hook(first_gat),
             fixture.gat_layers[-1].register_forward_hook(last_gat),
             fixture.parallel_neighbor_context.register_forward_hook(branch),
             fixture.bond_head[0].register_forward_pre_hook(head)]
    with torch.no_grad(): output=fixture(batch)
    for h in handles:h.remove()
    assert torch.equal(captured['shared_input'],captured['branch_input'])
    fused=captured['original_output']+captured['residual']
    src=[];dst=[];offset=0
    for length,nodes in zip(batch['seq_lens'].tolist(),batch['node_lens'].tolist()):
        src.extend(range(offset,offset+max(length-1,0)))
        dst.extend(range(offset+1,offset+max(length-1,0)+1))
        offset+=nodes
    assert offset==fused.shape[0] and len(src)==int(batch['label_mask'].sum())
    torch.testing.assert_close(captured['bond_features'][:,:256],fused[src],atol=1e-7,rtol=1e-7)
    torch.testing.assert_close(captured['bond_features'][:,256:512],fused[dst],atol=1e-7,rtol=1e-7)
    assert torch.isfinite(output[batch['label_mask'].bool()]).all()
    return {'initial_logits_and_four_states_exact':True,'initial_forward_rng_exact':True,
            'original_two_GAT_layers_present':len(fixture.gat_layers)==2,
            'same_post_GCN_input_for_both_paths':True,'nonzero_context_reaches_both_bond_endpoints':True,
            'packed_nodes':offset,'valid_bonds':len(src)}
