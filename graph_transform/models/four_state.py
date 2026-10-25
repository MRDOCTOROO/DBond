"""Joint operational b/y observability, state order none/b-only/y-only/both."""
import math
import torch
from torch import nn


def replace_binary_output(binary):
    """Initialize from the fresh binary head, preserving its initial union q.

    L0=0 and L1=L2=L3=z-log(3) imply logsumexp(L1:L3)-L0=z.
    No pretrained checkpoint is used. Fork RNG to preserve backbone/dropout RNG.
    """
    with torch.random.fork_rng(devices=[]):
        joint = nn.Linear(binary.in_features, 4)
        with torch.no_grad():
            joint.weight[0].zero_()
            joint.bias[0].zero_()
            joint.weight[1:].copy_(binary.weight.expand(3, -1))
            joint.bias[1:].copy_(binary.bias.expand(3) - math.log(3))
    return joint


def union_logit(logits):
    logits = logits.float()
    return torch.logsumexp(logits[..., 1:], dim=-1) - logits[..., 0]
