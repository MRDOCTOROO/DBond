"""Order-aware residual encoders before graph message passing.

The multiscale arm is inspired by PredFull's parallel sequence convolutions;
the BiGRU arm by Prosit's directional sequence encoder. These compact adapters
are experiments, not reproductions of those spectrum-prediction networks.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SequenceAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, kind, width=96, kernels=(3, 5, 7)):
        super().__init__()
        if kind not in ('multiscale', 'bigru'):
            raise ValueError(f'Unknown sequence adapter: {kind}')
        if width <= 0:
            raise ValueError('sequence_adapter_dim must be positive')
        self.kind = kind
        if kind == 'multiscale':
            if not kernels or any(k <= 0 or k % 2 == 0 for k in kernels):
                raise ValueError('sequence_conv_kernels must be positive odd integers')
            if width % len(kernels):
                raise ValueError('sequence_adapter_dim must be divisible by number of kernels')
            self.convs = nn.ModuleList([
                nn.Conv1d(input_dim, width // len(kernels), kernel_size=k, padding=k // 2) for k in kernels
            ])
        else:
            if width % 2:
                raise ValueError('sequence_adapter_dim must be even for bidirectional GRU')
            self.gru = nn.GRU(input_dim, width // 2, batch_first=True, bidirectional=True)
        self.output = nn.Linear(width, hidden_dim)
        # Constructed after GraphTransformer's legacy initialization. Zero
        # correction keeps shared starting predictions and parameters paired.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, nodes, lengths):
        """AA embeddings [B,L,input_dim], lengths excluding global/padding."""
        lengths = torch.as_tensor(lengths, dtype=torch.long, device='cpu')
        if lengths.numel() != nodes.size(0):
            raise ValueError('sequence lengths and batch size differ')
        if lengths.numel() == 0:
            return nodes.new_zeros((*nodes.shape[:2], self.output.out_features))
        if (lengths <= 0).any() or (lengths > nodes.size(1)).any():
            raise ValueError('sequence lengths must be in [1, padded sequence length]')
        valid = torch.arange(nodes.size(1), device=nodes.device)[None, :] < lengths.to(nodes.device)[:, None]
        # The existing embedding's padding row need not be zero after Xavier.
        x = nodes.masked_fill(~valid.unsqueeze(-1), 0.0)
        if self.kind == 'multiscale':
            channels = x.transpose(1, 2)
            context = torch.cat([torch.nn.functional.silu(conv(channels)) for conv in self.convs], dim=1)
            context = context.transpose(1, 2)
        else:
            # Packing is necessary: a backward GRU over padding would change
            # a peptide's representation when batched with a longer peptide.
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            context, _ = self.gru(packed)
            context, _ = pad_packed_sequence(context, batch_first=True, total_length=nodes.size(1))
        return self.output(context).masked_fill(~valid.unsqueeze(-1), 0.0)
