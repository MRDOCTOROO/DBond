"""
序列衍生理论键离子特征（torch 向量化版，模型 forward 内直接计算）。

对每根相邻肽键 j（res_j 与 res_{j+1} 之间，j = 0..L-2，与 dbond_s 的
bond_pos 0-based 语义一致），完全从序列计算理论碎片描述。不读取 MGF 峰、
不读取碎片强度——合成前可计算的先验。

特征（每键 15 维，与 graph_transform/data/theory_features.py 同公式同序）：
    0  prefix_mass   Σres(0..j)                /2000
    1  suffix_mass   Σres(j+1..L-1)            /2000
    2  b1_mz         (prefix + H+)             /2000
    3  b2_mz         (prefix + 2H+)/2          /2000
    4  y1_mz         (suffix + H2O + H+)       /2000
    5  y2_mz         (suffix + H2O + 2H+)/2    /2000
    6  rel_pos       (j+1)/(L-1)
    7  mass_left     res_j                     /200
    8  mass_right    res_{j+1}                 /200
    9  h2o_loss_left  res_j ∈ {S,T,E,D}        0/1
   10  h2o_loss_right 同上 res_{j+1}
   11  nh3_loss_left  res_j ∈ {N,Q,K,R}        0/1
   12  nh3_loss_right 同上 res_{j+1}
   13  pro_nterm     res_{j+1} == P（Xxx-Pro）  0/1
   14  pro_cterm     res_j == P                0/1

质量表与 PBCLA/utils.py / pyteomics 单同位素残基质量同源（含 B/O/X/Z）。
pad 键位置的特征置 0（下游输出侧已有 masked_fill / label_mask 保护）。
"""

from __future__ import annotations

import torch

PROTON = 1.00727646688
H2O = 18.0105646863
_MASS_NORM = 2000.0
_RES_MASS_NORM = 200.0
BOND_THEORY_DIM = 15

RESIDUE_MONO_MASS = {
    'A': 71.03711, 'R': 156.10111, 'N': 114.04293, 'D': 115.02694,
    'C': 103.00919, 'E': 129.04259, 'Q': 128.05858, 'G': 57.02146,
    'H': 137.05891, 'I': 113.08406, 'L': 113.08406, 'K': 128.09496,
    'M': 131.04049, 'F': 147.06841, 'P': 97.05276, 'S': 87.03203,
    'T': 101.04768, 'W': 186.07931, 'Y': 163.06333, 'V': 99.06841,
    # 特殊 D-残基（论文确认：B=D-Dap, O=D-Orn, X=3-(3-Pyridyl)-D-Ala, Z=D-Cha）
    'B': 86.04801, 'O': 114.07931, 'X': 148.06366, 'Z': 153.11536,
}


class BondTheoryEncoder(torch.nn.Module):
    """从 seq_index_batch 查表向量化计算逐键理论特征并投影。

    alphabet 传模型的 token 顺序（索引 i → token，pad 字符在 alphabet 内），
    pad 索引的查表值置 0。
    """

    def __init__(self, alphabet: str, pad_char: str, out_dim: int):
        super().__init__()
        vocab = len(alphabet) + 1
        mass = torch.zeros(vocab)
        h2o = torch.zeros(vocab)
        nh3 = torch.zeros(vocab)
        pro = torch.zeros(vocab)
        for i, ch in enumerate(alphabet):
            if ch == pad_char:
                continue
            m = RESIDUE_MONO_MASS.get(ch)
            if m is None:
                raise ValueError(f"Unknown residue {ch!r} in alphabet {alphabet!r}")
            mass[i + 1] = m
            h2o[i + 1] = 1.0 if ch in 'STED' else 0.0
            nh3[i + 1] = 1.0 if ch in 'NQKR' else 0.0
            pro[i + 1] = 1.0 if ch == 'P' else 0.0
        # 1-D 查找表 [vocab]：mass_lut[seq_index] → [B,L]，索引 < vocab 安全
        self.register_buffer('mass_lut', mass)
        self.register_buffer('h2o_lut', h2o)
        self.register_buffer('nh3_lut', nh3)
        self.register_buffer('pro_lut', pro)
        self.proj = torch.nn.Linear(BOND_THEORY_DIM, out_dim)

    def raw_features(self, seq_index_batch: torch.Tensor, seq_padding_mask_batch: torch.Tensor) -> torch.Tensor:
        """[B,L] token 索引（0=pad）→ [B, L-1, 15] 逐键理论特征。"""
        masses = self.mass_lut[seq_index_batch]            # [B,L]
        prefix = torch.cumsum(masses, dim=1)               # prefix[:, i] = Σmass(0..i)
        total = prefix[:, -1:]
        pre = prefix[:, :-1]                               # 键 j：Σres(0..j)
        suf = total - pre
        lengths = (seq_index_batch > 0).sum(dim=1)         # pad 索引恒为 0
        j = torch.arange(seq_index_batch.size(1) - 1,
                         device=seq_index_batch.device, dtype=prefix.dtype)
        rel_pos = (j + 1.0) / (lengths.to(prefix.dtype) - 1.0).clamp_min(1.0).unsqueeze(1)
        left_idx, right_idx = seq_index_batch[:, :-1], seq_index_batch[:, 1:]
        feat = torch.stack([
            pre / _MASS_NORM,
            suf / _MASS_NORM,
            (pre + PROTON) / _MASS_NORM,
            (pre + 2.0 * PROTON) / 2.0 / _MASS_NORM,
            (suf + H2O + PROTON) / _MASS_NORM,
            (suf + H2O + 2.0 * PROTON) / 2.0 / _MASS_NORM,
            rel_pos,
            self.mass_lut[left_idx] / _RES_MASS_NORM,
            self.mass_lut[right_idx] / _RES_MASS_NORM,
            self.h2o_lut[left_idx], self.h2o_lut[right_idx],
            self.nh3_lut[left_idx], self.nh3_lut[right_idx],
            self.pro_lut[right_idx], self.pro_lut[left_idx],
        ], dim=-1)
        # 无效键位置（j+1 >= 有效长度）置 0
        bond_valid = (j.unsqueeze(0) < (lengths - 1).clamp_min(0).unsqueeze(1))
        return feat * bond_valid.to(feat.dtype).unsqueeze(-1)

    def forward(self, seq_index_batch: torch.Tensor, seq_padding_mask_batch: torch.Tensor) -> torch.Tensor:
        """[B,L] → 投影后 [B, L-1, out_dim]。"""
        return self.proj(self.raw_features(seq_index_batch, seq_padding_mask_batch))
