"""
序列衍生理论键离子特征（sequence-derived theoretical bond-ion features）。

对每根相邻肽键 i（res_i 与 res_{i+1} 之间，i = 1..L-1），完全从序列计算
合成前可得的理论碎片描述，供模型作为输入先验。不读取 MGF 峰、不读取
碎片强度——与 pre-synthesis 场景自洽，候选评分时同样可得。

特征（每键 15 维，见 BOND_THEORY_DIM）：
    0  prefix_mass   前缀中性质量 Σres(1..i)          /2000
    1  suffix_mass   后缀中性质量 Σres(i+1..L)        /2000
    2  b1_mz         (prefix + H+) / 1               /2000
    3  b2_mz         (prefix + 2H+) / 2               /2000
    4  y1_mz         (suffix + H2O + H+) / 1          /2000
    5  y2_mz         (suffix + H2O + 2H+) / 2         /2000
    6  rel_pos       i / (L-1)                        (L=1 时取 0)
    7  mass_left     res_i 单同位素残基质量            /200
    8  mass_right    res_{i+1} 单同位素残基质量        /200
    9  h2o_loss_left  res_i ∈ {S,T,E,D}（易失水侧链）  0/1
   10  h2o_loss_right 同上对 res_{i+1}
   11  nh3_loss_left  res_i ∈ {N,Q,K,R}（易失氨侧链）  0/1
   12  nh3_loss_right 同上对 res_{i+1}
   13  pro_nterm     res_{i+1} == P（Xxx-Pro 特异断裂） 0/1
   14  pro_cterm     res_i == P（Pro-Xxx）             0/1

质量表与 PBCLA/utils.py 及 pyteomics 单同位素残基质量保持一致
（20 标准 + B/O/X/Z 特殊 D-残基），确保与 score_presynthesis 的
理论 m/z 同源。归一化惯例与模型一致（质量 /2000 对齐 pep_mass）。
"""

from __future__ import annotations

from typing import List

import torch

BOND_THEORY_DIM = 15

# 单同位素残基质量（残基 = 氨基酸 - H2O）。20 标准值与 pyteomics
# mass.std_aa_mass 一致；B/O/X/Z 取 PBCLA/utils.py 的值。
RESIDUE_MONO_MASS = {
    'A': 71.03711, 'R': 156.10111, 'N': 114.04293, 'D': 115.02694,
    'C': 103.00919, 'E': 129.04259, 'Q': 128.05858, 'G': 57.02146,
    'H': 137.05891, 'I': 113.08406, 'L': 113.08406, 'K': 128.09496,
    'M': 131.04049, 'F': 147.06841, 'P': 97.05276, 'S': 87.03203,
    'T': 101.04768, 'W': 186.07931, 'Y': 163.06333, 'V': 99.06841,
    # 特殊 D-残基（论文确认：B=D-Dap, O=D-Orn, X=3-(3-Pyridyl)-D-Ala, Z=D-Cha）
    'B': 86.04801, 'O': 114.07931, 'X': 148.06366, 'Z': 153.11536,
}

_PROTON = 1.00727646688   # H+ 质量
_H2O = 18.0105646863      # 水分子质量
_MASS_NORM = 2000.0       # 与 pep_mass/2000 同惯例
_RES_MASS_NORM = 200.0    # 残基质量归一（最大 W=186.08 → ~0.93）

_H2O_LOSS_RESIDUES = set('STED')   # 羟基/羧基侧链易失水
_NH3_LOSS_RESIDUES = set('NQKR')   # 酰胺/氨基侧链易失氨


def bond_theory_dim() -> int:
    return BOND_THEORY_DIM


def compute_bond_theory(sequence: str) -> torch.Tensor:
    """计算一条序列的逐键理论特征，返回 [L-1, 15] float32 张量。

    L<2（无键）时返回 [0, 15] 空张量。未知残基抛 ValueError（fail-fast，
    与序列编码路径的非法字符策略一致）。
    """
    seq = sequence.strip().upper()
    L = len(seq)
    n_bonds = max(L - 1, 0)
    if n_bonds == 0:
        return torch.zeros((0, BOND_THEORY_DIM), dtype=torch.float32)

    masses: List[float] = []
    for ch in seq:
        m = RESIDUE_MONO_MASS.get(ch)
        if m is None:
            raise ValueError(f"Unknown amino acid: {ch} (sequence={sequence})")
        masses.append(m)

    prefix = [0.0] * (L + 1)
    for i in range(L):
        prefix[i + 1] = prefix[i] + masses[i]
    total = prefix[L]

    rows = torch.zeros((n_bonds, BOND_THEORY_DIM), dtype=torch.float32)
    for i in range(1, L):  # 键 i：res_i 与 res_{i+1} 之间
        pre = prefix[i]              # 前缀中性质量
        suf = total - pre            # 后缀中性质量
        r = rows[i - 1]
        r[0] = pre / _MASS_NORM
        r[1] = suf / _MASS_NORM
        r[2] = (pre + _PROTON) / _MASS_NORM                       # b1+
        r[3] = (pre + 2.0 * _PROTON) / 2.0 / _MASS_NORM           # b2+
        r[4] = (suf + _H2O + _PROTON) / _MASS_NORM                # y1+
        r[5] = (suf + _H2O + 2.0 * _PROTON) / 2.0 / _MASS_NORM    # y2+
        r[6] = i / n_bonds                                        # 相对位置
        r[7] = masses[i - 1] / _RES_MASS_NORM
        r[8] = masses[i] / _RES_MASS_NORM
        r[9] = 1.0 if seq[i - 1] in _H2O_LOSS_RESIDUES else 0.0
        r[10] = 1.0 if seq[i] in _H2O_LOSS_RESIDUES else 0.0
        r[11] = 1.0 if seq[i - 1] in _NH3_LOSS_RESIDUES else 0.0
        r[12] = 1.0 if seq[i] in _NH3_LOSS_RESIDUES else 0.0
        r[13] = 1.0 if seq[i] == 'P' else 0.0
        r[14] = 1.0 if seq[i - 1] == 'P' else 0.0
    return rows
