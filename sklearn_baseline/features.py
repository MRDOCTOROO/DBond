"""键级特征构建：dataset/5fold 的 multi CSV → 传统模型可用的逐键特征矩阵。

特征口径严格对齐最新 film 臂（pre-synthesis 设定，
graph_transform/config/pre_synthesis_fold1222_theory_film.yaml）：
  - 15 维逐键理论特征：graph_transform.data.theory_features.compute_bond_theory
    （纯序列计算，与 use_theory_features: true 完全同源）
  - 条件特征 charge / pep_mass / nce（state mask [T,T,F] + env mask [T,F] 的保留项）
  - intensity / scan_num / rt 属合成后信息，pre-synthesis 口径下排除

标签 = true_multi 展开的逐键 0/1；每键一行（与 ludbond dbond_s 行口径一致，
训练集一折约 38 万谱图行 → 约 1100 万键行级样本由 410 条唯一序列重复贡献）。

树模型对单调尺度不敏感，连续列不做归一化（LR 侧在 Pipeline 里加 StandardScaler）。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph_transform.data.theory_features import BOND_THEORY_DIM, compute_bond_theory  # noqa: E402

# 理论特征维名（与 theory_features.py docstring 逐维对应）
THEORY_NAMES = [
    "prefix_mass", "suffix_mass", "b1_mz", "b2_mz", "y1_mz", "y2_mz",
    "rel_pos", "mass_left", "mass_right",
    "h2o_loss_left", "h2o_loss_right", "nh3_loss_left", "nh3_loss_right",
    "pro_nterm", "pro_cterm",
]
# film 臂 pre-synthesis 保留的条件列（顺序即特征矩阵列序）
PARITY_COND_COLS = ["charge", "pep_mass", "nce"]

FEATURE_NAMES = THEORY_NAMES + PARITY_COND_COLS
assert len(FEATURE_NAMES) == BOND_THEORY_DIM + len(PARITY_COND_COLS)


def parse_multi(text: str) -> np.ndarray:
    """解析 true_multi/soft_multi 的 ';'-分隔 0/1 串（与 precompute_soft_labels.parse_labels 同义）。"""
    toks = [t.strip() for t in str(text).strip().split(";") if t.strip() != ""]
    return np.asarray([float(t) for t in toks], dtype=np.float64)


@dataclass
class BondFrame:
    """一个折 CSV 展开后的键级数据。"""

    frame: pd.DataFrame        # 原始谱图行（含 seq/charge/nce/true_multi，行序即 CSV 行序）
    X: np.ndarray              # [N_bonds, len(FEATURE_NAMES)] float32
    y: np.ndarray              # [N_bonds] int8 逐键 0/1
    row_of_bond: np.ndarray    # [N_bonds] int64 每键所属谱图行号（CSV 行序 0-based）
    bond_index: np.ndarray     # [N_bonds] int32 键位 0-based（键 j 连接残基 j 与 j+1）

    @property
    def n_bonds_per_row(self) -> np.ndarray:
        return np.asarray([max(len(str(s)) - 1, 0) for s in self.frame["seq"]], dtype=np.int64)


def build_bond_frame(
    csv_path: str | Path,
    feature_set: str = "parity",
    theory_cache: Optional[Dict[str, np.ndarray]] = None,
    limit_rows: Optional[int] = None,
) -> BondFrame:
    """把 multi CSV 展开成键级特征矩阵。

    feature_set: 目前仅 'parity'（15 理论维 + charge/pep_mass/nce，对齐 film 臂）；
      预留 'extended'（两侧氨基酸 one-hot 等）供后续消融。
    theory_cache: 跨 train/test 共享的 {seq: [L-1,15] ndarray} 缓存（513 序列级，
      传入 dict 即原地更新）。
    limit_rows: 调试用，只读前 N 行。
    """
    if feature_set != "parity":
        raise NotImplementedError(f"feature_set={feature_set} 尚未实现（本轮只跑 parity 口径）")

    frame = pd.read_csv(csv_path)
    if limit_rows is not None:
        frame = frame.iloc[:limit_rows].reset_index(drop=True)
    if theory_cache is None:
        theory_cache = {}

    seqs = [str(s) for s in frame["seq"]]
    charges = frame["charge"].to_numpy(dtype=np.float32)
    pep_mass = frame["pep_mass"].to_numpy(dtype=np.float32)
    nces = frame["nce"].to_numpy(dtype=np.float32)

    # 逐行校验 true_multi 键数 == len(seq)-1（与 dataset/make_dbond_s_folds.py 同款硬校验）
    labels_per_row = [parse_labels for parse_labels in (parse_multi(t) for t in frame["true_multi"])]
    n_bonds = np.asarray([len(s) - 1 for s in seqs], dtype=np.int64)
    for i, (lab, nb, s) in enumerate(zip(labels_per_row, n_bonds, seqs)):
        if len(lab) != nb:
            raise ValueError(f"行 {i} true_multi 键数 {len(lab)} != len(seq)-1 {nb} (seq={s})")

    total = int(n_bonds.sum())
    n_feat = len(FEATURE_NAMES)
    X = np.empty((total, n_feat), dtype=np.float32)
    y = np.empty(total, dtype=np.int8)
    row_of_bond = np.empty(total, dtype=np.int64)
    bond_index = np.empty(total, dtype=np.int32)

    pos = 0
    for i, (seq, lab) in enumerate(zip(seqs, labels_per_row)):
        nb = n_bonds[i]
        if nb == 0:
            continue
        theory = theory_cache.get(seq)
        if theory is None:
            theory = compute_bond_theory(seq).numpy()  # [L-1, 15]，未知残基 fail-fast
            theory_cache[seq] = theory
        X[pos:pos + nb, :BOND_THEORY_DIM] = theory
        X[pos:pos + nb, BOND_THEORY_DIM:] = (charges[i], pep_mass[i], nces[i])
        y[pos:pos + nb] = lab.astype(np.int8)
        row_of_bond[pos:pos + nb] = i
        bond_index[pos:pos + nb] = np.arange(nb, dtype=np.int32)
        pos += nb

    return BondFrame(frame=frame, X=X, y=y, row_of_bond=row_of_bond, bond_index=bond_index)
