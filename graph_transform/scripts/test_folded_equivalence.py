#!/usr/bin/env python3
"""条件组折叠训练路径的等价性/正确性单测（纯 CPU，无需数据集）。

验证三件事（对应 20260906 审计的等价性定理与折叠实现）：
  1. 线性恒等式在实现层面成立：行级重复 hard BCE == 折叠 + spectrum 权重 BCE
     （sum_j BCE(z, y_j) = n_g * BCE(z, q_g)），即 sanity 臂期望上复现 hard 训练；
  2. BinaryBondLoss 加权路径与手工加权计算一致；
  3. metrics 条件级/序列级 q 指标：同组重复行去重、序列内条件均匀聚合正确。

用法：
    python graph_transform/scripts/test_folded_equivalence.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from graph_transform.training.loss_functions import BinaryBondLoss  # noqa: E402
from graph_transform.training.metrics import BinaryBondMetrics  # noqa: E402


def test_weighted_bce_identity() -> None:
    """(1) 行级 hard 重复 == 折叠 q + spectrum 权重（实现级恒等式）。"""
    torch.manual_seed(0)
    loss_cfg = {"main_loss": "binary_cross_entropy",
                "handle_imbalance": False, "use_auxiliary_losses": False}
    criterion = BinaryBondLoss(loss_cfg)

    n_spec, n_bonds = 5, 3
    z = torch.randn(n_bonds)                       # 同组输入相同 → 同一 logits
    y = torch.randint(0, 2, (n_spec, n_bonds)).float()
    q = y.mean(dim=0)

    # 行级：5 行重复 z，逐元素 BCE 取均值
    row_logits = z.unsqueeze(0).repeat(n_spec, 1).reshape(-1)
    row_targets = y.reshape(-1)
    loss_row = criterion(row_logits, row_targets)

    # 折叠：1 行 z，目标 q，spectrum 权重逐键展开（行权重 n_g 重复到该行 3 个键）
    loss_folded = criterion(z, q, weights=torch.full((n_bonds,), float(n_spec)))

    assert abs(loss_row.item() - loss_folded.item()) < 1e-6, (
        f"等价性破坏: row={loss_row.item():.8f} folded={loss_folded.item():.8f}")
    print(f"[1] 加权恒等式 OK: row={loss_row.item():.8f} folded={loss_folded.item():.8f}")


def test_weighted_mean_semantics() -> None:
    """(2) 加权路径 == 手工 (elem*w).sum()/w.sum()，且权重和不影响归一。"""
    loss_cfg = {"main_loss": "binary_cross_entropy",
                "handle_imbalance": False, "use_auxiliary_losses": False}
    criterion = BinaryBondLoss(loss_cfg)
    logits = torch.randn(7)
    targets = torch.rand(7)
    w = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 5.0, 5.0])
    got = criterion(logits, targets, weights=w)
    elem = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    want = (elem * w).sum() / w.sum()
    assert torch.allclose(got, want, atol=1e-7)
    # group_uniform 两行同 logits 同目标时权重尺度不变（归一化验证）
    a = criterion(logits, targets, weights=torch.ones(7))
    b = criterion(logits, targets, weights=torch.full((7,), 3.0))
    assert torch.allclose(a, b, atol=1e-7), "权重整体缩放不应改变加权均值"
    print(f"[2] 加权语义 OK: {got.item():.8f}")


def _fill_metrics() -> BinaryBondMetrics:
    """构造 2 序列 / 3 条件组 / 8 谱图行的合成累积状态。

    组内行完全相同（eval 确定性下的真实情形）；q: A1=0.2, A2=0.6, B1=0.9。
    预测 logits 单调对应（sigmoid 均值排序 A1 < A2 < B1）。
    """
    metrics = BinaryBondMetrics({"threshold": 0.5, "threshold_strategy": "fixed"})
    logits_by_group = {"A1": -2.0, "A2": 0.5, "B1": 3.0}
    q_by_group = {"A1": 0.2, "A2": 0.6, "B1": 0.9}
    # 条件键必须真实可分：A1/A2 是同序列不同 nce（同 charge+nce 会并成一组）
    key_by_group = {"A1": "SEQAA|2|30", "A2": "SEQAA|2|35", "B1": "SEQBB|2|30"}
    seq_by_group = {"A1": "SEQAA", "A2": "SEQAA", "B1": "SEQBB"}
    # A1×3 行, A2×3 行（序列 A 各 3 张谱图）, B1×2 行（序列 B 2 张）
    rows = ["A1"] * 3 + ["A2"] * 3 + ["B1"] * 2
    for g in rows:
        logits = np.full(4, logits_by_group[g], dtype=np.float32)
        q = np.full(4, q_by_group[g], dtype=np.float32)
        metrics.sample_predictions.append(logits)
        metrics.sample_targets.append(np.zeros(4, dtype=np.int32))
        metrics.sample_soft_targets.append(q)
        metrics.all_valid_predictions.append(logits)
        metrics.all_valid_targets.append(np.zeros(4, dtype=np.int32))
        metrics.all_valid_soft.append(q)
        metrics.row_group_keys.append(key_by_group[g])
        metrics.row_seq_keys.append(seq_by_group[g])
    return metrics


def test_group_level_metrics() -> None:
    """(3) 条件级去重与序列级聚合。"""
    metrics = _fill_metrics()
    out = metrics._compute_group_level_q_metrics()

    # 条件级：3 个组，预测与 q 完全同序 → spearman=1
    assert abs(out["q_spearman_pep_cond"] - 1.0) < 1e-6, out
    # 条件级 top10%（n=3 → 1 个）= 最高 q 组 0.9 / 总均值 (0.2+0.6+0.9)/3
    expect_top10 = 0.9 / ((0.2 + 0.6 + 0.9) / 3)
    assert abs(out["q_top10_enrichment_cond"] - expect_top10) < 1e-6, out
    # 序列级：A=(0.2+0.6)/2=0.4 < B=0.9，2 个序列预测同序 → spearman=1
    assert abs(out["q_spearman_pep_seq"] - 1.0) < 1e-6, out
    # 序列级 top10%（n=2 → 1 个）= 0.9 / ((0.4+0.9)/2)
    expect_top10_seq = 0.9 / ((0.4 + 0.9) / 2)
    assert abs(out["q_top10_enrichment_seq"] - expect_top10_seq) < 1e-6, out

    # 行级（旧口径）仍在：8 行全部参与（flat 概率 = 8 行 × 4 键 sigmoid 展平）
    flat_logits = np.concatenate([np.full(4, lg, dtype=np.float32)
                                  for lg in (-2.0, -2.0, -2.0, 0.5, 0.5, 0.5, 3.0, 3.0)])
    row_level = metrics._compute_expected_behavior_metrics(1.0 / (1.0 + np.exp(-flat_logits)))
    assert "q_spearman_pep" in row_level
    # 行级 enrichment 应被 B 组 2 行 / A 组 6 行的采集频率扭曲：
    # top10%（8 行 → 1 行）= 0.9 / 行均值(3*0.2+3*0.6+2*0.9)/8
    row_mean = (3 * 0.2 + 3 * 0.6 + 2 * 0.9) / 8
    assert abs(row_level["q_top10_enrichment"] - 0.9 / row_mean) < 1e-6
    print(f"[3] 条件级/序列级指标 OK: cond_top10={out['q_top10_enrichment_cond']:.4f} "
          f"seq_top10={out['q_top10_enrichment_seq']:.4f} (行级={0.9 / row_mean:.4f})")


def test_sequence_ranking_loss() -> None:
    """(4) 序列级 pairwise ranking loss：违序>0、顺序=0、同序列行聚合。"""
    from types import SimpleNamespace
    from graph_transform.training.trainer import Trainer

    dummy = SimpleNamespace(config={"loss": {"ranking_margin": 0.0}})
    B, L = 4, 4
    mask = torch.zeros(B, L)
    mask[:, :3] = 1.0                      # 每行 4 键只有前 3 个有效
    seqs = ["A", "A", "B", "B"]            # 每序列 2 行（条件组），聚合到序列级
    soft = torch.full((B, L), 0.2)
    soft[2:] = 0.8                         # A 目标低、B 目标高
    batch = {"label_mask": mask, "soft_labels": soft, "labels": soft.round(),
             "sequences": seqs}

    # A 预测高、B 预测低 → 与目标方向相反，hinge 必须给正损失
    logits_bad = torch.zeros(B, L)
    logits_bad[:2, :3] = 4.0
    logits_bad[2:, :3] = -4.0
    l_bad = Trainer._sequence_ranking_loss(dummy, logits_bad, batch)
    assert l_bad.item() > 0.5, l_bad.item()

    # 反向 → 已满足 margin=0，损失恰为 0
    l_good = Trainer._sequence_ranking_loss(dummy, -logits_bad, batch)
    assert l_good.item() == 0.0, l_good.item()

    # 单序列批 → 无可排序对，返回 0
    batch_one = dict(batch, sequences=["A"] * B)
    assert Trainer._sequence_ranking_loss(dummy, logits_bad, batch_one).item() == 0.0
    print(f"[4] ranking loss OK: 违序={l_bad.item():.4f} 顺序={l_good.item():.4f} 单序列=0")


def main() -> None:
    test_weighted_bce_identity()
    test_weighted_mean_semantics()
    test_group_level_metrics()
    test_sequence_ranking_loss()
    print("\n全部等价性/正确性测试通过 ✅")


if __name__ == "__main__":
    main()
