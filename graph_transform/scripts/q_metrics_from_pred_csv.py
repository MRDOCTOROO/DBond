#!/usr/bin/env python3
"""离线 q 口径补评：从 ludbond 1D 基线的 pred/test.pred.csv（键级长表）计算 q_* 排序指标。

背景：q_*（expected-behavior）口径需要逐谱图行的概率 + (seq,charge,nce) 组键，
ludbond 评估脚本不产 q 指标，但每折保存了键级长表 pred/test.pred.csv
（evaluation_id,threshold,bond_index,true,pred,pred_prob，按测试文件行序展开）。
本脚本把长表重排成 [R, L] 概率矩阵，与 dataset/5fold_soft 同名测试折（行序相同、
多一列 soft_multi=q 真值）对齐，喂给 graph_transform.training.metrics.
BinaryBondMetrics —— 与 DBond-GT 评估完全同一套指标实现，保证跨模型可比。

对齐断言（防错位）：
  1) 长表行数 == Σ(len(seq)-1)；
  2) 按行序切分后，每行 true 串与该测试行 true_multi 逐键一致（硬断言）；
  3) 由概率重建的 f1 与该折 test_metric.csv 参考 f1 交叉校验（Δ≥0.01 硬失败，
     否则打印 Δ 作参考；指标定义差异可能带来 <0.01 的正常偏差）。

用法（pod 项目根目录）：
  .venv/bin/python graph_transform/scripts/q_metrics_from_pred_csv.py \
      --pred_csv result/cv/dbond_s_pre/<ts>/fold_1222/pred/test.pred.csv \
      --test_csv dataset/5fold_soft/1222.test.fbr.multi.csv \
      --ref_metric_csv result/cv/dbond_s_pre/<ts>/fold_1222/metric/test_metric.csv \
      --out_csv result/metric/q_reeval_cond/dbond_s_pre_fold1222.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.metrics import BinaryBondMetrics, batch_group_keys  # noqa: E402


def parse_labels(text: str) -> np.ndarray:
    toks = [t.strip() for t in str(text).strip().split(";") if t.strip() != ""]
    return np.asarray([float(t) for t in toks], dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--test_csv", required=True, help="dataset/5fold_soft 同名测试折（含 soft_multi）")
    ap.add_argument("--ref_metric_csv", default=None, help="该折 metric/test_metric.csv，f1 交叉校验用")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    test = pd.read_csv(args.test_csv)
    pred = pd.read_csv(args.pred_csv)

    n_bonds = np.asarray([max(len(str(s)) - 1, 0) for s in test["seq"]], dtype=np.int64)
    R = len(test)
    probs_all = pred["pred_prob"].to_numpy(dtype=np.float32)
    trues_all = pred["true"].to_numpy(dtype=np.int32)

    # 两种长表格式：
    #   变长（dbond_s，含 bond_index 列）：每谱图行只写 len(seq)-1 个有效键；
    #   定宽（dbond_m/af 等多标签模型，无 bond_index 列）：每行固定 W 个键，含 padding 位。
    if "bond_index" in pred.columns and len(pred) == int(n_bonds.sum()):
        mode = "variable"
        starts = np.concatenate([[0], np.cumsum(n_bonds)])
    elif len(pred) % R == 0 and (len(pred) // R) >= int(n_bonds.max()):
        mode = "fixed"
        W = len(pred) // R
        TB = trues_all.reshape(R, W)
        pad_true = TB[np.arange(W)[None, :] >= n_bonds[:, None]]
        if pad_true.size and int(pad_true.max()) != 0:
            raise ValueError("定宽长表 padding 位 true 非全 0，宽度假设可疑")
        PB = probs_all.reshape(R, W)
        print(f"[格式] 定宽长表 W={W}（含 padding 位，已剔除并断言其 true 全 0）")
    else:
        raise ValueError(
            f"pred 行数 {len(pred)} 与测试行不匹配（Σ有效键={int(n_bonds.sum())}, "
            f"R={R}），行序对齐失败")

    L = int(n_bonds.max())
    P = np.zeros((R, L), dtype=np.float32)
    Y = np.zeros((R, L), dtype=np.int32)
    Q = np.zeros((R, L), dtype=np.float32)
    MASK = np.zeros((R, L), dtype=bool)
    seqs = [str(s) for s in test["seq"]]
    for i in range(R):
        b = int(n_bonds[i])
        if b == 0:
            continue
        if mode == "variable":
            s, e = int(starts[i]), int(starts[i + 1])
            P[i, :b] = probs_all[s:e]
            Y[i, :b] = trues_all[s:e]
        else:
            P[i, :b] = PB[i, :b]
            Y[i, :b] = TB[i, :b]
        Q[i, :b] = parse_labels(test["soft_multi"].iloc[i])
        MASK[i, :b] = True
        tm = parse_labels(test["true_multi"].iloc[i]).astype(np.int32)
        if not np.array_equal(tm, Y[i, :b]):
            raise ValueError(f"行 {i} 长表 true 与 true_multi 不一致，对齐失败")

    # compute() 内部固定按 logits 过 sigmoid（training/metrics.py 的 from_logits
    # 只在 update 侧生效），因此概率先做 logit 逆变换、按默认 logits 路径喂入
    # （与 ensemble_inference.py 的概率平均→logit 逆变换同一做法）。
    eps = 1e-7
    p_clipped = np.clip(P, eps, 1.0 - eps)
    LOGITS = np.log(p_clipped / (1.0 - p_clipped)).astype(np.float32)

    metrics = BinaryBondMetrics({"threshold": 0.5, "threshold_strategy": "fixed"})
    metrics.update(
        LOGITS, Y,
        label_mask=MASK,
        soft_targets=Q,
        sequences=seqs,
        group_keys=batch_group_keys({
            "sequences": seqs,
            "charges": test["charge"].to_numpy(),
            "nces": test["nce"].to_numpy(),
        }),
    )
    out = metrics.compute()

    if args.ref_metric_csv and os.path.exists(args.ref_metric_csv):
        ref = pd.read_csv(args.ref_metric_csv)
        # 口径基准：lab_f1_mi = 有效键 micro F1（docs §10 跨模型对齐口径），应与重建
        # f1_micro 精确一致；顶层 "f1" 是各 1D 模型自有测试行集口径（如 dbond_s 的
        # 0.7812 vs 有效键 0.7713），仅作参考打印，不作为对齐判据。
        row = ref.loc[ref["metric"] == "lab_f1_mi", "value"]
        if len(row):
            delta = abs(float(row.iloc[0]) - float(out.get("f1_micro", float("nan"))))
            tag = "OK" if delta < 1e-6 else ("WARN" if delta < 0.01 else "FAIL")
            print(f"[f1 校验:{tag}] 重建 f1_micro={out.get('f1_micro'):.6f} vs 参考 lab_f1_mi={float(row.iloc[0]):.6f} Δ={delta:.2e}")
            if delta >= 0.01:
                raise ValueError("f1 偏差 ≥0.01，概率对齐可疑，中止")
        row = ref.loc[ref["metric"] == "f1", "value"]
        if len(row):
            print(f"[参考] 该模型自有口径顶层 f1={float(row.iloc[0]):.6f}（行集口径不同，不比对）")

    keys = [
        "f1_micro", "q_brier", "q_mae", "q_spearman",
        "q_spearman_pep", "q_spearman_pep_cond", "q_spearman_pep_seq",
        "q_top10_enrichment", "q_top10_enrichment_cond", "q_top10_enrichment_seq",
    ]
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    rows = [{"metric": k, "value": float(out.get(k, float("nan")))} for k in keys]
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    for r in rows:
        print(f"{r['metric']:26s} {r['value']:.4f}")
    print(f"saved -> {args.out_csv}")


if __name__ == "__main__":
    main()
