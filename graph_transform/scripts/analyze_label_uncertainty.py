#!/usr/bin/env python3
"""序列内标签不确定性分析（PBCLA multi 标签的噪声瓶颈检测）。

问题背景：pre 模型（屏蔽 intensity/scan_num/rt）对同一序列的多张谱图只能给出
同一个预测，而标签来自单次实测。若同一序列在不同谱图间标签本身不稳定
（within-sequence label variance 高），则任何序列条件模型（GAT/Transformer/
Graphormer）都存在由标签随机性决定的性能上限，架构改进无法突破。

对每个唯一序列 s 的每根键 i 计算：
    q_{s,i} = 该序列第 i 根键在所有谱图中的平均断裂率

输出（全部 stdout，--save_q 可导出逐 (seq, bond) 的 q 表）：
    1. 谱图/序列基础统计（每序列谱图数分布）
    2. within-sequence label variance（q(1-q)）与模糊键占比
    3. per-bond entropy H(q)
    4. 方差分解与 sequence-level ICC（between / total）
    5. 同序列谱图间 label disagreement（总体 + 按条件 charge/nce 分层）
    6. leave-one-out oracle 上限：用同序列其他谱图的 q 预测当前谱图标签，
       得到序列条件模型的实证天花板（acc/F1/AUC），可与当前 pre 模型对比。

用法（pod，项目根目录）：
    .venv/bin/python graph_transform/scripts/analyze_label_uncertainty.py \
        --inputs dataset/5fold/1222.train.fbr.shuffle.multi.csv \
                  dataset/5fold/1222.test.fbr.multi.csv
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np
import pandas as pd


def parse_labels(true_multi: str) -> np.ndarray:
    toks = [t.strip() for t in str(true_multi).strip().split(";") if t.strip() != ""]
    return np.asarray([float(t) for t in toks], dtype=np.float64)


def load_rows(paths):
    frames = [pd.read_csv(p, usecols=["seq", "charge", "nce", "true_multi"]) for p in paths]
    frame = pd.concat(frames, ignore_index=True)
    frame["label_vec"] = frame["true_multi"].map(parse_labels)
    bad = sum(
        1 for lv, seq in zip(frame["label_vec"], frame["seq"])
        if lv.size != max(len(str(seq)) - 1, 0)
    )
    if bad:
        raise ValueError(f"{bad} rows with |true_multi| != len(seq)-1")
    return frame


def spectra_groups(frame: pd.DataFrame):
    """按序列聚合：返回 seq -> list of (label_vec, charge, nce)。"""
    groups = defaultdict(list)
    for seq, sub in frame.groupby("seq", sort=False):
        for lv, ch, nce in zip(sub["label_vec"], sub["charge"], sub["nce"]):
            groups[str(seq)].append((lv, float(ch), float(nce)))
    return groups


def pairwise_disagreement(label_vecs):
    """同序列谱图两两间：键级 disagreement 率与任意差异标志。"""
    n = len(label_vecs)
    if n < 2:
        return None
    rates = []
    differ_any = False
    for a in range(n):
        for b in range(a + 1, n):
            va, vb = label_vecs[a], label_vecs[b]
            if va.size != vb.size:
                continue
            rate = float(np.mean(va != vb))
            rates.append(rate)
            if rate > 0:
                differ_any = True
    if not rates:
        return None
    return float(np.mean(rates)), differ_any


def prf_micro(y_true: np.ndarray, p_hat: np.ndarray, thr: float):
    pred = (p_hat >= thr).astype(np.float64)
    tp = float(np.sum((pred == 1) & (y_true == 1)))
    fp = float(np.sum((pred == 1) & (y_true == 0)))
    fn = float(np.sum((pred == 0) & (y_true == 1)))
    acc = float(np.mean(pred == y_true))
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return acc, prec, rec, f1


def rank_auc(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    order = np.argsort(p_hat, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_true) + 1)
    # 并列取平均秩
    ser = pd.Series(p_hat)
    ranks = ser.rank(method="average").to_numpy()
    pos = float(np.sum(y_true == 1))
    neg = float(np.sum(y_true == 0))
    if pos == 0 or neg == 0:
        return float("nan")
    return float((ranks[y_true == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="multi 格式 CSV（train+test 全量）")
    ap.add_argument("--save_q", default=None, help="导出 (seq, bond, q, n_spectra) 表到该 CSV")
    args = ap.parse_args()

    frame = load_rows(args.inputs)
    groups = spectra_groups(frame)
    n_rows, n_seq = len(frame), len(groups)
    bond_counts = [max(len(s) - 1, 0) for s in groups]
    total_bonds = sum(bond_counts)
    print(f"rows={n_rows}  unique_seq={n_seq}  unique_bonds={total_bonds}")

    # 1) 每序列谱图数分布
    ns = np.array([len(v) for v in groups.values()], dtype=np.float64)
    qs = np.percentile(ns, [0, 25, 50, 75, 100])
    print(f"spectra/seq: min={qs[0]:.0f} p25={qs[1]:.1f} median={qs[2]:.1f} p75={qs[3]:.1f} max={qs[4]:.0f}")
    print(f"  n_s=1: {int((ns == 1).sum())} seqs ({(ns == 1).mean() * 100:.1f}%); "
          f"n_s>=2: {int((ns >= 2).sum())} ({(ns >= 2).mean() * 100:.1f}%); "
          f"n_s>=5: {int((ns >= 5).sum())} ({(ns >= 5).mean() * 100:.1f}%)")

    # 2) q_{s,i}、within variance、模糊键占比、3) entropy
    q_rows = []  # (seq, bond_idx, q, n)
    per_seq_mean = []
    within_vars = []
    entropies = []
    for seq, obs in groups.items():
        n_bond = obs[0][0].size
        if n_bond == 0:
            continue
        mat = np.stack([o[0] for o in obs])  # [n_s, n_bond]
        q = mat.mean(axis=0)
        per_seq_mean.append(float(q.mean()))
        for i in range(n_bond):
            q_rows.append((seq, i, float(q[i]), len(obs)))
        wv = q * (1.0 - q)
        within_vars.append(wv)
        ent = -np.where((q > 0) & (q < 1),
                        q * np.log2(np.clip(q, 1e-12, 1)) + (1 - q) * np.log2(np.clip(1 - q, 1e-12, 1)),
                        0.0)
        entropies.append(ent)
    within_vars = np.concatenate(within_vars)
    entropies = np.concatenate(entropies)
    q_all = np.array([r[2] for r in q_rows])

    print(f"\n[within-sequence label variance]")
    print(f"  mean q(1-q) = {within_vars.mean():.4f}  (理论最大 0.25)")
    print(f"  ambiguous bonds (0.1<q<0.9): {(np.abs(q_all - 0.5) < 0.4).mean() * 100:.1f}%")
    print(f"  strongly ambiguous (0.25<q<0.75): {(np.abs(q_all - 0.5) < 0.25).mean() * 100:.1f}%")
    print(f"  deterministic bonds (q==0 or q==1): {np.mean((q_all == 0) | (q_all == 1)) * 100:.1f}% "
          f"(其中 n_s=1 的独占 {np.mean([r[3] == 1 for r in q_rows]) * 100:.1f}%)")
    print(f"[per-bond entropy]  mean H(q) = {entropies.mean():.4f} bits (max 1.0); "
          f"H>0.5: {(entropies > 0.5).mean() * 100:.1f}%")

    # 4) 方差分解 + ICC（one-way random effect：标签按序列分组）
    within_mean = float(within_vars.mean())
    between = float(np.var(per_seq_mean, ddof=1)) if len(per_seq_mean) > 1 else 0.0
    total = between + within_mean
    print(f"\n[variance decomposition]")
    print(f"  between-seq var = {between:.4f}  within-seq var = {within_mean:.4f}  total = {total:.4f}")
    print(f"  ICC(seq) = between/total = {between / total if total > 0 else float('nan'):.4f}")
    grand = float(np.mean([np.mean(o[0]) for obs in groups.values() for o in obs]))
    print(f"  grand cleavage rate = {grand:.4f}")

    # 5) 谱图间 disagreement（总体 + 条件分层）
    any_diff = 0
    multi = 0
    rates_all, rates_same_cond, rates_diff_cond = [], [], []
    for seq, obs in groups.items():
        if len(obs) < 2:
            continue
        multi += 1
        lvs = [o[0] for o in obs]
        dis = pairwise_disagreement(lvs)
        if dis is None:
            continue
        rate, differ = dis
        rates_all.append(rate)
        if differ:
            any_diff += 1
        for a in range(len(obs)):
            for b in range(a + 1, len(obs)):
                va, vb = lvs[a], lvs[b]
                if va.size != vb.size:
                    continue
                r = float(np.mean(va != vb))
                same_cond = (obs[a][1] == obs[b][1]) and (obs[a][2] == obs[b][2])
                (rates_same_cond if same_cond else rates_diff_cond).append(r)
    print(f"\n[label disagreement across spectra of same sequence]  (n_seq>=2: {multi})")
    if multi:
        print(f"  sequences with ANY differing labels: {any_diff}/{multi} ({any_diff / multi * 100:.1f}%)")
        print(f"  mean pairwise bond disagreement rate: {np.mean(rates_all) * 100:.2f}%")
        if rates_same_cond:
            print(f"    same (charge,nce) pairs:   {np.mean(rates_same_cond) * 100:.2f}%  (n_pairs={len(rates_same_cond)})")
        if rates_diff_cond:
            print(f"    diff (charge,nce) pairs:   {np.mean(rates_diff_cond) * 100:.2f}%  (n_pairs={len(rates_diff_cond)})")

    # 6) leave-one-out oracle 上限（序列条件模型天花板）
    y_true, p_hat, p_hat_cond = [], [], []
    for seq, obs in groups.items():
        n_bond = obs[0][0].size
        if n_bond == 0:
            continue
        mat = np.stack([o[0] for o in obs])
        for r in range(len(obs)):
            loo = np.delete(mat, r, axis=0).mean(axis=0)
            y_true.append(mat[r])
            p_hat.append(loo)
            # 条件匹配版：只用同 (charge,nce) 的其他谱图；不足 1 张则退回全体 LOO
            same = [j for j in range(len(obs)) if j != r
                    and obs[j][1] == obs[r][1] and obs[j][2] == obs[r][2]]
            p_hat_cond.append(np.delete(mat, same, axis=0).mean(axis=0) if same else loo)
    y_true = np.concatenate(y_true)
    p_hat = np.concatenate(p_hat)
    p_hat_cond = np.concatenate(p_hat_cond)

    print(f"\n[leave-one-out oracle ceiling]  (pre 模型对照: lab_acc_mi≈0.800 / lab_f1_mi≈0.795 / AUC≈0.882)")
    for name, ph in (("all-spectra q", p_hat), ("condition-matched q", p_hat_cond)):
        acc, prec, rec, f1 = prf_micro(y_true, ph, 0.5)
        auc = rank_auc(y_true, ph)
        best = max((prf_micro(y_true, ph, t)[3], t) for t in np.arange(0.1, 0.95, 0.05))
        print(f"  {name}: thr0.5 acc={acc:.4f} P={prec:.4f} R={rec:.4f} F1={f1:.4f} AUC={auc:.4f} "
              f"| best-F1={best[0]:.4f}@thr{best[1]:.2f}")

    if args.save_q:
        out = pd.DataFrame(q_rows, columns=["seq", "bond_idx", "q", "n_spectra"])
        out.to_csv(args.save_q, index=False)
        print(f"\nsaved q table -> {args.save_q}")


if __name__ == "__main__":
    main()
