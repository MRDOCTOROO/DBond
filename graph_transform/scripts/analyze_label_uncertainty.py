#!/usr/bin/env python3
"""序列内标签不确定性分析（PBCLA multi 标签的噪声瓶颈检测）。

问题背景：pre 模型（屏蔽 intensity/scan_num/rt）对同一序列的多张谱图只能给出
同一个预测，而标签来自单次实测。若同一序列在不同谱图间标签本身不稳定
（within-sequence label variance 高），则任何序列条件模型（GAT/Transformer/
Graphormer）都存在由标签随机性决定的性能上限，架构改进无法突破。

对每个唯一序列 s 的每根键 i 计算：
    q_{s,i} = 该序列第 i 根键在所有谱图中的平均断裂率

实现说明：disagreement 与 leave-one-out oracle 均用组合恒等式向量化——
bond i 上取值不同的谱图对数恰为 k_i(n-k_i)，LOO 均值 = (k - y_r)/(n-1)，
复杂度 O(n_s)，全量数据分钟级内完成。

输出（stdout；--save_q 导出逐 (seq, bond) 的 q 表）：
    1. 谱图/序列基础统计（每序列谱图数分布）
    2. within-sequence label variance（q(1-q)）与模糊键占比
    3. per-bond entropy H(q)
    4. 方差分解与 sequence-level ICC（between / total）
    5. 同序列谱图间 label disagreement（总体 + 按条件 charge/nce 分层）
    6. leave-one-out oracle 上限（acc/F1/AUC，全谱 q 版 + 条件匹配 q 版）

用法（pod，项目根目录）：
    .venv/bin/python graph_transform/scripts/analyze_label_uncertainty.py \
        --inputs dataset/5fold/1222.train.fbr.shuffle.multi.csv \
                  dataset/5fold/1222.test.fbr.multi.csv
"""

from __future__ import annotations

import argparse
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
    ranks = pd.Series(p_hat).rank(method="average").to_numpy()
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
    total_bonds = sum(max(len(s) - 1, 0) for s in groups)
    print(f"rows={n_rows}  unique_seq={n_seq}  unique_bonds={total_bonds}", flush=True)

    # 1) 每序列谱图数分布
    ns = np.array([len(v) for v in groups.values()], dtype=np.float64)
    qs = np.percentile(ns, [0, 25, 50, 75, 100])
    print(f"spectra/seq: min={qs[0]:.0f} p25={qs[1]:.1f} median={qs[2]:.1f} p75={qs[3]:.1f} max={qs[4]:.0f}")
    print(f"  n_s=1: {int((ns == 1).sum())} seqs ({(ns == 1).mean() * 100:.1f}%); "
          f"n_s>=2: {int((ns >= 2).sum())} ({(ns >= 2).mean() * 100:.1f}%); "
          f"n_s>=5: {int((ns >= 5).sum())} ({(ns >= 5).mean() * 100:.1f}%)")

    q_rows = []
    per_seq_mean = []
    within_vars = []
    entropies = []
    grand_sum, grand_cnt = 0.0, 0
    # disagreement 全局累积（以 (谱图对, 键) 为计数单位）
    dis_num_total, pair_bond_total = 0.0, 0.0
    same_dis_num, same_pair_bonds = 0.0, 0.0
    diff_dis_num, diff_pair_bonds = 0.0, 0.0
    seq_any_diff = 0
    seq_multi = 0
    # LOO oracle 累积
    y_chunks, p_chunks, pc_chunks = [], [], []

    for seq, obs in groups.items():
        n_bond = obs[0][0].size
        if n_bond == 0:
            continue
        mat = np.stack([o[0] for o in obs])          # [n, L-1]
        n = mat.shape[0]
        k = mat.sum(axis=0)                           # 每键断裂次数
        q = k / n
        per_seq_mean.append(float(q.mean()))
        grand_sum += float(k.sum())
        grand_cnt += int(mat.size)
        for i in range(n_bond):
            q_rows.append((seq, i, float(q[i]), n))
        within_vars.append(q * (1.0 - q))
        ent = -np.where((q > 0) & (q < 1),
                        q * np.log2(np.clip(q, 1e-12, 1)) + (1 - q) * np.log2(np.clip(1 - q, 1e-12, 1)),
                        0.0)
        entropies.append(ent)

        # --- 5) 谱图两两 disagreement（组合恒等式：bond i 取值不同的对数 = k_i(n-k_i)）
        cond_keys = [(o[1], o[2]) for o in obs]
        cond_groups = defaultdict(list)
        for r, ck in enumerate(cond_keys):
            cond_groups[ck].append(r)

        # --- 6) leave-one-out oracle（(k - y_r)/(n-1)；n=1 记 NaN 后回退全局先验）
        if n >= 2:
            loo = (k[None, :] - mat) / (n - 1.0)
            dis_bonds = k * (n - k)
            pairs = n * (n - 1) / 2.0
            dis_num_total += float(dis_bonds.sum())
            pair_bond_total += pairs * n_bond
            seq_multi += 1
            if np.any((k > 0) & (k < n)):
                seq_any_diff += 1
            # 同条件（charge,nce）组内对
            for idxs in cond_groups.values():
                m = len(idxs)
                if m >= 2:
                    kg = mat[idxs].sum(axis=0)
                    same_dis_num += float((kg * (m - kg)).sum())
                    same_pair_bonds += m * (m - 1) / 2.0 * n_bond
            # 跨条件组对：二值标签下组间期望差异率 = q_g(1-q_h)+q_h(1-q_g)（精确）
            keys = list(cond_groups.keys())
            for a in range(len(keys)):
                for b in range(a + 1, len(keys)):
                    ia, ib = cond_groups[keys[a]], cond_groups[keys[b]]
                    qa = mat[ia].mean(axis=0)
                    qb = mat[ib].mean(axis=0)
                    diff_dis_num += float((qa * (1 - qb) + qb * (1 - qa)).sum()) * len(ia) * len(ib)
                    diff_pair_bonds += len(ia) * len(ib) * n_bond
        else:
            loo = np.full_like(mat, np.nan)

        # 条件匹配 LOO：同条件组内 leave-one-out；组内不足 2 张回退全体 LOO
        loo_cond = loo.copy()
        for idxs in cond_groups.values():
            m = len(idxs)
            if m >= 2:
                sub = mat[idxs]
                kg = sub.sum(axis=0)
                loo_cond[np.asarray(idxs)] = (kg[None, :] - sub) / (m - 1.0)
        y_chunks.append(mat)
        p_chunks.append(loo)
        pc_chunks.append(loo_cond)

    within_vars = np.concatenate(within_vars)
    entropies = np.concatenate(entropies)
    q_all = np.array([r[2] for r in q_rows])

    print(f"\n[within-sequence label variance]")
    print(f"  mean q(1-q) = {within_vars.mean():.4f}  (理论最大 0.25)")
    print(f"  ambiguous bonds (0.1<q<0.9): {(np.abs(q_all - 0.5) < 0.4).mean() * 100:.1f}%")
    print(f"  strongly ambiguous (0.25<q<0.75): {(np.abs(q_all - 0.5) < 0.25).mean() * 100:.1f}%")
    print(f"  deterministic bonds (q==0 or q==1): {np.mean((q_all == 0) | (q_all == 1)) * 100:.1f}%")
    print(f"[per-bond entropy]  mean H(q) = {entropies.mean():.4f} bits (max 1.0); "
          f"H>0.5: {(entropies > 0.5).mean() * 100:.1f}%")

    # 4) 方差分解 + ICC（one-way random effect： realized 标签按序列分组）
    within_mean = float(within_vars.mean())
    between = float(np.var(per_seq_mean, ddof=1)) if len(per_seq_mean) > 1 else 0.0
    total = between + within_mean
    grand = grand_sum / max(grand_cnt, 1)
    print(f"\n[variance decomposition]")
    print(f"  between-seq var = {between:.4f}  within-seq var = {within_mean:.4f}  total = {total:.4f}")
    print(f"  ICC(seq) = between/total = {between / total if total > 0 else float('nan'):.4f}")
    print(f"  grand cleavage rate = {grand:.4f}")

    # 5) disagreement 输出
    print(f"\n[label disagreement across spectra of same sequence]  (n_seq>=2: {seq_multi})")
    if seq_multi:
        print(f"  sequences with ANY differing labels: {seq_any_diff}/{seq_multi} ({seq_any_diff / seq_multi * 100:.1f}%)")
        print(f"  mean pairwise bond disagreement rate: {dis_num_total / pair_bond_total * 100:.2f}%")
        if same_pair_bonds > 0:
            print(f"    same (charge,nce) pairs:   {same_dis_num / same_pair_bonds * 100:.2f}%")
        if diff_pair_bonds > 0:
            print(f"    diff (charge,nce) pairs:   {diff_dis_num / diff_pair_bonds * 100:.2f}%")

    # 6) oracle 上限（n_s=1 行回退全局先验 p̄）
    y_true = np.concatenate(y_chunks)
    p_hat = np.concatenate(p_chunks)
    p_hat_cond = np.concatenate(pc_chunks)
    p_hat[np.isnan(p_hat)] = grand
    p_hat_cond[np.isnan(p_hat_cond)] = grand

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
