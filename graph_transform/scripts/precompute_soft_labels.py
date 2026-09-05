#!/usr/bin/env python3
"""q 软标签预计算：按 (seq, charge, nce) 分组求逐键平均断裂率，写入 soft_multi 列。

背景（标签不确定性分析结论）：93% 的标签方差在序列内部，其中大部分是
charge/nce 条件效应、约 15% 是同条件随机碎裂噪声。单谱 0/1 标签是
P(y | seq, charge, nce) 的一次含噪实现；直接拿它训练会让梯度被实现噪声污染。
软标签 q_{s,c,i} = 同 (seq,charge,nce) 组内第 i 键的平均断裂率，正是 pre 特征集
（序列+charge+nce）下的贝叶斯最优回归目标。

防泄漏：q 只从每折自己的 train 文件计算；test 文件原样复制（评估口径不变，
仍用 realized 标签）。组内只有 1 张谱图时退化为 realized 标签本身。

用法（pod，项目根目录）：
    .venv/bin/python graph_transform/scripts/precompute_soft_labels.py \
        --fold_dir dataset/5fold --out_fold_dir dataset/5fold_soft
之后五折训练加 --fold_data_dir dataset/5fold_soft。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_5fold import TEST_SUFFIX, TRAIN_SUFFIX, discover_folds  # noqa: E402


def parse_labels(true_multi: str) -> np.ndarray:
    toks = [t.strip() for t in str(true_multi).strip().split(";") if t.strip() != ""]
    return np.asarray([float(t) for t in toks], dtype=np.float64)


def compute_group_soft(frame: pd.DataFrame, group_keys) -> pd.DataFrame:
    """按 group_keys 分组，组内逐键求均值 → soft_multi 字符串列。"""
    soft_cols: list = [None] * len(frame)
    n_groups = 0
    group_sizes = []
    for _, idxs in frame.groupby(list(group_keys), sort=False).indices.items():
        idxs = np.asarray(idxs)
        mats = np.stack([parse_labels(v) for v in frame["true_multi"].iloc[idxs]])
        q = mats.mean(axis=0)
        text = ";".join(f"{v:.4f}" for v in q)
        for i in idxs:
            soft_cols[int(i)] = text
        n_groups += 1
        group_sizes.append(len(idxs))
    frame = frame.copy()
    frame["soft_multi"] = soft_cols
    stats = {
        "n_groups": n_groups,
        "group_size_mean": float(np.mean(group_sizes)),
        "group_size_min": int(np.min(group_sizes)),
        "group_size_max": int(np.max(group_sizes)),
    }
    return frame, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold_dir", default="dataset/5fold")
    ap.add_argument("--out_fold_dir", default="dataset/5fold_soft")
    ap.add_argument("--group_keys", default="seq,charge,nce",
                    help="逗号分隔分组键（默认条件匹配版；只用 seq 则为序列均值版）")
    ap.add_argument("--folds", default=None, help="逗号分隔 fold 子集，默认全部")
    args = ap.parse_args()

    group_keys = tuple(k.strip() for k in args.group_keys.split(",") if k.strip())
    os.makedirs(args.out_fold_dir, exist_ok=True)
    folds = args.folds.split(",") if args.folds else discover_folds(args.fold_dir)

    for fold in folds:
        train_path = os.path.join(args.fold_dir, f"{fold}{TRAIN_SUFFIX}")
        test_path = os.path.join(args.fold_dir, f"{fold}{TEST_SUFFIX}")
        out_train = os.path.join(args.out_fold_dir, f"{fold}{TRAIN_SUFFIX}")
        out_test = os.path.join(args.out_fold_dir, f"{fold}{TEST_SUFFIX}")

        frame = pd.read_csv(train_path)
        bad = sum(
            1 for lv, seq in zip(frame["true_multi"], frame["seq"])
            if len(str(lv).strip().split(";")) != max(len(str(seq)) - 1, 0)
        )
        if bad:
            raise ValueError(f"fold {fold}: {bad} rows with |true_multi| != len(seq)-1")

        frame, stats = compute_group_soft(frame, group_keys)
        frame.to_csv(out_train, index=False)
        shutil.copyfile(test_path, out_test)  # test 原样：评估仍用 realized 标签

        # 软标签与 realized 的平均差异 ≈ 标签被平滑掉的噪声量
        diffs = [
            np.abs(parse_labels(s) - parse_labels(y)).mean()
            for s, y in zip(frame["soft_multi"], frame["true_multi"])
        ]
        print(f"fold {fold}: rows={len(frame)} groups={stats['n_groups']} "
              f"group_size mean/min/max = {stats['group_size_mean']:.1f}"
              f"/{stats['group_size_min']}/{stats['group_size_max']} "
              f"mean|q-y|={float(np.mean(diffs)):.4f}")
    print(f"\nsoft fold dir ready: {args.out_fold_dir} (group_keys={list(group_keys)})")


if __name__ == "__main__":
    main()
