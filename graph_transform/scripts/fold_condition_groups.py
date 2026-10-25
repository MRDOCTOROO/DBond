#!/usr/bin/env python3
"""条件组折叠：把 (seq, charge, nce) 相同的谱图行折叠为一条训练样本。

背景（标签不确定性分析 + 等价性审计 20260906）：precompute_soft_labels.py 生成的
5fold_soft 里，同一条件组的 q 被逐行复制后仍全部送入 BCE。由线性恒等式
    sum_j BCE(z, y_j) = n_g * BCE(z, q_g)
重复行软标签与 hard BCE 的损失函数完全相同（gat_aux_soft 五折实证 f1 0.7956 vs
hard 0.7972，种子噪声内），不构成新的训练目标。

折叠后每个条件组只保留一行（soft_multi=q 即训练目标），组权重交给
data.weighting_scheme 在 loss 侧显式选择：
    spectrum         w = n_g   （期望上复现行级 hard BCE，实现自检用）
    group_uniform    w = 1     （每个 (seq,charge,nce) 条件等权）
    sequence_balanced w = 1/K_s（每序列等权，序列内条件平分，最贴合候选肽筛选）

test 文件不折叠（realized 行级指标 + q_* 指标的真值保持原口径），以符号链接
放入输出目录；缺列/缺文件直接报错。

用法（pod，项目根目录，先跑 precompute_soft_labels.py 得到 5fold_soft）：
    .venv/bin/python graph_transform/scripts/fold_condition_groups.py \
        --fold_dir dataset/5fold_soft --out_fold_dir dataset/5fold_folded
之后单折试点：
    train_5fold.py --config <folded 配置> --folds 1222 --fold_data_dir dataset/5fold_folded
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_5fold import TEST_SUFFIX, TRAIN_SUFFIX, discover_folds  # noqa: E402

GROUP_KEYS = ("seq", "charge", "nce")


def fold_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """按 (seq,charge,nce) 折叠：每组取首行 + group_n 列。

    pep_mass/scan_num/rt/intensity 等取组内首行（同 seq+charge 下 pep_mass 恒定，
    其余为 pre_synthesis 掩码特征，取值不影响模型输入）。true_multi 保留首行
    realized 标签仅为满足数据列契约，use_soft_labels 训练时不消费。
    """
    sizes = frame.groupby(list(GROUP_KEYS), sort=False).size().rename("group_n")
    folded = (
        frame.groupby(list(GROUP_KEYS), sort=False)
        .first()
        .reset_index()
        .merge(sizes.reset_index(), on=list(GROUP_KEYS), how="left")
    )
    return folded


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold_dir", default="dataset/5fold_soft",
                    help="precompute_soft_labels.py 的输出目录（需含 soft_multi）")
    ap.add_argument("--out_fold_dir", default="dataset/5fold_folded")
    ap.add_argument("--folds", default=None, help="逗号分隔 fold 子集，默认全部")
    args = ap.parse_args()

    os.makedirs(args.out_fold_dir, exist_ok=True)
    folds = args.folds.split(",") if args.folds else discover_folds(args.fold_dir)

    for fold in folds:
        train_path = os.path.join(args.fold_dir, f"{fold}{TRAIN_SUFFIX}")
        test_path = os.path.join(args.fold_dir, f"{fold}{TEST_SUFFIX}")
        out_train = os.path.join(args.out_fold_dir, f"{fold}{TRAIN_SUFFIX}")
        out_test = os.path.join(args.out_fold_dir, f"{fold}{TEST_SUFFIX}")

        frame = pd.read_csv(train_path)
        if "soft_multi" not in frame.columns:
            raise ValueError(f"{train_path} 缺 soft_multi 列：请先跑 precompute_soft_labels.py")
        folded = fold_frame(frame)
        # 折叠行数必须等于唯一条件组数，防 groupby/merge 错位
        n_groups = frame.groupby(list(GROUP_KEYS)).ngroups
        if len(folded) != n_groups:
            raise AssertionError(f"fold {fold}: 折叠行数 {len(folded)} != 组数 {n_groups}")
        folded.to_csv(out_train, index=False)

        # 软标签列完整性（折叠后每行必须有 q）
        if folded["soft_multi"].isna().any():
            raise ValueError(f"fold {fold}: 折叠后存在 soft_multi 缺失行")

        # test 原样链接进输出目录（不折叠：realized/q 指标保持行级口径）
        if not os.path.exists(out_test):
            try:
                os.symlink(os.path.abspath(test_path), out_test)
            except OSError:
                pd.read_csv(test_path).to_csv(out_test, index=False)

        seqs = folded["seq"].astype(str)
        per_seq_groups = folded.groupby("seq").size()
        print(f"fold {fold}: rows {len(frame)} -> {len(folded)} groups "
              f"({len(seqs.unique())} seqs, {per_seq_groups.mean():.1f} groups/seq, "
              f"min {per_seq_groups.min()} max {per_seq_groups.max()}); "
              f"sum(group_n)={int(folded['group_n'].sum())} (应等于 {len(frame)})")

    print(f"\nfolded dir ready: {args.out_fold_dir}")


if __name__ == "__main__":
    main()
