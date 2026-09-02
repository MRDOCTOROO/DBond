#!/usr/bin/env python3
"""从 multi（fbr）五折 CSV 生成 dbond-s 逐键格式折文件。

dbond-s 数据格式：每行一个 (谱图, 肽键) 样本，列：
    bond_aa(两侧二肽 seq[j:j+2]), bond_pos(0-based), bond_label(0/1),
    seq, charge, pep_mass, intensity, nce, scan_num, rt

本脚本把 dataset/5fold/{f}.train.fbr.shuffle.multi.csv（一行=一条谱图，
true_multi 为逐键 0/1 串）展开为 dataset/5fold/{f}.train.shuffle.csv，
{f}.test.fbr.multi.csv 同理 → {f}.test.csv。

生成前做严格标签校验（token∈{0,1} 且数量=len(seq)-1），不一致直接失败，
与训练入口 data/label_validation.py 的口径一致。

用法（仓库根目录）：
    .venv/bin/python dataset/make_dbond_s_folds.py                # 全部 5 折
    .venv/bin/python dataset/make_dbond_s_folds.py --folds 1222   # 指定折
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

FOLDS = ["1222", "2252", "3514", "6072", "9075"]
OUT_COLUMNS = ["bond_aa", "bond_pos", "bond_label", "seq",
               "charge", "pep_mass", "intensity", "nce", "scan_num", "rt"]


def expand_multi_to_s(df: pd.DataFrame, source: str) -> pd.DataFrame:
    seqs = df["seq"].astype(str)
    tokens = df["true_multi"].fillna("").astype(str).str.split(";")
    # 严格标签校验：token∈{0,1} 且数量 = len(seq)-1
    for i, (seq, toks) in enumerate(zip(seqs, tokens)):
        toks = [t for t in toks if t != ""]
        bad = [t for t in toks if t not in ("0", "1")]
        if bad or len(toks) != len(seq) - 1:
            raise ValueError(
                f"{source} row {i}: 标签校验失败 "
                f"(len(seq)={len(seq)}, tokens={len(toks)}, bad={bad[:3]})")

    out = df[["seq", "charge", "pep_mass", "intensity", "nce", "scan_num", "rt"]].copy()
    out["bond_pos"] = tokens.apply(lambda toks: list(range(len([t for t in toks if t != ""]))))
    out["bond_label"] = tokens.apply(lambda toks: [int(t) for t in toks if t != ""])
    out["bond_aa"] = seqs.map(lambda s: [s[j:j + 2] for j in range(len(s) - 1)])
    out = out.explode(["bond_pos", "bond_label", "bond_aa"], ignore_index=True)
    out["bond_pos"] = out["bond_pos"].astype(int)
    out["bond_label"] = out["bond_label"].astype(int)
    return out[OUT_COLUMNS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold_dir", default="dataset/5fold")
    ap.add_argument("--folds", nargs="+", default=FOLDS)
    args = ap.parse_args()

    for fold in args.folds:
        jobs = [
            (os.path.join(args.fold_dir, f"{fold}.train.fbr.shuffle.multi.csv"),
             os.path.join(args.fold_dir, f"{fold}.train.shuffle.csv")),
            (os.path.join(args.fold_dir, f"{fold}.test.fbr.multi.csv"),
             os.path.join(args.fold_dir, f"{fold}.test.csv")),
        ]
        for src, dst in jobs:
            if not os.path.exists(src):
                print(f"SKIP(缺源) {src}")
                continue
            df = pd.read_csv(src)
            out = expand_multi_to_s(df, src)
            out.to_csv(dst, index=False)
            pos = out["bond_label"].mean()
            print(f"{dst}: {len(df)} 谱图 → {len(out)} 键行 (正率 {pos:.4f})")


if __name__ == "__main__":
    main()
