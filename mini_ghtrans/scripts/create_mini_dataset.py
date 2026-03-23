#!/usr/bin/env python3
"""
从完整多标签数据集中抽取一个可重复的小型训练/测试集，供 mini_ghtrans 全流程验证。
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Create a small debug dataset for mini_ghtrans.")
    parser.add_argument("--train_csv", default="dataset/dataset_private/1222.train.fbr.shuffle.multi.csv")
    parser.add_argument("--test_csv", default="dataset/dataset_private/1222.test.fbr.multi.csv")
    parser.add_argument("--output_dir", default="mini_ghtrans/mini_data")
    parser.add_argument("--train_per_len", type=int, default=8)
    parser.add_argument("--test_per_len", type=int, default=4)
    return parser.parse_args()


def sample_by_length(csv_path: str, per_len: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["seq_len"] = df["seq"].astype(str).str.len()
    sampled = (
        df.sort_values(["seq_len", "seq", "charge", "nce", "rt"])
        .groupby("seq_len", group_keys=False)
        .head(per_len)
        .reset_index(drop=True)
    )
    return sampled.drop(columns=["seq_len"])


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = sample_by_length(args.train_csv, args.train_per_len)
    test_df = sample_by_length(args.test_csv, args.test_per_len)

    train_path = output_dir / "train.mini.multi.csv"
    test_path = output_dir / "test.mini.multi.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Created {train_path} with {len(train_df)} rows")
    print(f"Created {test_path} with {len(test_df)} rows")
    print("Train length counts:")
    print(train_df["seq"].astype(str).str.len().value_counts().sort_index().to_string())
    print("Test length counts:")
    print(test_df["seq"].astype(str).str.len().value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
