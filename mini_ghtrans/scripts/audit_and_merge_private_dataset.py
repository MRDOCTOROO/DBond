#!/usr/bin/env python3
"""
审计 dataset_private 下 train/test 划分，并生成未拆分全量数据文件。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


BASE_DIR = Path("dataset/dataset_private")


def positive_ratio(values: Iterable[str]) -> float:
    positives = 0
    total = 0
    for value in values:
        labels = [item.strip() for item in str(value).split(";") if item.strip() in {"0", "1"}]
        positives += sum(label == "1" for label in labels)
        total += len(labels)
    return positives / total if total else 0.0


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(BASE_DIR / name)


def length_distribution(df: pd.DataFrame) -> dict[int, int]:
    lengths = df["seq"].astype(str).str.len()
    return {int(idx): int(count) for idx, count in lengths.value_counts().sort_index().items()}


def key_set(df: pd.DataFrame, columns: list[str]) -> set[str]:
    return set(df[columns].astype(str).agg("|".join, axis=1))


def audit_split(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    train = train.copy()
    test = test.copy()
    train["seq_len"] = train["seq"].astype(str).str.len()
    test["seq_len"] = test["seq"].astype(str).str.len()

    all_lengths = sorted(set(train["seq_len"]).union(test["seq_len"]))
    per_length = []
    for seq_len in all_lengths:
        train_count = int((train["seq_len"] == seq_len).sum())
        test_count = int((test["seq_len"] == seq_len).sum())
        total = train_count + test_count
        per_length.append(
            {
                "seq_len": int(seq_len),
                "train_count": train_count,
                "test_count": test_count,
                "train_ratio_within_len": train_count / total if total else 0.0,
                "test_ratio_within_len": test_count / total if total else 0.0,
                "train_pct": train_count / len(train) if len(train) else 0.0,
                "test_pct": test_count / len(test) if len(test) else 0.0,
            }
        )

    overlap_counts = {
        "seq": len(set(train["seq"].astype(str)) & set(test["seq"].astype(str))),
        "seq_charge_nce": len(
            set(train["seq"].astype(str) + "|" + train["charge"].astype(str) + "|" + train["nce"].astype(str))
            & set(test["seq"].astype(str) + "|" + test["charge"].astype(str) + "|" + test["nce"].astype(str))
        ),
        "seq_charge_nce_rt": len(
            set(
                train["seq"].astype(str)
                + "|"
                + train["charge"].astype(str)
                + "|"
                + train["nce"].astype(str)
                + "|"
                + train["rt"].astype(str)
            )
            & set(
                test["seq"].astype(str)
                + "|"
                + test["charge"].astype(str)
                + "|"
                + test["nce"].astype(str)
                + "|"
                + test["rt"].astype(str)
            )
        ),
    }
    if "name" in train.columns and "name" in test.columns:
        overlap_counts["name"] = len(set(train["name"].astype(str)) & set(test["name"].astype(str)))

    if "true_multi" in train.columns and "true_multi" in test.columns:
        overlap_counts["full_multi"] = len(key_set(train, list(train.columns)) & key_set(test, list(test.columns)))
        train_pos_ratio = positive_ratio(train["true_multi"])
        test_pos_ratio = positive_ratio(test["true_multi"])
    else:
        overlap_counts["full_single"] = len(key_set(train, list(train.columns)) & key_set(test, list(test.columns)))
        train_pos_ratio = None
        test_pos_ratio = None

    missing_test_lengths = [row["seq_len"] for row in per_length if row["train_count"] > 0 and row["test_count"] == 0]

    return {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_length_distribution": length_distribution(train),
        "test_length_distribution": length_distribution(test),
        "train_max_len": int(train["seq_len"].max()),
        "test_max_len": int(test["seq_len"].max()),
        "train_positive_ratio": train_pos_ratio,
        "test_positive_ratio": test_pos_ratio,
        "per_length_split": per_length,
        "missing_test_lengths": missing_test_lengths,
        "overlap_counts": overlap_counts,
        "conclusion": {
            "has_obvious_leakage": any(value > 0 for value in overlap_counts.values()),
            "length_distribution_is_fully_covered": len(missing_test_lengths) == 0,
        },
    }


def merge_files(train_name: str, test_name: str, output_name: str) -> dict:
    train_df = load_csv(train_name)
    test_df = load_csv(test_name)
    merged = pd.concat([train_df, test_df], ignore_index=True)
    output_path = BASE_DIR / output_name
    merged.to_csv(output_path, index=False)
    return {"output": str(output_path), "rows": int(len(merged)), "columns": list(merged.columns)}


def main():
    multi_train = load_csv("1222.train.fbr.multi.csv")
    multi_test = load_csv("1222.test.fbr.multi.csv")
    single_train = load_csv("1222.train.csv")
    single_test = load_csv("1222.test.csv")

    report = {
        "multi_label_split_audit": audit_split(multi_train, multi_test),
        "single_label_split_audit": audit_split(single_train, single_test),
        "merged_files": [
            merge_files("1222.train.csv", "1222.test.csv", "1222.full.csv"),
            merge_files("1222.train.fbr.csv", "1222.test.fbr.csv", "1222.full.fbr.csv"),
            merge_files("1222.train.fbr.multi.csv", "1222.test.fbr.multi.csv", "1222.full.fbr.multi.csv"),
        ],
    }

    report_path = BASE_DIR / "1222.split_audit_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
