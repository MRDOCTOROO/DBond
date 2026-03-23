#!/usr/bin/env python3
"""
统计数据集序列长度分布与标签正例比例。
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Report sequence length statistics for a DBond CSV file.")
    parser.add_argument("--csv", required=True, help="Path to the CSV file.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a text table.")
    return parser.parse_args()


def count_positive_ratio(series: pd.Series) -> float:
    positives = 0
    total = 0
    for value in series.fillna(""):
        labels = [item.strip() for item in str(value).split(";") if item.strip() in {"0", "1"}]
        total += len(labels)
        positives += sum(item == "1" for item in labels)
    return positives / total if total else 0.0


def build_report(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path, usecols=["seq", "true_multi"])
    lengths = df["seq"].astype(str).str.len()
    length_counts = lengths.value_counts().sort_index()

    return {
        "csv_path": str(csv_path),
        "num_samples": int(len(df)),
        "min_seq_len": int(lengths.min()),
        "max_seq_len": int(lengths.max()),
        "avg_seq_len": float(lengths.mean()),
        "positive_ratio": float(count_positive_ratio(df["true_multi"])),
        "length_counts": {int(idx): int(count) for idx, count in length_counts.items()},
    }


def main():
    args = parse_args()
    report = build_report(Path(args.csv))

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print(f"CSV: {report['csv_path']}")
    print(f"Samples: {report['num_samples']}")
    print(f"Min length: {report['min_seq_len']}")
    print(f"Max length: {report['max_seq_len']}")
    print(f"Avg length: {report['avg_seq_len']:.4f}")
    print(f"Positive ratio: {report['positive_ratio']:.6f}")
    print("Length counts:")
    for seq_len, count in report["length_counts"].items():
        print(f"  {seq_len}: {count}")


if __name__ == "__main__":
    main()
