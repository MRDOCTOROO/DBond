#!/usr/bin/env python3
"""
五折交叉验证训练脚本。

默认扫描 dataset/5_fold 下的 CSV，并根据路径中的 fold/train/val/test 关键字自动识别：
1. dataset/5_fold/fold1/train.csv
2. dataset/5_fold/fold_1_val.csv
3. dataset/5_fold/train_fold1.csv

每折都会调用主训练流程完成训练、测试，并把测试指标写入 result 目录。
额外输出跨折汇总结果，便于比较均值与标准差。
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_graph_model import load_config, run_training


FOLD_PATTERNS = (
    re.compile(r"fold[\s_\-]*(\d+)", re.IGNORECASE),
    re.compile(r"(\d+)[\s_\-]*fold", re.IGNORECASE),
)
SPLIT_PATTERNS = {
    "train": re.compile(r"(?:^|[^a-z])(train)(?:[^a-z]|$)", re.IGNORECASE),
    "val": re.compile(r"(?:^|[^a-z])(validation|valid|val)(?:[^a-z]|$)", re.IGNORECASE),
    "test": re.compile(r"(?:^|[^a-z])(test)(?:[^a-z]|$)", re.IGNORECASE),
}


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s[%(levelname)s]:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("graph_transform_5fold")


def _extract_fold_id(text: str) -> str | None:
    for pattern in FOLD_PATTERNS:
        match = pattern.search(text)
        if match:
            return str(int(match.group(1)))
    return None


def _extract_split(text: str) -> str | None:
    for split_name, pattern in SPLIT_PATTERNS.items():
        if pattern.search(text):
            return split_name
    return None


def discover_folds(folds_dir: str) -> Dict[str, Dict[str, str]]:
    if not os.path.isdir(folds_dir):
        raise FileNotFoundError(f"Fold dataset directory not found: {folds_dir}")

    fold_map: Dict[str, Dict[str, str]] = {}
    for root, _, files in os.walk(folds_dir):
        for filename in files:
            if not filename.lower().endswith(".csv"):
                continue

            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, folds_dir).replace("\\", "/")
            fold_id = _extract_fold_id(rel_path)
            split = _extract_split(rel_path)

            if fold_id is None or split is None:
                continue

            fold_entry = fold_map.setdefault(fold_id, {})
            fold_entry[split] = os.path.abspath(full_path)

    return dict(sorted(fold_map.items(), key=lambda item: int(item[0])))


def validate_folds(fold_map: Dict[str, Dict[str, str]], expected_folds: int | None) -> None:
    if not fold_map:
        raise ValueError(
            "No fold CSV files were discovered. Expected names/paths containing both fold id and split, "
            "for example dataset/5_fold/fold1/train.csv or dataset/5_fold/fold_1_test.csv."
        )

    if expected_folds is not None and len(fold_map) != expected_folds:
        raise ValueError(f"Expected {expected_folds} folds, but discovered {len(fold_map)} folds: {list(fold_map)}")

    invalid_folds = []
    for fold_id, split_map in fold_map.items():
        missing = [split for split in ("train", "test") if split not in split_map]
        if missing:
            invalid_folds.append(f"fold {fold_id} missing required split(s): {', '.join(missing)}")

    if invalid_folds:
        raise ValueError("; ".join(invalid_folds))


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.device:
        config["device"]["device_type"] = args.device
    return config


def build_fold_config(
    base_config: Dict[str, Any],
    *,
    fold_id: str,
    split_map: Dict[str, str],
    checkpoint_root: str,
    metric_root: str,
    pred_root: str,
    log_root: str,
    seed: int,
) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)

    data_config = config.setdefault("data", {})
    data_config["train_csv_path"] = split_map["train"]
    data_config["test_csv_path"] = split_map["test"]
    data_config["val_csv_path"] = split_map.get("val")

    training_config = config.setdefault("training", {})
    training_config["checkpoint_dir"] = os.path.join(checkpoint_root, f"fold_{fold_id}")

    evaluation_config = config.setdefault("evaluation", {})
    evaluation_config["output_metric_dir"] = os.path.join(metric_root, f"fold_{fold_id}")
    evaluation_config["output_pred_dir"] = os.path.join(pred_root, f"fold_{fold_id}")

    logging_config = config.setdefault("logging", {})
    logging_config["log_dir"] = os.path.join(log_root, f"fold_{fold_id}")

    experiment_config = config.setdefault("experiment", {})
    experiment_name = experiment_config.get("name", "graph_transform_binary_bond")
    experiment_config["name"] = f"{experiment_name}_fold_{fold_id}"
    experiment_config["seed"] = seed

    return config


def build_summary_rows(per_fold_df: pd.DataFrame) -> List[Dict[str, Any]]:
    excluded_columns = {
        "fold",
        "train_csv_path",
        "val_csv_path",
        "test_csv_path",
        "checkpoint_dir",
        "best_checkpoint_path",
        "best_epoch",
        "test_dataset_size",
    }
    metric_columns = [
        column for column in per_fold_df.columns
        if column not in excluded_columns and pd.api.types.is_numeric_dtype(per_fold_df[column])
    ]

    rows: List[Dict[str, Any]] = []
    for metric_name in metric_columns:
        series = per_fold_df[metric_name].dropna().astype(float)
        if series.empty:
            continue
        rows.append(
            {
                "metric": metric_name,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "min": float(series.min()),
                "max": float(series.max()),
                "num_folds": int(series.shape[0]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphTransformer 五折交叉验证训练")
    parser.add_argument("--config", type=str, required=True, help="基础配置文件路径")
    parser.add_argument("--folds_dir", type=str, default="dataset/5_fold", help="五折数据目录")
    parser.add_argument("--expected_folds", type=int, default=5, help="期望发现的 fold 数量")
    parser.add_argument("--epochs", type=int, help="覆盖训练轮数")
    parser.add_argument("--batch_size", type=int, help="覆盖批大小")
    parser.add_argument("--learning_rate", type=float, help="覆盖学习率")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], help="覆盖设备")
    parser.add_argument("--seed", type=int, default=None, help="基础随机种子")
    args = parser.parse_args()

    logger = setup_logging()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")

    config_args = argparse.Namespace(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    base_config = load_config(args.config, config_args)
    base_config = apply_overrides(base_config, args)

    fold_map = discover_folds(args.folds_dir)
    validate_folds(fold_map, args.expected_folds)

    base_checkpoint_root = base_config.get("training", {}).get("checkpoint_dir", "checkpoints/graph_transform")
    checkpoint_root = os.path.join(base_checkpoint_root, "5_fold")
    evaluation_config = base_config.setdefault("evaluation", {})
    metric_root = os.path.join(evaluation_config.get("output_metric_dir", "result/metric/graph_transform"), "5_fold")
    pred_root = os.path.join(evaluation_config.get("output_pred_dir", "result/pred/graph_transform"), "5_fold")
    log_root = os.path.join(base_config.get("logging", {}).get("log_dir", "logs/graph_transform"), "5_fold")

    os.makedirs(checkpoint_root, exist_ok=True)
    os.makedirs(metric_root, exist_ok=True)
    os.makedirs(pred_root, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)

    logger.info("Discovered %s folds under %s", len(fold_map), os.path.abspath(args.folds_dir))
    for fold_id, split_map in fold_map.items():
        logger.info(
            "Fold %s -> train=%s, val=%s, test=%s",
            fold_id,
            split_map.get("train"),
            split_map.get("val"),
            split_map.get("test"),
        )

    base_seed = args.seed if args.seed is not None else int(base_config.get("experiment", {}).get("seed", 42))
    per_fold_results: List[Dict[str, Any]] = []

    for fold_index, (fold_id, split_map) in enumerate(fold_map.items(), start=1):
        fold_seed = base_seed + fold_index - 1
        logger.info("========== Starting fold %s/%s (fold_id=%s, seed=%s) ==========", fold_index, len(fold_map), fold_id, fold_seed)

        fold_config = build_fold_config(
            base_config,
            fold_id=fold_id,
            split_map=split_map,
            checkpoint_root=checkpoint_root,
            metric_root=metric_root,
            pred_root=pred_root,
            log_root=log_root,
            seed=fold_seed,
        )

        result = run_training(fold_config, seed=fold_seed)
        fold_row: Dict[str, Any] = {
            "fold": int(fold_id),
            "train_csv_path": split_map["train"],
            "val_csv_path": split_map.get("val"),
            "test_csv_path": split_map["test"],
            "checkpoint_dir": result.get("checkpoint_dir"),
            "best_checkpoint_path": result.get("best_checkpoint_path"),
            "best_epoch": result.get("best_epoch"),
            "best_val_f1": result.get("best_val_f1"),
            "test_dataset_size": result.get("test_dataset_size"),
        }
        test_metrics = result.get("test_metrics") or {}
        for key, value in test_metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                fold_row[key] = float(value)
        per_fold_results.append(fold_row)

    per_fold_df = pd.DataFrame(per_fold_results).sort_values("fold").reset_index(drop=True)
    summary_rows = build_summary_rows(per_fold_df)
    summary_df = pd.DataFrame(summary_rows)

    per_fold_csv = os.path.join(metric_root, "five_fold_metrics.csv")
    summary_csv = os.path.join(metric_root, "five_fold_summary.csv")
    per_fold_df.to_csv(per_fold_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    logger.info("Saved per-fold metrics to %s", per_fold_csv)
    logger.info("Saved aggregated summary to %s", summary_csv)

    if not summary_df.empty:
        focus_metrics = summary_df[summary_df["metric"].isin(["loss", "f1", "auc", "ex_f1", "lab_f1_mi"])]
        for _, row in focus_metrics.iterrows():
            logger.info(
                "Summary %s -> mean=%.6f std=%.6f",
                row["metric"],
                row["mean"],
                row["std"],
            )


if __name__ == "__main__":
    main()
