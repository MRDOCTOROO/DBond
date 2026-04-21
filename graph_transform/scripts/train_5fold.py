#!/usr/bin/env python3
"""五折交叉验证封装脚本。"""

import argparse
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime

import pandas as pd
import torch
import yaml

TRAIN_SUFFIX = ".train.fbr.shuffle.multi.csv"
TEST_SUFFIX = ".test.fbr.multi.csv"
DEFAULT_FOLD_DIR = "dataset/5fold"
SUMMARY_COLUMNS = ["metric", "mean", "std", "min", "max", "num_folds"]
SUMMARY_METRIC_EXCLUDE_PREFIXES = ("gpu_mem_",)


def discover_folds(fold_dir: str):
    if not os.path.isdir(fold_dir):
        raise FileNotFoundError(f"Fold directory not found: {fold_dir}")
    train_ids, test_ids = set(), set()
    for name in os.listdir(fold_dir):
        if name.endswith(TRAIN_SUFFIX):
            train_ids.add(name[:-len(TRAIN_SUFFIX)])
        elif name.endswith(TEST_SUFFIX):
            test_ids.add(name[:-len(TEST_SUFFIX)])
    fold_ids = sorted(train_ids & test_ids, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
    if not fold_ids:
        raise ValueError(f"No valid fold pairs found in {fold_dir}")
    return fold_ids


def latest_subdir(base_dir: str) -> str:
    subdirs = [os.path.join(base_dir, name) for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]
    if not subdirs:
        raise FileNotFoundError(f"No run directory found under {base_dir}")
    return max(subdirs, key=os.path.getmtime)


def load_metric_csv(metric_path: str):
    df = pd.read_csv(metric_path)
    metrics = {}
    for _, row in df.iterrows():
        metric_name = str(row["metric"]).strip().lower()
        metric_value = row["value"]
        if pd.isna(metric_value):
            continue
        metrics[metric_name] = float(metric_value)
    return metrics


def should_aggregate_metric(metric_name: str) -> bool:
    return not any(metric_name.startswith(prefix) for prefix in SUMMARY_METRIC_EXCLUDE_PREFIXES)


def load_best_metrics(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metrics = checkpoint.get("metrics", {}) or {}
    return checkpoint.get("epoch", -1) + 1, metrics


def make_fold_config(base_config, fold_id: str, fold_dir: str, cv_root: str, fold_seed: int):
    config = deepcopy(base_config)
    train_csv = os.path.join(fold_dir, f"{fold_id}{TRAIN_SUFFIX}")
    test_csv = os.path.join(fold_dir, f"{fold_id}{TEST_SUFFIX}")
    run_root = os.path.join(cv_root, f"fold_{fold_id}")
    checkpoint_root = os.path.join(run_root, "checkpoints")
    config.setdefault("experiment", {})["seed"] = fold_seed
    config["data"]["train_csv_path"] = train_csv
    config["data"]["test_csv_path"] = test_csv
    config["data"]["val_csv_path"] = None
    config["training"]["checkpoint_dir"] = checkpoint_root
    config.setdefault("logging", {})["log_dir"] = os.path.join(run_root, "logs")
    config["logging"]["tensorboard_log_dir"] = os.path.join(run_root, "tensorboard")
    config.setdefault("evaluation", {})["output_metric_dir"] = os.path.join(run_root, "metrics")
    config["evaluation"]["output_pred_dir"] = os.path.join(run_root, "preds")
    return config, run_root, checkpoint_root


def build_command(train_script: str, fold_config_path: str, args, fold_seed: int):
    cmd = [sys.executable, train_script, "--config", fold_config_path, "--seed", str(fold_seed)]
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.learning_rate is not None:
        cmd.extend(["--learning_rate", str(args.learning_rate)])
    if args.device is not None:
        cmd.extend(["--device", args.device])
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run 5-fold cross validation")
    parser.add_argument("--config", required=True, help="Base config yaml")
    parser.add_argument("--fold_data_dir", default=DEFAULT_FOLD_DIR, help="Directory containing 5-fold csv files")
    parser.add_argument("--folds", nargs="+", help="Optional subset of fold ids")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    base_seed = args.seed or base_config.get("experiment", {}).get("seed", 42)
    available_folds = discover_folds(args.fold_data_dir)
    fold_ids = args.folds or available_folds
    missing = [fold_id for fold_id in fold_ids if fold_id not in available_folds]
    if missing:
        raise ValueError(f"Requested folds not found: {missing}; available folds: {available_folds}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_base = base_config.get("training", {}).get("checkpoint_dir", "checkpoints/graph_transform")
    cv_root = os.path.join(ckpt_base, "5fold", timestamp)
    os.makedirs(cv_root, exist_ok=True)

    train_script = os.path.join(os.path.dirname(__file__), "train_graph_model.py")
    results = []
    summary_metric_order = []
    overall_start = time.perf_counter()

    for fold_index, fold_id in enumerate(fold_ids):
        fold_seed = base_seed + fold_index
        fold_config, run_root, checkpoint_root = make_fold_config(base_config, fold_id, args.fold_data_dir, cv_root, fold_seed)
        os.makedirs(run_root, exist_ok=True)
        fold_config_path = os.path.join(run_root, "config.yaml")
        with open(fold_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(fold_config, f, sort_keys=False, allow_unicode=True)

        print(f"[5fold] start fold={fold_id} seed={fold_seed}")
        subprocess.run(build_command(train_script, fold_config_path, args, fold_seed), check=True)

        run_dir = latest_subdir(checkpoint_root)
        best_epoch, best_metrics = load_best_metrics(os.path.join(run_dir, "best_model.pt"))
        test_metrics = load_metric_csv(os.path.join(run_root, "metrics", "latest_test_metric.csv"))
        fold_result = {
            "fold_id": fold_id,
            "seed": fold_seed,
            "best_epoch": best_epoch,
            "best_val_f1": best_metrics.get("f1"),
            "checkpoint_dir": run_dir,
        }
        for metric_name, metric_value in test_metrics.items():
            fold_result[metric_name] = metric_value
            if should_aggregate_metric(metric_name) and metric_name not in summary_metric_order:
                summary_metric_order.append(metric_name)
        results.append(fold_result)

        fold_f1 = fold_result.get("f1")
        if fold_f1 is not None:
            print(f"[5fold] done fold={fold_id} f1={fold_f1:.4f}")
        else:
            print(f"[5fold] done fold={fold_id}")

    per_fold_df = pd.DataFrame(results)
    agg_rows = []
    for metric in summary_metric_order:
        if metric not in per_fold_df.columns:
            continue
        series = pd.to_numeric(per_fold_df[metric], errors="coerce").dropna()
        if not series.empty:
            agg_rows.append({
                "metric": metric,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "min": float(series.min()),
                "max": float(series.max()),
                "num_folds": int(series.shape[0]),
            })
    agg_df = pd.DataFrame(agg_rows, columns=SUMMARY_COLUMNS)

    metrics_path = os.path.join(cv_root, "5fold_metrics.csv")
    summary_path = os.path.join(cv_root, "5fold_summary.csv")
    aggregate_path = os.path.join(cv_root, "5fold_aggregate.csv")
    per_fold_df.to_csv(metrics_path, index=False)
    agg_df.to_csv(summary_path, index=False)
    agg_df.to_csv(aggregate_path, index=False)

    print(f"[5fold] per-fold metrics saved to {metrics_path}")
    print(f"[5fold] summary saved to {summary_path}")
    print(f"[5fold] aggregate saved to {aggregate_path}")
    print(f"[5fold] total_time={time.perf_counter() - overall_start:.2f}s")


if __name__ == "__main__":
    main()
