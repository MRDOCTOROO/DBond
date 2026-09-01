#!/usr/bin/env python3
"""五折交叉验证封装脚本。"""

import argparse
import hashlib
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from typing import Optional

import pandas as pd
import torch
import yaml

from train_graph_model import apply_ablation_config

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
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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


def resolve_fold_runtime_paths(fold_config, checkpoint_root: str, run_root: str):
    """按 train_graph_model 的 ablation 规则推导实际运行输出路径。"""
    resolved_config = apply_ablation_config(deepcopy(fold_config))
    checkpoint_search_root = resolved_config["training"]["checkpoint_dir"]
    metric_csv_path = os.path.join(
        resolved_config.get("evaluation", {}).get("output_metric_dir", os.path.join(run_root, "metrics")),
        "latest_test_metric.csv",
    )
    pred_csv_path = os.path.join(
        resolved_config.get("evaluation", {}).get("output_pred_dir", os.path.join(run_root, "preds")),
        "latest_test.pred.csv",
    )
    return checkpoint_search_root, metric_csv_path, pred_csv_path


def is_fold_complete(checkpoint_search_root: str, metric_csv_path: str) -> bool:
    """判断某 fold 是否已完成（best_model.pt + latest_test_metric.csv 同时存在）。

    用于断点续跑：已完成的 fold 跳过训练，直接读取已有指标。
    任何异常情况（目录缺失、结构不完整）都返回 False，触发重训，保证安全。
    """
    if not os.path.isdir(checkpoint_search_root):
        return False
    if not os.path.isfile(metric_csv_path):
        return False
    # checkpoint_search_root 下应有含 best_model.pt 的 run 子目录
    try:
        run_dir = latest_subdir(checkpoint_search_root)
    except FileNotFoundError:
        return False
    return os.path.isfile(os.path.join(run_dir, "best_model.pt"))


def find_resumable_cv_root(ckpt_base: str) -> Optional[str]:
    """在 ckpt_base/5fold/ 下找最新的、可续跑的 cv_root（含已完成的 fold 但未生成 5fold_summary.csv）。

    续跑判定：cv_root 存在且未完成（无 5fold_summary.csv），且有至少一个 fold 目录。
    若最新 cv_root 已完成（有 5fold_summary.csv）则返回 None（应新建）。
    """
    fivefold_base = os.path.join(ckpt_base, "5fold")
    if not os.path.isdir(fivefold_base):
        return None
    subdirs = [os.path.join(fivefold_base, name) for name in os.listdir(fivefold_base)
               if os.path.isdir(os.path.join(fivefold_base, name))]
    if not subdirs:
        return None
    # 按修改时间倒序找第一个"未完成"的 cv_root
    for cv_root in sorted(subdirs, key=os.path.getmtime, reverse=True):
        summary = os.path.join(cv_root, "5fold_summary.csv")
        if os.path.isfile(summary):
            continue  # 已完成，跳过找更老的
        # 检查是否有 fold 目录（说明跑过但未完成）
        fold_dirs = [d for d in os.listdir(cv_root) if d.startswith("fold_")]
        if fold_dirs:
            return cv_root
    return None



def _stable_fold_seed(base_seed: int, fold_id: str) -> int:
    """fold seed 只依赖 (base_seed, fold_id)，与 fold 的运行顺序无关。"""
    digest = hashlib.sha1(f"{base_seed}:{fold_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


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
    parser.add_argument(
        "--resume_from", default=None,
        help="断点续跑：指定已有 cv_root（含 5fold/<timestamp>/ 结构）复用，跳过已完成 fold。"
             "设为 'auto' 自动在 ckpt_base/5fold/ 下找最新未完成的 cv_root。",
    )
    parser.add_argument(
        "--force_new", action="store_true",
        help="强制新建 cv_root（忽略 --resume_from 和自动续跑检测）。",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    base_seed = args.seed or base_config.get("experiment", {}).get("seed", 42)
    available_folds = discover_folds(args.fold_data_dir)
    fold_ids = args.folds or available_folds
    missing = [fold_id for fold_id in fold_ids if fold_id not in available_folds]
    if missing:
        raise ValueError(f"Requested folds not found: {missing}; available folds: {available_folds}")

    ckpt_base = base_config.get("training", {}).get("checkpoint_dir", "checkpoints/graph_transform")

    # 确定 cv_root：resume 优先，否则新建
    if args.force_new:
        cv_root = os.path.join(ckpt_base, "5fold", datetime.now().strftime("%Y%m%d_%H%M%S"))
        print(f"[5fold] force_new: 新建 cv_root = {cv_root}")
    elif args.resume_from == "auto":
        cv_root = find_resumable_cv_root(ckpt_base)
        if cv_root:
            print(f"[5fold] auto resume: 复用 cv_root = {cv_root}")
        else:
            cv_root = os.path.join(ckpt_base, "5fold", datetime.now().strftime("%Y%m%d_%H%M%S"))
            print(f"[5fold] auto resume: 无可续跑 cv_root，新建 = {cv_root}")
    elif args.resume_from:
        cv_root = args.resume_from
        if not os.path.isdir(cv_root):
            raise FileNotFoundError(f"--resume_from 指定的 cv_root 不存在: {cv_root}")
        print(f"[5fold] resume from: {cv_root}")
    else:
        cv_root = os.path.join(ckpt_base, "5fold", datetime.now().strftime("%Y%m%d_%H%M%S"))
        print(f"[5fold] 新建 cv_root = {cv_root}")
    os.makedirs(cv_root, exist_ok=True)

    train_script = os.path.join(os.path.dirname(__file__), "train_graph_model.py")
    results = []
    summary_metric_order = []
    overall_start = time.perf_counter()

    for fold_index, fold_id in enumerate(fold_ids):
        # P0: fold seed 绑定 fold_id（稳定）而非顺序索引——补跑子集或改变
        # --folds 顺序时，同一 fold 的 seed 不再漂移，保证可复现。
        fold_seed = _stable_fold_seed(base_seed, fold_id)
        fold_config, run_root, checkpoint_root = make_fold_config(base_config, fold_id, args.fold_data_dir, cv_root, fold_seed)
        checkpoint_search_root, metric_csv_path, pred_csv_path = resolve_fold_runtime_paths(
            fold_config,
            checkpoint_root,
            run_root,
        )
        os.makedirs(run_root, exist_ok=True)
        fold_config_path = os.path.join(run_root, "config.yaml")
        with open(fold_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(fold_config, f, sort_keys=False, allow_unicode=True)

        print(f"[5fold] start fold={fold_id} seed={fold_seed}")
        # 断点续跑：该 fold 已完成（best_model.pt + latest_test_metric.csv 都在）则跳过训练，直接读指标
        if is_fold_complete(checkpoint_search_root, metric_csv_path):
            run_dir = latest_subdir(checkpoint_search_root)
            print(f"[5fold] skip fold={fold_id} (已完成，复用 {run_dir})")
        else:
            subprocess.run(build_command(train_script, fold_config_path, args, fold_seed), check=True)
            run_dir = latest_subdir(checkpoint_search_root)

        best_epoch, best_metrics = load_best_metrics(os.path.join(run_dir, "best_model.pt"))
        test_metrics = load_metric_csv(metric_csv_path)
        fold_result = {
            "fold_id": fold_id,
            "seed": fold_seed,
            "best_epoch": best_epoch,
            "best_val_f1": best_metrics.get("f1"),
            "checkpoint_dir": run_dir,
            "metric_csv_path": metric_csv_path,
            "pred_csv_path": pred_csv_path,
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
