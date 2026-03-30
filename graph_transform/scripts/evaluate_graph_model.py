#!/usr/bin/env python3
"""
图神经网络评估脚本

参考 dbond 的评估流程，支持加载配置/模型权重、评估指标、输出预测与指标结果。
"""

import argparse
import logging
import os
import re
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GraphTransformer
from models.utils import ModelConfig, CheckpointManager
from data import GraphDataset, GraphDataLoader, CachedGraphDataset
from evaluation import Evaluator
from evaluation.metrics import metric_rows


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s[%(levelname)s]:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("graph_transform_eval")


def load_config(config_path: str, test_csv_path: str | None) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if test_csv_path:
        config["data"]["test_csv_path"] = test_csv_path
    return config


def setup_device(config: Dict[str, Any]) -> torch.device:
    device_config = config.get("device", {})
    if device_config.get("auto_detect", True):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_id = device_config.get("gpu_id", 0)
            torch.cuda.set_device(gpu_id)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device_type = device_config.get("device_type", "cpu")
        device = torch.device(device_type)
        if device_type == "cuda":
            gpu_id = device_config.get("gpu_id", 0)
            torch.cuda.set_device(gpu_id)
    return device


def sanitize_filename_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return sanitized or "unknown"


def build_evaluation_id(checkpoint_path: str, phase: str) -> str:
    checkpoint_stem = sanitize_filename_component(os.path.splitext(os.path.basename(checkpoint_path))[0])
    checkpoint_parent = sanitize_filename_component(os.path.basename(os.path.dirname(checkpoint_path)))
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{checkpoint_parent}_{checkpoint_stem}_{phase}"


def append_eval_id_to_path(filepath: str, evaluation_id: str) -> str:
    directory, filename = os.path.split(filepath)
    stem, ext = os.path.splitext(filename)
    return os.path.join(directory, f"{stem}__{evaluation_id}{ext}")


def save_evaluation_outputs(
    *,
    metrics: Dict[str, Any],
    output_df: pd.DataFrame,
    metric_csv_path: str,
    pred_csv_path: str,
    evaluation_id: str,
    logger: logging.Logger,
) -> Tuple[str, str, str, str]:
    metric_dir = os.path.dirname(metric_csv_path)
    pred_dir = os.path.dirname(pred_csv_path)
    if metric_dir:
        os.makedirs(metric_dir, exist_ok=True)
    if pred_dir:
        os.makedirs(pred_dir, exist_ok=True)

    metric_df = pd.DataFrame(metric_rows(metrics))
    metric_df.to_csv(metric_csv_path, index=False)
    output_df.to_csv(pred_csv_path, index=False)

    archive_metric_path = append_eval_id_to_path(metric_csv_path, evaluation_id)
    archive_pred_path = append_eval_id_to_path(pred_csv_path, evaluation_id)
    metric_df.to_csv(archive_metric_path, index=False)
    output_df.to_csv(archive_pred_path, index=False)

    logger.info(f"Saved latest metrics to {metric_csv_path}")
    logger.info(f"Saved latest predictions to {pred_csv_path}")
    logger.info(f"Archived metrics to {archive_metric_path}")
    logger.info(f"Archived predictions to {archive_pred_path}")
    return metric_csv_path, pred_csv_path, archive_metric_path, archive_pred_path


def main():
    parser = argparse.ArgumentParser(description="GraphTransformer evaluate")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="配置文件路径",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/graph_transform/best_model.pt",
        help="模型权重路径",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=None,
        help="测试集CSV路径（覆盖配置）",
    )
    parser.add_argument(
        "--out_pred_csv",
        type=str,
        default=None,
        help="预测结果输出路径",
    )
    parser.add_argument(
        "--out_metric_csv",
        type=str,
        default=None,
        help="评估指标输出路径",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="预测阈值（覆盖配置）",
    )
    args = parser.parse_args()

    logger = setup_logging()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    config = load_config(args.config, args.test_csv)
    threshold = args.threshold if args.threshold is not None else config.get("evaluation", {}).get("threshold", 0.5)
    evaluation_config = config.get("evaluation", {})

    device = setup_device(config)
    logger.info(f"Using device: {device}")

    model_config = ModelConfig(config["model"])
    model = GraphTransformer(model_config).to(device)
    CheckpointManager.load_checkpoint(args.checkpoint, model=model, device=device)

    data_config = config["data"]
    dataset_cls = CachedGraphDataset if data_config.get("cache_graphs", False) else GraphDataset
    test_kwargs = {
        "csv_path": data_config["test_csv_path"],
        "config": model_config,
        "max_seq_len": data_config["max_seq_len"],
        "graph_strategy": data_config["graph_strategy"],
        "augmentation": False,
        "split": "test",
    }
    if dataset_cls is CachedGraphDataset:
        test_kwargs.update({
            "cache_dir": data_config.get("cache_dir", "cache/graph_data"),
            "rebuild_cache": data_config.get("rebuild_cache", False),
            "cache_full_graphs": data_config.get("cache_full_graphs", False),
        })
    test_dataset = dataset_cls(**test_kwargs)
    test_loader = GraphDataLoader(
        dataset=test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True),
        drop_last=False,
    )

    evaluator = Evaluator(model=model, device=device, config=config, logger=logger)
    metrics = evaluator.evaluate(test_loader)
    prediction_outputs = evaluator.collect_prediction_outputs(test_loader, threshold=threshold)
    evaluation_id = build_evaluation_id(args.checkpoint, "test")
    out_metric_csv = args.out_metric_csv or os.path.join(
        evaluation_config.get("output_metric_dir", "result/metric/graph_transform"),
        "latest_metric.csv",
    )
    out_pred_csv = args.out_pred_csv or os.path.join(
        evaluation_config.get("output_pred_dir", "result/pred/graph_transform"),
        "latest.pred.csv",
    )
    output_df = test_dataset.data.copy()
    output_df["evaluation_id"] = evaluation_id
    output_df["checkpoint_path"] = os.path.abspath(args.checkpoint)
    output_df["threshold"] = prediction_outputs["threshold"]
    output_df["true"] = prediction_outputs["true_strings"]
    output_df["pred"] = prediction_outputs["pred_strings"]
    output_df["pred_prob"] = prediction_outputs["prob_strings"]
    save_evaluation_outputs(
        metrics=metrics,
        output_df=output_df,
        metric_csv_path=out_metric_csv,
        pred_csv_path=out_pred_csv,
        evaluation_id=evaluation_id,
        logger=logger,
    )


if __name__ == "__main__":
    main()
