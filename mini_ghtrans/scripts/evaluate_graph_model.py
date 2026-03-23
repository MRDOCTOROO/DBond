#!/usr/bin/env python3
"""
图神经网络评估脚本

参考 dbond 的评估流程，支持加载配置/模型权重、评估指标、输出预测与指标结果。
"""

import argparse
import logging
import os
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


def move_batch_to_device(batch_data: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    device_batch = {}
    for key, value in batch_data.items():
        if isinstance(value, torch.Tensor):
            device_batch[key] = value.to(device, non_blocking=True)
        else:
            device_batch[key] = value
    return device_batch


def collect_predictions(
    model: torch.nn.Module,
    data_loader: GraphDataLoader,
    device: torch.device,
    threshold: float,
    use_amp: bool,
) -> Tuple[List[str], List[str]]:
    model.eval()
    pred_strings: List[str] = []
    true_strings: List[str] = []
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"

    with torch.no_grad():
        for batch_data in data_loader:
            batch_data = move_batch_to_device(batch_data, device)

            if use_amp and device.type == "cuda":
                with torch.amp.autocast(amp_device_type, enabled=True):
                    logits = model(batch_data)
            else:
                logits = model(batch_data)

            probs = torch.sigmoid(logits)
            labels = batch_data["labels"]
            seq_lens = batch_data["seq_lens"].tolist()

            for i, seq_len in enumerate(seq_lens):
                bond_len = max(seq_len - 1, 0)
                if bond_len == 0:
                    pred_strings.append("")
                    true_strings.append("")
                    continue

                pred_vals = (probs[i, :bond_len] >= threshold).long().cpu().numpy()
                true_vals = labels[i, :bond_len].long().cpu().numpy()

                pred_strings.append(";".join(map(str, pred_vals.tolist())))
                true_strings.append(";".join(map(str, true_vals.tolist())))

    return pred_strings, true_strings


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
        default="result/pred/graph_transform/pred.csv",
        help="预测结果输出路径",
    )
    parser.add_argument(
        "--out_metric_csv",
        type=str,
        default="result/metric/graph_transform/metric.csv",
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

    pred_strings, true_strings = collect_predictions(
        model=model,
        data_loader=test_loader,
        device=device,
        threshold=threshold,
        use_amp=config.get("device", {}).get("use_amp", False),
    )

    os.makedirs(os.path.dirname(args.out_metric_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_pred_csv), exist_ok=True)

    metric_df = pd.DataFrame(list(metrics.items()), columns=["metric", "value"])
    metric_df.to_csv(args.out_metric_csv, index=False)
    logger.info(f"Saved metrics to {args.out_metric_csv}")

    output_df = test_dataset.data.copy()
    output_df["true"] = true_strings
    output_df["pred"] = pred_strings
    output_df.to_csv(args.out_pred_csv, index=False)
    logger.info(f"Saved predictions to {args.out_pred_csv}")


if __name__ == "__main__":
    main()
