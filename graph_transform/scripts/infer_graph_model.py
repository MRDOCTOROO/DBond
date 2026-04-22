#!/usr/bin/env python3
"""
GraphTransformer 推理脚本。

支持以下输入方式，且不需要修改原始 CSV 结构：
1. 整个 CSV 文件推理
2. 指定若干行推理（行号从 1 开始，不含表头）
3. 指定连续行区间推理
4. 取前 N 行推理

输出结果会保留原始列，并追加预测列，便于直接查看或下游分析。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GraphTransformer
from models.utils import build_model_config, CheckpointManager
from data import GraphDataset, GraphDataLoader, CachedGraphDataset
from evaluation import Evaluator
from evaluation.metrics import metric_rows


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s[%(levelname)s]:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("graph_transform_infer")


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def parse_rows_arg(rows_text: str) -> List[int]:
    rows: List[int] = []
    seen = set()
    for part in rows_text.split(","):
        value = part.strip()
        if not value:
            continue
        row_num = int(value)
        if row_num <= 0:
            raise ValueError(f"Row numbers must be >= 1, got {row_num}")
        if row_num not in seen:
            rows.append(row_num)
            seen.add(row_num)
    if not rows:
        raise ValueError("--rows did not contain any valid row numbers")
    return rows


def select_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    total_rows = len(df)

    if args.rows:
        selected_row_numbers = parse_rows_arg(args.rows)
        missing = [row_num for row_num in selected_row_numbers if row_num > total_rows]
        if missing:
            raise IndexError(f"Requested rows exceed CSV length {total_rows}: {missing}")
        selected = df.iloc[[row_num - 1 for row_num in selected_row_numbers]].copy()
    elif args.start_row is not None or args.end_row is not None:
        start_row = args.start_row if args.start_row is not None else 1
        end_row = args.end_row if args.end_row is not None else total_rows
        if start_row <= 0 or end_row <= 0:
            raise ValueError("start_row/end_row must be >= 1")
        if start_row > end_row:
            raise ValueError(f"start_row ({start_row}) must be <= end_row ({end_row})")
        if start_row > total_rows:
            raise IndexError(f"start_row {start_row} exceeds CSV length {total_rows}")
        end_row = min(end_row, total_rows)
        selected = df.iloc[start_row - 1:end_row].copy()
    elif args.head is not None:
        if args.head <= 0:
            raise ValueError("--head must be > 0")
        selected = df.head(args.head).copy()
    else:
        selected = df.copy()

    selected.insert(0, "source_row_number", selected.index + 1)
    selected.insert(1, "source_row_index", selected.index)
    return selected.reset_index(drop=True)


def _positive_positions(binary_text: Any) -> str:
    if pd.isna(binary_text):
        return ""
    tokens = [token.strip() for token in str(binary_text).split(";") if token.strip() != ""]
    positions = [str(idx) for idx, token in enumerate(tokens, start=1) if token == "1"]
    return ";".join(positions)


def build_output_dataframe(
    selected_df: pd.DataFrame,
    prediction_outputs: Dict[str, Any],
    checkpoint_path: str,
) -> pd.DataFrame:
    output_df = selected_df.copy()
    output_df["checkpoint_path"] = os.path.abspath(checkpoint_path)
    output_df["threshold"] = prediction_outputs["threshold"]
    output_df["pred"] = prediction_outputs["pred_strings"]
    output_df["pred_prob"] = prediction_outputs["prob_strings"]

    if "true_multi" in output_df.columns:
        output_df["true"] = prediction_outputs["true_strings"]
        output_df["true_positive_positions"] = output_df["true"].apply(_positive_positions)

    output_df["pred_positive_positions"] = output_df["pred"].apply(_positive_positions)
    output_df["seq_len"] = output_df["seq"].astype(str).str.len() if "seq" in output_df.columns else None
    output_df["bond_len"] = output_df["seq_len"].fillna(0).astype(int).clip(lower=0) - 1
    output_df["bond_len"] = output_df["bond_len"].clip(lower=0)
    return output_df


def print_preview(output_df: pd.DataFrame, print_limit: int, logger: logging.Logger) -> None:
    if output_df.empty:
        logger.warning("No rows were selected for inference.")
        return

    preview_count = min(len(output_df), max(print_limit, 0))
    logger.info("Inference completed for %s row(s). Previewing %s row(s):", len(output_df), preview_count)

    for _, row in output_df.head(preview_count).iterrows():
        parts = [
            f"row={int(row['source_row_number'])}",
        ]
        if "name" in row and not pd.isna(row["name"]):
            parts.append(f"name={row['name']}")
        if "seq_len" in row and not pd.isna(row["seq_len"]):
            parts.append(f"seq_len={int(row['seq_len'])}")
        parts.append(f"threshold={float(row['threshold']):.4f}")
        parts.append(f"pred_pos={row.get('pred_positive_positions', '')}")
        logger.info(" | ".join(parts))
        logger.info("  pred      = %s", row.get("pred", ""))
        logger.info("  pred_prob = %s", row.get("pred_prob", ""))
        if "true" in row:
            logger.info("  true      = %s", row.get("true", ""))


def default_prediction_output_path(input_csv: str) -> str:
    stem = os.path.splitext(os.path.basename(input_csv))[0]
    return os.path.join("result", "pred", "graph_transform", f"{stem}.infer.pred.csv")


def default_metric_output_path(input_csv: str) -> str:
    stem = os.path.splitext(os.path.basename(input_csv))[0]
    return os.path.join("result", "metric", "graph_transform", f"{stem}.infer.metric.csv")


def maybe_save_metrics(metrics: Optional[Dict[str, Any]], metric_csv_path: Optional[str], logger: logging.Logger) -> None:
    if metrics is None or metric_csv_path is None:
        return
    metric_dir = os.path.dirname(metric_csv_path)
    if metric_dir:
        os.makedirs(metric_dir, exist_ok=True)
    pd.DataFrame(metric_rows(metrics)).to_csv(metric_csv_path, index=False)
    logger.info("Saved metrics to %s", metric_csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphTransformer CSV 推理脚本")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--input_csv", type=str, required=True, help="待推理的原始 CSV 路径")
    parser.add_argument("--out_pred_csv", type=str, default=None, help="预测结果输出路径")
    parser.add_argument("--out_metric_csv", type=str, default=None, help="指标输出路径")
    parser.add_argument("--threshold", type=float, default=None, help="预测阈值，默认取配置文件中的 evaluation.threshold")
    parser.add_argument("--batch_size", type=int, default=None, help="覆盖推理 batch size")
    parser.add_argument("--num_workers", type=int, default=None, help="覆盖 DataLoader num_workers")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default=None, help="覆盖设备")
    parser.add_argument("--head", type=int, default=None, help="只推理前 N 行")
    parser.add_argument("--rows", type=str, default=None, help="指定行号，逗号分隔，行号从 1 开始，如 1,5,10")
    parser.add_argument("--start_row", type=int, default=None, help="起始行号（含），从 1 开始")
    parser.add_argument("--end_row", type=int, default=None, help="结束行号（含），从 1 开始")
    parser.add_argument("--with_metrics", action="store_true", help="同时计算并保存指标")
    parser.add_argument("--print_limit", type=int, default=20, help="终端最多打印多少条结果预览")
    args = parser.parse_args()

    logger = setup_logging()

    for path_value, label in (
        (args.config, "Config"),
        (args.checkpoint, "Checkpoint"),
        (args.input_csv, "Input CSV"),
    ):
        if not os.path.exists(path_value):
            raise FileNotFoundError(f"{label} not found: {path_value}")

    if args.rows and (args.start_row is not None or args.end_row is not None or args.head is not None):
        raise ValueError("--rows cannot be combined with --start_row/--end_row/--head")
    if args.head is not None and (args.start_row is not None or args.end_row is not None):
        raise ValueError("--head cannot be combined with --start_row/--end_row")

    config = load_config(args.config)
    if args.device:
        config.setdefault("device", {})["auto_detect"] = False
        config["device"]["device_type"] = args.device
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config.setdefault("data", {})["num_workers"] = args.num_workers

    # 推理时没有必要构建完整图缓存，保留边缓存即可。
    config.setdefault("data", {})["cache_full_graphs"] = False

    input_df = pd.read_csv(args.input_csv)
    selected_df = select_rows(input_df, args)
    if selected_df.empty:
        raise ValueError("The selected rows are empty.")

    max_seq_len = config.get("data", {}).get("max_seq_len")
    if max_seq_len is not None and "seq" in selected_df.columns:
        too_long_mask = selected_df["seq"].astype(str).str.len() > int(max_seq_len)
        if bool(too_long_mask.any()):
            dropped_rows = selected_df.loc[too_long_mask, "source_row_number"].astype(int).tolist()
            logger.warning(
                "Rows %s exceed max_seq_len=%s and will be filtered by the dataset loader.",
                dropped_rows,
                max_seq_len,
            )

    out_pred_csv = args.out_pred_csv or default_prediction_output_path(args.input_csv)
    out_metric_csv = args.out_metric_csv or default_metric_output_path(args.input_csv)
    pred_dir = os.path.dirname(out_pred_csv)
    if pred_dir:
        os.makedirs(pred_dir, exist_ok=True)

    temp_csv_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="") as temp_file:
            temp_csv_path = temp_file.name
        selected_df.to_csv(temp_csv_path, index=False)
        logger.info("Selected %s row(s) from %s", len(selected_df), args.input_csv)
        logger.info("Temporary inference CSV: %s", temp_csv_path)

        config["data"]["test_csv_path"] = temp_csv_path

        device = setup_device(config)
        logger.info("Using device: %s", device)

        model_config = build_model_config(config)
        model = GraphTransformer(model_config).to(device)
        CheckpointManager.load_checkpoint(args.checkpoint, model=model, device=device)
        logger.info("Loaded checkpoint: %s", args.checkpoint)

        data_config = config["data"]
        dataset_cls = CachedGraphDataset if data_config.get("cache_graphs", False) else GraphDataset
        dataset_kwargs = {
            "csv_path": data_config["test_csv_path"],
            "config": model_config,
            "max_seq_len": data_config["max_seq_len"],
            "graph_strategy": data_config["graph_strategy"],
            "augmentation": False,
            "split": "test",
        }
        if dataset_cls is CachedGraphDataset:
            dataset_kwargs.update({
                "cache_dir": data_config.get("cache_dir", "cache/graph_data"),
                "rebuild_cache": data_config.get("rebuild_cache", False),
                "cache_full_graphs": data_config.get("cache_full_graphs", False),
            })

        test_dataset = dataset_cls(**dataset_kwargs)
        if len(test_dataset) == 0:
            raise ValueError("No rows remain after dataset filtering; check max_seq_len and input contents.")

        test_loader = GraphDataLoader(
            dataset=test_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=data_config.get("num_workers", 4),
            pin_memory=data_config.get("pin_memory", True),
            drop_last=False,
        )

        evaluator = Evaluator(model=model, device=device, config=config, logger=logger)
        metrics = evaluator.evaluate(test_loader) if args.with_metrics else None
        prediction_outputs = evaluator.collect_prediction_outputs(test_loader, threshold=args.threshold)

        output_df = build_output_dataframe(
            selected_df=test_dataset.data.copy(),
            prediction_outputs=prediction_outputs,
            checkpoint_path=args.checkpoint,
        )
        output_df.to_csv(out_pred_csv, index=False)
        logger.info("Saved predictions to %s", out_pred_csv)

        maybe_save_metrics(metrics, out_metric_csv if args.with_metrics else None, logger)
        print_preview(output_df, args.print_limit, logger)
    finally:
        if temp_csv_path and os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)


if __name__ == "__main__":
    main()
