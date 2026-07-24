#!/usr/bin/env python3
"""
残基对化学矩阵分析（Paper-grade baseline）

研究问题：不同残基对 (X-Y) 的内在断裂倾向是多少？模型预测是否与之一致？
模型在哪里高估 / 低估？

输出：
  - residue_pair_matrix.svg    3 子图：经验断裂率 / 模型预测 / 差异
  - residue_pair_stats.csv     完整数据表
  - residue_pair_summary.json  关键统计

使用方式：
  # 方式 1：直接推理（云端）
  python graph_transform/scripts/residue_pair_analysis.py \
      --config graph_transform/config/default.yaml \
      --checkpoint <ckpt.pt> \
      --input_csv dataset/5fold/6072.test.fbr.multi.csv \
      --output_dir results/residue_pair_matrix \
      --infer_config

  # 方式 2：复用已落盘的预测 CSV（result/pred/.../latest.pred.csv）
  python graph_transform/scripts/residue_pair_analysis.py \
      --config graph_transform/config/default.yaml \
      --input_csv dataset/5fold/6072.test.fbr.multi.csv \
      --pred_csv result/pred/graph_transform/latest.pred.csv \
      --output_dir results/residue_pair_matrix
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GraphTransformer
from models.utils import build_model_config, CheckpointManager
from data import GraphDataset, CachedGraphDataset, GraphDataLoader
from utils.visualization import plot_residue_pair_matrix

# 默认字母表（与 graph_transform/config/default.yaml 一致）
DEFAULT_ALPHABET = "#ABCDEFGHIKLMNOPQRSTVWXYZ"  # 25 chars (含 # padding)


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s[%(levelname)s]:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("residue_pair")


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_device(config: Dict[str, Any]) -> torch.device:
    device_config = config.get("device", {})
    if device_config.get("auto_detect", True):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.cuda.set_device(device_config.get("gpu_id", 0))
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device_type = device_config.get("device_type", "cpu")
        device = torch.device(device_type)
        if device_type == "cuda":
            torch.cuda.set_device(device_config.get("gpu_id", 0))
    return device


def infer_model_config_from_checkpoint(
    checkpoint_path: str, base_config: Dict[str, Any]
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    inferred = base_config.copy()
    model_cfg = inferred.setdefault("model", {})

    if "bond_head.0.weight" in state_dict:
        bond_w = state_dict["bond_head.0.weight"].shape
        bond_feature_dim, hidden_dim = bond_w[1], bond_w[0]
        logger = logging.getLogger("residue_pair")
        logger.info(
            f"Inferred: hidden_dim={hidden_dim}, bond_feature_dim={bond_feature_dim}"
        )
        model_cfg["hidden_dim"] = hidden_dim
        remaining = bond_feature_dim - hidden_dim * 2
        model_cfg["bond_use_edge_repr"] = remaining >= hidden_dim * 3
        model_cfg["bond_use_diff_feature"] = remaining >= hidden_dim * 2
        model_cfg["bond_use_product_feature"] = remaining >= hidden_dim * 1
    if "node_encoder.aa_embedding.weight" in state_dict:
        vocab = state_dict["node_encoder.aa_embedding.weight"].shape[0]
        logging.getLogger("residue_pair").info(f"Inferred vocab_size={vocab}")
    return inferred


# =============================================================================
# 预测获取：推理 or 复用落盘
# =============================================================================

def run_inference(
    model: GraphTransformer,
    dataset: GraphDataset,
    device: torch.device,
    batch_size: int,
    logger: logging.Logger,
) -> Tuple[List[str], List[str], List[str]]:
    """跑完整数据集推理，返回 (sequences, true_strings, prob_strings)。

    每个样本的 prob_strings[i] = "p0;p1;...;p_{L-2}" 分号分隔的逐键概率。
    """
    model.eval()
    loader = GraphDataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda"), drop_last=False,
    )

    sequences: List[str] = []
    true_strings: List[str] = []
    prob_strings: List[str] = []
    sigmoid = torch.nn.Sigmoid()

    n_done = 0
    with torch.no_grad():
        for batch in loader:
            # Move tensors to device
            batch_dev = _move_batch_to_device(batch, device)
            logits = model(batch_dev)  # [B, max_bonds]
            probs = sigmoid(logits)

            label_mask = batch_dev["label_mask"]            # [B, max_bonds]
            seq_lens = batch_dev["seq_lens"].cpu().tolist()
            labels = batch_dev["labels"].cpu().numpy()
            probs_np = probs.cpu().numpy()

            for b in range(len(seq_lens)):
                L = int(seq_lens[b])
                num_bonds = max(L - 1, 0)
                valid = label_mask[b, :num_bonds].cpu().numpy().astype(bool)
                p_str = ";".join(
                    f"{probs_np[b, i]:.6f}" for i in range(num_bonds) if valid[i]
                )
                t_str = ";".join(
                    f"{int(labels[b, i])}" for i in range(num_bonds) if valid[i]
                )
                prob_strings.append(p_str)
                true_strings.append(t_str)
                sequences.append(batch["sequences"][b])
                n_done += 1

            if n_done % (batch_size * 20) == 0:
                logger.info(f"  inference progress: {n_done} samples")

    logger.info(f"Inference done: {n_done} samples")
    return sequences, true_strings, prob_strings


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = dict(batch)
    tensor_keys = [
        "edge_index", "edge_attr", "edge_types", "edge_distances",
        "bond_edge_map", "labels", "label_mask",
        "state_vars", "env_vars", "seq_lens", "node_lens",
    ]
    for k in tensor_keys:
        if k in out and isinstance(out[k], torch.Tensor):
            out[k] = out[k].to(device)
    return out


def load_predictions_from_csv(
    pred_csv: str, input_csv: str, logger: logging.Logger
) -> Tuple[List[str], List[str], List[str]]:
    """从已落盘的预测 CSV 加载（与 evaluate_graph_model.py 输出格式兼容）。

    预期列：seq, true, pred_prob（分号分隔字符串）。
    若 pred_csv 没有 seq 列，则按行序与 input_csv 对齐。
    """
    pred_df = pd.read_csv(pred_csv)
    input_df = pd.read_csv(input_csv)
    logger.info(
        f"Loaded pred_csv: {len(pred_df)} rows; input_csv: {len(input_df)} rows"
    )

    # ---- 容错：检测错误的 CSV 类型 ----
    cols = set(pred_df.columns)
    # 1) 指标文件（metric,value 两列）
    if cols <= {"metric", "value"} or (
        len(cols) == 2 and "metric" in cols and "value" in cols
    ):
        raise ValueError(
            f"\n  [ERROR] '{pred_csv}' looks like a METRIC file "
            f"(columns: metric, value), not a per-bond prediction file.\n"
            f"  Predicted CSVs have columns: seq, true, pred, pred_prob "
            f"(one row per sample, semicolon-separated per-bond values).\n\n"
            f"  Searched locations for the prediction file:\n"
            f"    - Same parent dir, 'preds/latest_test.pred.csv'\n"
            f"    - Same parent dir, 'latest.pred.csv'\n"
            f"    - result/pred/graph_transform/latest.pred.csv\n\n"
            f"  Tip: re-run evaluate_graph_model.py with the same checkpoint to "
            f"regenerate the prediction CSV, OR drop --pred_csv and pass "
            f"--checkpoint to run inference directly."
        )
    # 2) 缺少关键列
    prob_col_candidates = [c for c in ("pred_prob", "prob", "probabilities") if c in cols]
    if not prob_col_candidates:
        # 尝试自动定位真正的预测文件
        sibling = _auto_locate_pred_csv(pred_csv, logger)
        if sibling is not None:
            logger.info(f"Auto-located actual prediction CSV: {sibling}")
            return load_predictions_from_csv(sibling, input_csv, logger)
        raise ValueError(
            f"\n  [ERROR] '{pred_csv}' does not contain a prediction probability "
            f"column (expected one of: pred_prob, prob, probabilities).\n"
            f"  Found columns: {sorted(cols)}\n"
            f"  Please provide the path to the per-sample prediction CSV "
            f"(produced by evaluate_graph_model.py)."
        )

    if len(pred_df) != len(input_df):
        logger.warning(
            f"Row count mismatch (pred={len(pred_df)}, input={len(input_df)}); "
            f"using min={min(len(pred_df), len(input_df))}"
        )
        n = min(len(pred_df), len(input_df))
        pred_df = pred_df.iloc[:n].reset_index(drop=True)
        input_df = input_df.iloc[:n].reset_index(drop=True)

    # seq 列：优先用 pred_csv 的，否则从 input_csv 对齐
    if "seq" not in pred_df.columns:
        pred_df["seq"] = input_df["seq"].values

    # true 列：优先用 pred_csv 的，否则从 input_csv.true_multi
    if "true" in pred_df.columns:
        true_strings = pred_df["true"].astype(str).tolist()
    else:
        if "true_multi" not in input_df.columns:
            raise ValueError(
                f"Neither pred_csv nor input_csv has true labels "
                f"(looked for 'true' in pred_csv, 'true_multi' in input_csv)."
            )
        logger.info("pred_csv lacks 'true' column, falling back to input_csv.true_multi")
        true_strings = input_df["true_multi"].astype(str).tolist()

    sequences = pred_df["seq"].astype(str).tolist()
    prob_strings = pred_df[prob_col_candidates[0]].astype(str).tolist()
    return sequences, true_strings, prob_strings


def _auto_locate_pred_csv(wrong_path: str, logger: logging.Logger) -> Optional[str]:
    """根据用户提供的错误路径（如 metric CSV）尝试定位真正的预测 CSV。

    检查兄弟目录下的常见预测文件名。
    """
    p = os.path.abspath(wrong_path)
    parent = os.path.dirname(p)
    grandparent = os.path.dirname(parent)
    candidates = [
        os.path.join(parent, "latest_test.pred.csv"),
        os.path.join(parent, "latest.pred.csv"),
        os.path.join(parent, "..", "preds", "latest_test.pred.csv"),
        os.path.join(grandparent, "preds", "latest_test.pred.csv"),
        os.path.join(grandparent, "latest_test.pred.csv"),
    ]
    for c in candidates:
        c_norm = os.path.normpath(c)
        if os.path.isfile(c_norm) and c_norm != os.path.normpath(p):
            return c_norm
    return None


# =============================================================================
# 残基对矩阵聚合
# =============================================================================

def aggregate_residue_pair_stats(
    sequences: List[str],
    true_strings: List[str],
    prob_strings: List[str],
    alphabet: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """对每个键聚合 (X, Y, true_label, pred_prob)，返回残基对统计 DataFrame。

    输出列：X, Y, N, n_broken, empirical, predicted, diff
    """
    aa_chars = [c for c in alphabet if c != "#"]
    aa_to_idx = {aa: i for i, aa in enumerate(aa_chars)}
    n_aa = len(aa_chars)

    counts = np.zeros((n_aa, n_aa), dtype=np.int64)
    n_broken = np.zeros((n_aa, n_aa), dtype=np.int64)
    sum_prob = np.zeros((n_aa, n_aa), dtype=np.float64)

    skipped = 0
    processed = 0
    for seq, t_str, p_str in zip(sequences, true_strings, prob_strings):
        seq = str(seq)
        L = len(seq)
        if L < 2:
            skipped += 1
            continue
        try:
            true_labels = [int(x) for x in str(t_str).split(";") if x.strip() != ""]
            pred_probs = [float(x) for x in str(p_str).split(";") if x.strip() != ""]
        except ValueError:
            skipped += 1
            continue

        n_bonds = L - 1
        if len(true_labels) < n_bonds or len(pred_probs) < n_bonds:
            # 长度不一致，跳过该样本（标签可能被截断）
            skipped += 1
            continue

        for i in range(n_bonds):
            x_aa = seq[i]
            y_aa = seq[i + 1]
            if x_aa not in aa_to_idx or y_aa not in aa_to_idx:
                continue  # 跳过未知字符（理论不应出现）
            xi, yi = aa_to_idx[x_aa], aa_to_idx[y_aa]
            counts[xi, yi] += 1
            n_broken[xi, yi] += int(true_labels[i])
            sum_prob[xi, yi] += float(pred_probs[i])
            processed += 1

    logger.info(
        f"Aggregated {processed} bonds across {len(sequences)} samples "
        f"(skipped {skipped})"
    )

    rows = []
    for xi, x_aa in enumerate(aa_chars):
        for yi, y_aa in enumerate(aa_chars):
            n = int(counts[xi, yi])
            if n == 0:
                emp = pred = diff = float("nan")
            else:
                emp = float(n_broken[xi, yi]) / n
                pred = float(sum_prob[xi, yi]) / n
                diff = pred - emp
            rows.append({
                "X": x_aa,
                "Y": y_aa,
                "N": n,
                "n_broken": int(n_broken[xi, yi]),
                "empirical": emp,
                "predicted": pred,
                "diff": diff,
            })

    return pd.DataFrame(rows), aa_chars


def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Residue-pair cleavage chemistry matrix analysis."
    )
    parser.add_argument("--config", type=str, required=True, help="YAML 配置路径")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型 ckpt 路径")
    parser.add_argument("--input_csv", type=str, required=True, help="测试集 CSV 路径")
    parser.add_argument(
        "--pred_csv", type=str, default=None,
        help="已落盘的预测 CSV（含 seq/true/pred_prob 列）。提供此参数时跳过推理。",
    )
    parser.add_argument("--output_dir", type=str,
                        default="results/residue_pair_matrix")
    parser.add_argument("--infer_config", action="store_true",
                        help="从 ckpt 反推模型配置（hidden_dim / bond_feature_dim）")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="推理 batch_size（仅当 --pred_csv 未提供时生效）")
    parser.add_argument("--max_seq_len", type=int, default=32)
    parser.add_argument("--figure_format", type=str, default="svg",
                        choices=["svg", "png"])
    parser.add_argument("--alphabet", type=str, default=DEFAULT_ALPHABET,
                        help="字母表（默认与 default.yaml 一致）")
    parser.add_argument("--filter_empty", action="store_true",
                        help="过滤掉数据中完全不存在的 AA 行/列 "
                             "(如本项目测试集缺 C/M/V/W/Z)。输出图更紧凑。")
    parser.add_argument("--min_total_n", type=int, default=1,
                        help="filter_empty 时保留 AA 的最低总样本数阈值")
    args = parser.parse_args()

    img_ext = ".svg" if args.figure_format == "svg" else ".png"
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Residue-Pair Chemistry Matrix Analysis")
    logger.info("=" * 60)

    # ---- 1. 获取预测 ----
    if args.pred_csv is not None:
        logger.info(f"Loading predictions from {args.pred_csv}")
        sequences, true_strings, prob_strings = load_predictions_from_csv(
            args.pred_csv, args.input_csv, logger,
        )
    else:
        if args.checkpoint is None:
            raise ValueError(
                "Either --pred_csv or --checkpoint must be provided."
            )
        for p, label in [(args.config, "Config"), (args.checkpoint, "Checkpoint"),
                         (args.input_csv, "Input CSV")]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"{label} not found: {p}")

        config = load_config(args.config)
        if args.infer_config:
            config = infer_model_config_from_checkpoint(args.checkpoint, config)
        device = setup_device(config)
        logger.info(f"Using device: {device}")

        model_config = build_model_config(config)
        model = GraphTransformer(model_config).to(device)
        CheckpointManager.load_checkpoint(args.checkpoint, model=model, device=device)
        logger.info(f"Loaded checkpoint: {args.checkpoint}")

        data_config = config["data"]
        data_config["test_csv_path"] = args.input_csv
        data_config["max_seq_len"] = args.max_seq_len

        dataset_cls = CachedGraphDataset if data_config.get("cache_graphs", False) else GraphDataset
        dataset_kwargs = {
            "csv_path": data_config["test_csv_path"],
            "config": model_config,
            "max_seq_len": data_config.get("max_seq_len"),
            "graph_strategy": data_config.get("graph_strategy", "distance"),
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
        logger.info(f"Loaded dataset: {len(test_dataset)} samples")

        sequences, true_strings, prob_strings = run_inference(
            model, test_dataset, device, args.batch_size, logger,
        )

    # ---- 2. 聚合残基对统计 ----
    stats_df, aa_chars = aggregate_residue_pair_stats(
        sequences, true_strings, prob_strings, args.alphabet, logger,
    )

    n_aa = len(aa_chars)
    empirical_mat = stats_df["empirical"].to_numpy().reshape(n_aa, n_aa)
    predicted_mat = stats_df["predicted"].to_numpy().reshape(n_aa, n_aa)
    counts_mat = stats_df["N"].to_numpy(dtype=np.int64).reshape(n_aa, n_aa)

    # NaN -> 0 for visualization (kept as nan in CSV)
    empirical_viz = np.nan_to_num(empirical_mat, nan=0.0)
    predicted_viz = np.nan_to_num(predicted_mat, nan=0.0)

    # ---- 3. 绘图 ----
    fig_path = os.path.join(args.output_dir, f"residue_pair_matrix{img_ext}")
    plot_residue_pair_matrix(
        empirical=empirical_viz,
        predicted=predicted_viz,
        counts=counts_mat,
        aa_labels=aa_chars,
        save_path=fig_path,
        min_n_for_label=50,
        rare_thresholds=(10, 50),
        filter_empty=args.filter_empty,
        min_total_n=args.min_total_n,
    )
    logger.info(f"Saved figure: {fig_path}")
    if args.filter_empty:
        logger.info("  (filter_empty=True: 已过滤数据中不存在的 AA 行/列)")

    # ---- 4. 落盘 CSV + JSON 摘要 ----
    csv_path = os.path.join(args.output_dir, "residue_pair_stats.csv")
    stats_df.to_csv(csv_path, index=False)
    logger.info(f"Saved stats CSV: {csv_path}")

    # 计算 Wilson CI for cells with N >= 50
    reliable = stats_df[stats_df["N"] >= 50].copy()
    if len(reliable) > 0:
        ci = reliable.apply(
            lambda r: wilson_ci(r["empirical"], int(r["N"])), axis=1,
            result_type="expand",
        )
        reliable["empirical_ci_low"] = ci[0]
        reliable["empirical_ci_high"] = ci[1]
        reliable["abs_diff"] = reliable["diff"].abs()
        reliable_sorted = reliable.sort_values("abs_diff", ascending=False)
    else:
        reliable_sorted = reliable

    # 按预测残差 diff = predicted - empirical 的符号拆分高估/低估，
    # 各自取 |diff| 最大的 10 个对。
    # 注：早期版本用 reliable_sorted.head/tail(10)，但 reliable_sorted 是
    # 按 abs_diff 降序排列的，head 取的是“绝对误差最大”（不分方向），
    # tail 取的是“绝对误差最小”（拟合最好），与字段名含义不符。此处修正。
    summary_cols = ["X", "Y", "N", "empirical", "predicted", "diff"]
    reliable_over = (
        reliable[reliable["diff"] > 0]
        .sort_values("abs_diff", ascending=False)
        .head(10)
    )
    reliable_under = (
        reliable[reliable["diff"] < 0]
        .sort_values("abs_diff", ascending=False)
        .head(10)
    )

    # 全局指标
    valid_mask = ~np.isnan(empirical_mat)
    if valid_mask.sum() > 0:
        global_corr = float(np.corrcoef(
            empirical_mat[valid_mask], predicted_mat[valid_mask]
        )[0, 1])
        mae = float(np.mean(np.abs(
            predicted_mat[valid_mask] - empirical_mat[valid_mask]
        )))
    else:
        global_corr = float("nan")
        mae = float("nan")

    summary = {
        "total_samples": len(sequences),
        "total_residue_pairs_filled": int((counts_mat > 0).sum()),
        "total_residue_pairs_possible": int(counts_mat.size),
        "coverage": float((counts_mat > 0).sum() / counts_mat.size),
        "reliable_pairs_n_ge_50": int((counts_mat >= 50).sum()),
        "global_pearson_r_empirical_vs_predicted": global_corr,
        "global_mae_predicted_vs_empirical": mae,
        "top_overestimates": (
            reliable_over[summary_cols].to_dict(orient="records")
            if len(reliable) > 0 else []
        ),
        "top_underestimates": (
            reliable_under[summary_cols].to_dict(orient="records")
            if len(reliable) > 0 else []
        ),
        "best_fit_pairs": (
            reliable_sorted.tail(10)[summary_cols].to_dict(orient="records")
            if len(reliable) > 0 else []
        ),
    }
    json_path = os.path.join(args.output_dir, "residue_pair_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved summary JSON: {json_path}")
    logger.info(f"Global Pearson r (empirical vs predicted) = {global_corr:+.4f}")
    logger.info(f"Global MAE = {mae:.4f}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
