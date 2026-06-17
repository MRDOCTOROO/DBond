#!/usr/bin/env python3
"""
Attention 交叉验证分析（Method 5/6/7）

目的：证明（或证伪）模型 attention 确实承载了 functional meaning。
针对 GCN 修复后的可解释性结果做三层独立交叉验证：

  Method 5 - Attention Rank Correlation：
      逐样本计算 L0 与 L1 bond-level attention 的 Spearman ρ。
      ρ≈1：浅层 pattern 被深层保留（仅平滑）
      ρ≈0：深层完全重构 attention（migration）

  Method 6 - Attention Rollout (Abnar & Zuidema 2020)：
      A_rollout = (A_0 + I) @ (A_1 + I) @ ... @ (A_L + I)
      反映输入对最终输出的累积影响。
      验证 rollout 是聚焦还是扩散，以及最像哪一层。

  Method 7 - Attention–Occlusion Correlation：
      逐层计算 attention 与 occlusion 敏感度的 Pearson r / Spearman ρ。
      高相关 → 该层 attention ≈ functional importance
      低相关 → 该层 attention ≈ information mixing

输出（默认 results/cross_validation/）：
  - method5_rank_correlation.svg       层间 rank 相关性
  - method6_attention_rollout.svg      rollout 与各层对比
  - method7_attention_occlusion.svg    每层 attention ↔ occlusion 相关性
  - cross_validation_summary.json      所有数值指标（论文制表用）
  - cross_validation_per_sample.json   每样本原始 ρ/r 值

使用方式：
  python graph_transform/scripts/cross_validation_analysis.py \
      --config graph_transform/config/default.yaml \
      --checkpoint <ckpt.pt> \
      --input_csv dataset/5fold/6072.test.fbr.multi.csv \
      --output_dir results/cross_validation \
      --num_samples 15 \
      --infer_config

  # 复用已有 occlusion 结果（避免重新跑 occlusion，节省 GPU 时间）：
  --occlusion_json results/occlusion_attribution_v2/occlusion_per_sample.json

  # 自定义 rollout 归一化模式：
  --rollout_normalize row   # 默认：每行归一化（"扩散"解释）
  --rollout_normalize col   # 每列归一化（GAT 默认 softmax over src per dst）
  --rollout_normalize none  # 不归一化
"""

from __future__ import annotations

import argparse
import json
import logging
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
from data import GraphDataset, CachedGraphDataset
from utils.attention_extractor import AttentionExtractor
from utils.visualization import (
    compute_attention_rank_correlation,
    plot_attention_rank_correlation,
    compute_attention_rollout,
    plot_attention_rollout,
    compute_attention_occlusion_correlation,
    plot_attention_occlusion_correlation,
)

DEFAULT_ALPHABET = "#ABCDEFGHIKLMNOPQRSTVWXYZ"


# =============================================================================
# 基础设施（与 interpretability_analysis.py 共用模式）
# =============================================================================

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s[%(levelname)s]:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("cross_validation")


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
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    inferred = base_config.copy()
    model_cfg = inferred.setdefault("model", {})
    if "bond_head.0.weight" in state_dict:
        bond_w = state_dict["bond_head.0.weight"].shape
        bond_feature_dim, hidden_dim = bond_w[1], bond_w[0]
        model_cfg["hidden_dim"] = hidden_dim
        remaining = bond_feature_dim - hidden_dim * 2
        model_cfg["bond_use_edge_repr"] = remaining >= hidden_dim * 3
        model_cfg["bond_use_diff_feature"] = remaining >= hidden_dim * 2
        model_cfg["bond_use_product_feature"] = remaining >= hidden_dim * 1
    return inferred


# =============================================================================
# 分层抽样（与 occlusion_analysis.py 一致）
# =============================================================================

def stratified_sample(
    dataset: GraphDataset,
    num_samples: int,
    random_seed: int,
    logger: logging.Logger,
) -> List[int]:
    rng = np.random.RandomState(random_seed)
    df = dataset.data.reset_index(drop=True).copy()
    df["seq_len"] = df["seq"].astype(str).str.len()
    df["fbr"] = pd.to_numeric(df["fbr"], errors="coerce")
    df["nce"] = pd.to_numeric(df["nce"], errors="coerce")

    def _len_bin(L):
        return "L_short" if L <= 24 else ("L_mid" if L <= 29 else "L_long")

    def _nce_bin(n):
        if pd.isna(n):
            return "N_missing"
        return "N_low" if n < 22 else ("N_mid" if n <= 27 else "N_high")

    def _fbr_bin(f):
        if pd.isna(f):
            return "F_missing"
        return "F_low" if f < 0.3 else ("F_mid" if f <= 0.7 else "F_high")

    df["stratum"] = (
        df["seq_len"].apply(_len_bin) + "|"
        + df["nce"].apply(_nce_bin) + "|"
        + df["fbr"].apply(_fbr_bin)
    )
    grouped = df.groupby("stratum")
    strata = list(grouped.groups.keys())
    logger.info(f"Found {len(strata)} non-empty strata")

    strata_shuffled = list(strata)
    rng.shuffle(strata_shuffled)

    selected: List[int] = []
    for s in strata_shuffled:
        if len(selected) >= num_samples:
            break
        candidates = grouped.groups[s].tolist()
        if candidates:
            pick = candidates[rng.randint(0, len(candidates))]
            selected.append(int(pick))

    if len(selected) < num_samples:
        sizes = {s: len(grouped.groups[s]) for s in strata}
        sorted_strata = sorted(sizes, key=lambda x: -sizes[x])
        for s in sorted_strata:
            while len(selected) < num_samples:
                candidates = grouped.groups[s].tolist()
                if not candidates:
                    break
                pick = candidates[rng.randint(0, len(candidates))]
                if pick not in selected:
                    selected.append(int(pick))
                else:
                    break

    logger.info(
        f"Selected {len(selected)} samples across "
        f"{len(set(df.loc[selected, 'stratum']))} strata"
    )
    return selected[:num_samples]


# =============================================================================
# Occlusion：现跑 or 复用
# =============================================================================

def run_occlusion_for_samples(
    model: GraphTransformer,
    dataset: GraphDataset,
    selected_indices: List[int],
    alphabet: str,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """对选定样本现跑 occlusion，返回 (occlusion_matrices, sample_meta_list)。"""
    from utils.attention_extractor import AttentionExtractor  # 已 import 但保持显式
    from scripts.occlusion_analysis import occlusion_for_sample

    sigmoid = torch.nn.Sigmoid()
    matrices: List[np.ndarray] = []
    meta: List[Dict[str, Any]] = []

    for k, idx in enumerate(selected_indices):
        sample = dataset[idx]
        seq = sample["sequence"]
        sample_id = f"idx{idx}_len{sample['seq_len']}"
        logger.info(
            f"  [occlusion {k+1}/{len(selected_indices)}] sample={sample_id} seq={seq[:32]}"
        )
        sensitivity, p_orig = occlusion_for_sample(
            model, dataset, sample, alphabet, device, logger, sample_id,
        )
        matrices.append(sensitivity)
        meta.append({
            "sample_id": sample_id,
            "dataset_idx": int(idx),
            "sequence": seq,
            "seq_len": int(sample["seq_len"]),
            "nce": float(sample["nce"]),
            "charge": float(sample["charge"]),
        })

    return matrices, meta


def load_occlusion_from_json(
    occlusion_json: str,
    selected_indices: List[int],
    dataset: GraphDataset,
    logger: logging.Logger,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """从 occlusion_per_sample.json 加载已有结果，匹配 selected_indices。

    匹配键：dataset_idx。
    若某 idx 在 json 中无对应记录，会跳过该样本（同时从 selected_indices 移除）。
    """
    with open(occlusion_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    by_idx = {rec["dataset_idx"]: rec for rec in records}
    logger.info(
        f"Loaded {len(records)} occlusion records; matching against "
        f"{len(selected_indices)} selected indices"
    )

    matrices: List[np.ndarray] = []
    meta: List[Dict[str, Any]] = []
    matched_indices: List[int] = []
    for idx in selected_indices:
        if idx not in by_idx:
            logger.warning(f"  idx={idx} not in occlusion json; will skip for Method 7")
            continue
        rec = by_idx[idx]
        # occlusion_matrix 在 json 中是 [[...], ...] 嵌套 list
        mat = np.array(rec.get("occlusion_matrix", []), dtype=np.float64)
        if mat.size == 0:
            logger.warning(f"  idx={idx} has empty occlusion_matrix; skip")
            continue
        matrices.append(mat)
        meta.append({
            "sample_id": rec.get("sample_id", f"idx{idx}"),
            "dataset_idx": int(idx),
            "sequence": rec.get("sequence", ""),
            "seq_len": int(rec.get("seq_len", 0)),
            "nce": float(rec.get("nce", 0.0)),
            "charge": float(rec.get("charge", 0.0)),
        })
        matched_indices.append(idx)

    logger.info(f"Matched {len(matrices)}/{len(selected_indices)} samples")
    # 把 selected_indices 替换为 matched（返回供调用方更新）
    selected_indices[:] = matched_indices
    return matrices, meta


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-validation analysis: prove attention is functional "
                    "(Method 5/6/7).",
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="results/cross_validation")
    parser.add_argument("--infer_config", action="store_true")
    parser.add_argument("--num_samples", type=int, default=15)
    parser.add_argument("--max_seq_len", type=int, default=32)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--figure_format", type=str, default="svg",
                        choices=["svg", "png"])
    parser.add_argument("--alphabet", type=str, default=DEFAULT_ALPHABET)
    # Method 6 选项
    parser.add_argument("--rollout_normalize", type=str, default="row",
                        choices=["row", "col", "none"],
                        help="rollout 节点矩阵归一化模式："
                             "row=每行归一化（默认,信息扩散解释）/ "
                             "col=每列归一化（GAT softmax over src per dst）/ "
                             "none=不归一化")
    # Method 7 选项
    parser.add_argument("--occlusion_json", type=str, default=None,
                        help="已有 occlusion_per_sample.json 路径（避免重跑）。"
                             "若不提供，本脚本会现跑 occlusion。")
    parser.add_argument("--skip_occlusion", action="store_true",
                        help="完全跳过 Method 7（不跑也不加载 occlusion）")
    args = parser.parse_args()

    img_ext = ".svg" if args.figure_format == "svg" else ".png"
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Cross-Validation Analysis: Method 5/6/7")
    logger.info("=" * 60)

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
    extractor = AttentionExtractor(model, device)

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

    # 报告层数
    num_gat = getattr(model_config, "num_gat_layers", "?")
    num_gcn = getattr(model_config, "num_gcn_layers", "?")
    num_heads = getattr(model_config, "num_attention_heads", "?")
    logger.info(f"Model layers: GCN×{num_gcn} → GAT×{num_gat} "
                f"({num_heads} heads/GAT layer)")
    logger.info(f"=> {num_gat} attention layer(s) to analyze: "
                f"L0 .. L{num_gat - 1}")

    # ---- 1. 分层抽样 ----
    selected = stratified_sample(
        test_dataset, args.num_samples, args.random_seed, logger,
    )

    # 提取每个样本的 attention
    attention_weights_list: List[List[torch.Tensor]] = []
    sequences: List[str] = []
    edge_indices: List[Optional[torch.Tensor]] = []
    for idx in selected:
        sample = test_dataset[idx]
        aw = extractor.extract_attention_for_sample(sample)
        attention_weights_list.append(aw)
        sequences.append(sample["sequence"])
        edge_indices.append(sample.get("edge_index"))
    logger.info(f"Extracted attention for {len(attention_weights_list)} samples, "
                f"{len(attention_weights_list[0])} layers each")

    # ---- Method 5: Attention Rank Correlation ----
    logger.info("\n" + "=" * 60)
    logger.info("Method 5: Attention Rank Correlation (Spearman ρ between layers)")
    logger.info("=" * 60)
    m5_summary = compute_attention_rank_correlation(
        attention_weights_list, edge_indices, sequences, args.max_seq_len,
    )
    m5_path = os.path.join(args.output_dir, f"method5_rank_correlation{img_ext}")
    plot_attention_rank_correlation(m5_summary, save_path=m5_path)
    logger.info(f"Saved: {m5_path}")
    for (a, b), metrics in m5_summary["per_pair_metrics"].items():
        logger.info(
            f"  L{a} vs L{b}: mean ρ = {metrics['mean_rho']:+.3f}, "
            f"median = {metrics['median_rho']:+.3f}, "
            f"std = {metrics['std_rho']:.3f} (n={metrics['n_valid']})"
        )

    # ---- Method 6: Attention Rollout ----
    logger.info("\n" + "=" * 60)
    logger.info(f"Method 6: Attention Rollout (normalize={args.rollout_normalize})")
    logger.info("=" * 60)
    m6_summary = compute_attention_rollout(
        attention_weights_list, edge_indices, sequences,
        args.max_seq_len, normalize=args.rollout_normalize,
    )
    m6_path = os.path.join(args.output_dir, f"method6_attention_rollout{img_ext}")
    plot_attention_rollout(
        m6_summary, attention_weights_list, edge_indices, sequences,
        max_seq_len=args.max_seq_len,
        save_path=m6_path,
    )
    logger.info(f"Saved: {m6_path}")
    for m in m6_summary["per_layer_correlation"]:
        logger.info(
            f"  rollout vs L{m['layer']}: mean ρ = {m['mean_rho_rollout_vs_layer']:+.3f} "
            f"(n={m['n_samples']})"
        )

    # ---- Method 7: Attention–Occlusion Correlation ----
    m7_summary: Optional[Dict[str, Any]] = None
    m7_per_sample: List[Dict[str, Any]] = []
    if args.skip_occlusion:
        logger.info("\nSkipping Method 7 (--skip_occlusion given)")
    else:
        logger.info("\n" + "=" * 60)
        logger.info("Method 7: Attention–Occlusion Correlation")
        logger.info("=" * 60)

        if args.occlusion_json:
            logger.info(f"Loading occlusion from {args.occlusion_json}")
            occ_matrices, occ_meta = load_occlusion_from_json(
                args.occlusion_json, selected, test_dataset, logger,
            )
            # selected 可能被缩减（只保留 json 中匹配的样本）
            # 同步缩减 attention/sequences/edge_indices
            matched_set = set(meta["dataset_idx"] for meta in occ_meta)
            keep_mask = [idx in matched_set for idx in selected]
            attention_weights_list = [a for a, k in zip(attention_weights_list, keep_mask) if k]
            sequences = [s for s, k in zip(sequences, keep_mask) if k]
            edge_indices = [e for e, k in zip(edge_indices, keep_mask) if k]
            selected = [idx for idx, k in zip(selected, keep_mask) if k]
            logger.info(f"After matching: {len(selected)} samples for Method 7")
        else:
            logger.info(f"Running fresh occlusion on {len(selected)} samples...")
            occ_matrices, occ_meta = run_occlusion_for_samples(
                model, test_dataset, selected, args.alphabet, device, logger,
            )

        if occ_matrices and len(occ_matrices) == len(attention_weights_list):
            m7_summary = compute_attention_occlusion_correlation(
                attention_weights_list, edge_indices, sequences, occ_matrices,
                args.max_seq_len,
            )
            m7_path = os.path.join(args.output_dir, f"method7_attention_occlusion{img_ext}")
            plot_attention_occlusion_correlation(m7_summary, save_path=m7_path)
            logger.info(f"Saved: {m7_path}")
            for m in m7_summary["per_layer_metrics"]:
                logger.info(
                    f"  L{m['layer']}: mean r = {m['mean_pearson_r']:+.3f}, "
                    f"median = {m['median_pearson_r']:+.3f}, "
                    f"ρ = {m['mean_spearman_rho']:+.3f} (n={m['n_valid']})"
                )
        else:
            logger.warning(
                f"Method 7 skipped: {len(occ_matrices)} occlusion vs "
                f"{len(attention_weights_list)} attention"
            )

    # ---- 落盘 JSON 摘要 ----
    summary = {
        "config": {
            "num_samples": len(selected),
            "max_seq_len": args.max_seq_len,
            "num_gat_layers": num_gat,
            "num_gcn_layers": num_gcn,
            "num_attention_heads": num_heads,
            "rollout_normalize": args.rollout_normalize,
        },
        "method5_rank_correlation": {
            "per_pair_metrics": {
                f"L{a}_vs_L{b}": {k: v for k, v in metrics.items() if k != "per_sample_rho"}
                for (a, b), metrics in m5_summary["per_pair_metrics"].items()
            },
        },
        "method6_rollout": {
            "normalize": m6_summary["normalize"],
            "per_layer_correlation": m6_summary["per_layer_correlation"],
        },
        "method7_attention_occlusion": (
            {
                "per_layer_metrics": m7_summary["per_layer_metrics"],
            } if m7_summary else None
        ),
        "samples": [
            {"dataset_idx": int(idx),
             "seq_len": len(seq),
             "sequence": seq[:32] + "..." if len(seq) > 32 else seq}
            for idx, seq in zip(selected, sequences)
        ],
    }
    summary_path = os.path.join(args.output_dir, "cross_validation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\nSaved summary: {summary_path}")

    # Per-sample 详情
    per_sample_path = os.path.join(args.output_dir, "cross_validation_per_sample.json")
    per_sample_data: Dict[str, Any] = {
        "method5_per_sample_rho": {
            f"L{a}_vs_L{b}": metrics["per_sample_rho"]
            for (a, b), metrics in m5_summary["per_pair_metrics"].items()
        },
        "method7_per_sample_r": (
            {f"L{i}": m7_summary["per_layer_per_sample_r"][i]
             for i in range(len(m7_summary["per_layer_per_sample_r"]))}
            if m7_summary else {}
        ),
    }
    with open(per_sample_path, "w", encoding="utf-8") as f:
        json.dump(per_sample_data, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved per-sample details: {per_sample_path}")
    logger.info("\nDone. Cross-validation complete.")


if __name__ == "__main__":
    main()
