#!/usr/bin/env python3
"""
Occlusion 因果归因分析（Paper-grade attribution）

研究问题：每个残基位置 j 因果上对各键预测的贡献是多少？
        与 functional-saliency attention 是否一致？

方法：对每个精选样本，逐位将残基 j 替换为 24 种氨基酸之一（排除原始），
     记录 Δp[j, i] = |p_mut[i] - p_orig[i]|，构建敏感度矩阵 M[j, i]。
     与 attention 矩阵对比，计算 Pearson r 评估一致性。

样本选择：自动分层抽样（3 维 × 3 桶 = 27 格，每格 1 个样本至 15 个）。
        - 序列长度：24 / 25-29 / ≥30
        - NCE：<22 / 22-27 / >27
        - FBR：<0.3 / 0.3-0.7 / >0.7

计算成本（15 样本）：~10,800 次推理 ≈ GPU 2 分钟

输出：
  - occlusion_<sample_id>.svg   每样本 2 子图：occlusion vs attention
  - occlusion_aggregate.svg     聚合一致性图（散点 + 箱线图）
  - occlusion_summary.json      关键统计
  - occlusion_per_sample.json   每样本原始矩阵

使用方式：
  python graph_transform/scripts/occlusion_analysis.py \
      --config graph_transform/config/default.yaml \
      --checkpoint <ckpt.pt> \
      --input_csv dataset/5fold/6072.test.fbr.multi.csv \
      --output_dir results/occlusion_attribution \
      --num_samples 15 \
      --infer_config
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
    build_residue_attention_matrix,
    collapse_to_residue_bond_attention,
    plot_occlusion_vs_attention,
    plot_occlusion_attention_consistency,
)

DEFAULT_ALPHABET = "#ABCDEFGHIKLMNOPQRSTVWXYZ"  # 25 chars (含 #)


# =============================================================================
# 基础设施
# =============================================================================

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s[%(levelname)s]:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("occlusion")


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
# 分层抽样
# =============================================================================

def stratified_sample(
    dataset: GraphDataset,
    num_samples: int,
    random_seed: int,
    logger: logging.Logger,
) -> List[int]:
    """按 (长度桶 × NCE 桶 × FBR 桶) 分层抽样。

    返回 dataset 索引列表（长度 ≤ num_samples）。
    """
    rng = np.random.RandomState(random_seed)
    df = dataset.data.reset_index(drop=True).copy()

    df["seq_len"] = df["seq"].astype(str).str.len()
    df["fbr"] = pd.to_numeric(df["fbr"], errors="coerce")
    df["nce"] = pd.to_numeric(df["nce"], errors="coerce")

    def _len_bin(L):
        if L <= 24:
            return "L_short"
        elif L <= 29:
            return "L_mid"
        else:
            return "L_long"

    def _nce_bin(n):
        if pd.isna(n):
            return "N_missing"
        if n < 22:
            return "N_low"
        elif n <= 27:
            return "N_mid"
        else:
            return "N_high"

    def _fbr_bin(f):
        if pd.isna(f):
            return "F_missing"
        if f < 0.3:
            return "F_low"
        elif f <= 0.7:
            return "F_mid"
        else:
            return "F_high"

    df["len_bin"] = df["seq_len"].apply(_len_bin)
    df["nce_bin"] = df["nce"].apply(_nce_bin)
    df["fbr_bin"] = df["fbr"].apply(_fbr_bin)

    df["stratum"] = df["len_bin"] + "|" + df["nce_bin"] + "|" + df["fbr_bin"]

    grouped = df.groupby("stratum")
    strata = list(grouped.groups.keys())
    logger.info(f"Found {len(strata)} non-empty strata")

    # Round-robin sample one from each stratum (shuffled order) until target reached
    strata_shuffled = list(strata)
    rng.shuffle(strata_shuffled)

    selected_idx: List[int] = []
    # Pass 1: one per stratum
    for s in strata_shuffled:
        if len(selected_idx) >= num_samples:
            break
        candidates = grouped.groups[s].tolist()
        if candidates:
            pick = candidates[rng.randint(0, len(candidates))]
            selected_idx.append(int(pick))

    # Pass 2: fill up by re-sampling from largest strata
    if len(selected_idx) < num_samples:
        sizes = {s: len(grouped.groups[s]) for s in strata}
        sorted_strata = sorted(sizes, key=lambda x: -sizes[x])
        for s in sorted_strata:
            while len(selected_idx) < num_samples:
                candidates = grouped.groups[s].tolist()
                if not candidates:
                    break
                pick = candidates[rng.randint(0, len(candidates))]
                if pick not in selected_idx:
                    selected_idx.append(int(pick))
                else:
                    break

    logger.info(f"Selected {len(selected_idx)} samples across "
                f"{len(set(df.loc[selected_idx, 'stratum']))} strata")
    return selected_idx[:num_samples]


# =============================================================================
# 单样本前向 + 突变扫描
# =============================================================================

def _single_sample_to_batch(sample: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """单样本 dict → batch_size=1 的 batch dict。

    仿照 AttentionExtractor.extract_attention_for_sample 的转换逻辑，
    但只为前向推理准备 batch。
    """
    batch: Dict[str, Any] = {}
    if "sequence" in sample:
        batch["sequences"] = [sample["sequence"]]

    no_batch_keys = {"edge_index", "edge_types", "edge_distances",
                     "edge_attr", "bond_edge_map"}
    for key, value in sample.items():
        if key == "sequence":
            continue
        if isinstance(value, torch.Tensor):
            if key in no_batch_keys:
                batch[key] = value.to(device)
            else:
                batch[key] = value.unsqueeze(0).to(device)
        elif isinstance(value, list):
            batch[key] = value
        elif isinstance(value, (int, float)):
            batch[key] = torch.tensor([value], dtype=torch.float32).to(device)
        else:
            batch[key] = value

    if "seq_lens" not in batch and "seq_len" in sample:
        batch["seq_lens"] = torch.tensor([sample["seq_len"]], dtype=torch.long).to(device)
    if "node_lens" not in batch and "node_len" in sample:
        batch["node_lens"] = torch.tensor([sample["node_len"]], dtype=torch.long).to(device)
    for singular, plural in [("charge", "charges"), ("pep_mass", "pep_masses"),
                             ("intensity", "intensities"), ("nce", "nces"),
                             ("rt", "rts")]:
        if singular in sample and plural not in batch:
            batch[plural] = torch.tensor([sample[singular]], dtype=torch.float32).to(device)
    if "state_vars" not in batch and all(k in sample for k in
                                          ("charge", "pep_mass", "intensity")):
        batch["state_vars"] = torch.tensor(
            [[sample["charge"], sample["pep_mass"], sample["intensity"]]],
            dtype=torch.float32,
        ).to(device)
    if "env_vars" not in batch and "nce" in sample:
        env_value = sample.get("env_feature_value", sample.get("rt", 0.0))
        batch["env_vars"] = torch.tensor(
            [[sample["nce"], env_value]], dtype=torch.float32,
        ).to(device)
    if "secondary_envs" not in batch:
        if "env_feature_value" in sample:
            batch["secondary_envs"] = torch.tensor(
                [sample["env_feature_value"]], dtype=torch.float32,
            ).to(device)
        elif "rt" in sample:
            batch["secondary_envs"] = torch.tensor(
                [sample["rt"]], dtype=torch.float32,
            ).to(device)
    # label_mask: 所有有效键为 True
    if "label_mask" not in batch and "labels" in batch:
        num_bonds = batch["labels"].shape[-1]
        batch["label_mask"] = torch.ones_like(batch["labels"])
    return batch


def forward_sample(
    model: GraphTransformer,
    sample: Dict[str, Any],
    device: torch.device,
    sigmoid: torch.nn.Module,
) -> torch.Tensor:
    """单样本前向，返回 [num_bonds] 概率（CPU tensor）。"""
    batch = _single_sample_to_batch(sample, device)
    with torch.no_grad():
        logits = model(batch)            # [1, max_bonds]
        probs = sigmoid(logits)[0]       # [max_bonds]
    seq_len = sample["seq_len"]
    num_bonds = max(seq_len - 1, 0)
    return probs[:num_bonds].cpu()


def mutate_sample(
    base_sample: Dict[str, Any],
    position: int,
    new_aa: str,
    dataset: GraphDataset,
) -> Dict[str, Any]:
    """构建突变样本：将 base_sample 的 sequence 在 position 处替换为 new_aa，
    重建图（edge_attr 依赖残基身份），其余字段保持不变。
    """
    new_seq = base_sample["sequence"][:position] + new_aa + base_sample["sequence"][position + 1:]

    sample_features = {
        "charge": base_sample["charge"],
        "pep_mass": base_sample["pep_mass"],
        "intensity": base_sample["intensity"],
        "nce": base_sample["nce"],
        dataset.env_feature_name: base_sample.get(
            "env_feature_value", base_sample.get("rt", 0.0)
        ),
    }
    if "scan_num" in base_sample:
        sample_features["scan_num"] = base_sample["scan_num"]

    graph_data = dataset.graph_builder.build_graph(
        new_seq, sample_features, dataset.graph_strategy,
    )

    new_sample = dict(base_sample)
    new_sample["sequence"] = new_seq
    new_sample["edge_index"] = graph_data["edge_index"]
    new_sample["edge_attr"] = graph_data["edge_attr"]
    new_sample["edge_types"] = graph_data["edge_types"]
    new_sample["edge_distances"] = graph_data["edge_distances"]
    new_sample["bond_edge_map"] = graph_data["bond_edge_map"]
    # seq_len / node_len 不变（残基数不变）
    return new_sample


def occlusion_for_sample(
    model: GraphTransformer,
    dataset: GraphDataset,
    base_sample: Dict[str, Any],
    alphabet: str,
    device: torch.device,
    logger: logging.Logger,
    sample_id: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """对单样本执行 occlusion 扫描。

    返回:
        sensitivity: [seq_len, num_bonds]  M[j, i] = mean_aa |p_mut[i] - p_orig[i]|
        p_orig:      [num_bonds] 原始预测概率
    """
    sigmoid = torch.nn.Sigmoid()
    seq = base_sample["sequence"]
    seq_len = base_sample["seq_len"]
    num_bonds = max(seq_len - 1, 0)

    p_orig = forward_sample(model, base_sample, device, sigmoid)  # [num_bonds]

    mutation_aas = [c for c in alphabet if c != "#"]
    sensitivity = np.zeros((seq_len, num_bonds), dtype=np.float64)

    n_total = seq_len * len(mutation_aas)
    n_done = 0
    for j in range(seq_len):
        original_aa = seq[j]
        for aa in mutation_aas:
            if aa == original_aa:
                continue
            try:
                mut_sample = mutate_sample(base_sample, j, aa, dataset)
                p_mut = forward_sample(model, mut_sample, device, sigmoid)
                delta = (p_mut - p_orig).numpy()
                sensitivity[j, :] += np.abs(delta)
            except Exception as e:
                logger.warning(f"  sample={sample_id} pos={j} aa={aa} failed: {e}")
            n_done += 1
        # 平均：除以 (len(mutation_aas) - 1) 因为跳过了 original_aa
        sensitivity[j, :] /= max(len(mutation_aas) - 1, 1)

        if (j + 1) % 10 == 0 or j == seq_len - 1:
            logger.info(f"  sample={sample_id} pos {j+1}/{seq_len} "
                        f"({n_done}/{n_total} mutations done)")

    return sensitivity, p_orig.numpy()


# =============================================================================
# Attention 提取（per-sample）
# =============================================================================

def attention_matrix_for_sample(
    extractor: AttentionExtractor,
    sample: Dict[str, Any],
    layer_idx: int = -1,
) -> np.ndarray:
    """提取单样本指定层的 attention，转为 [seq_len, num_bonds] 残基-键矩阵。"""
    attn_list = extractor.extract_attention_for_sample(sample)
    if not attn_list:
        return np.zeros((0, 0), dtype=np.float64)
    if layer_idx < 0:
        layer_idx = len(attn_list) - 1
    attn = attn_list[layer_idx]                # [num_edges, num_heads]
    edge_index = sample["edge_index"]
    seq_len = sample["seq_len"]
    residue_attn = build_residue_attention_matrix(attn, edge_index, seq_len)
    return collapse_to_residue_bond_attention(residue_attn, seq_len)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Occlusion causal attribution analysis.",
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="results/occlusion_attribution")
    parser.add_argument("--infer_config", action="store_true")
    parser.add_argument("--num_samples", type=int, default=15)
    parser.add_argument("--max_seq_len", type=int, default=32)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--figure_format", type=str, default="svg",
                        choices=["svg", "png"])
    parser.add_argument("--attention_layer", type=int, default=-1,
                        help="使用哪一层 attention 做对比（-1 = 最后一层）")
    parser.add_argument("--alphabet", type=str, default=DEFAULT_ALPHABET)
    args = parser.parse_args()

    img_ext = ".svg" if args.figure_format == "svg" else ".png"
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Occlusion Causal Attribution Analysis")
    logger.info("=" * 60)

    # 检查文件
    for p, label in [(args.config, "Config"),
                     (args.checkpoint, "Checkpoint"),
                     (args.input_csv, "Input CSV")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{label} not found: {p}")

    # 加载
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

    # ---- 1. 分层抽样 ----
    selected = stratified_sample(
        test_dataset, args.num_samples, args.random_seed, logger,
    )

    per_sample_results: List[Dict[str, Any]] = []
    all_attention_flat: List[np.ndarray] = []
    all_occlusion_flat: List[np.ndarray] = []
    per_sample_r: List[float] = []
    sample_ids: List[str] = []

    # ---- 2. 逐样本 occlusion + attention ----
    for k, idx in enumerate(selected):
        sample = test_dataset[idx]
        seq = sample["sequence"]
        seq_len = sample["seq_len"]
        sample_id = f"idx{idx}_len{seq_len}"

        logger.info(f"\n[{k+1}/{len(selected)}] sample={sample_id}  seq={seq[:32]}...")

        # 原始 attention 矩阵
        try:
            attn_mat = attention_matrix_for_sample(
                extractor, sample, layer_idx=args.attention_layer,
            )
        except Exception as e:
            logger.warning(f"  attention extraction failed: {e}; skipping sample")
            continue

        # Occlusion 扫描
        sensitivity, p_orig = occlusion_for_sample(
            model, test_dataset, sample, args.alphabet, device, logger, sample_id,
        )

        num_bonds = sensitivity.shape[1]
        occ_mat = sensitivity[:seq_len, :num_bonds]
        att_mat = attn_mat[:seq_len, :num_bonds] if attn_mat.size else np.zeros_like(occ_mat)

        # 一致性 r
        if occ_mat.size > 1 and np.std(occ_mat) > 0 and np.std(att_mat) > 0:
            r = float(np.corrcoef(occ_mat.flatten(), att_mat.flatten())[0, 1])
        else:
            r = float("nan")

        per_sample_r.append(r)
        sample_ids.append(sample_id)
        all_attention_flat.append(att_mat.flatten())
        all_occlusion_flat.append(occ_mat.flatten())

        # 每样本图
        fig_path = os.path.join(
            args.output_dir, f"occlusion_{sample_id}{img_ext}",
        )
        fig, info = plot_occlusion_vs_attention(
            occlusion_matrix=occ_mat,
            attention_matrix=att_mat,
            sequence=seq,
            sample_id=sample_id,
            layer_idx=args.attention_layer,
            save_path=fig_path,
        )
        logger.info(f"  saved figure: {fig_path}  r={r:+.3f}")

        per_sample_results.append({
            "sample_id": sample_id,
            "dataset_idx": int(idx),
            "sequence": seq,
            "seq_len": int(seq_len),
            "nce": float(sample["nce"]),
            "charge": float(sample["charge"]),
            "p_orig": p_orig.tolist(),
            "pearson_r": r,
            "consistency": info["consistency"],
            "occlusion_matrix": occ_mat.tolist(),
            "attention_matrix": att_mat.tolist(),
        })

    # ---- 3. 聚合一致性分析 ----
    if all_attention_flat and all_occlusion_flat:
        attn_concat = np.concatenate(all_attention_flat)
        occ_concat = np.concatenate(all_occlusion_flat)
    else:
        attn_concat = np.array([])
        occ_concat = np.array([])

    if attn_concat.size > 1:
        agg_path = os.path.join(
            args.output_dir, f"occlusion_aggregate{img_ext}",
        )
        agg_fig, agg_info = plot_occlusion_attention_consistency(
            per_sample_r=per_sample_r,
            sample_ids=sample_ids,
            all_attention_flat=attn_concat,
            all_occlusion_flat=occ_concat,
            save_path=agg_path,
        )
        logger.info(f"Saved aggregate figure: {agg_path}")
    else:
        agg_info = {"error": "no valid samples"}

    # ---- 4. 落盘 JSON ----
    per_sample_path = os.path.join(args.output_dir, "occlusion_per_sample.json")
    with open(per_sample_path, "w", encoding="utf-8") as f:
        json.dump(per_sample_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved per-sample JSON: {per_sample_path}")

    summary = {
        "num_samples_requested": args.num_samples,
        "num_samples_completed": len(per_sample_results),
        "attention_layer": args.attention_layer,
        "aggregate": agg_info,
        "per_sample_pearson_r": dict(zip(sample_ids, per_sample_r)),
    }
    summary_path = os.path.join(args.output_dir, "occlusion_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved summary JSON: {summary_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
