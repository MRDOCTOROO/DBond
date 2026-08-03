#!/usr/bin/env python3
"""
可解释性分析脚本

本脚本用于生成GraphTransformer模型的可解释性分析图表，包括：
1. 新版4子图案例分析（注意力分布、层间演化、断裂对比、热力图）
2. 单样本各层注意力分布图
3. 层间演化趋势图

使用方法：
python graph_transform/scripts/interpretability_analysis.py \
    --config graph_transform/config/default.yaml \
    --checkpoint best_model/graph_transform.pt \
    --input_csv test_data.csv \
    --output_dir results/interpretability \
    --num_samples 5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
import json
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GraphTransformer
from models.utils import build_model_config, CheckpointManager
from data import GraphDataset, CachedGraphDataset
from utils.attention_extractor import AttentionExtractor
from utils.visualization import (
    plot_single_sample_layer_attention,
    plot_layer_evolution_trend,
    plot_bond_type_comparison,
    plot_new_interpretability_case_study,
    plot_aggregate_layer_attention,
    compute_layer_correlations,
    compute_layer_separation_metrics,
    compute_effect_size,
    detect_attention_mode,
    extract_bond_level_attention,
    bootstrap_ci,
    select_representative_case,
    plot_panel_a_case_study,
    plot_panel_c_violin,
    plot_grouped_alignment,
    VALID_ATTENTION_MODES,
    DEFAULT_ATTENTION_MODE,
)


def setup_logging() -> logging.Logger:
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s[%(levelname)s]:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("interpretability")


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_device(config: Dict[str, Any]) -> torch.device:
    """设置计算设备"""
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


def infer_model_config_from_checkpoint(checkpoint_path: str, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """从检查点推断模型配置"""
    logger = logging.getLogger("interpretability")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    inferred_config = base_config.copy()
    model_config = inferred_config.setdefault('model', {})
    
    if 'bond_head.0.weight' in state_dict:
        bond_weight_shape = state_dict['bond_head.0.weight'].shape
        bond_feature_dim = bond_weight_shape[1]
        hidden_dim = bond_weight_shape[0]
        
        logger.info(f"Inferred from checkpoint: hidden_dim={hidden_dim}, bond_feature_dim={bond_feature_dim}")
        
        base_dim = hidden_dim * 2
        remaining_dim = bond_feature_dim - base_dim
        
        bond_use_edge_repr = True
        bond_use_diff_feature = True
        bond_use_product_feature = True
        
        if remaining_dim == hidden_dim * 2:
            bond_use_product_feature = False
        elif remaining_dim == hidden_dim:
            bond_use_diff_feature = False
            bond_use_product_feature = False
        elif remaining_dim == 0:
            bond_use_edge_repr = False
            bond_use_diff_feature = False
            bond_use_product_feature = False
        
        model_config['bond_use_edge_repr'] = bond_use_edge_repr
        model_config['bond_use_diff_feature'] = bond_use_diff_feature
        model_config['bond_use_product_feature'] = bond_use_product_feature
        model_config['hidden_dim'] = hidden_dim
    
    if 'node_encoder.aa_embedding.weight' in state_dict:
        vocab_size = state_dict['node_encoder.aa_embedding.weight'].shape[0]
        logger.info(f"Inferred vocab_size: {vocab_size}")
    
    return inferred_config


def select_diverse_samples(dataset, num_samples: int, logger) -> List[int]:
    """选择多样化的样本（按序列长度分桶 + 桶内随机抽样）。

    旧版实现按序列长度排序后等间隔取样，每种长度只取「第一个」样本，
    导致选出的样本高度依赖数据顺序，容易挑到异常样本（如 padding 边缘、
    罕见 AA 富集序列等），case study 图会呈现与聚合统计相反的模式。

    改进版：
      1. 按序列长度分桶（每 1 个长度一个桶）
      2. 在桶内随机抽样（带 random_seed）
      3. 跨桶均匀分配 num_samples 个样本
    这样既能覆盖长度多样性，又避免「第一个样本」的偏置。
    """
    import random
    random.seed(42)

    if not (hasattr(dataset, 'data') and 'seq' in dataset.data.columns):
        return sorted(random.sample(range(len(dataset)), min(num_samples, len(dataset))))

    df = dataset.data.reset_index(drop=True)
    df['seq_len'] = df['seq'].astype(str).str.len()

    # 按长度分桶
    length_bins: dict[int, list[int]] = {}
    for idx, L in zip(df.index, df['seq_len']):
        length_bins.setdefault(int(L), []).append(int(idx))

    unique_lengths = sorted(length_bins.keys())
    if len(unique_lengths) <= num_samples:
        # 长度种类不够：每个长度抽 1 个
        selected = [random.choice(length_bins[L]) for L in unique_lengths]
        # 不足则从最大桶补
        while len(selected) < num_samples:
            biggest = max(unique_lengths, key=lambda L: len(length_bins[L]))
            pick = random.choice(length_bins[biggest])
            if pick not in selected:
                selected.append(pick)
            else:
                break
        return sorted(selected[:num_samples])

    # 均匀分布 num_samples 到各长度桶
    # 策略：把长度范围等分为 num_samples 个区间，每个区间随机抽 1 个
    min_L, max_L = unique_lengths[0], unique_lengths[-1]
    selected: list[int] = []
    for i in range(num_samples):
        lo = min_L + (max_L - min_L) * i // num_samples
        hi = min_L + (max_L - min_L) * (i + 1) // num_samples
        # 找 [lo, hi] 范围内的所有可用长度
        candidate_lengths = [L for L in unique_lengths if lo <= L <= hi]
        if not candidate_lengths:
            # 回退：用最近的长度
            nearest = min(unique_lengths, key=lambda L: abs(L - (lo + hi) / 2))
            candidate_lengths = [nearest]
        chosen_length = random.choice(candidate_lengths)
        pick = random.choice(length_bins[chosen_length])
        # 避免重复
        attempts = 0
        while pick in selected and attempts < 5:
            pick = random.choice(length_bins[chosen_length])
            attempts += 1
        selected.append(pick)

    logger.info(f"Selected {len(selected)} samples across "
                f"{len(set(df.loc[selected, 'seq_len']))} length bins; "
                f"lengths = {sorted(set(df.loc[selected, 'seq_len'].tolist()))}")
    return sorted(selected)


def select_statistical_samples(dataset, num_samples: int, random_seed: int,
                               logger) -> List[int]:
    """Proportionally sample length x charge x FBR strata.

    The allocation follows the test-set distribution while assigning at least
    one sample to every non-empty stratum when the requested sample count
    permits. This preserves population representativeness and guarantees that
    the reviewer-requested subgroup analyses are not silently missing strata.
    """
    import random

    rng = random.Random(random_seed)
    total = len(dataset)
    target = min(num_samples, total)
    if not hasattr(dataset, "data"):
        return sorted(rng.sample(range(total), target))

    df = dataset.data.reset_index(drop=True).copy()
    required = {"seq", "charge", "fbr"}
    if not required.issubset(df.columns):
        logger.warning(
            "Missing one of seq/charge/fbr; falling back to global random sampling"
        )
        return sorted(rng.sample(range(total), target))

    lengths = df["seq"].astype(str).str.len()
    df["length_group"] = pd.cut(
        lengths, bins=[-np.inf, 24, 29, np.inf],
        labels=["Short (<=24)", "Medium (25-29)", "Long (>=30)"],
    ).astype(str)
    charge_numeric = pd.to_numeric(df["charge"], errors="coerce")
    df["charge_group"] = charge_numeric.map(
        lambda value: "Charge missing" if pd.isna(value) else f"Charge {value:g}"
    )
    fbr_numeric = pd.to_numeric(df["fbr"], errors="coerce")
    df["fbr_group"] = pd.cut(
        fbr_numeric, bins=[-np.inf, 0.3, 0.7, np.inf], right=False,
        labels=["Low FBR (<0.3)", "Medium FBR (0.3-0.7)", "High FBR (>=0.7)"],
    ).astype(str)
    df["stratum"] = (
        df["length_group"] + " | " + df["charge_group"] + " | " + df["fbr_group"]
    )

    groups = {name: indices.tolist() for name, indices in df.groupby("stratum").groups.items()}
    strata = sorted(groups)
    logger.info(f"Statistical sampling found {len(strata)} non-empty "
                "length x charge x FBR strata")

    counts = np.asarray([len(groups[name]) for name in strata], dtype=float)
    exact = counts / counts.sum() * target
    allocation = np.floor(exact).astype(int)
    if target >= len(strata):
        allocation = np.maximum(allocation, 1)

    # Reconcile the integer allocation with the exact requested total.
    while allocation.sum() > target:
        candidates = np.where(allocation > (1 if target >= len(strata) else 0))[0]
        if not len(candidates):
            break
        idx = min(candidates, key=lambda i: exact[i] - allocation[i])
        allocation[idx] -= 1
    while allocation.sum() < target:
        candidates = [i for i in range(len(strata)) if allocation[i] < counts[i]]
        if not candidates:
            break
        idx = max(candidates, key=lambda i: exact[i] - allocation[i])
        allocation[idx] += 1

    selected: List[int] = []
    for name, requested in zip(strata, allocation):
        candidates = groups[name]
        sampled = rng.sample(candidates, min(int(requested), len(candidates)))
        selected.extend(int(index) for index in sampled)
        if requested:
            logger.info(f"  {name}: total={len(candidates)}, sampled={len(sampled)}")
    return sorted(selected)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="GraphTransformer可解释性分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python graph_transform/scripts/interpretability_analysis.py \
      --config graph_transform/config/default.yaml \
      --checkpoint best_model/graph_transform.pt \
      --input_csv test_data.csv \
      --output_dir results/interpretability \
      --num_samples 5
        """
    )
    
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--input_csv", type=str, required=True, help="输入CSV文件路径")
    parser.add_argument("--output_dir", type=str, default="results/interpretability", help="输出目录")
    parser.add_argument("--num_samples", type=int, default=5, help="用于案例分析的样本数量")
    parser.add_argument("--num_stat_samples", type=int, default=500, help="用于统计分析的样本数量")
    parser.add_argument("--bootstrap_iters", type=int, default=1000,
                        help="bootstrap置信区间重抽样次数")
    parser.add_argument("--max_seq_len", type=int, default=25, help="最大序列长度")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default=None, help="计算设备")
    parser.add_argument("--infer_config", action="store_true", help="从检查点自动推断模型配置")
    parser.add_argument("--random_seed", type=int, default=42, help="随机种子")
    # Paper-grade semantics + format
    parser.add_argument("--attention_mode", type=str, default=DEFAULT_ATTENTION_MODE,
                        choices=sorted(VALID_ATTENTION_MODES),
                        help="注意力解释模式(统一为 functional saliency): functional(默认) / "
                             "cleavage / importance / auto / stability(已废弃,自动映射)")
    parser.add_argument("--figure_format", type=str, default="svg", choices=["svg", "png"],
                        help="图像输出格式: svg(矢量,浏览器可编辑文本) 或 png")
    parser.add_argument("--heatmap_normalize", type=str, default="row",
                        choices=["row", "global", "absolute"],
                        help="Panel (d) 色阶模式: row(默认,每层独立归一化,视觉与指标一致) / "
                             "global(同 row, 兼容旧名) / absolute(保留层间绝对大小,但低量级层"
                             "的内部模式不可见,视觉可能与 entropy 指标矛盾)")
    
    args = parser.parse_args()
    img_ext = ".svg" if args.figure_format == "svg" else ".png"
    
    # 设置日志
    logger = setup_logging()
    logger.info("Starting interpretability analysis")
    analysis_start_time = time.time()
    
    # 检查文件
    for path_value, label in [
        (args.config, "Config"),
        (args.checkpoint, "Checkpoint"),
        (args.input_csv, "Input CSV"),
    ]:
        if not os.path.exists(path_value):
            raise FileNotFoundError(f"{label} not found: {path_value}")
    
    # 加载配置
    config = load_config(args.config)
    
    # 从检查点推断配置
    if args.infer_config:
        logger.info("Inferring model config from checkpoint...")
        config = infer_model_config_from_checkpoint(args.checkpoint, config)
    
    # 设置设备
    if args.device:
        config.setdefault("device", {})["auto_detect"] = False
        config["device"]["device_type"] = args.device
    
    device = setup_device(config)
    logger.info(f"Using device: {device}")
    
    # 加载模型
    model_config = build_model_config(config)
    model = GraphTransformer(model_config).to(device)
    
    try:
        CheckpointManager.load_checkpoint(args.checkpoint, model=model, device=device)
        logger.info(f"Loaded checkpoint: {args.checkpoint}")
    except RuntimeError as e:
        if "size mismatch" in str(e) and not args.infer_config:
            logger.error(f"Config mismatch detected: {e}")
            logger.error("Try using --infer_config to automatically infer model config")
            raise
        else:
            raise
    
    # 创建注意力提取器
    extractor = AttentionExtractor(model, device)
    
    # 加载数据
    data_config = config["data"]
    data_config["test_csv_path"] = args.input_csv
    
    if args.max_seq_len:
        data_config["max_seq_len"] = args.max_seq_len
    
    dataset_cls = CachedGraphDataset if data_config.get("cache_graphs", False) else GraphDataset
    dataset_kwargs = {
        "csv_path": data_config["test_csv_path"],
        "config": model_config,
        "max_seq_len": data_config.get("max_seq_len"),
        "graph_strategy": data_config.get("graph_strategy", "hybrid"),
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
    logger.info(f"Loaded dataset with {len(test_dataset)} samples")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置随机种子
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        import random
        random.seed(args.random_seed)
    
    # ========== 第一部分：案例分析样本 ==========
    logger.info("=" * 60)
    logger.info("Part 1: Case Study Analysis")
    logger.info("=" * 60)
    
    # 选择样本
    case_indices = select_diverse_samples(test_dataset, args.num_samples, logger)
    logger.info(f"Selected {len(case_indices)} case study samples: {case_indices}")
    
    # 提取注意力权重
    all_attention_weights = []
    all_sequences = []
    all_bond_labels = []
    all_edge_indices = []
    
    for idx in case_indices:
        if idx >= len(test_dataset):
            continue
        
        sample_data = test_dataset[idx]
        
        # 提取序列
        if hasattr(test_dataset, 'data') and 'seq' in test_dataset.data.columns:
            sequence = test_dataset.data.iloc[idx]['seq']
        else:
            sequence = f"Sample_{idx}"
        
        # 提取标签
        bond_labels = sample_data.get('labels', torch.zeros(10))
        
        # 提取注意力权重
        attention_weights = extractor.extract_attention_for_sample(sample_data)
        
        all_attention_weights.append(attention_weights)
        all_sequences.append(sequence)
        all_bond_labels.append(bond_labels)
        all_edge_indices.append(sample_data.get('edge_index'))
        
        logger.info(f"  Sample {idx}: seq_len={len(sequence)}, num_layers={len(attention_weights)}")
    
    # 生成单样本各层注意力图（移入 single_samples/ 子目录，作为 supplementary）
    single_sample_dir = os.path.join(args.output_dir, "single_samples")
    os.makedirs(single_sample_dir, exist_ok=True)
    logger.info("Generating single sample layer attention plots "
                "(supplementary, illustrative-not-representative)...")
    for i, (attn_weights, seq, labels, edge_idx) in enumerate(
        zip(all_attention_weights, all_sequences, all_bond_labels, all_edge_indices)
    ):
        save_path = os.path.join(single_sample_dir,
                                 f"sample_{case_indices[i]}_layer_attention{img_ext}")
        plot_single_sample_layer_attention(
            attn_weights, labels, seq, edge_idx,
            save_path=save_path, max_seq_len=args.max_seq_len,
            attention_mode=args.attention_mode,
        )
        plt.close('all')
    logger.info(f"  saved {len(case_indices)} single-sample figures to: {single_sample_dir}")
    
    # 生成层间演化趋势图
    logger.info("Generating layer evolution trend plot...")
    layer_metrics = compute_layer_separation_metrics(
        all_attention_weights, all_bond_labels, all_edge_indices, all_sequences
    )
    layer_corrs = [m["abs_r"] for m in layer_metrics]  # backward-compat display
    
    if layer_metrics:
        trend_path = os.path.join(args.output_dir, f"layer_evolution_trend{img_ext}")
        plot_layer_evolution_trend(
            layer_metrics, save_path=trend_path,
            attention_mode=args.attention_mode,
        )
        plt.close('all')
        # 保存完整的分层指标 (abs_r, signed r, spearman, auc)
        corr_df = pd.DataFrame([
            {"layer": f"Layer_{i}", **m} for i, m in enumerate(layer_metrics)
        ])
        corr_df.to_csv(os.path.join(args.output_dir, "layer_correlations.csv"), index=False)
    
    # NOTE: 合并图 interpretability_case_study_new 改在 Part 3 生成（(c)(d) 用统计样本）。
    # case_summary 在此处先置空，Part 3 会填充。
    case_summary = None
    
    # 生成断裂对比图
    logger.info("Generating bond type comparison plot...")
    comparison_path = os.path.join(args.output_dir, f"bond_type_comparison{img_ext}")
    _, case_effect = plot_bond_type_comparison(
        all_attention_weights, all_bond_labels, all_edge_indices, all_sequences,
        save_path=comparison_path, attention_mode=args.attention_mode,
        max_seq_len=args.max_seq_len,
    )
    plt.close('all')
    
    # 统计分析容器（用于JSON）
    stat_layer_metrics = []
    stat_layer_per_sample: List[List[Dict[str, float]]] = []
    stat_effect = {}
    # 统计样本容器（Part 3 的独立面板会用这些；若 Part 2 未运行则保持空）
    stat_attention_weights: List = []
    stat_bond_labels: List = []
    stat_sequences: List[str] = []
    stat_edge_indices: List = []
    stat_length_groups: List[str] = []
    stat_charge_groups: List[str] = []
    stat_fbr_groups: List[str] = []
    stat_sample_ids: List[int] = []
    grouped_summaries: Dict[str, object] = {}
    has_stat = False
    
    # ========== 第二部分：统计分析 ==========
    if args.num_stat_samples > args.num_samples:
        logger.info("=" * 60)
        logger.info(f"Part 2: Statistical Analysis ({args.num_stat_samples} samples)")
        logger.info("=" * 60)
        
        stat_indices = select_statistical_samples(
            test_dataset, args.num_stat_samples,
            (args.random_seed + 1) if args.random_seed is not None else int(time.time()),
            logger,
        )
        logger.info(f"Stratified sampling: selected {len(stat_indices)} samples")
        
        # 提取统计样本的注意力权重
        stat_attention_weights = []
        stat_bond_labels = []
        stat_sequences = []
        stat_edge_indices = []
        
        for idx, sample_idx in enumerate(stat_indices):
            if idx % 100 == 0:
                logger.info(f"Processing statistical sample {idx}/{len(stat_indices)}")
            
            try:
                sample_data = test_dataset[sample_idx]
                bond_labels = sample_data.get('labels')
                if bond_labels is None:
                    continue
                
                attention_weights = extractor.extract_attention_for_sample(sample_data)
                
                if hasattr(test_dataset, 'data') and 'seq' in test_dataset.data.columns:
                    sequence = test_dataset.data.iloc[sample_idx]['seq']
                else:
                    sequence = f"Sample_{sample_idx}"
                
                stat_attention_weights.append(attention_weights)
                stat_bond_labels.append(bond_labels)
                stat_sequences.append(sequence)
                stat_edge_indices.append(sample_data.get('edge_index'))
                stat_sample_ids.append(int(sample_idx))

                row = test_dataset.data.iloc[sample_idx] if hasattr(test_dataset, 'data') else {}
                seq_length = len(sequence)
                stat_length_groups.append(
                    "Short (<=24)" if seq_length <= 24
                    else "Medium (25-29)" if seq_length <= 29
                    else "Long (>=30)"
                )
                charge = pd.to_numeric(row.get('charge', np.nan), errors='coerce')
                stat_charge_groups.append(
                    "Charge missing" if pd.isna(charge) else f"Charge {charge:g}"
                )
                fbr = pd.to_numeric(row.get('fbr', np.nan), errors='coerce')
                stat_fbr_groups.append(
                    "FBR missing" if pd.isna(fbr)
                    else "Low FBR (<0.3)" if fbr < 0.3
                    else "Medium FBR (0.3-0.7)" if fbr < 0.7
                    else "High FBR (>=0.7)"
                )
            except Exception as e:
                logger.warning(f"Error processing sample {sample_idx}: {e}")
                continue
        
        # 统计分析的层间演化
        logger.info("Computing layer correlations for statistical analysis...")
        stat_layer_metrics, stat_layer_per_sample = compute_layer_separation_metrics(
            stat_attention_weights, stat_bond_labels, stat_edge_indices, stat_sequences,
            return_per_sample=True,
        )
        stat_layer_corrs = [m["abs_r"] for m in stat_layer_metrics]
        has_stat = len(stat_attention_weights) > 0
        
        if stat_layer_metrics:
            stat_trend_path = os.path.join(args.output_dir, f"layer_evolution_trend_statistical{img_ext}")
            plot_layer_evolution_trend(
                stat_layer_metrics, save_path=stat_trend_path,
                attention_mode=args.attention_mode,
                per_sample_metrics=stat_layer_per_sample,
                show_ci=True,
                n_bootstrap=args.bootstrap_iters,
            )
            plt.close('all')
            # 保存统计结果
            stat_corr_df = pd.DataFrame([
                {"layer": f"Layer_{i}", **m} for i, m in enumerate(stat_layer_metrics)
            ])
            stat_corr_df.to_csv(os.path.join(args.output_dir, "layer_correlations_statistical.csv"), index=False)
        
        # 统计分析的断裂对比
        logger.info("Generating statistical bond type comparison...")
        stat_comparison_path = os.path.join(args.output_dir, f"bond_type_comparison_statistical{img_ext}")
        _, stat_effect = plot_bond_type_comparison(
            stat_attention_weights, stat_bond_labels, stat_edge_indices, stat_sequences,
            save_path=stat_comparison_path, attention_mode=args.attention_mode,
            max_seq_len=args.max_seq_len,
        )
        plt.close('all')

        # ===== 聚合图层注意力图（基于多样本的群体平均模式）=====
        # 这是「真实平均」的可信视图，不受单样本选择偏置影响。
        # 与单样本图（sample_<idx>_layer_attention.svg）形成对比：
        #   - 单样本图：1 个样本，可能异常
        #   - 聚合图：N 个样本的中位数 + IQR，反映模型群体行为
        logger.info(f"Generating aggregate layer attention figure "
                    f"(n={len(stat_attention_weights)} samples)...")
        agg_path = os.path.join(
            args.output_dir, f"aggregate_layer_attention{img_ext}",
        )
        agg_fig, agg_summary = plot_aggregate_layer_attention(
            stat_attention_weights,
            stat_bond_labels,
            stat_edge_indices,
            stat_sequences,
            save_path=agg_path,
            max_seq_len=args.max_seq_len,
            max_bonds_show=min(20, args.max_seq_len - 1),
        )
        plt.close('all')
        logger.info(f"Saved aggregate layer attention: {agg_path}")
        logger.info(f"  Layer focus progression: {agg_summary['interpretation']}")
    
    # ========== 第三部分：4 个独立 SVG 面板 + 修复版合并图 ==========
    # 生成 4 个独立 SVG（panel_a/b/c/d）便于 LaTeX 自由排版，避免单张超大图。
    # 面板 (a) 单样本；面板 (b)(c)(d) 优先使用大样本统计集（满足审稿人 R-22）。
    # 同时修复 interpretability_case_study_new：其 (c)(d) 改用统计样本。
    logger.info("=" * 60)
    logger.info("Part 3: 4 separate SVG panels + fixed combined figure")
    logger.info("=" * 60)

    # 确定面板 (b)(c)(d) 的群体数据源：优先统计样本，回退到案例样本
    pop_attn = stat_attention_weights if has_stat else all_attention_weights
    pop_labels = stat_bond_labels if has_stat else all_bond_labels
    pop_edge = stat_edge_indices if has_stat else all_edge_indices
    pop_seq = stat_sequences if has_stat else all_sequences
    pop_source_label = "statistical" if has_stat else "case (fallback)"
    logger.info(f"Panels (b)(c)(d) data source: {pop_source_label} "
                f"(n={len(pop_attn)})")

    # ---- 面板 (a): 代表性单样本（按 median |r| 选择，反 cherry-pick）----
    rep_idx, rep_info = select_representative_case(
        pop_attn, pop_labels, pop_edge, pop_seq,
        max_seq_len=args.max_seq_len,
    )
    if has_stat and rep_idx < len(stat_sample_ids):
        rep_info["dataset_idx"] = stat_sample_ids[rep_idx]
    logger.info(f"Panel (a) representative sample index={rep_idx} ({rep_info})")
    panel_a_path = os.path.join(args.output_dir, f"panel_a_case_study{img_ext}")
    plot_panel_a_case_study(
        pop_attn[rep_idx], pop_labels[rep_idx],
        pop_seq[rep_idx], pop_edge[rep_idx],
        save_path=panel_a_path, max_seq_len=args.max_seq_len,
        attention_mode=args.attention_mode,
        selection_info=rep_info,
    )
    plt.close('all')

    # ---- 面板 (b): 层演化 + bootstrap CI（基于群体样本）----
    panel_b_metrics, panel_b_per_sample = compute_layer_separation_metrics(
        pop_attn, pop_labels, pop_edge, pop_seq,
        max_seq_len=args.max_seq_len, return_per_sample=True,
    )
    panel_b_path = os.path.join(args.output_dir, f"panel_b_layer_evolution{img_ext}")
    plot_layer_evolution_trend(
        panel_b_metrics, save_path=panel_b_path,
        attention_mode=args.attention_mode,
        per_sample_metrics=panel_b_per_sample,
        show_ci=True, n_bootstrap=args.bootstrap_iters,
    )
    plt.close('all')

    # ---- 面板 (c): violin 图 + bootstrap CI on Cohen's d（基于群体样本）----
    panel_c_path = os.path.join(args.output_dir, f"panel_c_bond_violin{img_ext}")
    _, panel_c_effect = plot_panel_c_violin(
        pop_attn, pop_labels, pop_edge, pop_seq,
        save_path=panel_c_path, attention_mode=args.attention_mode,
        max_seq_len=args.max_seq_len, n_bootstrap=args.bootstrap_iters,
    )
    plt.close('all')

    # ---- 面板 (d): 聚合图层注意力（基于群体样本）----
    panel_d_path = os.path.join(args.output_dir, f"panel_d_aggregate{img_ext}")
    _, panel_d_summary = plot_aggregate_layer_attention(
        pop_attn, pop_labels, pop_edge, pop_seq,
        save_path=panel_d_path, max_seq_len=args.max_seq_len,
        max_bonds_show=min(20, args.max_seq_len - 1),
    )
    plt.close('all')

    # ---- 审稿人 R-22：按长度 / 电荷 / 断裂率的独立补充图 ----
    if has_stat and len(stat_length_groups) == len(stat_attention_weights):
        grouped_specs = [
            (
                "length", stat_length_groups,
                ["Short (<=24)", "Medium (25-29)", "Long (>=30)"],
                "Sequence Length",
            ),
            (
                "charge", stat_charge_groups, None, "Precursor Charge",
            ),
            (
                "fbr", stat_fbr_groups,
                ["Low FBR (<0.3)", "Medium FBR (0.3-0.7)", "High FBR (>=0.7)"],
                "Cleavage Ratio (FBR)",
            ),
        ]
        for key, values, order, title in grouped_specs:
            grouped_path = os.path.join(
                args.output_dir, f"grouped_alignment_{key}{img_ext}",
            )
            _, grouped_summary = plot_grouped_alignment(
                stat_attention_weights, stat_bond_labels,
                stat_edge_indices, stat_sequences,
                group_values=values, group_order=order,
                group_title=title, save_path=grouped_path,
                max_seq_len=args.max_seq_len, n_bootstrap=args.bootstrap_iters,
            )
            plt.close('all')
            grouped_summaries[key] = grouped_summary
            logger.info(f"Saved grouped {key} analysis: {grouped_path}")
    else:
        logger.warning("Grouped analyses skipped because statistical metadata "
                       "are unavailable or misaligned")

    # ---- 修复版合并图（(c)(d) 用统计样本）----
    logger.info("Regenerating combined figure with stat data for (c)(d)...")
    case_study_path = os.path.join(args.output_dir, f"interpretability_case_study_new{img_ext}")
    fig_case, case_summary = plot_new_interpretability_case_study(
        all_attention_weights, all_bond_labels, all_sequences, all_edge_indices,
        save_path=case_study_path, max_seq_len=args.max_seq_len,
        attention_mode=args.attention_mode,
        heatmap_normalize=args.heatmap_normalize,
        stat_attention_weights=pop_attn if has_stat else None,
        stat_bond_labels=pop_labels if has_stat else None,
        stat_sequences=pop_seq if has_stat else None,
        stat_edge_indices=pop_edge if has_stat else None,
        n_bootstrap=args.bootstrap_iters,
    )
    plt.close('all')
    logger.info(f"Saved FIXED combined figure (c)(d) source={pop_source_label}")

    computation_cost_sec = round(time.time() - analysis_start_time, 2)
    logger.info(f"Total analysis wall time: {computation_cost_sec}s")

    # ========== 论文级 JSON 摘要 ==========
    logger.info("Writing paper-grade JSON summary...")
    json_summary = {
        "attention_mode": case_summary.get("attention_mode") if case_summary else args.attention_mode,
        "mode_caption": case_summary.get("mode_caption", ""),
        "sample_selection": {
            "case_study_rule": "length-stratified (select_diverse_samples), seed=42",
            "statistical_rule": ("length (<=24 / 25-29 / >=30) x charge x FBR "
                                 "(<0.3 / 0.3-0.7 / >=0.7) proportional stratified "
                                 "sampling, seed=random_seed+1"),
            "panel_a_representative_rule": rep_info,
            "panel_population_source": pop_source_label,
            "n_case_samples": len(all_attention_weights),
            "n_stat_samples": len(pop_attn),
        },
        "uncertainty": {
            "bootstrap_iters": args.bootstrap_iters,
            "ci_level": 0.95,
            "panel_b_ci": "per-layer bootstrap 95% CI on |r|",
            "panel_c_ci": ("cohen_d_ci_low / cohen_d_ci_high "
                           "(bootstrap 95% CI on signed Cohen's d)"),
        },
        "layer_trend": {
            "case_study": case_summary.get("panel_b_layer_metrics", []) if case_summary else [],
            "statistical": stat_layer_metrics,
            "panel_b_metrics": panel_b_metrics,
            "primary_metric": "abs_r",
            "note": "abs_r is the direction-agnostic strength; "
                    "pearson_r keeps the sign for direction.",
        },
        "effect_size": {
            "case_study": case_summary.get("panel_c_effect_size", {}) if case_summary else {},
            "statistical": stat_effect,
            "panel_c_violin": panel_c_effect,
            "convention": "cohen_d_signed > 0 always means cleavage bonds carry "
                          "the stronger attention signal, regardless of mode.",
        },
        "panels": {
            "panel_a_case_study": f"panel_a_case_study{img_ext}",
            "panel_b_layer_evolution": f"panel_b_layer_evolution{img_ext}",
            "panel_c_bond_violin": f"panel_c_bond_violin{img_ext}",
            "panel_d_aggregate": f"panel_d_aggregate{img_ext}",
            "combined_fixed": f"interpretability_case_study_new{img_ext}",
            "panel_c_source": case_summary.get("panel_c_source", pop_source_label) if case_summary else pop_source_label,
            "panel_d_source": case_summary.get("panel_d_source", pop_source_label) if case_summary else pop_source_label,
        },
        "grouped_robustness": grouped_summaries,
        "physicochemical_stats": {},
        "rule_agreement": {},
        "heatmap_normalize": args.heatmap_normalize,
        "figure_format": args.figure_format,
        "computation_cost_sec": computation_cost_sec,
    }
    json_path = os.path.join(args.output_dir, "interpretability_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved JSON summary to {json_path}")
    
    # ========== 完成 ==========
    logger.info("=" * 60)
    logger.info("Interpretability analysis completed!")
    logger.info("=" * 60)
    
    # 打印输出文件
    print("\n" + "=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)
    
    for root, dirs, files in os.walk(args.output_dir):
        level = root.replace(args.output_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in sorted(files):
            if file.endswith(('.png', '.svg', '.csv', '.json')):
                print(f"{subindent}{file}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Case study samples: {len(case_indices)}")
    if args.num_stat_samples > args.num_samples:
        print(f"Statistical analysis samples: {len(stat_attention_weights)}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # 打印关键结果
    if layer_metrics:
        print("\nLayer Separation Metrics (Case Study):")
        print(f"  primary = abs_r (strength, >=0)  | mode = {args.attention_mode}")
        for i, m in enumerate(layer_metrics):
            print(f"  Layer {i}: |r|={m['abs_r']:.4f}  signed_r={m['pearson_r']:+.4f}  "
                  f"spearman={m['spearman_r']:+.4f}  AUC={m['auc']:.4f}")


if __name__ == "__main__":
    main()
