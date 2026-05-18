#!/usr/bin/env python3
"""
注意力可视化脚本

本脚本用于可视化GraphTransformer模型的注意力权重，支持：
1. 注意力热力图
2. 肽段结构图叠加注意力
3. 注意力头模式分析

使用方法：
python graph_transform/scripts/attention_visualization.py \
    --config graph_transform/config/default.yaml \
    --checkpoint path/to/model.pt \
    --input_csv path/to/test_data.csv \
    --output_dir results/attention_viz \
    --num_samples 5
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

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GraphTransformer
from models.utils import build_model_config, CheckpointManager
from data import GraphDataset, GraphDataLoader, CachedGraphDataset
from utils.attention_extractor import AttentionExtractor
from utils.visualization import (
    plot_attention_heatmap,
    plot_peptide_attention_graph,
    plot_attention_head_comparison,
    analyze_attention_patterns,
    plot_attention_analysis,
    create_attention_report,
)


def setup_logging() -> logging.Logger:
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s[%(levelname)s]:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("attention_visualization")


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


def parse_sample_indices(indices_text: str) -> List[int]:
    """解析样本索引"""
    indices = []
    for part in indices_text.split(","):
        value = part.strip()
        if value:
            indices.append(int(value))
    return indices


def infer_model_config_from_checkpoint(checkpoint_path: str, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    从检查点推断模型配置
    
    Args:
        checkpoint_path: 检查点路径
        base_config: 基础配置
        
    Returns:
        Dict[str, Any]: 推断的配置
    """
    logger = logging.getLogger("attention_visualization")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # 推断配置
    inferred_config = base_config.copy()
    model_config = inferred_config.setdefault('model', {})
    
    # 从 bond_head.0.weight 推断 bond_feature_dim
    if 'bond_head.0.weight' in state_dict:
        bond_weight_shape = state_dict['bond_head.0.weight'].shape
        bond_feature_dim = bond_weight_shape[1]  # [hidden_dim, bond_feature_dim]
        hidden_dim = bond_weight_shape[0]
        
        logger.info(f"Inferred from checkpoint: hidden_dim={hidden_dim}, bond_feature_dim={bond_feature_dim}")
        
        # 推断哪些特征被启用
        # bond_feature_dim = hidden_dim * 2 + (edge_repr ? hidden_dim : 0) + (diff ? hidden_dim : 0) + (product ? hidden_dim : 0)
        base_dim = hidden_dim * 2
        remaining_dim = bond_feature_dim - base_dim
        
        # 默认都启用
        bond_use_edge_repr = True
        bond_use_diff_feature = True
        bond_use_product_feature = True
        
        # 根据剩余维度推断
        if remaining_dim == hidden_dim * 2:
            # 只启用了两个特征
            # 默认禁用 product feature
            bond_use_product_feature = False
            logger.info("Inferred: bond_use_product_feature=False")
        elif remaining_dim == hidden_dim:
            # 只启用了一个特征
            bond_use_diff_feature = False
            bond_use_product_feature = False
            logger.info("Inferred: bond_use_diff_feature=False, bond_use_product_feature=False")
        elif remaining_dim == 0:
            # 没有启用额外特征
            bond_use_edge_repr = False
            bond_use_diff_feature = False
            bond_use_product_feature = False
            logger.info("Inferred: all bond features disabled")
        elif remaining_dim == hidden_dim * 3:
            # 所有特征都启用
            logger.info("Inferred: all bond features enabled")
        else:
            logger.warning(f"Unexpected bond_feature_dim: {bond_feature_dim}, using defaults")
        
        model_config['bond_use_edge_repr'] = bond_use_edge_repr
        model_config['bond_use_diff_feature'] = bond_use_diff_feature
        model_config['bond_use_product_feature'] = bond_use_product_feature
        model_config['hidden_dim'] = hidden_dim
    
    # 从其他权重推断更多配置
    # 例如从 node_encoder 推断 alphabet 大小
    if 'node_encoder.aa_embedding.weight' in state_dict:
        aa_embedding_shape = state_dict['node_encoder.aa_embedding.weight'].shape
        vocab_size = aa_embedding_shape[0]
        logger.info(f"Inferred vocab_size: {vocab_size}")
    
    return inferred_config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="GraphTransformer注意力可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python graph_transform/scripts/attention_visualization.py \
      --config graph_transform/config/default.yaml \
      --checkpoint best_model/graph_transform.pt \
      --input_csv test_data.csv \
      --output_dir results/attention_viz \
      --num_samples 3

  python graph_transform/scripts/attention_visualization.py \
      --config graph_transform/config/default.yaml \
      --checkpoint best_model/graph_transform.pt \
      --input_csv test_data.csv \
      --output_dir results/attention_viz \
      --sample_indices 0,5,10
        """
    )
    
    parser.add_argument("--config", type=str, required=True,
                       help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="模型检查点路径")
    parser.add_argument("--input_csv", type=str, required=True,
                       help="输入CSV文件路径")
    parser.add_argument("--output_dir", type=str, default="results/attention_viz",
                       help="输出目录")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="要可视化的样本数量（用于案例研究）")
    parser.add_argument("--num_stat_samples", type=int, default=500,
                       help="用于统计分析的样本数量（建议500-1000）")
    parser.add_argument("--sample_indices", type=str, default=None,
                       help="指定样本索引，逗号分隔，如 0,5,10")
    parser.add_argument("--skip_case_study", action="store_true",
                       help="跳过案例研究，只进行统计分析")
    parser.add_argument("--sampling_strategy", type=str, choices=["random", "stratified"], 
                       default="stratified",
                       help="抽样策略：random=随机抽样，stratified=按序列长度分层抽样（推荐）")
    parser.add_argument("--num_length_bins", type=int, default=5,
                       help="分层抽样时的序列长度分组数（默认5组）")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="批处理大小")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default=None,
                       help="计算设备")
    parser.add_argument("--max_seq_len", type=int, default=None,
                       help="最大序列长度")
    parser.add_argument("--infer_config", action="store_true",
                       help="从检查点自动推断模型配置（解决配置不匹配问题）")
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging()
    logger.info("Starting attention visualization")
    
    # 检查文件是否存在
    for path_value, label in [
        (args.config, "Config"),
        (args.checkpoint, "Checkpoint"),
        (args.input_csv, "Input CSV"),
    ]:
        if not os.path.exists(path_value):
            raise FileNotFoundError(f"{label} not found: {path_value}")
    
    # 加载配置
    config = load_config(args.config)
    
    # 从检查点推断配置（如果启用）
    if args.infer_config:
        logger.info("Inferring model config from checkpoint...")
        config = infer_model_config_from_checkpoint(args.checkpoint, config)
        logger.info("Using inferred config from checkpoint")
    
    # 设置设备
    if args.device:
        config.setdefault("device", {})["auto_detect"] = False
        config["device"]["device_type"] = args.device
    
    device = setup_device(config)
    logger.info(f"Using device: {device}")
    
    # 加载模型
    model_config = build_model_config(config)
    model = GraphTransformer(model_config).to(device)
    
    # 加载检查点（使用 strict=False 以处理可能的配置差异）
    try:
        CheckpointManager.load_checkpoint(args.checkpoint, model=model, device=device)
        logger.info(f"Loaded checkpoint: {args.checkpoint}")
    except RuntimeError as e:
        if "size mismatch" in str(e) and not args.infer_config:
            logger.warning(f"Config mismatch detected: {e}")
            logger.warning("Try using --infer_config to automatically infer model config from checkpoint")
            logger.warning("Or use the correct config file that matches the checkpoint")
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
    
    # 创建数据集
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
    
    # 选择要可视化的样本
    if args.sample_indices:
        sample_indices = parse_sample_indices(args.sample_indices)
    else:
        # 按序列去重，选择不同的序列进行案例研究
        logger.info(f"Selecting {args.num_samples} diverse sequences for case study...")
        
        if hasattr(test_dataset, 'data') and 'seq' in test_dataset.data.columns:
            # 获取所有唯一序列
            unique_seqs = test_dataset.data['seq'].unique()
            num_unique = len(unique_seqs)
            
            if num_unique <= args.num_samples:
                # 唯一序列不足，直接取前N个
                sample_indices = list(range(min(args.num_samples, len(test_dataset))))
            else:
                # 按序列长度均匀抽样，确保多样性
                import numpy as np
                
                # 获取每个唯一序列的长度
                seq_lengths = {seq: len(str(seq)) for seq in unique_seqs}
                
                # 按长度排序后均匀抽样
                sorted_seqs = sorted(unique_seqs, key=lambda s: seq_lengths[s])
                
                # 均匀选择不同长度的序列
                step = len(sorted_seqs) // args.num_samples
                selected_seqs = [sorted_seqs[i * step] for i in range(args.num_samples)]
                
                # 找到这些序列在数据集中的第一个索引
                sample_indices = []
                for seq in selected_seqs:
                    idx = test_dataset.data[test_dataset.data['seq'] == seq].index[0]
                    sample_indices.append(int(idx))
                
                logger.info(f"Selected sequences with lengths: {[seq_lengths[s] for s in selected_seqs]}")
        else:
            # 无法访问序列信息，随机抽样
            import random
            random.seed(42)
            sample_indices = sorted(random.sample(range(len(test_dataset)), 
                                                  min(args.num_samples, len(test_dataset))))
        
        sample_indices = sorted(sample_indices)
    
    logger.info(f"Will visualize {len(sample_indices)} samples: {sample_indices}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 存储结果
    all_attention_weights = []
    all_sequences = []
    all_bond_labels = []
    
    # 处理每个样本
    for sample_idx in sample_indices:
        if sample_idx >= len(test_dataset):
            logger.warning(f"Sample index {sample_idx} out of range, skipping")
            continue
        
        logger.info(f"Processing sample {sample_idx}")
        
        # 获取样本数据
        sample_data = test_dataset[sample_idx]
        
        # 提取序列（从原始数据中）
        if hasattr(test_dataset, 'data') and 'seq' in test_dataset.data.columns:
            sequence = test_dataset.data.iloc[sample_idx]['seq']
        else:
            # 如果无法获取序列，使用占位符
            sequence = f"Sample_{sample_idx}"
        
        # 提取键断裂标签
        if 'labels' in sample_data:
            bond_labels = sample_data['labels']
        else:
            bond_labels = torch.zeros(10)  # 占位符
        
        # 提取注意力权重
        attention_weights = extractor.extract_attention_for_sample(sample_data)
        
        # 存储结果
        all_attention_weights.append(attention_weights)
        all_sequences.append(sequence)
        all_bond_labels.append(bond_labels)
        
        # 为当前样本生成可视化
        sample_dir = os.path.join(args.output_dir, f"sample_{sample_idx}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # 为每个层生成可视化
        for layer_idx, layer_weights in enumerate(attention_weights):
            # 热力图
            heatmap_path = os.path.join(sample_dir, f"attention_heatmap_layer{layer_idx}.png")
            plot_attention_heatmap(
                layer_weights, layer_idx, sequence=sequence, save_path=heatmap_path
            )
            
            # 肽段结构图
            peptide_path = os.path.join(sample_dir, f"peptide_attention_layer{layer_idx}.png")
            plot_peptide_attention_graph(
                sequence, layer_weights, layer_idx, 
                edge_index=sample_data.get('edge_index'),
                bond_labels=bond_labels, save_path=peptide_path
            )
            
            # 如果是多头注意力，生成头比较图
            if layer_weights.dim() == 2 and layer_weights.shape[1] > 1:
                heads_path = os.path.join(sample_dir, f"attention_heads_layer{layer_idx}.png")
                plot_attention_head_comparison(
                    layer_weights, layer_idx, 
                    edge_index=sample_data.get('edge_index'),
                    sequence=sequence, save_path=heads_path
                )
            
            # 分析注意力模式
            analysis = analyze_attention_patterns(
                layer_weights, bond_labels, layer_idx,
                edge_index=sample_data.get('edge_index')
            )
            
            # 保存分析结果
            analysis_path = os.path.join(sample_dir, f"attention_analysis_layer{layer_idx}.txt")
            with open(analysis_path, 'w') as f:
                f.write(f"Attention Analysis - Sample {sample_idx}, Layer {layer_idx}\n")
                f.write("=" * 50 + "\n\n")
                for key, value in analysis.items():
                    if key not in ['adjacent_weights', 'bond_labels']:
                        f.write(f"{key}: {value}\n")
            
            logger.info(f"  Layer {layer_idx}: Generated visualizations")
    
    # 生成综合分析报告
    if len(all_attention_weights) > 1:
        logger.info("Generating comprehensive analysis report")
        
        # 收集所有层的分析结果
        all_analysis_results = []
        for sample_idx, (attention_weights, sequence, bond_labels) in enumerate(
            zip(all_attention_weights, all_sequences, all_bond_labels)
        ):
            for layer_idx, layer_weights in enumerate(attention_weights):
                analysis = analyze_attention_patterns(
                    layer_weights, bond_labels, layer_idx
                )
                analysis['sample_index'] = sample_idx
                all_analysis_results.append(analysis)
        
        # 绘制综合分析图
        if all_analysis_results:
            analysis_plot_path = os.path.join(args.output_dir, "comprehensive_analysis.png")
            plot_attention_analysis(all_analysis_results, save_path=analysis_plot_path)
    
    # 统计分析（使用更多样本）
    if args.num_stat_samples > args.num_samples:
        logger.info(f"Running statistical analysis with {args.num_stat_samples} samples...")
        
        import random
        random.seed(42)  # 固定随机种子以便复现
        
        total_samples = len(test_dataset)
        stat_sample_count = min(args.num_stat_samples, total_samples)
        
        # 根据抽样策略选择样本
        if args.sampling_strategy == "stratified":
            # 按序列长度分层抽样
            logger.info(f"Using stratified sampling by sequence length ({args.num_length_bins} bins)")
            
            # 获取所有样本的序列长度
            if hasattr(test_dataset, 'data') and 'seq' in test_dataset.data.columns:
                seq_lengths = test_dataset.data['seq'].astype(str).apply(len).values
            else:
                # 如果无法获取序列长度，使用随机抽样
                logger.warning("Cannot access sequence lengths, falling back to random sampling")
                seq_lengths = None
            
            if seq_lengths is not None:
                # 将序列长度分组
                import numpy as np
                length_bins = np.percentile(seq_lengths, np.linspace(0, 100, args.num_length_bins + 1))
                length_bins = np.unique(length_bins)  # 去除重复值
                
                # 为每个分组分配样本数
                bin_counts = np.histogram(seq_lengths, bins=length_bins)[0]
                bin_proportions = bin_counts / bin_counts.sum()
                bin_sample_counts = np.round(bin_proportions * stat_sample_count).astype(int)
                
                # 调整样本数以匹配总数
                diff = stat_sample_count - bin_sample_counts.sum()
                if diff > 0:
                    bin_sample_counts[:int(diff)] += 1
                elif diff < 0:
                    bin_sample_counts[:int(-diff)] -= 1
                
                # 从每个分组中抽样
                stat_indices = []
                for i in range(len(length_bins) - 1):
                    # 获取当前分组的样本索引
                    bin_mask = (seq_lengths >= length_bins[i]) & (seq_lengths < length_bins[i + 1])
                    if i == len(length_bins) - 2:  # 最后一个分组包含右边界
                        bin_mask = (seq_lengths >= length_bins[i]) & (seq_lengths <= length_bins[i + 1])
                    
                    bin_indices = np.where(bin_mask)[0].tolist()
                    
                    # 从当前分组中抽样
                    n_samples = min(bin_sample_counts[i], len(bin_indices))
                    if n_samples > 0:
                        sampled = random.sample(bin_indices, n_samples)
                        stat_indices.extend(sampled)
                
                stat_indices = sorted(stat_indices)
                logger.info(f"Stratified sampling: selected {len(stat_indices)} samples from {len(length_bins)-1} length bins")
                
                # 打印每个分组的样本数
                for i in range(len(length_bins) - 1):
                    bin_mask = (seq_lengths >= length_bins[i]) & (seq_lengths < length_bins[i + 1])
                    if i == len(length_bins) - 2:
                        bin_mask = (seq_lengths >= length_bins[i]) & (seq_lengths <= length_bins[i + 1])
                    bin_count = bin_mask.sum()
                    logger.info(f"  Bin {i}: length [{length_bins[i]:.0f}-{length_bins[i+1]:.0f}], "
                              f"total={bin_count}, sampled={bin_sample_counts[i]}")
            else:
                # 随机抽样
                stat_indices = sorted(random.sample(range(total_samples), stat_sample_count))
        else:
            # 随机抽样
            logger.info("Using random sampling")
            stat_indices = sorted(random.sample(range(total_samples), stat_sample_count))
        
        logger.info(f"Selected {len(stat_indices)} samples for statistical analysis")
        
        # 存储统计分析结果
        stat_analysis_results = []
        
        # 处理每个统计样本
        for idx, sample_idx in enumerate(stat_indices):
            if idx % 100 == 0:
                logger.info(f"Processing statistical sample {idx}/{stat_sample_count}")
            
            try:
                sample_data = test_dataset[sample_idx]
                
                # 提取键断裂标签
                if 'labels' in sample_data:
                    bond_labels = sample_data['labels']
                else:
                    continue
                
                # 提取注意力权重
                attention_weights = extractor.extract_attention_for_sample(sample_data)
                
                # 为每个层进行分析
                for layer_idx, layer_weights in enumerate(attention_weights):
                    analysis = analyze_attention_patterns(
                        layer_weights, bond_labels, layer_idx,
                        edge_index=sample_data.get('edge_index')
                    )
                    analysis['sample_index'] = sample_idx
                    stat_analysis_results.append(analysis)
                    
            except Exception as e:
                logger.warning(f"Error processing sample {sample_idx}: {e}")
                continue
        
        # 绘制统计分析图
        if stat_analysis_results:
            stat_analysis_path = os.path.join(args.output_dir, "comprehensive_analysis_statistical.png")
            plot_attention_analysis(stat_analysis_results, save_path=stat_analysis_path)
            logger.info(f"Statistical analysis saved to {stat_analysis_path}")
            
            # 保存统计结果到CSV
            stat_csv_path = os.path.join(args.output_dir, "statistical_analysis_results.csv")
            import pandas as pd
            stat_df = pd.DataFrame(stat_analysis_results)
            stat_df.to_csv(stat_csv_path, index=False)
            logger.info(f"Statistical results saved to {stat_csv_path}")
    
    logger.info(f"Attention visualization completed. Results saved to {args.output_dir}")
    
    # 打印输出文件列表
    print("\n" + "="*60)
    print("ATTENTION VISUALIZATION RESULTS")
    print("="*60)
    
    for root, dirs, files in os.walk(args.output_dir):
        level = root.replace(args.output_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith(('.png', '.txt', '.csv')):
                print(f"{subindent}{file}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Case study samples: {len(sample_indices)}")
    print(f"Statistical analysis samples: {args.num_stat_samples}")
    print(f"Output directory: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()