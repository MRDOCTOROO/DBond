"""
注意力可视化工具

本模块提供注意力权重的可视化功能，包括热力图、肽段结构图等。
所有图表使用英文标签，确保兼容性。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
import logging
import os

logger = logging.getLogger(__name__)

# 设置matplotlib样式
plt.style.use('default')
sns.set_palette("husl")


def plot_attention_heatmap(attention_weights: torch.Tensor,
                          layer_index: int,
                          head_index: Optional[int] = None,
                          sequence: Optional[str] = None,
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (10, 8),
                          cmap: str = 'viridis',
                          show_values: bool = False) -> plt.Figure:
    """
    绘制注意力权重热力图
    
    Args:
        attention_weights: 注意力权重张量 [num_nodes, num_nodes] 或 [num_heads, num_nodes, num_nodes]
        layer_index: 注意力层索引
        head_index: 注意力头索引（如果为None，则显示所有头的平均值）
        sequence: 肽段序列（用于标签）
        save_path: 保存路径
        figsize: 图形大小
        cmap: 颜色映射
        show_values: 是否显示数值
        
    Returns:
        plt.Figure: matplotlib图形对象
    """
    if attention_weights.dim() == 3:
        if head_index is not None:
            weights = attention_weights[head_index].cpu().numpy()
            title = f"Attention Weights - Layer {layer_index}, Head {head_index}"
        else:
            weights = attention_weights.mean(dim=0).cpu().numpy()
            title = f"Attention Weights - Layer {layer_index}, Average of All Heads"
    else:
        weights = attention_weights.cpu().numpy()
        title = f"Attention Weights - Layer {layer_index}"
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制热力图
    im = ax.imshow(weights, cmap=cmap, aspect='auto', interpolation='nearest')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', fontsize=12)
    
    # 设置标题和标签
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    
    # 设置刻度标签
    if sequence is not None:
        seq_len = len(sequence)
        if seq_len <= 30:  # 只有当序列不太长时才显示字符标签
            ax.set_xticks(range(seq_len))
            ax.set_xticklabels(list(sequence), fontsize=9, rotation=45)
            ax.set_yticks(range(seq_len))
            ax.set_yticklabels(list(sequence), fontsize=9)
    
    # 显示数值
    if show_values and weights.size <= 100:  # 只有当矩阵不太大时才显示数值
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                ax.text(j, i, f'{weights[i, j]:.2f}',
                       ha='center', va='center', fontsize=8,
                       color='white' if weights[i, j] > weights.max() / 2 else 'black')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved attention heatmap to {save_path}")
    
    return fig


def plot_peptide_attention_graph(sequence: str,
                                attention_weights: torch.Tensor,
                                layer_index: int,
                                edge_index: Optional[torch.Tensor] = None,
                                head_index: Optional[int] = None,
                                bond_labels: Optional[torch.Tensor] = None,
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (12, 6),
                                node_size: int = 300,
                                edge_scale: float = 5.0) -> plt.Figure:
    """
    绘制肽段结构图叠加注意力权重
    
    Args:
        sequence: 肽段序列
        attention_weights: 注意力权重张量 [num_edges, num_heads] 或 [num_edges]
        layer_index: 注意力层索引
        edge_index: 边索引 [2, num_edges]，用于映射边权重到节点对
        head_index: 注意力头索引
        bond_labels: 键断裂标签（0或1）
        save_path: 保存路径
        figsize: 图形大小
        node_size: 节点大小
        edge_scale: 边宽度缩放因子
        
    Returns:
        plt.Figure: matplotlib图形对象
    """
    seq_len = len(sequence)
    
    # 处理注意力权重
    if attention_weights.dim() == 2:
        # [num_edges, num_heads]
        if head_index is not None:
            edge_weights_np = attention_weights[:, head_index].cpu().numpy()
            title = f"Peptide Attention - Layer {layer_index}, Head {head_index}"
        else:
            edge_weights_np = attention_weights.mean(dim=1).cpu().numpy()
            title = f"Peptide Attention - Layer {layer_index}, Average of All Heads"
    else:
        # [num_edges]
        edge_weights_np = attention_weights.cpu().numpy()
        title = f"Peptide Attention - Layer {layer_index}"
    
    # 构建节点到节点的注意力矩阵
    num_nodes = seq_len
    if edge_index is not None:
        # 从 edge_index 获取节点数
        num_nodes = int(edge_index.max().item()) + 1
    
    # 使用序列长度和节点数的最小值
    effective_len = min(seq_len, num_nodes)
    
    # 创建节点到节点的注意力矩阵
    attention_matrix = np.zeros((effective_len, effective_len))
    
    if edge_index is not None:
        # 使用 edge_index 映射边权重到节点对
        edge_index_np = edge_index.cpu().numpy()
        for i in range(edge_index_np.shape[1]):
            src, dst = int(edge_index_np[0, i]), int(edge_index_np[1, i])
            if src < effective_len and dst < effective_len:
                # 累加权重（可能有多条边连接同一对节点）
                attention_matrix[src, dst] += edge_weights_np[i]
    else:
        # 如果没有 edge_index，假设是顺序连接
        for i in range(effective_len - 1):
            if i < len(edge_weights_np):
                attention_matrix[i, i + 1] = edge_weights_np[i]
                attention_matrix[i + 1, i] = edge_weights_np[i]
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 左图：肽段序列图（线性）
    G = nx.DiGraph()
    
    # 添加节点（氨基酸）
    for i in range(effective_len):
        G.add_node(i, label=sequence[i])
    
    # 添加边（肽键和注意力权重）
    edges = []
    edge_weights_list = []
    edge_colors = []
    
    # 只考虑相邻节点之间的注意力权重（肽键）
    for i in range(effective_len - 1):
        # 注意力权重（从i到i+1和从i+1到i）
        weight_forward = attention_matrix[i, i + 1]
        weight_backward = attention_matrix[i + 1, i]
        avg_weight = (weight_forward + weight_backward) / 2
        
        edges.append((i, i + 1))
        edge_weights_list.append(avg_weight)
        
        # 根据键断裂标签设置颜色
        if bond_labels is not None and i < bond_labels.size(0):
            if bond_labels[i].item() == 1:
                edge_colors.append('red')  # 断裂的键
            else:
                edge_colors.append('blue')  # 完整的键
        else:
            edge_colors.append('blue')
    
    # 添加边到图
    for i, (src, dst) in enumerate(edges):
        G.add_edge(src, dst, weight=edge_weights_list[i], color=edge_colors[i])
    
    # 绘制图形
    pos = nx.spring_layout(G, seed=42)  # 使用弹簧布局
    pos = {i: (i, 0) for i in range(effective_len)}  # 线性布局
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=node_size, 
                          node_color='lightblue', alpha=0.8)
    
    # 绘制边（宽度根据注意力权重）
    edge_widths = [w * edge_scale for w in edge_weights_list]
    edge_colors_list = [G[u][v]['color'] for u, v in G.edges()]
    
    nx.draw_networkx_edges(G, pos, ax=ax1, width=edge_widths, 
                          edge_color=edge_colors_list, alpha=0.7,
                          arrows=True, arrowsize=10)
    
    # 添加节点标签（氨基酸）
    labels = {i: sequence[i] for i in range(effective_len)}
    nx.draw_networkx_labels(G, pos, labels, ax=ax1, font_size=10)
    
    ax1.set_title(f"Peptide Sequence Graph\n{title}", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 右图：注意力权重矩阵（只显示相邻位置）
    adjacent_weights = np.zeros((effective_len, effective_len))
    for i in range(effective_len - 1):
        adjacent_weights[i, i + 1] = attention_matrix[i, i + 1]
        adjacent_weights[i + 1, i] = attention_matrix[i + 1, i]
    
    im = ax2.imshow(adjacent_weights, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, ax=ax2, label='Attention Weight')
    
    # 设置刻度
    if effective_len <= 30:
        ax2.set_xticks(range(effective_len))
        ax2.set_xticklabels(list(sequence[:effective_len]), fontsize=9, rotation=45)
        ax2.set_yticks(range(effective_len))
        ax2.set_yticklabels(list(sequence[:effective_len]), fontsize=9)
    
    ax2.set_title("Adjacent Attention Weights", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Key Position", fontsize=10)
    ax2.set_ylabel("Query Position", fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved peptide attention graph to {save_path}")
    
    return fig


def plot_attention_head_comparison(attention_weights: torch.Tensor,
                                 layer_index: int,
                                 edge_index: Optional[torch.Tensor] = None,
                                 sequence: Optional[str] = None,
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (15, 10),
                                 max_heads: int = 8) -> plt.Figure:
    """
    比较不同注意力头的权重
    
    Args:
        attention_weights: 注意力权重张量 [num_edges, num_heads] 或 [num_edges]
        layer_index: 注意力层索引
        edge_index: 边索引 [2, num_edges]
        sequence: 肽段序列
        save_path: 保存路径
        figsize: 图形大小
        max_heads: 最大显示头数
        
    Returns:
        plt.Figure: matplotlib图形对象
    """
    # 处理注意力权重
    if attention_weights.dim() == 2:
        # [num_edges, num_heads]
        num_heads = attention_weights.shape[1]
        num_heads = min(num_heads, max_heads)
        edge_weights_np = attention_weights.cpu().numpy()
    else:
        # [num_edges] - 单头情况
        num_heads = 1
        edge_weights_np = attention_weights.cpu().numpy().reshape(-1, 1)
    
    # 计算子图布局
    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f"Attention Heads Comparison - Layer {layer_index}", 
                fontsize=16, fontweight='bold')
    
    # 确定节点数
    num_nodes = 0
    if sequence is not None:
        num_nodes = len(sequence)
    if edge_index is not None:
        num_nodes = max(num_nodes, int(edge_index.max().item()) + 1)
    if num_nodes == 0:
        # 无法确定节点数，使用边数估计
        num_nodes = int(np.sqrt(edge_weights_np.shape[0])) + 1
    
    for head_idx in range(num_heads):
        row = head_idx // cols
        col = head_idx % cols
        ax = axes[row, col]
        
        # 构建节点到节点的注意力矩阵
        attention_matrix = np.zeros((num_nodes, num_nodes))
        
        if edge_index is not None:
            # 使用 edge_index 映射边权重到节点对
            edge_index_np = edge_index.cpu().numpy()
            for i in range(edge_index_np.shape[1]):
                src, dst = int(edge_index_np[0, i]), int(edge_index_np[1, i])
                if src < num_nodes and dst < num_nodes:
                    attention_matrix[src, dst] += edge_weights_np[i, head_idx]
        else:
            # 如果没有 edge_index，假设是顺序连接
            for i in range(min(num_nodes - 1, edge_weights_np.shape[0])):
                attention_matrix[i, i + 1] = edge_weights_np[i, head_idx]
                attention_matrix[i + 1, i] = edge_weights_np[i, head_idx]
        
        im = ax.imshow(attention_matrix, cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax)
        
        ax.set_title(f"Head {head_idx}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Key Position", fontsize=10)
        ax.set_ylabel("Query Position", fontsize=10)
        
        # 设置刻度
        if sequence is not None and len(sequence) <= 20:
            ax.set_xticks(range(len(sequence)))
            ax.set_xticklabels(list(sequence), fontsize=8, rotation=45)
            ax.set_yticks(range(len(sequence)))
            ax.set_yticklabels(list(sequence), fontsize=8)
    
    # 隐藏空的子图
    for idx in range(num_heads, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved attention head comparison to {save_path}")
    
    return fig


def analyze_attention_patterns(attention_weights: torch.Tensor,
                              bond_labels: torch.Tensor,
                              layer_index: int,
                              edge_index: Optional[torch.Tensor] = None,
                              head_index: Optional[int] = None) -> Dict[str, Any]:
    """
    分析注意力模式与键断裂的关系
    
    Args:
        attention_weights: 注意力权重张量 [num_edges, num_heads] 或 [num_edges]
        bond_labels: 键断裂标签（0或1）
        layer_index: 注意力层索引
        edge_index: 边索引 [2, num_edges]
        head_index: 注意力头索引
        
    Returns:
        Dict[str, Any]: 分析结果
    """
    # 处理注意力权重
    if attention_weights.dim() == 2:
        # [num_edges, num_heads]
        if head_index is not None:
            edge_weights_np = attention_weights[:, head_index].cpu().numpy()
        else:
            edge_weights_np = attention_weights.mean(dim=1).cpu().numpy()
    else:
        # [num_edges]
        edge_weights_np = attention_weights.cpu().numpy()
    
    bond_labels_np = bond_labels.cpu().numpy()
    seq_len = len(bond_labels_np)
    
    # 构建节点到节点的注意力矩阵
    num_nodes = seq_len
    if edge_index is not None:
        num_nodes = int(edge_index.max().item()) + 1
    
    attention_matrix = np.zeros((num_nodes, num_nodes))
    
    if edge_index is not None:
        # 使用 edge_index 映射边权重到节点对
        edge_index_np = edge_index.cpu().numpy()
        for i in range(edge_index_np.shape[1]):
            src, dst = int(edge_index_np[0, i]), int(edge_index_np[1, i])
            if src < num_nodes and dst < num_nodes:
                attention_matrix[src, dst] += edge_weights_np[i]
    else:
        # 如果没有 edge_index，假设是顺序连接
        for i in range(min(seq_len - 1, len(edge_weights_np))):
            attention_matrix[i, i + 1] = edge_weights_np[i]
            attention_matrix[i + 1, i] = edge_weights_np[i]
    
    # 计算相邻位置的注意力权重
    adjacent_weights = []
    for i in range(seq_len - 1):
        # 取两个方向的平均值
        w1 = attention_matrix[i, i + 1]
        w2 = attention_matrix[i + 1, i]
        adjacent_weights.append((w1 + w2) / 2)
    
    adjacent_weights = np.array(adjacent_weights)
    
    # 分析断裂键和完整键的注意力权重差异
    broken_mask = bond_labels_np == 1
    intact_mask = bond_labels_np == 0
    
    # 确保长度匹配
    min_len = min(len(adjacent_weights), len(bond_labels_np))
    adjacent_weights = adjacent_weights[:min_len]
    broken_mask = broken_mask[:min_len]
    intact_mask = intact_mask[:min_len]
    
    broken_weights = adjacent_weights[broken_mask] if np.any(broken_mask) else np.array([0])
    intact_weights = adjacent_weights[intact_mask] if np.any(intact_mask) else np.array([0])
    
    analysis = {
        'layer_index': layer_index,
        'head_index': head_index,
        'sequence_length': seq_len,
        'num_broken_bonds': int(np.sum(broken_mask)),
        'num_intact_bonds': int(np.sum(intact_mask)),
        'broken_bond_attention_mean': float(np.mean(broken_weights)),
        'broken_bond_attention_std': float(np.std(broken_weights)),
        'intact_bond_attention_mean': float(np.mean(intact_weights)),
        'intact_bond_attention_std': float(np.std(intact_weights)),
        'attention_difference': float(np.mean(broken_weights) - np.mean(intact_weights)),
        'adjacent_weights': adjacent_weights.tolist(),
        'bond_labels': bond_labels_np.tolist(),
    }
    
    # 计算注意力权重与键断裂的相关性
    if len(adjacent_weights) > 1 and np.std(adjacent_weights) > 0:
        correlation = np.corrcoef(adjacent_weights, bond_labels_np[:min_len])[0, 1]
        analysis['correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
    else:
        analysis['correlation'] = 0.0
    
    return analysis


def plot_attention_analysis(analysis_results: List[Dict[str, Any]],
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    绘制注意力分析结果
    
    Args:
        analysis_results: 分析结果列表
        save_path: 保存路径
        figsize: 图形大小
        
    Returns:
        plt.Figure: matplotlib图形对象
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 提取数据
    layers = [r['layer_index'] for r in analysis_results]
    heads = [r.get('head_index', 'avg') for r in analysis_results]
    broken_means = [r['broken_bond_attention_mean'] for r in analysis_results]
    intact_means = [r['intact_bond_attention_mean'] for r in analysis_results]
    correlations = [r.get('correlation', 0) for r in analysis_results]
    
    # 子图标签
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']
    
    # 图1：断裂键vs完整键的注意力权重
    ax1 = axes[0, 0]
    x = range(len(analysis_results))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], broken_means, width, label='Broken Bonds', color='red', alpha=0.7)
    ax1.bar([i + width/2 for i in x], intact_means, width, label='Intact Bonds', color='blue', alpha=0.7)
    
    ax1.set_xlabel('Layer/Head', fontsize=12)
    ax1.set_ylabel('Mean Attention Weight', fontsize=12)
    ax1.set_title(f'{subplot_labels[0]} Attention on Broken vs Intact Bonds', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2：注意力差异
    ax2 = axes[0, 1]
    differences = [r['attention_difference'] for r in analysis_results]
    colors = ['red' if d > 0 else 'blue' for d in differences]
    
    ax2.bar(x, differences, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Layer/Head', fontsize=12)
    ax2.set_ylabel('Attention Difference (Broken - Intact)', fontsize=12)
    ax2.set_title(f'{subplot_labels[1]} Attention Preference for Broken Bonds', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 图3：相关性
    ax3 = axes[1, 0]
    colors = ['green' if c > 0 else 'orange' for c in correlations]
    
    ax3.bar(x, correlations, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Layer/Head', fontsize=12)
    ax3.set_ylabel('Correlation Coefficient', fontsize=12)
    ax3.set_title(f'{subplot_labels[2]} Attention-Breakage Correlation', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 图4：键数量统计
    ax4 = axes[1, 1]
    broken_counts = [r['num_broken_bonds'] for r in analysis_results]
    intact_counts = [r['num_intact_bonds'] for r in analysis_results]
    
    ax4.bar([i - width/2 for i in x], broken_counts, width, label='Broken Bonds', color='red', alpha=0.7)
    ax4.bar([i + width/2 for i in x], intact_counts, width, label='Intact Bonds', color='blue', alpha=0.7)
    
    ax4.set_xlabel('Layer/Head', fontsize=12)
    ax4.set_ylabel('Number of Bonds', fontsize=12)
    ax4.set_title(f'{subplot_labels[3]} Bond Distribution in Analysis', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved attention analysis plot to {save_path}")
    
    return fig


def create_attention_report(attention_weights_list: List[torch.Tensor],
                           sequences: List[str],
                           bond_labels_list: List[torch.Tensor],
                           output_dir: str,
                           prefix: str = "attention") -> Dict[str, str]:
    """
    创建完整的注意力分析报告
    
    Args:
        attention_weights_list: 注意力权重列表
        sequences: 肽段序列列表
        bond_labels_list: 键断裂标签列表
        output_dir: 输出目录
        prefix: 文件名前缀
        
    Returns:
        Dict[str, str]: 生成的文件路径字典
    """
    os.makedirs(output_dir, exist_ok=True)
    
    generated_files = {}
    
    # 为每个样本生成可视化
    for sample_idx, (attention_weights, sequence, bond_labels) in enumerate(
        zip(attention_weights_list, sequences, bond_labels_list)
    ):
        sample_dir = os.path.join(output_dir, f"sample_{sample_idx}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # 为每个层生成可视化
        for layer_idx, layer_weights in enumerate(attention_weights):
            # 热力图
            heatmap_path = os.path.join(sample_dir, f"{prefix}_heatmap_layer{layer_idx}.png")
            plot_attention_heatmap(
                layer_weights, layer_idx, sequence=sequence, save_path=heatmap_path
            )
            
            # 肽段结构图
            peptide_path = os.path.join(sample_dir, f"{prefix}_peptide_layer{layer_idx}.png")
            plot_peptide_attention_graph(
                sequence, layer_weights, layer_idx, bond_labels=bond_labels, save_path=peptide_path
            )
            
            # 如果是多头注意力，生成头比较图
            if layer_weights.dim() == 3:
                heads_path = os.path.join(sample_dir, f"{prefix}_heads_layer{layer_idx}.png")
                plot_attention_head_comparison(
                    layer_weights, layer_idx, sequence=sequence, save_path=heads_path
                )
            
            # 分析注意力模式
            analysis = analyze_attention_patterns(
                layer_weights, bond_labels, layer_idx
            )
            
            # 保存分析结果
            analysis_path = os.path.join(sample_dir, f"{prefix}_analysis_layer{layer_idx}.txt")
            with open(analysis_path, 'w') as f:
                f.write(f"Attention Analysis - Sample {sample_idx}, Layer {layer_idx}\n")
                f.write("=" * 50 + "\n\n")
                for key, value in analysis.items():
                    if key not in ['adjacent_weights', 'bond_labels']:
                        f.write(f"{key}: {value}\n")
            
            generated_files[f"sample_{sample_idx}_layer_{layer_idx}"] = {
                'heatmap': heatmap_path,
                'peptide': peptide_path,
                'analysis': analysis_path,
            }
            
            if layer_weights.dim() == 3:
                generated_files[f"sample_{sample_idx}_layer_{layer_idx}"]['heads'] = heads_path
    
    logger.info(f"Generated attention report in {output_dir}")
    return generated_files


def plot_peptide_attention_combined(sequence: str,
                                   attention_weights_list: List[torch.Tensor],
                                   edge_index: Optional[torch.Tensor] = None,
                                   bond_labels: Optional[torch.Tensor] = None,
                                   save_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (8, 7),
                                   save_individual: bool = True) -> plt.Figure:
    """
    为学术论文优化的紧凑型注意力热力图（去除冗余的序列图）
    
    Args:
        sequence: 肽段序列
        attention_weights_list: 各层注意力权重列表 [layer0_weights, layer1_weights, ...]
        edge_index: 边索引 [2, num_edges]
        bond_labels: 键断裂标签（0或1）
        save_path: 保存路径（总图）
        figsize: 图形大小（宽, 高），默认正方形
        save_individual: 是否同时保存单层图片
        
    Returns:
        plt.Figure: matplotlib图形对象
    """
    import seaborn as sns
    
    num_layers = len(attention_weights_list)
    seq_len = len(sequence)
    
    # 计算有效长度
    num_nodes = seq_len
    if edge_index is not None:
        num_nodes = int(edge_index.max().item()) + 1
    effective_len = min(seq_len, num_nodes)
    
    # 生成各层的注意力矩阵
    all_matrices = []
    for layer_idx, attention_weights in enumerate(attention_weights_list):
        # 处理注意力权重
        if attention_weights.dim() == 2:
            edge_weights_np = attention_weights.mean(dim=1).cpu().numpy()
        else:
            edge_weights_np = attention_weights.cpu().numpy()
        
        # 创建注意力矩阵（只保留相邻位置）
        attention_matrix = np.zeros((effective_len, effective_len))
        
        if edge_index is not None:
            edge_index_np = edge_index.cpu().numpy()
            for i in range(edge_index_np.shape[1]):
                src, dst = int(edge_index_np[0, i]), int(edge_index_np[1, i])
                if src < effective_len and dst < effective_len:
                    attention_matrix[src, dst] += edge_weights_np[i]
        else:
            for i in range(effective_len - 1):
                if i < len(edge_weights_np):
                    attention_matrix[i, i + 1] = edge_weights_np[i]
                    attention_matrix[i + 1, i] = edge_weights_np[i]
        
        # 只保留相邻位置的权重
        adjacent_weights = np.zeros((effective_len, effective_len))
        for i in range(effective_len - 1):
            adjacent_weights[i, i + 1] = attention_matrix[i, i + 1]
            adjacent_weights[i + 1, i] = attention_matrix[i + 1, i]
        
        all_matrices.append(adjacent_weights)
        
        # 保存单层图片（用于并排摆放）
        if save_individual and save_path:
            individual_path = save_path.replace('.png', f'_layer{layer_idx}.png')
            plot_peptide_attention_compact(
                adjacent_weights, sequence[:effective_len], 
                layer_idx, individual_path
            )
    
    # 绘制合并图（所有层横向排列）
    fig, axes = plt.subplots(1, num_layers, figsize=figsize, dpi=300)
    if num_layers == 1:
        axes = [axes]
    
    # 设置子图间距
    plt.subplots_adjust(wspace=0.3, left=0.08, right=0.95, top=0.92, bottom=0.08)
    
    seq_list = list(sequence[:effective_len])
    
    for layer_idx, (ax, matrix) in enumerate(zip(axes, all_matrices)):
        # 绘制热力图
        sns.heatmap(
            matrix,
            cmap='YlOrRd',
            ax=ax,
            cbar_kws={'label': 'Attention Weight', 'shrink': 0.8},
            square=True,
            linewidths=0.5,
            linecolor='white'
        )
        
        # 设置坐标轴标签
        ax.set_xticks(np.arange(len(seq_list)) + 0.5)
        ax.set_yticks(np.arange(len(seq_list)) + 0.5)
        
        # 设置刻度标签
        ax.set_xticklabels(seq_list, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(seq_list, rotation=0, fontsize=8)
        
        # 设置轴标题
        ax.set_xlabel('Key Position (Residue)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Query Position (Residue)', fontsize=10, fontweight='bold')
        
        # 设置子图标题
        title = f'( {"a" if layer_idx == 0 else "b"} ) Layer {layer_idx} Attention Weights'
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    # 添加总标题
    seq_display = sequence[:20] + ('...' if len(sequence) > 20 else '')
    fig.suptitle(f'Peptide: {seq_display}', fontsize=13, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved combined peptide attention to {save_path}")
    
    return fig


def plot_peptide_attention_compact(attention_matrix: np.ndarray,
                                  sequence: str,
                                  layer_idx: int,
                                  save_path: Optional[str] = None,
                                  figsize: Tuple[int, int] = (8, 7)) -> plt.Figure:
    """
    为学术论文优化的单层紧凑型注意力热力图
    
    Args:
        attention_matrix: (N, N) 的注意力权重 numpy 数组
        sequence: 长度为 N 的氨基酸序列字符串
        layer_idx: 当前的层数（用于标题）
        save_path: 保存路径
        figsize: 图形大小（宽, 高）
        
    Returns:
        plt.Figure: matplotlib图形对象
    """
    import seaborn as sns
    
    # 创建画布
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    # 绘制热力图
    sns.heatmap(
        attention_matrix,
        cmap='YlOrRd',
        ax=ax,
        cbar_kws={'label': 'Attention Weight'},
        square=True,
        linewidths=0.5,
        linecolor='white'
    )
    
    # 设置坐标轴标签
    seq_list = list(sequence)
    ax.set_xticks(np.arange(len(seq_list)) + 0.5)
    ax.set_yticks(np.arange(len(seq_list)) + 0.5)
    
    # 设置刻度标签，并旋转 x 轴标签以防重叠
    ax.set_xticklabels(seq_list, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(seq_list, rotation=0, fontsize=10)
    
    # 设置轴标题和图表主标题
    ax.set_xlabel('Key Position (Residue)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Query Position (Residue)', fontsize=12, fontweight='bold')
    
    # 设置主标题
    title = f'Adjacent Attention Weights - Layer {layer_idx}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # 调整边缘并保存
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved compact attention heatmap to {save_path}")
    
    return fig


def plot_attention_heads_combined(attention_weights_list: List[torch.Tensor],
                                 edge_index: Optional[torch.Tensor] = None,
                                 sequence: Optional[str] = None,
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (16, 8),
                                 max_heads: int = 4) -> plt.Figure:
    """
    绘制多层多头注意力的合并版本（横向排列）
    
    Args:
        attention_weights_list: 各层注意力权重列表 [layer0_weights, layer1_weights, ...]
        edge_index: 边索引 [2, num_edges]
        sequence: 肽段序列
        save_path: 保存路径
        figsize: 图形大小
        max_heads: 每层最多显示的头数
        
    Returns:
        plt.Figure: matplotlib图形对象
    """
    num_layers = len(attention_weights_list)
    
    # 确定每层显示的头数
    heads_per_layer = min(max_heads, attention_weights_list[0].shape[1] if attention_weights_list[0].dim() == 2 else 1)
    
    # 创建子图布局：每层 heads_per_layer 个图
    fig, axes = plt.subplots(num_layers, heads_per_layer, figsize=figsize)
    
    if num_layers == 1:
        axes = axes.reshape(1, -1)
    if heads_per_layer == 1:
        axes = axes.reshape(-1, 1)
    
    plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.05, right=0.95, top=0.9, bottom=0.1)
    
    # 确定节点数
    num_nodes = 0
    if sequence is not None:
        num_nodes = len(sequence)
    if edge_index is not None:
        num_nodes = max(num_nodes, int(edge_index.max().item()) + 1)
    
    for layer_idx, attention_weights in enumerate(attention_weights_list):
        for head_idx in range(heads_per_layer):
            ax = axes[layer_idx, head_idx]
            
            # 构建该头的注意力矩阵
            attention_matrix = np.zeros((num_nodes, num_nodes))
            
            if attention_weights.dim() == 2:
                # [num_edges, num_heads]
                edge_weights_np = attention_weights[:, head_idx].cpu().numpy()
            else:
                edge_weights_np = attention_weights.cpu().numpy()
            
            if edge_index is not None:
                edge_index_np = edge_index.cpu().numpy()
                for i in range(edge_index_np.shape[1]):
                    src, dst = int(edge_index_np[0, i]), int(edge_index_np[1, i])
                    if src < num_nodes and dst < num_nodes:
                        attention_matrix[src, dst] += edge_weights_np[i]
            
            im = ax.imshow(attention_matrix, cmap='viridis', aspect='auto')
            
            ax.set_title(f"L{layer_idx} Head {head_idx}", fontsize=10, fontweight='bold')
            
            if sequence is not None and len(sequence) <= 15:
                ax.set_xticks(range(len(sequence)))
                ax.set_xticklabels(list(sequence), fontsize=7, rotation=45)
                ax.set_yticks(range(len(sequence)))
                ax.set_yticklabels(list(sequence), fontsize=7)
            
            if head_idx == 0:
                ax.set_ylabel("Query", fontsize=9)
            if layer_idx == num_layers - 1:
                ax.set_xlabel("Key", fontsize=9)
    
    fig.suptitle("Multi-head Attention Comparison", fontsize=14, fontweight='bold')
    
    # 添加颜色条
    cbar_ax = fig.add_axes([0.96, 0.1, 0.015, 0.8])
    plt.colorbar(im, cax=cbar_ax, label='Attention Weight')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved combined attention heads to {save_path}")
    
    return fig


# ============================================================================
# Paper-grade interpretability analysis (unified functional-saliency frame)
#
# Canonical semantic convention
# -----------------------------
# Attention is interpreted as a FUNCTIONAL SALIENCY SCORE: it represents the
# model's structural role specialization in bond cleavage prediction. We do
# NOT label attention as a "cleavage score" or "stability score"; attention
# marks which bonds play a functionally important role in differentiating bond
# outcomes, and its relationship to the cleavage label is reported through
# alignment statistics rather than causal claims.
#
# Modes (kept for backward compatibility; all unify under the functional frame):
#      "functional" / "cleavage" / "importance" / "auto" / "stability"
#      Every mode renders the SAME functional-saliency narrative on the figures.
#      The mode value only governs the internal sign convention for statistics;
#      "stability" is DEPRECATED and remapped to the canonical mode.
#
# Consistency guarantees
# ----------------------
# 1. Every statistic is orientation-consistent by construction:
#      - Pearson r / Spearman r:  r > 0  =>  attention co-varies with label=1
#      - Cohen's d (signed):      d > 0  =>  broken bonds have higher saliency
#      - AUC:                    > 0.5  =>  broken bonds have higher saliency
#    These are reported as "alignment strength", never as causal effects.
# 2. Layer-wise evolution uses abs(r) as PRIMARY (alignment strength, >= 0);
#    the signed raw r is shown as a secondary gray dashed line.
# 3. All axis labels, legends, and captions use the unified vocabulary:
#    "Functional Saliency (Attention)" and "Attention–Label Alignment".
#    Boxplots use non-causal wording ("associated with" / "differentiates").
# 4. A single shared palette (INTERP_COLORS) and a single bond-indexing helper
#    (extract_bond_level_attention) are reused by every panel so colors,
#    indexing, and normalization are consistent across (a)-(d).
# ============================================================================

# Shared palette — imported by every panel for cross-figure consistency.
INTERP_COLORS = {
    'broken': '#E74C3C',      # cleaved bonds (label=1)
    'intact': '#3498DB',      # non-cleaved bonds (label=0)
    'line': '#2C3E50',        # primary lines / text
    'highlight': '#F39C12',   # top-k annotations
    'raw_r': '#95A5A6',       # secondary (raw signed r) trace
    'fill': '#3498DB',        # area fills
}

VALID_ATTENTION_MODES = {"functional", "cleavage", "importance", "auto", "stability"}

# Canonical default / unified interpretation mode.
DEFAULT_ATTENTION_MODE = "functional"

# Unified, non-causal caption shown on every figure (single source of truth).
FUNCTIONAL_SALIENCY_CAPTION = (
    "Attention = functional saliency "
    "(structural role specialization in bond cleavage prediction)"
)


def _resolve_attention_mode(mode: str, broken_weights=None, intact_weights=None) -> str:
    """Resolve a (possibly deprecated/auto) mode into the canonical concrete mode.

    "stability" is DEPRECATED -> remapped to the canonical mode.
    "auto" -> detected from the sign of mean(broken) - mean(intact), falling
              back to the canonical mode when data is unavailable.
    """
    if mode not in VALID_ATTENTION_MODES:
        raise ValueError(
            f"Unknown attention mode '{mode}'. "
            f"Expected one of {sorted(VALID_ATTENTION_MODES)}."
        )
    if mode == "stability":
        logger.warning(
            "Attention mode 'stability' is DEPRECATED and remapped to the "
            "unified functional-saliency interpretation. Update your call site."
        )
        return DEFAULT_ATTENTION_MODE
    if mode in ("functional", "cleavage", "importance"):
        return mode
    # auto
    if broken_weights is None or intact_weights is None:
        return DEFAULT_ATTENTION_MODE
    diff = float(np.mean(broken_weights) - np.mean(intact_weights))
    return DEFAULT_ATTENTION_MODE if diff > 0 else "importance"


def _mode_caption(mode: str = DEFAULT_ATTENTION_MODE) -> str:
    """Unified, non-causal figure caption (single source of truth).

    Returns the same functional-saliency statement for every mode so all
    figures share one interpretation framework.
    """
    return FUNCTIONAL_SALIENCY_CAPTION


def _save_figure(fig: plt.Figure, save_path: Optional[str]) -> None:
    """Save a figure, auto-selecting SVG (vector, editable in browser/Illustrator)
    or PNG (raster) from the file extension."""
    if save_path is None:
        return
    path_str = str(save_path)
    if path_str.lower().endswith('.svg'):
        fig.savefig(path_str, format='svg', bbox_inches='tight')
    else:
        fig.savefig(path_str, dpi=300, bbox_inches='tight')
    logger.info(f"Saved figure to {save_path}")


def detect_attention_mode(broken_weights, intact_weights) -> str:
    """Auto-detect attention semantics from the sign of the group-mean gap."""
    return _resolve_attention_mode("auto", broken_weights, intact_weights)


def compute_effect_size(broken_weights, intact_weights, mode: str = "auto") -> Dict[str, float]:
    """Unified effect-size computation.

    Guarantees that the returned ``cohen_d_signed`` is oriented so that
    **positive always means cleavage bonds carry the stronger signal**:
    cohen_d_signed = (mean(broken) - mean(intact)) / pooled_std.

    The ``mode`` argument only controls the ``interpretation`` narrative string,
    NOT the numerical direction, so the statistic is unambiguous to reviewers.

    Returns a dict with: raw_mean_diff, cohen_d_signed, cohen_d_abs, t_stat,
    p_value, auc, n_broken, n_intact, mode, interpretation.
    """
    from scipy import stats

    broken = np.asarray(broken_weights, dtype=float)
    intact = np.asarray(intact_weights, dtype=float)
    mode = _resolve_attention_mode(mode, broken, intact)

    n_b, n_i = len(broken), len(intact)
    mean_diff = float(np.mean(broken) - np.mean(intact)) if n_b and n_i else 0.0
    pooled_std = float(np.sqrt((np.var(broken) + np.var(intact)) / 2)) if n_b and n_i else 0.0
    cohen_d_signed = mean_diff / pooled_std if pooled_std > 0 else 0.0

    t_stat, p_value = (float('nan'), float('nan'))
    if n_b > 1 and n_i > 1:
        t_stat, p_value = stats.ttest_ind(broken, intact, equal_var=False)

    # AUC: P(broken attention > intact attention). > 0.5 -> broken bonds higher saliency.
    auc = 0.5
    if n_b and n_i:
        try:
            from sklearn.metrics import roc_auc_score
            labels = np.concatenate([np.ones(n_b), np.zeros(n_i)])
            scores = np.concatenate([broken, intact])
            auc = float(roc_auc_score(labels, scores))
        except Exception:
            auc = 0.5

    # Narrative — the NUMBER is invariant; wording is non-causal and unified.
    resolved = _resolve_attention_mode(mode, broken, intact)
    if resolved == "importance":
        interpretation = "Functional saliency (direction-agnostic)"
    else:
        interpretation = (
            "Saliency associated with broken bonds (broken > intact)" if cohen_d_signed > 0
            else "Saliency associated with intact bonds (broken < intact)"
        )

    return {
        "mode": resolved,
        "raw_mean_diff": mean_diff,
        "cohen_d_signed": cohen_d_signed,
        "cohen_d_abs": float(abs(cohen_d_signed)),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "auc": auc,
        "n_broken": n_b,
        "n_intact": n_i,
        "interpretation": interpretation,
    }


def compute_separation_metrics(bond_attn: np.ndarray, bond_labels_np: np.ndarray) -> Dict[str, float]:
    """Compute the full set of attention/breakage separation metrics for one
    layer-sample pair.

    Returns: pearson_r (signed), abs_r, spearman_r (signed), auc (signed,
    >0.5 means broken bonds have higher attention), separation_auc
    (direction-agnostic, in [0.5, 1.0]).
    """
    from scipy import stats

    bond_attn = np.asarray(bond_attn, dtype=float)
    bond_labels_np = np.asarray(bond_labels_np, dtype=float)
    n = min(len(bond_attn), len(bond_labels_np))
    bond_attn = bond_attn[:n]
    bond_labels_np = bond_labels_np[:n]

    out = {"pearson_r": 0.0, "abs_r": 0.0, "spearman_r": 0.0,
           "auc": 0.5, "separation_auc": 0.5, "n": int(n)}

    if n < 3 or np.std(bond_attn) == 0 or np.std(bond_labels_np) == 0:
        return out

    r = float(np.corrcoef(bond_attn, bond_labels_np)[0, 1])
    if np.isnan(r):
        r = 0.0
    out["pearson_r"] = r
    out["abs_r"] = abs(r)

    try:
        rho = float(stats.spearmanr(bond_attn, bond_labels_np).correlation)
        if np.isnan(rho):
            rho = 0.0
    except Exception:
        rho = 0.0
    out["spearman_r"] = rho

    if bond_labels_np.sum() > 0 and (bond_labels_np == 0).sum() > 0:
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(bond_labels_np, bond_attn))
        except Exception:
            auc = 0.5
    else:
        auc = 0.5
    out["auc"] = auc
    out["separation_auc"] = float(max(auc, 1.0 - auc))
    return out


def _normalize_heatmap(matrix: np.ndarray, mode: str) -> np.ndarray:
    """Normalize a [num_layers, num_bonds] attention matrix.

    mode = 'row'    -> each layer normalized to [0,1] (within-layer pattern)
    mode = 'global' -> shared scale across layers (cross-layer magnitude)
    """
    matrix = np.asarray(matrix, dtype=float)
    if mode == "row":
        out = np.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            row_max = matrix[i].max()
            out[i] = matrix[i] / row_max if row_max > 0 else matrix[i]
        return out
    # global
    g_max = matrix.max()
    return matrix / g_max if g_max > 0 else matrix


def extract_bond_level_attention(
    attention_weights: torch.Tensor,
    edge_index: Optional[torch.Tensor],
    sequence: str,
    max_seq_len: int = 30
) -> Tuple[np.ndarray, List[str]]:
    """Extract peptide-bond-level attention from edge-level weights.

    Returns (bond_attn, bond_labels) where bond_labels[i] = "X-Y" for the
    peptide bond between residue i and i+1. This is the SINGLE source of truth
    for bond indexing across all interpretability panels.
    """
    seq_len = min(len(sequence), max_seq_len)

    if attention_weights.dim() == 2:
        attn_np = attention_weights.mean(dim=1).cpu().numpy()
    else:
        attn_np = attention_weights.cpu().numpy()

    bond_attn = np.zeros(seq_len - 1)
    bond_counts = np.zeros(seq_len - 1)

    if edge_index is not None:
        edge_index_np = edge_index.cpu().numpy()
        for i in range(edge_index_np.shape[1]):
            src, dst = int(edge_index_np[0, i]), int(edge_index_np[1, i])
            # 仅取相邻残基边（肽键边）
            if abs(src - dst) != 1:
                continue
            # 排除涉及 global node（位于 seq_len 索引）的边。
            # 修复：旧版条件 `src < seq_len-1 and dst < seq_len` 不对称，
            # 导致最后一个键 (seq_len-2, seq_len-1) 的反向边 (seq_len-1 → seq_len-2)
            # 被错误丢弃（因为 src=seq_len-1 不满足 < seq_len-1）。
            # 新版条件保留所有合法肽键边的双向贡献，仍能正确排除 global node。
            if not (min(src, dst) < seq_len - 1 and max(src, dst) < seq_len):
                continue
            bond_pos = min(src, dst)
            if i < len(attn_np):
                bond_attn[bond_pos] += attn_np[i]
                bond_counts[bond_pos] += 1
    else:
        for i in range(min(seq_len - 1, len(attn_np))):
            bond_attn[i] = attn_np[i]
            bond_counts[i] = 1

    bond_counts = np.maximum(bond_counts, 1)
    bond_attn = bond_attn / bond_counts
    bond_labels = [f"{sequence[i]}-{sequence[i+1]}" for i in range(seq_len - 1)]
    return bond_attn, bond_labels


def plot_single_sample_layer_attention(
    attention_weights_list: List[torch.Tensor],
    bond_labels: torch.Tensor,
    sequence: str,
    edge_index: Optional[torch.Tensor],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
    max_seq_len: int = 25,
    attention_mode: str = DEFAULT_ATTENTION_MODE,
) -> plt.Figure:
    """Per-layer attention bar plot for a single peptide (case study).

    Attention is a functional-saliency score. Title reports |r| (alignment
    strength) and signed r (direction); r > 0 means saliency co-varies with
    the cleavage label.
    """
    num_layers = len(attention_weights_list)
    seq_len = min(len(sequence), max_seq_len)
    num_bonds = seq_len - 1

    all_bond_attn = []
    for layer_weights in attention_weights_list:
        bond_attn, _ = extract_bond_level_attention(layer_weights, edge_index, sequence, max_seq_len)
        all_bond_attn.append(bond_attn)

    bond_labels_np = bond_labels[:num_bonds].cpu().numpy() if isinstance(bond_labels, torch.Tensor) else np.asarray(bond_labels[:num_bonds])

    fig, axes = plt.subplots(1, num_layers, figsize=figsize, sharey=True)
    if num_layers == 1:
        axes = [axes]
    plt.subplots_adjust(wspace=0.15, left=0.08, right=0.95, top=0.84, bottom=0.18)

    x = np.arange(num_bonds)
    _, bond_labels_str = extract_bond_level_attention(attention_weights_list[0], edge_index, sequence, max_seq_len)
    for ax, bond_attn in zip(axes, all_bond_attn):
        for i in range(num_bonds):
            if i < len(bond_labels_np) and bond_labels_np[i] == 1:
                ax.axvspan(i - 0.4, i + 0.4, alpha=0.15, color=INTERP_COLORS['broken'])
        bar_colors = [INTERP_COLORS['broken'] if (i < len(bond_labels_np) and bond_labels_np[i] == 1)
                      else INTERP_COLORS['intact'] for i in range(num_bonds)]
        ax.bar(x, bond_attn[:num_bonds], color=bar_colors, alpha=0.7, edgecolor='white')

        top_indices = np.argsort(bond_attn[:num_bonds])[-min(3, num_bonds):]
        for idx in top_indices:
            ax.annotate(bond_labels_str[idx], xy=(idx, bond_attn[idx]),
                        xytext=(0, 8), textcoords='offset points',
                        ha='center', fontsize=7, fontweight='bold',
                        color=INTERP_COLORS['highlight'])

        m = compute_separation_metrics(bond_attn[:num_bonds], bond_labels_np)
        ax.set_xlabel('Bond Position (Residue Pair)', fontsize=9)
        ax.set_title(f"alignment |r|={m['abs_r']:.2f}  (signed r={m['pearson_r']:+.2f})",
                     fontsize=10, fontweight='bold')
        ax.set_xlim(-0.5, num_bonds - 0.5)
        if num_bonds <= 15:
            ax.set_xticks(x)
            ax.set_xticklabels(bond_labels_str, fontsize=7, rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

    axes[0].set_ylabel('Functional Saliency (Attention)', fontsize=10)
    seq_display = sequence[:20] + ('...' if len(sequence) > 20 else '')
    fig.suptitle(f'Sample Peptide: {seq_display}\n'
                 f'Cleavage-Relevance Attention Distribution\n'
                 f'{_mode_caption(attention_mode)}',
                 fontsize=10, fontweight='bold', y=0.99)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=INTERP_COLORS['broken'], alpha=0.7, label='Broken Bond (label=1)'),
        Patch(facecolor=INTERP_COLORS['intact'], alpha=0.7, label='Intact Bond (label=0)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 0.02))

    _save_figure(fig, save_path)
    return fig


def compute_layer_separation_metrics(
    attention_weights_list: List,
    bond_labels_list: List[torch.Tensor],
    edge_indices: List[Optional[torch.Tensor]],
    sequences: List[str],
    max_seq_len: int = 30,
) -> List[Dict[str, float]]:
    """Per-layer separation metrics, averaged across samples.

    Returns a list (one dict per layer) of {pearson_r, abs_r, spearman_r,
    auc, separation_auc}. Use abs_r / separation_auc as the primary,
    direction-agnostic strength metric for the evolution trend.
    """
    if not attention_weights_list or not attention_weights_list[0]:
        return []
    num_layers = len(attention_weights_list[0])
    per_layer: List[List[Dict[str, float]]] = [[] for _ in range(num_layers)]

    for attn_weights, labels, edge_idx, seq in zip(
        attention_weights_list, bond_labels_list, edge_indices, sequences
    ):
        labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
        for layer_idx in range(min(num_layers, len(attn_weights))):
            bond_attn, _ = extract_bond_level_attention(attn_weights[layer_idx], edge_idx, seq, max_seq_len)
            m = compute_separation_metrics(bond_attn, labels_np)
            if m["n"] >= 3:
                per_layer[layer_idx].append(m)

    summary = []
    keys = ("pearson_r", "abs_r", "spearman_r", "auc", "separation_auc")
    for samples in per_layer:
        if not samples:
            summary.append({k: 0.0 for k in keys} | {"n_samples": 0})
            continue
        agg = {k: float(np.mean([s[k] for s in samples])) for k in keys}
        agg["n_samples"] = len(samples)
        summary.append(agg)
    return summary


def compute_layer_correlations(
    attention_weights_list: List,
    bond_labels_list: List[torch.Tensor],
    edge_indices: List[Optional[torch.Tensor]],
    sequences: List[str],
    max_seq_len: int = 30,
) -> List[float]:
    """Backward-compatible wrapper: returns abs(r) per layer (the primary,
    direction-agnostic strength metric). Prefer compute_layer_separation_metrics
    for the full metric set."""
    metrics = compute_layer_separation_metrics(
        attention_weights_list, bond_labels_list, edge_indices, sequences, max_seq_len
    )
    return [m["abs_r"] for m in metrics]


def plot_layer_evolution_trend(
    layer_metrics,
    layer_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5),
    attention_mode: str = DEFAULT_ATTENTION_MODE,
    primary_metric: str = "abs_r",
) -> plt.Figure:
    """Layer-wise evolution of the attention–label alignment.

    Attention is a functional-saliency score. PRIMARY (solid): |r| — alignment
    strength, always >= 0, so an increasing trend means attention becomes more
    informative about bond-role differentiation with depth. SECONDARY (gray
    dashed): signed raw r — keeps the direction visible; r > 0 means saliency
    co-varies with the cleavage label (broken bonds receive higher saliency).

    ``layer_metrics`` may be either:
      * a list of floats (treated as the primary metric directly), or
      * a list of dicts as produced by compute_layer_separation_metrics().
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize input to (primary_values, raw_r_values)
    if layer_metrics and isinstance(layer_metrics[0], dict):
        primary_vals = [float(m[primary_metric]) for m in layer_metrics]
        raw_r_vals = [float(m["pearson_r"]) for m in layer_metrics]
    else:
        primary_vals = [float(v) for v in layer_metrics]
        raw_r_vals = primary_vals  # fall back; no signed info available

    num_layers = len(primary_vals)
    x = np.arange(num_layers)
    if layer_names is None:
        layer_names = [f'Layer {i}' for i in range(num_layers)]

    # Secondary: signed raw r (gray dashed) — alignment strength (direction)
    if raw_r_vals is not primary_vals:
        ax.plot(x, raw_r_vals, 'o--', color=INTERP_COLORS['raw_r'], linewidth=1.5,
                markersize=6, alpha=0.7, label='Signed r (alignment strength, direction)', zorder=2)
        for xi, yi in zip(x, raw_r_vals):
            ax.annotate(f'{yi:+.2f}', xy=(xi, yi), xytext=(0, -16),
                        textcoords='offset points', ha='center', fontsize=8,
                        color=INTERP_COLORS['raw_r'])

    # Primary: abs(r) — alignment strength
    ax.plot(x, primary_vals, 'o-', color=INTERP_COLORS['line'], linewidth=2.5,
            markersize=11, markerfacecolor=INTERP_COLORS['broken'],
            markeredgecolor='black', markeredgewidth=1.5,
            label=f'{primary_metric} (alignment strength)', zorder=4)
    for xi, yi in zip(x, primary_vals):
        ax.annotate(f'{yi:.3f}', xy=(xi, yi), xytext=(0, 12),
                    textcoords='offset points', ha='center', fontsize=10,
                    fontweight='bold', color=INTERP_COLORS['line'])
    ax.fill_between(x, primary_vals, alpha=0.15, color=INTERP_COLORS['fill'])

    # Linear trend on the primary metric
    if num_layers > 1:
        z = np.polyfit(x, primary_vals, 1)
        ax.plot(x, np.poly1d(z)(x), ':', color=INTERP_COLORS['highlight'],
                alpha=0.8, linewidth=1.5, label=f'Trend slope={z[0]:+.4f}')

    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Network Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{primary_metric}  (alignment strength, ≥ 0)', fontsize=12, fontweight='bold')
    ax.set_title('Layer-wise Attention–Label Alignment\n'
                 f'{_mode_caption(attention_mode)}',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, fontsize=10)
    lo = min(min(primary_vals), min(raw_r_vals) if raw_r_vals else 0)
    hi = max(max(primary_vals), max(raw_r_vals) if raw_r_vals else 0)
    ax.set_ylim(lo - 0.12, hi + 0.18)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

    # Reviewer-proof annotation (non-causal, unified frame)
    note = ("Increasing |r| ⇒ stronger attention–label alignment.\n"
            "Signed r > 0 ⇒ higher saliency associated with broken bonds.")
    ax.text(0.02, 0.98, note, transform=ax.transAxes, fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_bond_type_comparison(
    attention_weights_list: List,
    bond_labels_list: List[torch.Tensor],
    edge_indices: List[Optional[torch.Tensor]],
    sequences: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    attention_mode: str = "auto",
    max_seq_len: int = 30,
) -> Tuple[plt.Figure, Dict[str, float]]:
    """Boxplot of attention on broken vs intact bonds, annotated with the
    UNIFIED effect size (positive ⇒ cleavage bonds stronger) and the active
    semantic mode. Returns (fig, effect_size_dict) so callers can serialize
    the stats into the JSON summary.
    """
    broken_weights, intact_weights = [], []
    for attn_weights, labels, edge_idx, seq in zip(
        attention_weights_list, bond_labels_list, edge_indices, sequences
    ):
        last_layer = attn_weights[-1] if isinstance(attn_weights, list) else attn_weights
        bond_attn, _ = extract_bond_level_attention(last_layer, edge_idx, seq, max_seq_len)
        labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
        for i in range(min(len(bond_attn), len(labels_np))):
            (broken_weights if labels_np[i] == 1 else intact_weights).append(bond_attn[i])

    effect = compute_effect_size(broken_weights, intact_weights, mode=attention_mode)
    mode = effect["mode"]

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot([broken_weights, intact_weights],
                    patch_artist=True, widths=0.5, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='white', markersize=8))
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Broken Bonds', 'Intact Bonds'])
    bp['boxes'][0].set_facecolor(INTERP_COLORS['broken']); bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(INTERP_COLORS['intact']); bp['boxes'][1].set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('black'); median.set_linewidth(2)

    rng = np.random.default_rng(42)
    ax.scatter(rng.normal(1, 0.06, len(broken_weights)), broken_weights,
               alpha=0.4, color=INTERP_COLORS['broken'], s=20, zorder=2)
    ax.scatter(rng.normal(2, 0.06, len(intact_weights)), intact_weights,
               alpha=0.4, color=INTERP_COLORS['intact'], s=20, zorder=2)

    stats_text = (
        f"Cohen's d (signed): {effect['cohen_d_signed']:+.3f}\n"
        f"  ↑ positive ⇒ higher saliency associated with broken bonds\n"
        f"Cohen's d (|·|): {effect['cohen_d_abs']:.3f}\n"
        f"AUC: {effect['auc']:.3f}  (>0.5 ⇒ broken > intact)\n"
        f"t = {effect['t_stat']:.2f}, p = {effect['p_value']:.2e}\n"
        f"n_broken = {effect['n_broken']}, n_intact = {effect['n_intact']}\n"
        f"{effect['interpretation']}"
    )
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

    ax.set_ylabel('Functional Saliency (Attention)', fontsize=12, fontweight='bold')
    ax.set_title('Broken vs Intact Bonds: Attention-based Functional Separation\n'
                 f'{_mode_caption(mode)}',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig, effect


def plot_new_interpretability_case_study(
    attention_weights_list: List,
    bond_labels_list: List[torch.Tensor],
    sequences: List[str],
    edge_indices: List[Optional[torch.Tensor]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    max_seq_len: int = 25,
    attention_mode: str = "auto",
    heatmap_normalize: str = "row",
) -> Tuple[plt.Figure, Dict[str, object]]:
    """Paper-grade 2x2 interpretability figure.

    (a) Single-sample attention (final layer) with |r| and signed r.
    (b) Layer-wise evolution: primary |r| (strength) + secondary signed r.
    (c) Broken vs intact boxplot with UNIFIED effect size + mode label.
    (d) Attention heatmap with explicit normalization + semantic caption.

    Shared palette (INTERP_COLORS), shared bond indexing
    (extract_bond_level_attention), and shared metrics
    (compute_separation_metrics / compute_effect_size) guarantee consistency.

    Returns (fig, summary_dict) for JSON serialization.
    """
    # Create 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    plt.subplots_adjust(hspace=0.40, wspace=0.30, left=0.08, right=0.95, top=0.90, bottom=0.08)

    # ---------- (a) Single-sample final-layer attention ----------
    ax1 = axes[0, 0]
    sample_attn = attention_weights_list[0]
    sample_labels = bond_labels_list[0]
    sample_seq = sequences[0]
    sample_edge_idx = edge_indices[0]

    num_layers = len(sample_attn)
    seq_len = min(len(sample_seq), max_seq_len)
    num_bonds = seq_len - 1

    last_layer = sample_attn[-1]
    last_attn, bond_labels_str = extract_bond_level_attention(
        last_layer, sample_edge_idx, sample_seq, max_seq_len
    )
    bond_labels_np = (sample_labels[:num_bonds].cpu().numpy()
                      if isinstance(sample_labels, torch.Tensor)
                      else np.asarray(sample_labels[:num_bonds]))
    last_attn = last_attn[:num_bonds]

    x = np.arange(num_bonds)
    for i in range(num_bonds):
        if i < len(bond_labels_np) and bond_labels_np[i] == 1:
            ax1.axvspan(i - 0.4, i + 0.4, alpha=0.15, color=INTERP_COLORS['broken'])
    bar_colors = [INTERP_COLORS['broken'] if (i < len(bond_labels_np) and bond_labels_np[i] == 1)
                  else INTERP_COLORS['intact'] for i in range(num_bonds)]
    ax1.bar(x, last_attn, color=bar_colors, alpha=0.7, edgecolor='white')

    for idx in np.argsort(last_attn)[-min(3, num_bonds):]:
        ax1.annotate(bond_labels_str[idx], xy=(idx, last_attn[idx]),
                     xytext=(0, 8), textcoords='offset points',
                     ha='center', fontsize=7, fontweight='bold',
                     color=INTERP_COLORS['highlight'])

    m_a = compute_separation_metrics(last_attn, bond_labels_np)
    ax1.set_xlabel('Bond Position (Residue Pair)', fontsize=10)
    ax1.set_ylabel('Functional Saliency (Attention)', fontsize=10)
    ax1.set_title('(a) Cleavage-Relevance Attention Distribution\n'
                  f"alignment |r|={m_a['abs_r']:.2f}, signed r={m_a['pearson_r']:+.2f}",
                  fontsize=11, fontweight='bold')
    if num_bonds <= 15:
        ax1.set_xticks(x)
        ax1.set_xticklabels(bond_labels_str, fontsize=7, rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    # ---------- (b) Layer-wise evolution ----------
    ax2 = axes[0, 1]
    layer_metrics = compute_layer_separation_metrics(
        attention_weights_list, bond_labels_list, edge_indices, sequences, max_seq_len
    )

    if layer_metrics:
        layer_x = np.arange(len(layer_metrics))
        abs_r = [m["abs_r"] for m in layer_metrics]
        raw_r = [m["pearson_r"] for m in layer_metrics]

        # Secondary signed r (gray dashed)
        ax2.plot(layer_x, raw_r, 'o--', color=INTERP_COLORS['raw_r'], linewidth=1.5,
                 markersize=6, alpha=0.7, label='Signed r (alignment strength)', zorder=2)
        # Primary |r| (solid)
        ax2.plot(layer_x, abs_r, 'o-', color=INTERP_COLORS['line'], linewidth=2.5,
                 markersize=11, markerfacecolor=INTERP_COLORS['broken'],
                 markeredgecolor='black', markeredgewidth=1.5,
                 label='|r| (alignment strength)', zorder=4)
        for xi, yi in zip(layer_x, abs_r):
            ax2.annotate(f'{yi:.3f}', xy=(xi, yi), xytext=(0, 12),
                         textcoords='offset points', ha='center', fontsize=10,
                         fontweight='bold', color=INTERP_COLORS['line'])
        ax2.fill_between(layer_x, abs_r, alpha=0.15, color=INTERP_COLORS['fill'])
        ax2.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax2.set_xticks(layer_x)
        ax2.set_xticklabels([f'L{i}' for i in range(len(layer_metrics))], fontsize=10)
        ax2.legend(loc='lower right', fontsize=9, framealpha=0.9)

    ax2.set_xlabel('Network Layer', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Alignment strength (≥ 0)', fontsize=10, fontweight='bold')
    ax2.set_title('(b) Layer-wise Attention–Label Alignment',
                  fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # ---------- (c) Broken vs intact (unified effect size) ----------
    ax3 = axes[1, 0]
    broken_weights, intact_weights = [], []
    for attn_weights, labels, edge_idx, seq in zip(
        attention_weights_list, bond_labels_list, edge_indices, sequences
    ):
        ll = attn_weights[-1] if isinstance(attn_weights, list) else attn_weights
        bond_attn, _ = extract_bond_level_attention(ll, edge_idx, seq, max_seq_len)
        labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
        for i in range(min(len(bond_attn), len(labels_np))):
            (broken_weights if labels_np[i] == 1 else intact_weights).append(bond_attn[i])

    effect = compute_effect_size(broken_weights, intact_weights, mode=attention_mode)
    mode = effect["mode"]

    bp = ax3.boxplot([broken_weights, intact_weights], patch_artist=True, widths=0.5,
                     showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='white', markersize=8))
    ax3.set_xticks([1, 2])
    ax3.set_xticklabels(['Broken Bonds', 'Intact Bonds'])
    bp['boxes'][0].set_facecolor(INTERP_COLORS['broken']); bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(INTERP_COLORS['intact']); bp['boxes'][1].set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('black'); median.set_linewidth(2)

    rng = np.random.default_rng(42)
    ax3.scatter(rng.normal(1, 0.06, len(broken_weights)), broken_weights,
                alpha=0.4, color=INTERP_COLORS['broken'], s=20, zorder=2)
    ax3.scatter(rng.normal(2, 0.06, len(intact_weights)), intact_weights,
                alpha=0.4, color=INTERP_COLORS['intact'], s=20, zorder=2)

    # 精简 stats_text：保留关键数值，去掉冗长的自然语言 interpretation
    # （interpretation 由用户在 figure caption 中自行补充）
    stats_text = (
        f"d = {effect['cohen_d_signed']:+.2f}  "
        f"(n_broken={effect['n_broken']}, n_intact={effect['n_intact']})\n"
        f"AUC = {effect['auc']:.3f}\n"
        f"p = {effect['p_value']:.2e}"
    )
    ax3.text(0.98, 0.98, stats_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
    ax3.set_ylabel('Functional Saliency (Attention)', fontsize=10, fontweight='bold')
    ax3.set_title('(c) Broken vs Intact Bonds: Attention-based Functional Separation',
                  fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # ---------- (d) Aggregate layer attention (replaces old heatmap) ----------
    # 旧版：mean + row-normalized heatmap（5 个样本）
    # 新版：median + 量化聚焦度指标（N 个案例样本）
    # 色阶映射 heatmap_normalize 参数：'row'（默认，视觉与指标一致）
    #                          或 'absolute'（保留层间大小，但低量级层不可见）
    ax4 = axes[1, 1]
    # 调整 panel (d) 子图位置，给右侧 metrics 文本列留空间
    pos = ax4.get_position()
    ax4.set_position([pos.x0, pos.y0, pos.width * 0.72, pos.height])

    # heatmap_normalize 历史值: 'row' 或 'global'。映射到新的 color_scale：
    #   'row' / 'global' → 'row'（视觉一致，推荐）
    #   'absolute'       → 'absolute'（保留大小，但视觉可能与指标矛盾）
    panel_d_color_scale = 'absolute' if heatmap_normalize == 'absolute' else 'row'

    _, panel_d_summary = plot_aggregate_layer_attention_compact(
        attention_weights_list, edge_indices, sequences,
        ax=ax4,
        max_seq_len=max_seq_len,
        max_bonds_show=min(15, max_seq_len - 1),
        focus_thresholds=(0.85, 0.95),
        show_right_metrics=True,
        color_scale=panel_d_color_scale,
    )
    panel_d_focus_metrics = panel_d_summary['layer_focus_metrics']
    # 在 panel (d) 标题前加 "(d)" 标识
    ax4.set_title('(d) ' + ax4.get_title(), fontsize=10, fontweight='bold', pad=8)

    # 精简 suptitle：只保留主标题，详细 interpretation 留给 figure caption
    fig.suptitle('DBond-GT Interpretability Case Study',
                 fontsize=13, fontweight='bold', y=0.97)

    _save_figure(fig, save_path)

    summary = {
        "attention_mode": mode,
        "mode_caption": _mode_caption(mode),
        "panel_a_metrics": m_a,
        "panel_b_layer_metrics": layer_metrics,
        "panel_c_effect_size": effect,
        "panel_d_layer_focus_metrics": panel_d_focus_metrics,
        "panel_d_interpretation": panel_d_summary['interpretation'],
        "panel_d_n_samples": panel_d_summary['n_samples'],
        "panel_d_color_scale": "absolute",
    }
    return fig, summary


# =============================================================================
# Residue-pair chemistry matrix (Paper-grade baseline)
# =============================================================================

def plot_residue_pair_matrix(
    empirical: np.ndarray,
    predicted: np.ndarray,
    counts: np.ndarray,
    aa_labels: List[str],
    save_path: Optional[str] = None,
    min_n_for_label: int = 50,
    rare_thresholds: Tuple[int, int] = (10, 50),
    figsize: Tuple[float, float] = (30, 10.5),
    filter_empty: bool = False,
    min_total_n: int = 1,
) -> plt.Figure:
    """3-panel residue-pair chemistry matrix.

    Panel (a): empirical bond cleavage rate per residue pair (X-Y).
    Panel (b): model's mean predicted cleavage probability per residue pair.
    Panel (c): difference (predicted - empirical), diverging colormap.

    Rows = N-terminal residue X, columns = C-terminal residue Y.
    Cells with sample count below `rare_thresholds[1]` are hatched and marked
    with asterisks to indicate statistical uncertainty (rare AAs B/O/X/Z).

    Args:
        empirical:    [N_aa, N_aa] empirical cleavage rate, values in [0, 1].
        predicted:    [N_aa, N_aa] model mean predicted probability, [0, 1].
        counts:       [N_aa, N_aa] integer sample count per cell.
        aa_labels:    list of N_aa single-char amino acid labels.
        save_path:    output path; format inferred from extension (svg/png).
        min_n_for_label: cells with N < this get asterisk annotation.
        rare_thresholds: (n_star_star, n_star) thresholds for double/triple marker.
        figsize: figure size; default (30, 10.5) gives ~0.4" per cell with 3
                 square subplots side-by-side, enough room for 2-decimal text.
        filter_empty: if True, drop AAs that have zero total observations
                      (rows+cols all empty) from the visualization. Useful when
                      the dataset by-design excludes certain AAs (e.g., C/M/W
                      in D-amino-acid mirror peptides).
        min_total_n:  minimum total count (sum over row + column) for an AA
                      to be retained when filter_empty=True.
    """
    n_aa = len(aa_labels)
    assert empirical.shape == (n_aa, n_aa)
    assert predicted.shape == (n_aa, n_aa)
    assert counts.shape == (n_aa, n_aa)

    # 可选：过滤掉数据中完全不存在的 AA（按行+列总样本数）
    if filter_empty:
        # 行+列总样本数 = 该 AA 作为 N 端或 C 端出现的总次数
        row_sums = counts.sum(axis=1)
        col_sums = counts.sum(axis=0)
        total_per_aa = row_sums + col_sums
        keep_mask = total_per_aa >= min_total_n
        kept_idx = np.where(keep_mask)[0]
        dropped = [aa_labels[i] for i in np.where(~keep_mask)[0]]
        if len(dropped) > 0:
            empirical = empirical[keep_mask][:, keep_mask]
            predicted = predicted[keep_mask][:, keep_mask]
            counts = counts[keep_mask][:, keep_mask]
            aa_labels = [aa_labels[i] for i in kept_idx]
            n_aa = len(aa_labels)
            # 缩小 figsize 以匹配
            scale = n_aa / max(len(aa_labels), 1)  # 实际无变化，仅为占位
            # 自适应：每个 cell 保持 ~0.42" 宽
            per_cell = 0.42
            subplot_w = max(7.0, n_aa * per_cell)
            figsize = (subplot_w * 3 + 2.0, max(7.5, n_aa * per_cell + 2.5))

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    # 拓宽左右留白以放下 colorbar；wspace 控制子图间距；
    # 底部留 0.10 给坐标轴说明；顶部留 0.17 给总标题。
    plt.subplots_adjust(wspace=0.40, left=0.05, right=0.97, top=0.83, bottom=0.10)

    # Shared colour scale for (a) and (b) so they are directly comparable
    rate_vmin, rate_vmax = 0.0, 1.0
    diff_vmin, diff_vmax = -0.5, 0.5

    def _draw_panel(ax, matrix, vmin, vmax, cmap, title, annotate_values: bool,
                    is_diff: bool = False):
        im = ax.imshow(matrix, cmap=cmap, aspect='equal',
                       vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_xticks(np.arange(n_aa))
        ax.set_yticks(np.arange(n_aa))
        ax.set_xticklabels(aa_labels, fontsize=10)
        ax.set_yticklabels(aa_labels, fontsize=10)
        ax.set_xlabel('C-terminal residue  Y', fontsize=12, fontweight='bold',
                      labelpad=6)
        ax.set_ylabel('N-terminal residue  X', fontsize=12, fontweight='bold',
                      labelpad=6)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        # 让每个 cell 的边界清晰（小网格线）
        ax.set_xticks(np.arange(-0.5, n_aa, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_aa, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.6)
        ax.tick_params(which='minor', bottom=False, left=False)

        # Annotate values with rare-AA markers
        if annotate_values:
            for i in range(n_aa):
                for j in range(n_aa):
                    n = int(counts[i, j])
                    if n == 0:
                        # empty cell: hatch background
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                                   fill=True, facecolor='#EEEEEE',
                                                   edgecolor='#BBBBBB',
                                                   linewidth=0.3, zorder=1))
                        continue
                    val = matrix[i, j]
                    # 动态字体大小 + 颜色：稀有 AA 用更小字体 + 灰色
                    if n < rare_thresholds[0]:
                        marker, color, fs = '**', '#34495E', 7.0
                    elif n < rare_thresholds[1]:
                        marker, color, fs = '*', '#34495E', 7.5
                    else:
                        marker, color, fs = '', 'black', 8.5
                    # diff 面板：带符号的百分点（percentage points, pp），
                    # 与 (a)(b) 保持同样 2-3 字符宽度，避免数字堆叠。
                    # 例: +0.12 → "+12",  -0.34 → "−34"
                    if is_diff:
                        sign = '+' if val >= 0 else '−'
                        text = f'{sign}{abs(val*100):.0f}{marker}'
                    else:
                        text = f'{val*100:.0f}{marker}'
                    ax.text(j, i, text, ha='center', va='center',
                            fontsize=fs, color=color)
        return im

    im_a = _draw_panel(axes[0], empirical, rate_vmin, rate_vmax, 'YlOrRd',
                       '(a) Empirical cleavage rate  P(broken | X-Y)  [%]',
                       annotate_values=True)
    im_b = _draw_panel(axes[1], predicted, rate_vmin, rate_vmax, 'YlOrRd',
                       '(b) Model predicted  E[σ(model) | X-Y]  [%]',
                       annotate_values=True)
    im_c = _draw_panel(axes[2], predicted - empirical, diff_vmin, diff_vmax, 'RdBu_r',
                       '(c) Difference  (predicted − empirical)  [pp]',
                       annotate_values=True, is_diff=True)

    # Colorbars: 放在右侧，pad 略大避免与子图挤在一起
    cb_a = plt.colorbar(im_a, ax=axes[0], fraction=0.038, pad=0.04)
    cb_a.set_label('Cleavage rate', fontsize=10)
    cb_b = plt.colorbar(im_b, ax=axes[1], fraction=0.038, pad=0.04)
    cb_b.set_label('Predicted probability', fontsize=10)
    cb_c = plt.colorbar(im_c, ax=axes[2], fraction=0.038, pad=0.04)
    # 用百分点 (pp) 显示 colorbar 刻度，与 cell 注释一致
    cb_c.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f'{v*100:+.0f}')
    )
    cb_c.set_label('Bias (model − empirical)  [pp]', fontsize=10)

    fig.suptitle('Residue-Pair Cleavage Chemistry: Empirical vs Model\n'
                 'Rows X = N-terminal side,  Cols Y = C-terminal side   '
                 '|   Panels (a)(b) shown as percentage [%],   '
                 'Panel (c) shown as signed bias in percentage points [pp]\n'
                 f'* N∈[{rare_thresholds[0]},{rare_thresholds[1]}), '
                 f'** N<{rare_thresholds[0]} (statistically uncertain, rare AAs B/O/X/Z)',
                 fontsize=13, fontweight='bold', y=0.965)

    _save_figure(fig, save_path)
    return fig


# =============================================================================
# Occlusion causal attribution
# =============================================================================

def build_residue_attention_matrix(
    attention_weights: torch.Tensor,
    edge_index: torch.Tensor,
    seq_len: int,
) -> np.ndarray:
    """Convert GAT edge attention to a [seq_len, seq_len] residue-residue matrix.

    Each edge (src, dst) contributes its (head-averaged) attention weight to
    both matrix entries [src, dst] and [dst, src] (attention is treated as a
    symmetric association for visualisation purposes; this is a display
    convention, not a claim about the underlying directed computation).
    """
    if attention_weights.dim() == 2:
        attn_np = attention_weights.mean(dim=1).cpu().numpy()
    else:
        attn_np = attention_weights.cpu().numpy()

    matrix = np.zeros((seq_len, seq_len), dtype=float)
    edge_index_np = edge_index.cpu().numpy()
    n_edges = edge_index_np.shape[1]
    for i in range(min(n_edges, len(attn_np))):
        src, dst = int(edge_index_np[0, i]), int(edge_index_np[1, i])
        if 0 <= src < seq_len and 0 <= dst < seq_len and src != dst:
            w = float(attn_np[i])
            matrix[src, dst] += w
            matrix[dst, src] += w
    return matrix


def collapse_to_residue_bond_attention(
    residue_attn: np.ndarray,
    seq_len: int,
) -> np.ndarray:
    """Collapse [L, L] residue-residue attention to [L, L-1] residue-bond matrix.

    Entry [j, i] = mean of residue j's attention toward residue i and residue
    i+1 (the two residues that form bond i). This produces a matrix directly
    comparable to the occlusion sensitivity matrix M[j, i].
    """
    bond_attn = np.zeros((seq_len, max(seq_len - 1, 0)), dtype=float)
    for i in range(seq_len - 1):
        if i < residue_attn.shape[1] and (i + 1) < residue_attn.shape[1]:
            bond_attn[:, i] = (residue_attn[:, i] + residue_attn[:, i + 1]) / 2.0
    return bond_attn


def plot_occlusion_vs_attention(
    occlusion_matrix: np.ndarray,
    attention_matrix: np.ndarray,
    sequence: str,
    sample_id: str = '',
    layer_idx: int = -1,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Per-sample 2-panel heatmap: occlusion sensitivity vs attention saliency.

    Both matrices are [seq_len, seq_len-1]:
        rows   = residue position j (the mutated / attending residue)
        cols   = bond position i    (the predicted / attended-to bond)

    Pearson r between flattened matrices is reported in the suptitle as a
    consistency metric (non-causal wording: "consistent with" not "caused by").

    Figure width auto-scales with sequence length to prevent label overlap.
    """
    seq_len = len(sequence)
    num_bonds = seq_len - 1

    # 自适应宽度：每个键至少 0.45 inch
    if figsize is None:
        min_width_per_bond = 0.45
        min_total_width = 12.0
        computed = max(min_total_width, num_bonds * min_width_per_bond * 2 + 3)
        figsize = (min(computed, 28.0), max(6.5, seq_len * 0.32 + 3.0))

    occ = occlusion_matrix[:seq_len, :num_bonds]
    att = attention_matrix[:seq_len, :num_bonds]

    # Consistency (Pearson r over flattened upper-relevant entries)
    if occ.size > 1 and np.std(occ) > 0 and np.std(att) > 0:
        r = float(np.corrcoef(occ.flatten(), att.flatten())[0, 1])
    else:
        r = float('nan')

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    # 底部留更多空间给旋转 45° 的键标签
    plt.subplots_adjust(wspace=0.3, left=0.07, right=0.96, top=0.83, bottom=0.28)

    # 紧凑的键标签：仅在键位置变化时显示残基对，否则只显示索引；
    # 旋转 45° 防止重叠；底部留 0.28 给旋转后的标签空间。
    bond_labels = [f'{sequence[i]}-{sequence[i+1]}' for i in range(num_bonds)]
    residue_labels = [f'{i}:{sequence[i]}' for i in range(seq_len)]

    im0 = axes[0].imshow(occ, cmap='YlOrRd', aspect='auto',
                         interpolation='nearest')
    axes[0].set_xticks(np.arange(num_bonds))
    axes[0].set_xticklabels(bond_labels, fontsize=8, rotation=45,
                            ha='right', rotation_mode='anchor')
    axes[0].set_yticks(np.arange(seq_len))
    axes[0].set_yticklabels(residue_labels, fontsize=8)
    axes[0].set_xlabel('Bond (predicted)  [X-Y]', fontsize=10, fontweight='bold',
                       labelpad=8)
    axes[0].set_ylabel('Mutated residue position j', fontsize=10, fontweight='bold')
    axes[0].set_title('(a) Occlusion sensitivity  mean |Δp[j→aa] on bond i|',
                      fontsize=10, fontweight='bold')
    cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.set_label('Sensitivity', fontsize=9)

    im1 = axes[1].imshow(att, cmap='YlOrRd', aspect='auto',
                         interpolation='nearest')
    axes[1].set_xticks(np.arange(num_bonds))
    axes[1].set_xticklabels(bond_labels, fontsize=8, rotation=45,
                            ha='right', rotation_mode='anchor')
    axes[1].set_yticks(np.arange(seq_len))
    axes[1].set_yticklabels(residue_labels, fontsize=8)
    axes[1].set_xlabel('Bond (attended-to)  [X-Y]', fontsize=10, fontweight='bold',
                       labelpad=8)
    axes[1].set_ylabel('Attending residue position j', fontsize=10, fontweight='bold')
    axes[1].set_title('(b) Functional-saliency attention (residue → bond)',
                      fontsize=10, fontweight='bold')
    cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.set_label('Attention (functional saliency)', fontsize=9)

    r_str = f'{r:+.3f}' if not np.isnan(r) else 'N/A'
    consistency_verdict = (
        'strongly consistent' if (not np.isnan(r) and r >= 0.5)
        else ('moderately consistent' if (not np.isnan(r) and r >= 0.3)
              else ('weakly consistent' if not np.isnan(r) else 'undefined'))
    )
    fig.suptitle(
        f'Occlusion vs Functional-Saliency Attention   |   sample = {sample_id}   '
        f'|   layer = {layer_idx}\n'
        f'Pearson r (attention vs occlusion) = {r_str}   →   {consistency_verdict}',
        fontsize=11, fontweight='bold', y=0.96,
    )

    _save_figure(fig, save_path)
    return fig, {'pearson_r': r, 'consistency': consistency_verdict}


def plot_occlusion_attention_consistency(
    per_sample_r: List[float],
    sample_ids: List[str],
    all_attention_flat: np.ndarray,
    all_occlusion_flat: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 6),
) -> plt.Figure:
    """Aggregate consistency figure: scatter + per-sample r distribution.

    Left  : 2D hexbin/scatter of (attention, occlusion) over all (j,i) entries
            across all analysed samples; reports global Pearson r and Spearman ρ.
    Right : box plot of per-sample Pearson r values; each sample is one point.
    """
    from scipy.stats import spearmanr

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plt.subplots_adjust(wspace=0.3, left=0.07, right=0.97, top=0.86, bottom=0.15)

    valid_r = [r for r in per_sample_r if not np.isnan(r)]
    global_r = (float(np.corrcoef(all_attention_flat, all_occlusion_flat)[0, 1])
                if len(all_attention_flat) > 1 else float('nan'))
    try:
        global_rho, _ = spearmanr(all_attention_flat, all_occlusion_flat)
        global_rho = float(global_rho)
    except Exception:
        global_rho = float('nan')

    # Left: scatter (hexbin if very dense)
    ax = axes[0]
    if len(all_attention_flat) > 5000:
        hb = ax.hexbin(all_attention_flat, all_occlusion_flat, gridsize=40,
                       cmap='Blues', mincnt=1)
        plt.colorbar(hb, ax=ax, label='count')
    else:
        ax.scatter(all_attention_flat, all_occlusion_flat, s=8,
                   alpha=0.35, color=INTERP_COLORS['broken'], edgecolors='none')
    ax.set_xlabel('Functional-saliency attention (residue → bond)',
                  fontsize=10, fontweight='bold')
    ax.set_ylabel('Occlusion sensitivity (residue → bond)',
                  fontsize=10, fontweight='bold')
    r_str = f'{global_r:+.3f}' if not np.isnan(global_r) else 'N/A'
    rho_str = f'{global_rho:+.3f}' if not np.isnan(global_rho) else 'N/A'
    ax.set_title(
        f'(a) Global consistency across all samples\n'
        f'Pearson r = {r_str}   |   Spearman ρ = {rho_str}',
        fontsize=10, fontweight='bold',
    )
    ax.grid(True, alpha=0.3)

    # Right: per-sample r distribution
    ax2 = axes[1]
    if valid_r:
        bp = ax2.boxplot(valid_r, vert=True, patch_artist=True, widths=0.4,
                         medianprops=dict(color='black', linewidth=1.5))
        for patch in bp['boxes']:
            patch.set_facecolor(INTERP_COLORS['fill'])
            patch.set_alpha(0.5)
        # overlay individual sample points
        jitter = np.random.RandomState(42).uniform(-0.08, 0.08, size=len(valid_r))
        ax2.scatter(np.ones(len(valid_r)) + jitter, valid_r,
                    s=40, color=INTERP_COLORS['broken'],
                    edgecolor='black', linewidth=0.5, zorder=3)
        for k, (r_val, sid) in enumerate(zip(per_sample_r, sample_ids)):
            if not np.isnan(r_val):
                # annotate top-3 and bottom-3 only to avoid clutter
                pass
        mean_r = float(np.mean(valid_r))
        median_r = float(np.median(valid_r))
        ax2.axhline(mean_r, color=INTERP_COLORS['line'], linestyle='--',
                    linewidth=1.2, alpha=0.7,
                    label=f'mean = {mean_r:+.3f}')
        ax2.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax2.axhline(0.5, color=INTERP_COLORS['highlight'], linestyle=':',
                    linewidth=1.0, alpha=0.6, label='r = 0.5 (strong)')
        ax2.set_xticklabels([f'per-sample\n(n={len(valid_r)})'])
        ax2.set_ylabel('Pearson r (attention vs occlusion)',
                       fontsize=10, fontweight='bold')
        ax2.set_title(
            f'(b) Per-sample consistency distribution\n'
            f'median = {median_r:+.3f}   |   '
            f'{sum(1 for r in valid_r if r >= 0.5)}/{len(valid_r)} '
            f'samples r ≥ 0.5',
            fontsize=10, fontweight='bold',
        )
        ax2.legend(loc='lower right', fontsize=9, framealpha=0.9)
    else:
        ax2.text(0.5, 0.5, 'No valid r values', ha='center', va='center',
                 transform=ax2.transAxes)
    ax2.grid(True, alpha=0.3, axis='y')

    n_total = len(per_sample_r)
    n_strong = sum(1 for r in valid_r if r >= 0.5)
    verdict = (
        f'{n_strong}/{len(valid_r)} samples show strong consistency (r ≥ 0.5)'
        if valid_r else 'Consistency undefined'
    )
    fig.suptitle(
        f'Occlusion vs Attention Consistency  ({n_total} samples)\n'
        f'{verdict}',
        fontsize=12, fontweight='bold', y=0.98,
    )

    _save_figure(fig, save_path)
    return fig, {
        'global_pearson_r': global_r,
        'global_spearman_rho': global_rho,
        'mean_per_sample_r': float(np.mean(valid_r)) if valid_r else float('nan'),
        'median_per_sample_r': float(np.median(valid_r)) if valid_r else float('nan'),
        'n_strong_consistency': n_strong,
        'n_valid': len(valid_r),
    }


# =============================================================================
# Aggregate per-layer attention (trustworthy mean across many samples)
# =============================================================================

def _compute_layer_focus_stats(
    layer_bond_values: List[List[List[float]]],
    num_layers: int,
    max_bonds: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """从「每层每键的 attention 值列表」计算聚合统计量。

    Args:
        layer_bond_values: shape [num_layers][max_bonds][n_samples] 的嵌套 list
        num_layers:        层数
        max_bonds:         键数

    Returns:
        layer_median:  [num_layers, max_bonds]  跨样本中位数
        layer_q25:     [num_layers, max_bonds]  25 分位
        layer_q75:     [num_layers, max_bonds]  75 分位
        layer_mean:    [num_layers, max_bonds]  跨样本均值
        layer_std:     [num_layers, max_bonds]  跨样本标准差
        focus_metrics: list[dict]，每层一个，含 normalized_entropy / top1_share /
                       top3_share / mean_attention / std_across_bonds
    """
    layer_median = np.zeros((num_layers, max_bonds))
    layer_q25 = np.zeros((num_layers, max_bonds))
    layer_q75 = np.zeros((num_layers, max_bonds))
    layer_mean = np.zeros((num_layers, max_bonds))
    layer_std = np.zeros((num_layers, max_bonds))

    for layer_idx in range(num_layers):
        for bond_pos in range(max_bonds):
            vals = layer_bond_values[layer_idx][bond_pos]
            if len(vals) >= 2:
                layer_median[layer_idx, bond_pos] = float(np.median(vals))
                layer_q25[layer_idx, bond_pos] = float(np.percentile(vals, 25))
                layer_q75[layer_idx, bond_pos] = float(np.percentile(vals, 75))
                layer_mean[layer_idx, bond_pos] = float(np.mean(vals))
                layer_std[layer_idx, bond_pos] = float(np.std(vals))
            elif len(vals) == 1:
                layer_median[layer_idx, bond_pos] = vals[0]
                layer_q25[layer_idx, bond_pos] = vals[0]
                layer_q75[layer_idx, bond_pos] = vals[0]
                layer_mean[layer_idx, bond_pos] = vals[0]
                layer_std[layer_idx, bond_pos] = 0.0

    focus_metrics: List[Dict[str, float]] = []
    for layer_idx in range(num_layers):
        row = layer_median[layer_idx]
        row_nonneg = np.maximum(row, 0)
        s = row_nonneg.sum()
        if s > 0:
            normalized = row_nonneg / s
            entropy = -float(np.sum(normalized * np.log(normalized + 1e-12)))
            max_entropy = float(np.log(max_bonds)) if max_bonds > 1 else 0.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            top1_share = float(np.max(normalized))
            top3_share = float(np.sum(np.sort(normalized)[-min(3, max_bonds):]))
        else:
            normalized_entropy = 1.0
            top1_share = 0.0
            top3_share = 0.0
        focus_metrics.append({
            'layer': layer_idx,
            'normalized_entropy': normalized_entropy,
            'top1_share': top1_share,
            'top3_share': top3_share,
            'mean_attention': float(row_nonneg.mean()),
            'std_across_bonds': float(row_nonneg.std()),
        })

    return layer_median, layer_q25, layer_q75, layer_mean, layer_std, focus_metrics


def _collect_layer_bond_values(
    attention_weights_list: List[List[torch.Tensor]],
    edge_indices: List[Optional[torch.Tensor]],
    sequences: List[str],
    num_layers: int,
    max_bonds: int,
    max_seq_len: int,
) -> List[List[List[float]]]:
    """收集每层每键的 attention 值（跨样本）。"""
    layer_bond_values: List[List[List[float]]] = [
        [[] for _ in range(max_bonds)] for _ in range(num_layers)
    ]
    for attn_weights, edge_idx, seq in zip(
        attention_weights_list, edge_indices, sequences
    ):
        seq_len = min(len(seq), max_seq_len)
        for layer_idx in range(num_layers):
            bond_attn, _ = extract_bond_level_attention(
                attn_weights[layer_idx], edge_idx, seq, max_seq_len
            )
            for bond_pos in range(min(max_bonds, len(bond_attn))):
                layer_bond_values[layer_idx][bond_pos].append(float(bond_attn[bond_pos]))
    return layer_bond_values


def plot_aggregate_layer_attention_compact(
    attention_weights_list: List[List[torch.Tensor]],
    edge_indices: List[Optional[torch.Tensor]],
    sequences: List[str],
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
    max_seq_len: int = 30,
    max_bonds_show: int = 20,
    focus_thresholds: Tuple[float, float] = (0.85, 0.95),
    figsize: Tuple[float, float] = (10, 4),
    show_right_metrics: bool = True,
    color_scale: str = "row",
) -> Tuple[Optional[plt.Figure], Dict[str, object]]:
    """紧凑版聚合图层注意力图（单 subplot，适配嵌入 case study panel d）。

    与 `plot_aggregate_layer_attention`（多 subplot 详细版）的区别：
        - 本函数输出/绘制到单个 axes，便于嵌入 2×2 case study 的 panel (d)
        - 用 heatmap（layer × bond），可切换 absolute / row-normalized 色阶
        - 右侧文本列显示每层的 entropy + top-3 share + 聚焦度判定

    解读：
        - 浅层（L0）应熵低（< 0.85）= focused（少数键占大头）
        - 深层（LN）应熵高（≥ 0.95）= diffuse（均匀分布）
        - 若所有层熵都接近 1.0：模型未学到层间分化

    Args:
        attention_weights_list: 每个样本的 [layer_0, layer_1, ...] attention
        edge_indices:           每个样本的 edge_index
        sequences:              每个样本的序列
        ax:                     若提供，绘制到此 axes（嵌入模式）；
                                否则创建新 figure（standalone 模式）。
        save_path:              保存路径（仅 standalone 模式生效）
        max_seq_len:            最大序列长度
        max_bonds_show:         最多显示键数
        focus_thresholds:       (focused, diffuse) 的 entropy 阈值
                                entropy < thr[0] → focused
                                thr[0] ≤ entropy < thr[1] → moderate
                                entropy ≥ thr[1] → diffuse
        figsize:                standalone 模式的 figure 大小
        show_right_metrics:     是否在右侧显示 entropy + top-3 文本列
        color_scale:            "row" = 每层独立归一化 [0,1]（默认，视觉与指标一致）
                                "absolute" = 所有层共用 [0, global_max]（保留层间大小差异，
                                但低量级层的内部模式不可见）

    Returns:
        (fig, summary) — fig 在嵌入模式下为 None
    """
    n_samples = len(attention_weights_list)
    if n_samples == 0:
        raise ValueError("attention_weights_list is empty")
    num_layers = len(attention_weights_list[0])
    max_bonds = min(max_bonds_show, max_seq_len - 1)

    layer_bond_values = _collect_layer_bond_values(
        attention_weights_list, edge_indices, sequences,
        num_layers, max_bonds, max_seq_len,
    )
    layer_median, _, _, _, _, focus_metrics = _compute_layer_focus_stats(
        layer_bond_values, num_layers, max_bonds,
    )

    # 嵌入 vs standalone
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        plt.subplots_adjust(left=0.10, right=0.75, top=0.88, bottom=0.18)
    else:
        fig = None

    # 色阶选择
    if color_scale == "absolute":
        # 绝对值色阶：所有层共用 [0, global_max]
        # 注意：低量级层（如 L0 attention 很小）会全部映射到浅色，
        # 导致即使该层 entropy 低（focused）也看不出热点。视觉与指标会矛盾。
        global_max = layer_max_safe(layer_median)
        if global_max <= 0:
            global_max = 1.0
        display_matrix = layer_median
        vmin, vmax = 0.0, float(global_max)
        cbar_label = 'Median attention (absolute)'
    elif color_scale == "row":
        # 行归一化：每层独立缩放到 [0, 1]
        # 视觉与指标一致：focused 层会看到少数深色热点，diffuse 层会看到均匀中色。
        # 代价：丢失层间绝对大小比较。
        display_matrix = np.zeros_like(layer_median)
        for i in range(num_layers):
            row = layer_median[i]
            r_max = float(row.max()) if row.max() > 0 else 1.0
            display_matrix[i] = row / r_max
        vmin, vmax = 0.0, 1.0
        cbar_label = 'Median attention (per-layer normalized)'
    else:
        raise ValueError(f"Unknown color_scale: {color_scale!r}. Use 'row' or 'absolute'.")

    im = ax.imshow(display_matrix, cmap='YlOrRd', aspect='auto',
                   vmin=vmin, vmax=vmax, interpolation='nearest',
                   origin='lower')  # L0 在底部，LN 在顶部

    # 主图：层 × 键热力图
    ax.set_xticks(np.arange(max_bonds))
    ax.set_xticklabels([str(i) for i in range(max_bonds)], fontsize=8)
    ax.set_yticks(np.arange(num_layers))
    ax.set_yticklabels([f'L{i}' for i in range(num_layers)], fontsize=10,
                       fontweight='bold')
    ax.set_xlabel('Bond position (i)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Network layer', fontsize=10, fontweight='bold')

    # 标题：单行精简版（层进度信息已在右侧 H= 文本列显示，无需重复）
    scale_note = 'absolute' if color_scale == 'absolute' else 'row-normalized'
    ax.set_title(
        f'Aggregate median attention (n={n_samples}, {scale_note})',
        fontsize=10, fontweight='bold', pad=8,
    )

    # 每行右侧加 entropy + top-3 share 文本
    focused_thr, diffuse_thr = focus_thresholds
    if show_right_metrics:
        for i, m in enumerate(focus_metrics):
            ent = m['normalized_entropy']
            top3 = m['top3_share']
            if ent < focused_thr:
                verdict, verdict_color = 'focused', '#C0392B'
            elif ent < diffuse_thr:
                verdict, verdict_color = 'moderate', '#F39C12'
            else:
                verdict, verdict_color = 'diffuse', '#27AE60'
            text = f'H={ent:.2f}  T3={top3:.2f}\n{verdict}'
            # 用 axes 坐标在右侧定位
            ax.text(1.02, i / max(num_layers - 1, 1), text,
                    transform=ax.transAxes, fontsize=8, va='center', ha='left',
                    color=verdict_color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=verdict_color, linewidth=0.8, alpha=0.9))

    # Colorbar（仅在 standalone 模式，避免嵌入时挤占空间）
    if fig is not None:
        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label(cbar_label, fontsize=9)

    summary = {
        'n_samples': n_samples,
        'num_layers': num_layers,
        'max_bonds_shown': max_bonds,
        'layer_focus_metrics': focus_metrics,
        'color_scale': color_scale,
        'global_max_absolute': layer_max_safe(layer_median),
        'interpretation': (
            'Layer-wise focus progression (median across samples): '
            + ' → '.join(
                f"L{m['layer']}(H={m['normalized_entropy']:.3f})"
                for m in focus_metrics
            )
        ),
    }

    if fig is not None:
        _save_figure(fig, save_path)
    return fig, summary


def layer_max_safe(arr: np.ndarray) -> float:
    """安全获取数组最大值（空数组返回 0）。"""
    if arr.size == 0:
        return 0.0
    return float(arr.max())


def plot_aggregate_layer_attention(
    attention_weights_list: List[List[torch.Tensor]],
    bond_labels_list: List[torch.Tensor],
    edge_indices: List[Optional[torch.Tensor]],
    sequences: List[str],
    save_path: Optional[str] = None,
    max_seq_len: int = 30,
    max_bonds_show: int = 20,
    figsize: Optional[Tuple[float, float]] = None,
    show_quantiles: bool = True,
) -> Tuple[plt.Figure, Dict[str, object]]:
    """Aggregate per-layer attention: mean ± IQR across many samples.

    回答的问题：「在群体平均水平上，每层的 attention 是更分散还是更聚焦？」

    与单样本图的关键区别：
        - 单样本图（plot_single_sample_layer_attention）：1 个样本，可能异常
        - 本图（aggregate）：N 个样本的平均，反映模型的真实群体行为

    每层一行：
        - 实线 = 跨样本的中位数 attention
        - 阴影带 = IQR (25%~75% 分位)
        - 可选虚线 = 均值

    解读：
        - L0 行呈"平缓"分布（中位数均匀，IQR 大） → 浅层分散，符合 GCN 平滑后预期
        - L1 行呈"尖峰"分布（少数键高中位数，IQR 窄） → 深层聚焦到任务相关键
        - 这是修复 GCN 跳过 bug 后的预期模式

    Args:
        attention_weights_list: 每个样本的 [layer_0_weights, layer_1_weights, ...]
        bond_labels_list:       每个样本的键标签（仅用于诊断，不参与绘图）
        edge_indices:           每个样本的边索引
        sequences:              每个样本的序列
        max_seq_len:            最大序列长度（裁剪）
        max_bonds_show:         最多显示多少个键（避免长序列拥挤）
        show_quantiles:         是否绘制 IQR 阴影带
    """
    n_samples = len(attention_weights_list)
    if n_samples == 0:
        raise ValueError("attention_weights_list is empty")

    num_layers = len(attention_weights_list[0])
    max_bonds = min(max_bonds_show, max_seq_len - 1)

    layer_bond_values = _collect_layer_bond_values(
        attention_weights_list, edge_indices, sequences,
        num_layers, max_bonds, max_seq_len,
    )
    layer_median, layer_q25, layer_q75, layer_mean, _, focus_metrics = (
        _compute_layer_focus_stats(layer_bond_values, num_layers, max_bonds)
    )

    # 绘图：每层一行
    n_rows = num_layers
    if figsize is None:
        figsize = (14, 3.2 * n_rows + 1.5)
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
    if n_rows == 1:
        axes = [axes]
    plt.subplots_adjust(hspace=0.45, left=0.08, right=0.97, top=0.90, bottom=0.10)

    bond_x = np.arange(max_bonds)
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        median = layer_median[layer_idx]
        q25 = layer_q25[layer_idx]
        q75 = layer_q75[layer_idx]
        mean = layer_mean[layer_idx]

        if show_quantiles:
            ax.fill_between(bond_x, q25, q75, alpha=0.30,
                            color=INTERP_COLORS['fill'], label='IQR (25%–75%)')
        ax.plot(bond_x, median, 'o-', color=INTERP_COLORS['line'],
                linewidth=2.2, markersize=7, label='Median', zorder=3)
        ax.plot(bond_x, mean, '--', color=INTERP_COLORS['raw_r'],
                linewidth=1.4, alpha=0.8, label='Mean', zorder=2)

        m = focus_metrics[layer_idx]
        focus_verdict = (
            'focused' if m['normalized_entropy'] < 0.85
            else ('moderate' if m['normalized_entropy'] < 0.95 else 'diffuse')
        )
        ax.set_title(
            f'Layer {layer_idx}   |   '
            f'normalized entropy = {m["normalized_entropy"]:.3f} ({focus_verdict})   |   '
            f'top-1 share = {m["top1_share"]:.3f}   |   '
            f'top-3 share = {m["top3_share"]:.3f}',
            fontsize=10, fontweight='bold', pad=8,
        )
        ax.set_ylabel('Attention\n(functional saliency)',
                      fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        if layer_idx == 0:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    axes[-1].set_xticks(bond_x)
    axes[-1].set_xticklabels([f'i={i}' for i in bond_x], fontsize=8)
    axes[-1].set_xlabel('Bond position (i)', fontsize=10, fontweight='bold')

    fig.suptitle(
        f'Aggregate Per-Layer Attention Pattern (n = {n_samples} samples)\n'
        f'Lower entropy = more focused; higher top-K share = sharper peaks.  '
        f'Expected after GCN fix: L0 diffuse → L_last focused.',
        fontsize=12, fontweight='bold', y=0.97,
    )

    _save_figure(fig, save_path)

    summary = {
        'n_samples': n_samples,
        'num_layers': num_layers,
        'max_bonds_shown': max_bonds,
        'layer_focus_metrics': focus_metrics,
        'interpretation': (
            'Layer-wise focus progression (median across samples): '
            + ' → '.join(
                f"L{m['layer']}(H={m['normalized_entropy']:.3f})"
                for m in focus_metrics
            )
        ),
    }
    return fig, summary


# =============================================================================
# Cross-validation: proving attention is (or isn't) functionally meaningful
# =============================================================================

# Method 5: Attention Rank Correlation (Spearman ρ between layers)
# Method 6: Attention Rollout (Abnar & Zuidema 2020)
# Method 7: Attention–Occlusion Correlation (per layer)


def compute_attention_rank_correlation(
    attention_weights_list: List[List[torch.Tensor]],
    edge_indices: List[Optional[torch.Tensor]],
    sequences: List[str],
    max_seq_len: int = 30,
) -> Dict[str, Any]:
    """Method 5: 逐样本计算 L0 与 L1（及更深层）bond-level attention 的 Spearman ρ。

    解读：
        - ρ ≈ +1：两层关注相同位置（只是平滑度不同）→ 浅层 pattern 被保留
        - ρ ≈ 0：两层关注完全不同位置 → 发生了 attention migration / reconstruction
        - ρ < 0：两层关注方向相反 → 反直觉，提示训练异常

    对每个样本：
        1. 提取 L0 bond attention (length = num_bonds)
        2. 提取 L1 bond attention (length = num_bonds)
        3. 计算 Spearman ρ
    跨样本聚合：返回每对层组合的 ρ 分布。

    Returns:
        Dict 含：
          - per_sample_rho: List[float] 每个样本的 ρ 值（L0 vs L1）
          - mean_rho, median_rho, std_rho
          - layer_pairs: [(0,1), (0,2), ...] 所有层对组合
          - per_pair_metrics: 每对的统计
    """
    from scipy.stats import spearmanr

    n_samples = len(attention_weights_list)
    if n_samples == 0:
        raise ValueError("empty attention_weights_list")
    num_layers = len(attention_weights_list[0])

    # 收集每层每样本的 bond attention
    per_layer_bond_attn: List[List[np.ndarray]] = [[] for _ in range(num_layers)]
    for attn_weights, edge_idx, seq in zip(attention_weights_list, edge_indices, sequences):
        for layer_idx in range(num_layers):
            bond_attn, _ = extract_bond_level_attention(
                attn_weights[layer_idx], edge_idx, seq, max_seq_len,
            )
            per_layer_bond_attn[layer_idx].append(bond_attn)

    # 所相邻层对（含 L0 vs L_last 等远端对）
    layer_pairs = []
    for i in range(num_layers):
        for j in range(i + 1, num_layers):
            layer_pairs.append((i, j))
    if num_layers == 2:
        # 2 层时只有 (0, 1)
        pass

    per_pair_metrics: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for (la, lb) in layer_pairs:
        rhos: List[float] = []
        for k in range(n_samples):
            a = per_layer_bond_attn[la][k]
            b = per_layer_bond_attn[lb][k]
            if len(a) < 2 or len(b) < 2:
                continue
            if np.std(a) == 0 or np.std(b) == 0:
                rhos.append(float('nan'))
                continue
            try:
                rho, _ = spearmanr(a, b)
                rhos.append(float(rho))
            except Exception:
                rhos.append(float('nan'))
        valid = [r for r in rhos if not np.isnan(r)]
        per_pair_metrics[(la, lb)] = {
            'per_sample_rho': rhos,
            'mean_rho': float(np.mean(valid)) if valid else float('nan'),
            'median_rho': float(np.median(valid)) if valid else float('nan'),
            'std_rho': float(np.std(valid)) if valid else float('nan'),
            'iqr': (
                float(np.percentile(valid, 25)),
                float(np.percentile(valid, 75)),
            ) if valid else (float('nan'), float('nan')),
            'n_valid': len(valid),
            'n_total': len(rhos),
        }

    return {
        'num_samples': n_samples,
        'num_layers': num_layers,
        'layer_pairs': layer_pairs,
        'per_pair_metrics': per_pair_metrics,
    }


def plot_attention_rank_correlation(
    rank_corr_summary: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 6),
) -> plt.Figure:
    """可视化 Method 5 结果：左图分布，右图示例样本 L0 vs L1 scatter。

    由于示例样本 scatter 需要原始 attention 数据（不在 summary 中），
    此函数仅绘制左图分布。如需 scatter，调用方自行绘制。
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plt.subplots_adjust(wspace=0.3, left=0.07, right=0.97, top=0.85, bottom=0.15)

    per_pair = rank_corr_summary['per_pair_metrics']

    # 左：每对层的 ρ 分布 boxplot
    ax = axes[0]
    pair_labels = [f"L{a}vsL{b}" for (a, b) in rank_corr_summary['layer_pairs']]
    box_data = []
    for (a, b) in rank_corr_summary['layer_pairs']:
        rhos = per_pair[(a, b)]['per_sample_rho']
        box_data.append([r for r in rhos if not np.isnan(r)])

    if box_data and any(len(d) > 0 for d in box_data):
        bp = ax.boxplot(box_data, tick_labels=pair_labels, patch_artist=True, widths=0.5,
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='white', markersize=7))
        for patch in bp['boxes']:
            patch.set_facecolor(INTERP_COLORS['fill'])
            patch.set_alpha(0.6)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        # 散点
        rng = np.random.default_rng(42)
        for i, d in enumerate(box_data):
            if d:
                jitter = rng.normal(0, 0.06, size=len(d))
                ax.scatter(np.full(len(d), i + 1) + jitter, d,
                           alpha=0.5, color=INTERP_COLORS['broken'],
                           s=25, edgecolor='black', linewidth=0.3, zorder=3)
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax.axhline(0.7, color=INTERP_COLORS['highlight'], linestyle=':',
                   linewidth=1.0, alpha=0.7, label='ρ=0.7 (strong)')
        ax.axhline(0.3, color=INTERP_COLORS['raw_r'], linestyle=':',
                   linewidth=1.0, alpha=0.7, label='ρ=0.3 (weak)')
        ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.set_ylabel('Spearman ρ (L_a vs L_b bond attention)',
                  fontsize=10, fontweight='bold')
    ax.set_title('(a) Per-sample rank correlation between layers',
                 fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 右：聚合统计柱状图
    ax = axes[1]
    pairs = rank_corr_summary['layer_pairs']
    if pairs:
        means = [per_pair[p]['mean_rho'] for p in pairs]
        medians = [per_pair[p]['median_rho'] for p in pairs]
        stds = [per_pair[p]['std_rho'] for p in pairs]
        x = np.arange(len(pairs))
        w = 0.35
        ax.bar(x - w/2, means, w, yerr=stds, label='Mean ± std',
               color=INTERP_COLORS['broken'], alpha=0.75,
               capsize=5, edgecolor='black', linewidth=0.5)
        ax.bar(x + w/2, medians, w, label='Median',
               color=INTERP_COLORS['intact'], alpha=0.75,
               edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f'L{a}vsL{b}' for (a, b) in pairs])
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axhline(0.7, color=INTERP_COLORS['highlight'], linestyle=':',
                   linewidth=1.0, alpha=0.6, label='strong (0.7)')
        ax.legend(loc='lower right', fontsize=8)
    ax.set_ylabel('Spearman ρ', fontsize=10, fontweight='bold')
    ax.set_title('(b) Aggregate rank correlation per layer pair',
                 fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 总标题 + 解读提示
    interpretations = []
    for (a, b) in pairs:
        m = per_pair[(a, b)]['mean_rho']
        if np.isnan(m):
            verdict = 'undefined'
        elif m >= 0.7:
            verdict = 'positions preserved (only smoothed)'
        elif m >= 0.3:
            verdict = 'partial migration'
        elif m >= -0.3:
            verdict = 'complete reconstruction'
        else:
            verdict = 'inversion (suspicious)'
        interpretations.append(f'L{a}vsL{b}: ρ={m:+.3f} → {verdict}')

    fig.suptitle(
        'Method 5: Attention Rank Correlation Between Layers\n'
        + ' | '.join(interpretations),
        fontsize=11, fontweight='bold', y=0.97,
    )

    _save_figure(fig, save_path)
    return fig


# -----------------------------------------------------------------------------
# Method 6: Attention Rollout (Abnar & Zuidema 2020)
# -----------------------------------------------------------------------------

def _build_node_attention_matrix(
    attention_weights: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    add_identity: bool = True,
    normalize: str = 'row',
) -> np.ndarray:
    """从 GAT edge attention 构建节点 × 节点 attention 矩阵。

    Args:
        attention_weights: [num_edges] 或 [num_edges, num_heads]，head 维会平均
        edge_index:        [2, num_edges]，edge_index[0]=src, [1]=dst
        num_nodes:         节点数（含或不含 global node 由调用方决定）
        add_identity:      是否加 I（rollout 标准做法）
        normalize:         'row' / 'col' / 'none'
                           - 'row': 每行归一化（信息"扩散"解释）
                           - 'col': 每列归一化（GAT 默认 softmax over src per dst）
                           - 'none': 不归一化
    """
    if attention_weights.dim() == 2:
        attn_np = attention_weights.mean(dim=1).cpu().numpy()
    else:
        attn_np = attention_weights.cpu().numpy()

    A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    edge_np = edge_index.cpu().numpy()
    n_edges = edge_np.shape[1]
    for i in range(min(n_edges, len(attn_np))):
        src, dst = int(edge_np[0, i]), int(edge_np[1, i])
        if 0 <= src < num_nodes and 0 <= dst < num_nodes:
            A[src, dst] += float(attn_np[i])

    if add_identity:
        A = A + np.eye(num_nodes)

    if normalize == 'row':
        row_sum = A.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        A = A / row_sum
    elif normalize == 'col':
        col_sum = A.sum(axis=0, keepdims=True)
        col_sum[col_sum == 0] = 1.0
        A = A / col_sum

    return A


def compute_attention_rollout(
    attention_weights_list: List[List[torch.Tensor]],
    edge_indices: List[Optional[torch.Tensor]],
    sequences: List[str],
    max_seq_len: int = 30,
    normalize: str = 'row',
) -> Dict[str, Any]:
    """Method 6: Attention Rollout (Abnar & Zuidema 2020).

    对每个样本：
        A_rollout = (A_0 + I) @ (A_1 + I) @ ... @ (A_L + I)

    其中 A_l 是第 l 层 GAT 的 node × node attention 矩阵（head 平均）。
    加 I 是为了保留节点自身信息（标准 rollout 做法）。

    解读：
        - Rollout 显示"输入节点对最终表示的累积影响"
        - 若 rollout 仍聚焦在 L0 的热点位置 → 浅层 pattern 通过深层被保留
        - 若 rollout 完全扩散 → 深层确实重新分配了 attention
        - 若 rollout 聚焦在 L1 之外的新位置 → 多层累积导致新的关键点

    Returns:
        Dict 含：
          - rollout_bond_attn: List[np.ndarray] 每个样本的 rollout bond-level attention
          - per_layer_focus: rollout 与每层 attention 的 Spearman ρ
                             (判断 rollout 最像哪一层)
    """
    from scipy.stats import spearmanr

    n_samples = len(attention_weights_list)
    if n_samples == 0:
        raise ValueError("empty attention_weights_list")
    num_layers = len(attention_weights_list[0])

    rollout_bond_attn_list: List[np.ndarray] = []
    per_layer_bond_attn_for_compare: List[List[np.ndarray]] = [[] for _ in range(num_layers)]

    for attn_weights, edge_idx, seq in zip(attention_weights_list, edge_indices, sequences):
        if edge_idx is None:
            continue
        seq_len = min(len(seq), max_seq_len)
        # rollout 在 residue 子图上做（排除 global node）
        num_nodes = seq_len

        # 逐层构建 + 累积乘
        rollout = np.eye(num_nodes)
        for layer_idx in range(num_layers):
            A_l = _build_node_attention_matrix(
                attn_weights[layer_idx], edge_idx, num_nodes,
                add_identity=True, normalize=normalize,
            )
            rollout = rollout @ A_l

        # 从 rollout 矩阵提取 bond-level attention
        # rollout[i, j] = info flow from i to j
        # 键 i 连接 i 和 i+1，取 rollout[i, i+1] 和 rollout[i+1, i] 的平均
        bond_attn = np.zeros(seq_len - 1)
        for i in range(seq_len - 1):
            bond_attn[i] = (rollout[i, i + 1] + rollout[i + 1, i]) / 2.0
        rollout_bond_attn_list.append(bond_attn)

        # 同时收集每层 bond attention 用于比较
        for layer_idx in range(num_layers):
            layer_bond_attn, _ = extract_bond_level_attention(
                attn_weights[layer_idx], edge_idx, seq, max_seq_len,
            )
            per_layer_bond_attn_for_compare[layer_idx].append(layer_bond_attn)

    # rollout 与每层 attention 的相关性
    per_layer_correlation: List[Dict[str, float]] = []
    for layer_idx in range(num_layers):
        rhos: List[float] = []
        for k in range(min(len(rollout_bond_attn_list),
                           len(per_layer_bond_attn_for_compare[layer_idx]))):
            a = rollout_bond_attn_list[k]
            b = per_layer_bond_attn_for_compare[layer_idx][k]
            n = min(len(a), len(b))
            if n < 2:
                continue
            if np.std(a[:n]) == 0 or np.std(b[:n]) == 0:
                continue
            try:
                rho, _ = spearmanr(a[:n], b[:n])
                rhos.append(float(rho))
            except Exception:
                pass
        per_layer_correlation.append({
            'layer': layer_idx,
            'mean_rho_rollout_vs_layer': float(np.mean(rhos)) if rhos else float('nan'),
            'median_rho': float(np.median(rhos)) if rhos else float('nan'),
            'n_samples': len(rhos),
        })

    return {
        'num_samples': len(rollout_bond_attn_list),
        'num_layers': num_layers,
        'normalize': normalize,
        'rollout_bond_attn': rollout_bond_attn_list,
        'per_layer_correlation': per_layer_correlation,
    }


def plot_attention_rollout(
    rollout_summary: Dict[str, Any],
    attention_weights_list: List[List[torch.Tensor]],
    edge_indices: List[Optional[torch.Tensor]],
    sequences: List[str],
    max_seq_len: int = 30,
    max_bonds_show: int = 15,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (16, 6),
) -> plt.Figure:
    """可视化 Method 6：rollout 与各层 attention 的 bond-level pattern 对比。

    左：每层 + rollout 的群体中位数 bond attention 折线图
    右：rollout 与每层的 Spearman ρ 柱状图
    """
    n_samples = rollout_summary['num_samples']
    num_layers = rollout_summary['num_layers']
    max_bonds = min(max_bonds_show, max_seq_len - 1)

    # 收集每层 + rollout 的 bond attention
    per_layer_bond = [[] for _ in range(num_layers)]
    rollout_bond = []
    for k, (attn_weights, edge_idx, seq) in enumerate(
        zip(attention_weights_list, edge_indices, sequences)
    ):
        if k >= n_samples:
            break
        seq_len = min(len(seq), max_seq_len)
        for layer_idx in range(num_layers):
            ba, _ = extract_bond_level_attention(
                attn_weights[layer_idx], edge_idx, seq, max_seq_len,
            )
            per_layer_bond[layer_idx].append(ba[:max_bonds])
        if k < len(rollout_summary['rollout_bond_attn']):
            r = rollout_summary['rollout_bond_attn'][k]
            rollout_bond.append(r[:max_bonds])

    # 计算中位数
    def _median_padded(lst):
        if not lst:
            return np.zeros(max_bonds)
        max_len = max(len(x) for x in lst)
        padded = [np.pad(x, (0, max_len - len(x)), constant_values=np.nan) for x in lst]
        stacked = np.vstack(padded)
        with np.errstate(all='ignore'):
            return np.nanmedian(stacked, axis=0)[:max_bonds]

    layer_medians = [_median_padded(per_layer_bond[i]) for i in range(num_layers)]
    rollout_median = _median_padded(rollout_bond)

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plt.subplots_adjust(wspace=0.3, left=0.07, right=0.97, top=0.85, bottom=0.15)

    # 左：折线对比
    ax = axes[0]
    x = np.arange(max_bonds)
    layer_colors = ['#95A5A6', '#3498DB', '#9B59B6', '#16A085', '#F39C12']
    for layer_idx in range(num_layers):
        color = layer_colors[layer_idx % len(layer_colors)]
        # 归一化到自身最大值便于对比形状
        m = layer_medians[layer_idx]
        m_norm = m / (m.max() if m.max() > 0 else 1.0)
        ax.plot(x, m_norm, 'o--', color=color, linewidth=1.5,
                markersize=5, alpha=0.7, label=f'L{layer_idx} (per-row normed)', zorder=2)
    rollout_norm = rollout_median / (rollout_median.max() if rollout_median.max() > 0 else 1.0)
    ax.plot(x, rollout_norm, 's-', color=INTERP_COLORS['broken'], linewidth=2.8,
            markersize=10, markeredgecolor='black', markeredgewidth=1.3,
            label='Rollout (per-row normed)', zorder=4)
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x], fontsize=8)
    ax.set_xlabel('Bond position', fontsize=10, fontweight='bold')
    ax.set_ylabel('Median attention (per-row normalized)', fontsize=10, fontweight='bold')
    ax.set_title('(a) Layer-wise vs Rollout bond attention pattern',
                 fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # 右：rollout 与每层的 ρ 柱状图
    ax = axes[1]
    per_layer_corr = rollout_summary['per_layer_correlation']
    layer_ids = [m['layer'] for m in per_layer_corr]
    rhos = [m['mean_rho_rollout_vs_layer'] for m in per_layer_corr]
    x = np.arange(len(layer_ids))
    bars = ax.bar(x, rhos, color=[layer_colors[i % len(layer_colors)] for i in layer_ids],
                  alpha=0.8, edgecolor='black', linewidth=0.5)
    for bar, r in zip(bars, rhos):
        if not np.isnan(r):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{r:+.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axhline(0.7, color=INTERP_COLORS['highlight'], linestyle=':',
               linewidth=1.0, alpha=0.7, label='ρ=0.7 (strong)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{i}' for i in layer_ids])
    ax.set_ylim(min(-0.1, min(rhos + [0]) - 0.1), max(0.9, max(rhos + [0]) + 0.15))
    ax.set_ylabel('Spearman ρ (rollout vs layer)', fontsize=10, fontweight='bold')
    ax.set_title('(b) Which layer does rollout resemble?',
                 fontsize=10, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        f'Method 6: Attention Rollout (n={n_samples} samples, '
        f'normalize={rollout_summary["normalize"]})\n'
        f'Rollout reveals cumulative input→output attention flow across all layers',
        fontsize=11, fontweight='bold', y=0.97,
    )

    _save_figure(fig, save_path)
    return fig


# -----------------------------------------------------------------------------
# Method 7: Attention–Occlusion Correlation (per layer)
# -----------------------------------------------------------------------------

def compute_attention_occlusion_correlation(
    attention_weights_list: List[List[torch.Tensor]],
    edge_indices: List[Optional[torch.Tensor]],
    sequences: List[str],
    occlusion_matrices: List[np.ndarray],
    max_seq_len: int = 30,
) -> Dict[str, Any]:
    """Method 7: 逐层计算 attention 与 occlusion 敏感度的相关性。

    验证：哪一层的 attention 与因果归因（occlusion）最一致？
        - L_X attention ≈ occlusion → 该层 attention 反映 functional importance
        - L_Y attention ⊥ occlusion → 该层 attention 仅做 information mixing

    每层 attention 转为 residue × bond 矩阵 [seq_len, num_bonds]，
    与 occlusion sensitivity [seq_len, num_bonds] 计算 Pearson r 和 Spearman ρ。

    Args:
        attention_weights_list: 每个样本的 [layer_0, layer_1, ...]
        edge_indices:           每个样本的 edge_index
        sequences:              每个样本的序列
        occlusion_matrices:     每个样本的 occlusion sensitivity [seq_len, num_bonds]
    """
    from scipy.stats import spearmanr

    n_samples = len(attention_weights_list)
    if n_samples == 0:
        raise ValueError("empty attention_weights_list")
    if len(occlusion_matrices) != n_samples:
        raise ValueError(
            f"Length mismatch: {n_samples} attention samples vs "
            f"{len(occlusion_matrices)} occlusion matrices"
        )
    num_layers = len(attention_weights_list[0])

    per_layer_per_sample_r: List[List[float]] = [[] for _ in range(num_layers)]
    per_layer_per_sample_rho: List[List[float]] = [[] for _ in range(num_layers)]

    for k in range(n_samples):
        attn_weights = attention_weights_list[k]
        edge_idx = edge_indices[k]
        seq = sequences[k]
        occ = occlusion_matrices[k]

        seq_len = min(len(seq), max_seq_len)
        num_bonds = seq_len - 1
        occ_mat = occ[:seq_len, :num_bonds] if occ.shape[0] >= seq_len and occ.shape[1] >= num_bonds else None
        if occ_mat is None or occ_mat.size < 2:
            continue

        for layer_idx in range(num_layers):
            # 构建 residue × bond attention 矩阵
            residue_attn = build_residue_attention_matrix(
                attn_weights[layer_idx], edge_idx, seq_len,
            )
            res_bond_attn = collapse_to_residue_bond_attention(residue_attn, seq_len)
            att_mat = res_bond_attn[:seq_len, :num_bonds]
            if att_mat.size < 2 or np.std(att_mat) == 0 or np.std(occ_mat) == 0:
                per_layer_per_sample_r[layer_idx].append(float('nan'))
                per_layer_per_sample_rho[layer_idx].append(float('nan'))
                continue
            try:
                r = float(np.corrcoef(att_mat.flatten(), occ_mat.flatten())[0, 1])
                rho, _ = spearmanr(att_mat.flatten(), occ_mat.flatten())
                per_layer_per_sample_r[layer_idx].append(r)
                per_layer_per_sample_rho[layer_idx].append(float(rho))
            except Exception:
                per_layer_per_sample_r[layer_idx].append(float('nan'))
                per_layer_per_sample_rho[layer_idx].append(float('nan'))

    per_layer_metrics: List[Dict[str, Any]] = []
    for layer_idx in range(num_layers):
        valid_r = [x for x in per_layer_per_sample_r[layer_idx] if not np.isnan(x)]
        valid_rho = [x for x in per_layer_per_sample_rho[layer_idx] if not np.isnan(x)]
        per_layer_metrics.append({
            'layer': layer_idx,
            'mean_pearson_r': float(np.mean(valid_r)) if valid_r else float('nan'),
            'median_pearson_r': float(np.median(valid_r)) if valid_r else float('nan'),
            'std_pearson_r': float(np.std(valid_r)) if valid_r else float('nan'),
            'mean_spearman_rho': float(np.mean(valid_rho)) if valid_rho else float('nan'),
            'median_spearman_rho': float(np.median(valid_rho)) if valid_rho else float('nan'),
            'n_valid': len(valid_r),
        })

    return {
        'num_samples': n_samples,
        'num_layers': num_layers,
        'per_layer_metrics': per_layer_metrics,
        'per_layer_per_sample_r': per_layer_per_sample_r,
        'per_layer_per_sample_rho': per_layer_per_sample_rho,
    }


def plot_attention_occlusion_correlation(
    attn_occ_summary: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 6),
) -> plt.Figure:
    """可视化 Method 7：左图 per-sample 分布 boxplot，右图 per-layer 聚合柱状图。

    解读：
        - 若 L0 mean r ≈ 0.4-0.5, L1 mean r ≈ 0.1-0.2：
            → "L0 attention ≈ functional importance"（你的猜想得到验证）
            → "L1 attention ≈ information mixing"（深层不再代表 importance）
        - 若两层都 ≈ 0：attention 都不解释 model behavior
        - 若两层都 > 0.5：attention 始终是 importance 的良好代理
    """
    per_layer = attn_occ_summary['per_layer_metrics']
    per_sample_r = attn_occ_summary['per_layer_per_sample_r']
    num_layers = attn_occ_summary['num_layers']

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plt.subplots_adjust(wspace=0.3, left=0.07, right=0.97, top=0.85, bottom=0.15)

    # 左：per-sample 分布 boxplot
    ax = axes[0]
    layer_labels = [f'L{m["layer"]}' for m in per_layer]
    box_data = [[x for x in per_sample_r[i] if not np.isnan(x)]
                for i in range(num_layers)]
    if any(len(d) > 0 for d in box_data):
        bp = ax.boxplot(box_data, tick_labels=layer_labels, patch_artist=True, widths=0.5,
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='white', markersize=8))
        layer_colors = ['#95A5A6', '#3498DB', '#9B59B6', '#16A085']
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(layer_colors[i % len(layer_colors)])
            patch.set_alpha(0.6)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        rng = np.random.default_rng(42)
        for i, d in enumerate(box_data):
            if d:
                jitter = rng.normal(0, 0.06, size=len(d))
                ax.scatter(np.full(len(d), i + 1) + jitter, d,
                           alpha=0.5, color=INTERP_COLORS['broken'],
                           s=25, edgecolor='black', linewidth=0.3, zorder=3)
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax.axhline(0.5, color=INTERP_COLORS['highlight'], linestyle=':',
                   linewidth=1.0, alpha=0.7, label='r=0.5 (strong)')
        ax.axhline(0.3, color=INTERP_COLORS['raw_r'], linestyle=':',
                   linewidth=1.0, alpha=0.7, label='r=0.3 (moderate)')
        ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.set_ylabel('Pearson r (attention vs occlusion)',
                  fontsize=10, fontweight='bold')
    ax.set_title('(a) Per-sample correlation distribution per layer',
                 fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 右：per-layer 聚合
    ax = axes[1]
    layer_ids = [m['layer'] for m in per_layer]
    means_r = [m['mean_pearson_r'] for m in per_layer]
    means_rho = [m['mean_spearman_rho'] for m in per_layer]
    stds_r = [m['std_pearson_r'] for m in per_layer]
    x = np.arange(len(layer_ids))
    w = 0.35
    ax.bar(x - w/2, means_r, w, yerr=stds_r, label='Pearson r (mean ± std)',
           color=INTERP_COLORS['broken'], alpha=0.75,
           capsize=5, edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, means_rho, w, label='Spearman ρ (mean)',
           color=INTERP_COLORS['intact'], alpha=0.75,
           edgecolor='black', linewidth=0.5)
    for i, (r, rho) in enumerate(zip(means_r, means_rho)):
        if not np.isnan(r):
            ax.text(i - w/2, r + 0.02, f'{r:+.2f}',
                    ha='center', fontsize=9, fontweight='bold')
        if not np.isnan(rho):
            ax.text(i + w/2, rho + 0.02, f'{rho:+.2f}',
                    ha='center', fontsize=9, fontweight='bold')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axhline(0.5, color=INTERP_COLORS['highlight'], linestyle=':',
               linewidth=1.0, alpha=0.6, label='strong (0.5)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{i}' for i in layer_ids])
    ax.set_ylabel('Correlation with occlusion', fontsize=10, fontweight='bold')
    ax.set_title('(b) Per-layer attention ↔ occlusion agreement',
                 fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # 总标题 + 解读
    interpretations = []
    for m in per_layer:
        r = m['mean_pearson_r']
        if np.isnan(r):
            verdict = 'undefined'
        elif r >= 0.5:
            verdict = '≈ functional importance'
        elif r >= 0.3:
            verdict = 'partial importance'
        elif r >= 0.1:
            verdict = 'information mixing'
        else:
            verdict = 'orthogonal to importance'
        interpretations.append(f'L{m["layer"]}: r={r:+.3f} → {verdict}')

    fig.suptitle(
        f'Method 7: Attention–Occlusion Correlation per Layer\n'
        + ' | '.join(interpretations),
        fontsize=11, fontweight='bold', y=0.97,
    )

    _save_figure(fig, save_path)
    return fig
