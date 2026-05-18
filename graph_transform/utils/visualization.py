"""
注意力可视化工具

本模块提供注意力权重的可视化功能，包括热力图、肽段结构图等。
所有图表使用英文标签，确保兼容性。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
                                   figsize: Tuple[int, int] = (14, 6),
                                   node_size: int = 250,
                                   edge_scale: float = 4.0,
                                   show_sequence: bool = False) -> plt.Figure:
    """
    绘制多层肽段注意力图的合并版本（横向排列，减少留白）
    
    Args:
        sequence: 肽段序列
        attention_weights_list: 各层注意力权重列表 [layer0_weights, layer1_weights, ...]
        edge_index: 边索引 [2, num_edges]
        bond_labels: 键断裂标签（0或1）
        save_path: 保存路径
        figsize: 图形大小（宽, 高）
        node_size: 节点大小
        edge_scale: 边宽度缩放因子
        show_sequence: 是否在图片下方显示序列（默认False，因为坐标轴已经有序列）
        
    Returns:
        plt.Figure: matplotlib图形对象
    """
    num_layers = len(attention_weights_list)
    seq_len = len(sequence)
    
    # 创建子图布局：每层2个图（序列图 + 热力图），横向排列
    # 使用 gridspec 使布局更紧凑
    fig = plt.figure(figsize=figsize)
    
    if show_sequence:
        # 如果显示序列，需要额外空间
        gs = fig.add_gridspec(2, num_layers * 2, height_ratios=[5, 0.3], 
                             hspace=0.15, wspace=0.3, 
                             left=0.05, right=0.92, top=0.88, bottom=0.12)
    else:
        # 不显示序列，布局更紧凑
        gs = fig.add_gridspec(1, num_layers * 2, 
                             wspace=0.3, 
                             left=0.05, right=0.92, top=0.88, bottom=0.08)
    
    im = None  # 用于存储最后一个 imshow 对象，用于 colorbar
    
    for layer_idx, attention_weights in enumerate(attention_weights_list):
        # 处理注意力权重
        if attention_weights.dim() == 2:
            edge_weights_np = attention_weights.mean(dim=1).cpu().numpy()
        else:
            edge_weights_np = attention_weights.cpu().numpy()
        
        # 构建节点到节点的注意力矩阵
        num_nodes = seq_len
        if edge_index is not None:
            num_nodes = int(edge_index.max().item()) + 1
        
        effective_len = min(seq_len, num_nodes)
        
        # 创建注意力矩阵
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
        
        # 获取子图轴
        ax_graph = fig.add_subplot(gs[0, layer_idx * 2])      # 序列图
        ax_heatmap = fig.add_subplot(gs[0, layer_idx * 2 + 1])  # 热力图
        
        # === 左侧：肽段序列图 ===
        G = nx.DiGraph()
        
        for i in range(effective_len):
            G.add_node(i, label=sequence[i])
        
        edges = []
        edge_weights_list = []
        edge_colors = []
        
        for i in range(effective_len - 1):
            weight_forward = attention_matrix[i, i + 1]
            weight_backward = attention_matrix[i + 1, i]
            avg_weight = (weight_forward + weight_backward) / 2
            
            edges.append((i, i + 1))
            edge_weights_list.append(avg_weight)
            
            if bond_labels is not None and i < bond_labels.size(0):
                if bond_labels[i].item() == 1:
                    edge_colors.append('red')
                else:
                    edge_colors.append('blue')
            else:
                edge_colors.append('blue')
        
        for i, (src, dst) in enumerate(edges):
            G.add_edge(src, dst, weight=edge_weights_list[i], color=edge_colors[i])
        
        pos = {i: (i, 0) for i in range(effective_len)}
        
        nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_size=node_size, 
                              node_color='lightblue', alpha=0.8)
        
        edge_widths = [w * edge_scale for w in edge_weights_list]
        edge_colors_list = [G[u][v]['color'] for u, v in G.edges()]
        
        nx.draw_networkx_edges(G, pos, ax=ax_graph, width=edge_widths, 
                              edge_color=edge_colors_list, alpha=0.7,
                              arrows=True, arrowsize=8)
        
        # 节点标签（氨基酸单字母）- 使用更小的字体
        labels = {i: sequence[i] for i in range(effective_len)}
        nx.draw_networkx_labels(G, pos, labels, ax=ax_graph, font_size=8, font_weight='bold')
        
        # 标题放在上方
        ax_graph.set_title(f"(a) Layer {layer_idx} Sequence Graph", fontsize=10, fontweight='bold', pad=5)
        ax_graph.axis('off')
        
        # === 右侧：注意力热力图 ===
        adjacent_weights = np.zeros((effective_len, effective_len))
        for i in range(effective_len - 1):
            adjacent_weights[i, i + 1] = attention_matrix[i, i + 1]
            adjacent_weights[i + 1, i] = attention_matrix[i + 1, i]
        
        im = ax_heatmap.imshow(adjacent_weights, cmap='YlOrRd', aspect='auto')
        
        # 设置刻度标签为氨基酸序列
        if effective_len <= 30:
            ax_heatmap.set_xticks(range(effective_len))
            ax_heatmap.set_xticklabels(list(sequence[:effective_len]), fontsize=7, rotation=0)
            ax_heatmap.set_yticks(range(effective_len))
            ax_heatmap.set_yticklabels(list(sequence[:effective_len]), fontsize=7)
        else:
            # 序列太长时只显示部分刻度
            tick_interval = max(1, effective_len // 20)
            ticks = list(range(0, effective_len, tick_interval))
            ax_heatmap.set_xticks(ticks)
            ax_heatmap.set_xticklabels([sequence[i] if i < len(sequence) else '' for i in ticks], 
                                       fontsize=7, rotation=0)
            ax_heatmap.set_yticks(ticks)
            ax_heatmap.set_yticklabels([sequence[i] if i < len(sequence) else '' for i in ticks], 
                                       fontsize=7)
        
        # 标题放在上方
        ax_heatmap.set_title(f"(b) Layer {layer_idx} Attention Weights", fontsize=10, fontweight='bold', pad=5)
        ax_heatmap.set_xlabel("Position", fontsize=9)
        if layer_idx == 0:
            ax_heatmap.set_ylabel("Position", fontsize=9)
        
        # 在热力图下方显示序列（可选）
        if show_sequence:
            ax_seq = fig.add_subplot(gs[1, layer_idx * 2 + 1])
            ax_seq.text(0.5, 0.5, sequence[:effective_len], fontsize=8, 
                       fontfamily='monospace', ha='center', va='center', transform=ax_seq.transAxes)
            ax_seq.axis('off')
    
    # 添加总标题（简洁版，只显示序列前20个字符）
    seq_display = sequence[:20] + ('...' if len(sequence) > 20 else '')
    fig.suptitle(f"Peptide: {seq_display}", fontsize=11, fontweight='bold', y=0.98)
    
    # 添加颜色条（共享，放在右侧）
    if im is not None:
        cbar_ax = fig.add_axes([0.94, 0.08, 0.015, 0.8])
        plt.colorbar(im, cax=cbar_ax, label='Attention Weight')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved combined peptide attention to {save_path}")
    
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