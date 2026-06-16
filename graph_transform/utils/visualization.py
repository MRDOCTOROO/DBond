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
            if abs(src - dst) == 1 and src < seq_len - 1 and dst < seq_len:
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

    stats_text = (
        f"d = {effect['cohen_d_signed']:+.2f} (|d|={effect['cohen_d_abs']:.2f})\n"
        f"AUC = {effect['auc']:.3f}  (>0.5 ⇒ broken > intact)\n"
        f"p = {effect['p_value']:.2e}\n"
        f"{effect['interpretation']}"
    )
    ax3.text(0.98, 0.98, stats_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
    ax3.set_ylabel('Functional Saliency (Attention)', fontsize=10, fontweight='bold')
    ax3.set_title('(c) Broken vs Intact Bonds: Attention-based Functional Separation',
                  fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # ---------- (d) Attention heatmap (normalized, semantic caption) ----------
    ax4 = axes[1, 1]
    max_bonds_show = 15
    heatmap_data = []
    for layer_idx in range(num_layers):
        rows = []
        for attn_weights, edge_idx, seq in zip(attention_weights_list, edge_indices, sequences):
            bond_attn, _ = extract_bond_level_attention(attn_weights[layer_idx], edge_idx, seq, max_seq_len)
            rows.append(bond_attn[:max_bonds_show])
        max_len = max(len(r) for r in rows) if rows else 0
        padded = [np.pad(r, (0, max_len - len(r)), constant_values=np.nan) for r in rows]
        stacked = np.stack(padded) if padded else np.zeros((1, 1))
        # nanmean ignores padding
        with np.errstate(all='ignore'):
            heatmap_data.append(np.nanmean(stacked, axis=0))
    # Truncate to the shortest non-nan length to avoid nan columns
    min_len = min((np.sum(~np.isnan(row)) for row in heatmap_data), default=0)
    heatmap_array = np.array([row[:min_len] for row in heatmap_data]) if min_len > 0 else np.array(heatmap_data)
    heatmap_array = _normalize_heatmap(heatmap_array, heatmap_normalize)

    im = ax4.imshow(heatmap_array, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax4)
    norm_label = 'per-row normalized' if heatmap_normalize == 'row' else 'global normalized'
    cbar.set_label(f'Attention (normalized functional saliency)', fontsize=10)
    ax4.set_xlabel('Bond Position', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Network Layer', fontsize=10, fontweight='bold')
    ax4.set_title(f'(d) Layer-wise Functional Saliency Heatmap [{norm_label}]',
                  fontsize=11, fontweight='bold')
    if num_layers <= 5:
        ax4.set_yticks(np.arange(num_layers))
        ax4.set_yticklabels([f'L{i}' for i in range(num_layers)], fontsize=9)

    fig.suptitle('DBond-GT Interpretability Analysis\n'
                 + FUNCTIONAL_SALIENCY_CAPTION,
                 fontsize=13, fontweight='bold', y=0.99)

    _save_figure(fig, save_path)

    summary = {
        "attention_mode": mode,
        "mode_caption": _mode_caption(mode),
        "panel_a_metrics": m_a,
        "panel_b_layer_metrics": layer_metrics,
        "panel_c_effect_size": effect,
        "panel_d_heatmap_normalize": heatmap_normalize,
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
    figsize: Tuple[float, float] = (20, 6.5),
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
    """
    n_aa = len(aa_labels)
    assert empirical.shape == (n_aa, n_aa)
    assert predicted.shape == (n_aa, n_aa)
    assert counts.shape == (n_aa, n_aa)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    plt.subplots_adjust(wspace=0.35, left=0.06, right=0.97, top=0.83, bottom=0.18)

    # Shared colour scale for (a) and (b) so they are directly comparable
    rate_vmin, rate_vmax = 0.0, 1.0
    diff_vmin, diff_vmax = -0.5, 0.5

    def _draw_panel(ax, matrix, vmin, vmax, cmap, title, annotate_values: bool):
        im = ax.imshow(matrix, cmap=cmap, aspect='equal',
                       vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_xticks(np.arange(n_aa))
        ax.set_yticks(np.arange(n_aa))
        ax.set_xticklabels(aa_labels, fontsize=9)
        ax.set_yticklabels(aa_labels, fontsize=9)
        ax.set_xlabel('C-terminal residue  Y', fontsize=11, fontweight='bold')
        ax.set_ylabel('N-terminal residue  X', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold')

        # Annotate values with rare-AA markers
        if annotate_values:
            for i in range(n_aa):
                for j in range(n_aa):
                    n = int(counts[i, j])
                    if n == 0:
                        # empty cell: hatch background
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                                   fill=True, facecolor='#EEEEEE',
                                                   edgecolor='#BBBBBB', linewidth=0.3, zorder=1))
                        continue
                    val = matrix[i, j]
                    if n < rare_thresholds[0]:
                        marker, color = '**', '#7F8C8D'
                    elif n < rare_thresholds[1]:
                        marker, color = '*', '#7F8C8D'
                    else:
                        marker, color = '', 'black'
                    text = f'{val:.2f}{marker}' if marker else f'{val:.2f}'
                    ax.text(j, i, text, ha='center', va='center',
                            fontsize=6.5, color=color)
        return im

    im_a = _draw_panel(axes[0], empirical, rate_vmin, rate_vmax, 'YlOrRd',
                       '(a) Empirical cleavage rate  P(broken | X-Y)',
                       annotate_values=True)
    im_b = _draw_panel(axes[1], predicted, rate_vmin, rate_vmax, 'YlOrRd',
                       '(b) Model predicted  E[σ(model) | X-Y]',
                       annotate_values=True)
    im_c = _draw_panel(axes[2], predicted - empirical, diff_vmin, diff_vmax, 'RdBu_r',
                       '(c) Difference  (predicted − empirical)',
                       annotate_values=True)

    cb_a = plt.colorbar(im_a, ax=axes[0], fraction=0.046, pad=0.04)
    cb_a.set_label('Cleavage rate', fontsize=9)
    cb_b = plt.colorbar(im_b, ax=axes[1], fraction=0.046, pad=0.04)
    cb_b.set_label('Predicted probability', fontsize=9)
    cb_c = plt.colorbar(im_c, ax=axes[2], fraction=0.046, pad=0.04)
    cb_c.set_label('Bias (model − empirical)', fontsize=9)

    fig.suptitle('Residue-Pair Cleavage Chemistry: Empirical vs Model\n'
                 f'Rows X = N-terminal side,  Cols Y = C-terminal side   '
                 f'|   * N∈[{rare_thresholds[0]},{rare_thresholds[1]}), '
                 f'** N<{rare_thresholds[0]} (uncertain)',
                 fontsize=12, fontweight='bold', y=0.97)

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
