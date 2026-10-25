"""
可解释性分析模块

提供模型可解释性分析和可视化功能，用于论文图表生成。
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


def plot_interpretability_case_study(
    attention_weights_list: List[torch.Tensor],
    bond_labels_list: List[torch.Tensor],
    sequences: List[str],
    edge_indices: List[Optional[torch.Tensor]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    num_cases: int = 2,
    max_seq_len: int = 30
) -> plt.Figure:
    """
    绘制4子图可解释性案例分析图
    
    Args:
        attention_weights_list: 各样本的注意力权重列表 [sample_idx][layer_idx]
        bond_labels_list: 各样本的键断裂标签列表 [sample_idx]
        sequences: 各样本的序列列表 [sample_idx]
        edge_indices: 各样本的边索引列表 [sample_idx]
        save_path: 保存路径
        figsize: 图形大小
        num_cases: 案例数量（默认2个）
        max_seq_len: 最大序列长度（超过则截断显示）
        
    Returns:
        plt.Figure: matplotlib图形对象
    """
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # 创建4子图布局
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.subplots_adjust(hspace=0.35, wspace=0.3, left=0.08, right=0.95, top=0.92, bottom=0.08)
    
    # 颜色配置
    colors = {
        'broken': '#E74C3C',      # 红色
        'intact': '#3498DB',      # 蓝色
        'highlight': '#F39C12',   # 橙色高亮
        'line': '#2C3E50',        # 深色线条
        'scatter_broken': '#E74C3C',
        'scatter_intact': '#3498DB',
    }
    
    # ========== 子图(a)：案例1 ==========
    ax1 = axes[0, 0]
    _plot_case_study_single(
        ax1, attention_weights_list[0], bond_labels_list[0], 
        sequences[0], edge_indices[0], 
        case_id=1, colors=colors, max_seq_len=max_seq_len
    )
    ax1.set_title('(a) Case Study 1: Attention Distribution', 
                  fontsize=12, fontweight='bold', pad=10)
    
    # ========== 子图(b)：案例2 ==========
    ax2 = axes[0, 1]
    if len(attention_weights_list) > 1:
        _plot_case_study_single(
            ax2, attention_weights_list[1], bond_labels_list[1],
            sequences[1], edge_indices[1],
            case_id=2, colors=colors, max_seq_len=max_seq_len
        )
    else:
        ax2.text(0.5, 0.5, 'Insufficient samples', ha='center', va='center', fontsize=12)
    ax2.set_title('(b) Case Study 2: Attention Distribution', 
                  fontsize=12, fontweight='bold', pad=10)
    
    # ========== 子图(c)：注意力权重箱线图 ==========
    ax3 = axes[1, 0]
    _plot_attention_boxplot(
        ax3, attention_weights_list, bond_labels_list, 
        edge_indices, sequences, colors
    )
    ax3.set_title('(c) Attention Weight Distribution: Broken vs Intact', 
                  fontsize=12, fontweight='bold', pad=10)
    
    # ========== 子图(d)：注意力-断裂相关性散点图 ==========
    ax4 = axes[1, 1]
    _plot_attention_correlation_scatter(
        ax4, attention_weights_list, bond_labels_list,
        edge_indices, sequences, colors
    )
    ax4.set_title('(d) Attention-Breakage Correlation', 
                  fontsize=12, fontweight='bold', pad=10)
    
    # 添加总标题
    fig.suptitle('DBond-GT Model Interpretability Analysis', 
                fontsize=14, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved interpretability case study to {save_path}")
    
    return fig


def _plot_case_study_single(
    ax: plt.Axes,
    attention_weights,  # 可以是 Tensor 或 List[Tensor]
    bond_labels: torch.Tensor,
    sequence: str,
    edge_index: Optional[torch.Tensor],
    case_id: int,
    colors: Dict[str, str],
    max_seq_len: int = 30
) -> None:
    """
    绘制单个案例的注意力权重分布
    """
    seq_len = min(len(sequence), max_seq_len)
    
    # 处理注意力权重：如果是列表，取所有层的平均
    if isinstance(attention_weights, list):
        # 列表格式：[layer0_weights, layer1_weights, ...]
        # 取所有层的平均
        if len(attention_weights) > 0:
            if isinstance(attention_weights[0], torch.Tensor):
                # 如果是张量列表，取平均
                attn_weights = torch.stack([w.mean(dim=1) if w.dim() == 2 else w 
                                           for w in attention_weights]).mean(dim=0).cpu().numpy()
            else:
                attn_weights = np.mean(attention_weights, axis=0)
        else:
            attn_weights = np.zeros(seq_len)
    elif isinstance(attention_weights, torch.Tensor):
        # 张量格式
        if attention_weights.dim() == 2:
            # [num_edges, num_heads] - 取平均
            attn_weights = attention_weights.mean(dim=1).cpu().numpy()
        else:
            attn_weights = attention_weights.cpu().numpy()
    else:
        attn_weights = np.array(attention_weights)
    
    # 构建位置级别的注意力权重
    position_weights = np.zeros(seq_len)
    position_counts = np.zeros(seq_len)
    
    if edge_index is not None:
        edge_index_np = edge_index.cpu().numpy()
        for i in range(edge_index_np.shape[1]):
            src, dst = int(edge_index_np[0, i]), int(edge_index_np[1, i])
            if src < seq_len and dst < seq_len and i < len(attn_weights):
                # 只处理相邻位置
                if abs(src - dst) == 1:
                    position_weights[src] += attn_weights[i]
                    position_weights[dst] += attn_weights[i]
                    position_counts[src] += 1
                    position_counts[dst] += 1
    
    # 避免除零
    position_counts = np.maximum(position_counts, 1)
    position_weights = position_weights / position_counts
    
    # 归一化到[0,1]
    if position_weights.max() > 0:
        position_weights = position_weights / position_weights.max()
    
    # 获取键标签（每个位置对应一个键）
    # 键i连接位置i和i+1
    num_bonds = min(seq_len - 1, len(bond_labels))
    bond_labels_np = bond_labels[:num_bonds].cpu().numpy() if isinstance(bond_labels, torch.Tensor) else bond_labels[:num_bonds]
    
    # 绘制注意力权重曲线
    positions = np.arange(num_bonds)
    attn_values = position_weights[:num_bonds]
    
    # 绘制背景区域（断裂键用红色背景）
    for i in range(num_bonds):
        if bond_labels_np[i] == 1:
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.2, color=colors['broken'])
    
    # 绘制注意力权重
    ax.plot(positions, attn_values, color=colors['line'], linewidth=2, 
            marker='o', markersize=6, label='Attention Weight', zorder=3)
    
    # 标记断裂键和完整键
    broken_mask = bond_labels_np == 1
    intact_mask = bond_labels_np == 0
    
    ax.scatter(positions[broken_mask], attn_values[broken_mask], 
              color=colors['broken'], s=100, zorder=5, 
              label='Broken Bonds', edgecolors='black', linewidth=1)
    ax.scatter(positions[intact_mask], attn_values[intact_mask], 
              color=colors['intact'], s=100, zorder=5,
              label='Intact Bonds', edgecolors='black', linewidth=1)
    
    # 高亮Top-3注意力位置
    top_k = min(3, num_bonds)
    top_indices = np.argsort(attn_values)[-top_k:]
    for idx in top_indices:
        ax.annotate(f'{sequence[idx]}-{sequence[idx+1]}', 
                   xy=(idx, attn_values[idx]),
                   xytext=(0, 15), textcoords='offset points',
                   ha='center', fontsize=8, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color=colors['highlight']),
                   color=colors['highlight'])
    
    # 设置坐标轴
    ax.set_xlabel('Bond Position', fontsize=10)
    ax.set_ylabel('Normalized Attention Weight', fontsize=10)
    ax.set_xlim(-0.5, num_bonds - 0.5)
    ax.set_ylim(0, 1.1)
    
    # X轴标签显示氨基酸
    if num_bonds <= 20:
        ax.set_xticks(range(num_bonds))
        ax.set_xticklabels([f'{sequence[i]}' for i in range(num_bonds)], 
                          fontsize=8, rotation=0)
    
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_attention_boxplot(
    ax: plt.Axes,
    attention_weights_list,  # 可以是 List[Tensor] 或 List[List[Tensor]]
    bond_labels_list: List[torch.Tensor],
    edge_indices: List[Optional[torch.Tensor]],
    sequences: List[str],
    colors: Dict[str, str]
) -> None:
    """
    绘制注意力权重箱线图（断裂键 vs 完整键）
    """
    broken_weights = []
    intact_weights = []
    
    for idx, (attn_weights, bond_labels, edge_index, seq) in enumerate(
        zip(attention_weights_list, bond_labels_list, edge_indices, sequences)
    ):
        seq_len = min(len(seq), 30)
        
        # 处理注意力权重：如果是列表，取所有层的平均
        if isinstance(attn_weights, list):
            if len(attn_weights) > 0:
                if isinstance(attn_weights[0], torch.Tensor):
                    attn_np = torch.stack([w.mean(dim=1) if w.dim() == 2 else w 
                                          for w in attn_weights]).mean(dim=0).cpu().numpy()
                else:
                    attn_np = np.mean(attn_weights, axis=0)
            else:
                attn_np = np.zeros(seq_len)
        elif isinstance(attn_weights, torch.Tensor):
            if attn_weights.dim() == 2:
                attn_np = attn_weights.mean(dim=1).cpu().numpy()
            else:
                attn_np = attn_weights.cpu().numpy()
        else:
            attn_np = np.array(attn_weights)
        
        # 构建位置级别的注意力权重
        position_weights = np.zeros(seq_len)
        position_counts = np.zeros(seq_len)
        
        if edge_index is not None:
            edge_index_np = edge_index.cpu().numpy()
            for i in range(edge_index_np.shape[1]):
                src, dst = int(edge_index_np[0, i]), int(edge_index_np[1, i])
                if src < seq_len and dst < seq_len and i < len(attn_np):
                    if abs(src - dst) == 1:
                        position_weights[src] += attn_np[i]
                        position_weights[dst] += attn_np[i]
                        position_counts[src] += 1
                        position_counts[dst] += 1
        
        position_counts = np.maximum(position_counts, 1)
        position_weights = position_weights / position_counts
        
        # 归一化
        if position_weights.max() > 0:
            position_weights = position_weights / position_weights.max()
        
        # 按键类型分组
        num_bonds = min(seq_len - 1, len(bond_labels))
        bond_labels_np = bond_labels[:num_bonds].cpu().numpy() if isinstance(bond_labels, torch.Tensor) else bond_labels[:num_bonds]
        
        for i in range(num_bonds):
            if bond_labels_np[i] == 1:
                broken_weights.append(position_weights[i])
            else:
                intact_weights.append(position_weights[i])
    
    # 绘制箱线图
    if broken_weights and intact_weights:
        data_to_plot = [broken_weights, intact_weights]
        bp = ax.boxplot(data_to_plot, 
                       patch_artist=True,
                       widths=0.6)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Broken Bonds', 'Intact Bonds'])
        
        # 设置颜色
        bp['boxes'][0].set_facecolor(colors['broken'])
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor(colors['intact'])
        bp['boxes'][1].set_alpha(0.7)
        
        # 设置中位数颜色
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        
        # 添加散点
        np.random.seed(42)
        x1 = np.random.normal(1, 0.04, size=len(broken_weights))
        x2 = np.random.normal(2, 0.04, size=len(intact_weights))
        ax.scatter(x1, broken_weights, alpha=0.5, color=colors['broken'], s=20)
        ax.scatter(x2, intact_weights, alpha=0.5, color=colors['intact'], s=20)
        
        # 计算统计检验
        from scipy import stats
        if len(broken_weights) > 1 and len(intact_weights) > 1:
            t_stat, p_value = stats.ttest_ind(broken_weights, intact_weights)
            ax.text(0.5, 0.95, f'p = {p_value:.4f}', 
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_ylabel('Normalized Attention Weight', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')


def _plot_attention_correlation_scatter(
    ax: plt.Axes,
    attention_weights_list,  # 可以是 List[Tensor] 或 List[List[Tensor]]
    bond_labels_list: List[torch.Tensor],
    edge_indices: List[Optional[torch.Tensor]],
    sequences: List[str],
    colors: Dict[str, str]
) -> None:
    """
    绘制注意力权重与断裂标签的散点图
    """
    all_attn = []
    all_labels = []
    
    for idx, (attn_weights, bond_labels, edge_index, seq) in enumerate(
        zip(attention_weights_list, bond_labels_list, edge_indices, sequences)
    ):
        seq_len = min(len(seq), 30)
        
        # 处理注意力权重：如果是列表，取所有层的平均
        if isinstance(attn_weights, list):
            if len(attn_weights) > 0:
                if isinstance(attn_weights[0], torch.Tensor):
                    attn_np = torch.stack([w.mean(dim=1) if w.dim() == 2 else w 
                                          for w in attn_weights]).mean(dim=0).cpu().numpy()
                else:
                    attn_np = np.mean(attn_weights, axis=0)
            else:
                attn_np = np.zeros(seq_len)
        elif isinstance(attn_weights, torch.Tensor):
            if attn_weights.dim() == 2:
                attn_np = attn_weights.mean(dim=1).cpu().numpy()
            else:
                attn_np = attn_weights.cpu().numpy()
        else:
            attn_np = np.array(attn_weights)
        
        # 构建位置级别的注意力权重
        position_weights = np.zeros(seq_len)
        position_counts = np.zeros(seq_len)
        
        if edge_index is not None:
            edge_index_np = edge_index.cpu().numpy()
            for i in range(edge_index_np.shape[1]):
                src, dst = int(edge_index_np[0, i]), int(edge_index_np[1, i])
                if src < seq_len and dst < seq_len and i < len(attn_np):
                    if abs(src - dst) == 1:
                        position_weights[src] += attn_np[i]
                        position_weights[dst] += attn_np[i]
                        position_counts[src] += 1
                        position_counts[dst] += 1
        
        position_counts = np.maximum(position_counts, 1)
        position_weights = position_weights / position_counts
        
        # 归一化
        if position_weights.max() > 0:
            position_weights = position_weights / position_weights.max()
        
        # 收集数据
        num_bonds = min(seq_len - 1, len(bond_labels))
        bond_labels_np = bond_labels[:num_bonds].cpu().numpy() if isinstance(bond_labels, torch.Tensor) else bond_labels[:num_bonds]
        
        for i in range(num_bonds):
            all_attn.append(position_weights[i])
            all_labels.append(bond_labels_np[i])
    
    all_attn = np.array(all_attn)
    all_labels = np.array(all_labels)
    
    # 添加抖动避免重叠
    np.random.seed(42)
    jitter = np.random.normal(0, 0.05, size=len(all_labels))
    y_jittered = all_labels + jitter
    
    # 绘制散点图
    scatter_broken = ax.scatter(all_attn[all_labels == 1], y_jittered[all_labels == 1],
                               color=colors['scatter_broken'], s=50, alpha=0.6,
                               label='Broken Bonds', edgecolors='black', linewidth=0.5)
    scatter_intact = ax.scatter(all_attn[all_labels == 0], y_jittered[all_labels == 0],
                               color=colors['scatter_intact'], s=50, alpha=0.6,
                               label='Intact Bonds', edgecolors='black', linewidth=0.5)
    
    # 添加趋势线
    try:
        from scipy import stats
        z = np.polyfit(all_attn, all_labels, 1)
        p = np.poly1d(z)
        x_line = np.linspace(all_attn.min(), all_attn.max(), 100)
        ax.plot(x_line, p(x_line), color=colors['line'], linestyle='--', 
                linewidth=2, label=f'Trend (r={np.corrcoef(all_attn, all_labels)[0,1]:.3f})')
    except:
        pass
    
    ax.set_xlabel('Normalized Attention Weight', fontsize=10)
    ax.set_ylabel('Bond Label (0=Intact, 1=Broken)', fontsize=10)
    ax.set_ylim(-0.2, 1.3)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Intact', 'Broken'])
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)


def generate_interpretability_report(
    attention_weights_list: List[torch.Tensor],
    bond_labels_list: List[torch.Tensor],
    sequences: List[str],
    edge_indices: List[Optional[torch.Tensor]],
    output_dir: str
) -> Dict[str, str]:
    """
    生成完整的可解释性分析报告
    
    Returns:
        生成的文件路径字典
    """
    os.makedirs(output_dir, exist_ok=True)
    
    generated_files = {}
    
    # 1. 生成4子图案例分析图
    case_study_path = os.path.join(output_dir, 'interpretability_case_study.png')
    plot_interpretability_case_study(
        attention_weights_list, bond_labels_list,
        sequences, edge_indices, save_path=case_study_path
    )
    generated_files['case_study'] = case_study_path
    
    # 2. 生成统计摘要
    summary_path = os.path.join(output_dir, 'interpretability_summary.txt')
    _generate_statistical_summary(
        attention_weights_list, bond_labels_list,
        sequences, edge_indices, summary_path
    )
    generated_files['summary'] = summary_path
    
    logger.info(f"Generated interpretability report in {output_dir}")
    return generated_files


def _generate_statistical_summary(
    attention_weights_list,  # 可以是 List[Tensor] 或 List[List[Tensor]]
    bond_labels_list: List[torch.Tensor],
    sequences: List[str],
    edge_indices: List[Optional[torch.Tensor]],
    save_path: str
) -> None:
    """生成统计摘要"""
    from scipy import stats
    
    broken_weights = []
    intact_weights = []
    
    for attn_weights, bond_labels, edge_index, seq in zip(
        attention_weights_list, bond_labels_list, edge_indices, sequences
    ):
        seq_len = min(len(seq), 30)
        
        # 处理注意力权重：如果是列表，取所有层的平均
        if isinstance(attn_weights, list):
            if len(attn_weights) > 0:
                if isinstance(attn_weights[0], torch.Tensor):
                    attn_np = torch.stack([w.mean(dim=1) if w.dim() == 2 else w 
                                          for w in attn_weights]).mean(dim=0).cpu().numpy()
                else:
                    attn_np = np.mean(attn_weights, axis=0)
            else:
                attn_np = np.zeros(seq_len)
        elif isinstance(attn_weights, torch.Tensor):
            if attn_weights.dim() == 2:
                attn_np = attn_weights.mean(dim=1).cpu().numpy()
            else:
                attn_np = attn_weights.cpu().numpy()
        else:
            attn_np = np.array(attn_weights)
        
        position_weights = np.zeros(seq_len)
        position_counts = np.zeros(seq_len)
        
        if edge_index is not None:
            edge_index_np = edge_index.cpu().numpy()
            for i in range(edge_index_np.shape[1]):
                src, dst = int(edge_index_np[0, i]), int(edge_index_np[1, i])
                if src < seq_len and dst < seq_len and i < len(attn_np):
                    if abs(src - dst) == 1:
                        position_weights[src] += attn_np[i]
                        position_weights[dst] += attn_np[i]
                        position_counts[src] += 1
                        position_counts[dst] += 1
        
        position_counts = np.maximum(position_counts, 1)
        position_weights = position_weights / position_counts
        
        if position_weights.max() > 0:
            position_weights = position_weights / position_weights.max()
        
        num_bonds = min(seq_len - 1, len(bond_labels))
        bond_labels_np = bond_labels[:num_bonds].cpu().numpy() if isinstance(bond_labels, torch.Tensor) else bond_labels[:num_bonds]
        
        for i in range(num_bonds):
            if bond_labels_np[i] == 1:
                broken_weights.append(position_weights[i])
            else:
                intact_weights.append(position_weights[i])
    
    # 计算统计量
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("Interpretability Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total samples analyzed: {len(sequences)}\n")
        f.write(f"Total bonds analyzed: {len(broken_weights) + len(intact_weights)}\n")
        f.write(f"Broken bonds: {len(broken_weights)}\n")
        f.write(f"Intact bonds: {len(intact_weights)}\n\n")
        
        if broken_weights and intact_weights:
            f.write("Attention Weight Statistics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Broken bonds - Mean: {np.mean(broken_weights):.4f}, Std: {np.std(broken_weights):.4f}\n")
            f.write(f"Intact bonds - Mean: {np.mean(intact_weights):.4f}, Std: {np.std(intact_weights):.4f}\n\n")
            
            t_stat, p_value = stats.ttest_ind(broken_weights, intact_weights)
            f.write(f"Statistical Test (t-test):\n")
            f.write(f"  t-statistic: {t_stat:.4f}\n")
            f.write(f"  p-value: {p_value:.6f}\n")
            f.write(f"  Significant (p < 0.05): {'Yes' if p_value < 0.05 else 'No'}\n\n")
            
            correlation = np.corrcoef(
                np.array(broken_weights + intact_weights),
                np.array([1]*len(broken_weights) + [0]*len(intact_weights))
            )[0, 1]
            f.write(f"Correlation with breakage label: r = {correlation:.4f}\n")
    
    logger.info(f"Generated statistical summary to {save_path}")
