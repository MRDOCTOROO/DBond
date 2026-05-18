"""
GraphTransformer 可视化工具包

本包提供注意力权重提取和可视化功能，用于模型可解释性分析。
"""

from .attention_extractor import AttentionExtractor, extract_attention_weights_from_model
from .visualization import (
    plot_attention_heatmap,
    plot_peptide_attention_graph,
    plot_attention_head_comparison,
    plot_peptide_attention_combined,
    plot_attention_heads_combined,
    analyze_attention_patterns,
    plot_attention_analysis,
    create_attention_report,
)

__all__ = [
    'AttentionExtractor',
    'extract_attention_weights_from_model',
    'plot_attention_heatmap',
    'plot_peptide_attention_graph',
    'plot_attention_head_comparison',
    'plot_peptide_attention_combined',
    'plot_attention_heads_combined',
    'analyze_attention_patterns',
    'plot_attention_analysis',
    'create_attention_report',
]