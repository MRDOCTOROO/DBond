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
    plot_peptide_attention_compact,
    plot_attention_heads_combined,
    analyze_attention_patterns,
    plot_attention_analysis,
    create_attention_report,
    # 新版可解释性分析函数
    extract_bond_level_attention,
    plot_single_sample_layer_attention,
    plot_layer_evolution_trend,
    plot_bond_type_comparison,
    compute_layer_correlations,
    plot_new_interpretability_case_study,
    # 语义/统计辅助（paper-grade）
    INTERP_COLORS,
    VALID_ATTENTION_MODES,
    DEFAULT_ATTENTION_MODE,
    detect_attention_mode,
    compute_effect_size,
    compute_separation_metrics,
    compute_layer_separation_metrics,
)
from .interpretability import (
    plot_interpretability_case_study,
    generate_interpretability_report,
)

__all__ = [
    'AttentionExtractor',
    'extract_attention_weights_from_model',
    'plot_attention_heatmap',
    'plot_peptide_attention_graph',
    'plot_attention_head_comparison',
    'plot_peptide_attention_combined',
    'plot_peptide_attention_compact',
    'plot_attention_heads_combined',
    'analyze_attention_patterns',
    'plot_attention_analysis',
    'create_attention_report',
    'plot_interpretability_case_study',
    'generate_interpretability_report',
    # 新版可解释性分析函数
    'extract_bond_level_attention',
    'plot_single_sample_layer_attention',
    'plot_layer_evolution_trend',
    'plot_bond_type_comparison',
    'compute_layer_correlations',
    'plot_new_interpretability_case_study',
    # 语义/统计辅助（paper-grade）
    'INTERP_COLORS',
    'VALID_ATTENTION_MODES',
    'DEFAULT_ATTENTION_MODE',
    'detect_attention_mode',
    'compute_effect_size',
    'compute_separation_metrics',
    'compute_layer_separation_metrics',
]