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
    FUNCTIONAL_SALIENCY_CAPTION,
    detect_attention_mode,
    compute_effect_size,
    compute_separation_metrics,
    compute_layer_separation_metrics,
    # 残基对化学矩阵 + Occlusion 归因（paper-grade extension）
    plot_residue_pair_matrix,
    build_residue_attention_matrix,
    collapse_to_residue_bond_attention,
    plot_occlusion_vs_attention,
    plot_occlusion_attention_consistency,
    # 聚合图层注意力（基于多样本的群体平均模式，比单样本更可信）
    plot_aggregate_layer_attention,
    plot_aggregate_layer_attention_compact,
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
    'FUNCTIONAL_SALIENCY_CAPTION',
    'detect_attention_mode',
    'compute_effect_size',
    'compute_separation_metrics',
    'compute_layer_separation_metrics',
    # 残基对化学矩阵 + Occlusion 归因（paper-grade extension）
    'plot_residue_pair_matrix',
    'build_residue_attention_matrix',
    'collapse_to_residue_bond_attention',
    'plot_occlusion_vs_attention',
    'plot_occlusion_attention_consistency',
    # 聚合图层注意力（基于多样本的群体平均模式，比单样本更可信）
    'plot_aggregate_layer_attention',
    'plot_aggregate_layer_attention_compact',
]