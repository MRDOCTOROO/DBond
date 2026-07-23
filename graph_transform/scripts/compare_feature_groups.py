#!/usr/bin/env python3
"""Feature-group progressive addition 消融实验对比分析。

实验设计：6 个 setting（baseline_none + 4 个 +X + full），每个 5 折交叉验证。
本脚本汇总对比，回答"charge / mass+intensity / NCE / scan_num 哪类实验条件贡献最大"。

输入：各 setting 的 per-fold 明细（由 run_feature_group_ablation.py 调 train_5fold.py 产出）
  - <ckpt_base>/<setting_key>/5fold/<ts>/5fold_metrics.csv   per-fold 明细(每折一行,用于配对检验)
  - <ckpt_base>/<setting_key>/5fold/<ts>/5fold_summary.csv   聚合统计(每指标一行)

输出(默认 result/metric/graph_transform/feature_group_comparison/):
  1. feature_group_summary.csv     各 setting × 各指标的 mean±std 宽表
  2. feature_group_delta.csv       各 +X vs baseline_none 的 Δ + Δ%（单组贡献）
  3. feature_group_stat_test.csv   各 +X vs baseline_none 的 Wilcoxon signed-rank
                                  （含原始 p 值与 BH-FDR 校正后 q 值，5 次多重比较）
  4. feature_group_barplot.pdf     分组柱状图（6 setting × 关键指标，误差棒=std，柱顶标显著性）

用法:
  # 自动发现 ckpt_base 下各 setting 的最新 5fold_metrics.csv
  python graph_transform/scripts/compare_feature_groups.py \
      --ckpt_base checkpoints/graph_transform/feature_group_ablation

  # 显式指定各 setting 的 metrics.csv（可只传部分 setting）
  python graph_transform/scripts/compare_feature_groups.py \
      --metrics_csv baseline_none=checkpoints/.../5fold_metrics.csv \
                    charge_only=checkpoints/.../5fold_metrics.csv
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# 6 个 setting 的固定顺序与显示标签（与 run_feature_group_ablation.py 对齐）
SETTING_ORDER = [
    "baseline_none",
    "charge_only",
    "mass_intensity_only",
    "nce_only",
    "scan_num_only",
    "full",
]
SETTING_LABELS = {
    "baseline_none": "Baseline\n(no state/env)",
    "charge_only": "+Charge",
    "mass_intensity_only": "+Mass+Intensity",
    "nce_only": "+NCE",
    "scan_num_only": "+Scan_num",
    "full": "Full\n(all features)",
}
# 默认对比的指标及绘图顺序（按重要性）
DEFAULT_METRICS = ["accuracy", "precision", "recall", "f1", "auc"]
# 不参与统计检验/绘图的列（per-fold 明细里的元数据）
META_COLUMNS = {
    "fold_id", "seed", "best_epoch", "best_val_f1",
    "checkpoint_dir", "metric_csv_path", "pred_csv_path",
}
# 默认的 +X setting（与 baseline_none 配对做 Wilcoxon 的实验组）
DELTA_SETTINGS = ["charge_only", "mass_intensity_only", "nce_only", "scan_num_only"]
BASELINE_KEY = "baseline_none"
ALPHA = 0.05
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CKPT_BASE = "checkpoints/graph_transform/feature_group_ablation"


def resolve_rel(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def find_latest_metrics(setting_dir: str) -> Optional[str]:
    """在 setting 目录下递归找最新的 5fold_metrics.csv。"""
    if not os.path.isdir(setting_dir):
        return None
    found = []
    for dirpath, _, files in os.walk(setting_dir):
        if "5fold_metrics.csv" in files:
            found.append(os.path.join(dirpath, "5fold_metrics.csv"))
    if not found:
        return None
    return max(found, key=os.path.getmtime)


def discover_setting_metrics(ckpt_base: str, settings: List[str]) -> Dict[str, str]:
    """在 ckpt_base 下为每个 setting 找最新的 5fold_metrics.csv。"""
    result = {}
    for s in settings:
        setting_dir = os.path.join(ckpt_base, s)
        path = find_latest_metrics(setting_dir)
        if path:
            result[s] = path
        else:
            print(f"[warn] 未找到 setting={s} 的 5fold_metrics.csv (搜索 {setting_dir})")
    return result


def load_per_fold(metrics_csv: str) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    if "fold_id" not in df.columns:
        raise ValueError(f"{metrics_csv} 缺少 fold_id 列，无法配对。")
    return df.sort_values("fold_id").reset_index(drop=True)


def get_metric_columns(df: pd.DataFrame) -> list:
    cols = [c for c in df.columns if c not in META_COLUMNS]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def align_all_folds(per_fold: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """取所有 setting 的 fold_id 交集并按 fold_id 对齐，保证配对检验成立。"""
    common = None
    for df in per_fold.values():
        ids = set(df["fold_id"])
        common = ids if common is None else (common & ids)
    common = sorted(common) if common else []
    if len(common) < 3:
        raise ValueError(
            f"共同 fold_id 仅 {len(common)} 个（<3），Wilcoxon 无法做。"
            f"各 setting fold_id：{ {k: sorted(v['fold_id'].tolist()) for k, v in per_fold.items()} }"
        )
    aligned = {}
    for key, df in per_fold.items():
        missing = sorted(set(df["fold_id"]) - set(common))
        if missing:
            print(f"[warn] setting={key} 丢弃非共同 fold：{missing}")
        aligned[key] = df.set_index("fold_id").loc[common].reset_index()
    return aligned


def fmt_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f}±{std:.4f}"


def significance_marker(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def benjamini_hochberg(pvalues: List[float]) -> List[float]:
    """BH-FDR 校正，返回 q 值列表（与输入顺序一致）。

    手写实现避免引入 statsmodels 依赖。公式：
      q_(i) = min over k>=i of (p_(k) * m / k)，其中 m=比较总数，按 p 升序排。
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    q = [0.0] * m
    running_min = 1.0
    # 从大到小（即 p 值最大的开始）回溯取 min
    for rank_from_top, idx in enumerate(reversed(order)):
        rank = m - rank_from_top  # 1-based 升序 rank
        corrected = pvalues[idx] * m / rank
        if corrected > 1.0:
            corrected = 1.0
        if corrected < running_min:
            running_min = corrected
        q[idx] = running_min
    return q


def build_summary_table(
    per_fold: Dict[str, pd.DataFrame], metrics: List[str]
) -> pd.DataFrame:
    """宽表：每行一个指标，列为各 setting 的 mean±std。"""
    rows = []
    for m in metrics:
        row = {"metric": m}
        for s in SETTING_ORDER:
            if s not in per_fold:
                continue
            series = pd.to_numeric(per_fold[s][m], errors="coerce").dropna()
            if series.empty:
                continue
            mean, std = float(series.mean()), float(series.std(ddof=0))
            row[f"{s}_mean"] = mean
            row[f"{s}_std"] = std
            row[f"{s}_mean_std"] = fmt_mean_std(mean, std)
        rows.append(row)
    return pd.DataFrame(rows)


def build_delta_table(
    per_fold: Dict[str, pd.DataFrame], metrics: List[str],
    baseline: str = BASELINE_KEY, delta_settings: List[str] = DELTA_SETTINGS,
) -> pd.DataFrame:
    """各 +X vs baseline 的 Δ + Δ%（单组贡献）。"""
    if baseline not in per_fold:
        print(f"[warn] baseline={baseline} 缺失，无法计算 Δ。")
        return pd.DataFrame()
    rows = []
    for m in metrics:
        base = pd.to_numeric(per_fold[baseline][m], errors="coerce").dropna()
        if base.empty:
            continue
        base_mean = float(base.mean())
        for s in delta_settings:
            if s not in per_fold:
                continue
            cur = pd.to_numeric(per_fold[s][m], errors="coerce").dropna()
            if cur.empty:
                continue
            cur_mean = float(cur.mean())
            delta = cur_mean - base_mean
            delta_pct = (delta / abs(base_mean) * 100.0) if base_mean != 0 else float("nan")
            rows.append({
                "metric": m,
                "setting": s,
                "baseline_mean": base_mean,
                "setting_mean": cur_mean,
                "delta": delta,
                "delta_pct": delta_pct,
            })
    return pd.DataFrame(rows)


def run_statistical_tests(
    per_fold: Dict[str, pd.DataFrame], metrics: List[str],
    baseline: str = BASELINE_KEY, delta_settings: List[str] = DELTA_SETTINGS,
) -> pd.DataFrame:
    """各 +X vs baseline 的 Wilcoxon signed-rank + BH-FDR。"""
    if baseline not in per_fold:
        print(f"[warn] baseline={baseline} 缺失，无法做统计检验。")
        return pd.DataFrame()
    rows = []
    raw_pvals = []  # 平行收集原始 p 值，用于 BH 校正
    for m in metrics:
        base = pd.to_numeric(per_fold[baseline][m], errors="coerce").dropna().to_numpy()
        for s in delta_settings:
            if s not in per_fold:
                continue
            cur = pd.to_numeric(per_fold[s][m], errors="coerce").dropna().to_numpy()
            n = min(len(base), len(cur))
            if n < 3:
                rows.append({
                    "metric": m, "setting": s, "n_pairs": n,
                    "statistic": np.nan, "p_value": np.nan,
                    "significant_raw": False, "marker_raw": "n/a(<3)",
                })
                raw_pvals.append(np.nan)
                continue
            b, c = base[:n], cur[:n]
            diff = c - b
            if np.all(diff == 0):
                rows.append({
                    "metric": m, "setting": s, "n_pairs": n,
                    "statistic": np.nan, "p_value": 1.0,
                    "significant_raw": False, "marker_raw": "n.s.(identical)",
                })
                raw_pvals.append(1.0)
                continue
            try:
                stat, p = stats.wilcoxon(c, b, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                stat, p = np.nan, np.nan
            rows.append({
                "metric": m, "setting": s, "n_pairs": n,
                "statistic": float(stat) if not np.isnan(stat) else np.nan,
                "p_value": float(p) if not np.isnan(p) else np.nan,
                "significant_raw": bool(not np.isnan(p) and p < ALPHA),
                "marker_raw": significance_marker(p),
            })
            raw_pvals.append(float(p) if not np.isnan(p) else np.nan)

    # BH-FDR 校正（忽略 nan）
    qvals = benjamini_hochberg(raw_pvals)
    for row, q in zip(rows, qvals):
        row["q_value_bh"] = q
        row["significant_fdr"] = bool(not np.isnan(q) and q < ALPHA)
        row["marker_fdr"] = significance_marker(q)
    return pd.DataFrame(rows)


def plot_grouped_bar(
    summary: pd.DataFrame, test: pd.DataFrame, out_path: str,
    metrics: List[str], present_settings: List[str],
):
    """分组柱状图：每个指标一组，组内各 setting 一根柱。"""
    if summary.empty:
        print("[warn] summary 表为空，跳过绘图。")
        return
    n_metrics = len(metrics)
    n_settings = len(present_settings)
    x = np.arange(n_metrics)
    width = 0.8 / max(n_settings, 1)

    fig, ax = plt.subplots(figsize=(max(8, 1.8 * n_metrics + 2), 5.0))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_settings))
    # baseline 用灰色突出对比
    for i, s in enumerate(present_settings):
        means = []
        stds = []
        for m in metrics:
            row = summary[summary["metric"] == m]
            if row.empty or f"{s}_mean" not in row.columns:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(float(row[f"{s}_mean"].iloc[0]))
                stds.append(float(row[f"{s}_std"].iloc[0]))
        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)
        color = "#999999" if s == BASELINE_KEY else colors[i]
        label = SETTING_LABELS.get(s, s)
        ax.bar(x + i * width - 0.4 + width / 2, means, width,
               yerr=np.nan_to_num(stds), capsize=3,
               label=label, color=color, edgecolor="black", linewidth=0.5)

    # 在各 +X 柱顶标 FDR 显著性（与 baseline_none 对比）
    if not test.empty and "marker_fdr" in test.columns:
        test_lookup = {(r["metric"], r["setting"]): r["marker_fdr"] for _, r in test.iterrows()}
        ymax_per_metric = []
        for j, m in enumerate(metrics):
            max_top = -np.inf
            for i, s in enumerate(present_settings):
                if s == BASELINE_KEY or s == "full":
                    continue
                row = summary[summary["metric"] == m]
                if row.empty or f"{s}_mean" not in row.columns:
                    continue
                top = float(row[f"{s}_mean"].iloc[0]) + float(row[f"{s}_std"].iloc[0])
                max_top = max(max_top, top)
                marker = test_lookup.get((m, s), "")
                if marker and marker not in ("n/a", "n/a(<3)"):
                    xpos = x[j] + i * width - 0.4 + width / 2
                    ax.text(xpos, top + 0.005, marker, ha="center", va="bottom",
                            fontsize=9, fontweight="bold", color=colors[i])
            ymax_per_metric.append(max_top)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=0)
    ax.set_ylabel("score")
    ax.set_title("Feature-group progressive addition (5-fold CV, paired Wilcoxon vs baseline, BH-FDR)")
    # y 轴上限留出标注空间
    all_means, all_stds = [], []
    for m in metrics:
        row = summary[summary["metric"] == m]
        for s in present_settings:
            if f"{s}_mean" in row.columns:
                all_means.append(float(row[f"{s}_mean"].iloc[0]))
                all_stds.append(float(row[f"{s}_std"].iloc[0]))
    if all_means:
        ymax = max(np.array(all_means) + np.array(all_stds))
        ax.set_ylim(bottom=0.0, top=min(1.0, ymax * 1.15) if ymax > 0 else 1.0)
    ax.legend(loc="lower right", ncol=2, frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def discover_inputs(args) -> Dict[str, str]:
    """解析各 setting 的 per-fold metrics.csv 路径。返回 {setting_key: csv_path}。"""
    result = {}
    # 1. 显式 --metrics_csv key=path（优先）
    if args.metrics_csv:
        for kv in args.metrics_csv:
            if "=" not in kv:
                sys.exit(f"[error] --metrics_csv 格式应为 setting=path，收到：{kv}")
            key, path = kv.split("=", 1)
            result[key.strip()] = resolve_rel(path.strip())
        return result
    # 2. 自动按 ckpt_base 发现
    ckpt_base = resolve_rel(args.ckpt_base)
    print(f"[discover] 在 {ckpt_base} 下自动发现各 setting 的 5fold_metrics.csv ...")
    discovered = discover_setting_metrics(ckpt_base, SETTING_ORDER)
    for s, path in discovered.items():
        print(f"[discover] {s:25s} -> {path}")
    return discovered


def main():
    parser = argparse.ArgumentParser(
        description="Feature-group progressive addition 消融对比（6 setting × 5 fold）"
    )
    parser.add_argument(
        "--ckpt_base", default=DEFAULT_CKPT_BASE,
        help=f"各 setting 的根目录（默认 {DEFAULT_CKPT_BASE}）",
    )
    parser.add_argument(
        "--metrics_csv", nargs="+",
        help="显式指定各 setting 的 metrics.csv，格式 setting_key=path（可多个）",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=DEFAULT_METRICS,
        help=f"对比的指标列，默认 {DEFAULT_METRICS}",
    )
    parser.add_argument(
        "--baseline", default=BASELINE_KEY,
        help=f"Δ 与统计检验的参照 setting，默认 {BASELINE_KEY}",
    )
    parser.add_argument(
        "--delta_settings", nargs="+", default=DELTA_SETTINGS,
        help=f"与 baseline 配对做 Wilcoxon 的 +X setting，默认 {DELTA_SETTINGS}",
    )
    parser.add_argument(
        "--output_dir",
        default="result/metric/graph_transform/feature_group_comparison",
        help="输出目录",
    )
    args = parser.parse_args()

    setting_metrics = discover_inputs(args)
    if not setting_metrics:
        sys.exit(f"[error] 未发现任何 setting 的 metrics.csv。请检查 --ckpt_base 或用 --metrics_csv 显式指定。")

    # 校验文件存在
    for s, path in setting_metrics.items():
        if not os.path.isfile(path):
            sys.exit(f"[error] setting={s} 的 metrics.csv 不存在：{path}")

    # 加载 per-fold 明细
    per_fold_raw = {s: load_per_fold(path) for s, path in setting_metrics.items()}
    print(f"[load] 已加载 settings：{list(per_fold_raw.keys())}")

    # 校验请求的指标列都存在
    available = get_metric_columns(next(iter(per_fold_raw.values())))
    missing = [m for m in args.metrics if m not in available]
    if missing:
        print(f"[warn] 以下指标在 per-fold 表中不存在，已忽略：{missing}")
    metrics = [m for m in args.metrics if m in available]
    if not metrics:
        sys.exit(f"[error] 没有可对比的指标。per-fold 表数值列：{available}")

    # 对齐 fold
    per_fold = align_all_folds(per_fold_raw)
    common_folds = per_fold[next(iter(per_fold))]["fold_id"].tolist()
    print(f"[align] 配对折数 = {len(common_folds)} (fold_id={common_folds})")

    # 按固定顺序过滤出存在的 setting
    present_settings = [s for s in SETTING_ORDER if s in per_fold]

    # 三张表
    summary = build_summary_table(per_fold, metrics)
    delta = build_delta_table(per_fold, metrics, baseline=args.baseline, delta_settings=args.delta_settings)
    test = run_statistical_tests(per_fold, metrics, baseline=args.baseline, delta_settings=args.delta_settings)

    out_dir = resolve_rel(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "feature_group_summary.csv")
    delta_path = os.path.join(out_dir, "feature_group_delta.csv")
    stat_path = os.path.join(out_dir, "feature_group_stat_test.csv")
    plot_path = os.path.join(out_dir, "feature_group_barplot.pdf")

    summary.to_csv(summary_path, index=False)
    delta.to_csv(delta_path, index=False)
    test.to_csv(stat_path, index=False)
    print(f"[save] feature_group_summary   -> {summary_path}")
    print(f"[save] feature_group_delta     -> {delta_path}")
    print(f"[save] feature_group_stat_test -> {stat_path}")

    # 打印人读视图：各 setting 的 mean±std
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    view_cols = ["metric"] + [f"{s}_mean_std" for s in present_settings if f"{s}_mean_std" in summary.columns]
    print("\n=== 各 setting mean±std (5-fold) ===")
    print(summary[view_cols].to_string(index=False))

    if not delta.empty:
        print("\n=== Δ vs baseline (单组贡献) ===")
        print(delta[["metric", "setting", "baseline_mean", "setting_mean", "delta", "delta_pct"]].to_string(index=False))

    if not test.empty:
        print("\n=== Wilcoxon (vs baseline, BH-FDR) ===")
        print(test[["metric", "setting", "n_pairs", "p_value", "marker_raw",
                    "q_value_bh", "marker_fdr"]].to_string(index=False))

    plot_grouped_bar(summary, test, plot_path, metrics, present_settings)
    print(f"\n[done] 全部产出在 {out_dir}")


if __name__ == "__main__":
    main()
