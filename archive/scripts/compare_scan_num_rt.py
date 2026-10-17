#!/usr/bin/env python3
"""scan_num vs rt 五折交叉验证对比分析。

输入: 两组 5 折实验产物(train_5fold.py 输出)
  - <cv_root>/5fold_metrics.csv   per-fold 明细(每折一行,用于配对检验)
  - <cv_root>/5fold_summary.csv   聚合统计(每指标一行)

输出(默认 result/metric/graph_transform/env_feature_comparison/):
  1. comparison_table.csv     每指标 scan_num vs rt 的 mean±std + Δ + Δ%
  2. statistical_test.csv     每指标 Wilcoxon signed-rank(5 折配对): statistic / p / significant
  3. comparison_barplot.pdf   分组柱状图(误差棒=std,柱顶标 p 值显著性)

用法:
  # 自动发现最新两组结果(按 tag 子目录)
  python graph_transform/scripts/compare_scan_num_rt.py

  # 显式指定两组 5fold 输出根目录
  python graph_transform/scripts/compare_scan_num_rt.py \
      --scan_num_root checkpoints/graph_transform/5fold/<ts_scan> \
      --rt_root checkpoints/graph_transform/5fold/<ts_rt>

  # 直接指定 per-fold metrics.csv
  python graph_transform/scripts/compare_scan_num_rt.py \
      --scan_num_metrics checkpoints/.../5fold_metrics.csv \
      --rt_metrics checkpoints/.../5fold_metrics.csv
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# 默认对比的指标及绘图顺序(按重要性)
DEFAULT_METRICS = ["accuracy", "precision", "recall", "f1", "auc"]
# 不参与统计检验/绘图的列(per-fold 明细里的元数据)
META_COLUMNS = {
    "fold_id", "seed", "best_epoch", "best_val_f1",
    "checkpoint_dir", "metric_csv_path", "pred_csv_path",
}
ALPHA = 0.05
SCAN_TAG = "scan_num_env"
RT_TAG = "rt_env"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_rel(path: str) -> str:
    """相对路径基于项目根解析。"""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def latest_cv_root_by_tag(tag: str) -> str:
    """在 checkpoints/graph_transform/ 下找 tag 匹配的最新 cv_root。

    apply_ablation_config 会把 checkpoint_dir 改成 .../graph_transform/<tag>,
    train_5fold 在其下建 5fold/<timestamp>/。因此 cv_root 的祖路径含 tag。
    匹配要求 tag 作为独立路径段出现(前后为分隔符),避免 rt_env 误中其它字样。
    """
    base = resolve_rel("checkpoints/graph_transform")
    candidates = []
    for dirpath, _dirs, files in os.walk(base):
        if "5fold_metrics.csv" not in files:
            continue
        # 归一化为 / 分隔,首尾补分隔符,用 /tag/ 做整段匹配
        norm = "/" + dirpath.replace(os.sep, "/").strip("/") + "/"
        if f"/{tag}/" in norm:
            candidates.append(dirpath)
    if not candidates:
        raise FileNotFoundError(
            f"未找到 tag={tag} 的 5fold_metrics.csv,已搜索 {base}。"
            f"请先用 --scan_num_metrics / --rt_metrics 显式指定。"
        )
    # 取最新修改时间
    latest = max(candidates, key=lambda p: os.path.getmtime(os.path.join(p, "5fold_metrics.csv")))
    return latest


def load_per_fold(metrics_csv: str) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    # 按 fold_id 排序,保证两组配对对齐(两组 fold_id 集合应一致)
    if "fold_id" not in df.columns:
        raise ValueError(f"{metrics_csv} 缺少 fold_id 列,无法配对。")
    df = df.sort_values("fold_id").reset_index(drop=True)
    return df


def align_folds(scan_df: pd.DataFrame, rt_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """取两组 fold_id 的交集并按 fold_id 对齐,保证 Wilcoxon 配对成立。"""
    common = sorted(set(scan_df["fold_id"]) & set(rt_df["fold_id"]))
    if len(common) < 3:
        raise ValueError(
            f"两组共同 fold_id 仅 {len(common)} 个(<3),Wilcoxon 无法做。"
            f"scan_num folds={sorted(scan_df['fold_id'].tolist())}, "
            f"rt folds={sorted(rt_df['fold_id'].tolist())}"
        )
    missing_scan = sorted(set(scan_df["fold_id"]) - set(rt_df["fold_id"]))
    missing_rt = sorted(set(rt_df["fold_id"]) - set(scan_df["fold_id"]))
    if missing_scan or missing_rt:
        print(f"[warn] 两组 fold_id 不完全一致,已取交集 {common}。"
              f" scan_num 独有={missing_scan}, rt 独有={missing_rt}")
    scan_aligned = scan_df.set_index("fold_id").loc[common].reset_index()
    rt_aligned = rt_df.set_index("fold_id").loc[common].reset_index()
    return scan_aligned, rt_aligned


def get_metric_columns(df: pd.DataFrame) -> list:
    cols = [c for c in df.columns if c not in META_COLUMNS]
    # 只保留数值列
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def fmt_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f}±{std:.4f}"


def significance_marker(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def build_comparison_table(
    scan_df: pd.DataFrame, rt_df: pd.DataFrame, metrics: list
) -> pd.DataFrame:
    rows = []
    for m in metrics:
        s = pd.to_numeric(scan_df[m], errors="coerce").dropna()
        r = pd.to_numeric(rt_df[m], errors="coerce").dropna()
        if s.empty or r.empty:
            continue
        s_mean, s_std = float(s.mean()), float(s.std(ddof=0))
        r_mean, r_std = float(r.mean()), float(r.std(ddof=0))
        delta = s_mean - r_mean
        delta_pct = (delta / abs(r_mean) * 100.0) if r_mean != 0 else float("nan")
        rows.append({
            "metric": m,
            "scan_num_mean": s_mean,
            "scan_num_std": s_std,
            "scan_num_mean_std": fmt_mean_std(s_mean, s_std),
            "rt_mean": r_mean,
            "rt_std": r_std,
            "rt_mean_std": fmt_mean_std(r_mean, r_std),
            "delta_scan_minus_rt": delta,
            "delta_pct": delta_pct,
        })
    return pd.DataFrame(rows)


def run_statistical_tests(
    scan_df: pd.DataFrame, rt_df: pd.DataFrame, metrics: list
) -> pd.DataFrame:
    rows = []
    for m in metrics:
        s = pd.to_numeric(scan_df[m], errors="coerce").dropna().to_numpy()
        r = pd.to_numeric(rt_df[m], errors="coerce").dropna().to_numpy()
        n = min(len(s), len(r))
        if n < 3:
            rows.append({"metric": m, "n_pairs": n, "statistic": np.nan,
                         "p_value": np.nan, "significant": "", "marker": "n/a(<3)"})
            continue
        # 配对前对齐到同一长度(理论上 align_folds 已保证,这里防御)
        s = s[:n]
        r = r[:n]
        diff = s - r
        # 若所有差值同号且无零差,Wilcoxon 仍可计算;若全部差值为0则跳过
        if np.all(diff == 0):
            rows.append({"metric": m, "n_pairs": n, "statistic": np.nan,
                         "p_value": 1.0, "significant": False, "marker": "n.s.(identical)"})
            continue
        try:
            stat, p = stats.wilcoxon(s, r, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            stat, p = np.nan, np.nan
        rows.append({
            "metric": m,
            "n_pairs": n,
            "statistic": float(stat) if not np.isnan(stat) else np.nan,
            "p_value": float(p) if not np.isnan(p) else np.nan,
            "significant": bool(not np.isnan(p) and p < ALPHA),
            "marker": significance_marker(p) if not np.isnan(p) else "n/a",
        })
    return pd.DataFrame(rows)


def plot_grouped_bar(
    comparison: pd.DataFrame, test: pd.DataFrame, out_path: str,
    scan_label: str = "scan_num", rt_label: str = "rt",
):
    if comparison.empty:
        print("[warn] comparison 表为空,跳过绘图。")
        return
    metrics = comparison["metric"].tolist()
    n = len(metrics)
    x = np.arange(n)
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * n + 2), 4.5))
    s_means = comparison["scan_num_mean"].to_numpy()
    s_stds = comparison["scan_num_std"].to_numpy()
    r_means = comparison["rt_mean"].to_numpy()
    r_stds = comparison["rt_std"].to_numpy()

    bars_s = ax.bar(x - width / 2, s_means, width, yerr=s_stds, capsize=4,
                    label=scan_label, color="#2c7fb8", edgecolor="black", linewidth=0.6)
    bars_r = ax.bar(x + width / 2, r_means, width, yerr=r_stds, capsize=4,
                    label=rt_label, color="#d95f0e", edgecolor="black", linewidth=0.6)

    # 柱顶标注 p 值显著性
    test_map = test.set_index("metric")["marker"].to_dict() if not test.empty else {}
    ymax = max(np.nanmax(s_means + s_stds), np.nanmax(r_means + r_stds)) if n else 1.0
    ymin = min(np.nanmin(s_means - s_stds), np.nanmin(r_means - r_stds)) if n else 0.0
    pad = (ymax - ymin) * 0.08 if ymax != ymin else 0.05
    for i, m in enumerate(metrics):
        marker = test_map.get(m, "")
        top = max(s_means[i] + s_stds[i], r_means[i] + r_stds[i])
        if marker and marker != "n/a(<3)":
            ax.text(x[i], top + pad, marker, ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=0)
    ax.set_ylabel("score")
    ax.set_title("scan_num vs rt (5-fold CV, paired Wilcoxon)")
    ax.set_ylim(bottom=max(0.0, ymin - pad * 2), top=ymax + pad * 3)
    ax.legend(loc="lower right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def discover_inputs(args) -> Tuple[str, str]:
    """按优先级解析 scan_num / rt 的 per-fold metrics.csv 路径。"""
    # 1. 显式 metrics.csv
    if args.scan_num_metrics and args.rt_metrics:
        return resolve_rel(args.scan_num_metrics), resolve_rel(args.rt_metrics)
    # 2. 显式 cv_root
    if args.scan_num_root and args.rt_root:
        s = os.path.join(resolve_rel(args.scan_num_root), "5fold_metrics.csv")
        r = os.path.join(resolve_rel(args.rt_root), "5fold_metrics.csv")
        return s, r
    # 3. 自动按 tag 发现
    print("[discover] 自动按 tag 发现最新 cv_root ...")
    scan_root = latest_cv_root_by_tag(SCAN_TAG)
    rt_root = latest_cv_root_by_tag(RT_TAG)
    print(f"[discover] scan_num cv_root = {scan_root}")
    print(f"[discover] rt       cv_root = {rt_root}")
    return os.path.join(scan_root, "5fold_metrics.csv"), os.path.join(rt_root, "5fold_metrics.csv")


def main():
    parser = argparse.ArgumentParser(
        description="scan_num vs rt 五折对比分析(表 + Wilcoxon + 柱状图)"
    )
    parser.add_argument("--scan_num_metrics", help="scan_num 组 per-fold 5fold_metrics.csv")
    parser.add_argument("--rt_metrics", help="rt 组 per-fold 5fold_metrics.csv")
    parser.add_argument("--scan_num_root", help="scan_num 组 cv_root(含 5fold_metrics.csv 的目录)")
    parser.add_argument("--rt_root", help="rt 组 cv_root")
    parser.add_argument(
        "--metrics", nargs="+", default=DEFAULT_METRICS,
        help=f"对比的指标列,默认 {DEFAULT_METRICS}",
    )
    parser.add_argument(
        "--output_dir",
        default="result/metric/graph_transform/env_feature_comparison",
        help="输出目录(默认 result/metric/graph_transform/env_feature_comparison)",
    )
    parser.add_argument("--scan_label", default="scan_num", help="图中 scan_num 组的标签")
    parser.add_argument("--rt_label", default="rt", help="图中 rt 组的标签")
    args = parser.parse_args()

    scan_csv, rt_csv = discover_inputs(args)
    for path in (scan_csv, rt_csv):
        if not os.path.isfile(path):
            sys.exit(f"[error] 找不到 per-fold metrics: {path}")

    print(f"[load] scan_num metrics: {scan_csv}")
    print(f"[load] rt       metrics: {rt_csv}")
    scan_df = load_per_fold(scan_csv)
    rt_df = load_per_fold(rt_csv)

    # 校验请求的指标列都存在
    available = get_metric_columns(scan_df)
    missing = [m for m in args.metrics if m not in available]
    if missing:
        print(f"[warn] 以下指标在 per-fold 表中不存在,已忽略: {missing}")
    metrics = [m for m in args.metrics if m in available]
    if not metrics:
        sys.exit(f"[error] 没有可对比的指标。per-fold 表数值列: {available}")

    scan_df, rt_df = align_folds(scan_df, rt_df)
    print(f"[align] 配对折数 = {len(scan_df)} (fold_id={scan_df['fold_id'].tolist()})")

    comparison = build_comparison_table(scan_df, rt_df, metrics)
    test = run_statistical_tests(scan_df, rt_df, metrics)

    out_dir = resolve_rel(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    table_path = os.path.join(out_dir, "comparison_table.csv")
    stat_path = os.path.join(out_dir, "statistical_test.csv")
    plot_path = os.path.join(out_dir, "comparison_barplot.pdf")

    comparison.to_csv(table_path, index=False)
    test.to_csv(stat_path, index=False)
    print(f"[save] comparison_table   -> {table_path}")
    print(f"[save] statistical_test   -> {stat_path}")

    # 合并打印一张人读表
    merged = comparison.merge(test[["metric", "p_value", "marker"]], on="metric", how="left")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    view = merged[["metric", "scan_num_mean_std", "rt_mean_std",
                   "delta_scan_minus_rt", "delta_pct", "p_value", "marker"]]
    print("\n=== scan_num vs rt (5-fold paired) ===")
    print(view.to_string(index=False))

    plot_grouped_bar(comparison, test, plot_path,
                     scan_label=args.scan_label, rt_label=args.rt_label)
    print(f"\n[done] 全部产出在 {out_dir}")


if __name__ == "__main__":
    main()
