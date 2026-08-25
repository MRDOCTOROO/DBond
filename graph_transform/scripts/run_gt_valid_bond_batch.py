#!/usr/bin/env python3
"""GT 变体批量 valid-bond 统一指标生成（tab:ablation / LOFO / 主表 GT 行的 label 列重算）。

背景：GT 管线各变体的 label 指标矩阵宽 = 实际最大键数（内部可比，无 m/af 的
35 维 padding 问题），但主对比表换成 valid-bond 口径后，所有 GT 表格的 label
列必须同口径重算，否则跨表数字冲突（R-09）。

对注册表（DEFAULT_RUNS）中每个 GT 变体：
  1. 若 {cv_root}/r20_aggregation/per_fold/fold_{id}/pred.csv 五折不全 →
     自动调用 aggregate_r20_5fold.py 补评估（逐折加载 best_model 重跑推理，
     生成 metric/pred/ranking；已存在的折自动跳过 → 断点续跑安全）
  2. 全部就绪后一次性调用 valid_bond_metrics.py 统一重算
     bond_acc / bond_precision / bond_recall / bond_f1 / bond_mcc
     （valid bonds only，阈值 0.5，padding 无关）
  3. 汇总宽表：每变体一行 = subset/ex 五项（取自 {cv_root}/5fold_summary.csv，
     训练汇总、padding 无关）+ valid-bond 五项（替换原 label 四列 + MCC）

tag 自动探测：扫描 fold_*/checkpoints/ 的公共子目录名，无需手填。
5 个旧消融变体路径未知已留占位（None）——填路径或用 --extra name=path 追加。

用法（graphtrans 机，DBond 仓库根目录）：
  python graph_transform/scripts/run_gt_valid_bond_batch.py
  python graph_transform/scripts/run_gt_valid_bond_batch.py --only gt_pre lofo_no_nce
  python graph_transform/scripts/run_gt_valid_bond_batch.py \
      --extra wo_edge_features=checkpoints/graph_transform/5fold/<ts>_wo_edge

输出（默认 result/valid_bond_metrics_gt_all/）：
  valid_bond_wide.csv     宽表（"58.81±0.34" 百分数格式，论文直接取数）
  valid_bond_metrics.csv  valid_bond_metrics.py 原始长表（model × metric）
  per_model/*.csv         每变体逐折明细
  run.log
"""

from __future__ import annotations

import argparse
import collections
import glob
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
AGGREGATE_SCRIPT = os.path.join(SCRIPT_DIR, "aggregate_r20_5fold.py")
VALID_BOND_SCRIPT = os.path.join(SCRIPT_DIR, "valid_bond_metrics.py")

DEFAULT_FOLDS = ["1222", "2252", "3514", "6072", "9075"]

# 注册表：(显示名, cv_root 或 glob 模式)。tag 自动探测；None = 占位待填路径。
DEFAULT_RUNS: List[Tuple[str, Optional[str]]] = [
    # 主模型 obs full（r20_aggregation 已存在，只补 valid-bond 计算）
    ("dbond_gt_obs", "checkpoints/graph_transform/5fold/20260421_181316base"),
    # R-01 pre 版（pred 已生成）
    ("gt_pre", "checkpoints/graph_transform/pre_synthesis/5fold/*"),
    # R-03 LOFO 5 个 setting（每 setting 5fold/ 下单一时间戳，glob 直接匹配）
    ("lofo_no_charge", "checkpoints/graph_transform/lofo/lofo_no_charge/5fold/*"),
    ("lofo_no_mass", "checkpoints/graph_transform/lofo/lofo_no_mass/5fold/*"),
    ("lofo_no_intensity", "checkpoints/graph_transform/lofo/lofo_no_intensity/5fold/*"),
    ("lofo_no_nce", "checkpoints/graph_transform/lofo/lofo_no_nce/5fold/*"),
    ("lofo_no_scan", "checkpoints/graph_transform/lofo/lofo_no_scan/5fold/*"),
    # 旧消融表 6 行：sequence_graph 路径已知；其余 5 个待填（None 占位）
    ("sequence_graph", "checkpoints/graph_transform/5fold/20260422_232825*"),
    ("wo_message_passing", None),
    ("wo_edge_features", None),
    ("wo_state_env", None),
    ("wo_global_node", None),
    ("gcn_only", None),
]

# 宽表列：训练汇总（padding 无关） + valid-bond 口径（替换原 label 列）
SUMMARY_KEYS = ["subset_acc", "ex_acc", "ex_precision", "ex_recall", "ex_f1"]
BOND_KEYS = ["bond_acc", "bond_precision", "bond_recall", "bond_f1", "bond_mcc"]
WIDE_COLUMNS = SUMMARY_KEYS + BOND_KEYS

logger = logging.getLogger("gt_valid_bond_batch")


def setup_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s[%(levelname)s]:%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(os.path.join(output_dir, "run.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)


def resolve_cv_root(pattern: str) -> Optional[str]:
    """glob → 单一 cv_root。多个候选时取含 fold_*/config.yaml 且 mtime 最新的。"""
    candidates = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    if not candidates:
        return None
    with_folds = [p for p in candidates
                  if glob.glob(os.path.join(p, "fold_*", "config.yaml"))]
    pool = with_folds or candidates
    pool.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    if len(pool) > 1:
        logger.warning(f"  '{pattern}' 匹配到 {len(pool)} 个目录，取最新: {pool[0]}")
    return pool[0]


def detect_tag(cv_root: str) -> Optional[str]:
    """扫描各折 checkpoints/ 的公共子目录名作 tag（aggregate 定位 best_model 用）。"""
    counter: collections.Counter = collections.Counter()
    for fold_dir in sorted(glob.glob(os.path.join(cv_root, "fold_*"))):
        for sub in glob.glob(os.path.join(fold_dir, "checkpoints", "*")):
            if os.path.isdir(sub) and glob.glob(os.path.join(sub, "*", "best_model.pt")):
                counter[os.path.basename(sub)] += 1
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def all_pred_csv_exist(cv_root: str, folds: List[str]) -> bool:
    for fold_id in folds:
        path = os.path.join(cv_root, "r20_aggregation", "per_fold", f"fold_{fold_id}", "pred.csv")
        if not os.path.exists(path):
            return False
    return True


def run_aggregate(cv_root: str, tag: str, folds: List[str]) -> bool:
    """调用 aggregate_r20_5fold.py 补 pred.csv（已存在的折它自己会跳过）。"""
    cmd = [sys.executable, AGGREGATE_SCRIPT,
           "--cv_root", cv_root, "--tag", tag, "--folds", *folds]
    logger.info("  AGGREGATE CMD: " + " ".join(cmd))
    rc = subprocess_run(cmd)
    if rc != 0:
        logger.error(f"  aggregate 失败 (rc={rc}): {cv_root}")
        return False
    return all_pred_csv_exist(cv_root, folds)


def subprocess_run(cmd: List[str]) -> int:
    import subprocess
    return subprocess.run(cmd).returncode


def load_summary_mean_std(cv_root: str) -> Dict[str, Tuple[float, float]]:
    """读训练 5fold_summary.csv（metric,mean,std,... 长表）→ {metric: (mean,std)}。"""
    path = os.path.join(cv_root, "5fold_summary.csv")
    if not os.path.exists(path):
        logger.warning(f"  5fold_summary.csv 不存在，subset/ex 列将留空: {path}")
        return {}
    df = pd.read_csv(path)
    if not {"metric", "mean"}.issubset(df.columns):
        return {}
    std_col = "std" if "std" in df.columns else None
    out: Dict[str, Tuple[float, float]] = {}
    for _, row in df.iterrows():
        try:
            mean = float(row["mean"])
            std = float(row[std_col]) if std_col else 0.0
        except (ValueError, TypeError):
            continue
        out[str(row["metric"])] = (mean, std)
    return out


def fmt_pct(pair: Optional[Tuple[float, float]]) -> str:
    if pair is None:
        return ""
    mean, std = pair
    return f"{mean * 100:.2f}±{std * 100:.2f}"


def main():
    parser = argparse.ArgumentParser(description="GT 变体批量 valid-bond 统一指标")
    parser.add_argument("--output_dir", type=str, default="result/valid_bond_metrics_gt_all")
    parser.add_argument("--folds", type=str, nargs="+", default=DEFAULT_FOLDS)
    parser.add_argument("--only", type=str, nargs="+", default=None,
                        help="只处理指定名字的变体（注册表名或 --extra 名）")
    parser.add_argument("--extra", type=str, nargs="+", default=[],
                        help="追加变体 name=cv_root（glob 可用），如 wo_edge_features=checkpoints/...")
    parser.add_argument("--skip_aggregate", action="store_true",
                        help="不补评估（pred 已齐全时直接算指标）")
    parser.add_argument("--force_aggregate", action="store_true",
                        help="忽略已有 pred 强制重跑评估（慎用）")
    args = parser.parse_args()

    setup_logging(args.output_dir)
    logger.info("=" * 70)
    logger.info("GT valid-bond batch | output=" + args.output_dir)
    logger.info("=" * 70)

    runs: List[Tuple[str, Optional[str]]] = list(DEFAULT_RUNS)
    for spec in args.extra:
        if "=" not in spec:
            parser.error(f"--extra 格式应为 name=cv_root: {spec}")
        name, path = spec.split("=", 1)
        runs.append((name, path))
    if args.only:
        runs = [r for r in runs if r[0] in set(args.only)]

    # ---- 阶段 1：逐变体准备 pred.csv ----
    ready: List[Tuple[str, str]] = []  # (name, cv_root)
    summary_stats: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for name, pattern in runs:
        logger.info("\n" + "=" * 60)
        if not pattern:
            logger.info(f"[{name}] 跳过：占位路径未填（DEFAULT_RUNS 中补上或用 --extra）")
            continue
        logger.info(f"[{name}] cv_root pattern = {pattern}")
        cv_root = resolve_cv_root(pattern)
        if not cv_root:
            logger.error(f"[{name}] 跳过：glob 无匹配目录")
            continue
        logger.info(f"[{name}] cv_root = {cv_root}")

        tag = detect_tag(cv_root)
        if not tag:
            logger.error(f"[{name}] 跳过：fold_*/checkpoints/*/ 下未找到 best_model.pt")
            continue
        logger.info(f"[{name}] tag = {tag}")

        if all_pred_csv_exist(cv_root, args.folds):
            if args.force_aggregate:
                logger.info(f"[{name}] pred 五折齐全，但 --force_aggregate → 重跑评估")
                if not run_aggregate(cv_root, tag, args.folds):
                    continue
            else:
                logger.info(f"[{name}] pred 五折齐全，跳过评估")
        else:
            if args.skip_aggregate:
                logger.error(f"[{name}] 跳过：pred 不全且指定 --skip_aggregate")
                continue
            if not run_aggregate(cv_root, tag, args.folds):
                continue

        ready.append((name, cv_root))
        summary_stats[name] = load_summary_mean_std(cv_root)

    if not ready:
        logger.error("没有就绪的变体，退出。")
        return

    # ---- 阶段 2：统一 valid-bond 指标（一次调用，全部同代码路径） ----
    names = [n for n, _ in ready]
    roots = [r for _, r in ready]
    cmd = [sys.executable, VALID_BOND_SCRIPT,
           "--models", *names,
           "--cv_roots", *roots,
           "--types", *["gt"] * len(names),
           "--folds", *args.folds,
           "--output_dir", args.output_dir]
    logger.info("\n" + "=" * 60)
    logger.info("VALID-BOND CMD: " + " ".join(cmd))
    rc = subprocess_run(cmd)
    if rc != 0:
        logger.error(f"valid_bond_metrics 失败 (rc={rc})，宽表不生成。")
        return

    # ---- 阶段 3：宽表（subset/ex 来自训练汇总 + valid-bond 五项） ----
    vb_csv = os.path.join(args.output_dir, "valid_bond_metrics.csv")
    vb = pd.read_csv(vb_csv)
    bond_lookup: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for _, row in vb.iterrows():
        bond_lookup.setdefault(str(row["model"]), {})[str(row["metric"])] = (
            float(row["mean"]), float(row["std"]))

    wide_rows = []
    for name, _cv_root in ready:
        row = {"model": name}
        for key in SUMMARY_KEYS:
            row[key] = fmt_pct(summary_stats.get(name, {}).get(key))
        for key in BOND_KEYS:
            row[key] = fmt_pct(bond_lookup.get(name, {}).get(key))
        wide_rows.append(row)
    wide_df = pd.DataFrame(wide_rows)[["model"] + WIDE_COLUMNS]
    wide_csv = os.path.join(args.output_dir, "valid_bond_wide.csv")
    wide_df.to_csv(wide_csv, index=False)
    logger.info("\n" + "=" * 70)
    logger.info(f"宽表已保存: {wide_csv}")
    logger.info("=" * 70)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        logger.info("\n" + wide_df.to_string(index=False))
    logger.info("Done.")


if __name__ == "__main__":
    main()
