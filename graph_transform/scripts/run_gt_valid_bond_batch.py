#!/usr/bin/env python3
"""统一口径批量指标生成（GT 变体 + 基线 pre/obs，valid-bond 口径，全模型可比）。

与 valid_bond_metrics.py 的关系：后者是计算引擎（读 pred CSV → 有效键指标），
本脚本是编排器——自动补缺失的 pred 评估（仅 GT 需要）→ 调用后者 → 拼宽表。
不算重复实现。

注册表（DEFAULT_RUNS）三类条目：
  - gt         : GT 管线运行。缺 pred.csv 时自动调 aggregate_r20_5fold.py 补评估
                 （逐折 best_model 推理，已存在折自动跳过，断点续跑安全），tag 自动探测
  - multilabel : 基线 m/af/af_opt（pred/test.pred.csv 训练时已落盘，无需评估）
  - single     : 基线 dbond_s（同上）
路径不存在的条目自动跳过 → 同一脚本在两台机器各跑各的：
  - graphtrans 机：GT 变体行（obs full / gt_pre / LOFO / 消融）
  - dbond-gt-2 机：基线行（4 obs + 4 pre）
跑完把两台的 valid_bond_wide.csv 汇总即得全模型统一口径大表。

待填路径：5 个旧消融（wo_*）与 4 个基线 obs 的 cv_root（None 占位）——
填进 DEFAULT_RUNS 或用 --extra 追加。

用法（各自仓库根目录）：
  python graph_transform/scripts/run_gt_valid_bond_batch.py            # 全部已注册条目
  python graph_transform/scripts/run_gt_valid_bond_batch.py --only gt_pre lofo_no_nce
  python graph_transform/scripts/run_gt_valid_bond_batch.py \
      --extra wo_edge_features=gt=checkpoints/graph_transform/5fold/<ts>_wo_edge \
      --extra dbond_s=single=result/cv/dbond_s/<obs_ts>

输出（默认 result/valid_bond_metrics_gt_all/）：
  valid_bond_wide.csv     宽表（"58.81±0.34" 百分数格式）：
                          subset/ex 五项（各 cv_root 的 5fold_summary.csv，padding 无关）
                          + valid-bond 五项（替换原 label 四列 + MCC）
  valid_bond_metrics.csv / per_model/*.csv   valid_bond_metrics.py 原始输出
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
AGGREGATE_SCRIPT = os.path.join(SCRIPT_DIR, "aggregate_r20_5fold.py")
VALID_BOND_SCRIPT = os.path.join(SCRIPT_DIR, "valid_bond_metrics.py")

DEFAULT_FOLDS = ["1222", "2252", "3514", "6072", "9075"]

# 注册表：(显示名, cv_root 或 glob 模式, 类型 gt/multilabel/single)。None = 占位待填。
# GT 变体（dbond_gt_obs 在 dbond-gt-2 机；其余在 graphtrans 机，各自机器自动跳过对方条目）
GT_RUNS: List[Tuple[str, Optional[str], str]] = [
    ("dbond_gt_obs", "checkpoints/graph_transform/feature_group_ablation/full/5fold/20260725_201242", "gt"),
    ("gt_pre", "checkpoints/graph_transform/pre_synthesis/5fold/*", "gt"),
    ("lofo_no_charge", "checkpoints/graph_transform/lofo/lofo_no_charge/5fold/*", "gt"),
    ("lofo_no_mass", "checkpoints/graph_transform/lofo/lofo_no_mass/5fold/*", "gt"),
    ("lofo_no_intensity", "checkpoints/graph_transform/lofo/lofo_no_intensity/5fold/*", "gt"),
    ("lofo_no_nce", "checkpoints/graph_transform/lofo/lofo_no_nce/5fold/*", "gt"),
    ("lofo_no_scan", "checkpoints/graph_transform/lofo/lofo_no_scan/5fold/*", "gt"),
    ("sequence_graph", "checkpoints/graph_transform/5fold/20260422_232825*", "gt"),
    ("wo_message_passing", None, "gt"),
    ("wo_edge_features", None, "gt"),
    ("wo_state_env", None, "gt"),
    ("wo_global_node", None, "gt"),
    ("gcn_only", None, "gt"),
]
# 基线（dbond-gt-2 机）：obs 四个路径待填；pre 四个时间戳已定（glob 自动匹配）
BASELINE_RUNS: List[Tuple[str, Optional[str], str]] = [
    ("dbond_s", None, "single"),
    ("dbond_m", None, "multilabel"),
    ("dbond_af", None, "multilabel"),
    ("dbond_af_opt", None, "multilabel"),
    ("dbond_s_pre", "result/cv/dbond_s_pre/*", "single"),
    ("dbond_m_pre", "result/cv/dbond_m_pre/*", "multilabel"),
    ("dbond_af_pre", "result/cv/dbond_af_pre/*", "multilabel"),
    ("dbond_af_opt_pre", "result/cv/dbond_af_opt_pre/*", "multilabel"),
]
DEFAULT_RUNS = GT_RUNS + BASELINE_RUNS

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
                  if glob.glob(os.path.join(p, "fold_*"))]
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


def all_gt_pred_csv_exist(cv_root: str, folds: List[str]) -> bool:
    for fold_id in folds:
        path = os.path.join(cv_root, "r20_aggregation", "per_fold", f"fold_{fold_id}", "pred.csv")
        if not os.path.exists(path):
            return False
    return True


def baseline_pred_status(cv_root: str, folds: List[str]) -> Tuple[int, List[str]]:
    """基线 pred 检查（与 valid_bond_metrics 的 find_pred_csv 相同的两种布局）。
    返回 (找到 pred 的折数, 缺失折 id 列表)。"""
    found, missing = 0, []
    for fold_id in folds:
        patterns = [
            os.path.join(cv_root, "*", f"fold_{fold_id}", "pred", "test.pred.csv"),
            os.path.join(cv_root, f"fold_{fold_id}", "pred", "test.pred.csv"),
        ]
        if any(glob.glob(p) for p in patterns):
            found += 1
        else:
            missing.append(fold_id)
    return found, missing


def run_aggregate(cv_root: str, tag: str, folds: List[str]) -> bool:
    """调用 aggregate_r20_5fold.py 补 pred.csv（已存在的折它自己会跳过）。"""
    import subprocess
    cmd = [sys.executable, AGGREGATE_SCRIPT,
           "--cv_root", cv_root, "--tag", tag, "--folds", *folds]
    logger.info("  AGGREGATE CMD: " + " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        logger.error(f"  aggregate 失败 (rc={rc}): {cv_root}")
        return False
    return all_gt_pred_csv_exist(cv_root, folds)


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
    parser = argparse.ArgumentParser(description="GT 变体 + 基线 pre/obs 批量 valid-bond 统一指标")
    parser.add_argument("--output_dir", type=str, default="result/valid_bond_metrics_gt_all")
    parser.add_argument("--fold_data_dir", type=str, default="dataset/5fold",
                        help="5fold 测试数据目录（基线条目需要）")
    parser.add_argument("--folds", type=str, nargs="+", default=DEFAULT_FOLDS)
    parser.add_argument("--only", type=str, nargs="+", default=None,
                        help="只处理指定名字的变体（注册表名或 --extra 名）")
    parser.add_argument("--extra", type=str, nargs="+", default=[],
                        help="追加条目 name=kind=cv_root（gt/multilabel/single），"
                             "或 name=cv_root（默认 gt）")
    parser.add_argument("--skip_aggregate", action="store_true",
                        help="不补评估（pred 已齐全时直接算指标）")
    parser.add_argument("--force_aggregate", action="store_true",
                        help="忽略已有 pred 强制重跑评估（慎用，仅对 gt 条目生效）")
    args = parser.parse_args()

    setup_logging(args.output_dir)
    logger.info("=" * 70)
    logger.info("Valid-bond batch (GT variants + baselines) | output=" + args.output_dir)
    logger.info("=" * 70)

    runs: List[Tuple[str, Optional[str], str]] = list(DEFAULT_RUNS)
    for spec in args.extra:
        parts = spec.split("=")
        if len(parts) == 3:
            name, kind, path = parts
        elif len(parts) == 2:
            name, path = parts
            kind = "gt"
        else:
            parser.error(f"--extra 格式应为 name=kind=cv_root 或 name=cv_root: {spec}")
        if kind not in ("gt", "multilabel", "single"):
            parser.error(f"--extra kind 只能是 gt/multilabel/single: {spec}")
        runs.append((name, path, kind))
    if args.only:
        wanted = set(args.only)
        runs = [r for r in runs if r[0] in wanted]

    # ---- 阶段 1：逐条目准备 pred ----
    ready: List[Tuple[str, str, str]] = []  # (name, cv_root, kind)
    summary_stats: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for name, pattern, kind in runs:
        logger.info("\n" + "=" * 60)
        if not pattern:
            logger.info(f"[{name}] 跳过：占位路径未填（DEFAULT_RUNS 中补上或用 --extra）")
            continue
        logger.info(f"[{name}] kind={kind} | cv_root pattern = {pattern}")
        cv_root = resolve_cv_root(pattern)
        if not cv_root:
            logger.warning(f"[{name}] 跳过：本机 glob 无匹配目录（另一台机器的条目属正常跳过）")
            continue
        logger.info(f"[{name}] cv_root = {cv_root}")

        if kind == "gt":
            tag = detect_tag(cv_root)
            if not tag:
                logger.error(f"[{name}] 跳过：fold_*/checkpoints/*/ 下未找到 best_model.pt")
                continue
            logger.info(f"[{name}] tag = {tag}")
            if all_gt_pred_csv_exist(cv_root, args.folds):
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
        else:
            # 基线：pred 由训练落盘，无需评估；只做存在性预检
            found, missing = baseline_pred_status(cv_root, args.folds)
            if found == 0:
                logger.error(f"[{name}] 跳过：未找到任何 fold 的 pred/test.pred.csv")
                continue
            if missing:
                logger.warning(f"[{name}] pred 缺失折 {missing}（这些折将不计入，其余照常）")

        ready.append((name, cv_root, kind))
        summary_stats[name] = load_summary_mean_std(cv_root)

    if not ready:
        logger.error("没有就绪的条目，退出。")
        return

    # ---- 阶段 2：统一 valid-bond 指标（一次调用，全部同代码路径） ----
    names = [n for n, _, _ in ready]
    roots = [r for _, r, _ in ready]
    kinds = [k for _, _, k in ready]
    cmd = [sys.executable, VALID_BOND_SCRIPT,
           "--models", *names,
           "--cv_roots", *roots,
           "--types", *kinds,
           "--fold_data_dir", args.fold_data_dir,
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
    for name, _cv_root, _kind in ready:
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
