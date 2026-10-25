#!/usr/bin/env python3
"""基线多种子集成（同折同划分、seed 概率平均）→ 统一 q 口径评测 → GT / GT-Ens 对照。

背景：论文 GT 行有一个 5-seed 集成版本（result/metric/ensemble_seed5_5fold，
同折同划分概率平均，与 ensemble_inference.py 同一做法；跨折混合属协议泄漏，禁止）。
"GT 的领先是否只是集成带来的、基线也集成后差距会不会拉大"需要同规格的基线集成：
把每个基线的同一折、5 个不同 seed 的 run 逐键概率平均后，走 GT 的同一指标实现评测。

流程（每个基线模型）：
  1. 自动发现 result/cv/<model>/ 下所有 5fold run（按 5fold_metrics.csv 中 fold_1222
     行的 seed 识别 base_seed；seed 42 即正式基线 run，43-46 为本批新训）；
  2. 每折读取 5 个 seed 的 fold_<f>/pred/test.pred.csv，逐键概率平均
     （断言各 seed 行长一致、true 列全等）；
  3. 平均后的键级长表交给 q_metrics_from_pred_csv.py（与 GT 完全同一指标实现，
     概率 → logit 逆变换 → BinaryBondMetrics）；
  4. 5 折 mean±std(ddof=1) 聚合，拼出 GT 单模型 / GT-Ens / 各基线-Ens / 各基线单模型
     的对照表并打印"GT 领先量（单模型 → 集成后）"。

用法（pod 项目根目录，先 dry-run 核对 run 映射）：
  python graph_transform/scripts/ensemble_baseline_preds.py \
      --models dbond_m_pre dbond_af_pre dbond_af_opt_pre --dry_run
  python graph_transform/scripts/ensemble_baseline_preds.py \
      --models dbond_m_pre dbond_af_pre dbond_af_opt_pre \
      --out_dir result/metric/ens_baselines

产物：
  <out_dir>/pred/<model>_fold<f>.ens.pred.csv   概率平均后的键级长表
  <out_dir>/metric/<model>_fold<f>.csv          统一口径逐折指标（同 q_reeval_cond 格式）
  <out_dir>/<model>_ens_5fold_metrics.csv       集成行逐折指标
  <out_dir>/comparison_mean.csv / comparison_std.csv   全部对照行（宽表）
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd

FOLDS = ["1222", "2252", "3514", "6072", "9075"]
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
GT_SINGLE_DIR = "checkpoints/graph_transform/pre_synthesis_theory_folded_film/5fold/20260907_124432"
GT_ENS_DIR = "result/metric/ensemble_seed5_5fold"
Q_REEVAL_DIR = "result/metric/q_reeval_cond"
Q_METRICS_SCRIPT = "graph_transform/scripts/q_metrics_from_pred_csv.py"

# (显示名, 指标键, 缩放倍数)——缩放对齐论文表：比例类 ×100，Top10 富集比不动
DISPLAY = [
    ("Lab-F1", "f1_micro", 100),
    ("BondAcc", "accuracy", 100),
    ("Subset%", "subset_acc", 100),
    ("Ex-Acc", "ex_acc", 100),
    ("Ex-F1h", "ex_f1_harm", 100),  # 调和口径 2PR/(P+R)（§15.16），非 samples 口径 ex_f1
    ("Brier", "q_brier", 100),
    ("MAE", "q_mae", 100),
    ("r_key", "q_spearman", 100),
    ("r_pep", "q_spearman_pep", 100),
    ("r_cond", "q_spearman_pep_cond", 100),
    ("r_seq", "q_spearman_pep_seq", 100),
    ("Top10", "q_top10_enrichment_seq", 1),
]
KEY_GAP = ["Lab-F1", "BondAcc", "Ex-F1h", "Brier", "r_key", "r_seq", "Top10"]
LOWER_BETTER = {"Brier", "MAE"}


def short_name(model: str) -> str:
    return model.replace("dbond_", "").replace("_pre", "")


def read_long(fp: str) -> dict:
    df = pd.read_csv(fp)
    return dict(zip(df["metric"], df["value"]))


def read_wide(fp: str) -> dict:
    df = pd.read_csv(fp)
    if len(df) != 1:
        raise ValueError(f"非单行宽表: {fp}（{len(df)} 行）")
    return {c: df.iloc[0][c] for c in df.columns}


def finish(d: dict) -> dict:
    """补 ex_f1 调和口径（samples 口径 ex_f1 跨模型不可比，见 §15.16）。"""
    try:
        p = float(d.get("ex_precision", np.nan))
        r = float(d.get("ex_recall", np.nan))
        d["ex_f1_harm"] = 2 * p * r / (p + r) if np.isfinite(p) and np.isfinite(r) and (p + r) > 0 else np.nan
    except (TypeError, ValueError):
        d["ex_f1_harm"] = np.nan
    return d


def discover_runs(model: str) -> dict:
    """{base_seed: run_dir}；base_seed = 该 run 中 fold_1222 行的 seed。重名取最新 mtime。"""
    found: dict = {}
    for d in sorted(glob.glob(os.path.join(REPO, "result/cv", model, "*"))):
        mcsv = os.path.join(d, "5fold_metrics.csv")
        if not os.path.isfile(mcsv):
            continue
        try:
            md = pd.read_csv(mcsv)
        except Exception:
            continue
        if "fold_id" not in md.columns or "seed" not in md.columns:
            continue
        row = md.loc[md["fold_id"].astype(str) == "1222"]
        if row.empty:
            continue
        base_seed = int(row.iloc[0]["seed"])
        if base_seed in found:
            old = os.path.getmtime(os.path.join(found[base_seed], "5fold_metrics.csv"))
            if old > os.path.getmtime(mcsv):
                continue
            print(f"[发现] {model}: base_seed {base_seed} 出现重复 run，取最新 {os.path.relpath(d, REPO)}")
        found[base_seed] = d
    return found


def build_ensemble_pred(runs: dict, fold: str, seeds: list, out_path: str) -> str:
    """同折多种子概率平均，写出平均后的键级长表（保留首个成员的列结构）。"""
    paths = []
    for s in seeds:
        if s not in runs:
            raise FileNotFoundError(f"缺少 seed {s} 的 run（训练未完成？）")
        p = os.path.join(runs[s], f"fold_{fold}", "pred", "test.pred.csv")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"缺少成员 pred: {p}")
        paths.append(p)
    df0 = pd.read_csv(paths[0])
    prob = np.array(df0["pred_prob"], dtype=np.float64, copy=True)  # to_numpy 可能只读，须拷贝
    for p in paths[1:]:
        df = pd.read_csv(p)
        if len(df) != len(df0):
            raise ValueError(f"行长不一致（{len(df)} vs {len(df0)}）: {p}")
        if not np.array_equal(df["true"].to_numpy(), df0["true"].to_numpy()):
            raise ValueError(f"true 列不一致，行序对齐失败: {p}")
        prob += df["pred_prob"].to_numpy(dtype=np.float64)
    prob /= len(paths)
    out = df0.copy()
    out["pred_prob"] = prob
    out["pred"] = (prob >= 0.5).astype(np.int64)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path


def eval_pred(pred_csv: str, fold: str, out_csv: str) -> dict:
    cmd = [
        sys.executable, os.path.join(REPO, Q_METRICS_SCRIPT),
        "--pred_csv", pred_csv,
        "--test_csv", os.path.join(REPO, f"dataset/5fold_soft/{fold}.test.fbr.multi.csv"),
        "--out_csv", out_csv,
    ]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-2000:] + "\n" + r.stderr[-2000:] + "\n")
        raise RuntimeError(f"q_metrics_from_pred_csv 失败: {pred_csv}")
    return finish(read_long(out_csv))


def aggregate(fold_dicts: list) -> dict:
    """逐折 dict 列表 → {显示名: (mean, std_ddof1)}。"""
    agg = {}
    for name, key, scale in DISPLAY:
        vals = []
        for d in fold_dicts:
            try:
                v = float(d.get(key, np.nan))
            except (TypeError, ValueError):
                v = np.nan
            if np.isfinite(v):
                vals.append(v * scale)
        if vals:
            a = np.asarray(vals, dtype=float)
            agg[name] = (float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else float("nan"))
        else:
            agg[name] = (float("nan"), float("nan"))
    return agg


def load_gt_single() -> list:
    files = sorted(glob.glob(os.path.join(REPO, GT_SINGLE_DIR, "5fold_metrics.fold_*.csv")))
    if not files:
        print(f"[警告] 未找到 GT 单模型逐折 csv：{GT_SINGLE_DIR}/5fold_metrics.fold_*.csv")
    return [finish(read_wide(fp)) for fp in files]


def load_gt_ens() -> list:
    out = []
    for f in FOLDS:
        fp = os.path.join(REPO, GT_ENS_DIR, f"{f}.csv")
        if not os.path.isfile(fp):
            print(f"[警告] 缺少 GT-Ens 逐折文件 {fp}")
            return []
        out.append(finish(read_long(fp)))
    return out


def load_baseline_single(model: str, runs: dict) -> list:
    """基线单模型行：realized/ex/subset 取 seed-42 run 的 5fold_metrics.csv 行，
    bondacc/q_* 取 result/metric/q_reeval_cond/<model>_fold<f>.csv（冻结表同一来源）。"""
    base = runs.get(42)
    if base is None:
        print(f"[警告] {model} 无 seed-42 run，单模型行跳过")
        return []
    df = pd.read_csv(os.path.join(base, "5fold_metrics.csv"))
    out = []
    for f in FOLDS:
        row = df.loc[df["fold_id"].astype(str) == f]
        if row.empty:
            continue
        d = {c: row.iloc[0][c] for c in df.columns}
        qf = os.path.join(REPO, Q_REEVAL_DIR, f"{model}_fold{f}.csv")
        if os.path.isfile(qf):
            d.update(read_long(qf))
        out.append(finish(d))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="基线 5-seed 集成评测 + GT 对照表")
    ap.add_argument("--models", nargs="+", default=["dbond_m_pre", "dbond_af_pre", "dbond_af_opt_pre"])
    ap.add_argument("--seeds", type=str, default="42,43,44,45,46")
    ap.add_argument("--out_dir", type=str, default="result/metric/ens_baselines")
    ap.add_argument("--skip_eval", action="store_true", help="跳过 pred 平均与评测，只重聚合已有逐折 csv")
    ap.add_argument("--dry_run", action="store_true", help="只发现 run 映射，不计算")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ---- 发现各模型 run ----
    run_maps = {}
    ok = True
    for model in args.models:
        rm = discover_runs(model)
        run_maps[model] = rm
        missing = [s for s in seeds if s not in rm]
        print(f"[发现] {model}: found seeds {sorted(rm)}")
        for s in sorted(rm):
            print(f"        seed {s:>2}: {os.path.relpath(rm[s], REPO)}")
        if missing:
            ok = False
            print(f"        !! 缺少 seed {missing} 的 run（训练未完成？）")
    if args.dry_run:
        return
    if not ok:
        raise SystemExit("有模型缺少成员 run，先完成训练再聚合（或改 --seeds）")

    rows: dict = {}
    rows["GT-single"] = aggregate(load_gt_single())
    rows["GT-Ens"] = aggregate(load_gt_ens())

    for model in args.models:
        short = short_name(model)
        rm = run_maps[model]
        fold_dicts = []
        for f in FOLDS:
            met_out = os.path.join(out_dir, "metric", f"{model}_fold{f}.csv")
            pred_out = os.path.join(out_dir, "pred", f"{model}_fold{f}.ens.pred.csv")
            if args.skip_eval and os.path.isfile(met_out):
                d = finish(read_long(met_out))
            else:
                build_ensemble_pred(rm, f, seeds, pred_out)
                d = eval_pred(pred_out, f, met_out)
            fold_dicts.append(d)
            print(f"[eval] {short} fold{f}: f1_micro={d.get('f1_micro', float('nan')):.4f} "
                  f"bondacc={d.get('accuracy', float('nan')):.4f} "
                  f"q_brier={d.get('q_brier', float('nan')):.4f} "
                  f"q_seq={d.get('q_spearman_pep_seq', float('nan')):.4f} "
                  f"top10={d.get('q_top10_enrichment_seq', float('nan')):.4f}")
        pd.DataFrame(fold_dicts).to_csv(os.path.join(out_dir, f"{model}_ens_5fold_metrics.csv"), index=False)
        rows[f"{short}-Ens"] = aggregate(fold_dicts)
        rows[f"{short}-single"] = aggregate(load_baseline_single(model, rm))

    order = ["GT-single", "GT-Ens"]
    for model in args.models:
        s = short_name(model)
        order += [f"{s}-Ens", f"{s}-single"]
    order = [o for o in order if o in rows]

    # ---- 打印对照表 ----
    print("\n== 5 折 mean±std（比例/Brier/MAE 已 ×100；Top10 为富集比）==")
    hdr = "行".ljust(13) + "".join(n.rjust(11) for n, _, _ in DISPLAY)
    print(hdr)
    for lab in order:
        cells = []
        for name, _, _ in DISPLAY:
            m, s = rows[lab].get(name, (np.nan, np.nan))
            cells.append(f"{m:.2f}±{s:.2f}" if np.isfinite(m) else "—")
        print(lab.ljust(13) + "".join(c.rjust(11) for c in cells))

    # ---- GT 领先量：单模型 vs 集成后 ----
    print("\n== GT 领先量（GT − 基线，正=GT 更好；Brier/MAE 已取反）==")
    for model in args.models:
        s = short_name(model)
        if f"{s}-single" not in rows or f"{s}-Ens" not in rows:
            continue
        for name in KEY_GAP:
            direction = -1.0 if name in LOWER_BETTER else 1.0
            gs, bs = rows["GT-single"][name][0], rows[f"{s}-single"][name][0]
            ge, be = rows["GT-Ens"][name][0], rows[f"{s}-Ens"][name][0]
            if not all(np.isfinite([gs, bs, ge, be])):
                continue
            print(f"{s:7s} {name:8s} 单模型 {direction * (gs - bs):+7.2f}  →  集成后 {direction * (ge - be):+7.2f}")

    # ---- 落盘 ----
    mean_df = pd.DataFrame({lab: {n: rows[lab][n][0] for n, _, _ in DISPLAY} for lab in order})
    std_df = pd.DataFrame({lab: {n: rows[lab][n][1] for n, _, _ in DISPLAY} for lab in order})
    mean_df.to_csv(os.path.join(out_dir, "comparison_mean.csv"))
    std_df.to_csv(os.path.join(out_dir, "comparison_std.csv"))
    print(f"\n[saved] {os.path.join(args.out_dir, 'comparison_mean.csv')} / comparison_std.csv")


if __name__ == "__main__":
    main()