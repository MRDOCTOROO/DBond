#!/usr/bin/env python3
"""扫描 GT 训练运行目录，自动识别消融身份（双重判定：ablation 标志 + model 段实际形状）。

背景：早期消融实验是手动改 model 段布尔值（use_state_features: false 等）跑的，
fold config 的 ablation.tag 可能是过期或手写的（例：tag=gatonly 但 model 段
3GCN+2GAT 全开）。因此本脚本同时给出两个视角：

  flag_identity : ablation 段特征键判定（新机制运行：gt_pre / lofo_* 等，可靠）
  shape         : model 段实际形状描述（旧手改运行的真实证据）
                  gcn{n}/gat{m} + noMP/noEdge*/noState/noEnv/noGlobal + 图策略

身份判定规则（shape → 消融行，与 apply_ablation_config 的落点一致）：
  num_gcn_layers==0 且 num_gat_layers==0          → w/o Message Passing
  num_gat_layers==0 且 num_gcn_layers>0           → GCN Only（机制版为 5/0，手改版 3/0 也算）
  num_gcn_layers==0 且 num_gat_layers>0           → GAT Only（论文表无此行，仅报告）
  use_state_features==False 且 use_env_features==False → w/o State/Env
  use_global_node==False                          → w/o Global Node
  use_edge_features==False（含 edge 子开关关闭）    → w/o Edge Features

用法（任一机器 DBond 仓库根目录；默认 4 类根都扫，不存在的自动跳过）：
  python graph_transform/scripts/scan_ablation_runs.py
  python graph_transform/scripts/scan_ablation_runs.py --roots 'checkpoints/graph_transform/5fold/*'

输出：控制台表（flag 身份、shape、tag、完整度）+ 对匹配 5 个旧消融的目录打印
可复制的 --extra 行（flag 与 shape 不一致或仅 shape 判定时标注"需人工确认"）。
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Optional, Tuple

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_ROOTS = [
    "checkpoints/graph_transform/5fold/*",
    "checkpoints/graph_transform/feature_group_ablation/*/5fold/*",
    "checkpoints/graph_transform/pre_synthesis/5fold/*",
    "checkpoints/graph_transform/lofo/*/5fold/*",
]

# ablation 特征键 → 身份（新机制；与 train_graph_model.apply_ablation_config 一致）
FLAG_TO_IDENTITY = [
    ("pre_synthesis", "gt_pre"),
    ("lofo_no_charge", "lofo_no_charge"),
    ("lofo_no_mass", "lofo_no_mass"),
    ("lofo_no_intensity", "lofo_no_intensity"),
    ("lofo_no_nce", "lofo_no_nce"),
    ("lofo_no_scan", "lofo_no_scan"),
    ("use_sequence_graph", "sequence_graph"),
    ("use_hybrid_graph", "hybrid_graph"),
    ("no_message_passing", "wo_message_passing"),
    ("no_edge_attr", "wo_edge_features"),
    ("no_state_env", "wo_state_env"),
    ("disable_global_node", "wo_global_node"),
    ("gcn_only", "gcn_only"),
    ("gat_only", "gat_only"),
    ("baseline_no_state_env", "table9_baseline_no_state_env"),
    ("state_charge_only", "table9_state_charge_only"),
    ("state_mass_intensity_only", "table9_state_mass_intensity"),
    ("state_mass_only", "table9_state_mass"),
    ("state_intensity_only", "table9_state_intensity"),
    ("env_nce_only", "table9_env_nce"),
    ("env_scan_num_only", "table9_env_scan"),
    ("env_rt_only", "table9_env_rt"),
]

WANTED = {"wo_message_passing", "wo_edge_features", "wo_state_env",
          "wo_global_node", "gcn_only"}


def classify_flags(ablation_cfg: Dict) -> Tuple[str, List[str]]:
    active = [flag for flag, _ in FLAG_TO_IDENTITY if ablation_cfg.get(flag, False)]
    if not active:
        return "full", []
    identity = dict((f, i) for f, i in FLAG_TO_IDENTITY)[active[0]]
    return identity, active


def classify_shape(model_cfg: Dict, data_cfg: Dict) -> Tuple[str, List[str]]:
    """model 段实际形状 → (shape 描述, 推断的旧消融身份列表)。"""
    gcn = int(model_cfg.get("num_gcn_layers", 3))
    gat = int(model_cfg.get("num_gat_layers", 2))
    parts = [f"gcn{gcn}/gat{gat}"]
    candidates: List[str] = []

    if gcn == 0 and gat == 0:
        parts.append("noMP")
        candidates.append("wo_message_passing")
    elif gat == 0 and gcn > 0:
        parts.append("gcnOnly")
        candidates.append("gcn_only")
    elif gcn == 0 and gat > 0:
        parts.append("gatOnly")
        candidates.append("gat_only")

    no_state = model_cfg.get("use_state_features") is False
    no_env = model_cfg.get("use_env_features") is False
    if no_state:
        parts.append("noState")
    if no_env:
        parts.append("noEnv")
    if no_state and no_env:
        candidates.append("wo_state_env")

    if model_cfg.get("use_edge_features") is False:
        parts.append("noEdge")
        candidates.append("wo_edge_features")
    else:
        edge_subs = ["use_raw_edge_attr", "gat_use_edge_bias", "gat_use_edge_gate",
                     "bond_use_edge_repr", "use_edge_type_embedding", "use_distance_embedding"]
        off = [k for k in edge_subs if model_cfg.get(k) is False]
        if len(off) >= 4:
            parts.append(f"noEdgeSub({len(off)})")
            candidates.append("wo_edge_features")

    if model_cfg.get("use_global_node") is False:
        parts.append("noGlobal")
        candidates.append("wo_global_node")

    strategy = str(data_cfg.get("graph_strategy", ""))
    parts.append(strategy or "?")
    return "+".join(parts), candidates


def read_first_config(cv_root: str) -> Optional[Dict]:
    for cfg_path in sorted(glob.glob(os.path.join(cv_root, "fold_*", "config.yaml"))):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            continue
    return None


def count_folds_with_best_model(cv_root: str) -> int:
    return len(glob.glob(os.path.join(cv_root, "fold_*", "checkpoints", "*", "*", "best_model.pt")))


def count_r20_pred(cv_root: str) -> int:
    return len(glob.glob(os.path.join(cv_root, "r20_aggregation", "per_fold", "fold_*", "pred.csv")))


def main():
    parser = argparse.ArgumentParser(description="扫描并识别 GT 消融运行目录（ablation 标志 + model 形状双判定）")
    parser.add_argument("--roots", type=str, nargs="+", default=DEFAULT_ROOTS)
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()

    rows: List[Dict] = []
    seen = set()
    for root_pat in args.roots:
        for cv_root in sorted(glob.glob(root_pat)):
            cv_root = cv_root.rstrip("/\\")
            if not os.path.isdir(cv_root) or cv_root in seen:
                continue
            seen.add(cv_root)
            fold_cfg = read_first_config(cv_root)
            if fold_cfg is None:
                rows.append({"dir": cv_root, "flag_identity": "(无 fold config，失败/中断)",
                             "shape": "", "tag": "", "folds_best_model": 0,
                             "has_summary": False, "r20_pred_folds": 0,
                             "shape_candidates": ""})
                continue
            ablation_cfg = fold_cfg.get("ablation", {}) or {}
            model_cfg = fold_cfg.get("model", {}) or {}
            data_cfg = fold_cfg.get("data", {}) or {}
            flag_identity, _ = classify_flags(ablation_cfg)
            shape, shape_candidates = classify_shape(model_cfg, data_cfg)
            rows.append({
                "dir": cv_root,
                "flag_identity": flag_identity,
                "shape": shape,
                "tag": str(ablation_cfg.get("tag", "")),
                "folds_best_model": count_folds_with_best_model(cv_root),
                "has_summary": os.path.exists(os.path.join(cv_root, "5fold_summary.csv")),
                "r20_pred_folds": count_r20_pred(cv_root),
                "shape_candidates": ",".join(shape_candidates),
            })

    print(f"{'dir':<58} {'flag_identity':<26} {'shape':<30} {'tag':<14} {'bm':>3} {'sum':>4} {'r20':>4}")
    print("-" * 145)
    for r in rows:
        print(f"{r['dir']:<58} {r['flag_identity']:<26} {r['shape']:<30} "
              f"{r['tag']:<14} {r['folds_best_model']:>3} "
              f"{'Y' if r['has_summary'] else '-':>4} {r['r20_pred_folds']:>4}")
    print("-" * 145)
    print("bm = 各折 best_model.pt 数(最多5) | sum = 根目录 5fold_summary.csv | r20 = 已有 pred.csv 折数")
    print("shape 列: gcn{n}/gat{m} + noMP/gcnOnly/gatOnly/noState/noEnv/noEdge/noGlobal + 图策略")

    # 5 个旧消融的候选：flag 判定 或 shape 判定命中，且训练完整（bm>=5）
    print("\n# tab:ablation 需要的 5 个旧消融候选（bm>=5 才列）——仅 shape 判定或与 tag 冲突的需人工确认：")
    suggestion_lines = []
    for r in rows:
        if r["folds_best_model"] < 5:
            continue
        hit_flags = r["flag_identity"] in WANTED
        shape_hits = [c for c in r["shape_candidates"].split(",") if c in WANTED]
        if hit_flags:
            mark = "OK" if not shape_hits or r["flag_identity"] in shape_hits else f"冲突(shape={shape_hits})"
            suggestion_lines.append((mark, f"        {r['flag_identity']}=gt={r['dir']} \\"))
        elif shape_hits:
            for cand in shape_hits:
                suggestion_lines.append(("需确认(仅shape)", f"        {cand}=gt={r['dir']} \\"))
    if suggestion_lines:
        print("    --extra \\")
        for mark, line in suggestion_lines:
            print(f"    # [{mark}]")
            print(line)
    else:
        print("#   （无完整候选，见上表 bm 列 <5 的目录）")

    if args.output_csv:
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)
        print(f"\nCSV 已保存: {args.output_csv}")


if __name__ == "__main__":
    main()
