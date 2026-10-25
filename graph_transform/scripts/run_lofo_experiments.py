#!/usr/bin/env python3
"""R-03 leave-one-feature-out (LOFO) 批量运行器（DBond-GT）。

5 个 setting（互不影响，可任意并行）：每次从 full model 只屏蔽一个特征（5 缺 1），
全部以同一 full 参照（主模型 20260421_181316base）做差，回答"该特征的必要性"
（与 Table 9 single-only 的充分性互补）：

    lofo_no_charge     state=[F,T,T] env=[T,T]
    lofo_no_mass       state=[T,F,T] env=[T,T]   (pep_mass)
    lofo_no_intensity  state=[T,T,F] env=[T,T]
    lofo_no_nce        state=[T,T,T] env=[F,T]
    lofo_no_scan       state=[T,T,T] env=[T,F]   (scan_num)

⚠️ --source 必须是 full/obs 主模型的 fold config 快照（tag 为空/baseline、无消融开关），
   即 checkpoints/graph_transform/5fold/20260421_181316base/fold_9075/config.yaml。
   不要用 20260422_232825（那是 Sequence Graph 消融，use_sequence_graph=true）。
   脚本会校验 source 无激活开关，否则拒绝生成。

配置派生：从 source 逐字段克隆，仅改 ablation 段（单 setting 开关 + tag）与输出目录
（checkpoints/graph_transform/lofo/<setting>/ 独立隔离），保证与 full 的唯一差异 =
被屏蔽的那一个特征。派生文件写出到 graph_transform/config/lofo_derived/。

断点续跑：--resume_from auto 交给 train_5fold.py（在 setting 专属 ckpt_base/5fold/ 下
自动找未完成 cv_root，已有 best_model.pt + latest_test_metric.csv 的 fold 跳过）。
本脚本额外做完成判定（5fold_summary.csv 存在且 5 折指标齐全 → 跳过该 setting）。

用法（云端, ~/graphtrans/DBond 下）：
  python graph_transform/scripts/run_lofo_experiments.py \
      --source checkpoints/graph_transform/5fold/20260421_181316base/fold_9075/config.yaml

  # 多卡并行：卡0 与 卡1 各跑部分 setting
  python graph_transform/scripts/run_lofo_experiments.py --source <...> -e lofo_no_charge lofo_no_mass --gpu 0
  python graph_transform/scripts/run_lofo_experiments.py --source <...> -e lofo_no_intensity lofo_no_nce lofo_no_scan --gpu 1

  # 中断后恢复：原命令重跑即可
  # 调试单折 / 强制重跑：
  python graph_transform/scripts/run_lofo_experiments.py --source <...> -e lofo_no_charge --folds 1222
  python graph_transform/scripts/run_lofo_experiments.py --source <...> -e lofo_no_charge --force_new

日志：logs/lofo/<setting>_<时间戳>.log（同时写终端）。
"""

from __future__ import annotations

import argparse
import copy
import datetime
import os
import subprocess
import sys

import yaml

# 5 个 LOFO setting：名称 -> (开关名, 屏蔽说明)。mask 映射在 apply_ablation_config 里。
SETTINGS = {
    "lofo_no_charge": ("lofo_no_charge", "state=[F,T,T] env=[T,T]"),
    "lofo_no_mass": ("lofo_no_mass", "state=[T,F,T] env=[T,T]"),
    "lofo_no_intensity": ("lofo_no_intensity", "state=[T,T,F] env=[T,T]"),
    "lofo_no_nce": ("lofo_no_nce", "state=[T,T,T] env=[F,T]"),
    "lofo_no_scan": ("lofo_no_scan", "state=[T,T,T] env=[T,F]"),
}
SETTING_ORDER = list(SETTINGS.keys())

# 与 apply_ablation_config 的互斥开关表保持一致（生成 ablation 段时全部置 false）
ALL_EXCLUSIVE_FLAGS = [
    'use_sequence_graph', 'use_hybrid_graph', 'disable_global_node',
    'gcn_only', 'gat_only', 'no_message_passing', 'no_edge_attr', 'no_state_env',
    'baseline_no_state_env', 'state_charge_only', 'state_mass_intensity_only',
    'env_nce_only', 'env_scan_num_only', 'state_mass_only', 'state_intensity_only',
    'env_rt_only', 'pre_synthesis',
    'lofo_no_charge', 'lofo_no_mass', 'lofo_no_intensity', 'lofo_no_nce', 'lofo_no_scan',
]

EXPECTED_FOLDS = 5
LOG_DIR = "logs/lofo"


def beijing_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=8)


def ckpt_base_for(setting: str) -> str:
    """每个 setting 独立的 ckpt_base（train_5fold 在其下建 5fold/<时间戳>/，
    resume auto 也只扫描该 setting 自己的目录，不与其他 setting 混淆）。"""
    return os.path.join("checkpoints", "graph_transform", "lofo", setting)


def derive_configs(source_path: str, out_dir: str) -> dict:
    """从 full 模型 fold config 快照派生 5 个 LOFO 配置，返回 {setting: 配置路径}。"""
    with open(source_path, "r", encoding="utf-8") as f:
        source = yaml.safe_load(f)

    ablation = source.get("ablation", {}) or {}
    active = [flag for flag in ALL_EXCLUSIVE_FLAGS if ablation.get(flag, False)]
    explicit_tag = ablation.get("tag")
    if active:
        sys.exit(
            f"[lofo] source 不是 full/baseline 快照（激活了消融开关: {active}）。\n"
            f"请使用 20260421_181316base/fold_*/config.yaml（主模型 Distance/full）。"
        )
    if explicit_tag and explicit_tag not in ("baseline", "null", "None", ""):
        sys.exit(
            f"[lofo] source 的 ablation.tag={explicit_tag!r} 不是 baseline。"
            f"请使用 20260421_181316base/fold_*/config.yaml。"
        )

    os.makedirs(out_dir, exist_ok=True)
    derived_paths = {}
    for setting in SETTING_ORDER:
        flag, note = SETTINGS[setting]
        cfg = copy.deepcopy(source)
        new_ablation = {f: False for f in ALL_EXCLUSIVE_FLAGS}
        new_ablation.update({
            "tag": setting,
            "base_experiment_name": None,
            flag: True,
            "rebuild_cache": True,  # edge_attr 数值随 mask 变化，必须重建缓存
        })
        cfg["ablation"] = new_ablation
        # 输出目录按 setting 隔离（train_5fold 会在此 base 下建 5fold/<时间戳>/）
        cfg.setdefault("training", {})["checkpoint_dir"] = ckpt_base_for(setting)
        cfg.setdefault("evaluation", {})["output_pred_dir"] = os.path.join("result", "pred", "graph_transform", "lofo", setting)
        cfg.setdefault("evaluation", {})["output_metric_dir"] = os.path.join("result", "metric", "graph_transform", "lofo", setting)
        cfg.setdefault("logging", {})["log_dir"] = os.path.join("logs", "graph_transform", "lofo", setting)
        cfg.setdefault("logging", {})["tensorboard_log_dir"] = os.path.join("tensorboard", "graph_transform", "lofo", setting)
        cfg.setdefault("experiment", {})["name"] = f"graph_transform_{setting}"
        cfg["experiment"]["description"] = f"LOFO (R-03): derived from {source_path}; mask {note}"

        path = os.path.join(out_dir, f"{setting}.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        derived_paths[setting] = path
        print(f"[lofo] 派生配置: {path}  ({note})")
    return derived_paths


def count_done_folds(cv_root: str, setting: str) -> int:
    """该 cv_root 下已有指标的 fold 数（metrics/<tag>/latest_test_metric.csv，
    tag 经 apply_ablation_config 追加到 fold 目录内）。"""
    n = 0
    if not os.path.isdir(cv_root):
        return 0
    for name in os.listdir(cv_root):
        if not name.startswith("fold_"):
            continue
        metric = os.path.join(cv_root, name, "metrics", setting, "latest_test_metric.csv")
        if os.path.exists(metric):
            n += 1
    return n


def find_cv_state(setting: str) -> tuple:
    """扫描该 setting 的 ckpt_base/5fold/，返回 (complete, latest_cv_root, done_folds)。"""
    fivefold_base = os.path.join(ckpt_base_for(setting), "5fold")
    if not os.path.isdir(fivefold_base):
        return False, None, 0
    cv_roots = [
        os.path.join(fivefold_base, d)
        for d in os.listdir(fivefold_base)
        if os.path.isdir(os.path.join(fivefold_base, d))
    ]
    if not cv_roots:
        return False, None, 0
    latest = max(cv_roots, key=os.path.getmtime)
    done = count_done_folds(latest, setting)
    complete = os.path.exists(os.path.join(latest, "5fold_summary.csv")) and done >= EXPECTED_FOLDS
    return complete, latest, done


def run_setting(setting: str, config_path: str, args: argparse.Namespace, env: dict) -> str:
    cmd = [
        sys.executable,
        os.path.join("graph_transform", "scripts", "train_5fold.py"),
        "--config", config_path,
        "--fold_data_dir", args.fold_data_dir,
    ]
    if not args.force_new:
        complete, latest, done = find_cv_state(setting)
        if complete and not args.folds:
            print(f"[lofo] {setting}: 已完成({latest}), 跳过")
            return "SKIPPED_DONE"
        if latest is not None:
            # resume auto: train_5fold 在本 setting 的 ckpt_base/5fold/ 下自动找未完成 cv_root
            print(f"[lofo] {setting}: 续跑检查(最新 {latest}, 已完成 fold {done}/{EXPECTED_FOLDS})")
        cmd.extend(["--resume_from", "auto"])
    if args.folds:
        cmd.extend(["--folds"] + list(args.folds))
    if args.force_new:
        cmd.append("--force_new")
        print(f"[lofo] {setting}: force_new 强制重跑(新目录)")

    print(f"[lofo] command: {' '.join(cmd)}")
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{setting}_{beijing_now().strftime('%Y%m%d_%H%M%S')}.log")
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    log_file.write(f"# command: {' '.join(cmd)}\n")
    log_file.write(f"# CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '(unset)')}\n\n")
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, text=True, encoding="utf-8", errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
        returncode = proc.wait()
    finally:
        log_file.close()
    status = "SUCCEEDED" if returncode == 0 else f"FAILED(exit={returncode})"
    print(f"[lofo] {setting}: {status}  日志: {log_path}")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="R-03 LOFO 批量运行器(5 缺 1 消融, 配置派生 + 断点续跑 + 多卡并行)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", required=True,
                        help="full/obs 主模型 fold config 快照(20260421_181316base/fold_*/config.yaml)")
    parser.add_argument("--out_config_dir", default="graph_transform/config/lofo_derived",
                        help="派生配置输出目录")
    parser.add_argument("-e", "--experiments", nargs="+", choices=SETTING_ORDER, default=None,
                        help=f"要跑的 setting 子集(默认全部: {' '.join(SETTING_ORDER)})")
    parser.add_argument("--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES(如 0 / 1 / 0,1)")
    parser.add_argument("--fold_data_dir", default="dataset/5fold", help="5fold 数据目录")
    parser.add_argument("--folds", nargs="+", default=None, help="子集 fold id(调试用)")
    parser.add_argument("--force_new", action="store_true", help="忽略旧结果强制重跑(新目录)")
    parser.add_argument("--refresh_configs", action="store_true",
                        help="强制重新生成派生配置(默认已存在则跳过生成)")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    os.chdir(repo_root)
    if not os.path.exists(args.source):
        sys.exit(f"[lofo] source 不存在: {args.source}")

    selected = args.experiments if args.experiments else SETTING_ORDER

    derived_dir = args.out_config_dir
    marker = os.path.join(derived_dir, ".source")
    need_gen = args.refresh_configs or not all(
        os.path.exists(os.path.join(derived_dir, f"{s}.yaml")) for s in SETTING_ORDER
    )
    if need_gen or not (os.path.exists(marker) and open(marker, encoding="utf-8").read().strip() == os.path.abspath(args.source)):
        print(f"[lofo] 从 source 派生 5 个 LOFO 配置: {args.source}")
        derived_paths = derive_configs(args.source, derived_dir)
        with open(marker, "w", encoding="utf-8") as f:
            f.write(os.path.abspath(args.source))
    else:
        derived_paths = {s: os.path.join(derived_dir, f"{s}.yaml") for s in SETTING_ORDER}
        print(f"[lofo] 派生配置已存在({derived_dir}, source 匹配), 跳过生成")

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"[lofo] CUDA_VISIBLE_DEVICES = {args.gpu}")
    if not os.path.isdir(args.fold_data_dir):
        sys.exit(f"[lofo] 数据目录不存在: {args.fold_data_dir}")

    print(f"[lofo] 计划执行({len(selected)}): {', '.join(selected)}")
    results = {}
    for setting in selected:
        print("\n" + "=" * 70)
        print(f"[lofo] ===== {setting} ({SETTINGS[setting][1]}) =====")
        print("=" * 70)
        results[setting] = run_setting(setting, derived_paths[setting], args, env)

    print("\n[lofo] ===== 汇总 =====")
    failed = 0
    for setting, status in results.items():
        print(f"  {setting:<20} {status}")
        if status.startswith("FAILED"):
            failed += 1
    if failed:
        sys.exit(f"[lofo] {failed} 个 setting 失败; 重跑本命令即可断点续跑。")
    print("[lofo] 全部完成。")


if __name__ == "__main__":
    main()
