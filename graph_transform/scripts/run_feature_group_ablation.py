#!/usr/bin/env python3
"""Feature-group progressive addition 消融实验调度脚本。

实验设计：从裸模型（无 state 无 env）出发，每次只加一组特征，明确 4 类实验条件
（charge / mass+intensity / NCE / scan_num）对键断裂预测的贡献排序。
共 6 个 setting（baseline_none + 4 个 +X + full），每个跑 5 折交叉验证。

工作流程：
  1. 读 base config（ablation_feature_group.yaml，3GCN+2GAT + scan_num）
  2. 对每个 setting：
     a. 深拷贝 base config
     b. 重置 ablation 段，只开该 setting 对应的开关（或全 false = full）
     c. 把 checkpoint_dir / 输出路径改为带 setting tag 的固定路径（便于对比脚本定位）
     d. 写临时 config 到磁盘
     e. subprocess 调 train_5fold.py 跑 5 折
  3. 汇总各 setting 的 5fold_summary.csv 路径，供 compare_feature_groups.py 使用

用法：
  python graph_transform/scripts/run_feature_group_ablation.py
  python graph_transform/scripts/run_feature_group_ablation.py --settings baseline_no_state_env state_charge_only
  python graph_transform/scripts/run_feature_group_ablation.py --skip_existing  # 跳过已有结果的 setting

输出目录约定（便于 compare_feature_groups.py 定位）：
  {checkpoint_dir}/{setting_tag}/5fold/{timestamp}/5fold_summary.csv
  其中 checkpoint_dir 来自 base config（默认 checkpoints/graph_transform/feature_group_ablation）
"""

import argparse
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime

import yaml

# 6 个 setting 的定义：(setting_key, ablation_switch_or_None, 描述)
# setting_key 用于命名输出子目录；ablation_switch 是 apply_ablation_config 里的开关名，
# None 表示全特征（full，ablation 段全 false）。
SETTINGS = [
    ("baseline_none", "baseline_no_state_env", "裸模型：无 state 无 env（起点）"),
    ("charge_only", "state_charge_only", "+charge"),
    ("mass_intensity_only", "state_mass_intensity_only", "+mass+intensity"),
    ("nce_only", "env_nce_only", "+NCE"),
    ("scan_num_only", "env_scan_num_only", "+scan_num"),
    ("full", None, "全特征（终点参照）"),
]

# apply_ablation_config 里的全部互斥开关（重置时全部置 false）
ALL_ABLATION_SWITCHES = [
    "use_sequence_graph", "use_hybrid_graph", "disable_global_node",
    "gcn_only", "gat_only", "no_message_passing", "no_edge_attr", "no_state_env",
    "baseline_no_state_env", "state_charge_only", "state_mass_intensity_only",
    "env_nce_only", "env_scan_num_only",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_5FOLD = os.path.join(SCRIPT_DIR, "train_5fold.py")
DEFAULT_CONFIG = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "config", "ablation_feature_group.yaml"))


def make_setting_config(base_config: dict, setting_key: str, ablation_switch, setting_root: str) -> dict:
    """为单个 setting 派生 config：重置 ablation 段，改输出路径到 setting_root 下。"""
    cfg = deepcopy(base_config)

    # 重置 ablation 段：全部开关 false，再开目标（保证单变量）
    ablation_cfg = cfg.setdefault("ablation", {})
    ablation_cfg["tag"] = None  # null = 自动推导 tag（由 build_ablation_tag 生成）
    ablation_cfg["base_experiment_name"] = None
    for sw in ALL_ABLATION_SWITCHES:
        ablation_cfg[sw] = False
    ablation_cfg["rebuild_cache"] = True  # edge_attr 含 state/env，mask 不同必须重建
    if ablation_switch is not None:
        ablation_cfg[ablation_switch] = True

    # 改输出路径：各 setting 隔离到独立子目录，便于 compare_feature_groups.py 定位。
    # train_5fold.py 会在 checkpoint_dir 下建 5fold/<timestamp>/fold_<id>/，
    # apply_ablation_config 会再附加 /<ablation_tag>，所以这里设 checkpoint_dir 为 setting_root。
    cfg.setdefault("training", {})["checkpoint_dir"] = setting_root
    cfg.setdefault("evaluation", {})["output_metric_dir"] = os.path.join(
        "result/metric/graph_transform/feature_group_ablation", setting_key
    )
    cfg.setdefault("evaluation", {})["output_pred_dir"] = os.path.join(
        "result/pred/graph_transform/feature_group_ablation", setting_key
    )
    cfg.setdefault("logging", {})["log_dir"] = os.path.join(
        "logs/graph_transform/feature_group_ablation", setting_key
    )
    cfg.setdefault("logging", {})["tensorboard_log_dir"] = os.path.join(
        "tensorboard/graph_transform/feature_group_ablation", setting_key
    )
    return cfg


def find_latest_summary(setting_root: str):
    """在 setting_root 下找最新的 5fold_summary.csv（train_5fold.py 输出）。

    路径模式：{setting_root}/5fold/{timestamp}/5fold_summary.csv
    （apply_ablation_config 可能在 setting_root 后再附加 /<tag>，所以递归搜索）
    """
    if not os.path.isdir(setting_root):
        return None
    summaries = []
    for dirpath, _, filenames in os.walk(setting_root):
        if "5fold_summary.csv" in filenames:
            summaries.append(os.path.join(dirpath, "5fold_summary.csv"))
    if not summaries:
        return None
    return max(summaries, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description="Run feature-group progressive addition ablation (6 settings × 5 folds)")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Base config yaml")
    parser.add_argument(
        "--settings", nargs="+",
        choices=[s[0] for s in SETTINGS],
        help="Optional subset of settings to run (default: all 6)",
    )
    parser.add_argument("--skip_existing", action="store_true", help="Skip settings that already have 5fold_summary.csv")
    parser.add_argument("--fold_data_dir", default="dataset/5fold", help="5-fold csv directory")
    parser.add_argument("--epochs", type=int, help="Override epochs per fold")
    parser.add_argument("--batch_size", type=int, help="Override batch_size per fold")
    parser.add_argument("--learning_rate", type=float, help="Override learning_rate per fold")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], help="Override device")
    parser.add_argument("--seed", type=int, help="Override base seed (each fold uses base_seed + fold_index)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    selected = args.settings or [s[0] for s in SETTINGS]
    settings_map = {s[0]: (s[1], s[2]) for s in SETTINGS}

    ckpt_base = base_config.get("training", {}).get("checkpoint_dir", "checkpoints/graph_transform/feature_group_ablation")
    overall_start = time.perf_counter()

    print("=" * 60)
    print(" Feature-group progressive addition ablation")
    print(f" Config: {args.config}")
    print(f" Settings: {selected}")
    print(f" Checkpoint base: {ckpt_base}")
    print(f" Fold data dir: {args.fold_data_dir}")
    print("=" * 60)

    completed = []
    skipped = []
    failed = []

    for setting_key in selected:
        ablation_switch, desc = settings_map[setting_key]
        setting_root = os.path.join(ckpt_base, setting_key)

        print("\n" + "-" * 60)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Setting: {setting_key}")
        print(f"  描述: {desc}")
        print(f"  开关: {ablation_switch or '(全 false = full)'}")
        print(f"  输出根目录: {setting_root}")
        print("-" * 60)

        if args.skip_existing:
            existing = find_latest_summary(setting_root)
            if existing:
                print(f"  [skip] 已有结果: {existing}")
                skipped.append((setting_key, existing))
                completed.append((setting_key, existing))
                continue

        # 派生 config 并写到 setting_root/config.yaml
        setting_config = make_setting_config(base_config, setting_key, ablation_switch, setting_root)
        os.makedirs(setting_root, exist_ok=True)
        setting_config_path = os.path.join(setting_root, "config.yaml")
        with open(setting_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(setting_config, f, sort_keys=False, allow_unicode=True)
        print(f"  [config] 写入 {setting_config_path}")

        # 跑 5 折
        cmd = [sys.executable, TRAIN_5FOLD, "--config", setting_config_path, "--fold_data_dir", args.fold_data_dir]
        if args.epochs is not None:
            cmd.extend(["--epochs", str(args.epochs)])
        if args.batch_size is not None:
            cmd.extend(["--batch_size", str(args.batch_size)])
        if args.learning_rate is not None:
            cmd.extend(["--learning_rate", str(args.learning_rate)])
        if args.device is not None:
            cmd.extend(["--device", args.device])
        if args.seed is not None:
            cmd.extend(["--seed", str(args.seed)])

        print(f"  [run] {' '.join(cmd)}")
        setting_start = time.perf_counter()
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  [FAIL] {setting_key} 训练失败: {e}")
            failed.append(setting_key)
            continue
        elapsed = time.perf_counter() - setting_start

        summary = find_latest_summary(setting_root)
        if summary:
            print(f"  [done] {setting_key} 完成 ({elapsed:.0f}s), summary: {summary}")
            completed.append((setting_key, summary))
        else:
            print(f"  [WARN] {setting_key} 完成但未找到 5fold_summary.csv")
            failed.append(setting_key)

    # 汇总
    print("\n" + "=" * 60)
    print(" 全部 setting 完成")
    print("=" * 60)
    print(f" 成功: {len(completed)} / {len(selected)}")
    print(f" 跳过: {len(skipped)}")
    print(f" 失败: {len(failed)}")
    if failed:
        print(f" 失败列表: {failed}")
    print("\n 各 setting 的 5fold_summary.csv 路径：")
    for setting_key, summary in completed:
        print(f"  {setting_key:25s} -> {summary}")
    print(f"\n 总耗时: {time.perf_counter() - overall_start:.0f}s")
    print("\n 下一步：运行 compare_feature_groups.py 汇总对比 + 统计检验：")
    print("  python graph_transform/scripts/compare_feature_groups.py \\\n"
          "    --ckpt_base checkpoints/graph_transform/feature_group_ablation")


if __name__ == "__main__":
    main()
