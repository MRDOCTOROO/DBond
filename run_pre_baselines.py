#!/usr/bin/env python3
"""R-01 基线 pre 变体批量运行器(dbond_s/m/af/af_opt 的 *_pre 四个实验)。

功能:
  1. 顺序执行选定的实验(默认全部 4 个, 按依赖无关顺序);
  2. 中断/打断后重跑同一命令即可恢复:
     - 已完成(5 个 fold 都有 test_metric.csv 且汇总已写出)的实验自动跳过;
     - 未完成的实验自动 --resume_from 最新未完成目录, 逐 fold 断点续跑
       (底层 _5fold_common: 已有 test_metric.csv 的 fold 跳过训练);
  3. --experiments 指定先跑哪个/哪几个, 配合 --gpu 实现多卡并行
     (不同实验可并行, 同一实验不要同时跑两个实例);
  4. 单个实验失败不阻塞后续实验, 结尾汇总各实验状态, 有失败则退出码非 0。

用法(云端, ~/dbond-gt-2/DBond 下):
  python run_pre_baselines.py                                   # 全部 4 个顺序跑
  python run_pre_baselines.py --gpu 0                           # 指定 0 号卡
  python run_pre_baselines.py -e dbond_s_pre dbond_m_pre --gpu 0    # 卡 0: 前两个
  python run_pre_baselines.py -e dbond_af_pre dbond_af_opt_pre --gpu 1  # 卡 1: 后两个(并行)
  python run_pre_baselines.py -e dbond_s_pre --folds 1222       # 调试单折
  python run_pre_baselines.py -e dbond_s_pre --force_new        # 忽略旧结果强制重跑(新目录)

  中断后恢复: 直接重跑原命令即可(无需记住目录), 例如
  python run_pre_baselines.py --gpu 0

日志: 每个实验的完整输出同时写终端与 logs/pre_baselines/<实验名>_<时间戳>.log
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys

# 实验注册表: 名称 -> 训练器脚本路径(相对仓库根)。
# 配置无需传入: 各训练器默认指向对应 pre.yaml。
EXPERIMENTS = {
    "dbond_s_pre": "ludbond/train_dbond_s_pre_5fold.py",
    "dbond_m_pre": "ludbond/train_dbond_m_pre_5fold.py",
    "dbond_af_pre": "ludbondaf/train_dbond_af_pre_5fold.py",
    "dbond_af_opt_pre": "ludbondaf/train_dbond_af_opt_pre_5fold.py",
}
EXPERIMENT_ORDER = list(EXPERIMENTS.keys())

EXPECTED_FOLDS = 5  # 完成判定: 5fold_metrics.csv 数据行数 >= 5
LOG_DIR = "logs/pre_baselines"


def beijing_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=8)


def find_cv_state(model_name: str) -> tuple[str | None, str | None]:
    """扫描 result/cv/{model_name}/ 下的时间戳目录。

    返回 (complete_dir, incomplete_dir):
      complete_dir  = 最新"已完成"目录(5fold_metrics.csv 行数 >= 5)
      incomplete_dir= 最新"未完成"目录(存在 fold_* 但未完成), 无则 None
    完成判定用 5fold_metrics.csv 行数而非 5fold_summary.csv 的存在:
    调试 --folds 子集也会写出 summary, 只看 summary 会误判。
    """
    base = os.path.join("result", "cv", model_name)
    if not os.path.isdir(base):
        return None, None

    def mtime(path: str) -> float:
        return os.path.getmtime(path)

    complete = []
    incomplete = []
    for name in os.listdir(base):
        cv_root = os.path.join(base, name)
        if not os.path.isdir(cv_root):
            continue
        metrics_csv = os.path.join(cv_root, "5fold_metrics.csv")
        if not os.path.exists(metrics_csv):
            # 无汇总文件: 有 fold 目录则视为进行中
            if any(f.startswith("fold_") for f in os.listdir(cv_root)):
                incomplete.append(cv_root)
            continue
        with open(metrics_csv, "r", encoding="utf-8") as f:
            n_rows = sum(1 for _ in f) - 1  # 减表头
        if n_rows >= EXPECTED_FOLDS:
            complete.append(cv_root)
        else:
            incomplete.append(cv_root)

    complete_dir = max(complete, key=mtime) if complete else None
    incomplete_dir = max(incomplete, key=mtime) if incomplete else None
    return complete_dir, incomplete_dir


def count_done_folds(cv_root: str) -> int:
    """目录下已有 metric/test_metric.csv 的 fold 数(用于日志展示进度)。"""
    n = 0
    if not os.path.isdir(cv_root):
        return 0
    for name in os.listdir(cv_root):
        if name.startswith("fold_"):
            if os.path.exists(os.path.join(cv_root, name, "metric", "test_metric.csv")):
                n += 1
    return n


def run_experiment(name: str, script: str, args: argparse.Namespace, env: dict) -> str:
    """执行单个实验, 返回状态字符串(SUCCEEDED / FAILED)。"""
    cmd = [sys.executable, script, "--fold_data_dir", args.fold_data_dir]
    if args.folds:
        cmd.extend(["--folds"] + list(args.folds))
    if args.force_new:
        cmd.append("--force_new")

    resume_dir = None
    if not args.force_new:
        complete_dir, incomplete_dir = find_cv_state(name)
        if complete_dir and not args.folds:
            print(f"[runner] {name}: 已完成({complete_dir}), 跳过")
            return "SKIPPED_DONE"
        if incomplete_dir:
            resume_dir = incomplete_dir
            cmd.extend(["--resume_from", resume_dir])
            print(f"[runner] {name}: 断点续跑 {resume_dir} "
                  f"(已完成 fold {count_done_folds(resume_dir)}/{EXPECTED_FOLDS})")
        else:
            print(f"[runner] {name}: 全新运行")
    else:
        print(f"[runner] {name}: force_new 强制重跑(新目录)")

    print(f"[runner] command: {' '.join(cmd)}")

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{name}_{beijing_now().strftime('%Y%m%d_%H%M%S')}.log")
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    log_file.write(f"# command: {' '.join(cmd)}\n")
    log_file.write(f"# CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '(unset)')}\n\n")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
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
    print(f"[runner] {name}: {status}  日志: {log_path}")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="R-01 基线 pre 变体批量运行器(顺序执行 + 断点续跑 + 多卡并行)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "多卡并行示例:\n"
            "  卡0: python run_pre_baselines.py -e dbond_s_pre dbond_m_pre --gpu 0\n"
            "  卡1: python run_pre_baselines.py -e dbond_af_pre dbond_af_opt_pre --gpu 1\n"
            "中断恢复: 重跑原命令即可, 已完成实验跳过, 未完成实验逐 fold 断点续跑。"
        ),
    )
    parser.add_argument(
        "-e", "--experiments", nargs="+", choices=EXPERIMENT_ORDER, default=None,
        help=f"要运行的实验子集(默认全部, 顺序: {' '.join(EXPERIMENT_ORDER)})",
    )
    parser.add_argument("--gpu", type=str, default=None,
                        help="设置 CUDA_VISIBLE_DEVICES(如 0 / 1 / 0,1), 默认不改动环境")
    parser.add_argument("--fold_data_dir", type=str, default="dataset/5fold", help="5fold 数据目录")
    parser.add_argument("--folds", nargs="+", default=None,
                        help="子集 fold id(调试用; 注意子集跑完的目录在后续全量运行时会按断点续跑处理)")
    parser.add_argument("--force_new", action="store_true", help="忽略旧结果强制重跑(新目录)")
    args = parser.parse_args()

    # 固定 CWD 为仓库根(脚本所在目录), 保证相对路径(trainers/config/result)一致
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)

    selected = args.experiments if args.experiments else EXPERIMENT_ORDER
    print(f"[runner] 计划执行({len(selected)}): {', '.join(selected)}")
    if args.gpu is not None:
        print(f"[runner] CUDA_VISIBLE_DEVICES = {args.gpu}")

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not os.path.isdir(args.fold_data_dir):
        sys.exit(f"[runner] 数据目录不存在: {args.fold_data_dir}(请在仓库根目录运行)")

    results = {}
    for name in selected:
        print("\n" + "=" * 70)
        print(f"[runner] ===== 实验 {name} =====")
        print("=" * 70)
        results[name] = run_experiment(name, EXPERIMENTS[name], args, env)

    print("\n[runner] ===== 汇总 =====")
    failed = 0
    for name, status in results.items():
        print(f"  {name:<20} {status}")
        if status.startswith("FAILED"):
            failed += 1
    if failed:
        sys.exit(f"[runner] {failed} 个实验失败; 重跑本命令即可断点续跑。")
    print("[runner] 全部完成。")


if __name__ == "__main__":
    main()
