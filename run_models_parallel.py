#!/usr/bin/env python3
"""多模型五折并行调度器：每张卡跑一个完整实验（模型级并行）。

与 run_pre_baselines.py 的关系：本脚本只做"实验 × GPU"的调度，每个任务
调一次 run_pre_baselines.py（内部自带 5 折循环、断点续跑、完成跳过）。
4 个模型 + 4 张卡 = 一轮全并行，总耗时 ≈ 最慢模型的单模型五折时间，
避免"4 折并行 + 第 5 折单卡"的两轮等待。

用法（仓库根目录）：
  # pre 基线四件套并行（各占一卡）
  python run_models_parallel.py
  # pre+theory 四件套并行
  python run_models_parallel.py -m dbond_s_pre_theory dbond_m_pre_theory dbond_af_pre_theory dbond_af_opt_pre_theory
  # 只跑其中两个（卡 0/1）
  python run_models_parallel.py -m dbond_s_pre_theory dbond_m_pre_theory --gpus 0,1
  # 调试单折
  python run_models_parallel.py -m dbond_s_pre_theory --folds 1222

前台运行即可看进度（各实验日志同时写 logs/models_parallel/<名>_<ts>.log）。
中断恢复：直接重跑同一命令——已完成的实验自动跳过，未完成的续跑。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
import threading
import time

# 实验注册表：名称 -> (训练器脚本, 配置文件)。
# 与 run_pre_baselines.py 的 EXPERIMENTS 保持一致（该脚本按名称自行解析，
# 本表仅用于名称校验与文档；新增实验时两处同步）。
EXPERIMENTS = {
    # pre 基线（已有结果，重跑会自动跳过）
    "dbond_s_pre": ("ludbond/train_dbond_s_pre_5fold.py", "ludbond/dbond_s_config/pre.yaml"),
    "dbond_m_pre": ("ludbond/train_dbond_m_pre_5fold.py", "ludbond/dbond_m_config/pre.yaml"),
    "dbond_af_pre": ("ludbondaf/train_dbond_af_pre_5fold.py", "ludbondaf/dbond_m_exp_af_config/af_pre.yaml"),
    "dbond_af_opt_pre": ("ludbondaf/train_dbond_af_opt_pre_5fold.py", "ludbondaf/dbond_m_exp_af_config/af_opt_pre.yaml"),
    # pre + 理论键离子特征（单变量：与同名 pre 基线对照）
    "dbond_s_pre_theory": ("ludbond/train_dbond_s_pre_theory_5fold.py", "ludbond/dbond_s_config/pre_theory.yaml"),
    "dbond_m_pre_theory": ("ludbond/train_dbond_m_pre_theory_5fold.py", "ludbond/dbond_m_config/pre_theory.yaml"),
    "dbond_af_pre_theory": ("ludbondaf/train_dbond_af_pre_theory_5fold.py", "ludbondaf/dbond_m_exp_af_config/af_pre_theory.yaml"),
    "dbond_af_opt_pre_theory": ("ludbondaf/train_dbond_af_opt_pre_theory_5fold.py", "ludbondaf/dbond_m_exp_af_config/af_opt_pre_theory.yaml"),
}

_PRINT_LOCK = threading.Lock()


def run_experiment(gpu: int, name: str, extra_args: list[str], work_root: str) -> None:
    log_path = os.path.join(work_root, f"{name}_gpu{gpu}.log")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # run_pre_baselines.py 按实验名从其内部注册表解析训练器与配置；
    # --gpu 传 0 是因为 CUDA_VISIBLE_DEVICES 已把本任务的卡映射为 0 号
    cmd = [sys.executable, "run_pre_baselines.py", "-e", name, "--gpu", "0"] + extra_args
    # 说明：--gpu 传 0 是因为 CUDA_VISIBLE_DEVICES 已把该卡映射为 0 号
    with _PRINT_LOCK:
        print(f"[matrix] gpu={gpu} {name} start {time.strftime('%H:%M:%S')} log={log_path}", flush=True)
    with open(log_path, "w", encoding="utf-8") as log_f, \
            subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
        for line in proc.stdout:
            line = line.rstrip("\r\n")
            if not line:
                continue
            log_f.write(line + "\n")
            log_f.flush()
            # 关键行透传终端（折完成/汇总/异常）
            if ("done" in line or "summary saved" in line or "COMPLETE" in line.upper()
                    or "Error" in line or "Traceback" in line):
                with _PRINT_LOCK:
                    print(f"[{name}] {line}", flush=True)
        rc = proc.wait()
    status = "OK" if rc == 0 else f"FAIL({rc})"
    with _PRINT_LOCK:
        print(f"[matrix] gpu={gpu} {name} {status} end {time.strftime('%H:%M:%S')}", flush=True)
    if rc != 0:
        raise RuntimeError(f"{name} failed rc={rc}, see {log_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--models", nargs="+", default=list(EXPERIMENTS.keys()),
                    help="要跑的实验名（默认注册表全部）")
    ap.add_argument("--gpus", default="0,1,2,3", help="逗号分隔 GPU 列表")
    ap.add_argument("--folds", nargs="+", default=None, help="可选：只跑指定折（调试用）")
    ap.add_argument("--force_new", action="store_true", help="忽略旧结果强制新目录")
    args = ap.parse_args()

    for name in args.models:
        if name not in EXPERIMENTS:
            print(f"ERROR: unknown experiment {name!r}; 可选: {list(EXPERIMENTS.keys())}")
            sys.exit(1)
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    models = list(args.models)
    if not gpus or not models:
        print("ERROR: gpus/models empty")
        sys.exit(1)

    extra_args: list[str] = []
    if args.folds:
        extra_args += ["--folds"] + args.folds
    if args.force_new:
        extra_args += ["--force_new"]

    work_root = os.path.join("logs", f"models_parallel_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(work_root, exist_ok=True)
    rounds = (len(models) + len(gpus) - 1) // len(gpus)
    print(f"[matrix] {len(models)} 个实验 x {len(gpus)} GPUs -> {rounds} 轮", flush=True)

    pending = list(models)
    results = {}
    while pending:
        batch, pending = pending[: len(gpus)], pending[len(gpus):]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as ex:
            futures = {ex.submit(run_experiment, gpu, name, extra_args, work_root): name
                       for gpu, name in zip(gpus, batch)}
            for fut in concurrent.futures.as_completed(futures):
                name = futures[fut]
                try:
                    fut.result()
                    results[name] = "OK"
                except Exception as e:
                    results[name] = f"FAIL {e}"
        print(f"[matrix] 轮完成: {results}", flush=True)

    print(f"[matrix] 全部完成: {results}", flush=True)
    failed = [k for k, v in results.items() if v != "OK"]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
