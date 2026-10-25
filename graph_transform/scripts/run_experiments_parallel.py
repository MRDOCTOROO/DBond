#!/usr/bin/env python3
"""模型级并行训练调度（横向对比实验用）。

把多个独立实验（不同 config / 不同模型）并行调度到 GPU 池：
  例：卡0 md3, 卡1 md6, 卡2 md10, 卡3 sequence —— 四个模型同时训练，同 fold 同口径横向对比。
每个 job 是独立的 train_graph_model.py 子进程（各自 GPU / 独立日志 / 独立输出目录）。

用法（devpod，项目根目录）：
    .venv/bin/python graph_transform/scripts/run_experiments_parallel.py \
        --jobs "md3:graph_transform/config/pre_synthesis_fold1222_short_md3.yaml;\
                md6:graph_transform/config/pre_synthesis_fold1222_short_md6.yaml;\
                md10:graph_transform/config/pre_synthesis_fold1222_short.yaml;\
                seq:graph_transform/config/pre_synthesis_fold1222_short_seq.yaml" \
        --gpus 0,1,2,3

日志：logs/exp_par_<ts>/<name>_gpu<n>.log
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time


def parse_jobs(text: str):
    jobs = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"job 格式应为 name:config_path，得到 {part!r}")
        name, cfg = part.split(":", 1)
        jobs.append((name.strip(), cfg.strip()))
    return jobs


def run_job(gpu: int, name: str, config_path: str, work_root: str) -> None:
    """为 job 生成目录隔离的独立 config 后启动训练。

    修复并行冲突：train_graph_model 的 checkpoint run_id 用秒级 timestamp，
    多个 job 同秒启动会共享同一 best_model.pt 互相覆盖（曾导致 md3 加载到
    md10 的权重报 shape mismatch）。这里把 checkpoint/指标/预测/日志目录
    全部按 job name 隔离。
    """
    import yaml as _yaml

    with open(config_path, encoding="utf-8") as f:
        job_config = _yaml.safe_load(f)
    suffix = f"exp_par/{name}"
    for section, key in (
        ("training", "checkpoint_dir"),
        ("evaluation", "output_metric_dir"),
        ("evaluation", "output_pred_dir"),
        ("logging", "log_dir"),
        ("logging", "tensorboard_log_dir"),
    ):
        base = job_config.get(section, {}).get(key)
        if base:
            job_config[section][key] = os.path.join(base, suffix)
    job_config.setdefault("experiment", {})["name"] = f"{job_config.get('experiment', {}).get('name', 'gt_pre')}_{name}"
    job_config_path = os.path.join(work_root, f"config_{name}.yaml")
    with open(job_config_path, "w", encoding="utf-8") as f:
        _yaml.safe_dump(job_config, f, sort_keys=False, allow_unicode=True)

    log_path = os.path.join(work_root, f"{name}_gpu{gpu}.log")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [sys.executable, "graph_transform/scripts/train_graph_model.py",
           "--config", job_config_path]
    print(f"[exp-par] gpu={gpu} name={name} start {time.strftime('%H:%M:%S')}", flush=True)
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
    print(f"[exp-par] gpu={gpu} name={name} {status} end {time.strftime('%H:%M:%S')} log={log_path}", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with rc={proc.returncode}, see {log_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="分号分隔的 name:config_path 列表")
    ap.add_argument("--gpus", default="0,1,2,3", help="逗号分隔 GPU 列表")
    args = ap.parse_args()

    jobs = parse_jobs(args.jobs)
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    if not jobs or not gpus:
        print("ERROR: jobs/gpus empty")
        sys.exit(1)

    work_root = os.path.join("logs", f"exp_par_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(work_root, exist_ok=True)
    print(f"[exp-par] {len(jobs)} jobs x {len(gpus)} GPUs, work={work_root}", flush=True)

    pending = list(jobs)
    results = {}
    while pending:
        batch, pending = pending[: len(gpus)], pending[len(gpus):]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as ex:
            futures = {ex.submit(run_job, gpu, name, cfg, work_root): name
                       for (name, cfg), gpu in zip(batch, gpus)}
            for fut in concurrent.futures.as_completed(futures):
                name = futures[fut]
                try:
                    fut.result()
                    results[name] = "OK"
                except Exception as e:
                    results[name] = f"FAIL {e}"
        print(f"[exp-par] 轮完成: {results}", flush=True)

    print(f"[exp-par] 全部完成: {results}", flush=True)


if __name__ == "__main__":
    main()
