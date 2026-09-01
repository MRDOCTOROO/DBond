#!/usr/bin/env python3
"""折级并行 5 折训练调度（DBond-GT-pre），输出结构与 train_5fold.py 完全一致。

思路：5 折是 5 个独立模型，天然可并行——每卡跑一折（无需 DDP）。
所有折共享同一个标准 cv_root：<checkpoint_base>/5fold/<时间戳>/，
训练完成后由 --aggregate_only 在该 cv_root 下重建标准汇总三件套：
    5fold_metrics.csv / 5fold_summary.csv / 5fold_aggregate.csv
每折子进程的中间汇总写为 5fold_*.fold_<id>.csv（防并发写冲突，可删）。

用法（devpod，项目根目录）：
    .venv/bin/python graph_transform/scripts/train_5fold_parallel.py \
        --config graph_transform/config/pre_synthesis_5fold_md6.yaml \
        --gpus 0,1,2,3
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time

FOLDS = ["1222", "2252", "3514", "6072", "9075"]


def run_fold(gpu: int, fold_id: str, config_path: str, cv_root: str, work_root: str) -> None:
    """在指定 GPU 上跑单个 fold，产物落入共享 cv_root（标准结构）。"""
    log_path = os.path.join(work_root, f"fold{fold_id}_gpu{gpu}.log")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        sys.executable, "graph_transform/scripts/train_5fold.py",
        "--config", config_path,
        "--folds", fold_id,
        "--cv_root", cv_root,
        "--summary_suffix", f".fold_{fold_id}",
    ]
    print(f"[parallel] gpu={gpu} fold={fold_id} start {time.strftime('%H:%M:%S')}", flush=True)
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
    print(f"[parallel] gpu={gpu} fold={fold_id} {status} end {time.strftime('%H:%M:%S')} log={log_path}", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"fold {fold_id} failed rc={proc.returncode}, see {log_path}")


def aggregate(config_path: str, cv_root: str, work_root: str) -> None:
    """全部折完成后重建标准汇总三件套（单进程，无写冲突）。"""
    log_path = os.path.join(work_root, "aggregate.log")
    cmd = [
        sys.executable, "graph_transform/scripts/train_5fold.py",
        "--config", config_path,
        "--cv_root", cv_root,
        "--aggregate_only",
    ]
    print(f"[parallel] aggregate start {time.strftime('%H:%M:%S')}", flush=True)
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
    print(f"[parallel] aggregate {status} log={log_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpus", default="0,1,2,3", help="逗号分隔 GPU 列表")
    ap.add_argument("--folds", default=",".join(FOLDS), help="逗号分隔 fold 列表（默认 5 折）")
    args = ap.parse_args()

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    folds = [x.strip() for x in args.folds.split(",") if x.strip()]
    if not gpus or not folds:
        print("ERROR: gpus/folds empty")
        sys.exit(1)

    import yaml as _yaml
    with open(args.config, encoding="utf-8") as f:
        base_config = _yaml.safe_load(f)
    base_ckpt = base_config.get("training", {}).get(
        "checkpoint_dir", "checkpoints/graph_transform/pre_synthesis")

    # 单一共享 cv_root（与传统 train_5fold 相同的目录结构），由 runner 创建一次
    cv_root = os.path.join(base_ckpt, "5fold", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(cv_root, exist_ok=True)
    work_root = os.path.join("logs", f"5fold_par_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(work_root, exist_ok=True)

    print(f"[parallel] {len(folds)} folds x {len(gpus)} GPUs -> {len(folds) // len(gpus) + (1 if len(folds) % len(gpus) else 0)} 轮", flush=True)
    print(f"[parallel] cv_root = {cv_root}", flush=True)

    pending = list(folds)
    results = {}
    while pending:
        batch, pending = pending[: len(gpus)], pending[len(gpus):]
        # 每轮并行启动（每卡一折，线程池仅做调度，实际训练是独立子进程）
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as ex:
            futures = {ex.submit(run_fold, gpu, fold, args.config, cv_root, work_root): fold
                       for gpu, fold in zip(gpus, batch)}
            for fut in concurrent.futures.as_completed(futures):
                fold = futures[fut]
                try:
                    fut.result()
                    results[fold] = "OK"
                except Exception as e:
                    results[fold] = f"FAIL {e}"
        print(f"[parallel] 轮完成: {results}", flush=True)

    aggregate(args.config, cv_root, work_root)
    print(f"[parallel] 全部完成: {results}", flush=True)
    print(f"[parallel] 汇总目录: {cv_root} (5fold_metrics.csv / 5fold_summary.csv / 5fold_aggregate.csv)", flush=True)


if __name__ == "__main__":
    main()
