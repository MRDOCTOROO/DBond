#!/usr/bin/env python3
"""4 卡并行 5 折训练调度（DBond-GT-pre）。

思路：5 折是 5 个独立模型，天然可并行——每卡跑一折（无需 DDP）。
4 张卡分 2 轮：轮 1 跑 4 折（每卡一折），轮 2 跑剩余 1 折（复用 GPU 0）。
每折内部仍由 train_5fold.py 驱动（fold 配置 + 汇总）。

用法（devpod，项目根目录）：
    .venv/bin/python graph_transform/scripts/train_5fold_parallel.py \
        --config graph_transform/config/pre_synthesis_5fold_md3.yaml \
        --gpus 0,1,2,3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

FOLDS = ["1222", "2252", "3514", "6072", "9075"]


def run_fold(gpu: int, fold_id: str, config_path: str, work_root: str) -> None:
    """在指定 GPU 上跑单个 fold。

    修复并行冲突：train_5fold.py 的 cv_root 用秒级 timestamp，多进程同秒启动会
    互相覆盖。这里为每折生成独立 config（checkpoint_dir 加 fold 后缀），
    使 cv_root 落到独立目录。
    """
    import yaml as _yaml

    with open(config_path, encoding="utf-8") as f:
        fold_config = _yaml.safe_load(f)
    base_ckpt = fold_config.get("training", {}).get(
        "checkpoint_dir", "checkpoints/graph_transform/pre_synthesis")
    # 每折独立工作根目录：<base>/5fold_par/<fold_id>/（其下再按 train_5fold 建 5fold/<ts>）
    fold_work = os.path.join(base_ckpt, "5fold_par", fold_id)
    fold_config["training"]["checkpoint_dir"] = fold_work
    fold_config["experiment"]["name"] = f"{fold_config.get('experiment', {}).get('name', 'gt_pre')}_fold{fold_id}"
    fold_config_path = os.path.join(work_root, f"config_fold_{fold_id}.yaml")
    os.makedirs(work_root, exist_ok=True)
    with open(fold_config_path, "w", encoding="utf-8") as f:
        _yaml.safe_dump(fold_config, f, sort_keys=False, allow_unicode=True)

    log_path = os.path.join(work_root, f"fold{fold_id}_gpu{gpu}.log")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        sys.executable, "graph_transform/scripts/train_5fold.py",
        "--config", fold_config_path,
        "--folds", fold_id,
    ]
    print(f"[parallel] gpu={gpu} fold={fold_id} start {time.strftime('%H:%M:%S')}", flush=True)
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
    print(f"[parallel] gpu={gpu} fold={fold_id} {status} end {time.strftime('%H:%M:%S')} log={log_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpus", default="0,1,2,3", help="逗号分隔 GPU 列表")
    ap.add_argument("--folds", default=",".join(FOLDS), help="逗号分隔 fold 列表（默认 5 折）")
    ap.add_argument("--checkpoint_base", default="checkpoints/graph_transform/pre_synthesis/gt_pre",
                    help="汇总输出根目录（与 train_5fold 的 checkpoint_dir 对应）")
    args = ap.parse_args()

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    folds = [x.strip() for x in args.folds.split(",") if x.strip()]
    if not gpus or not folds:
        print("ERROR: gpus/folds empty")
        sys.exit(1)

    print(f"[parallel] {len(folds)} folds x {len(gpus)} GPUs -> {len(folds) // len(gpus) + (1 if len(folds) % len(gpus) else 0)} 轮", flush=True)

    pending = list(folds)
    results = {}
    work_root = os.path.join("logs", f"5fold_par_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(work_root, exist_ok=True)
    while pending:
        batch, pending = pending[: len(gpus)], pending[len(gpus):]
        # 每轮并行启动（每卡一折，线程池仅做调度，实际训练是独立子进程）
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as ex:
            futures = {ex.submit(run_fold, gpu, fold, args.config, work_root): fold
                       for gpu, fold in zip(gpus, batch)}
            for fut in concurrent.futures.as_completed(futures):
                fold = futures[fut]
                try:
                    fut.result()
                    results[fold] = "OK"
                except Exception as e:
                    results[fold] = f"FAIL {e}"
        print(f"[parallel] 轮完成: {results}", flush=True)

    print(f"[parallel] 全部完成: {results}", flush=True)


if __name__ == "__main__":
    main()
