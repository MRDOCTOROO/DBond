#!/usr/bin/env python3
"""训练吞吐基准（单配置模式）。用法: _bench_train.py --mode baseline|compile"""

import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from data import CachedGraphDataset, GraphDataLoader
from models import GraphTransformer
from models.utils import build_model_config
from train_graph_model import apply_ablation_config
from training.loss_functions import BinaryBondLoss
from training.trainer import Trainer


def build(config: dict, batch_size: int = 1024, compile_model: bool = False):
    data_config = config["data"]
    model_config = build_model_config(config)
    config["_model_config"] = model_config
    cached = CachedGraphDataset(
        csv_path=data_config["train_csv_path"],
        config=model_config,
        cache_dir=data_config["cache_dir"],
        max_seq_len=data_config["max_seq_len"],
        graph_strategy=data_config["graph_strategy"],
        augmentation=False,
        split="train",
        rebuild_cache=False,
        cache_full_graphs=True,
    )
    loader = GraphDataLoader(dataset=cached, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=True, drop_last=True)
    device = torch.device("cuda")
    model = GraphTransformer(model_config).to(device)
    if compile_model:
        model = torch.compile(model, dynamic=False)
    criterion = BinaryBondLoss(config.get("loss", {}))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion,
                      device=device, config=config)
    return trainer, loader


def bench(trainer: Trainer, loader, n_iters: int = 10, warmup: int = 3):
    batches = []
    for i, batch in enumerate(loader):
        if i >= warmup + n_iters:
            break
        batches.append(batch)
    for batch in batches[:warmup]:
        batch = {k: (v.to(trainer.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        loss, _ = trainer._forward_pass(batch)
        trainer._backward_pass(loss)
    torch.cuda.synchronize()
    times, fwd, bwd = [], [], []
    for batch in batches[warmup:]:
        batch = {k: (v.to(trainer.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        t0 = time.perf_counter()
        loss, _ = trainer._forward_pass(batch)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        trainer._backward_pass(loss)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        times.append(t2 - t0)
        fwd.append(t1 - t0)
        bwd.append(t2 - t1)
    print(f"  forward avg={sum(fwd)/len(fwd)*1000:.1f}ms backward+step avg={sum(bwd)/len(bwd)*1000:.1f}ms")
    return sum(times) / len(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "compile", "profile"], required=True)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    with open("graph_transform/config/pre_synthesis_fold1222_short.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = apply_ablation_config(config)
    trainer, loader = build(config, compile_model=(args.mode == "compile"))
    print(f"[{args.mode}] warmup+bench start {time.strftime('%H:%M:%S')}", flush=True)

    if args.mode == "profile":
        trainer.profile_time = True
        if hasattr(trainer.model, "enable_timing"):
            trainer.model.enable_timing = True
        it = iter(loader)
        for i, batch in enumerate(it):
            if i >= 5:
                break
            batch = {k: (v.to(trainer.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            loss, _ = trainer._forward_pass(batch)
            trainer._backward_pass(loss)
            torch.cuda.synchronize()
            if i == 2:  # 稳定后打印一次
                print("MODEL TIMING:", trainer.model.last_forward_timing, flush=True)
        return

    t = bench(trainer, loader, n_iters=args.iters)
    print(f"[{args.mode}] {t:.4f} s/batch ({1024/t:.1f} batch/s) done {time.strftime('%H:%M:%S')}", flush=True)
    with open(f"/tmp/bench_{args.mode}.txt", "w") as f:
        f.write(f"{t:.4f}\n")
    print(f"[{args.mode}] saved /tmp/bench_{args.mode}.txt", flush=True)


if __name__ == "__main__":
    main()
