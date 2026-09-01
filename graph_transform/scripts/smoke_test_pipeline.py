#!/usr/bin/env python3
"""最小化管线 smoke test：dataset → collate → forward → masked loss → backward → step。

用途（P0 修复前的可信基线验证）：
    1. 验证 gt-pre 配置下完整训练单步无异常（shape/NaN/显存）；
    2. 验证 gt-pre mask 端到端生效：扰动 intensity/scan_num 后，
       eval 模式下模型输出应逐位一致（协变量无关性）；
    3. 验证标签严格校验器（若已启用）能拒绝坏样本。

用法（devpod，项目根目录）：
    .venv/bin/python graph_transform/scripts/smoke_test_pipeline.py \
        --config graph_transform/config/pre_synthesis_5fold.yaml \
        --n_rows 512 --batch_size 128 --device cuda
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time

import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import GraphDataset, GraphDataLoader
from models import GraphTransformer
from models.utils import build_model_config
from train_graph_model import apply_ablation_config
from training.loss_functions import BinaryBondLoss
from training.trainer import Trainer

logger = logging.getLogger("smoke_test")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s[%(levelname)s]:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def make_subset_csv(csv_path: str, n_rows: int, strict_label_check: bool) -> str:
    frame = pd.read_csv(csv_path)
    if strict_label_check:
        from data.label_validation import validate_label_frame, label_error_report
        errors = label_error_report(frame)
        if not errors.empty:
            logger.warning("strict label check: %d bad rows in head sample", len(errors))
        kept = len(validate_label_frame(frame, on_error="drop"))
        logger.info("strict label check: %d/%d rows kept", kept, len(frame))
    subset = frame.head(n_rows).reset_index(drop=True)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="")
    with tmp:
        subset.to_csv(tmp.name, index=False)
    return tmp.name


def build_batch(config: dict, csv_path: str, batch_size: int, device: torch.device):
    model_config = build_model_config(config)
    config["_model_config"] = model_config
    dataset = GraphDataset(
        csv_path=csv_path,
        config=model_config,
        max_seq_len=config["data"]["max_seq_len"],
        graph_strategy=config["data"]["graph_strategy"],
        augmentation=False,
        split="train",
    )
    loader = GraphDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    batches = []
    for batch in loader:
        batches.append({k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()})
    return model_config, batches


def run_smoke(config: dict, csv_path: str, batch_size: int, device: torch.device) -> None:
    model_config, batches = build_batch(config, csv_path, batch_size, device)
    model = GraphTransformer(model_config).to(device)
    criterion = BinaryBondLoss(config.get("loss", {}))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config,
        logger=logger,
    )

    # (1) 单步训练：forward + masked loss + backward + step
    model.train()
    batch = batches[0]
    t0 = time.perf_counter()
    loss, stats = trainer._forward_pass(batch)
    trainer._backward_pass(loss)
    elapsed = time.perf_counter() - t0
    assert torch.isfinite(loss), "loss 非有限值"
    logger.info(
        "[1] 单步训练 OK: loss=%.4f valid_bonds=%d samples=%d time=%.3fs",
        loss.item(), stats["valid_bond_count"], stats["sample_count"], elapsed,
    )

    # (2) 协变量无关性：先测同输入两次的本底非确定性 Δ0（CUDA atomics 非确定），
    #     再测扰动 intensity/scan_num/rt 的 Δ1；只有 Δ1 >> Δ0 才判定 mask 泄漏。
    model.eval()
    with torch.no_grad():
        base_out = model(batch)
        repeat_out = model(batch)
        pert = dict(batch)
        pert["intensities"] = batch["intensities"] * 7.0 + 1000.0
        pert["rts"] = batch["rts"] + 33.0
        if "secondary_envs" in batch:
            pert["secondary_envs"] = batch["secondary_envs"] * 5.0 + 12345.0
        if "env_vars" in batch:
            ev = batch["env_vars"].clone()
            ev[:, 1] = ev[:, 1] * 5.0 + 12345.0
            pert["env_vars"] = ev
        pert_out = model(pert)
    delta0 = (base_out - repeat_out).abs().max().item()
    delta1 = (base_out - pert_out).abs().max().item()
    leak_threshold = max(delta0 * 10.0, 1e-4)
    if delta1 > leak_threshold:
        logger.error("[2] FAIL 协变量扰动 Δ1=%.3e 超过本底 Δ0=%.3e 阈值 %.3e —— mask 未端到端生效",
                     delta1, delta0, leak_threshold)
        sys.exit(1)
    logger.info("[2] 协变量无关性 OK: 本底 Δ0=%.3e, 扰动 Δ1=%.3e (阈值 %.3e, intensity/scan_num/rt 被屏蔽)",
                delta0, delta1, leak_threshold)

    # (3) 汇总
    n_valid = sum(int(b["label_mask"].sum().item()) for b in batches)
    logger.info("[3] smoke test 全部通过: %d batches, %d valid bonds, device=%s",
                len(batches), n_valid, device)


def main() -> None:
    parser = argparse.ArgumentParser(description="DBond gt-pre minimal pipeline smoke test")
    parser.add_argument("--config", default="graph_transform/config/pre_synthesis_5fold.yaml")
    parser.add_argument("--csv", default="dataset/5fold/1222.train.fbr.shuffle.multi.csv")
    parser.add_argument("--n_rows", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--strict_label_check", action="store_true",
                        help="启用标签严格校验（P0-2 实现后有效）")
    args = parser.parse_args()

    setup_logging()
    with open(args.config, "r", encoding="utf-8") as f:
        import yaml
        config = yaml.safe_load(f)
    config = apply_ablation_config(config)
    tag = config.get("ablation", {}).get("resolved_tag", "")
    logger.info("resolved_tag=%s state_mask=%s env_mask=%s",
                tag, config["model"].get("state_feature_mask"), config["model"].get("env_feature_mask"))

    device = torch.device(args.device)
    csv_path = make_subset_csv(args.csv, args.n_rows, args.strict_label_check)
    try:
        run_smoke(config, csv_path, args.batch_size, device)
    finally:
        os.remove(csv_path)


if __name__ == "__main__":
    main()
