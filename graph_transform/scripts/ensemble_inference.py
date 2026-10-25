#!/usr/bin/env python3
"""同折多种子集成（seed ensemble）推理：同一数据划分、不同随机种子的多个
best_model 概率平均后在该 fold 的 test 上评估。

【为什么必须是同折同划分——2026-09-07 泄漏实证】
五个 5fold 目录是相互重叠的随机重划分，不是互斥分区：fold 1222 的 test 有
83.57% 的谱图（按 name+scan_num+rt 唯一标识，且全行含标签完全相同）出现在
fold 2252 的 train 里（对 3514/6072 同样 76-84%）。若集成其他折的模型，测试
谱图大多在其训练集中，指标是记忆效应（实测 F1 0.795→0.856 的"增益"即此假象，
已作废）。fold 内部 train/test 严格互斥（overlap=0，已验证），所以唯一诚实的
集成方式是：同折同划分 + 不同 seed（权重初始化/dropout/批序不同）。

用法（pod，项目根目录）：
    .venv/bin/python graph_transform/scripts/ensemble_inference.py \
        --checkpoints ckpt_a/best_model.pt,ckpt_b/best_model.pt,ckpt_c/best_model.pt \
        --config graph_transform/config/pre_synthesis_fold1222_theory.yaml \
        --test_csv dataset/5fold_soft/1222.test.fbr.multi.csv \
        --out_csv result/metric/ensemble/seed5_fold1222.csv
前提：各 checkpoint 训练时用的 train split 与 --test_csv 对应的 fold 一致。
输出：完整口径指标 CSV（realized + q_* + cond/seq）。

概率平均经 logit 反变换喂给指标器（其内部固定再做一次 sigmoid，精确还原）。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GraphTransformer
from models.utils import build_model_config, CheckpointManager
from data import GraphDataset, GraphDataLoader, CachedGraphDataset
from evaluation.metrics import BinaryBondMetrics, metric_rows
from training.metrics import batch_group_keys
from train_graph_model import apply_ablation_config

logger = logging.getLogger("ensemble")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s[%(levelname)s]:%(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")


def main() -> None:
    parser = argparse.ArgumentParser(description="同折多种子集成推理（防跨折泄漏）")
    parser.add_argument("--checkpoints", required=True,
                        help="逗号分隔的 best_model.pt 列表（≥2，必须同折同划分不同 seed）")
    parser.add_argument("--config", required=True, help="该 fold 的 YAML（模型结构+阈值）")
    parser.add_argument("--test_csv", default=None,
                        help="覆盖配置的 test_csv_path（默认用配置里的路径）")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    setup_logging()
    ckpts = [p.strip() for p in args.checkpoints.split(",") if p.strip()]
    if len(ckpts) < 2:
        raise SystemExit("至少需要 2 个 checkpoint 才构成集成")
    for p in ckpts:
        if not os.path.exists(p):
            raise SystemExit(f"checkpoint 不存在: {p}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, "r", encoding="utf-8") as f:
        config = apply_ablation_config(yaml.safe_load(f))
    if args.test_csv:
        config.setdefault("data", {})["test_csv_path"] = args.test_csv
    # 集成评估固定尝试吃 soft_multi（q_* 口径）；无该列时数据集自动回退 realized
    config.setdefault("data", {})["use_soft_labels"] = True
    logger.info("ensemble of %d models | test=%s | device=%s",
                len(ckpts), config["data"]["test_csv_path"], device)

    model_config = build_model_config(config)
    data_config = config["data"]
    test_kwargs = {
        "csv_path": data_config["test_csv_path"],
        "config": config,
        "max_seq_len": data_config["max_seq_len"],
        "graph_strategy": data_config["graph_strategy"],
        "augmentation": False,
        "split": "test",
    }
    dataset_cls = CachedGraphDataset if data_config.get("cache_graphs", False) else GraphDataset
    if dataset_cls is CachedGraphDataset:
        test_kwargs.update({
            "cache_dir": data_config.get("cache_dir", "cache/graph_data"),
            "rebuild_cache": False,
            "cache_full_graphs": data_config.get("cache_full_graphs", False),
        })
    test_dataset = dataset_cls(**test_kwargs)
    loader = GraphDataLoader(dataset=test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=data_config.get("num_workers", 4),
                             pin_memory=data_config.get("pin_memory", True), drop_last=False)

    models = []
    for ck in ckpts:
        m = GraphTransformer(model_config).to(device)
        CheckpointManager.load_checkpoint(ck, model=m, device=device)
        m.eval()
        models.append(m)

    threshold_cfg = config.get("evaluation", {})
    metrics = BinaryBondMetrics({"threshold": threshold_cfg.get("threshold", 0.5),
                                 "threshold_strategy": threshold_cfg.get("threshold_strategy", "fixed")})
    eps = 1e-6
    with torch.no_grad():
        for batch_data in loader:
            batch_data = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                          for k, v in batch_data.items()}
            prob_sum = None
            for m in models:
                logits = m(batch_data)
                probs = torch.sigmoid(logits.float())
                prob_sum = probs if prob_sum is None else prob_sum + probs
            avg_prob = prob_sum / len(models)
            # 指标器内部固定 from_logits=True 再做一次 sigmoid：
            # 传 logit(avg_prob) 可精确还原概率平均（clamp 防无穷）
            avg_logit = torch.log(avg_prob.clamp(eps, 1 - eps) / (1 - avg_prob).clamp(eps, 1 - eps))
            metrics.update(
                avg_logit.cpu(),
                batch_data["labels"].cpu(),
                label_mask=batch_data.get("label_mask"),
                from_logits=True,
                soft_targets=batch_data.get("soft_labels").cpu() if batch_data.get("soft_labels") is not None else None,
                sequences=batch_data.get("sequences"),
                group_keys=batch_group_keys(batch_data),
            )

    fold_metrics = metrics.compute()
    import pandas as pd
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    pd.DataFrame(metric_rows(fold_metrics)).to_csv(args.out_csv, index=False)
    show = ["f1", "lab_f1_mi", "auc", "mcc", "q_brier", "q_spearman_pep",
            "q_spearman_pep_cond", "q_spearman_pep_seq", "q_top10_enrichment_seq"]
    logger.info("==== seed-ensemble (%d models) ====", len(models))
    for k in show:
        if k in fold_metrics:
            logger.info("%-24s %.4f", k, fold_metrics[k])
    logger.info("metrics → %s", args.out_csv)


if __name__ == "__main__":
    main()
