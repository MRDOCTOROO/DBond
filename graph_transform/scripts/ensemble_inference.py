#!/usr/bin/env python3
"""跨折集成（ensemble）推理：对 5 折 CV 的每个 fold f,用**其余折**的 best_model
概率平均后在 fold f 的 test 上评估（honest out-of-fold ensemble）。

为什么排除本折模型：五个 fold 是独立随机重划分（非互斥分区），fold f 的 test
谱图大量出现在 fold g(g≠f→含 f) 的 train 里——把本折模型放进集成会让"测试集"
变回训练集，指标虚高。因此每折 = 其余 4 模型平均（留一集成）。

用法（pod，项目根目录）：
    .venv/bin/python graph_transform/scripts/ensemble_inference.py \
        --cv_root checkpoints/graph_transform/pre_synthesis/5fold/20260902_073953 \
        --config graph_transform/config/pre_synthesis_5fold_md6_theory.yaml \
        --fold_data_dir dataset/5fold_soft \
        --out_dir result/metric/ensemble/theory
输出：<out_dir>/ensemble_fold_<f>.csv + 汇总 mean±std 打印与 ensemble_summary.csv。

指标口径与 evaluate_graph_model 一致（realized + q_* + cond/seq 新口径），概率
平均后经 logit 反变换喂给指标器（指标器内部固定做一次 sigmoid）。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List

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


def find_best_model(cv_root: str, fold: str) -> str:
    """fold 目录下找 best_model.pt（目录布局随版本有差异，用 find 语义兜底）。"""
    fold_dir = os.path.join(cv_root, f"fold_{fold}")
    hits = []
    for root, _, files in os.walk(fold_dir):
        if "best_model.pt" in files:
            hits.append(os.path.join(root, "best_model.pt"))
    if not hits:
        raise FileNotFoundError(f"{fold_dir} 下未找到 best_model.pt")
    return sorted(hits)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="跨折留一集成推理")
    parser.add_argument("--cv_root", required=True,
                        help="5fold 运行根目录（含 fold_<id>/ 子目录）")
    parser.add_argument("--config", required=True, help="该 run 的 YAML（模型结构+阈值）")
    parser.add_argument("--fold_data_dir", default="dataset/5fold_soft",
                        help="test CSV 目录（默认 5fold_soft 以获得 q 真值）")
    parser.add_argument("--folds", default=None, help="逗号分隔，默认从 cv_root 发现")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    setup_logging()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, "r", encoding="utf-8") as f:
        config = apply_ablation_config(yaml.safe_load(f))
    # 集成评估固定吃 5fold_soft 的 test（带 soft_multi 列）以输出 q_* 口径；
    # 无软列时数据集自动回退 realized，q 指标自然缺失。
    config.setdefault("data", {})["use_soft_labels"] = True

    model_config = build_model_config(config)
    data_config = config["data"]

    if args.folds:
        folds = [f.strip() for f in args.folds.split(",") if f.strip()]
    else:
        folds = sorted(d[len("fold_"):] for d in os.listdir(args.cv_root)
                       if d.startswith("fold_"))
    logger.info("folds: %s | device: %s", folds, device)

    all_rows: List[Dict[str, float]] = []
    for hold in folds:
        ckpts = [find_best_model(args.cv_root, f) for f in folds if f != hold]
        logger.info("fold %s: %d 个集成成员 %s", hold, len(ckpts),
                    [os.path.relpath(c, args.cv_root) for c in ckpts])

        test_kwargs = {
            "csv_path": os.path.join(args.fold_data_dir, f"{hold}.test.fbr.multi.csv"),
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
        out_csv = os.path.join(args.out_dir, f"ensemble_fold_{hold}.csv")
        import pandas as pd
        pd.DataFrame(metric_rows(fold_metrics)).to_csv(out_csv, index=False)
        logger.info("fold %s: f1=%.4f q_spearman_pep_seq=%.4f q_spearman_pep_cond=%.4f → %s",
                    hold, fold_metrics.get("f1", float("nan")),
                    fold_metrics.get("q_spearman_pep_seq", float("nan")),
                    fold_metrics.get("q_spearman_pep_cond", float("nan")), out_csv)
        all_rows.append(fold_metrics)

    # 汇总
    import pandas as pd
    keys = sorted({k for r in all_rows for k, v in r.items() if isinstance(v, (int, float))})
    summary_rows = []
    for k in keys:
        vals = np.array([r.get(k, np.nan) for r in all_rows], dtype=np.float64)
        summary_rows.append({"metric": k, "mean": float(np.nanmean(vals)),
                             "std": float(np.nanstd(vals)), "num_folds": int(len(folds))})
    summary_csv = os.path.join(args.out_dir, "ensemble_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    show = ["f1", "lab_f1_mi", "auc", "q_spearman_pep", "q_spearman_pep_cond",
            "q_spearman_pep_seq", "q_top10_enrichment_seq"]
    logger.info("==== ensemble mean±std (%d folds) ====", len(folds))
    for k in show:
        row = next((r for r in summary_rows if r["metric"] == k), None)
        if row:
            logger.info("%-24s %.4f±%.4f", k, row["mean"], row["std"])
    logger.info("summary → %s", summary_csv)


if __name__ == "__main__":
    main()
