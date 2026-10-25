# -*- coding: utf-8 -*-
"""panel_c 重生成（split 9075，film 臂）——pod 端执行脚本（20261013 审稿修订）。

相对原 interpretability_analysis.py part-2 输出的变更：
  1) 统计框不再显示 p 值（逐键 Welch t 检验易被误读为 AUC 检验；审稿 F03）；
  2) Cohen's d 的 95% CI 改为按序列整簇 bootstrap（重抽整条肽，审稿 F04）；
  3) 标题第二行取收窄后的 FUNCTIONAL_SALIENCY_CAPTION（"Attention = functional saliency"）。
保持不变：分层抽样协议（length x charge x FBR 比例分层，seed = 42+1 = 43）、
          500 统计样本、attention 提取方式（头平均 + 双向合并 → 残基 x 键 → 键两端平均）。

用法（pod，/mnt/pvc/graphtrans/DBond 下，先 source .venv/bin/activate）：
  python result/interp_regen_20261013/regen_panel_c_9075.py
scratch 目录内需有更新后的 visualization.py（与本地仓库同版本）。
"""
import importlib.util
import json
import logging
import os
import sys

REPO = "/mnt/pvc/graphtrans/DBond"
SCRATCH = os.path.join(REPO, "result", "interp_regen_20261013")
RUN = os.path.join(REPO, "checkpoints/graph_transform/pre_synthesis_theory_folded_film/5fold/20260907_124432")
CFG = os.path.join(RUN, "fold_9075/config.yaml")
CKPT = os.path.join(RUN, "fold_9075/checkpoints/gt_pre/20260907_130026/best_model.pt")
TEST_CSV = os.path.join(REPO, "dataset/5fold_folded/9075.test.fbr.multi.csv")
NUM_SAMPLES = 500
SEED = 42 + 1
OUT_SVG = os.path.join(SCRATCH, "panel_c_bond_violin.svg")
OUT_JSON = os.path.join(SCRATCH, "panel_c_effect.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("regen_panel_c")

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "graph_transform"))
sys.path.insert(0, os.path.join(REPO, "graph_transform", "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from models import GraphTransformer  # noqa: E402
from models.utils import build_model_config, CheckpointManager  # noqa: E402
from data import GraphDataset  # noqa: E402
from utils.attention_extractor import AttentionExtractor  # noqa: E402
from interpretability_analysis import (  # noqa: E402
    load_config, setup_device, select_statistical_samples,
)

# 更新后的 visualization 按独立模块加载（不动 pod 代码树）
spec = importlib.util.spec_from_file_location(
    "viz_fixed", os.path.join(SCRATCH, "visualization.py"))
viz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viz)

for path, label in [(CFG, "config"), (CKPT, "checkpoint"), (TEST_CSV, "test csv")]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")

config = load_config(CFG)
device = setup_device(config)
logger.info("device=%s", device)

model_config = build_model_config(config)
model = GraphTransformer(model_config).to(device)
CheckpointManager.load_checkpoint(CKPT, model=model, device=device)
model.eval()
extractor = AttentionExtractor(model, device)
logger.info("checkpoint loaded")

data_config = config["data"]
data_config["test_csv_path"] = TEST_CSV
max_seq_len = data_config.get("max_seq_len")
if max_seq_len is None:
    max_seq_len = config.get("model", {}).get("max_seq_len", 32)
    data_config["max_seq_len"] = max_seq_len
max_seq_len = int(max_seq_len)

test_dataset = GraphDataset(
    csv_path=TEST_CSV,
    config=model_config,
    max_seq_len=max_seq_len,
    graph_strategy=data_config.get("graph_strategy", "hybrid"),
    augmentation=False,
    split="test",
)
logger.info("dataset: %d samples", len(test_dataset))

stat_indices = select_statistical_samples(test_dataset, NUM_SAMPLES, SEED, logger)
logger.info("sampled %d", len(stat_indices))

pop_attn, pop_labels, pop_seq, pop_edge = [], [], [], []
for idx, sample_idx in enumerate(stat_indices):
    if idx % 100 == 0:
        logger.info("extracting %d/%d", idx, len(stat_indices))
    sample_data = test_dataset[sample_idx]
    bond_labels = sample_data.get("labels")
    if bond_labels is None:
        continue
    attention_weights = extractor.extract_attention_for_sample(sample_data)
    if hasattr(test_dataset, "data") and "seq" in test_dataset.data.columns:
        sequence = test_dataset.data.iloc[sample_idx]["seq"]
    else:
        sequence = f"Sample_{sample_idx}"
    pop_attn.append(attention_weights)
    pop_labels.append(bond_labels)
    pop_seq.append(sequence)
    pop_edge.append(sample_data.get("edge_index"))

logger.info("collected %d valid samples; plotting", len(pop_attn))
_, effect = viz.plot_panel_c_violin(
    pop_attn, pop_labels, pop_edge, pop_seq,
    save_path=OUT_SVG, attention_mode="functional",
    max_seq_len=max_seq_len, n_bootstrap=1000,
)

record = {
    "config": CFG,
    "checkpoint": CKPT,
    "test_csv": TEST_CSV,
    "num_requested": NUM_SAMPLES,
    "num_valid": len(pop_attn),
    "sampling_seed": SEED,
    "attention_mode": "functional",
    "max_seq_len": max_seq_len,
    "effect": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
               for k, v in effect.items()},
}
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(record, f, ensure_ascii=False, indent=1)
print("SAVED:", OUT_SVG)
print("EFFECT:", json.dumps(record["effect"], ensure_ascii=False))