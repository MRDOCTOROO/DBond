#!/usr/bin/env python3
"""DBond-GT-pre 合成前候选评分脚本（R-01 / R-02 验收）。

对任意候选序列（从未合成、没有任何谱图或实测协变量）输出唯一、可复现的候选级分数：

    sequence
      → 理论前体 m/z（PBCLA/utils.py 残基组成表，含 B/O/X/Z；pep_mass 列语义=前体 m/z）
      × 预设 charge×NCE 条件网格（默认 charge {2,3,4,5,6} × NCE {20,30,40,50}，20 点等权，
        与训练集分布一致，无外推）
      → DBond-GT-pre 逐键断裂概率
      → R_pred^pre = 网格等权平均（每个网格点 = 该条件下全部键概率的均值）

输入输出：
    --sequences  候选序列文件（txt 每行一条，或含 seq 列的 CSV）
    输出 presynthesis_scores.csv: seq, n_bonds, R_pred_pre, rank（按分数降序）

要求：
    * checkpoint 必须是以 ablation.pre_synthesis=true 训练的 DBond-GT-pre 权重；
    * --config 必须是对应配置（含 ablation.pre_synthesis: true）——mask 是
      persistent=False 的 buffer，不随 checkpoint 保存，由 config 在加载时生效；
    * 脚本启动时校验 resolved_tag == gt_pre 且 mask == state[T,T,F]/env[T,F]。

--verify 三重自检（R-01/R-02 验收证据）：
    (1) 重复运行一致性：同输入跑两遍，逐值完全一致；
    (2) 行序无关性：打乱输入行序，逐候选分数一致（容差 1e-9，批组合变化仅影响浮点归约末位）；
    (3) 实测协变量无关性：把 intensity/rt/scan_num 换成随机大值，分数不变
        —— 证明三条注入点的 mask 端到端生效，无实测信息泄漏。

用法（云端，~/graphtrans/DBond 下）：
    python graph_transform/scripts/score_presynthesis.py \
        --config <gt_pre 的 fold config.yaml 或 pre_synthesis_5fold.yaml> \
        --checkpoint <gt_pre best_model.pt> \
        --sequences candidates.txt \
        --output_dir result/presynthesis \
        --verify
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GraphTransformer
from models.utils import build_model_config, CheckpointManager
from data import GraphDataset, GraphDataLoader, CachedGraphDataset
from evaluation import Evaluator
from train_graph_model import apply_ablation_config

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEFAULT_CHARGES = [2, 3, 4, 5, 6]
DEFAULT_NCES = [20, 30, 40, 50]

logger = logging.getLogger("score_presynthesis")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s[%(levelname)s]:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_pbcla_utils():
    """以文件路径方式加载 PBCLA/utils.py —— import 副作用会更新 pyteomics 的
    B/O/X/Z 残基组成与质量表，随后 mass.fast_mass 即可计算含特殊残基的肽。"""
    path = os.path.join(REPO_ROOT, "PBCLA", "utils.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"PBCLA utils not found: {path}")
    spec = importlib.util.spec_from_file_location("pbcla_utils", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pbcla_utils"] = module
    spec.loader.exec_module(module)
    return module


def parse_int_list(text: str) -> List[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError(f"Empty list: {text}")
    return values


def load_sequences(path: str, alphabet: str) -> List[str]:
    sequences: List[str] = []
    if path.lower().endswith(".csv"):
        frame = pd.read_csv(path)
        if "seq" not in frame.columns:
            raise ValueError(f"--sequences CSV 必须含 seq 列，实际列: {list(frame.columns)}")
        raw = frame["seq"].astype(str).tolist()
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    seen = set()
    valid_chars = set(alphabet) - {"#"}
    for item in raw:
        seq = item.strip().upper()
        if not seq or seq in seen:
            continue
        bad = sorted(set(seq) - valid_chars)
        if bad:
            raise ValueError(f"序列含字母表外字符 {bad}: {seq}")
        sequences.append(seq)
        seen.add(seq)
    if not sequences:
        raise ValueError("No valid sequences loaded.")
    return sequences


def theoretical_mz(seq: str, charge: int, mass_module) -> float:
    """理论前体 m/z = fast_mass(seq, charge=c)。须在 load_pbcla_utils() 之后调用。"""
    from pyteomics import mass
    return float(mass_module.mass.fast_mass(sequence=seq, charge=charge))


def build_grid_frame(
    sequences: List[str],
    charges: List[int],
    nces: List[int],
    mass_module,
) -> pd.DataFrame:
    """构造合成推理表：每序列 × 每 (charge, NCE) 网格点一行。
    intensity/rt/scan_num 置 0（模型侧被 mask 屏蔽，置 0 为双保险）。"""
    rows = []
    for seq in sequences:
        bond_len = max(len(seq) - 1, 0)
        true_multi = ";".join(["0"] * bond_len)
        for charge in charges:
            mz = theoretical_mz(seq, charge, mass_module)
            for nce in nces:
                rows.append({
                    "name": f"{seq}|z{charge}|nce{nce}",
                    "seq": seq,
                    "charge": charge,
                    "pep_mass": mz,
                    "intensity": 0.0,
                    "nce": nce,
                    "scan_num": 0.0,
                    "rt": 0.0,
                    "true_multi": true_multi,
                })
    return pd.DataFrame(rows)


def run_inference(
    config: Dict,
    model: torch.nn.Module,
    device: torch.device,
    frame: pd.DataFrame,
    threshold: float,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    """对 grid frame 执行一次前向推理，返回带 mean_bond_prob 列的结果表。"""
    data_config = config["data"]
    # 评分时禁用缓存重建（边结构 cache 与 mask 无关；edge_attr 在 __getitem__ 实时计算）
    data_config["rebuild_cache"] = False
    data_config["cache_full_graphs"] = False
    data_config["num_workers"] = num_workers
    config["training"]["batch_size"] = batch_size

    temp_csv = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="")
    try:
        frame.to_csv(temp_csv.name, index=False)
        data_config["test_csv_path"] = temp_csv.name

        dataset_cls = CachedGraphDataset if data_config.get("cache_graphs", False) else GraphDataset
        kwargs = {
            "csv_path": data_config["test_csv_path"],
            "config": config["_model_config"],
            "max_seq_len": data_config["max_seq_len"],
            "graph_strategy": data_config["graph_strategy"],
            "augmentation": False,
            "split": "test",
        }
        if dataset_cls is CachedGraphDataset:
            kwargs.update({"cache_dir": data_config.get("cache_dir", "cache/graph_data"), "rebuild_cache": False})
        dataset = dataset_cls(**kwargs)
        if len(dataset) == 0:
            raise ValueError("No rows remained after dataset filtering; check max_seq_len and sequences.")
        loader = GraphDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        evaluator = Evaluator(model=model, device=device, config=config, logger=logger)
        outputs = evaluator.collect_prediction_outputs(loader, threshold=threshold)
    finally:
        os.remove(temp_csv.name)

    result = dataset.data.copy()
    result["pred_prob"] = outputs["prob_strings"]
    result["mean_bond_prob"] = result["pred_prob"].apply(
        lambda s: float(np.mean([float(x) for x in str(s).split(";") if x != ""]))
    )
    return result


def aggregate_scores(grid_result: pd.DataFrame) -> pd.DataFrame:
    """R_pred^pre = 网格等权平均；输出按分数降序的候选表。"""
    scores = (
        grid_result.groupby("seq", sort=True)
        .agg(n_bonds=("seq", lambda s: len(s.iloc[0]) - 1),
             n_grid_points=("mean_bond_prob", "size"),
             R_pred_pre=("mean_bond_prob", "mean"))
        .reset_index()
    )
    scores = scores.sort_values(["R_pred_pre", "seq"], ascending=[False, True]).reset_index(drop=True)
    scores.insert(0, "rank", scores.index + 1)
    return scores


def verify(
    config: Dict,
    model: torch.nn.Module,
    device: torch.device,
    frame: pd.DataFrame,
    threshold: float,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> List[str]:
    """三重自检，返回报告行列表。"""
    report: List[str] = []
    base = aggregate_scores(run_inference(config, model, device, frame, threshold, batch_size, num_workers))

    # (1) 重复运行一致性（同批组合，应逐位一致）
    repeat = aggregate_scores(run_inference(config, model, device, frame, threshold, batch_size, num_workers))
    merged = base.merge(repeat, on="seq", suffixes=("_a", "_b"))
    max_repeat = float(np.max(np.abs(merged["R_pred_pre_a"] - merged["R_pred_pre_b"])))
    report.append(f"(1) repeat-run max |ΔR_pred_pre| = {max_repeat:.3e}  (期望 0，逐位一致)")
    if max_repeat != 0.0:
        report.append("    [WARN] 重复运行不完全一致，检查 AMP/非确定性内核")

    # (2) 行序无关性（批组合改变，浮点归约末位允许 1e-9）
    rng = np.random.default_rng(seed)
    shuffled = frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)
    shuffled_scores = aggregate_scores(run_inference(config, model, device, shuffled, threshold, batch_size, num_workers))
    merged2 = base.merge(shuffled_scores, on="seq", suffixes=("_a", "_b"))
    max_shuffle = float(np.max(np.abs(merged2["R_pred_pre_a"] - merged2["R_pred_pre_b"])))
    report.append(f"(2) row-order-invariance max |ΔR_pred_pre| = {max_shuffle:.3e}  (阈值 1e-9)")
    if max_shuffle > 1e-9:
        report.append("    [FAIL] 行序改变影响了候选分数，超出容差")

    # (3) 实测协变量无关性：intensity/rt/scan_num 灌随机大值，分数应不变（mask 生效证明）
    perturbed = frame.copy()
    rng2 = np.random.default_rng(seed + 1)
    perturbed["intensity"] = 10 ** rng2.uniform(3, 8, len(perturbed))
    perturbed["rt"] = rng2.uniform(0.5, 61.0, len(perturbed))
    perturbed["scan_num"] = rng2.uniform(1, 90000, len(perturbed))
    perturbed_scores = aggregate_scores(run_inference(config, model, device, perturbed, threshold, batch_size, num_workers))
    merged3 = base.merge(perturbed_scores, on="seq", suffixes=("_a", "_b"))
    max_perturb = float(np.max(np.abs(merged3["R_pred_pre_a"] - merged3["R_pred_pre_b"])))
    report.append(f"(3) covariate-perturbation max |ΔR_pred_pre| = {max_perturb:.3e}  (期望 0：intensity/rt/scan 全程被屏蔽)")
    if max_perturb > 1e-9:
        report.append("    [FAIL] 实测协变量影响了分数 —— mask 未端到端生效，禁止用于 pre-synthesis 结论")

    return report


def reference_ppm_check(reference_csv: str, charges: List[int], mass_module) -> List[str]:
    """理论 m/z 与数据集观测 pep_mass 的 ppm 偏差（D2 语义验证）。"""
    df = pd.read_csv(reference_csv)
    if not {"seq", "charge", "pep_mass"}.issubset(df.columns):
        raise ValueError("--reference_csv 需含 seq/charge/pep_mass 列")
    obs = df.groupby(["seq", "charge"])["pep_mass"].mean().reset_index()
    obs["charge"] = obs["charge"].astype(int)
    obs = obs[obs["charge"].isin(charges)]
    if obs.empty:
        return ["[reference] 参考数据中无匹配 charge 的行，跳过"]
    ppm = obs.apply(
        lambda r: (theoretical_mz(r["seq"], int(r["charge"]), mass_module) - r["pep_mass"]) / r["pep_mass"] * 1e6,
        axis=1,
    )
    return [
        f"[reference] n(seq,charge)={len(obs)}  ppm偏差: median={ppm.median():.2f}, "
        f"p5={np.percentile(ppm, 5):.2f}, p95={np.percentile(ppm, 95):.2f}, "
        f"max|ppm|={ppm.abs().max():.2f}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="DBond-GT-pre presynthesis candidate scoring (R-01)")
    parser.add_argument("--config", required=True, help="gt_pre 配置 yaml（含 ablation.pre_synthesis: true）")
    parser.add_argument("--checkpoint", required=True, help="gt_pre 训练权重 best_model.pt")
    parser.add_argument("--sequences", required=True, help="候选序列：txt 每行一条，或含 seq 列的 CSV")
    parser.add_argument("--output_dir", default="result/presynthesis", help="输出目录")
    parser.add_argument("--charges", default=",".join(map(str, DEFAULT_CHARGES)), help="预设 charge 网格（逗号分隔）")
    parser.add_argument("--nces", default=",".join(map(str, DEFAULT_NCES)), help="预设 NCE 网格（逗号分隔）")
    parser.add_argument("--threshold", type=float, default=0.5, help="二值化阈值（仅 pred 字符串用，不影响分数）")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--grid_detail", action="store_true", help="输出逐网格点明细 CSV")
    parser.add_argument("--bond_detail", action="store_true", help="输出逐键概率明细 CSV（较大）")
    parser.add_argument("--verify", action="store_true", help="三重自检：重复一致 / 行序无关 / 协变量无关")
    parser.add_argument("--reference_csv", default=None, help="可选：数据集 CSV，核对理论 m/z 与观测 pep_mass 的 ppm 偏差")
    parser.add_argument("--seed", type=int, default=42, help="--verify 随机打乱/扰动种子")
    args = parser.parse_args()

    setup_logging()
    charges = parse_int_list(args.charges)
    nces = parse_int_list(args.nces)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.device:
        config.setdefault("device", {})["auto_detect"] = False
        config["device"]["device_type"] = args.device
    config = apply_ablation_config(config)

    # 防误用守卫：必须是 gt_pre 配置（mask 由 config 生效，checkpoint 不含 mask）
    resolved_tag = config.get("ablation", {}).get("resolved_tag", "")
    if resolved_tag != "gt_pre":
        sys.exit(f"[score_pre] 配置 resolved_tag={resolved_tag!r}，需 gt_pre。请用 pre_synthesis 训练配置。")
    model_cfg = config.get("model", {})
    if list(model_cfg.get("state_feature_mask", [])) != [True, True, False] or \
       list(model_cfg.get("env_feature_mask", [])) != [True, False]:
        sys.exit(f"[score_pre] mask 校验失败: state={model_cfg.get('state_feature_mask')}, env={model_cfg.get('env_feature_mask')}")
    logger.info("Config OK: resolved_tag=gt_pre, state_mask=[T,T,F], env_mask=[T,F]")

    mass_module = load_pbcla_utils()
    sequences = load_sequences(args.sequences, config["model"]["alphabet"])
    max_len = config["data"]["max_seq_len"]
    too_long = [s for s in sequences if len(s) > max_len]
    if too_long:
        logger.warning("以下序列超过 max_seq_len=%d 将被数据集过滤: %s", max_len, too_long[:5])
        sequences = [s for s in sequences if len(s) <= max_len]
    logger.info("Scoring %d sequence(s) on %d-point grid (charge %s × NCE %s, 等权)",
                len(sequences), len(charges) * len(nces), charges, nces)

    device_config = config.get("device", {})
    if device_config.get("auto_detect", True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_config.get("device_type", "cpu"))
    logger.info("Using device: %s", device)

    model_config = build_model_config(config)
    config["_model_config"] = model_config
    model = GraphTransformer(model_config).to(device)
    CheckpointManager.load_checkpoint(args.checkpoint, model=model, device=device)
    model.eval()
    logger.info("Loaded checkpoint: %s", args.checkpoint)

    frame = build_grid_frame(sequences, charges, nces, mass_module)
    if args.reference_csv:
        for line in reference_ppm_check(args.reference_csv, charges, mass_module):
            logger.info(line)

    grid_result = run_inference(config, model, device, frame, args.threshold, args.batch_size, args.num_workers)
    scores = aggregate_scores(grid_result)

    os.makedirs(args.output_dir, exist_ok=True)
    scores_path = os.path.join(args.output_dir, "presynthesis_scores.csv")
    scores.to_csv(scores_path, index=False)
    logger.info("Saved %d candidate scores to %s", len(scores), scores_path)

    if args.grid_detail:
        detail = grid_result[["seq", "charge", "nce", "pep_mass", "mean_bond_prob"]].copy()
        detail.to_csv(os.path.join(args.output_dir, "presynthesis_grid_detail.csv"), index=False)
        logger.info("Saved grid detail.")
    if args.bond_detail:
        rows = []
        for _, r in grid_result.iterrows():
            probs = [float(x) for x in str(r["pred_prob"]).split(";") if x != ""]
            for pos, prob in enumerate(probs, start=1):
                rows.append({"seq": r["seq"], "charge": r["charge"], "nce": r["nce"],
                             "bond_position": pos, "prob": prob})
        pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "presynthesis_bond_probs.csv"), index=False)
        logger.info("Saved bond-level detail.")

    if args.verify:
        logger.info("Running verification (repeat / row-order / covariate-perturbation)...")
        report = verify(config, model, device, frame, args.threshold, args.batch_size, args.num_workers, args.seed)
        report_path = os.path.join(args.output_dir, "verify_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report) + "\n")
        for line in report:
            logger.info(line)
        logger.info("Verification report saved to %s", report_path)

    preview = scores.head(min(10, len(scores)))
    for _, row in preview.iterrows():
        logger.info("rank=%d  R_pred_pre=%.6f  n_bonds=%d  %s",
                    int(row["rank"]), row["R_pred_pre"], int(row["n_bonds"]), row["seq"])


if __name__ == "__main__":
    main()
