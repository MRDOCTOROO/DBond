"""Independent, pod-only audit of DBond-GT Stage 12-14 same-pass predictions.

The producer writes an ``audit_manifest.json`` with one entry per trained arm
and outer fold. This auditor never selects a checkpoint, threshold, seed, or
hyperparameter. It verifies the selected checkpoint's prediction source and
recomputes all reported metrics from unpadded bond rows.

Run inside the authorized pod Python environment only::

    python audit_core.py --manifest /path/to/audit_manifest.json --out /path/to/audit

The CSV output deliberately keeps each fold/seed separate. Five outer folds
reuse sequences across experiments and are not independent test sets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


FOLDS = ("1222", "2252", "3514", "6072", "9075")
METRICS = (
    "bond_f1", "bond_accuracy", "roc_auc", "pr_auc", "mcc",
    "spectrum_precision", "spectrum_recall", "spectrum_f1_harmonic",
    "q_brier", "ece_15", "q_seq", "top10_enrichment", "top20_enrichment",
)
LOWER_BETTER = {"q_brier", "ece_15"}
PRODUCER_NAMES = {
    "bond_f1": "f1", "bond_accuracy": "accuracy", "roc_auc": "auc",
    "spectrum_precision": "ex_precision", "spectrum_recall": "ex_recall",
    "q_brier": "q_brier", "q_seq": "q_spearman_pep_seq",
    "top10_enrichment": "q_top10_enrichment_seq",
    "top20_enrichment": "q_top20_enrichment_seq",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def vector(value: object, cast=float) -> list:
    raw = str(value).strip()
    if not raw or raw.lower() == "nan":
        return []
    if raw.endswith(";"):
        raw = raw[:-1]
    return [cast(part) for part in raw.split(";")]


def _source_rows(path: Path, max_seq_len: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require(all(col in frame for col in ("seq", "charge", "nce", "true_multi", "soft_multi")),
            f"{path}: source columns missing")
    # Independent implementation of the formal loader's strict label check.
    # A trailing semicolon is legal; every retained token must be binary and
    # its count must be the number of bonds in that sequence.
    keep = []
    for seq, labels in zip(frame["seq"], frame["true_multi"]):
        if pd.isna(seq) or pd.isna(labels):
            keep.append(False)
            continue
        name, raw = str(seq).strip(), str(labels).strip()
        tokens = raw.split(";")
        if tokens and tokens[-1] == "":
            tokens = tokens[:-1]
        keep.append(2 <= len(name) <= max_seq_len and
                    len(tokens) == len(name) - 1 and
                    all(token in {"0", "1"} for token in tokens))
    frame = frame.iloc[np.flatnonzero(keep)]
    return frame.reset_index(drop=True)


def _producer_metrics(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(rows and {"metric", "value"}.issubset(rows[0]),
            f"{path}: unrecognized metric CSV")
    result = {row["metric"]: float(row["value"]) for row in rows}
    require(len(result) == len(rows), f"{path}: duplicate metric names")
    return result


def _ece_equal_width(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """Conventional realized-label ECE; fixed 15 equal-width probability bins."""
    indices = np.minimum(np.floor(p * bins).astype(int), bins - 1)
    total = len(p)
    return float(sum(np.count_nonzero(indices == index) / total *
                     abs(float(p[indices == index].mean()) -
                         float(y[indices == index].mean()))
                     for index in range(bins) if np.any(indices == index)))


def _ranking(scores: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    if len(scores) < 2:
        return 0.0, 0.0, 0.0
    rho = float(spearmanr(scores, truth).statistic)
    if not math.isfinite(rho):
        rho = 0.0
    order = np.argsort(-scores, kind="mergesort")
    overall = float(np.mean(truth))
    enrichments = []
    for percentage in (10, 20):
        count = max(1, int(round(percentage / 100.0 * len(scores))))
        enrichments.append(float(truth[order[:count]].mean() / overall)
                           if overall > 1e-8 else 0.0)
    return rho, *enrichments


def _expected_binding(entry: dict, key: str, path: Path) -> None:
    recorded = entry.get(key + "_sha256") or entry.get(key + "_sha")
    if key == "validation_source_csv" and entry.get("stage") == "S13":
        # Stage13's producer names this checksum without the `_csv` infix.
        alternate = entry.get("validation_source_sha256")
        require(recorded in (None, alternate),
                f"Conflicting Stage13 validation-source SHA fields: {path}")
        recorded = alternate
    require(isinstance(recorded, str) and len(recorded) == 64,
            f"Missing required {key} SHA binding: {path}")
    require(sha256(path) == recorded, f"{key} SHA mismatch: {path}")


def _manifest_binding(manifest: dict, manifest_path: Path) -> tuple[Path | None, dict]:
    """Independently bind an official audit manifest to its frozen pod run."""
    stage = str(manifest.get("stage", ""))
    if stage == "historical_stage11_single_control":
        return None, {"reference_only": True}
    require(stage in {"S12", "S13", "S14"},
            f"Unknown stage cannot be marked audited: {stage!r}")
    for key in ("ops", "root", "source_commit", "source_root"):
        require(isinstance(manifest.get(key), str) and manifest[key],
                f"Official manifest missing {key}")
    ops = Path(manifest["ops"]).resolve()
    root = Path(manifest["root"]).resolve()
    source_root = Path(manifest["source_root"]).resolve()
    require(manifest_path.resolve() == ops / "audit_manifest.json"
            and ops.name.endswith("_ops")
            and Path('/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform') in root.parents
            and source_root.is_dir(),
            "Manifest path, ops, run root or source root escaped the authorized pod locations")
    launch_path, terminal_path = ops / "launch.json", ops / "terminal.json"
    require(launch_path.is_file() and terminal_path.is_file(),
            "Launch or terminal completion record missing")
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    require(launch.get("root") == terminal.get("root") == str(root)
            and terminal.get("status") in {"completed", "failed"},
            "Launch/terminal root or completion status mismatch")
    recorded_manifest_sha = terminal.get("audit_manifest_sha256")
    if recorded_manifest_sha is not None:
        require(isinstance(recorded_manifest_sha, str)
                and recorded_manifest_sha == sha256(manifest_path),
                "Terminal audit-manifest SHA differs from the file under review")
    if terminal["status"] == "failed":
        require(stage == "S12" and manifest.get("status") == "partial_pending_audit"
                and isinstance(manifest.get("failed_arms"), list)
                and manifest["failed_arms"],
                "Failed terminal cannot be marked audited without explicit partial-arm evidence")
    source_commit = manifest["source_commit"]
    require(source_commit in {launch.get("source_commit"),
                              launch.get("source_revision"), launch.get("gitrev")},
            "Manifest source commit differs from launch")
    if stage == "S14":
        require(isinstance(manifest.get("stage13_audit"), dict)
                and manifest["stage13_audit"] == launch.get("stage13_audit"),
                "Stage14 predecessor audit proof differs from frozen launch")
    actual = subprocess.check_output(["git", "-C", str(source_root),
                                      "rev-parse", "HEAD"], text=True).strip()
    require(actual == source_commit, "Frozen source checkout HEAD changed")
    bound_files = manifest.get("source_files_sha256") or launch.get("source_sha256")
    if stage == "S12":
        bound_path = Path(manifest["source_manifest"]).resolve()
        require(bound_path == ops / "source_manifest.json"
                and bound_path.is_file()
                and sha256(bound_path) == manifest["source_manifest_sha256"]
                == launch["source_manifest_sha256"],
                "Stage12 source file manifest changed")
        bound_files = json.loads(bound_path.read_text(encoding="utf-8"))
    elif stage == "S13":
        require(bound_files == launch.get("source_files_sha256"),
                "Stage13 source-file SHA map differs from frozen launch")
        require(manifest.get("budget_profile") == launch.get("budget_profile")
                == "minimal_s13_c64_5fold_v1",
                "Reduced Stage13 budget profile differs from frozen launch")
        stage12 = manifest.get("stage12_audit")
        require(stage12 == {
                    "path": launch.get("stage12_audit"),
                    "sha256": launch.get("stage12_audit_sha256"),
                    "root": launch.get("stage12_root"),
                    "launch_sha256": launch.get("stage12_launch_sha256"),
                }, "Stage12 predecessor proof differs from frozen Stage13 launch")
        stage12_ops = Path('/mnt/pvc/graphtrans/dbond_s12_gated_fusion_20261026_ops')
        stage12_receipt_path = Path(stage12["path"]).resolve()
        require(stage12_receipt_path ==
                stage12_ops / 'independent_audit/formal/audit_receipt.json'
                and sha256(stage12_receipt_path) == stage12["sha256"]
                and sha256(stage12_ops / 'launch.json') == stage12["launch_sha256"],
                "Stage12 formal receipt or launch SHA changed")
        stage12_receipt = json.loads(stage12_receipt_path.read_text(encoding='utf-8'))
        stage12_launch = json.loads((stage12_ops / 'launch.json').read_text(encoding='utf-8'))
        require(stage12_receipt.get('status') == 'audited'
                and stage12_receipt.get('stage') == 'S12'
                and stage12_receipt.get('ops') == str(stage12_ops)
                and stage12_receipt.get('root') == stage12_launch.get('root')
                == stage12['root'],
                "Stage12 predecessor is not the audited matching root")
        preserved = manifest.get("preserved_baseline")
        require(isinstance(preserved, dict)
                and preserved == launch.get("preserved_baseline"),
                "Reused completed baseline lineage differs from frozen launch")
        old_ops = Path(preserved["ops"]).resolve()
        old_root = Path(preserved["root"]).resolve()
        require(old_ops == Path('/mnt/pvc/graphtrans/dbond_s13_multiscale_conv_20261026_ops')
                and old_root == Path('/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/dbond_s13_multiscale_conv_20261026_20261026_010518_d4ec97ad'),
                "Baseline lineage points outside the preserved Stage13 batch")
        old_launch = json.loads((old_ops / "launch.json").read_text(encoding="utf-8"))
        old_manifest = json.loads((old_root / "manifest.json").read_text(encoding="utf-8"))
        old_status = json.loads((old_root / "status.json").read_text(encoding="utf-8"))
        require(sha256(old_ops / "launch.json") == preserved["launch_sha256"]
                and sha256(old_root / "manifest.json") == preserved["batch_manifest_sha256"]
                and old_launch["root"] == old_manifest["root"] == old_status["root"] == str(old_root)
                and old_launch["source_revision"] == preserved["source_commit"]
                and old_status["arms"][0]["status"] == "completed"
                and old_status["arms"][0]["returncode"] == 0
                and old_manifest["arms"][0]["config_sha256"] == preserved["baseline_config_sha256"]
                and sha256(old_manifest["arms"][0]["metrics_csv"]) == preserved["fivefold_metrics_sha256"],
                "Preserved baseline launch, completed arm, config, or fivefold metrics changed")
        for relative, expected in old_launch["source_files_sha256"].items():
            current = source_root / relative
            require(current.is_file() and sha256(current) == expected,
                    f"Production source differs from preserved baseline: {relative}")
    require(isinstance(bound_files, dict) and bound_files,
            "Missing frozen source-file SHA map")
    for relative, expected in bound_files.items():
        path = (source_root / relative).resolve()
        require(source_root in path.parents and path.is_file()
                and sha256(path) == expected,
                f"Frozen source file changed: {relative}")
    return source_root, {"stage": stage, "ops": str(ops), "root": str(root),
                         "source_commit": source_commit,
                         "launch_sha256": sha256(launch_path),
                         "terminal_sha256": sha256(terminal_path),
                         "terminal_status": terminal["status"],
                         "source_file_count": len(bound_files)}


def _strict_checkpoint(entry: dict, source_root: Path) -> dict:
    """Reconstruct the model on CPU outside the training producer's finalizer."""
    model_dir = str(source_root / "graph_transform")
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    from models.graph_transformer import GraphTransformer
    from models.node_context_conv import node_conv_mode_code, node_conv_spec
    from models.utils import build_model_config

    config = yaml.safe_load(Path(entry["config"]).read_text(encoding="utf-8"))
    saved = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False)
    require(config["experiment"]["seed"] == int(entry["seed"]),
            f"{entry['arm']}/{entry['fold']}: runtime training seed differs")
    require(config["data"]["max_seq_len"] == int(entry["max_seq_len"]) == 32,
            f"{entry['arm']}/{entry['fold']}: runtime sequence bound differs")
    split_seed = entry.get("split_seed", entry.get("validation_split_seed"))
    require(split_seed is not None and
            config["training"]["validation_split_seed"] == int(split_seed),
            f"{entry['arm']}/{entry['fold']}: runtime split seed differs")
    protocol = saved.get("extra_state", {}).get("training_rng_protocol", {})
    require(protocol.get("training_seed") == int(entry["seed"])
            and protocol.get("validation_split_seed") == int(split_seed),
            f"{entry['arm']}/{entry['fold']}: actual checkpoint seed protocol differs")
    require(saved["config"]["hidden_dim"] == 256
            and saved["config"]["bond_head_type"] == "four_state_mlp"
            and saved["config"]["state_feature_mask"] == [True, True, False]
            and saved["config"]["env_feature_mask"] == [True, False]
            and config["model"]["state_feature_mask"] == [True, True, True]
            and config["model"]["env_feature_mask"] == [True, True]
            and config["ablation"]["pre_synthesis"] is True
            and saved["config"]["theory_fusion_stage"] == "late"
            and saved["config"]["bond_fusion_mode"] == "concat"
            and saved["config"]["num_gcn_layers"] == 3
            and saved["config"]["num_gat_layers"] == 2
            and saved["config"]["num_attention_heads"] == 4
            and config["loss"]["four_state_conditional_weight"] == 1.0
            and config["loss"]["four_state_union_brier_weight"] == 0.0,
            f"{entry['arm']}/{entry['fold']}: checkpoint model recipe changed")
    require(saved["config"]["node_conv_mode"] == config["model"]["node_conv_mode"]
            and saved["config"].get("stage12_fusion_mode", "legacy") ==
                config["model"].get("stage12_fusion_mode", "legacy"),
            f"{entry['arm']}/{entry['fold']}: checkpoint mode differs from config")
    if entry["stage"] == "S14":
        require(float(config["loss"]["ranking_loss_weight"]) == float(entry["lambda"])
                and float(config["loss"]["ranking_temperature"]) == .1
                and float(config["loss"]["ranking_min_target_delta"]) == .05
                and config["loss"]["ranking_loss_mode"] == "pairwise_logistic",
                f"{entry['arm']}/{entry['fold']}: ranking loss recipe differs")
        actual_rank = saved.get("extra_state", {}).get("ranking_loss_protocol", {})
        require(actual_rank == entry.get("ranking_loss_protocol")
                and actual_rank.get("name") == "train_only_sequence_pairwise_logistic_v1"
                and actual_rank.get("mode") == "pairwise_logistic"
                and float(actual_rank.get("weight", -1)) == float(entry["lambda"])
                and float(actual_rank.get("temperature", -1)) == .1
                and float(actual_rank.get("min_target_delta", -1)) == .05
                and actual_rank.get("target_source") ==
                    "folded_outer_training_soft_multi_only"
                and actual_rank.get("source_config_sha256") == sha256(Path(entry["config"])),
                f"{entry['arm']}/{entry['fold']}: actual ranking loss protocol differs")
    effective = dict(config)
    effective["model"] = saved["config"]
    with torch.random.fork_rng(devices=[]):
        model = GraphTransformer(build_model_config(effective)).cpu()
    model.load_state_dict(saved["model_state_dict"], strict=True)
    kernels, channels = node_conv_spec(saved["config"]["node_conv_mode"])
    require(int(model.four_state_mode_code.item()) == 4
            and int(model.node_conv_mode_code.item()) ==
                node_conv_mode_code(saved["config"]["node_conv_mode"])
            and model.node_conv_kernel_sizes.tolist() == list(kernels)
            and model.node_conv_branch_channels.tolist() == list(channels)
            and all(torch.isfinite(parameter).all().item()
                    for parameter in model.parameters()),
            f"{entry['arm']}/{entry['fold']}: four-state marker/nonfinite parameter")
    if entry["stage"] == "S12" and saved["config"]["stage12_fusion_mode"] != "legacy":
        from models.stage12_fusion import _MODE_CODES
        require(int(model.stage12_fusion_mode_code.item()) ==
                _MODE_CODES[saved["config"]["stage12_fusion_mode"]]
                and int(model.stage12_fusion_schema_code.item()) == 1,
                f"{entry['arm']}/{entry['fold']}: Stage12 fusion marker differs")
    count = sum(parameter.numel() for parameter in model.parameters())
    if entry.get("parameter_count") is not None:
        require(count == int(entry["parameter_count"]),
                f"{entry['arm']}/{entry['fold']}: parameter count changed")
    return {"strict_cpu_restore": True, "parameter_count": count,
            "checkpoint_epoch": saved.get("epoch"),
            "training_seed": int(entry["seed"]),
            "validation_split_seed": int(split_seed),
            "node_conv_mode": saved["config"]["node_conv_mode"],
            "stage12_fusion_mode": saved["config"].get("stage12_fusion_mode", "legacy")}


def _resource_proof(entry: dict) -> dict:
    """Independently check available formal per-fold timing and memory records."""
    if entry.get("stage") == "S14":
        path = Path(entry["training_log"]).resolve()
        _expected_binding(entry, "training_log", path)
        log = path.read_text(encoding="utf-8", errors="replace")
        training = [float(value) for value in re.findall(
            r"Training completed in ([0-9]+(?:\.[0-9]+)?)s", log)]
        test = [float(value) for value in re.findall(
            r"Final test evaluation took ([0-9]+(?:\.[0-9]+)?)s", log)]
        peaks = [float(value) for value in re.findall(
            r"Epoch [0-9]+ GPU Memory - [^\n]*?peak_allocated: "
            r"([0-9]+(?:\.[0-9]+)?)MB", log)]
        recorded = entry.get("resource_observation", {})
        require(len(training) == len(test) == 1 and peaks
                and recorded.get("source") == "per_fold_completed_training_log"
                and recorded.get("training_plus_final_evaluation_seconds") == training[0]
                and recorded.get("test_evaluation_and_export_seconds") == test[0]
                and recorded.get("test_inference_only_seconds") is None
                and recorded.get("epoch_peak_gpu_allocated_mb_max") == max(peaks)
                and recorded.get("epoch_peak_count") == len(peaks),
                f"{entry['arm']}/{entry['fold']}: formal resource record differs from log")
        return {"resources": {
            "total_training_seconds": training[0],
            "test_evaluation_and_export_seconds": test[0],
            "peak_gpu_allocated_mb": max(peaks)},
            "training_log_sha256": sha256(path),
            "resource_limitation": recorded.get("limitations", "")}
    if entry.get("stage") != "S13":
        return {}
    resources = entry.get("resources")
    require(isinstance(resources, dict),
            f"{entry['arm']}/{entry['fold']}: missing formal resource record")
    history_path = Path(resources["training_history"]).resolve()
    log_path = Path(entry["fold_log"]).resolve()
    require(history_path.is_file() and log_path.is_file()
            and sha256(history_path) == resources["training_history_sha256"]
            and sha256(log_path) == entry["fold_log_sha256"],
            f"{entry['arm']}/{entry['fold']}: training resource source SHA changed")
    with history_path.open(encoding="utf-8-sig", newline="") as stream:
        history = list(csv.DictReader(stream))
    selected = [row for row in history if row.get("epoch") == str(entry["best_epoch"])]
    require(len(selected) == 1,
            f"{entry['arm']}/{entry['fold']}: best epoch absent from history")
    log = log_path.read_text(encoding="utf-8", errors="replace")
    epoch_seconds = {int(epoch): float(seconds) for epoch, seconds in re.findall(
        r"Epoch\s+(\d+)\s+wall time:\s*([0-9]+(?:\.[0-9]+)?)s", log)}
    inference = [float(value) for value in re.findall(
        r"Final test evaluation took\s+([0-9]+(?:\.[0-9]+)?)s", log)]
    training = [float(value) for value in re.findall(
        r"Training completed in\s+([0-9]+(?:\.[0-9]+)?)s", log)]
    peaks = [float(value) for value in re.findall(
        r"peak_allocated:\s*([0-9]+(?:\.[0-9]+)?)MB", log)]
    require(len(epoch_seconds) == len(history) and len(inference) == len(training) == 1
            and peaks, f"{entry['arm']}/{entry['fold']}: incomplete formal resource log")
    actual = {
        "best_epoch_train_f1": float(selected[0]["train_f1"]),
        "best_epoch_val_f1": float(selected[0]["val_f1"]),
        "best_epoch_train_minus_val_f1":
            float(selected[0]["train_f1"]) - float(selected[0]["val_f1"]),
        "mean_epoch_wall_seconds": mean(epoch_seconds.values()),
        "test_inference_seconds": inference[0],
        "total_training_seconds": training[0],
        "peak_gpu_allocated_mb": max(peaks),
    }
    require(all(math.isfinite(value) and
                abs(value - float(resources[key])) < 1e-6
                for key, value in actual.items()),
            f"{entry['arm']}/{entry['fold']}: formal resource record differs from log/history")
    require({int(key): float(value) for key, value in
             resources["epoch_wall_seconds"].items()} == epoch_seconds,
            f"{entry['arm']}/{entry['fold']}: epoch timing record differs")
    return {"resources": {
                "total_training_seconds": actual["total_training_seconds"],
                "test_evaluation_and_export_seconds": actual["test_inference_seconds"],
                "peak_gpu_allocated_mb": actual["peak_gpu_allocated_mb"],
                "mean_epoch_wall_seconds": actual["mean_epoch_wall_seconds"],
                "best_epoch_train_f1": actual["best_epoch_train_f1"],
                "best_epoch_val_f1": actual["best_epoch_val_f1"],
                "best_epoch_train_minus_val_f1":
                    actual["best_epoch_train_minus_val_f1"]},
            "training_history_sha256": sha256(history_path),
            "fold_log_sha256": sha256(log_path)}


def audit_fold(entry: dict) -> tuple[dict, dict]:
    arm = str(entry["arm"])
    fold = str(entry["fold"])
    official = str(entry.get("stage", "")) in {"S12", "S13", "S14"}
    require(fold in FOLDS, f"Unknown fold: {fold}")
    paths = {key: Path(entry[key]).resolve() for key in
             ("pred_csv", "test_csv", "metric_csv", "checkpoint", "config")}
    for key, path in paths.items():
        require(path.is_file(), f"{arm}/{fold}: missing {key}: {path}")
        _expected_binding(entry, key, path)
    extra_bindings = {}
    if official:
        target_key = ("target_manifest" if "target_manifest" in entry
                      else "four_state_manifest")
        for key in ("train_csv", "val_csv", "validation_source_csv", target_key):
            require(isinstance(entry.get(key), str) and entry[key],
                    f"{arm}/{fold}: required {key} path missing")
            path = Path(entry[key]).resolve()
            require(path.is_file(), f"{arm}/{fold}: missing {key}: {path}")
            _expected_binding(entry, key, path)
            extra_bindings[key + "_sha256"] = sha256(path)
            if key == target_key:
                extra_bindings["target_sha256"] = sha256(path)

    max_seq_len = int(entry.get("max_seq_len", 512))
    if official:
        require("max_seq_len" in entry and max_seq_len == 32,
                f"{arm}/{fold}: missing or changed 32-residue source bound")
    source = _source_rows(paths["test_csv"], max_seq_len)
    split_proof = {}
    if "train_csv" in entry and "val_csv" in entry:
        split_paths = {name: Path(entry[name]).resolve() for name in ("train_csv", "val_csv")}
        for name, path in split_paths.items():
            require(path.is_file(), f"{arm}/{fold}: missing {name}")
            _expected_binding(entry, name, path)
        train_frame = pd.read_csv(split_paths["train_csv"])
        val_frame = pd.read_csv(split_paths["val_csv"])
        require("seq" in train_frame and "seq" in val_frame,
                f"{arm}/{fold}: split sequence column missing")
        sequence_sets = [set(frame["seq"].dropna().astype(str)) for frame in
                         (train_frame, val_frame, source)]
        require(not (sequence_sets[0] & sequence_sets[1] or
                     sequence_sets[0] & sequence_sets[2] or
                     sequence_sets[1] & sequence_sets[2]),
                f"{arm}/{fold}: train/validation/test sequence overlap")
        split_proof = {
            "train_csv_sha256": sha256(split_paths["train_csv"]),
            "val_csv_sha256": sha256(split_paths["val_csv"]),
            "train_rows": len(train_frame), "val_rows": len(val_frame),
            "test_rows": len(source),
            "train_exact_duplicate_rows": int(train_frame.duplicated().sum()),
            "val_exact_duplicate_rows": int(val_frame.duplicated().sum()),
            "test_exact_duplicate_rows": int(source.duplicated().sum()),
            "train_unique_sequences": len(sequence_sets[0]),
            "val_unique_sequences": len(sequence_sets[1]),
            "test_unique_sequences": len(sequence_sets[2]),
            "sequence_overlap": 0,
        }
    with paths["pred_csv"].open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == len(source) and rows, f"{arm}/{fold}: source/prediction row mismatch")
    require(all(row.get("prediction_export_source") == "same_pass_evaluate_logits" and
                row.get("prediction_probability_precision") ==
                "float32_roundtrip_9_significant_digits" for row in rows),
            f"{arm}/{fold}: predictions are not same-pass nine-digit export")
    evaluation_ids = {row.get("evaluation_id", "") for row in rows}
    require(len(evaluation_ids) == 1 and next(iter(evaluation_ids)),
            f"{arm}/{fold}: mixed evaluation IDs")

    flat_y: list[int] = []
    flat_p: list[float] = []
    flat_q: list[float] = []
    row_confusion: list[tuple[int, int, int]] = []
    conditions: OrderedDict[str, tuple[str, float, float]] = OrderedDict()
    repeated_condition_prediction_max_mean_delta = 0.0
    repeated_condition_rows = 0
    valid_bonds = 0
    for index, (row, original) in enumerate(zip(rows, source.to_dict("records"))):
        seq = str(original["seq"])
        require(str(row["seq"]) == seq, f"{arm}/{fold}: seq/order mismatch at {index}")
        require(abs(float(row["charge"]) - float(original["charge"])) < 1e-8 and
                abs(float(row["nce"]) - float(original["nce"])) < 1e-8,
                f"{arm}/{fold}: charge/NCE mismatch at {index}")
        require(Path(row["checkpoint_path"]).resolve() == paths["checkpoint"] and
                float(row["threshold"]) == .5,
                f"{arm}/{fold}: checkpoint or threshold mismatch at {index}")
        expected_len = len(seq) - 1
        y = vector(row["true"], int)
        pred = vector(row["pred"], int)
        q = vector(row["soft_multi"])
        p = vector(row["pred_prob"])
        source_y = vector(original["true_multi"], int)
        source_q = vector(original["soft_multi"])
        require(all(len(array) == expected_len for array in
                    (y, pred, q, p, source_y, source_q)),
                f"{arm}/{fold}: padding or missing bonds at row {index}")
        require(y == source_y and y == vector(row["true_multi"], int),
                f"{arm}/{fold}: realized labels mismatch at row {index}")
        require(np.allclose(q, source_q, rtol=0, atol=1e-7),
                f"{arm}/{fold}: q labels mismatch at row {index}")
        require(all(math.isfinite(v) and 0 <= v <= 1 for v in (*p, *q)),
                f"{arm}/{fold}: invalid probabilities at row {index}")
        tokens = str(row["pred_prob"]).split(";") if expected_len else []
        require(all(format(float(np.float32(float(token))), ".9g") == token
                    for token in tokens),
                f"{arm}/{fold}: noncanonical float32 export at row {index}")
        require(all(value == int(prob > .5) for value, prob in zip(pred, p)),
                f"{arm}/{fold}: threshold decision changed by export at row {index}")
        yp = np.asarray(pred, dtype=int)
        yt = np.asarray(y, dtype=int)
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        row_confusion.append((tp, fp, fn))
        flat_y.extend(y)
        flat_p.extend(p)
        flat_q.extend(q)
        valid_bonds += expected_len
        # Group-level ranking uses the first repeated condition and then the
        # mean of unique conditions within each sequence, exactly as q_seq.
        key = f"{seq}|{float(row['charge']):g}|{float(row['nce']):g}"
        if key not in conditions:
            conditions[key] = (seq, float(np.mean(p)), float(np.mean(q)))
        else:
            old_seq, old_p, old_q = conditions[key]
            require(old_seq == seq and abs(old_q - float(np.mean(q))) <= 1e-7,
                    f"{arm}/{fold}: repeated condition q target differs")
            # The legacy q_seq metric deliberately keeps the first condition
            # prediction, even though same-condition spectra sometimes have
            # small prediction differences. Preserve that exact ranking rule
            # and expose the variability as a provenance diagnostic.
            repeated_condition_rows += 1
            repeated_condition_prediction_max_mean_delta = max(
                repeated_condition_prediction_max_mean_delta,
                abs(old_p - float(np.mean(p))))

    y_arr = np.asarray(flat_y, dtype=int)
    p_arr = np.asarray(flat_p, dtype=float)
    q_arr = np.asarray(flat_q, dtype=float)
    yp_arr = (p_arr > .5).astype(int)
    require(valid_bonds == len(y_arr) and valid_bonds > 0,
            f"{arm}/{fold}: invalid valid-bond count")
    per_row_p = [tp / (tp + fp) if tp + fp else 0.0 for tp, fp, _ in row_confusion]
    per_row_r = [tp / (tp + fn) if tp + fn else 0.0 for tp, _, fn in row_confusion]
    spectrum_p = float(np.mean(per_row_p))
    spectrum_r = float(np.mean(per_row_r))
    spectrum_f1 = (2 * spectrum_p * spectrum_r / (spectrum_p + spectrum_r)
                   if spectrum_p + spectrum_r else 0.0)
    sequence_scores: OrderedDict[str, list[tuple[float, float]]] = OrderedDict()
    for seq, score, truth in conditions.values():
        sequence_scores.setdefault(seq, []).append((score, truth))
    score_seq = np.asarray([np.mean([pair[0] for pair in pairs])
                            for pairs in sequence_scores.values()], dtype=float)
    truth_seq = np.asarray([np.mean([pair[1] for pair in pairs])
                            for pairs in sequence_scores.values()], dtype=float)
    q_seq, top10, top20 = _ranking(score_seq, truth_seq)
    unique_y = np.unique(y_arr)
    require(len(unique_y) == 2, f"{arm}/{fold}: ROC/PR undefined with one class")
    metrics = {
        "bond_f1": float(f1_score(y_arr, yp_arr, zero_division=0)),
        "bond_accuracy": float(accuracy_score(y_arr, yp_arr)),
        "roc_auc": float(roc_auc_score(y_arr, p_arr)),
        "pr_auc": float(average_precision_score(y_arr, p_arr)),
        "mcc": float(matthews_corrcoef(y_arr, yp_arr)),
        "spectrum_precision": spectrum_p,
        "spectrum_recall": spectrum_r,
        "spectrum_f1_harmonic": spectrum_f1,
        "q_brier": float(np.mean((p_arr - q_arr) ** 2)),
        "ece_15": _ece_equal_width(y_arr, p_arr),
        "q_seq": q_seq,
        "top10_enrichment": top10,
        "top20_enrichment": top20,
    }
    producer = _producer_metrics(paths["metric_csv"])
    tolerance = {"roc_auc": 2e-4, "q_seq": 5e-6, "top10_enrichment": 5e-6,
                 "top20_enrichment": 5e-6}
    differences = {}
    for ours, theirs in PRODUCER_NAMES.items():
        require(theirs in producer, f"{arm}/{fold}: producer missing {theirs}")
        differences[ours] = abs(metrics[ours] - producer[theirs])
        require(differences[ours] <= tolerance.get(ours, 2e-5),
                f"{arm}/{fold}: {ours} disagrees with producer by {differences[ours]:.3g}")
    return ({"stage": entry.get("stage", ""), "arm": arm, "fold": fold,
             "seed": str(entry.get("seed", "")), **metrics},
            {"arm": arm, "fold": fold, "rows": len(rows),
             "valid_bonds": valid_bonds, "unique_sequences": len(sequence_scores),
             "unique_conditions": len(conditions),
             "repeated_condition_rows": repeated_condition_rows,
             "repeated_condition_prediction_max_mean_delta":
                 repeated_condition_prediction_max_mean_delta,
             "source_sha256": sha256(paths["test_csv"]),
             "prediction_sha256": sha256(paths["pred_csv"]),
             "checkpoint_sha256": sha256(paths["checkpoint"]),
             "config_sha256": sha256(paths["config"]),
             "producer_metric_sha256": sha256(paths["metric_csv"]),
             "producer_differences": differences,
             "evaluation_id": next(iter(evaluation_ids)), **split_proof,
             **extra_bindings})


def _summarize(rows: list[dict], baseline: str) -> tuple[list[dict], list[dict]]:
    indexed = {(str(row["arm"]), str(row["fold"])): row for row in rows}
    arms = sorted({str(row["arm"]) for row in rows})
    require(baseline in arms, f"Missing baseline arm {baseline}")
    require(len(indexed) == len(rows), "Duplicate arm/fold")
    require(set(indexed) == {(arm, fold) for arm in arms for fold in FOLDS},
            "Incomplete five-fold arm matrix")
    summary, paired = [], []
    for arm in arms:
        for metric in METRICS:
            vals = [float(indexed[arm, fold][metric]) for fold in FOLDS]
            summary.append({"arm": arm, "metric": metric, "mean": mean(vals),
                            "sample_sd": stdev(vals)})
            if arm != baseline:
                diffs = [float(indexed[arm, fold][metric]) -
                         float(indexed[baseline, fold][metric]) for fold in FOLDS]
                sign = -1 if metric in LOWER_BETTER else 1
                paired.append({"arm": arm, "baseline": baseline, "metric": metric,
                               "mean_delta": mean(diffs), "paired_sample_sd": stdev(diffs),
                               "wins": sum(sign * value > 1e-12 for value in diffs),
                               "ties": sum(abs(value) <= 1e-12 for value in diffs),
                               "losses": sum(sign * value < -1e-12 for value in diffs),
                               **{f"delta_{fold}": diffs[i] for i, fold in enumerate(FOLDS)}})
    return summary, paired


def _write_csv(path: Path, rows: list[dict],
               empty_fieldnames: list[str] | None = None) -> None:
    require(rows or empty_fieldnames,
            f"Refuse to write CSV without fields: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else empty_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _engineering_decisions(stage: str, paired: list[dict]) -> dict:
    require(stage in {"S12", "S13", "S14", "historical_stage11_single_control"},
            f"Unknown stage cannot pass engineering gates: {stage}")
    by_arm: dict[str, dict[str, dict]] = {}
    for row in paired:
        by_arm.setdefault(row["arm"], {})[row["metric"]] = row
    decisions = {}
    for arm, values in by_arm.items():
        if stage in {"S12", "S13"}:
            checks = {
                "bond_f1_delta_at_least_0.002": values["bond_f1"]["mean_delta"] >= .002,
                "bond_f1_improved_at_least_4_of_5": values["bond_f1"]["wins"] >= 4,
                "roc_auc_drop_at_most_0.001": values["roc_auc"]["mean_delta"] >= -.001,
                "q_brier_worsening_at_most_0.001": values["q_brier"]["mean_delta"] <= .001,
            }
        elif stage == "S14":
            checks = {
                "q_seq_delta_at_least_0.01": values["q_seq"]["mean_delta"] >= .01,
                "q_seq_improved_at_least_4_of_5": values["q_seq"]["wins"] >= 4,
                "top10_delta_at_least_0.01": values["top10_enrichment"]["mean_delta"] >= .01,
                "bond_f1_drop_at_most_0.002": values["bond_f1"]["mean_delta"] >= -.002,
                "spectrum_f1_drop_at_most_0.002": values["spectrum_f1_harmonic"]["mean_delta"] >= -.002,
                "q_brier_worsening_at_most_0.001": values["q_brier"]["mean_delta"] <= .001,
            }
        else:
            checks = {}
        decisions[arm] = {"checks": checks, "passes_all": all(checks.values()) if checks else None}
    return decisions


def _stage_report(manifest: dict, summary: list[dict], paired: list[dict],
                  decisions: dict, provenance: list[dict], baseline: str) -> str:
    stage = str(manifest.get("stage", "unknown"))
    arms = sorted({row["arm"] for row in summary})
    summaries = {(row["arm"], row["metric"]): row for row in summary}
    lines = [f"# {stage} 同轮实验独立审计报告", "",
             f"- 批次：`{manifest.get('root', '')}`",
             f"- 冻结源码：`{manifest.get('source_commit', '')}`",
             f"- 配对基线：`{baseline}`；五个固定外部分组；每组训练 seed 见 `all_metrics.csv`。",
             "- 预测来自同一次正式评估，固定阈值 0.5，有效键严格按序列长度去 padding。",
             "- PR-AUC 定义为 average precision；ECE 使用真实 0/1 键标签与 15 个等宽概率箱。",
             "- Spectrum F1 为谱图平均 Precision 与 Recall 的调和均值。",
             "- q_seq/Top10/Top20 对重复的 (序列、电荷、NCE) 条件保留首行预测，再对每条序列的唯一条件均匀平均；重复行的预测均值可能略有差异，见审计回执。", "",
             "## 五组均值 ± 样本 SD", "",
             "所有指标保留原始 0–1 比例单位，Top10/Top20 是富集倍数。", "",
             "| 指标 | " + " | ".join(arms) + " |", "|---|" + "---:|" * len(arms)]
    for metric in METRICS:
        cells = [summaries[arm, metric] for arm in arms]
        lines.append("| " + metric + " | " + " | ".join(
            f"{cell['mean']:.6f} ± {cell['sample_sd']:.6f}" for cell in cells) + " |")
    lines += ["", "## 对同轮基线的配对差", "",
              "| 候选 | 指标 | 平均差 ± 配对样本 SD | 改善/持平/下降 |",
              "|---|---|---:|---:|"]
    report_metrics = ("bond_f1", "spectrum_f1_harmonic", "roc_auc", "pr_auc",
                      "q_brier", "q_seq", "top10_enrichment", "top20_enrichment")
    for row in paired:
        if row["metric"] in report_metrics:
            lines.append(f"| {row['arm']} | {row['metric']} | "
                         f"{row['mean_delta']:+.6f} ± {row['paired_sample_sd']:.6f} | "
                         f"{row['wins']}/{row['ties']}/{row['losses']} |")
    lines += ["", "## 预设工程门槛", ""]
    for arm, decision in decisions.items():
        lines.append(f"- `{arm}`：{'全部通过' if decision['passes_all'] else '未全部通过'}；" +
                     "；".join(f"{name}={'通过' if okay else '未过'}"
                              for name, okay in decision["checks"].items()))
    aliases = manifest.get("aliases", [])
    if aliases:
        lines += ["", "## 等价对照", ""]
        for alias in aliases:
            lines.append(f"- `{alias['arm']}` 与 `{alias['alias_of']}` 是同一模型与同一组训练权重；不作为新的独立训练样本。{alias['reason']}")
    lines += ["", "## 模型与资源", "",
              "| 模型 | 参数量 | 训练及最终评估秒数均值 ± SD | 测试评估及导出秒数均值 ± SD | 训练期 GPU 峰值 MB 均值 ± SD |",
              "|---|---:|---:|---:|---:|"]
    for arm in arms:
        proofs = [item for item in provenance if item["arm"] == arm]
        counts = {item.get("parameter_count") for item in proofs}
        require(len(counts) == 1, f"{arm}: inconsistent parameter count")
        count_label = (str(next(iter(counts))) if next(iter(counts)) is not None
                       else "历史只读参考")
        resource_cells = []
        for key in ("total_training_seconds", "test_evaluation_and_export_seconds",
                    "peak_gpu_allocated_mb"):
            values = [item["resources"][key] for item in proofs
                      if "resources" in item]
            resource_cells.append(f"{mean(values):.3f} ± {stdev(values):.3f}"
                                  if len(values) == len(proofs) else "未有完整绑定")
        lines.append(f"| {arm} | {count_label} | "
                     + " | ".join(resource_cells) + " |")
    lines += ["", "## 来源与边界", "",
              f"- {len(provenance)} 份权重对应的原始预测/配置/测试数据 SHA 已在 `audit_receipt.json` 逐一记录。",
              "- 训练、内部验证与外部测试的序列集合逐折不交叉；精确重复行数量也在审计回执中记录。",
              "- 五个外部分组已被多轮研发使用，且分组之间重叠；本报告属于开发性评估，不是独立确认性证据。",
              "- 工程门槛不是显著性检验；不以外部测试结果重新选择 epoch、阈值或训练 seed。",
              "- `all_metrics.csv`、`paired_comparison.csv` 给出全部逐折数值。", ""]
    return "\n".join(lines)


def _stage_dashboard(manifest: dict, summary: list[dict], paired: list[dict],
                     decisions: dict, baseline: str) -> str:
    payload = json.dumps({"stage": manifest.get("stage"), "root": manifest.get("root"),
                          "baseline": baseline, "summary": summary, "paired": paired,
                          "decisions": decisions,
                          "aliases": manifest.get("aliases", [])},
                         ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(str(manifest.get("stage", "Stage")))
    return """<!doctype html><html lang="zh"><meta charset="utf-8"><title>""" + title + """ 独立审计</title>
<style>body{font:15px/1.5 system-ui,sans-serif;background:#f5f7fb;color:#182238;margin:0;padding:26px}main{max-width:1150px;margin:auto}.card{background:white;padding:22px;border-radius:16px;margin:16px 0;box-shadow:0 3px 18px #12203a13}h1{margin:0}small{color:#5e6878}table{border-collapse:collapse;width:100%;margin:12px 0}td,th{padding:8px;border-bottom:1px solid #dce3ee;text-align:right}td:first-child,th:first-child{text-align:left}select{padding:7px;border-radius:7px;border:1px solid #bec9d9}.positive{color:#116c4d}.negative{color:#a02a34}svg{max-width:100%;height:auto}code{overflow-wrap:anywhere}</style>
<main><div class="card"><h1>""" + title + """：同轮配对审计</h1><p>基线 <b id="base"></b> · 五组固定序列划分 · seed42 首轮</p><small id="source"></small></div>
<div class="card"><label>指标 <select id="metric"></select></label><div id="summary"></div><svg id="chart" viewBox="0 0 1060 380" role="img" aria-label="五组配对差"></svg></div>
<div class="card"><h2>预设工程门槛</h2><div id="gates"></div><small>改善折数按各指标有利方向统计；q-Brier/ECE 越低越好。此为开发性评估，非独立确认性测试。</small></div><div class="card" id="aliases" hidden></div></main>
<script type="application/json" id="data">""" + payload + """</script><script>
const d=JSON.parse(document.querySelector('#data').textContent),s=document.querySelector('#metric');
document.querySelector('#base').textContent=d.baseline;document.querySelector('#source').textContent='批次：'+d.root;
for(const m of [...new Set(d.summary.map(x=>x.metric))]){const o=document.createElement('option');o.value=o.textContent=m;s.append(o)}
if(d.aliases.length){const box=document.querySelector('#aliases');box.hidden=false;const h=document.createElement('h2');h.textContent='等价对照';box.append(h);for(const a of d.aliases){const p=document.createElement('p');p.textContent=`${a.arm} = ${a.alias_of}，同一组训练权重，不作为新的独立样本。${a.reason}`;box.append(p)}}
function render(){const m=s.value,mean=d.summary.filter(x=>x.metric===m),diff=d.paired.filter(x=>x.metric===m),folds=['1222','2252','3514','6072','9075'];let t='<table><tr><th>模型</th><th>均值</th><th>样本 SD</th></tr>';for(const x of mean)t+=`<tr><td>${x.arm}</td><td>${x.mean.toFixed(6)}</td><td>${x.sample_sd.toFixed(6)}</td></tr>`;t+='</table><table><tr><th>相对基线</th><th>平均差</th><th>配对 SD</th><th>改善/持平/下降</th></tr>';for(const x of diff)t+=`<tr><td>${x.arm}</td><td>${x.mean_delta.toFixed(6)}</td><td>${x.paired_sample_sd.toFixed(6)}</td><td>${x.wins}/${x.ties}/${x.losses}</td></tr>`;document.querySelector('#summary').innerHTML=t+'</table>';const svg=document.querySelector('#chart');while(svg.firstChild)svg.firstChild.remove();const all=diff.flatMap(x=>folds.map(f=>x['delta_'+f])),span=Math.max(...all.map(Math.abs),1e-6),ns='http://www.w3.org/2000/svg';function el(tag,attrs,txt){const x=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))x.setAttribute(k,v);if(txt)x.textContent=txt;svg.append(x)}el('line',{x1:530,y1:20,x2:530,y2:350,stroke:'#70809a','stroke-width':2});diff.forEach((x,i)=>{el('text',{x:18,y:70+i*82,'font-size':16},x.arm);folds.forEach((f,j)=>{const v=x['delta_'+f],x0=530,x1=530+v/span*430,y=36+i*82+j*11;el('line',{x1:x0,y1:y,x2:x1,y2:y,stroke:v>=0?'#187d61':'#bf4554','stroke-width':7,'stroke-linecap':'round'});el('text',{x:975,y:y+4,'font-size':11},f)})})}s.onchange=render;render();let g='';for(const[a,v]of Object.entries(d.decisions)){g+=`<p><b>${a}</b>：<span class="${v.passes_all?'positive':'negative'}">${v.passes_all?'全部通过':'未全部通过'}</span></p><ul>`;for(const[k,ok]of Object.entries(v.checks))g+=`<li>${k}: ${ok?'通过':'未过'}</li>`;g+='</ul>'}document.querySelector('#gates').innerHTML=g;
</script></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", default=None)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_root, run_binding = _manifest_binding(manifest, args.manifest)
    entries = manifest["arms"]
    require(isinstance(entries, list) and entries, "Manifest has no arm/fold entries")
    baseline = args.baseline or manifest.get("baseline_arm") or str(entries[0]["arm"])
    args.out.mkdir(parents=True, exist_ok=True)
    metrics, provenance = [], []
    for entry in entries:
        item = {**entry, "stage": manifest.get("stage", "")}
        row, proof = audit_fold(item)
        if source_root is not None:
            proof.update(_strict_checkpoint(item, source_root))
            proof.update(_resource_proof(item))
        metrics.append(row)
        provenance.append(proof)
    arms = sorted({str(entry["arm"]) for entry in entries})
    for fold in FOLDS:
        paired = [entry for entry in provenance if entry["fold"] == fold]
        require(len(paired) == len(arms),
                f"Fold {fold}: expected one complete run per audited arm")
        keys = ["source_sha256"]
        if source_root is not None:
            keys += ["train_csv_sha256", "val_csv_sha256",
                     "validation_source_csv_sha256", "target_sha256",
                     "training_seed", "validation_split_seed"]
        for key in keys:
            present = [entry[key] for entry in paired if key in entry]
            require(len(present) == len(paired) and len(set(present)) == 1,
                    f"Fold {fold}: same-round arms differ in {key}")
    aliases = manifest.get("aliases", [])
    require(isinstance(aliases, list), "Malformed alias list")
    if str(manifest.get("stage")) == "S13":
        require(set(arms) == {"S13-A", "S13-C"}
                and aliases == []
                and {entry["parameter_count"] for entry in provenance
                     if entry["arm"] == "S13-A"} ==
                    {entry["parameter_count"] for entry in provenance
                     if entry["arm"] == "S13-C"},
                "Reduced S13 A/C arm set or parameter matching differs")
        expected_modes = {'S13-A': 'single_k5', 'S13-C': 'multiscale_357'}
        preserved_root = Path(manifest['preserved_baseline']['root']).resolve()
        candidate_root = Path(manifest['root']).resolve()
        for entry, proof in zip(entries, provenance):
            arm = entry['arm']
            expected_root = preserved_root if arm == 'S13-A' else candidate_root
            require(proof['node_conv_mode'] == expected_modes[arm],
                    f'{arm}/{entry["fold"]}: audited checkpoint mode differs')
            for path_key in ('config', 'checkpoint', 'pred_csv', 'metric_csv', 'fold_log'):
                path = Path(entry[path_key]).resolve()
                require(expected_root in path.parents,
                        f'{arm}/{entry["fold"]}: {path_key} escaped its bound batch root')
    summary, paired = _summarize(metrics, baseline)
    decisions = _engineering_decisions(str(manifest.get("stage", "")), paired)
    _write_csv(args.out / "all_metrics.csv", metrics)
    _write_csv(args.out / "summary_metrics.csv", summary)
    _write_csv(args.out / "paired_comparison.csv", paired,
               ["arm", "baseline", "metric", "mean_delta", "paired_sample_sd",
                "wins", "ties", "losses", *[f"delta_{fold}" for fold in FOLDS]])
    (args.out / "stage_report.md").write_text(
        _stage_report(manifest, summary, paired, decisions, provenance, baseline),
        encoding="utf-8")
    (args.out / "stage_dashboard.html").write_text(
        _stage_dashboard(manifest, summary, paired, decisions, baseline),
        encoding="utf-8")
    (args.out / "audit_receipt.json").write_text(json.dumps({
        "status": "audited", "stage": manifest.get("stage"),
        "root": manifest.get("root"),
        "ops": manifest.get("ops"),
        "source_commit": manifest.get("source_commit"),
        "manifest_sha256": sha256(args.manifest), "baseline_arm": baseline,
        "engineering_decisions": decisions,
        "run_binding": run_binding,
        "failed_arms": manifest.get("failed_arms", []),
        "aliases": aliases,
        "metric_definitions": {
            "spectrum_f1_harmonic": "Harmonic mean of per-spectrum mean P and mean R",
            "q_seq": "Spearman after unique (sequence,charge,NCE) conditions then sequence-uniform averaging",
            "top10_top20": "Top K percent true q average divided by all-sequence true q average",
            "ece_15": "Realized bond-label ECE over 15 equal-width probability bins",
            "pr_auc": "Average precision (stepwise precision-recall area)",
            "threshold": "Fixed p > 0.5, padding removed by exact sequence length minus one",
        },
        "fold_provenance": provenance,
        "warning": "Five overlapping outer folds and one training initialization are exploratory, not independent confirmation",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "audited", "stage": manifest.get("stage"),
                      "fold_runs": len(metrics), "baseline": baseline,
                      "output": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
