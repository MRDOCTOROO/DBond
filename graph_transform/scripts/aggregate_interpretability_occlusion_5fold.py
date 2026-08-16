#!/usr/bin/env python3
"""Run and aggregate repeated split interpretability and occlusion analyses.

The split manifest is intentionally explicit: a checkpoint, config and test CSV
must belong to the same split.  This prevents accidentally pairing a model from
one split with data from another split.  The splits are treated as repeated
analysis units, not as strict cross-validation folds.  In particular, results
from overlapping splits are never pooled as independent observations.

Example:
    python graph_transform/scripts/aggregate_interpretability_occlusion_5fold.py \
        --split_manifest graph_transform/config/interpretability_splits.json \
        --output_root results/interpretability_occlusion_repeated_splits \
        --infer_config

    The script produces one directory per split, JSON/CSV aggregation tables, and a
    representative split selected by a prespecified median-distance rule.  It does
not average SVG files: representative figures are copied from one complete,
paired fold, while cross-fold numerical summaries are written separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import shutil
import subprocess
import sys
from statistics import mean, median, stdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_logging(output_root: str) -> logging.Logger:
    os.makedirs(output_root, exist_ok=True)
    logger = logging.getLogger("interpretability_occlusion_5fold")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(
        os.path.join(output_root, "run.log"), encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)


def summarize_values(values: Iterable[Any]) -> Dict[str, Any]:
    numbers = [number for number in (finite_number(v) for v in values) if number is not None]
    if not numbers:
        return {"n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
    return {
        "n": len(numbers),
        "mean": mean(numbers),
        "std": stdev(numbers) if len(numbers) > 1 else 0.0,
        "median": median(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }


def weighted_mean(pairs: Iterable[Tuple[Any, Any]]) -> Optional[float]:
    numerator = 0.0
    denominator = 0.0
    for value, weight in pairs:
        number = finite_number(value)
        count = finite_number(weight)
        if number is not None and count is not None and count > 0:
            numerator += number * count
            denominator += count
    return numerator / denominator if denominator else None


def get_path(mapping: Dict[str, Any], path: Sequence[str]) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def add_common_args(command: List[str], args: argparse.Namespace) -> None:
    command.extend(["--bootstrap_iters", str(args.bootstrap_iters)])
    command.extend(["--random_seed", str(args.random_seed)])
    command.extend(["--figure_format", args.figure_format])
    if args.max_seq_len is not None:
        command.extend(["--max_seq_len", str(args.max_seq_len)])
    if args.device:
        command.extend(["--device", args.device])
    if args.infer_config:
        command.append("--infer_config")


def run_command(command: List[str], logger: logging.Logger) -> None:
    logger.info("Running: %s", " ".join(command))
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        logger.error(result.stdout[-4000:])
        logger.error(result.stderr[-4000:])
        raise RuntimeError("Analysis command failed with code %s" % result.returncode)
    if result.stdout:
        logger.info(result.stdout[-1500:])


def validate_manifest(manifest: Any) -> List[Dict[str, str]]:
    if isinstance(manifest, dict):
        folds = manifest.get("splits", manifest.get("folds"))
    else:
        folds = manifest
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError("split_manifest must contain exactly five split entries")
    required = ("id", "config", "checkpoint", "input_csv")
    normalized = []
    for entry in folds:
        if not isinstance(entry, dict) or any(key not in entry for key in required):
            raise ValueError("Each fold requires id, config, checkpoint and input_csv")
        normalized.append({key: str(entry[key]) for key in required})
    ids = [entry["id"] for entry in normalized]
    if len(set(ids)) != len(ids):
        raise ValueError("Split ids must be unique")
    return normalized


def run_fold(
    fold: Dict[str, str],
    args: argparse.Namespace,
    output_root: str,
    logger: logging.Logger,
) -> Dict[str, str]:
    fold_root = os.path.join(output_root, "per_split", "split_%s" % fold["id"])
    interpretability_root = os.path.join(fold_root, "interpretability")
    occlusion_root = os.path.join(fold_root, "occlusion")
    os.makedirs(interpretability_root, exist_ok=True)
    os.makedirs(occlusion_root, exist_ok=True)

    interpretability_summary = os.path.join(
        interpretability_root, "interpretability_summary.json"
    )
    occlusion_summary = os.path.join(occlusion_root, "occlusion_summary.json")

    interpretability_command = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "interpretability_analysis.py"),
        "--config", fold["config"],
        "--checkpoint", fold["checkpoint"],
        "--input_csv", fold["input_csv"],
        "--output_dir", interpretability_root,
        "--num_samples", str(args.num_case_samples),
        "--num_stat_samples", str(args.num_stat_samples),
        "--attention_mode", args.attention_mode,
        "--heatmap_normalize", args.heatmap_normalize,
    ]
    add_common_args(interpretability_command, args)

    occlusion_command = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "occlusion_analysis.py"),
        "--config", fold["config"],
        "--checkpoint", fold["checkpoint"],
        "--input_csv", fold["input_csv"],
        "--output_dir", occlusion_root,
        "--num_samples", str(args.num_occlusion_samples),
        "--num_case_figures", str(args.num_case_figures),
        "--attention_layer", str(args.attention_layer),
    ]
    add_common_args(occlusion_command, args)

    if not args.skip_existing or not os.path.exists(interpretability_summary):
        run_command(interpretability_command, logger)
    else:
        logger.info("Skipping existing interpretability output for split %s", fold["id"])
    if not args.skip_existing or not os.path.exists(occlusion_summary):
        run_command(occlusion_command, logger)
    else:
        logger.info("Skipping existing occlusion output for split %s", fold["id"])

    for path in (interpretability_summary, occlusion_summary):
        if not os.path.exists(path):
            raise FileNotFoundError("Missing expected output: %s" % path)
    return {
        "id": fold["id"],
        "config": fold["config"],
        "checkpoint": fold["checkpoint"],
        "input_csv": fold["input_csv"],
        "interpretability_dir": interpretability_root,
        "occlusion_dir": occlusion_root,
    }


def aggregate_interpretability(fold_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    summaries = [record["interpretability"] for record in fold_records]
    layer_count = max(
        len(summary.get("layer_trend", {}).get("statistical", []))
        for summary in summaries
    )
    layers = []
    for index in range(layer_count):
        entries = [
            summary.get("layer_trend", {}).get("statistical", [])[index]
            for summary in summaries
            if len(summary.get("layer_trend", {}).get("statistical", [])) > index
        ]
        layer_summary = {"layer_index": index}
        for metric in ("pearson_r", "abs_r", "spearman_r", "auc", "separation_auc", "n_samples"):
            layer_summary[metric] = summarize_values(entry.get(metric) for entry in entries)
        layers.append(layer_summary)

    effect_paths = [
        "effect_size.statistical.cohen_d_signed",
        "effect_size.statistical.cohen_d_abs",
        "effect_size.statistical.auc",
        "effect_size.statistical.p_value",
        "effect_size.statistical.n_broken",
        "effect_size.statistical.n_intact",
    ]
    effects = {}
    for path in effect_paths:
        values = [get_path(summary, path.split(".")) for summary in summaries]
        effects[path.split(".")[-1]] = summarize_values(values)

    grouped: Dict[str, Any] = {}
    group_types = set()
    for summary in summaries:
        group_types.update(summary.get("grouped_robustness", {}).keys())
    for group_type in sorted(group_types):
        names = set()
        for summary in summaries:
            names.update(summary.get("grouped_robustness", {}).get(group_type, {}).get("groups", {}).keys())
        grouped[group_type] = {}
        for name in sorted(names):
            entries = [
                summary.get("grouped_robustness", {}).get(group_type, {}).get("groups", {}).get(name, {})
                for summary in summaries
            ]
            grouped[group_type][name] = {
                metric: summarize_values(entry.get(metric) for entry in entries)
                for metric in ("n", "mean_abs_r", "mean_signed_r", "ci_low", "ci_high")
            }

    return {
        "method": "unweighted split-level mean/std; splits may overlap",
        "num_splits": len(summaries),
        "layers": layers,
        "effect_size": effects,
        "grouped_robustness": grouped,
        "n_stat_samples_per_split": summarize_values(
            summary.get("sample_selection", {}).get("n_stat_samples") for summary in summaries
        ),
    }


def aggregate_occlusion(fold_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    summaries = [record["occlusion"] for record in fold_records]
    aggregate_paths = {
        "global_pearson_r": "aggregate.global_pearson_r",
        "global_spearman_rho": "aggregate.global_spearman_rho",
        "mean_per_sample_r": "mean_per_sample_pearson_r",
        "median_per_sample_r": "aggregate.median_per_sample_r",
        "n_valid": "aggregate.n_valid",
        "computation_cost_sec": "computation_cost_sec",
    }
    values = {
        name: summarize_values(get_path(summary, path.split(".")) for summary in summaries)
        for name, path in aggregate_paths.items()
    }

    grouped: Dict[str, Any] = {}
    group_types = set()
    for summary in summaries:
        group_types.update(summary.get("grouped_robustness", {}).keys())
    for group_type in sorted(group_types):
        names = set()
        for summary in summaries:
            names.update(summary.get("grouped_robustness", {}).get(group_type, {}).get("groups", {}).keys())
        grouped[group_type] = {}
        for name in sorted(names):
            entries = [
                summary.get("grouped_robustness", {}).get(group_type, {}).get("groups", {}).get(name, {})
                for summary in summaries
            ]
            grouped[group_type][name] = {
                metric: summarize_values(entry.get(metric) for entry in entries)
                for metric in ("n", "mean", "median", "ci_low", "ci_high")
            }

    return {
        "method": "unweighted split-level mean/std; per-sample pooling disabled because splits may overlap",
        "num_splits": len(summaries),
        "split_level": values,
        "grouped_robustness": grouped,
    }


def representative_fold(fold_records: List[Dict[str, Any]]) -> Tuple[str, Dict[str, float]]:
    metric_paths = [
        ("interpretability_final_abs_r", "interpretability_final_abs_r"),
        ("interpretability_final_signed_r", "interpretability_final_signed_r"),
        ("interpretability_cohen_d", "interpretability.effect_size.statistical.cohen_d_signed"),
        ("occlusion_mean_r", "occlusion.mean_per_sample_pearson_r"),
        ("occlusion_global_r", "occlusion.aggregate.global_pearson_r"),
    ]

    def resolve(record: Dict[str, Any], path: str) -> Optional[float]:
        if path == "interpretability_final_abs_r":
            entries = record["interpretability"].get("layer_trend", {}).get("statistical", [])
            return finite_number(entries[-1].get("abs_r")) if entries else None
        if path == "interpretability_final_signed_r":
            entries = record["interpretability"].get("layer_trend", {}).get("statistical", [])
            return finite_number(entries[-1].get("pearson_r")) if entries else None
        parts = path.split(".")
        return finite_number(get_path(record, parts))

    medians = {
        name: median(
            values
        )
        for name, path in metric_paths
        for values in [[value for value in (resolve(record, path) for record in fold_records) if value is not None]]
        if values
    }
    ranges = {}
    for name, path in metric_paths:
        values = [value for value in (resolve(record, path) for record in fold_records) if value is not None]
        ranges[name] = max(max(values) - min(values), 1e-8) if values else 1.0

    scores = {}
    for record in fold_records:
        score = 0.0
        used = 0
        for name, path in metric_paths:
            value = resolve(record, path)
            if value is not None and name in medians:
                score += abs(value - medians[name]) / ranges[name]
                used += 1
        scores[record["id"]] = score / used if used else float("inf")
    selected = min(scores, key=scores.get)
    return selected, scores


def copy_representative_figures(record: Dict[str, Any], output_root: str) -> None:
    destination_root = os.path.join(output_root, "representative")
    for analysis_key, source_key in (
        ("interpretability", "interpretability_dir"),
        ("occlusion", "occlusion_dir"),
    ):
        destination = os.path.join(destination_root, analysis_key)
        os.makedirs(destination, exist_ok=True)
        source = record[source_key]
        for filename in os.listdir(source):
            if filename.endswith((".svg", ".pdf", ".png")):
                shutil.copy2(os.path.join(source, filename), os.path.join(destination, filename))


def write_metric_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flatten_aggregate_metrics(aggregate: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert nested aggregate JSON into a manuscript-table-friendly CSV."""
    rows: List[Dict[str, Any]] = []

    def add(scope: str, metric: str, summary: Dict[str, Any]) -> None:
        rows.append({
            "analysis": scope,
            "metric": metric,
            "n": summary.get("n"),
            "mean": summary.get("mean"),
            "std": summary.get("std"),
            "median": summary.get("median"),
            "min": summary.get("min"),
            "max": summary.get("max"),
        })

    interpretation = aggregate["interpretability"]
    for layer in interpretation.get("layers", []):
        layer_id = "Layer_%s" % layer["layer_index"]
        for metric in ("pearson_r", "abs_r", "spearman_r", "auc", "separation_auc", "n_samples"):
            add("interpretability.%s" % layer_id, metric, layer[metric])
    for metric, summary in interpretation.get("effect_size", {}).items():
        add("interpretability.effect_size", metric, summary)
    for group_type, groups in interpretation.get("grouped_robustness", {}).items():
        for group_name, metrics in groups.items():
            for metric, summary in metrics.items():
                add("interpretability.%s.%s" % (group_type, group_name), metric, summary)

    occlusion = aggregate["occlusion"]
    for metric, summary in occlusion.get("split_level", {}).items():
        add("occlusion.split_level", metric, summary)
    for group_type, groups in occlusion.get("grouped_robustness", {}).items():
        for group_name, metrics in groups.items():
            for metric, summary in metrics.items():
                add("occlusion.%s.%s" % (group_type, group_name), metric, summary)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_manifest", required=True, help="JSON manifest containing exactly five repeated splits")
    parser.add_argument("--output_root", default="results/interpretability_occlusion_repeated_splits")
    parser.add_argument("--num_stat_samples", type=int, default=500)
    parser.add_argument("--num_case_samples", type=int, default=5)
    parser.add_argument("--num_occlusion_samples", type=int, default=150)
    parser.add_argument("--num_case_figures", type=int, default=15)
    parser.add_argument("--bootstrap_iters", type=int, default=1000)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--attention_layer", type=int, default=-1)
    parser.add_argument("--attention_mode", default="functional")
    parser.add_argument("--heatmap_normalize", default="row", choices=["row", "global", "absolute"])
    parser.add_argument("--figure_format", default="svg", choices=["svg", "png"])
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--infer_config", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    manifest = validate_manifest(load_json(args.split_manifest))
    logger = setup_logging(args.output_root)
    records = []
    for fold in manifest:
        records.append(run_fold(fold, args, args.output_root, logger))

    for record in records:
        record["interpretability"] = load_json(
            os.path.join(record["interpretability_dir"], "interpretability_summary.json")
        )
        record["occlusion"] = load_json(
            os.path.join(record["occlusion_dir"], "occlusion_summary.json")
        )

    selected_id, scores = representative_fold(records)
    selected_record = next(record for record in records if record["id"] == selected_id)
    copy_representative_figures(selected_record, args.output_root)

    aggregate = {
        "protocol": {
            "num_splits": 5,
            "split_type": "repeated/random data splits; not strict cross-validation",
            "overlap_policy": "split-level summaries only; no cross-split pooling as independent observations",
            "num_stat_samples_per_split": args.num_stat_samples,
            "num_occlusion_samples_per_split": args.num_occlusion_samples,
            "bootstrap_iters_per_split": args.bootstrap_iters,
            "random_seed": args.random_seed,
            "representative_rule": "minimum normalized distance from split-wise medians across paired metrics",
            "interpretability_aggregation": "unweighted split-level mean/std; existing summaries do not contain raw per-sample layer metrics",
            "occlusion_aggregation": "unweighted split-level mean/std; per-sample pooling disabled because splits may overlap",
        },
        "representative_split": selected_id,
        "representative_split_scores": scores,
        "interpretability": aggregate_interpretability(records),
        "occlusion": aggregate_occlusion(records),
        "splits": [
            {
                "id": record["id"],
                "config": record["config"],
                "checkpoint": record["checkpoint"],
                "input_csv": record["input_csv"],
                "interpretability_dir": record["interpretability_dir"],
                "occlusion_dir": record["occlusion_dir"],
            }
            for record in records
        ],
    }
    write_json(os.path.join(args.output_root, "repeated_split_summary.json"), aggregate)
    write_metric_csv(
        os.path.join(args.output_root, "aggregate_metrics.csv"),
        flatten_aggregate_metrics(aggregate),
    )

    rows = []
    for record in records:
        interpretability = record["interpretability"]
        occlusion = record["occlusion"]
        layers = interpretability.get("layer_trend", {}).get("statistical", [])
        final_layer = layers[-1] if layers else {}
        rows.append({
            "split": record["id"],
            "interpretability_final_abs_r": final_layer.get("abs_r"),
            "interpretability_final_signed_r": final_layer.get("pearson_r"),
            "interpretability_cohen_d": get_path(interpretability, ["effect_size", "statistical", "cohen_d_signed"]),
            "interpretability_auc": get_path(interpretability, ["effect_size", "statistical", "auc"]),
            "occlusion_global_pearson": get_path(occlusion, ["aggregate", "global_pearson_r"]),
            "occlusion_global_spearman": get_path(occlusion, ["aggregate", "global_spearman_rho"]),
            "occlusion_mean_per_sample_r": occlusion.get("mean_per_sample_pearson_r"),
        })
    write_metric_csv(os.path.join(args.output_root, "split_metrics.csv"), rows)
    logger.info("Representative split: %s", selected_id)
    logger.info("Summary: %s", os.path.join(args.output_root, "repeated_split_summary.json"))


if __name__ == "__main__":
    main()
