#!/usr/bin/env python3
"""Summarize a manifest-bound architecture batch, without inference.

All values stay on their original scale. The five splits overlap: standard
deviations and paired differences are descriptive, not independent-sample tests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
from datetime import datetime, timezone

FOLDS = ("1222", "2252", "3514", "6072", "9075")
METRICS = ("f1", "accuracy", "auc", "ex_precision", "ex_recall",
           "ex_f1_harmonic", "q_seq", "q_brier")
SOURCE_COLUMNS = {metric: metric for metric in METRICS
                  if metric not in ("ex_f1_harmonic", "q_seq")}
SOURCE_COLUMNS["q_seq"] = "q_spearman_pep_seq"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False,
                                 allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fold_metrics(path: Path) -> dict[str, dict]:
    """Require all five unique split rows and finite, correctly scaled metrics."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"fold_id", *SOURCE_COLUMNS.values()}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        result = {}
        for raw in reader:
            fold = raw["fold_id"].strip()
            if fold not in FOLDS or fold in result:
                raise ValueError(f"{path}: unexpected or duplicate fold_id {fold!r}")
            row = {"fold_id": fold, "seed": (raw.get("seed") or "").strip()}
            for metric, column in SOURCE_COLUMNS.items():
                try:
                    value = float(raw[column])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{path}: fold {fold} invalid {column}") from error
                lower = -1.0 if metric == "q_seq" else 0.0
                if not math.isfinite(value) or not lower <= value <= 1.0:
                    raise ValueError(f"{path}: fold {fold} {column} outside [{lower}, 1]: {value}")
                row[metric] = value
            for canonical, alias in (("f1", "lab_f1_mi"), ("accuracy", "lab_acc_mi")):
                if raw.get(alias) not in (None, ""):
                    if not math.isclose(row[canonical], float(raw[alias]), abs_tol=1e-8, rel_tol=1e-8):
                        raise ValueError(f"{path}: fold {fold} {canonical} != {alias}")
            if raw.get("hamming_loss") not in (None, ""):
                if not math.isclose(row["accuracy"], 1 - float(raw["hamming_loss"]), abs_tol=1e-8):
                    raise ValueError(f"{path}: fold {fold} accuracy != 1 - hamming_loss")
            p, r = row["ex_precision"], row["ex_recall"]
            row["ex_f1_harmonic"] = 2 * p * r / (p + r) if p + r else 0.0
            result[fold] = row
    if set(result) != set(FOLDS):
        raise ValueError(f"{path}: missing splits {sorted(set(FOLDS) - set(result))}")
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def summarize(manifest_path: Path, output_dir: Path | None = None) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arms = manifest["arms"]
    if not arms or len({arm["name"] for arm in arms}) != len(arms):
        raise ValueError("Manifest must contain uniquely named arms; the first is the control")
    if tuple(manifest.get("folds", ())) != FOLDS:
        raise ValueError("Manifest must declare the five canonical repeated splits")
    output_dir = output_dir or manifest_path.parent / "summary"
    all_rows, aggregate_rows, paired_rows, sources, loaded = [], [], [], [], {}
    for arm in arms:
        path = Path(arm["metrics_csv"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        loaded[arm["name"]] = load_fold_metrics(path)
        sources.append({"arm": arm["name"], "config": arm["config"],
                        "metrics_csv": str(path.resolve()), "sha256": sha256(path)})
    control_name = arms[0]["name"]
    control = loaded[control_name]
    for arm in arms:
        name, rows = arm["name"], loaded[arm["name"]]
        for fold in FOLDS:
            if rows[fold]["seed"] != control[fold]["seed"]:
                raise ValueError(f"Unpaired seeds for {name}, split {fold}")
            all_rows.append({"arm": name, **rows[fold]})
        for metric in METRICS:
            values = [rows[fold][metric] for fold in FOLDS]
            deltas = [rows[fold][metric] - control[fold][metric] for fold in FOLDS]
            direction = -1 if metric == "q_brier" else 1
            wins = sum(direction * delta > 1e-12 for delta in deltas)
            losses = sum(direction * delta < -1e-12 for delta in deltas)
            aggregate_rows.append({
                "arm": name, "control": control_name, "metric": metric,
                "higher_is_better": direction == 1, "num_splits": len(values),
                "mean": statistics.mean(values), "std_ddof1": statistics.stdev(values),
                "paired_delta_mean": statistics.mean(deltas),
                "paired_delta_std_ddof1": statistics.stdev(deltas),
                "win_splits": wins, "tie_splits": len(FOLDS) - wins - losses,
                "loss_splits": losses,
            })
            if name != control_name:
                for fold, delta in zip(FOLDS, deltas):
                    paired_rows.append({"arm": name, "control": control_name,
                                        "fold_id": fold, "metric": metric,
                                        "value": rows[fold][metric],
                                        "control_value": control[fold][metric],
                                        "delta": delta, "improvement": direction * delta})
    report = {
        "schema_version": 1, "created_at": utc_now(),
        "manifest": str(manifest_path.resolve()), "gitrev": manifest.get("gitrev"),
        "control": control_name, "folds": list(FOLDS),
        "protocol": "Five repeated sequence-level 80/20 splits; overlapping across splits.",
        "interpretation": "Descriptive means, sample SD (ddof=1), and paired differences only; no t-test or p-values.",
        "scale": "Original unscaled metric values; delta = arm minus control; q_brier lower is better.",
        "example_f1": "Compute 2*ex_precision*ex_recall/(ex_precision+ex_recall) within each split, then average; ignore source ex_f1.",
        "warnings": (["Seed metadata absent: pairing verified by fold_id only."]
                     if any(not row["seed"] for row in all_rows) else []),
        "sources": sources, "metrics": aggregate_rows, "per_fold": all_rows,
        "paired_deltas": paired_rows,
    }
    write_csv(output_dir / "summary.csv", aggregate_rows)
    write_csv(output_dir / "per_fold_metrics.csv", all_rows)
    if paired_rows:
        write_csv(output_dir / "paired_deltas.csv", paired_rows)
    atomic_json(output_dir / "summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()
    report = summarize(args.manifest.resolve(), args.output_dir)
    print(f"Summarized {len(report['sources'])} arms; control={report['control']}", flush=True)


if __name__ == "__main__":
    main()
