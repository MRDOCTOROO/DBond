#!/usr/bin/env python3
"""CPU-only synthetic CSV tests; never start training subprocesses."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch

import run_architecture_batch as batch
import summarize_architecture_batch as summary
import train_5fold_parallel as parallel


def write_metrics(path: Path, delta: float = 0.0, folds=summary.FOLDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, fold in enumerate(folds):
        rows.append({"fold_id": fold, "seed": 42 + index,
                     "f1": 0.7 + index * 0.01 + delta,
                     "accuracy": 0.8 + delta, "auc": 0.88 + delta,
                     "ex_precision": 0.6 + index * 0.05,
                     "ex_recall": 0.9 - index * 0.02,
                     "ex_f1": 0.123,  # Must never use this incompatible caliber.
                     "q_spearman_pep_seq": 0.5 + delta,
                     "q_brier": 0.07 - delta})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def make_args(self):
        data = self.root / "data"
        data.mkdir()
        configs = []
        for name in ("control", "variant", "third"):
            path = self.root / f"{name}.yaml"
            path.write_text("training: {}\n", encoding="utf-8")
            configs.append(str(path))
        return argparse.Namespace(configs=configs, fold_data_dir=str(data),
                                  gpus="0,1,2,3,4", run_root=str(self.root / "run"))

    def test_sample_sd_harmonic_and_direction(self):
        arms = []
        for name, delta in (("control", 0), ("variant", 0.01)):
            path = self.root / name / "5fold_metrics.csv"
            write_metrics(path, delta)
            arms.append({"name": name, "config": f"{name}.yaml", "metrics_csv": str(path)})
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"arms": arms, "folds": summary.FOLDS}))
        report = summary.summarize(manifest)
        metrics = {(row["arm"], row["metric"]): row for row in report["metrics"]}
        self.assertAlmostEqual(metrics["control", "f1"]["std_ddof1"], statistics.stdev([.7, .71, .72, .73, .74]))
        self.assertAlmostEqual(metrics["variant", "f1"]["paired_delta_mean"], .01)
        self.assertAlmostEqual(metrics["variant", "q_brier"]["paired_delta_mean"], -.01)
        self.assertEqual(metrics["variant", "q_brier"]["win_splits"], 5)
        expected = statistics.mean([2 * (.6+i*.05) * (.9-i*.02) / (.6+i*.05+.9-i*.02) for i in range(5)])
        self.assertAlmostEqual(metrics["control", "ex_f1_harmonic"]["mean"], expected)
        self.assertNotAlmostEqual(expected, .123)
        self.assertEqual(len(report["paired_deltas"]), 5 * len(summary.METRICS))
        for filename in ("summary.json", "summary.csv", "paired_deltas.csv", "per_fold_metrics.csv"):
            self.assertTrue((self.root / "summary" / filename).is_file())

    def test_incomplete_duplicate_nan_rejected(self):
        path = self.root / "metrics.csv"
        for folds in (summary.FOLDS[:-1], (*summary.FOLDS[:-1], summary.FOLDS[0])):
            write_metrics(path, folds=folds)
            with self.assertRaises(ValueError):
                summary.load_fold_metrics(path)
        write_metrics(path, delta=float("nan"))
        with self.assertRaises(ValueError):
            summary.load_fold_metrics(path)

    def test_batch_failed_process_stops_next_arm(self):
        args = self.make_args()
        with patch.object(batch, "run_logged", return_value=7) as runner:
            self.assertEqual(batch.execute(args), 1)
        self.assertEqual(runner.call_count, 1)
        status = json.loads((Path(args.run_root) / "status.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertEqual([arm["status"] for arm in status["arms"]], ["failed", "pending", "pending"])
        self.assertEqual(status["arms"][0]["returncode"], 7)

    def test_zero_exit_incomplete_csv_is_failure(self):
        args = self.make_args()
        def incomplete(command, log, status, status_path):
            cv_root = Path(command[command.index("--cv_root") + 1])
            write_metrics(cv_root / "5fold_metrics.csv", folds=summary.FOLDS[:-1])
            return 0
        with patch.object(batch, "run_logged", side_effect=incomplete) as runner:
            self.assertEqual(batch.execute(args), 1)
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(json.loads((Path(args.run_root) / "status.json").read_text())["status"], "failed")

    def test_complete_batch_calls_summary_once(self):
        args = self.make_args()
        def succeed(command, log, status, status_path):
            if "--cv_root" in command:
                cv_root = Path(command[command.index("--cv_root") + 1])
                write_metrics(cv_root / "5fold_metrics.csv")
            else:
                summary.summarize(Path(command[command.index("--manifest") + 1]))
            return 0
        with patch.object(batch, "run_logged", side_effect=succeed) as runner:
            self.assertEqual(batch.execute(args), 0)
        self.assertEqual(runner.call_count, 4)
        status = json.loads((Path(args.run_root) / "status.json").read_text())
        self.assertEqual(status["status"], "completed")
        self.assertTrue(all(arm["status"] == "completed" for arm in status["arms"]))
        manifest = json.loads((Path(args.run_root) / "manifest.json").read_text())
        for arm in manifest["arms"]:
            self.assertEqual(Path(arm["config_snapshot"]).read_bytes(), Path(arm["config"]).read_bytes())
        with self.assertRaises(ValueError):
            batch.execute(args)  # Never overwrite an existing batch.

    def test_aggregation_nonzero_and_partial_are_failures(self):
        with patch.object(parallel.subprocess, "run") as run:
            run.return_value.returncode = 3
            with self.assertRaises(RuntimeError):
                parallel.aggregate("config", "cv", str(self.root))
        write_metrics(self.root / "5fold_metrics.csv", folds=summary.FOLDS[:-1])
        with self.assertRaises(RuntimeError):
            parallel.check_completed_folds(str(self.root), list(summary.FOLDS))


if __name__ == "__main__":
    unittest.main()
