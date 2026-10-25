"""Local unit tests for checkpoint selection; no dataset or training is run."""

from copy import deepcopy
import logging
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.checkpoint_selection import CheckpointSelection
from train_graph_model import restore_checkpoint_selection, save_checkpoint_with_selection


LOGGER = logging.getLogger("checkpoint_selection_test")


class CheckpointSelectionTest(unittest.TestCase):
    def test_default_legacy_matches_historical_loop(self):
        trace = [0.5, 0.51, 0.52, 0.49, 0.56, 0.565, 0.55, 0.57, 0.62]
        selection = CheckpointSelection(min_delta=0.05)
        self.assertEqual(selection.mode, "legacy")
        old_best, old_epoch, old_bad = float("-inf"), 0, 0
        for epoch, value in enumerate(trace, start=1):
            expected_save = value > old_best + 0.05
            if expected_save:
                old_best, old_epoch, old_bad = value, epoch, 0
            else:
                old_bad += 1
            decision = selection.update(value, epoch)
            self.assertEqual(decision.save_best, expected_save)
            self.assertEqual(decision.early_stopping_improved, expected_save)
            self.assertEqual(selection.best_f1, old_best)
            self.assertEqual(selection.best_epoch, old_epoch)
            self.assertEqual(selection.bad_validations, old_bad)

    def test_max_f1_saves_small_gains_without_resetting_patience(self):
        selection = CheckpointSelection("max_f1", min_delta=0.01)
        selection.update(0.8, 1)
        for epoch, value in ((2, 0.8001), (3, 0.8002), (4, 0.8003)):
            decision = selection.update(value, epoch)
            self.assertTrue(decision.save_best)
            self.assertFalse(decision.early_stopping_improved)
            self.assertEqual(selection.best_f1, value)
            self.assertEqual(selection.best_epoch, epoch)
            self.assertEqual(selection.early_stop_anchor, 0.8)
            self.assertEqual(selection.bad_validations, epoch - 1)
        # At patience=3, the final small gain is still saved before stopping.
        self.assertGreaterEqual(selection.bad_validations, 3)

    def test_anchor_uses_cumulative_gain_and_strict_boundary(self):
        selection = CheckpointSelection("max_f1", min_delta=0.125)
        selection.update(0.5, 1)
        boundary = selection.update(0.625, 3)
        self.assertTrue(boundary.save_best)
        self.assertFalse(boundary.early_stopping_improved)
        self.assertEqual(selection.bad_validations, 1)  # validations, not skipped epochs
        improved = selection.update(0.6875, 5)
        self.assertTrue(improved.save_best)
        self.assertTrue(improved.early_stopping_improved)
        self.assertEqual(selection.early_stop_anchor, 0.6875)
        self.assertEqual(selection.bad_validations, 0)

    def test_ties_and_regressions_do_not_replace_best(self):
        selection = CheckpointSelection("max_f1", min_delta=0.0)
        self.assertTrue(selection.update(0.0, 1).save_best)
        self.assertTrue(selection.update(0.7, 2).save_best)
        self.assertFalse(selection.update(0.7, 3).save_best)
        self.assertFalse(selection.update(0.6, 4).save_best)
        self.assertEqual(selection.best_epoch, 2)
        self.assertEqual(selection.bad_validations, 2)

    def test_restored_state_matches_uninterrupted_selection(self):
        trace = [0.5, 0.51, 0.52, 0.49, 0.56, 0.57, 0.59, 0.6]
        for mode in ("legacy", "max_f1"):
            with self.subTest(mode=mode):
                original = CheckpointSelection(mode, 0.05)
                for epoch, value in enumerate(trace[:4], start=1):
                    original.update(value, epoch)
                restored = CheckpointSelection(mode, 0.05)
                restored.load_state_dict(deepcopy(original.state_dict()))
                for epoch, value in enumerate(trace[4:], start=5):
                    self.assertEqual(original.update(value, epoch), restored.update(value, epoch))
                    self.assertEqual(original.state_dict(), restored.state_dict())

    def test_invalid_protocol_metrics_and_mismatched_resume_fail(self):
        for mode, delta in (("typo", 0), ("max_f1", -0.1), ("legacy", float("nan"))):
            with self.subTest(mode=mode, delta=delta), self.assertRaises(ValueError):
                CheckpointSelection(mode, delta)
        selection = CheckpointSelection("max_f1", 0.01)
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                selection.update(value, 1)
        selection.update(0.5, 1)
        with self.assertRaises(ValueError):
            selection.update(0.6, 1)
        for target in (CheckpointSelection("legacy", 0.01), CheckpointSelection("max_f1", 0.02)):
            with self.assertRaisesRegex(ValueError, "protocol mismatch"):
                target.load_state_dict(selection.state_dict())

    def test_best_and_periodic_checkpoints_preserve_selection_schema(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        selection = CheckpointSelection("max_f1", 0.01)
        with tempfile.TemporaryDirectory(prefix="dbond_selection_") as directory:
            best_path = str(Path(directory) / "best_model.pt")
            for epoch, value in enumerate((0.8, 0.8001, 0.799, 0.8002), start=1):
                decision = selection.update(value, epoch)
                if decision.save_best:
                    with torch.no_grad():
                        model.weight.fill_(epoch)
                    save_checkpoint_with_selection(
                        model, optimizer, epoch - 1, 0.2, {"f1": value},
                        best_path, selection, is_best=True,
                    )
            best = torch.load(best_path, weights_only=False)
            self.assertEqual(best["epoch"] + 1, 4)
            self.assertEqual(best["metrics"], {"f1": 0.8002})
            self.assertEqual(float(best["model_state_dict"]["weight"].item()), 4.0)
            self.assertEqual(best[CheckpointSelection.STATE_KEY], selection.state_dict())
            self.assertTrue((Path(directory) / "best_model_best.pt").exists())
            self.assertIn("optimizer_state_dict", best)

            selection.update(0.79, 5)
            periodic_path = str(Path(directory) / "checkpoint_epoch_5.pt")
            save_checkpoint_with_selection(
                model, optimizer, 4, 0.1, {"f1": 0.99}, periodic_path, selection,
            )
            periodic = torch.load(periodic_path, weights_only=False)
            restored = CheckpointSelection("max_f1", 0.01)
            restore_checkpoint_selection(restored, periodic, periodic_path, True, LOGGER)
            self.assertEqual(restored.state_dict(), selection.state_dict())
            self.assertEqual(restored.best_f1, 0.8002)  # never use periodic train F1=0.99
            self.assertEqual(restored.bad_validations, 4)

    def test_legacy_resume_uses_validation_checkpoint_and_warns(self):
        with tempfile.TemporaryDirectory(prefix="dbond_selection_") as directory:
            best_path = Path(directory) / "best_model.pt"
            best = {"epoch": 1, "metrics": {"f1": 0.7}}
            torch.save(best, best_path)
            periodic = {"epoch": 4, "metrics": {"f1": 0.99}}
            selection = CheckpointSelection("max_f1", 0.01)
            with self.assertLogs(LOGGER, level="WARNING") as logs:
                restore_checkpoint_selection(
                    selection, periodic, str(Path(directory) / "checkpoint_epoch_5.pt"), True, LOGGER,
                )
            self.assertEqual(selection.best_f1, 0.7)
            self.assertEqual(selection.best_epoch, 2)
            self.assertEqual(selection.bad_validations, 0)
            self.assertIn("Historical unsaved F1 maxima", "\n".join(logs.output))
            self.assertTrue(selection.update(0.7001, 6).save_best)

    def test_resume_rejects_mismatched_or_missing_best_weights(self):
        selection = CheckpointSelection("max_f1", 0.01)
        selection.update(0.7, 2)
        periodic = {"epoch": 4, "metrics": {"f1": 0.99},
                    CheckpointSelection.STATE_KEY: selection.state_dict()}
        with tempfile.TemporaryDirectory(prefix="dbond_selection_") as directory:
            path = str(Path(directory) / "checkpoint_epoch_5.pt")
            with self.assertRaises(FileNotFoundError):
                restore_checkpoint_selection(CheckpointSelection("max_f1", 0.01), periodic, path, True, LOGGER)
            torch.save({"epoch": 3, "metrics": {"f1": 0.8}}, Path(directory) / "best_model.pt")
            with self.assertRaisesRegex(ValueError, "matching best_model"):
                restore_checkpoint_selection(CheckpointSelection("max_f1", 0.01), periodic, path, True, LOGGER)

    def test_empty_state_and_no_validation_legacy_resume(self):
        selection = CheckpointSelection()
        restored = CheckpointSelection()
        restored.load_state_dict(selection.state_dict())
        self.assertEqual(restored.state_dict(), selection.state_dict())
        with self.assertLogs(LOGGER, level="WARNING"):
            restore_checkpoint_selection(restored, {"epoch": 1, "metrics": {"f1": 0.99}},
                                         "unused.pt", False, LOGGER)
        self.assertEqual(restored.best_f1, float("-inf"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
