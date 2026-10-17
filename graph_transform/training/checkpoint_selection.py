"""Validation-F1 checkpoint selection, separate from early-stopping progress.

``legacy`` preserves the historical min_delta-gated checkpoint selection.
``max_f1`` saves every strict new maximum; only the independent early-stopping
anchor uses min_delta. Epochs here are one-based validation epoch numbers.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SelectionDecision:
    save_best: bool
    early_stopping_improved: bool


class CheckpointSelection:
    STATE_KEY = "checkpoint_selection_state"

    def __init__(self, mode="legacy", min_delta=0.0):
        if mode not in ("legacy", "max_f1"):
            raise ValueError("training.checkpoint_selection must be 'legacy' or 'max_f1'")
        if not math.isfinite(float(min_delta)) or float(min_delta) < 0:
            raise ValueError("training.min_delta must be finite and non-negative")
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best_f1 = float("-inf")
        self.best_epoch = 0
        self.early_stop_anchor = float("-inf")
        self.bad_validations = 0
        self.last_validation_epoch = 0

    def update(self, f1, epoch):
        f1 = float(f1)
        if not math.isfinite(f1) or not 0.0 <= f1 <= 1.0:
            raise ValueError("Validation F1 must be finite and in [0, 1]")
        if int(epoch) != epoch or epoch <= self.last_validation_epoch:
            raise ValueError("Validation epochs must be strictly increasing positive integers")
        epoch = int(epoch)
        # The anchor intentionally does not follow every small new maximum.
        # Several small gains can eventually exceed min_delta cumulatively.
        early_improved = f1 > self.early_stop_anchor + self.min_delta
        if early_improved:
            self.early_stop_anchor = f1
            self.bad_validations = 0
        else:
            self.bad_validations += 1

        save_best = (f1 > self.best_f1 if self.mode == "max_f1"
                     else f1 > self.best_f1 + self.min_delta)
        if save_best:
            self.best_f1 = f1
            self.best_epoch = epoch
        self.last_validation_epoch = epoch
        return SelectionDecision(save_best, early_improved)

    def state_dict(self):
        return {
            "version": 1, "mode": self.mode, "min_delta": self.min_delta,
            "best_f1": self.best_f1, "best_epoch": self.best_epoch,
            "early_stop_anchor": self.early_stop_anchor,
            "bad_validations": self.bad_validations,
            "last_validation_epoch": self.last_validation_epoch,
        }

    def load_state_dict(self, state):
        if state.get("version") != 1:
            raise ValueError("Unsupported checkpoint selection state version")
        if state.get("mode") != self.mode or state.get("min_delta") != self.min_delta:
            raise ValueError(
                "Resume checkpoint selection protocol mismatch: use the saved "
                "checkpoint_selection/min_delta, or start a fresh run"
            )
        best = float(state["best_f1"])
        anchor = float(state["early_stop_anchor"])
        for name, value in (("best_f1", best), ("early_stop_anchor", anchor)):
            if value != float("-inf") and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"Invalid saved selection {name}")
        counts = {}
        for name in ("best_epoch", "bad_validations", "last_validation_epoch"):
            value = state[name]
            if int(value) != value or value < 0:
                raise ValueError(f"Invalid saved selection {name}")
            counts[name] = int(value)
        if counts["best_epoch"] > counts["last_validation_epoch"]:
            raise ValueError("Saved best epoch exceeds last validation epoch")
        if counts["last_validation_epoch"] == 0:
            if best != float("-inf") or anchor != float("-inf") or counts["best_epoch"] != 0 or counts["bad_validations"] != 0:
                raise ValueError("Invalid empty checkpoint selection state")
        elif not math.isfinite(best) or not math.isfinite(anchor) or counts["best_epoch"] == 0:
            raise ValueError("Saved validation state must have a finite best and anchor")
        if anchor > best or (self.mode == "legacy" and anchor != best):
            raise ValueError("Saved selection best and early-stopping anchor are inconsistent")
        self.best_f1 = best
        self.early_stop_anchor = anchor
        self.best_epoch = counts["best_epoch"]
        self.bad_validations = counts["bad_validations"]
        self.last_validation_epoch = counts["last_validation_epoch"]

    def initialize_from_legacy_validation(self, checkpoint):
        """Seed from a known validation checkpoint, never periodic train metrics.

        Historical unsaved maxima and patience counts cannot be reconstructed.
        The caller must log this limitation when loading an old checkpoint.
        """
        f1 = float((checkpoint.get("metrics") or {}).get("f1", float("nan")))
        epoch = int(checkpoint["epoch"]) + 1
        self.update(f1, epoch)
