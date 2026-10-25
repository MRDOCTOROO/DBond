#!/usr/bin/env python3
"""One external ion4 pretraining run; no MiPD labels, classification metrics or test access.

Example (launch separately from production training):
  python graph_transform/scripts/pretrain_ion_intensity.py --train_csv train.csv \
    --val_csv val.csv --dataset_manifest manifest.json --output_dir NEW_RUN --device cuda
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "graph_transform"))
from pretraining.intensity_model import (
    CHANNELS, IonIntensityDataset, IonIntensityModel, collate_ion_intensity,
    make_pretraining_config, masked_l1_per_spectrum,
)


PROTOCOL = {
    "seed": 42, "max_epochs": 60, "warmup_epochs": 5,
    "learning_rate": 1e-4, "weight_decay": 1e-4, "batch_size": 128,
    "dropout": 0.1, "early_stop_patience": 10,
    "loss": "masked_L1_mean_slots_per_spectrum_then_mean_spectra",
    "checkpoint_selection": "strict_min_validation_masked_L1",
    "optimizer": "AdamW", "schedule": "linear_warmup_then_cosine",
    "amp": "CUDA_FP16_only", "resume_supported": False,
}
SOURCE_FILES = [
    "graph_transform/models/graph_transformer.py", "graph_transform/models/utils.py",
    "graph_transform/models/gcn_layers.py", "graph_transform/models/attention_layers.py",
    "graph_transform/data/graph_dataset.py", "graph_transform/data/graph_builder.py",
    "graph_transform/data/preprocessing.py", "graph_transform/data/theory_features.py",
    "graph_transform/pretraining/intensity_model.py", "graph_transform/pretraining/msp_dataset.py",
    "graph_transform/scripts/pretrain_ion_intensity.py", "graph_transform/scripts/prepare_ion_pretraining.py",
]
EXPECTED_EXCLUSION_SETS = {
    "nist": (238178, "aad4fe6b2cca66bdaab09fca6aad49db5e3923bb27aa9bd9f8de4c86de596e84"),
    "mipd": (513, "ee70d1ede41b7359b74eec10a3c501ddf78388055d23ae8de972bb868bd44e64"),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sequence_set_sha256(sequences):
    ordered = sorted(set(sequences))
    text = "\n".join(ordered) + ("\n" if ordered else "")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_identity():
    hashes = {name: sha256_file(REPO_ROOT / name) for name in SOURCE_FILES}
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                            capture_output=True, text=True, check=True,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return {"git_revision": result.stdout.strip(), "source_sha256": hashes}


def validate_dataset_manifest(manifest_path, train_path, val_path, train, val):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "completed" or manifest.get("dataset_kind") != "ion_intensity_external"
            or manifest.get("channel_order") != CHANNELS):
        raise ValueError("wrong external dataset kind/channel order")
    exclusion = manifest["exclusion"]
    if exclusion.get("identity_rule") != "uppercase_base_sequence_I_to_L":
        raise ValueError("unsupported exclusion identity rule")
    if exclusion.get("certification", {}).get("certified_full_exclusion_sets") is not True:
        raise ValueError("full NIST/MiPD exclusion certification is required")
    for name, (count, digest) in EXPECTED_EXCLUSION_SETS.items():
        entry = exclusion.get(name, {})
        if entry.get("unique_sequences") != count or entry.get("sequence_set_sha256") != digest:
            raise ValueError(f"full {name} exclusion count/SHA256 mismatch")
    identity_info = exclusion["identity_file"]
    identity_path = Path(identity_info["path"])
    if not identity_path.is_absolute():
        identity_path = manifest_path.parent / identity_path
    if sha256_file(identity_path) != identity_info["sha256"]:
        raise ValueError("excluded identity file SHA256 mismatch")
    excluded_rows = identity_path.read_text(encoding="utf-8").splitlines()
    excluded = set(excluded_rows)
    if len(excluded_rows) != identity_info["rows"] or len(excluded) != len(excluded_rows) or not excluded:
        raise ValueError("invalid/empty exclusion identity file")
    if any(not seq or seq != seq.upper().replace("I", "L") for seq in excluded):
        raise ValueError("noncanonical excluded sequence identity")
    mipd = exclusion["mipd"]
    mipd_sequences = mipd.get("sequences", [])
    mipd_ids = {seq.replace("I", "L") for seq in mipd_sequences}
    if (mipd_sequences != sorted(set(mipd_sequences)) or len(mipd_sequences) != mipd["unique_sequences"]
            or sequence_set_sha256(mipd_sequences) != mipd["sequence_set_sha256"]
            or sequence_set_sha256(mipd_ids) != mipd["identity_set_sha256"]
            or not mipd_ids <= excluded):
        raise ValueError("MiPD exclusion sequence list/combined identity file mismatch")
    sequences, identities, input_hashes = {}, {}, {}
    for name, path, dataset in (("train", train_path, train), ("val", val_path, val)):
        entry = manifest["outputs"][name]
        seqs = sorted(set(dataset.data["seq"].tolist()))
        ids = sorted({seq.replace("I", "L") for seq in seqs})
        actual_hash = sha256_file(path)
        expected = {"sha256": actual_hash, "rows": len(dataset), "unique_sequences": len(seqs),
                    "sequence_set_sha256": sequence_set_sha256(seqs),
                    "identity_set_sha256": sequence_set_sha256(ids),
                    "sequences": seqs, "sequence_identities": ids}
        for key, value in expected.items():
            if entry.get(key) != value:
                raise ValueError(f"manifest outputs.{name}.{key} mismatch")
        if set(ids) & excluded:
            raise ValueError(f"external {name} intersects the excluded identities")
        sequences[name], identities[name] = seqs, ids
        input_hashes[name + "_csv"] = actual_hash
    if set(identities["train"]) & set(identities["val"]):
        raise ValueError("external train/val sequence identity overlap")
    return {
        "dataset_manifest_path": str(manifest_path), "dataset_manifest": manifest,
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "dataset_manifest_canonical_sha256": canonical_sha256(manifest),
        "exclusion_identity": exclusion, "input_sha256": input_hashes,
        "input_paths": {"train_csv": str(Path(train_path).resolve()), "val_csv": str(Path(val_path).resolve())},
        "train_sequences": sequences["train"], "val_sequences": sequences["val"],
        "train_sequences_sha256": sequence_set_sha256(sequences["train"]),
        "val_sequences_sha256": sequence_set_sha256(sequences["val"]),
        "trained_amino_acids": sorted(set("".join(sequences["train"]))),
        "trained_position_count": max(map(len, sequences["train"])),
        "train_spectra": len(train), "val_spectra": len(val),
        "exclusion_identity_file_path": str(identity_path.resolve()),
    }


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def learning_rate_for_epoch(epoch_index):
    warmup = PROTOCOL["warmup_epochs"]
    if epoch_index < warmup:
        return PROTOCOL["learning_rate"] * (epoch_index + 1) / warmup
    progress = (epoch_index - warmup) / (PROTOCOL["max_epochs"] - warmup - 1)
    return PROTOCOL["learning_rate"] * (1 + math.cos(math.pi * progress)) / 2


def run_epoch(model, loader, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total, spectra, slots = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = move_batch(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                predictions = model(batch)
                per_spectrum = masked_l1_per_spectrum(predictions, batch["ion_targets"], batch["ion_mask"])
                loss = per_spectrum.mean()
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total += float(per_spectrum.detach().double().sum().item())
            spectra += len(per_spectrum)
            slots += int(batch["ion_mask"].sum().item())
    if spectra == 0:
        raise ValueError("empty pretraining loader")
    return {"masked_l1": total / spectra, "spectra": spectra, "supervised_slots": slots}


def constant_baseline(train, val):
    medians, counts = [], []
    for channel in range(4):
        active = [values[:, channel][mask[:, channel]]
                  for values, mask in zip(train.ion_targets, train.ion_masks)]
        values = torch.cat(active)
        counts.append(values.numel())
        medians.append(float(torch.quantile(values, 0.5)) if values.numel() else None)
    result = {"kind": "training_observed_slot_per_channel_median", "values": medians,
              "train_observed_slots_per_channel": counts, "val_masked_l1": None,
              "role": "descriptive_sanity_only_not_selection_or_success_criterion"}
    if all(value is not None for value in medians):
        constant = torch.tensor(medians)
        losses = [float(masked_l1_per_spectrum(constant.expand_as(values)[None], values[None], mask[None])[0])
                  for values, mask in zip(val.ion_targets, val.ion_masks)]
        result["val_masked_l1"] = sum(losses) / len(losses)
    return result


def checkpoint_payload(model, metadata, epoch, val_loss, initial_val_loss, updates):
    # All metadata is plain JSON-compatible data; all weights are detached CPU tensors.
    return {
        "schema": 1, "schema_version": 1, "task": "ion_intensity_external", "channels": list(CHANNELS),
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.backbone.state_dict().items()},
        "ion_head_state_dict": {k: v.detach().cpu().clone() for k, v in model.ion_head.state_dict().items()},
        "model_config": model.backbone.config.to_dict(), "best_epoch": int(epoch),
        "val_loss": float(val_loss), "initial_val_loss": float(initial_val_loss),
        "optimizer_updates": int(updates), "protocol": dict(PROTOCOL), **metadata,
    }


def run(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"output_dir must be new or empty: {output}")
    status = {"status": "running", "started_at": now(), "finished_at": None,
              "error": None, "best_checkpoint": None}
    write_json_atomic(output / "status.json", status)
    try:
        device = torch.device(args.device)
        if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
            raise ValueError("requested CPU/CUDA device is unavailable")
        random.seed(PROTOCOL["seed"])
        np.random.seed(PROTOCOL["seed"])
        torch.manual_seed(PROTOCOL["seed"])
        if device.type == "cuda":
            torch.cuda.manual_seed_all(PROTOCOL["seed"])
        torch.backends.cudnn.benchmark = False
        config = make_pretraining_config()
        train = IonIntensityDataset(args.train_csv, config, split="train")
        val = IonIntensityDataset(args.val_csv, config, split="val")
        metadata = validate_dataset_manifest(args.dataset_manifest, args.train_csv, args.val_csv, train, val)
        metadata.update(source_identity())
        metadata["runtime"] = {"torch": str(torch.__version__), "python": sys.version,
                               "device": str(device), "cuda": torch.version.cuda}
        resolved_config = {"protocol": PROTOCOL, "model_config": config.to_dict(),
                           "num_workers": args.num_workers, "device": str(device)}
        metadata["config_sha256"] = canonical_sha256(resolved_config)
        write_json_atomic(output / "config.json", resolved_config)
        write_json_atomic(output / "manifest.json", {"schema_version": 1, **metadata})
        generator = torch.Generator().manual_seed(PROTOCOL["seed"])
        common = dict(batch_size=PROTOCOL["batch_size"], num_workers=args.num_workers,
                      collate_fn=collate_ion_intensity, pin_memory=device.type == "cuda", drop_last=False,
                      persistent_workers=args.num_workers > 0)
        train_loader = DataLoader(train, shuffle=True, generator=generator, **common)
        val_loader = DataLoader(val, shuffle=False, **common)
        model = IonIntensityModel(config).to(device)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=PROTOCOL["learning_rate"], weight_decay=PROTOCOL["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        update_count = [0]
        def count_update(*unused):
            update_count[0] += 1
        optimizer.register_step_post_hook(count_update)
        initial = run_epoch(model, val_loader, device)
        baseline = constant_baseline(train, val)
        history = {"schema_version": 1, "git_revision": metadata["git_revision"],
                   "source_sha256": metadata["source_sha256"], "config_sha256": metadata["config_sha256"],
                   "initial_validation": initial, "constant_baseline": baseline, "epochs": []}
        write_json_atomic(output / "history.json", history)
        best, best_epoch, bad_epochs = float("inf"), 0, 0
        best_path = output / "best_ion_intensity.pt"
        for epoch_index in range(PROTOCOL["max_epochs"]):
            started = time.monotonic()
            lr = learning_rate_for_epoch(epoch_index)
            for group in optimizer.param_groups:
                group["lr"] = lr
            updates_before = update_count[0]
            train_metrics = run_epoch(model, train_loader, device, optimizer, scaler)
            val_metrics = run_epoch(model, val_loader, device)
            improved = val_metrics["masked_l1"] < best
            if improved:
                best, best_epoch, bad_epochs = val_metrics["masked_l1"], epoch_index + 1, 0
                payload = checkpoint_payload(model, metadata, best_epoch, best,
                                             initial["masked_l1"], update_count[0])
                temporary = best_path.with_suffix(".pt.tmp")
                torch.save(payload, temporary)
                os.replace(temporary, best_path)
            else:
                bad_epochs += 1
            history["epochs"].append({"epoch": epoch_index + 1, "learning_rate": lr,
                                      "train": train_metrics, "val": val_metrics, "saved_best": improved,
                                      "bad_epochs": bad_epochs, "optimizer_updates": update_count[0],
                                      "epoch_updates": update_count[0] - updates_before,
                                      "amp_skipped_steps": len(train_loader) - (update_count[0] - updates_before),
                                      "elapsed_seconds": time.monotonic() - started})
            write_json_atomic(output / "history.json", history)
            if bad_epochs >= PROTOCOL["early_stop_patience"]:
                break
        # Fail closed if provenance changed while the process was running.
        for key, path in metadata["input_paths"].items():
            if sha256_file(path) != metadata["input_sha256"][key]:
                raise RuntimeError(f"training input changed during run: {key}")
        if sha256_file(args.dataset_manifest) != metadata["dataset_manifest_sha256"]:
            raise RuntimeError("dataset manifest changed during run")
        if sha256_file(metadata["exclusion_identity_file_path"]) != metadata["exclusion_identity"]["identity_file"]["sha256"]:
            raise RuntimeError("exclusion identities changed during run")
        if source_identity()["source_sha256"] != metadata["source_sha256"]:
            raise RuntimeError("pretraining source changed during run")
        # This also verifies the promised safe deployment serialization format.
        restored = torch.load(best_path, map_location="cpu", weights_only=True)
        if restored["best_epoch"] != best_epoch or restored["val_loss"] != best:
            raise RuntimeError("best checkpoint round-trip mismatch")
        status.update(status="completed", finished_at=now(), best_checkpoint=str(best_path),
                      best_checkpoint_sha256=sha256_file(best_path), best_epoch=best_epoch,
                      val_loss=best, initial_val_loss=initial["masked_l1"],
                      completed_epochs=len(history["epochs"]), optimizer_updates=update_count[0],
                      train_spectra_seen=sum(row["train"]["spectra"] for row in history["epochs"]),
                      stopped_early=bad_epochs >= PROTOCOL["early_stop_patience"])
        write_json_atomic(output / "status.json", status)
        return status
    except BaseException as exc:
        status.update(status="failed", finished_at=now(), error=f"{type(exc).__name__}: {exc}")
        write_json_atomic(output / "status.json", status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--dataset_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("num_workers must be nonnegative")
    print(json.dumps(run(args), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
