"""Opt-in discriminative AdamW rates for verified ion4 bond-context transfer.

Only fully copied encoder tensors use a reduced learning rate. Partly copied
embeddings/fusion tensors and the bond-context head keep the ordinary rate.
The scale=1 production optimizer and scheduler remain on their legacy paths.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path

import torch


POLICY = "fully_copied_encoder_lr_scale_v1"


def validate_transfer_optimizer_request(training, resume=None):
    value = training.get("transfer_encoder_lr_scale", 1.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError("transfer_encoder_lr_scale must be a finite number in (0, 1]")
    scale = float(value)
    if scale < 1:
        if resume:
            raise ValueError("--resume is unsupported with transfer_encoder_lr_scale < 1")
        if training.get("pretrained_transfer_scope") != "bond_context":
            raise ValueError("reduced encoder LR requires verified bond_context initialization")
    return scale


def _canonical_sha256(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verified_encoder_names(model, training, provenance):
    """Check the loader's returned contract; the caller must run that loader first.

    This function does not load a source or trust a scope/config flag alone. The
    production caller passes the successful load_pretrained_backbone result.
    """
    from pretraining.transfer import (
        AA_KEY, POSITION_KEY, FUSION_KEY, NODE_SHARED, CONTEXT_KEYS, GCN_LEAVES, GAT_LEAVES,
    )
    if not isinstance(provenance, dict) or provenance.get("schema_version") != 1:
        raise ValueError("reduced encoder LR requires the verified transfer loader's provenance")
    if (provenance.get("scope") != "bond_context" or provenance.get("source_task") != "ion_intensity_external"
            or provenance.get("checkpoint_sha256") != training.get("pretrained_backbone_sha256")):
        raise ValueError("transfer optimizer source/scope binding mismatch")
    requested_path = training.get("pretrained_backbone_path")
    if not isinstance(requested_path, str) or not requested_path:
        raise ValueError("missing verified pretrained source path")
    path = Path(requested_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    if str(path.resolve()) != provenance.get("checkpoint_path"):
        raise ValueError("transfer optimizer checkpoint path mismatch")
    binding = provenance.get("dataset_binding", {})
    if (binding.get("full_mipd513_exclusion_verified") is not True
            or binding.get("external_train_val_identity_disjoint") is not True):
        raise ValueError("missing verified transfer dataset binding")
    config = model.config.to_dict()
    if provenance.get("target_model_config") != config:
        raise ValueError("transfer optimizer target model mismatch")
    slow = set(NODE_SHARED)
    slow.update(f"gcn_layers.{index}.{leaf}" for index in range(config["num_gcn_layers"]) for leaf in GCN_LEAVES)
    slow.update(f"gat_layers.{index}.{leaf}" for index in range(config["num_gat_layers"]) for leaf in GAT_LEAVES)
    if provenance.get("fully_copied_keys") != sorted(slow | CONTEXT_KEYS):
        raise ValueError("fully copied transfer encoder/context allowlist mismatch")
    partial = [AA_KEY, POSITION_KEY, FUSION_KEY]
    if provenance.get("partially_copied_keys") != partial:
        raise ValueError("partial transfer tensor allowlist mismatch")
    named = list(model.named_parameters(remove_duplicate=False))
    if len({id(parameter) for _, parameter in named}) != len(named):
        raise ValueError("tied/aliased parameters are unsupported for discriminative transfer LR")
    names = {name for name, _ in named}
    if not slow or not slow < names or not (slow | CONTEXT_KEYS | set(partial)) <= names:
        raise ValueError("transfer optimizer names do not match live model parameters")
    expected_retained = sorted(set(model.state_dict()) - slow - CONTEXT_KEYS - set(partial))
    if provenance.get("retained_entire_keys") != expected_retained:
        raise ValueError("retained random/condition tensor allowlist mismatch")
    return slow


def create_transfer_optimizer(model, config, transfer_provenance):
    """Create the two genuine AdamW parameter groups, normal group first."""
    training, optimizer_config = config["training"], config["optimizer"]
    scale = validate_transfer_optimizer_request(training)
    if scale == 1:
        raise ValueError("scale=1 must use the unchanged legacy optimizer constructor")
    if optimizer_config.get("type", "adamw").lower() != "adamw":
        raise ValueError("reduced transfer encoder LR currently supports AdamW only")
    if training.get("scheduler", "cosine").lower() != "cosine":
        raise ValueError("reduced transfer encoder LR requires the ratio-preserving cosine scheduler")
    base_lr = float(training["learning_rate"])
    minimum = float(training.get("min_lr", 0.0))
    if not math.isfinite(base_lr) or base_lr <= 0 or not math.isfinite(minimum) or not 0 <= minimum <= base_lr:
        raise ValueError("require finite learning_rate > 0 and 0 <= min_lr <= learning_rate")
    slow_names = _verified_encoder_names(model, training, transfer_provenance)
    named = list(model.named_parameters())
    groups = [
        {"params": [parameter for name, parameter in named if name not in slow_names],
         "lr": base_lr, "name": "base", "lr_scale": 1.0},
        {"params": [parameter for name, parameter in named if name in slow_names],
         "lr": base_lr * scale, "name": "pretrained_encoder", "lr_scale": scale},
    ]
    optimizer = torch.optim.AdamW(
        groups, lr=base_lr, weight_decay=training.get("weight_decay", 0.0),
        betas=optimizer_config.get("betas", [0.9, 0.999]), eps=float(optimizer_config.get("eps", 1e-8)),
    )
    optimizer._verified_transfer_encoder_lr_scale = scale
    return optimizer


def create_transfer_scheduler(optimizer, config):
    """Reuse the legacy normal-rate formula with one identical multiplier per group.

    In particular, the absolute floor of the slow group is min_lr * scale,
    rather than the legacy multi-group helper's shared absolute min_lr.
    """
    training = config["training"]
    scale = validate_transfer_optimizer_request(training)
    if scale == 1 or getattr(optimizer, "_verified_transfer_encoder_lr_scale", None) != scale:
        raise ValueError("ratio-preserving scheduler requires the verified grouped optimizer")
    if training.get("scheduler", "cosine").lower() != "cosine":
        raise ValueError("unsupported scheduler for reduced transfer encoder LR")
    base_lr = float(training["learning_rate"])
    expected = [("base", 1.0), ("pretrained_encoder", scale)]
    if len(optimizer.param_groups) != 2 or any(
        group.get("name") != name or group.get("lr_scale") != factor or group["lr"] != base_lr * factor
        for group, (name, factor) in zip(optimizer.param_groups, expected)
    ):
        raise ValueError("unexpected transfer optimizer groups/base rates")
    warmup = training.get("warmup_epochs", 10)
    maximum = training["epochs"]
    min_ratio = min(max(float(training.get("min_lr", 0.0)) / base_lr, 0.0), 1.0)

    def multiplier(current_epoch):
        if current_epoch < warmup:
            return float(current_epoch + 1) / float(max(1, warmup))
        progress = float(current_epoch - warmup) / float(max(1, maximum - warmup))
        progress = min(max(progress, 0.0), 1.0)
        return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, [multiplier, multiplier])


def describe_transfer_optimizer(model, optimizer, config, transfer_provenance=None):
    """Freeze startup metadata; call after scheduler creation, before training."""
    training = config["training"]
    scale = validate_transfer_optimizer_request(training)
    named = list(model.named_parameters())
    by_id = {id(parameter): (name, parameter) for name, parameter in named}
    slow_names = _verified_encoder_names(model, training, transfer_provenance) if scale < 1 else set()
    base_lr = float(training["learning_rate"])
    scheduler_type = training.get("scheduler", "cosine").lower()
    minimum = float(training.get("min_lr", 1e-8 if scheduler_type == "plateau" else 0.0))
    minimum = min(max(minimum, 0.0), base_lr) if scheduler_type == "cosine" else minimum if scheduler_type == "plateau" else 0.0
    groups, visited = [], []
    expected_names = [[name for name, _ in named]] if scale == 1 else [
        [name for name, _ in named if name not in slow_names],
        [name for name, _ in named if name in slow_names],
    ]
    if len(optimizer.param_groups) != len(expected_names):
        raise ValueError("optimizer group count differs from requested transfer LR policy")
    for index, group in enumerate(optimizer.param_groups):
        ids = [id(parameter) for parameter in group["params"]]
        if any(identifier not in by_id for identifier in ids):
            raise ValueError("optimizer contains a parameter outside the model")
        names = [by_id[identifier][0] for identifier in ids]
        if names != expected_names[index]:
            raise ValueError("optimizer parameter names/order differ from transfer LR policy")
        factor = 1.0 if index == 0 else scale
        groups.append({"index": index, "name": "base" if index == 0 else "pretrained_encoder",
                       "param_names": names, "num_tensors": len(names),
                       "numel": sum(by_id[identifier][1].numel() for identifier in ids),
                       "lr_scale": factor, "base_lr": base_lr * factor,
                       "floor_lr": minimum * factor, "initial_lr": float(group["lr"])})
        visited.extend(ids)
    if len(visited) != len(set(visited)) or set(visited) != set(by_id):
        raise ValueError("optimizer parameter coverage has omissions or duplicates")
    return {
        "schema_version": 1, "policy": POLICY, "transfer_encoder_lr_scale": scale,
        "transfer_scope": (transfer_provenance or {}).get("scope", "none"),
        "transfer_checkpoint_sha256": (transfer_provenance or {}).get("checkpoint_sha256"),
        "transfer_provenance_sha256": _canonical_sha256(transfer_provenance) if transfer_provenance else None,
        "optimizer_type": config["optimizer"].get("type", "adamw").lower(),
        "scheduler": {"type": scheduler_type, "warmup_epochs": training.get("warmup_epochs", 10),
                      "max_epochs": training["epochs"], "min_lr_ratio": minimum / base_lr if base_lr > 0 else 0.0},
        "groups": groups, "total_parameter_tensors": len(named),
        "total_numel": sum(parameter.numel() for _, parameter in named),
        "base_lr_partial_parameter_names": list((transfer_provenance or {}).get("partially_copied_keys", [])),
        "initial_lr_definition": "effective first-epoch LR after scheduler construction; optimizer initial_lr itself is base_lr",
        "floor_application": "cosine phase only; unchanged linear warmup may start below the floor",
        "slow_parameter_policy": "fully_copied_keys minus bond_theory_proj and bond_head.0; partial AA/position/fusion and all random/condition/edge_encoder/global/bond-readout tensors keep base LR",
    }


def get_optimizer_provenance(model, optimizer, config, transfer_provenance=None, resume_checkpoint=None):
    """Opt in explicitly, preserving legacy checkpoint/q paths for older configs."""
    saved = ((resume_checkpoint or {}).get("extra_state") or {}).get("optimizer_provenance")
    if saved is not None:
        if not isinstance(saved, dict):
            raise ValueError("malformed resumed optimizer provenance")
        return deepcopy(saved)
    if "transfer_encoder_lr_scale" not in config["training"]:
        return None
    return describe_transfer_optimizer(model, optimizer, config, transfer_provenance)


def write_optimizer_provenance(provenance, checkpoint_dir):
    if provenance is None:
        return None
    path = Path(checkpoint_dir) / "optimizer_provenance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(provenance, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path
