"""Manifest-bound, allowlisted initialization from an ion4 checkpoint.

This is initialization, never resume: no conditions, output classifier, optimizer,
or training state is restored. All checks precede any parameter mutation.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import torch

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
EXPECTED_EXCLUSION_SETS = {
    "mipd": (513, "ee70d1ede41b7359b74eec10a3c501ddf78388055d23ae8de972bb868bd44e64"),
    "nist": (238178, "aad4fe6b2cca66bdaab09fca6aad49db5e3923bb27aa9bd9f8de4c86de596e84"),
}
SOURCE_FILES = (
    "graph_transform/models/graph_transformer.py", "graph_transform/models/utils.py",
    "graph_transform/models/gcn_layers.py", "graph_transform/models/attention_layers.py",
    "graph_transform/data/graph_dataset.py", "graph_transform/data/graph_builder.py",
    "graph_transform/data/preprocessing.py", "graph_transform/data/theory_features.py",
    "graph_transform/pretraining/intensity_model.py", "graph_transform/pretraining/msp_dataset.py",
    "graph_transform/scripts/pretrain_ion_intensity.py", "graph_transform/scripts/prepare_ion_pretraining.py",
)
AA_KEY = "node_encoder.aa_embedding.weight"
POSITION_KEY = "node_encoder.position_embedding.weight"
FUSION_KEY = "node_encoder.node_encoder.0.weight"
NODE_SHARED = {
    "node_encoder.physicochemical_encoder.weight", "node_encoder.physicochemical_encoder.bias",
    "node_encoder.node_encoder.0.bias", "node_encoder.node_encoder.1.weight",
    "node_encoder.node_encoder.1.bias",
}
CONTEXT_KEYS = {"bond_theory_proj.weight", "bond_theory_proj.bias",
                "bond_head.0.weight", "bond_head.0.bias"}
GCN_LEAVES = ("gcn.bias", "gcn.linear.weight", "gcn.linear.bias", "gcn.edge_transform.weight",
              "gcn.edge_transform.bias", "layer_norm.weight", "layer_norm.bias",
              "gate.0.weight", "gate.0.bias")
GAT_LEAVES = ("W", "a", "edge_attention_bias.weight", "edge_attention_bias.bias",
              "edge_message_gate.weight", "edge_message_gate.bias", "layer_norm.weight", "layer_norm.bias")
EDGE_LEAVES = ("edge_type_embedding.weight", "distance_embedding.weight", "edge_encoder.0.weight",
               "edge_encoder.0.bias", "edge_attr_encoder.weight", "edge_attr_encoder.bias",
               "edge_attr_norm.weight", "edge_attr_norm.bias", "edge_attr_fuse.0.weight", "edge_attr_fuse.0.bias")


def _require(condition, message):
    if not condition:
        raise ValueError("Pretrained transfer: " + message)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _sequence_sha(sequences):
    return _sha(("\n".join(sorted(set(sequences))) + "\n").encode("utf-8"))


def _identities(sequences):
    return {seq.replace("I", "L") for seq in sequences}


def _hash_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_transfer_request(training, resume=None):
    """Return False without I/O or RNG operations for a disabled control."""
    scope = training.get("pretrained_transfer_scope", "none")
    path = training.get("pretrained_backbone_path")
    checksum = training.get("pretrained_backbone_sha256")
    _require(scope in ("none", "encoder", "bond_context"), "unknown transfer scope")
    if scope == "none":
        _require(not path and not checksum, "scope none requires a null path and SHA256")
        return False
    _require(isinstance(path, str) and bool(path), "enabled scope requires a checkpoint path")
    _require(_hash_string(checksum), "enabled scope requires a pinned checkpoint SHA256")
    _require(not resume, "--resume cannot be combined with pretrained initialization")
    return True


def _checked_sequences(value, label):
    _require(isinstance(value, list) and value and all(isinstance(s, str) and s and
             s == s.strip().upper() for s in value), label + " must be uppercase sequences")
    _require(value == sorted(set(value)), label + " must be sorted and unique")
    return set(value)


def _resolve(path, parent):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (parent / value).resolve()


def _verify_manifest(checkpoint, target_csv_paths, repo):
    import pandas as pd
    manifest = checkpoint.get("dataset_manifest")
    _require(isinstance(manifest, dict), "missing dataset manifest")
    _require(manifest.get("status") == "completed" and manifest.get("dataset_kind") == "ion_intensity_external"
             and manifest.get("channel_order") == ["b1", "b2", "y1", "y2"], "wrong dataset status/kind/channels")
    _require(_sha(_canonical(manifest)) == checkpoint.get("dataset_manifest_canonical_sha256"),
             "embedded manifest canonical SHA256 mismatch")
    manifest_path = _resolve(checkpoint["dataset_manifest_path"], repo)
    raw = manifest_path.read_bytes()
    _require(_sha(raw) == checkpoint.get("dataset_manifest_sha256"), "manifest file SHA256 mismatch")
    _require(json.loads(raw.decode("utf-8-sig")) == manifest, "manifest file differs from embedded manifest")
    exclusion = manifest.get("exclusion", {})
    _require(checkpoint.get("exclusion_identity") == exclusion, "exclusion identity differs from manifest")
    _require(exclusion.get("identity_rule") == "uppercase_base_sequence_I_to_L", "unknown exclusion identity rule")
    _require(exclusion.get("certification", {}).get("certified_full_exclusion_sets") is True,
             "full exclusion certification is required")
    for name, (count, digest) in EXPECTED_EXCLUSION_SETS.items():
        entry = exclusion.get(name, {})
        _require(entry.get("unique_sequences") == count and entry.get("sequence_set_sha256") == digest,
                 "full " + name + " exclusion count/SHA256 mismatch")
    mipd = exclusion.get("mipd", {})
    mipd_seqs = _checked_sequences(mipd.get("sequences"), "MiPD exclusion")
    _require(len(mipd_seqs) == mipd.get("unique_sequences") == 513, "full MiPD513 exclusion is required")
    _require(_sequence_sha(mipd_seqs) == mipd.get("sequence_set_sha256"), "MiPD sequence hash mismatch")
    _require(_sequence_sha(_identities(mipd_seqs)) == mipd.get("identity_set_sha256"), "MiPD identity hash mismatch")
    nist = exclusion.get("nist", {})
    _require(_hash_string(nist.get("identity_set_sha256")), "missing complete NIST exclusion identity")
    identity_info = exclusion.get("identity_file", {})
    identity_path = _resolve(identity_info["path"], manifest_path.parent)
    identity_bytes = identity_path.read_bytes()
    _require(_sha(identity_bytes) == identity_info.get("sha256"), "exclusion identity file SHA256 mismatch")
    identity_list = identity_bytes.decode("utf-8-sig").splitlines()
    _require(len(identity_list) == identity_info.get("rows") and identity_list == sorted(set(identity_list)),
             "invalid exclusion identity file rows")
    _require(all(s and s == s.strip().upper().replace("I", "L") for s in identity_list),
             "noncanonical exclusion identity")
    excluded = set(identity_list)
    _require(_identities(mipd_seqs) <= excluded, "MiPD identities absent from combined exclusion file")
    external = {}
    for split in ("train", "val"):
        sequences = _checked_sequences(checkpoint.get(split + "_sequences"), "external " + split)
        _require(all(set(s) <= set(STANDARD_AA) for s in sequences), "external nonstandard residues")
        _require(_sequence_sha(sequences) == checkpoint.get(split + "_sequences_sha256"), "external sequence hash mismatch")
        output = manifest.get("outputs", {}).get(split, {})
        _require(output.get("sequences") == sorted(sequences) and output.get("unique_sequences") == len(sequences)
                 and output.get("sequence_set_sha256") == _sequence_sha(sequences), "manifest external sequence mismatch")
        identities = _identities(sequences)
        _require(output.get("sequence_identities") == sorted(identities) and
                 output.get("identity_set_sha256") == _sequence_sha(identities), "manifest external identity mismatch")
        _require(not identities & excluded, "external data overlap excluded NIST/MiPD identities")
        csv_path = _resolve(output["csv"], manifest_path.parent)
        csv_bytes = csv_path.read_bytes()
        _require(_sha(csv_bytes) == output.get("sha256") ==
                 checkpoint.get("input_sha256", {}).get(split + "_csv"), "external CSV SHA256 mismatch")
        frame = pd.read_csv(io.BytesIO(csv_bytes), usecols=["seq"], dtype={"seq": str}, keep_default_na=False)
        actual_sequences = set(frame["seq"].str.strip().str.upper())
        _require(actual_sequences == sequences and len(frame) == output.get("rows"), "external CSV sequence declarations mismatch")
        external[split] = sequences
    _require(not _identities(external["train"]) & _identities(external["val"]), "external train/val identity leakage")
    # Only the seq column is loaded from downstream files; no held-out labels.
    _require(target_csv_paths.get("train") and target_csv_paths.get("test"), "downstream train and test sequence files required")
    target_union, target_info = set(), {}
    for role, value in target_csv_paths.items():
        if not value:
            continue
        path = _resolve(value, repo)
        before = path.stat()
        sequences, rows = set(), 0
        for chunk in pd.read_csv(path, usecols=["seq"], dtype={"seq": str}, keep_default_na=False, chunksize=100000):
            clean = chunk["seq"].str.strip().str.upper()
            _require(not (clean == "").any(), "blank downstream sequence")
            sequences.update(clean.unique())
            rows += len(chunk)
        after = path.stat()
        _require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "downstream CSV changed during read")
        _require(sequences <= mipd_seqs, "downstream sequence absent from full MiPD exclusion")
        _require(not _identities(sequences) & _identities(external["train"] | external["val"]), "downstream/pretraining identity leakage")
        target_union.update(sequences)
        target_info[role] = {"path": str(path), "columns_read": ["seq"], "rows": rows,
                             "unique_sequences": len(sequences), "sequence_set_sha256": _sequence_sha(sequences)}
    _require(target_union == mipd_seqs, "downstream fold files must collectively verify the full excluded MiPD513 set")
    return external, {"manifest_path": str(manifest_path), "manifest_sha256": checkpoint["dataset_manifest_sha256"],
                      "manifest_canonical_sha256": checkpoint["dataset_manifest_canonical_sha256"],
                      "exclusion_identity_file_sha256": identity_info["sha256"],
                      "mipd_sequence_set_sha256": mipd["sequence_set_sha256"],
                      "nist_sequence_set_sha256": nist["sequence_set_sha256"], "downstream_sequence_files": target_info,
                      "full_mipd513_exclusion_verified": True, "external_train_val_identity_disjoint": True}


def _model_keys(config):
    keys = {AA_KEY, POSITION_KEY, FUSION_KEY, *NODE_SHARED, *CONTEXT_KEYS,
            "bond_head.3.weight", "bond_head.3.bias", "global_node_embedding",
            "global_node_proj.0.weight", "global_node_proj.0.bias"}
    keys.update("edge_encoder." + leaf for leaf in EDGE_LEAVES)
    keys.update(f"node_encoder.{group}_encoder.{layer}.{param}" for group in ("state", "env")
                for layer in (0, 3) for param in ("weight", "bias"))
    keys.update(f"gcn_layers.{i}.{leaf}" for i in range(config["num_gcn_layers"]) for leaf in GCN_LEAVES)
    keys.update(f"gat_layers.{i}.{leaf}" for i in range(config["num_gat_layers"]) for leaf in GAT_LEAVES)
    if config["use_condition_film"]:
        keys.update(f"film_mlp.{i}.{p}" for i in (0, 2) for p in ("weight", "bias"))
    return keys


def _verify_structure(source, target):
    _require(isinstance(source, dict) and set(source) == set(target), "unknown or missing model configuration fields")
    exceptions = {"alphabet", "max_seq_len", "env_feature_mask", "use_env_features", "use_condition_film"}
    for key in target:
        if key not in exceptions:
            _require(source[key] == target[key], "model structure mismatch: " + key)
    for cfg in (source, target):
        for key in ("use_global_node", "use_theory_features", "use_position_embedding", "use_physicochemical_features",
                    "use_state_features", "use_edge_features", "use_edge_type_embedding", "use_distance_embedding",
                    "use_raw_edge_attr", "gat_use_edge_bias", "gat_use_edge_gate", "concat_heads"):
            _require(cfg.get(key) is True, "unsupported architecture flag: " + key)
        for key in ("use_fragment_readout", "use_residue_chemistry", "use_deep_supervision", "use_peptide_aux_head"):
            _require(cfg.get(key) is False, "unsupported optional branch: " + key)
        _require(cfg.get("sequence_adapter_type") == "none" and cfg.get("bond_condition_mode") == "none", "unsupported optional adapter")
        _require(cfg.get("state_feature_mask") == [True, True, False], "observed state fields must be masked")
        _require(cfg.get("num_state_features") == 3 and cfg.get("num_env_features") == 2, "unknown metadata layout")
        _require(cfg.get("pad_char") == "#" and len(set(cfg["alphabet"])) == len(cfg["alphabet"]), "unknown token layout")
        _require(set(STANDARD_AA) <= set(cfg["alphabet"]), "missing standard residue tokens")
    _require(source.get("use_condition_film") is False and source.get("use_env_features") is False
             and source.get("env_feature_mask") == [False, False],
             "pretraining must disable FiLM and mask unknown NCE/observed environment")
    _require(target.get("use_env_features") is True and target.get("env_feature_mask") == [True, False],
             "target must use pre-synthesis environment mask")


def load_pretrained_backbone(model, training, target_csv_paths, *, repo_root=None, resume=None):
    """Validate the complete source before atomically copying the allowed tensors."""
    if not validate_transfer_request(training, resume):
        return None
    repo = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[2]
    path = _resolve(training["pretrained_backbone_path"], repo)
    raw = path.read_bytes()
    _require(_sha(raw) == training["pretrained_backbone_sha256"], "checkpoint SHA256 mismatch")
    checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    _require(isinstance(checkpoint, dict) and checkpoint.get("schema_version") == 1, "unsupported checkpoint schema")
    _require(checkpoint.get("schema", 1) == 1, "conflicting checkpoint schema")
    _require(checkpoint.get("task") == "ion_intensity_external" and checkpoint.get("channels") == ["b1", "b2", "y1", "y2"], "wrong source task/channels")
    hashes = checkpoint.get("source_sha256", {})
    _require(set(hashes) == set(SOURCE_FILES), "unknown or incomplete source hash allowlist")
    for relative in SOURCE_FILES:
        _require(_hash_string(hashes[relative]) and _sha((repo / relative).read_bytes()) == hashes[relative],
                 "source hash mismatch: " + relative)
    external, binding = _verify_manifest(checkpoint, target_csv_paths, repo)
    source_cfg, target_cfg = checkpoint.get("model_config"), model.config.to_dict()
    _verify_structure(source_cfg, target_cfg)
    source, target = checkpoint.get("model_state_dict"), model.state_dict()
    _require(isinstance(source, dict) and set(source) == _model_keys(source_cfg), "unknown/missing source state keys")
    _require(set(target) == _model_keys(target_cfg), "unknown target model structure")
    for key, tensor in source.items():
        _require(torch.is_tensor(tensor) and tensor.is_floating_point() and bool(torch.isfinite(tensor).all()), "invalid/nonfinite source tensor: " + key)
        _require(tensor.dtype == target[key].dtype, "source dtype mismatch: " + key)
        if key not in (AA_KEY, POSITION_KEY):
            _require(tensor.shape == target[key].shape, "source shape mismatch: " + key)
    trained_aa = sorted(set("".join(external["train"])))
    position_count = max(map(len, external["train"]))
    _require(checkpoint.get("trained_amino_acids") == trained_aa, "trained amino acid provenance mismatch")
    _require(checkpoint.get("trained_position_count") == position_count, "trained position provenance mismatch")
    _require(source[AA_KEY].shape == (len(source_cfg["alphabet"]) + 1, target_cfg["aa_embedding_dim"]), "source token table shape")
    _require(source[POSITION_KEY].shape == (source_cfg["max_seq_len"], target_cfg["position_embedding_dim"])
             and position_count <= min(source[POSITION_KEY].shape[0], target[POSITION_KEY].shape[0]), "source position table shape/range")
    scope = training["pretrained_transfer_scope"]
    complete = set(NODE_SHARED)
    complete.update(f"gcn_layers.{i}.{leaf}" for i in range(target_cfg["num_gcn_layers"]) for leaf in GCN_LEAVES)
    complete.update(f"gat_layers.{i}.{leaf}" for i in range(target_cfg["num_gat_layers"]) for leaf in GAT_LEAVES)
    if scope == "bond_context":
        complete.update(CONTEXT_KEYS)
    updates = {key: source[key] for key in complete}
    aa_value = target[AA_KEY].detach().clone()
    mappings = []
    for char in trained_aa:
        src_row, dst_row = source_cfg["alphabet"].index(char) + 1, target_cfg["alphabet"].index(char) + 1
        aa_value[dst_row].copy_(source[AA_KEY][src_row])
        mappings.append({"residue": char, "source_row": src_row, "target_row": dst_row})
    updates[AA_KEY] = aa_value
    pos_value = target[POSITION_KEY].detach().clone()
    pos_value[:position_count].copy_(source[POSITION_KEY][:position_count])
    updates[POSITION_KEY] = pos_value
    shared_columns = sum(target_cfg[k] for k in ("aa_embedding_dim", "position_embedding_dim", "physicochemical_dim"))
    _require(target[FUSION_KEY].shape[1] == shared_columns + 64, "unknown node fusion column layout")
    fusion_value = target[FUSION_KEY].detach().clone()
    fusion_value[:, :shared_columns].copy_(source[FUSION_KEY][:, :shared_columns])
    updates[FUSION_KEY] = fusion_value
    provenance = {"schema_version": 1, "scope": scope, "checkpoint_path": str(path),
                  "checkpoint_sha256": training["pretrained_backbone_sha256"], "source_task": checkpoint["task"],
                  "source_sha256": hashes, "dataset_binding": binding,
                  "fully_copied_keys": sorted(complete), "partially_copied_keys": [AA_KEY, POSITION_KEY, FUSION_KEY],
                  "aa_row_mapping": mappings, "copied_position_interval": [0, position_count],
                  "copied_node_fusion_column_interval": [0, shared_columns],
                  "retained_node_fusion_condition_columns": [shared_columns, shared_columns + 64],
                  "retained_entire_keys": sorted(set(target) - set(updates)),
                  "policy": "Preserve all edge_encoder/global-node/state/env/FiLM/final-classifier parameters; preserve special/padding/untrained AA and position rows; no optimizer or schedule transfer.",
                  "target_model_config": target_cfg, "source_model_config": source_cfg}
    # No model construction or random draws occur in this loader.
    with torch.no_grad():
        for key, value in updates.items():
            target[key].copy_(value)
    return provenance


def write_transfer_provenance(provenance, checkpoint_dir):
    if provenance is None:
        return None
    path = Path(checkpoint_dir) / "transfer_provenance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
