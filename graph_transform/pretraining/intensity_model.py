"""External ion-intensity task, isolated from production classification training.

Channels are [b1, b2, y1, y2], all aligned to N-to-C peptide-bond indices.
An unobserved ion is missing supervision, never an observed zero target.
"""
from __future__ import annotations

import json
import pandas as pd
import torch
from torch import nn

from data.graph_dataset import GraphDataset, GraphDataLoader
from data.theory_features import RESIDUE_MONO_MASS
from models.graph_transformer import GraphTransformer
from models.utils import ModelConfig


CHANNELS = ["b1", "b2", "y1", "y2"]
STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
WATER_MASS = 18.0105646863
PROTON_MASS = 1.00727646688


def make_pretraining_config() -> ModelConfig:
    """Match the production backbone geometry, with unknown/observed inputs off."""
    return ModelConfig({
        "alphabet": "#ABCDEFGHIKLMNOPQRSTVWXYZ", "pad_char": "#",
        "hidden_dim": 256, "num_attention_heads": 4, "dropout": 0.1,
        "aa_embedding_dim": 64, "position_embedding_dim": 32,
        "physicochemical_dim": 32, "edge_dim": 32,
        "num_physicochemical_features": 4, "num_state_features": 3, "num_env_features": 2,
        "edge_embedding_dim": 16, "distance_embedding_dim": 16,
        "edge_types": ["sequence", "distance", "functional", "long_range", "global"],
        "concat_heads": True, "leaky_relu_slope": 0.2,
        "use_edge_features": True, "use_edge_type_embedding": True,
        "use_distance_embedding": True, "use_raw_edge_attr": True,
        "gat_use_edge_bias": True, "gat_use_edge_gate": True,
        "bond_use_edge_repr": True, "bond_use_diff_feature": True,
        "bond_use_product_feature": True, "use_position_embedding": True,
        "use_physicochemical_features": True,
        "num_gcn_layers": 3, "num_gat_layers": 2, "num_classes": 1,
        "max_seq_len": 32, "max_distance": 6, "use_global_node": True,
        "use_long_range_edges": False, "long_range_hops": 1, "long_range_stride": 10,
        "use_theory_features": True,
        "use_state_features": True, "state_feature_mask": [True, True, False],
        "use_env_features": False, "env_feature_mask": [False, False],
        "env_feature_name": "scan_num", "env_feature_scale": 1.0,
        "use_condition_film": False, "use_deep_supervision": False,
        "use_peptide_aux_head": False, "use_fragment_readout": False,
        "use_residue_chemistry": False, "sequence_adapter_type": "none",
        "bond_condition_mode": "none", "bond_condition_dim": 32,
        "fragment_readout_dim": 64, "residue_chemistry_dim": 32,
        "sequence_adapter_dim": 96, "sequence_conv_kernels": [3, 5, 7],
    })


def validate_pretraining_config(config: ModelConfig) -> None:
    expected = {
        "use_state_features": True, "state_feature_mask": [True, True, False],
        "use_env_features": False, "env_feature_mask": [False, False],
        "use_condition_film": False, "use_theory_features": True,
        "use_deep_supervision": False, "use_peptide_aux_head": False,
        "use_fragment_readout": False, "use_residue_chemistry": False,
        "sequence_adapter_type": "none", "bond_condition_mode": "none",
    }
    for key, value in expected.items():
        if getattr(config, key, None) != value:
            raise ValueError(f"ion4 requires {key}={value!r}")


def parse_ion_targets(values_json: str, mask_json: str, bonds: int):
    values = torch.tensor(json.loads(values_json), dtype=torch.float32)
    raw_mask = torch.tensor(json.loads(mask_json))
    if values.shape != (bonds, 4) or raw_mask.shape != values.shape:
        raise ValueError(f"ion arrays must have shape ({bonds}, 4)")
    if not torch.all((raw_mask == 0) | (raw_mask == 1)):
        raise ValueError("ion_mask_json must contain only bool/0/1")
    mask = raw_mask.bool()
    if not mask.any():
        raise ValueError("each spectrum must contain at least one supervised ion")
    active = values[mask]
    if not torch.isfinite(active).all() or torch.any((active < 0) | (active > 1)):
        raise ValueError("observed ion intensities must be finite and in [0, 1]")
    # Discard any missing-slot placeholder before graph/model/loss consumption.
    return torch.where(mask, values, 0.0), mask


class IonIntensityDataset(GraphDataset):
    """Reuse graph construction; never read classification or observed-feature columns."""

    def __init__(self, csv_path, config, split="train"):
        validate_pretraining_config(config)
        full_config = {"model": config.to_dict(), "data": {
            "use_soft_labels": False, "weighting_scheme": "none",
            "strict_label_mode": "raise", "env_feature_name": "scan_num",
        }}
        super().__init__(str(csv_path), full_config, max_seq_len=config.max_seq_len,
                         graph_strategy="distance", augmentation=False, split=split)

    def _load_data(self):
        data = pd.read_csv(self.csv_path, usecols=[
            "seq", "charge", "ion_intensities_json", "ion_mask_json"],
            dtype={"seq": str})
        if data.empty:
            raise ValueError(f"empty external dataset: {self.csv_path}")
        self.ion_targets, self.ion_masks = [], []
        masses, placeholders = [], []
        for index, row in enumerate(data.itertuples(index=False)):
            seq = row.seq
            if not isinstance(seq, str) or not 2 <= len(seq) <= self.max_seq_len:
                raise ValueError(f"invalid sequence length at row {index}")
            if not set(seq) <= STANDARD_AA:
                raise ValueError(f"external pretraining accepts standard unmodified AA only: row {index}")
            charge = float(row.charge)
            if not 1 <= charge <= 6 or not charge.is_integer():
                raise ValueError(f"unknown/unsupported precursor charge at row {index}")
            values, mask = parse_ion_targets(row.ion_intensities_json, row.ion_mask_json, len(seq) - 1)
            self.ion_targets.append(values)
            self.ion_masks.append(mask)
            neutral_mass = sum(RESIDUE_MONO_MASS[aa] for aa in seq) + WATER_MASS
            masses.append((neutral_mass + charge * PROTON_MASS) / charge)
            placeholders.append(";".join(["0"] * (len(seq) - 1)))
        data["pep_mass"] = masses
        for key in ("nce", "intensity", "scan_num", "rt"):
            data[key] = 0.0
        data["true_multi"] = placeholders  # GraphDataset's structural contract only.
        return data.reset_index(drop=True)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["ion_targets"] = self.ion_targets[index]
        sample["ion_mask"] = self.ion_masks[index]
        return sample


def collate_ion_intensity(samples):
    # The existing collator only uses its batch argument (no loader state).
    batch = GraphDataLoader._default_collate_fn(None, samples)
    targets = torch.zeros(len(samples), batch["max_bonds"], 4)
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for i, sample in enumerate(samples):
        count = sample["seq_len"] - 1
        targets[i, :count] = sample["ion_targets"]
        mask[i, :count] = sample["ion_mask"]
    batch["ion_targets"], batch["ion_mask"] = targets, mask
    for key in ("labels", "label_mask", "soft_labels", "sample_weights"):
        batch.pop(key, None)
    return batch


def masked_l1_per_spectrum(predictions, targets, mask):
    """Return one equally weighted loss per spectrum, excluding missing/padded slots."""
    if predictions.shape != targets.shape or mask.shape != targets.shape or mask.dtype != torch.bool:
        raise ValueError("matching [batch, bonds, 4] tensors and a bool mask are required")
    if predictions.ndim != 3 or predictions.shape[-1] != 4:
        raise ValueError("ion4 tensors must have shape [batch, bonds, 4]")
    counts = mask.sum(dim=(1, 2))
    if torch.any(counts == 0):
        raise ValueError("a spectrum has no supervised ion slots")
    if not torch.isfinite(predictions[mask]).all() or not torch.isfinite(targets[mask]).all():
        raise ValueError("nonfinite supervised ion prediction/target")
    # Mask before subtracting: NaN placeholders must not contaminate gradients.
    errors = (torch.where(mask, predictions.float(), 0.0)
              - torch.where(mask, targets.float(), 0.0)).abs()
    return errors.sum(dim=(1, 2)) / counts


def masked_l1(predictions, targets, mask):
    return masked_l1_per_spectrum(predictions, targets, mask).mean()


class IonIntensityModel(nn.Module):
    """Read the existing post-ReLU, pre-dropout bond representation with one ion head."""

    def __init__(self, config=None):
        super().__init__()
        config = config or make_pretraining_config()
        validate_pretraining_config(config)
        self.backbone = GraphTransformer(config)
        self.ion_head = nn.Linear(config.hidden_dim, 4)
        # The old classifier is only traversed to reuse the unchanged forward method.
        self.backbone.bond_head[3].requires_grad_(False)
        self._bond_hidden = None
        self.backbone.bond_head[1].register_forward_hook(self._capture_hidden)

    def _capture_hidden(self, module, inputs, output):
        self._bond_hidden = output

    @staticmethod
    def _sanitize_inputs(batch):
        clean = dict(batch)
        for key in ("nces", "intensities", "rts", "secondary_envs", "env_vars", "scan_nums"):
            if key in clean and torch.is_tensor(clean[key]):
                clean[key] = torch.zeros_like(clean[key])
        if "state_vars" in clean:
            clean["state_vars"] = clean["state_vars"].clone()
            clean["state_vars"][..., 2] = 0.0
        return clean

    def forward(self, batch):
        self._bond_hidden = None
        clean = self._sanitize_inputs(batch)
        try:
            self.backbone(clean)
            hidden = self._bond_hidden
            if hidden is None:
                raise ValueError("ion4 requires at least one peptide bond")
            flat = torch.sigmoid(self.ion_head(hidden))
            lengths = clean["seq_lens"].to(flat.device)
            valid = torch.arange(clean["max_bonds"], device=flat.device)[None, :] < (lengths - 1)[:, None]
            if int(valid.sum()) != flat.shape[0]:
                raise RuntimeError("backbone bond order/shape no longer matches ion targets")
            output = flat.new_zeros((len(lengths), clean["max_bonds"], 4))
            output[valid] = flat
            return output
        finally:
            self._bond_hidden = None
