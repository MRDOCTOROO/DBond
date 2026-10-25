"""Route the same sequence-derived bond theory before or after propagation.

The early route reuses the existing 15->hidden projection. A residue receives
the mean of its incident valid bond vectors; optional global nodes are never
listed among bond endpoints and receive no direct addition. This is a routing
ablation, not a new source of measured spectra or a learned 3-D structure.
"""

import torch


THEORY_FUSION_MODE_KEY = 'theory_fusion_mode_code'
_EARLY_MODE_CODES = {'early_only': 1, 'early_plus_late': 2}
_CODE_TO_MODE = {code: mode for mode, code in _EARLY_MODE_CODES.items()}


def validate_theory_fusion_stage(value):
    if value not in ('late', 'early_only', 'early_plus_late'):
        raise ValueError(f'Unknown theory_fusion_stage: {value!r}')
    return value


def theory_fusion_mode_code(stage):
    """Only nonlegacy routes need a checkpoint marker; late keeps old keys."""
    stage = validate_theory_fusion_stage(stage)
    return _EARLY_MODE_CODES.get(stage)


def validate_theory_fusion_checkpoint(model, state_dict, checkpoint_config=None):
    """Reject a routing mismatch before any model parameter is copied.

    All three routes share learned parameter shapes, so strict=False cannot
    identify a wrong route. Missing mode markers mean the original late route;
    each early route must retain its exact scalar int64 marker. Optional saved
    config metadata must agree with the state, including legacy missing flags.
    """
    requested = validate_theory_fusion_stage(model.theory_fusion_stage)
    if THEORY_FUSION_MODE_KEY in state_dict:
        marker = state_dict[THEORY_FUSION_MODE_KEY]
        if (not torch.is_tensor(marker) or marker.dtype != torch.int64
                or marker.shape != torch.Size([]) or marker.item() not in _CODE_TO_MODE):
            raise ValueError('Invalid checkpoint theory_fusion_mode_code')
        saved = _CODE_TO_MODE[int(marker.item())]
    else:
        saved = 'late'
    if saved != requested:
        raise ValueError(f'Checkpoint theory_fusion_stage mismatch: {saved!r} != {requested!r}')
    if checkpoint_config is not None:
        if not isinstance(checkpoint_config, dict):
            raise ValueError('Checkpoint config must be a dictionary')
        saved_model_config = checkpoint_config.get('model', checkpoint_config)
        if not isinstance(saved_model_config, dict):
            raise ValueError('Checkpoint model config must be a dictionary')
        declared = validate_theory_fusion_stage(saved_model_config.get('theory_fusion_stage', 'late'))
        if declared != saved:
            raise ValueError('Checkpoint theory_fusion_stage config/state mismatch')


def add_packed_bond_theory(node_features, bond_vectors, bond_src, bond_dst):
    """Add mean incident bond vectors to valid packed residue nodes only.

    ``bond_src`` and ``bond_dst`` must be the existing row-major valid bond
    endpoint indices. Padding and each trailing global node are excluded by
    their construction. Reusing them also handles unequal peptide lengths and
    samples containing fewer than two residues without a separate alignment.
    """
    if (node_features.ndim != 2 or bond_vectors.ndim != 2
            or bond_vectors.size(1) != node_features.size(1)
            or bond_src.ndim != 1 or bond_dst.ndim != 1
            or bond_src.numel() != bond_vectors.size(0)
            or bond_dst.numel() != bond_vectors.size(0)):
        raise ValueError('Early theory fusion requires aligned packed nodes and valid bond endpoints')
    if bond_vectors.size(0) == 0:
        return node_features
    vectors = bond_vectors.to(device=node_features.device, dtype=node_features.dtype)
    additions = torch.zeros_like(node_features)
    additions.index_add_(0, bond_src, vectors)
    additions.index_add_(0, bond_dst, vectors)
    counts = node_features.new_zeros((node_features.size(0), 1))
    ones = counts.new_ones((bond_vectors.size(0), 1))
    counts.index_add_(0, bond_src, ones)
    counts.index_add_(0, bond_dst, ones)
    return node_features + additions / counts.clamp_min(1)
