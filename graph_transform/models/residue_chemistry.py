"""Fixed structural descriptors for DBond's 20 standard + B/O/X/Z residues.

Element counts describe a neutral amino acid minus H2O (a peptide residue),
consistent with data/theory_features.py and PBCLA/utils.py. Functional groups
are structural indicators, NOT pKa, gas-phase basicity or cleavage rules.
The AA identity embedding remains available (e.g. to distinguish I and L).
"""

import torch
from torch import nn


ELEMENTS = ("C", "H", "N", "O", "S")
RESIDUE_COMPOSITION = {
    "A": (3, 5, 1, 1, 0), "R": (6, 12, 4, 1, 0),
    "N": (4, 6, 2, 2, 0), "D": (4, 5, 1, 3, 0),
    "C": (3, 5, 1, 1, 1), "E": (5, 7, 1, 3, 0),
    "Q": (5, 8, 2, 2, 0), "G": (2, 3, 1, 1, 0),
    "H": (6, 7, 3, 1, 0), "I": (6, 11, 1, 1, 0),
    "L": (6, 11, 1, 1, 0), "K": (6, 12, 2, 1, 0),
    "M": (5, 9, 1, 1, 1), "F": (9, 9, 1, 1, 0),
    "P": (5, 7, 1, 1, 0), "S": (3, 5, 1, 2, 0),
    "T": (4, 7, 1, 2, 0), "W": (11, 10, 2, 1, 0),
    "Y": (9, 9, 1, 2, 0), "V": (5, 9, 1, 1, 0),
    "B": (3, 6, 2, 1, 0),   # D-Dap: side chain -CH2-NH2
    "O": (5, 10, 2, 1, 0),  # D-Orn: side chain -(CH2)3-NH2
    "X": (8, 8, 2, 1, 0),   # 3-(3-pyridyl)-D-Ala
    "Z": (9, 15, 1, 1, 0),  # D-Cha: side chain -CH2-cyclohexyl
}

# All side-chain flags except backbone_N_substituted; aromatic ring count W=2.
GROUP_MEMBERS = {
    "primary_amine": "BOK", "guanidino": "R", "imidazole": "H",
    "pyridyl": "X", "carboxyl": "DE", "amide": "NQ",
    "hydroxyl": "STY", "thiol": "C", "thioether": "M",
    "aromatic": "FHWYX",
    "saturated_carbocycle": "Z", "backbone_N_substituted": "P",
}
CHEMISTRY_FEATURE_NAMES = ELEMENTS + tuple(GROUP_MEMBERS)
CHEMISTRY_DIM = len(CHEMISTRY_FEATURE_NAMES)
# Fixed scales, independent of dataset/split composition; ring count scaled by 2.
CHEMISTRY_SCALES = (11, 20, 4, 3, 1) + (1,) * 9 + (2, 1, 1)


def residue_descriptor(residue):
    """Unscaled structural descriptor; unknown symbols fail closed."""
    if residue not in RESIDUE_COMPOSITION:
        raise ValueError(f"Unknown residue chemistry: {residue}")
    groups = [float(residue in members) for members in GROUP_MEMBERS.values()]
    if residue == "W":
        groups[list(GROUP_MEMBERS).index("aromatic")] = 2.0
    return tuple(RESIDUE_COMPOSITION[residue]) + tuple(groups)


class ResidueChemistryEncoder(nn.Module):
    """Small residual node embedding; starts as an exact zero correction."""

    def __init__(self, char_to_idx, vocab_size, hidden_dim, chemistry_dim=32):
        super().__init__()
        if chemistry_dim <= 0:
            raise ValueError("residue_chemistry_dim must be positive")
        lookup = torch.zeros(vocab_size, CHEMISTRY_DIM)
        scales = torch.tensor(CHEMISTRY_SCALES, dtype=torch.float32)
        for residue, idx in char_to_idx.items():
            if idx != 0:  # padding must stay zero, irrespective of pad symbol
                lookup[idx] = torch.tensor(residue_descriptor(residue)) / scales
        self.register_buffer("lookup", lookup, persistent=False)
        self.project = nn.Sequential(
            nn.Linear(CHEMISTRY_DIM, chemistry_dim), nn.SiLU(),
            nn.Linear(chemistry_dim, hidden_dim),
        )
        nn.init.zeros_(self.project[-1].weight)
        nn.init.zeros_(self.project[-1].bias)

    def forward(self, tokens):
        correction = self.project(self.lookup[tokens])
        return correction.masked_fill((tokens == 0).unsqueeze(-1), 0.0)
