from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .constants import FINGERPRINT


def featurize_hashed_morgan_counts(
    smiles: Sequence[str],
    radius: int = FINGERPRINT["radius"],
    n_bits: int = FINGERPRINT["n_bits"],
    use_chirality: bool = FINGERPRINT["use_chirality"],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert SMILES to the hashed Morgan count representation used for training."""
    try:
        from rdkit import Chem, rdBase
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError("RDKit is required to calculate Morgan fingerprints.") from error

    matrix = np.zeros((len(smiles), int(n_bits)), dtype=np.float32)
    valid = np.zeros(len(smiles), dtype=bool)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(radius),
        fpSize=int(n_bits),
        includeChirality=bool(use_chirality),
    )
    with rdBase.BlockLogs():
        for row, value in enumerate(smiles):
            molecule = Chem.MolFromSmiles(str(value))
            if molecule is None:
                continue
            fingerprint = generator.GetCountFingerprint(molecule)
            for bit_id, count in fingerprint.GetNonzeroElements().items():
                matrix[row, int(bit_id)] = float(count)
            valid[row] = True
    return matrix, valid
