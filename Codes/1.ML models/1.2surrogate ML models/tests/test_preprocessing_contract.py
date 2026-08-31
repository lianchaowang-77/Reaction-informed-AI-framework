from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surrogate_core.preprocessing import IdentityScaler, load_identity_scaler


class PreprocessingContractTests(unittest.TestCase):
    def test_legacy_identity_scaler_is_portably_loadable(self) -> None:
        scaler = load_identity_scaler(ROOT / "VIP" / "pretrained" / "Xdesc_scaler.pkl")
        self.assertIsInstance(scaler, IdentityScaler)
        values = np.array([[1.0, 2.0]], dtype=np.float32)
        np.testing.assert_array_equal(scaler.transform(values), values)


if __name__ == "__main__":
    unittest.main()
