from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdml.data import RD51_FEATURE_NAMES, load_split_csv, read_legacy_feature_matrix


class DataContractTests(unittest.TestCase):
    def test_reordered_rd_features_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.csv"
            labels = list(RD51_FEATURE_NAMES)
            labels[0], labels[1] = labels[1], labels[0]
            path.write_text(",".join(labels) + "\n" + ",".join(["0"] * 51) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "labels or their order"):
                read_legacy_feature_matrix(path)

    def test_optional_validation_rows_are_not_added_to_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.csv"
            pd.DataFrame(
                {"smiles": ["A", "B", "C"], "split": ["train", "val", "test"]}
            ).to_csv(path, index=False)
            train_idx, test_idx = load_split_csv(path, ["A", "B", "C"])
            self.assertEqual(train_idx.tolist(), [0])
            self.assertEqual(test_idx.tolist(), [2])


if __name__ == "__main__":
    unittest.main()
