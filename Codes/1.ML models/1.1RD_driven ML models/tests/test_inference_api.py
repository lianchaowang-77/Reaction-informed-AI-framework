from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdml.data import RD51_FEATURE_NAMES
from rdml.inference import predict_ensemble


MODEL_DIRS = {
    "dnn": "DNN",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "svr": "SVR",
    "lgbm": "LGBM",
    "xgb": "XGB",
    "rf": "RF",
    "et": "ET",
}


class InferenceApiTests(unittest.TestCase):
    @staticmethod
    def write_features(directory: str) -> Path:
        feature_csv = Path(directory) / "features.csv"
        labels = ",".join(RD51_FEATURE_NAMES)
        row1 = ",".join("0" for _ in range(51))
        row2 = ",".join("1" for _ in range(51))
        feature_csv.write_text(f"{labels}\n{row1}\n{row2}\n", encoding="utf-8")
        return feature_csv

    def test_all_released_ensembles_predict_finite_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            feature_csv = self.write_features(temporary)
            with warnings.catch_warnings(record=True) as emitted:
                warnings.simplefilter("always")
                for model_key, directory in MODEL_DIRS.items():
                    with self.subTest(model=model_key):
                        result = predict_ensemble(
                            model_key,
                            feature_csv,
                            ROOT / directory / "pretrained",
                        )
                        self.assertEqual(result.shape, (2, 7))
                        self.assertTrue(np.isfinite(result.to_numpy(dtype=float)).all())
                        self.assertEqual(
                            list(result.columns),
                            [
                                "seed1",
                                "seed2",
                                "seed3",
                                "seed4",
                                "seed5",
                                "predicted_log_kH_mean",
                                "predicted_log_kH_std",
                            ],
                        )
                self.assertEqual([str(item.message) for item in emitted], [])

    def test_single_released_member_can_be_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            feature_csv = self.write_features(temporary)
            result = predict_ensemble("ridge", feature_csv, ROOT / "Ridge" / "pretrained", seeds=[3])
            self.assertEqual(list(result.columns), ["seed3", "predicted_log_kH_mean", "predicted_log_kH_std"])
            np.testing.assert_allclose(result["seed3"], result["predicted_log_kH_mean"])
            np.testing.assert_allclose(result["predicted_log_kH_std"], 0.0)

    def test_mismatched_model_artifacts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            feature_csv = self.write_features(temporary)
            with self.assertRaisesRegex(ValueError, "Artifact/model mismatch"):
                predict_ensemble("ridge", feature_csv, ROOT / "Lasso" / "pretrained", seeds=[1])


if __name__ == "__main__":
    unittest.main()
