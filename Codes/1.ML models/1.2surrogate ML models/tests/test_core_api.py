from __future__ import annotations

import sys
import unittest
from pathlib import Path
import contextlib
import io
import tempfile
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surrogate_core.constants import FINGERPRINT, TARGETS
from surrogate_core.fingerprint import featurize_hashed_morgan_counts
from surrogate_core.inference import _predict_file, target_directory
from surrogate_core.training import _read_precomputed_features


class CoreApiTests(unittest.TestCase):
    def test_public_targets_and_fingerprint_are_fixed(self) -> None:
        self.assertEqual(len(TARGETS), 8)
        self.assertEqual(FINGERPRINT, {"type": "hashed_morgan_counts", "radius": 2, "n_bits": 1024, "use_chirality": False})

    def test_morgan_featurization_matches_release_shape(self) -> None:
        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            matrix, valid = featurize_hashed_morgan_counts(["COCOC", "not-a-smiles"])
        self.assertEqual(matrix.shape, (2, 1024))
        self.assertEqual(matrix.dtype, np.float32)
        self.assertEqual(valid.tolist(), [True, False])
        self.assertGreater(float(matrix[0].sum()), 0.0)
        self.assertEqual(float(matrix[1].sum()), 0.0)
        self.assertEqual(captured.getvalue(), "")

    def test_target_directory_rejects_unknown_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown electronic-property target"):
            target_directory(ROOT, "unknown")

    def test_precomputed_feature_width_must_be_1024(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.csv"
            path.write_text("0,1\n1.0,2.0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "1024"):
                _read_precomputed_features(path)

    def test_fingerprint_uses_current_quiet_rdkit_api(self) -> None:
        source = (ROOT / "surrogate_core" / "fingerprint.py").read_text(encoding="utf-8")
        self.assertNotIn("GetHashedMorganFingerprint", source)
        self.assertIn("BlockLogs", source)

    def test_inference_suppresses_tensorflow_runtime_chatter(self) -> None:
        source = (ROOT / "surrogate_core" / "inference.py").read_text(encoding="utf-8")
        self.assertIn('setLevel("ERROR")', source)

    def test_prediction_file_replaces_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.csv"
            output_path = Path(directory) / "output.csv"
            pd.DataFrame({"smiles": ["COCOC", "CC"]}).to_csv(input_path, index=False)
            output_path.write_text("stale\n", encoding="utf-8")
            fake_model = mock.Mock()
            fake_model.predict.return_value = np.array([[1.0], [2.0]])
            fake_scaler = mock.Mock()
            fake_scaler.inverse_transform.side_effect = lambda values: values
            with (
                mock.patch(
                    "surrogate_core.inference.featurize_hashed_morgan_counts",
                    return_value=(np.zeros((2, 1024), dtype=np.float32), np.ones(2, dtype=bool)),
                ),
                mock.patch(
                    "surrogate_core.inference._load_artifacts",
                    return_value={"model": fake_model, "y_scaler": fake_scaler},
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                _predict_file(ROOT, input_path, output_path, ("VIP",), 5000, 4096, None, None)
            self.assertNotIn("stale", output_path.read_text(encoding="utf-8"))

    def test_chunked_prediction_loads_each_model_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.csv"
            output_path = Path(directory) / "output.csv"
            pd.DataFrame({"smiles": ["COCOC", "CC", "CCC"]}).to_csv(input_path, index=False)
            fake_model = mock.Mock()
            fake_model.predict.side_effect = lambda values, **_: np.zeros((len(values), 1), dtype=np.float32)
            fake_scaler = mock.Mock()
            fake_scaler.inverse_transform.side_effect = lambda values: values
            with (
                mock.patch(
                    "surrogate_core.inference.featurize_hashed_morgan_counts",
                    side_effect=lambda smiles: (
                        np.zeros((len(smiles), 1024), dtype=np.float32),
                        np.ones(len(smiles), dtype=bool),
                    ),
                ),
                mock.patch(
                    "surrogate_core.inference._load_artifacts",
                    return_value={"model": fake_model, "y_scaler": fake_scaler},
                ) as load_artifacts,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                _predict_file(ROOT, input_path, output_path, ("VIP",), 2, 4096, None, None)
            self.assertEqual(load_artifacts.call_count, 1)


if __name__ == "__main__":
    unittest.main()
