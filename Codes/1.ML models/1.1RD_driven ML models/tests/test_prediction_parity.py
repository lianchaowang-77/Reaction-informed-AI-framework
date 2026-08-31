from __future__ import annotations

import json
import pickle
import tempfile
import unittest
import warnings
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdml.data import RD51_FEATURE_NAMES, read_legacy_feature_matrix
from rdml.dnn import DNN3, require_torch
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


def direct_dnn_prediction(member: Path, values: np.ndarray) -> np.ndarray:
    torch = require_torch()
    config = json.loads((member / "config.json").read_text(encoding="utf-8"))
    with (member / "Xdesc_scaler.pkl").open("rb") as handle:
        x_scaler = pickle.load(handle)
    with (member / "Yscaler.pkl").open("rb") as handle:
        y_scaler = pickle.load(handle)
    model = DNN3(input_dim=values.shape[1], hidden=config["architecture"])
    state = torch.load(member / "DNN3.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    scaled = x_scaler.transform(values).astype(np.float32)
    with torch.no_grad():
        predicted = model(torch.tensor(scaled, dtype=torch.float32)).numpy().reshape(-1, 1)
    return y_scaler.inverse_transform(predicted).reshape(-1)


class PredictionParityTests(unittest.TestCase):
    def test_ensemble_columns_equal_direct_member_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            feature_csv = Path(directory) / "features.csv"
            rows = np.vstack([np.zeros(51), np.linspace(-0.25, 0.25, 51)])
            feature_csv.write_text(
                ",".join(RD51_FEATURE_NAMES)
                + "\n"
                + "\n".join(",".join(map(str, row)) for row in rows)
                + "\n",
                encoding="utf-8",
            )
            values, _ = read_legacy_feature_matrix(feature_csv)
            for model_key, public_name in MODEL_DIRS.items():
                pretrained = ROOT / public_name / "pretrained"
                actual = predict_ensemble(model_key, feature_csv, pretrained)
                expected_members = []
                for seed in range(1, 6):
                    member = pretrained / f"seed{seed}"
                    if model_key == "dnn":
                        expected = direct_dnn_prediction(member, values)
                    else:
                        estimator = joblib.load(member / "model.pkl")
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=UserWarning)
                            expected = np.asarray(estimator.predict(values), dtype=float).reshape(-1)
                    expected_members.append(expected)
                    np.testing.assert_allclose(actual[f"seed{seed}"], expected, rtol=1e-7, atol=1e-7)
                matrix = np.column_stack(expected_members)
                np.testing.assert_allclose(actual["predicted_log_kH_mean"], matrix.mean(axis=1))
                np.testing.assert_allclose(actual["predicted_log_kH_std"], matrix.std(axis=1, ddof=0))


if __name__ == "__main__":
    unittest.main()
