from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdml.data import RD51_FEATURE_NAMES
from rdml.registry import MODEL_NAMES, build_estimator, model_search_space


EXPECTED_NAMES = {
    "ridge": "Ridge",
    "lasso": "Lasso",
    "svr": "SVR",
    "lgbm": "LGBM",
    "xgb": "XGB",
    "rf": "RF",
    "et": "ET",
    "dnn": "DNN",
}


class ModelRegistryTests(unittest.TestCase):
    def test_rd_feature_order_is_fixed(self) -> None:
        self.assertEqual(len(RD51_FEATURE_NAMES), 51)

    def test_public_model_names_are_fixed(self) -> None:
        self.assertEqual(MODEL_NAMES, EXPECTED_NAMES)

    def test_legacy_linear_and_kernel_search_spaces_are_preserved(self) -> None:
        self.assertEqual(model_search_space("ridge"), {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]})
        self.assertEqual(model_search_space("lasso"), {"model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]})
        self.assertEqual(
            model_search_space("svr"),
            {
                "model__C": [0.1, 1, 10, 100],
                "model__gamma": ["scale", 0.01, 0.1, 1.0],
                "model__epsilon": [0.01, 0.1, 0.2],
            },
        )

    def test_legacy_tree_search_space_is_preserved(self) -> None:
        expected = {
            "model__n_estimators": [200, 500, 1000],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 1.0],
        }
        self.assertEqual(model_search_space("rf"), expected)
        self.assertEqual(model_search_space("et"), expected)

    def test_sklearn_models_keep_standard_x_and_robust_y_scaling(self) -> None:
        estimator, search_space = build_estimator("ridge", seed=1, y_scaler="robust")
        self.assertEqual(estimator.transformer.__class__.__name__, "RobustScaler")
        self.assertEqual(estimator.regressor.named_steps["x_scaler"].__class__.__name__, "StandardScaler")
        self.assertEqual(search_space, {"regressor__model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]})


if __name__ == "__main__":
    unittest.main()
