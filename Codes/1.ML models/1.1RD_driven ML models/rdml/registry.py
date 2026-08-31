"""Model definitions and hyperparameter spaces preserved from the legacy run."""

from __future__ import annotations

from typing import Any

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR


MODEL_NAMES = {
    "ridge": "Ridge",
    "lasso": "Lasso",
    "svr": "SVR",
    "lgbm": "LGBM",
    "xgb": "XGB",
    "rf": "RF",
    "et": "ET",
    "dnn": "DNN",
}


def model_search_space(model_key: str) -> dict[str, list[Any]]:
    key = model_key.strip().lower()
    if key in {"et", "rf"}:
        return {
            "model__n_estimators": [200, 500, 1000],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 1.0],
        }
    if key == "svr":
        return {
            "model__C": [0.1, 1, 10, 100],
            "model__gamma": ["scale", 0.01, 0.1, 1.0],
            "model__epsilon": [0.01, 0.1, 0.2],
        }
    if key == "ridge":
        return {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    if key == "lasso":
        return {"model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]}
    if key == "xgb":
        return {
            "model__max_depth": [2, 3, 4, 6],
            "model__learning_rate": [0.01, 0.03, 0.1],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__min_child_weight": [1, 5, 10],
            "model__reg_lambda": [0.0, 1.0, 10.0],
        }
    if key == "lgbm":
        return {
            "model__num_leaves": [15, 31, 63, 127],
            "model__learning_rate": [0.01, 0.03, 0.1],
            "model__min_child_samples": [5, 10, 20, 40],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__reg_lambda": [0.0, 1.0, 10.0],
        }
    if key == "dnn":
        return {}
    raise ValueError(f"Unsupported model key: {model_key}")


def _base_estimator(model_key: str, seed: int):
    key = model_key.strip().lower()
    if key == "et":
        return ExtraTreesRegressor(random_state=int(seed))
    if key == "rf":
        return RandomForestRegressor(random_state=int(seed))
    if key == "svr":
        return SVR(kernel="rbf")
    if key == "ridge":
        return Ridge(random_state=int(seed))
    if key == "lasso":
        return Lasso(random_state=int(seed), max_iter=200000)
    if key == "xgb":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover
            raise ModuleNotFoundError("Install xgboost to train the XGB model.") from exc
        return XGBRegressor(
            random_state=int(seed),
            n_estimators=800,
            objective="reg:squarederror",
            verbosity=0,
            n_jobs=1,
        )
    if key == "lgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:  # pragma: no cover
            raise ModuleNotFoundError("Install lightgbm to train the LGBM model.") from exc
        return LGBMRegressor(
            random_state=int(seed),
            n_estimators=2000,
            verbosity=-1,
            verbose=-1,
            n_jobs=1,
        )
    raise ValueError(f"{model_key!r} is not a conventional estimator.")


def build_estimator(model_key: str, seed: int, y_scaler: str = "robust"):
    base = _base_estimator(model_key, seed)
    pipeline = Pipeline([("x_scaler", StandardScaler()), ("model", base)])
    mode = y_scaler.strip().lower()
    if mode == "none":
        return pipeline, model_search_space(model_key)
    if mode == "robust":
        transformer = RobustScaler()
    elif mode == "standard":
        transformer = StandardScaler()
    else:
        raise ValueError("y_scaler must be one of: none, robust, standard")
    estimator = TransformedTargetRegressor(regressor=pipeline, transformer=transformer)
    search_space = {f"regressor__{name}": values for name, values in model_search_space(model_key).items()}
    return estimator, search_space

