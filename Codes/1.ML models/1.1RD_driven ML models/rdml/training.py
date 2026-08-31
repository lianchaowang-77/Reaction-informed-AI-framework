"""Training entry points for the eight RD-driven hydrolysis-rate models."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, StandardScaler

from .data import load_split_csv, load_training_data, make_same_test_split, make_validation_split
from .dnn import DNN3, require_torch
from .registry import MODEL_NAMES, build_estimator


def _parse_seeds(value: str) -> list[int]:
    seeds = []
    for item in value.split(","):
        item = item.strip()
        if item and int(item) not in seeds:
            seeds.append(int(item))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def _safe_config(
    model_key: str,
    seed: int,
    feature_names: list[str],
    best_params: dict | None = None,
    split_method: str = "target-quantile stratified",
    cv_folds: int | None = None,
    n_iter: int | None = None,
) -> dict:
    config = {
        "model": MODEL_NAMES[model_key],
        "model_key": model_key,
        "seed": int(seed),
        "target": "log(k_H)",
        "feature_representation": "51D RD feature matrix",
        "feature_count": len(feature_names),
        "feature_labels": feature_names,
        "split": {"method": split_method},
        "x_scaler": "StandardScaler fitted on training data",
        "y_scaler": "RobustScaler fitted on training data",
    }
    if split_method == "target-quantile stratified":
        config["split"].update({"test_fraction": 0.2, "shared_test_seed": 1})
    if best_params is not None:
        config["best_params"] = best_params
        config["search"] = {
            "method": "RandomizedSearchCV",
            "cv_folds": int(cv_folds),
            "n_iter": int(n_iter),
        }
    return config


def _train_conventional(model_key: str, args, data, train_idx: np.ndarray, split_method: str) -> None:
    for seed in _parse_seeds(args.seeds):
        estimator, space = build_estimator(model_key, seed=seed, y_scaler="robust")
        n_space = int(np.prod([len(values) for values in space.values()])) if space else 0
        n_iter = min(int(args.n_iter), n_space) if n_space else int(args.n_iter)
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=space,
            n_iter=n_iter,
            scoring="neg_root_mean_squared_error",
            cv=KFold(n_splits=int(args.cv_folds), shuffle=True, random_state=seed),
            random_state=seed,
            n_jobs=int(args.n_jobs),
            refit=True,
            verbose=0,
        )
        search.fit(data.X[train_idx], data.y[train_idx])
        fitted = search.best_estimator_
        member_dir = Path(args.output_dir) / f"seed{seed}"
        member_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted, member_dir / "model.pkl")
        regressor = fitted.regressor_ if hasattr(fitted, "regressor_") else fitted
        joblib.dump(regressor.named_steps["x_scaler"], member_dir / "Xdesc_scaler.pkl")
        joblib.dump(fitted.transformer_, member_dir / "Yscaler.pkl")
        config = _safe_config(
            model_key,
            seed,
            data.feature_names,
            search.best_params_,
            split_method=split_method,
            cv_folds=args.cv_folds,
            n_iter=n_iter,
        )
        (member_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(f"Saved {MODEL_NAMES[model_key]} seed{seed} to {member_dir}")


def _train_dnn(args, data, train_idx: np.ndarray, split_method: str) -> None:
    torch = require_torch()
    base_train_idx, val_idx = make_validation_split(train_idx, split_seed=1, val_fraction=0.1)
    for seed in _parse_seeds(args.seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        x_scaler = StandardScaler()
        y_scaler = RobustScaler()
        X_train = x_scaler.fit_transform(data.X[base_train_idx]).astype(np.float32)
        X_val = x_scaler.transform(data.X[val_idx]).astype(np.float32)
        y_train_raw = data.y[base_train_idx].astype(np.float32).reshape(-1, 1)
        y_val_raw = data.y[val_idx].astype(np.float32).reshape(-1, 1)
        y_scaler.fit(y_train_raw)
        y_train = y_scaler.transform(y_train_raw).astype(np.float32)
        y_val = y_scaler.transform(y_val_raw).astype(np.float32)

        model = DNN3(input_dim=data.X.shape[1], hidden=(128, 64, 32, 16))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.MSELoss()
        Xt = torch.tensor(X_train, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.float32)
        Xv = torch.tensor(X_val, dtype=torch.float32)
        yv = torch.tensor(y_val, dtype=torch.float32)
        best_state = None
        best_loss = float("inf")
        stale = 0
        for _ in range(300):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(model(Xt), yt)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_loss = float(loss_fn(model(Xv), yv).item())
            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                stale = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= 30:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)

        member_dir = Path(args.output_dir) / f"seed{seed}"
        member_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), member_dir / "DNN3.pt")
        with (member_dir / "Xdesc_scaler.pkl").open("wb") as handle:
            pickle.dump(x_scaler, handle)
        with (member_dir / "Yscaler.pkl").open("wb") as handle:
            pickle.dump(y_scaler, handle)
        config = _safe_config("dnn", seed, data.feature_names, split_method=split_method)
        config.update(
            {
                "architecture": [128, 64, 32, 16],
                "activation": "ReLU",
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "loss": "mean squared error",
                "epochs_max": 300,
                "early_stopping_patience": 30,
                "validation_fraction": 0.1,
            }
        )
        (member_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(f"Saved DNN seed{seed} to {member_dir}")


def main_for_model(model_key: str) -> None:
    parser = argparse.ArgumentParser(description=f"Train the RD-driven {MODEL_NAMES[model_key]} hydrolysis-rate model.")
    parser.add_argument("--dataset_csv", required=True)
    parser.add_argument("--feature_csv", required=True)
    parser.add_argument("--feature_smiles_csv", required=True)
    parser.add_argument("--target_col", default="LOGk2")
    parser.add_argument("--split_csv", default="", help="Optional fixed split containing SMILES and split columns.")
    parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parents[1] / MODEL_NAMES[model_key] / "trained"))
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--cv_folds", type=int, default=10)
    parser.add_argument("--n_iter", type=int, default=40)
    parser.add_argument("--n_jobs", type=int, default=16)
    args = parser.parse_args()

    data = load_training_data(args.dataset_csv, args.feature_csv, args.feature_smiles_csv, args.target_col)
    if data.X.shape[1] != 51:
        raise ValueError(f"Expected 51 RD features, received {data.X.shape[1]}.")
    if args.split_csv:
        train_idx, _ = load_split_csv(args.split_csv, data.smiles)
        split_method = "supplied split CSV"
    else:
        train_idx, _ = make_same_test_split(data.y, split_seed=1, test_size=0.2, n_bins=10)
        split_method = "target-quantile stratified"
    if model_key == "dnn":
        _train_dnn(args, data, train_idx, split_method)
    else:
        _train_conventional(model_key, args, data, train_idx, split_method)
