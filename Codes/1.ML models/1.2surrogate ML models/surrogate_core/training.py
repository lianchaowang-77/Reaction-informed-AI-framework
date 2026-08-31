"""Focused retraining code for the released single-task DNN22 surrogates."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .constants import FINGERPRINT, MODEL_SETTINGS, TARGETS
from .fingerprint import featurize_hashed_morgan_counts
from .preprocessing import IdentityScaler


def _find_smiles_column(frame: pd.DataFrame) -> str:
    for column in frame.columns:
        if str(column).strip().lower() == "smiles":
            return str(column)
    raise ValueError(f"No SMILES column was found. Available columns: {list(frame.columns)}")


def _read_precomputed_features(feature_csv: str | Path) -> np.ndarray:
    path = Path(feature_csv)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        labels = next(csv.reader(handle))
    frame = pd.read_csv(path, header=None, skiprows=1)
    if frame.shape[1] != len(labels):
        raise ValueError(
            f"Feature dimension mismatch: {frame.shape[1]} data columns versus {len(labels)} labels."
        )
    if frame.shape[1] != int(FINGERPRINT["n_bits"]):
        raise ValueError(
            f"Expected {FINGERPRINT['n_bits']} precomputed Morgan features, received {frame.shape[1]}."
        )
    return frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32, copy=True)


def _align_features(
    dataset_smiles: list[str],
    features: np.ndarray,
    feature_smiles_csv: str | Path,
) -> np.ndarray:
    frame = pd.read_csv(feature_smiles_csv)
    smiles_column = _find_smiles_column(frame)
    source_smiles = frame[smiles_column].astype(str).str.strip().tolist()
    if len(source_smiles) != len(features):
        raise ValueError("The precomputed feature matrix and feature-SMILES table have different row counts.")
    if len(source_smiles) != len(set(source_smiles)):
        raise ValueError("Duplicate SMILES were found in the feature-SMILES table.")
    index = {smiles: row for row, smiles in enumerate(source_smiles)}
    missing = [smiles for smiles in dataset_smiles if smiles not in index]
    if missing:
        raise ValueError(f"{len(missing)} dataset SMILES are absent from the feature table: {missing[:5]}")
    return np.vstack([features[index[smiles]] for smiles in dataset_smiles]).astype(np.float32)


def load_training_data(
    dataset_csv: str | Path,
    target: str,
    feature_csv: str | Path | None = None,
    feature_smiles_csv: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    frame = pd.read_csv(dataset_csv)
    smiles_column = _find_smiles_column(frame)
    if target not in frame.columns:
        raise ValueError(f"Target column {target!r} was not found in the dataset.")
    smiles = frame[smiles_column].astype(str).str.strip().tolist()
    y = pd.to_numeric(frame[target], errors="raise").to_numpy(dtype=np.float32).reshape(-1, 1)
    if not np.isfinite(y).all():
        raise ValueError("The target contains NaN or infinite values.")
    if feature_csv:
        if not feature_smiles_csv:
            raise ValueError("--feature_smiles_csv is required with --feature_csv.")
        values = _align_features(smiles, _read_precomputed_features(feature_csv), feature_smiles_csv)
    else:
        values, valid = featurize_hashed_morgan_counts(smiles)
        if not valid.all():
            invalid = [smiles[index] for index in np.flatnonzero(~valid)[:5]]
            raise ValueError(f"The training dataset contains invalid SMILES: {invalid}")
    return values, y, smiles


def _stratified_subsplit(
    indices: np.ndarray,
    y: np.ndarray,
    fraction: float,
    seed: int,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = y[np.asarray(indices, dtype=int)].reshape(-1)
    bins = pd.qcut(values, q=int(n_bins), labels=False, duplicates="drop")
    first, second = train_test_split(
        np.asarray(indices, dtype=int),
        test_size=float(fraction),
        random_state=int(seed),
        stratify=bins,
    )
    return np.sort(np.asarray(first, dtype=int)), np.sort(np.asarray(second, dtype=int))


def make_splits(
    y: np.ndarray,
    seed: int = 1,
    test_fraction: float = 0.15,
    validation_fraction: float = 0.15,
    n_bins: int = 10,
    split_npz: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split_npz:
        loaded = np.load(split_npz)
        return tuple(np.sort(np.asarray(loaded[key], dtype=int)) for key in ("idx_tr", "idx_va", "idx_te"))
    all_indices = np.arange(len(y), dtype=int)
    train_pool, test = _stratified_subsplit(all_indices, y, test_fraction, seed, n_bins)
    train, validation = _stratified_subsplit(train_pool, y, validation_fraction, seed + 17, n_bins)
    return train, validation, test


def build_dnn(input_dim: int):
    try:
        import tensorflow as tf
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError("TensorFlow is required to train the DNN22 surrogate.") from error
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(int(input_dim),))])
    for width in MODEL_SETTINGS["hidden_units"]:
        model.add(tf.keras.layers.Dense(int(width), activation="relu"))
        model.add(tf.keras.layers.Dropout(float(MODEL_SETTINGS["dropout"])))
    model.add(tf.keras.layers.Dense(1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(MODEL_SETTINGS["learning_rate"])),
        loss="mse",
    )
    return model


def train_target(
    target: str,
    dataset_csv: str | Path,
    output_dir: str | Path,
    feature_csv: str | Path | None = None,
    feature_smiles_csv: str | Path | None = None,
    split_npz: str | Path | None = None,
    seed: int = 1,
    test_fraction: float = 0.15,
    validation_fraction: float = 0.15,
    stratified_bins: int = 10,
    epochs: int = MODEL_SETTINGS["epochs_max"],
    batch_size: int = MODEL_SETTINGS["batch_size"],
) -> Path:
    if target not in TARGETS:
        raise ValueError(f"Unknown electronic-property target {target!r}.")
    values, y, _ = load_training_data(dataset_csv, target, feature_csv, feature_smiles_csv)
    train_idx, validation_idx, test_idx = make_splits(
        y,
        seed=seed,
        test_fraction=test_fraction,
        validation_fraction=validation_fraction,
        n_bins=stratified_bins,
        split_npz=split_npz,
    )
    y_scaler = Pipeline(
        [
            ("y_transform", FunctionTransformer(func=None, inverse_func=None, validate=False)),
            ("y_scaler", StandardScaler()),
        ]
    )
    y_train = y_scaler.fit_transform(y[train_idx]).astype(np.float32)
    y_validation = y_scaler.transform(y[validation_idx]).astype(np.float32)

    try:
        import tensorflow as tf
    except ImportError as error:  # pragma: no cover
        raise ModuleNotFoundError("TensorFlow is required to train the DNN22 surrogate.") from error
    np.random.seed(int(seed))
    tf.keras.utils.set_random_seed(int(seed))
    model = build_dnn(values.shape[1])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(MODEL_SETTINGS["early_stopping_patience"]),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=int(MODEL_SETTINGS["reduce_lr_patience"]),
            factor=float(MODEL_SETTINGS["reduce_lr_factor"]),
            verbose=0,
        ),
    ]
    model.fit(
        values[train_idx],
        y_train,
        validation_data=(values[validation_idx], y_validation),
        epochs=int(epochs),
        batch_size=int(batch_size),
        verbose=0,
        callbacks=callbacks,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model.save(destination / "electronic_model.keras")
    with (destination / "Xdesc_scaler.pkl").open("wb") as handle:
        pickle.dump(IdentityScaler(), handle)
    with (destination / "Y_scaler.pkl").open("wb") as handle:
        pickle.dump(y_scaler, handle)
    config = {
        "target": target,
        "seed": int(seed),
        "fingerprint": FINGERPRINT,
        "feature_representation": (
            "precomputed 1024D matrix aligned by SMILES" if feature_csv else "1024D hashed Morgan count fingerprint"
        ),
        "x_scaler": "identity (Morgan count features are used directly)",
        "y_scaler": "StandardScaler fitted on the training subset",
        "model": MODEL_SETTINGS,
        "split": {
            "method": "supplied index archive" if split_npz else "per-target quantile-stratified",
            "test_fraction": float(test_fraction),
            "validation_fraction_of_training_pool": float(validation_fraction),
            "stratified_bins": int(stratified_bins),
            "n_total": int(len(y)),
            "n_train": int(len(train_idx)),
            "n_validation": int(len(validation_idx)),
            "n_test": int(len(test_idx)),
        },
        "training": {"epochs_max": int(epochs), "batch_size": int(batch_size)},
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved {target} surrogate artifacts to {destination}")
    return destination


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_csv", default=None)
    parser.add_argument("--feature_smiles_csv", default=None)
    parser.add_argument("--split_npz", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--stratified_bins", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=MODEL_SETTINGS["epochs_max"])
    parser.add_argument("--batch_size", type=int, default=MODEL_SETTINGS["batch_size"])


def main_for_target(target: str) -> None:
    parser = argparse.ArgumentParser(description=f"Retrain the {target} single-task DNN22 surrogate.")
    _add_training_arguments(parser)
    args = parser.parse_args()
    train_target(
        target,
        args.dataset_csv,
        args.output_dir,
        feature_csv=args.feature_csv,
        feature_smiles_csv=args.feature_smiles_csv,
        split_npz=args.split_npz,
        seed=args.seed,
        test_fraction=args.test_fraction,
        validation_fraction=args.validation_fraction,
        stratified_bins=args.stratified_bins,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


def main_all() -> None:
    parser = argparse.ArgumentParser(description="Retrain all eight single-task DNN22 electronic-property surrogates.")
    _add_training_arguments(parser)
    args = parser.parse_args()
    root = Path(args.output_dir)
    for target in TARGETS:
        train_target(
            target,
            args.dataset_csv,
            root / target,
            feature_csv=args.feature_csv,
            feature_smiles_csv=args.feature_smiles_csv,
            split_npz=args.split_npz,
            seed=args.seed,
            test_fraction=args.test_fraction,
            validation_fraction=args.validation_fraction,
            stratified_bins=args.stratified_bins,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
