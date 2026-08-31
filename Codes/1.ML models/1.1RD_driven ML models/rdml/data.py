"""Load and align the 51-dimensional reaction-descriptor matrix."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RD51_FEATURE_NAMES = (
    "220", "200", "216", "199", "204", "185", "175", "207", "184", "176", "206", "211", "183",
    "212", "186", "182", "198", "179", "208", "178", "197", "177", "201", "180", "336", "344",
    "351", "335", "338", "258", "312", "366", "420", "448", "440", "439", "496", "457", "497",
    "458", "503", "536", "537", "538", "539", "540", "541", "542", "543", "544", "545",
)


@dataclass(frozen=True)
class TrainingData:
    X: np.ndarray
    y: np.ndarray
    smiles: list[str]
    feature_names: list[str]


def _find_column_case_insensitive(frame: pd.DataFrame, name: str) -> str:
    for column in frame.columns:
        if str(column).strip().lower() == name.strip().lower():
            return str(column)
    raise ValueError(f"Column {name!r} was not found. Available columns: {list(frame.columns)}")


def read_legacy_feature_matrix(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Read the paper's feature format: labels in row 1, numeric data below."""
    feature_path = Path(path)
    with feature_path.open("r", encoding="utf-8-sig", newline="") as handle:
        first_row = next(csv.reader(handle))
    feature_names = [str(value).strip() for value in first_row if str(value).strip()]
    frame = pd.read_csv(feature_path, header=None, skiprows=1)
    if frame.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature dimension mismatch in {feature_path}: "
            f"{frame.shape[1]} data columns versus {len(feature_names)} labels."
        )
    if tuple(feature_names) != RD51_FEATURE_NAMES:
        raise ValueError(
            "The feature labels or their order do not match the released 51D RD representation. "
            "Use the canonical order documented in README.md."
        )
    matrix = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32, copy=True)
    return matrix, feature_names


def load_training_data(
    dataset_csv: str | Path,
    feature_csv: str | Path,
    feature_smiles_csv: str | Path,
    target_col: str = "LOGk2",
) -> TrainingData:
    dataset = pd.read_csv(dataset_csv)
    target_name = _find_column_case_insensitive(dataset, target_col)
    smiles_name = _find_column_case_insensitive(dataset, "smiles")
    y = pd.to_numeric(dataset[target_name], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("The target contains NaN or infinite values.")

    raw_X, feature_names = read_legacy_feature_matrix(feature_csv)
    feature_smiles = pd.read_csv(feature_smiles_csv)
    feature_smiles_name = _find_column_case_insensitive(feature_smiles, "smiles")
    if len(feature_smiles) != raw_X.shape[0]:
        raise ValueError("The feature matrix and feature-SMILES table have different row counts.")

    dataset_smiles = dataset[smiles_name].astype(str).str.strip().tolist()
    source_smiles = feature_smiles[feature_smiles_name].astype(str).str.strip().tolist()
    if len(source_smiles) != len(set(source_smiles)):
        raise ValueError("Duplicate SMILES were found in the feature-SMILES table.")
    source_index = {smiles: index for index, smiles in enumerate(source_smiles)}
    missing = [smiles for smiles in dataset_smiles if smiles not in source_index]
    if missing:
        raise ValueError(f"{len(missing)} dataset SMILES are absent from the feature table: {missing[:5]}")
    X = np.vstack([raw_X[source_index[smiles]] for smiles in dataset_smiles]).astype(np.float32)
    return TrainingData(X=X, y=y, smiles=dataset_smiles, feature_names=feature_names)


def make_same_test_split(
    y: np.ndarray,
    split_seed: int = 1,
    test_size: float = 0.2,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float).reshape(-1)
    indices = np.arange(len(y))
    bins = pd.qcut(y, q=int(n_bins), labels=False, duplicates="drop")
    train_idx, test_idx = train_test_split(
        indices,
        test_size=float(test_size),
        random_state=int(split_seed),
        stratify=bins,
    )
    return np.sort(np.asarray(train_idx, dtype=int)), np.sort(np.asarray(test_idx, dtype=int))


def load_split_csv(path: str | Path, smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    split = pd.read_csv(path)
    smiles_name = _find_column_case_insensitive(split, "smiles")
    split_name = _find_column_case_insensitive(split, "split")
    labels = {
        str(row[smiles_name]).strip(): str(row[split_name]).strip().lower()
        for _, row in split.iterrows()
    }
    missing = [value for value in smiles if value not in labels]
    if missing:
        raise ValueError(f"Split file is missing {len(missing)} SMILES: {missing[:5]}")
    # Match the authoritative mixed script: an optional ``val`` subset is not
    # folded back into the training pool.
    train_idx = np.asarray([i for i, value in enumerate(smiles) if labels[value] == "train"], dtype=int)
    test_idx = np.asarray([i for i, value in enumerate(smiles) if labels[value] == "test"], dtype=int)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("The split file must contain nonempty train and test subsets.")
    return np.sort(train_idx), np.sort(test_idx)


def make_validation_split(
    train_idx: np.ndarray,
    split_seed: int = 1,
    val_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    base_train, validation = train_test_split(
        np.asarray(train_idx, dtype=int),
        test_size=float(val_fraction),
        random_state=int(split_seed),
    )
    return np.sort(np.asarray(base_train, dtype=int)), np.sort(np.asarray(validation, dtype=int))
