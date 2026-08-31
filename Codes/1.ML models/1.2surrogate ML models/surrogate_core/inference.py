"""Inference APIs and command-line entry points for the eight surrogate models."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import FINGERPRINT, TARGETS
from .fingerprint import featurize_hashed_morgan_counts


def _require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError("TensorFlow is required to load the released Keras models.") from error
    tf.get_logger().setLevel("ERROR")
    return tf


def _find_column(frame: pd.DataFrame, requested: str | None, candidates: tuple[str, ...]) -> str | None:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"Column {requested!r} was not found. Available columns: {list(frame.columns)}")
        return requested
    exact = {str(column): str(column) for column in frame.columns}
    lowered = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def target_directory(release_root: str | Path, target: str) -> Path:
    if target not in TARGETS:
        raise ValueError(f"Unknown electronic-property target {target!r}. Available targets: {list(TARGETS)}")
    directory = Path(release_root) / target
    if not directory.is_dir():
        raise FileNotFoundError(f"Released target directory is missing: {directory}")
    return directory


def _load_artifacts(release_root: str | Path, target: str) -> dict[str, Any]:
    directory = target_directory(release_root, target) / "pretrained"
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config.get("target") != target:
        raise ValueError(
            f"Artifact/target mismatch in {directory}: requested {target!r}, config declares {config.get('target')!r}."
        )
    if config.get("fingerprint") != FINGERPRINT:
        raise ValueError(f"The fingerprint configuration in {directory} does not match this release.")
    if not str(config.get("x_scaler", "")).startswith("identity"):
        raise ValueError(f"Unsupported input preprocessing declared in {directory}.")
    tf = _require_tensorflow()
    model = tf.keras.models.load_model(directory / "electronic_model.keras", compile=False)
    with (directory / "Y_scaler.pkl").open("rb") as handle:
        y_scaler = pickle.load(handle)
    return {"model": model, "y_scaler": y_scaler}


def predict_matrix(
    values: np.ndarray,
    target: str,
    release_root: str | Path,
    batch_size: int = 4096,
) -> np.ndarray:
    bundle = _load_artifacts(release_root, target)
    return _predict_with_bundle(values, bundle, batch_size)


def _predict_with_bundle(values: np.ndarray, bundle: dict[str, Any], batch_size: int) -> np.ndarray:
    predicted_scaled = bundle["model"].predict(
        np.asarray(values, dtype=np.float32),
        batch_size=int(batch_size),
        verbose=0,
    ).reshape(-1, 1)
    return np.asarray(bundle["y_scaler"].inverse_transform(predicted_scaled), dtype=float).reshape(-1)


def predict_smiles(
    smiles: list[str],
    target: str,
    release_root: str | Path,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    values, valid = featurize_hashed_morgan_counts(smiles)
    return predict_matrix(values, target, release_root, batch_size=batch_size), valid


def predict_all_smiles(
    smiles: list[str],
    release_root: str | Path,
    batch_size: int = 4096,
) -> tuple[pd.DataFrame, np.ndarray]:
    values, valid = featurize_hashed_morgan_counts(smiles)
    predictions = {
        target: predict_matrix(values, target, release_root, batch_size=batch_size)
        for target in TARGETS
    }
    return pd.DataFrame(predictions), valid


def _predict_file(
    release_root: Path,
    input_csv: str | Path,
    output_csv: str | Path,
    targets: tuple[str, ...],
    chunksize: int,
    batch_size: int,
    molecule_col: str | None,
    smiles_col: str | None,
) -> None:
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    bundles = {target: _load_artifacts(release_root, target) for target in targets}
    wrote_header = False
    for chunk in pd.read_csv(input_csv, chunksize=int(chunksize)):
        selected_smiles = _find_column(chunk, smiles_col, ("smiles", "SMILES", "Smiles"))
        if selected_smiles is None:
            raise ValueError("No SMILES column was found in the input file.")
        selected_molecule = _find_column(chunk, molecule_col, ("molecule", "id", "name", "mol"))
        smiles = chunk[selected_smiles].astype(str).tolist()
        values, _ = featurize_hashed_morgan_counts(smiles)
        result = pd.DataFrame()
        if selected_molecule is not None:
            result["molecule"] = chunk[selected_molecule].astype(str).to_numpy()
        result["smiles"] = smiles
        for target in targets:
            result[target] = _predict_with_bundle(values, bundles[target], batch_size=batch_size)
        result.to_csv(
            output,
            mode="a" if wrote_header else "w",
            header=not wrote_header,
            index=False,
            encoding="utf-8",
        )
        wrote_header = True
    print(f"Saved predictions to {output}")


def main_for_target(target: str) -> None:
    parser = argparse.ArgumentParser(description=f"Predict {target} with its released single-task DNN22 surrogate.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--pretrained_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--chunksize", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--molecule_col", default=None)
    parser.add_argument("--smiles_col", default=None)
    args = parser.parse_args()
    _predict_file(
        Path(args.pretrained_root),
        args.input_csv,
        args.output_csv,
        (target,),
        args.chunksize,
        args.batch_size,
        args.molecule_col,
        args.smiles_col,
    )


def main_all() -> None:
    parser = argparse.ArgumentParser(description="Predict all eight DFT electronic properties with the released surrogates.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--pretrained_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--chunksize", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--molecule_col", default=None)
    parser.add_argument("--smiles_col", default=None)
    args = parser.parse_args()
    _predict_file(
        Path(args.pretrained_root),
        args.input_csv,
        args.output_csv,
        TARGETS,
        args.chunksize,
        args.batch_size,
        args.molecule_col,
        args.smiles_col,
    )
