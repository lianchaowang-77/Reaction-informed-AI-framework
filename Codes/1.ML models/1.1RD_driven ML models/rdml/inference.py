"""Load released pretrained artifacts and predict log(k_H) from a 51D RD matrix."""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from .data import read_legacy_feature_matrix
from .dnn import DNN3, require_torch
from .registry import MODEL_NAMES


def _seed_directories(pretrained_dir: str | Path, seeds: Sequence[int] | None = None) -> list[Path]:
    root = Path(pretrained_dir)
    selected = list(seeds) if seeds is not None else list(range(1, 6))
    if not selected or any(seed not in range(1, 6) for seed in selected) or len(selected) != len(set(selected)):
        raise ValueError("Seeds must be a nonempty, unique subset of 1,2,3,4,5.")
    members = [root / f"seed{seed}" for seed in selected]
    missing = [path for path in members if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing pretrained member directories: {missing}")
    return members


def _validate_member_model(member_dir: Path, model_key: str) -> None:
    config_path = member_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_key") != model_key:
        raise ValueError(
            f"Artifact/model mismatch in {member_dir}: requested {model_key!r}, "
            f"but config declares {config.get('model_key')!r}."
        )


def _predict_dnn(member_dir: Path, X: np.ndarray, device: str) -> np.ndarray:
    torch = require_torch()
    config = json.loads((member_dir / "config.json").read_text(encoding="utf-8"))
    hidden = config.get("architecture", [128, 64, 32, 16])
    with (member_dir / "Xdesc_scaler.pkl").open("rb") as handle:
        x_scaler = pickle.load(handle)
    with (member_dir / "Yscaler.pkl").open("rb") as handle:
        y_scaler = pickle.load(handle)
    model = DNN3(input_dim=X.shape[1], hidden=hidden).to(torch.device(device))
    try:
        state = torch.load(member_dir / "DNN3.pt", map_location=torch.device(device), weights_only=True)
    except TypeError:  # pragma: no cover - older PyTorch
        state = torch.load(member_dir / "DNN3.pt", map_location=torch.device(device))
    model.load_state_dict(state)
    model.eval()
    scaled = x_scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        prediction_scaled = model(torch.tensor(scaled, dtype=torch.float32, device=torch.device(device))).cpu().numpy()
    return y_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).reshape(-1)


def predict_ensemble(
    model_key: str,
    feature_csv: str | Path,
    pretrained_dir: str | Path,
    device: str = "cpu",
    seeds: Sequence[int] | None = None,
) -> pd.DataFrame:
    if model_key not in MODEL_NAMES:
        raise ValueError(f"Unknown model key {model_key!r}. Available keys: {sorted(MODEL_NAMES)}")
    X, _ = read_legacy_feature_matrix(feature_csv)
    if X.shape[1] != 51:
        raise ValueError(f"Expected 51 RD features, received {X.shape[1]}.")
    predictions: dict[str, np.ndarray] = {}
    members = _seed_directories(pretrained_dir, seeds=seeds)
    for member in members:
        _validate_member_model(member, model_key)
        if model_key == "dnn":
            values = _predict_dnn(member, X, device=device)
        else:
            estimator = joblib.load(member / "model.pkl")
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"X does not have valid feature names, but .* was fitted with feature names",
                    category=UserWarning,
                )
                values = np.asarray(estimator.predict(X), dtype=float).reshape(-1)
        predictions[member.name] = values
    frame = pd.DataFrame(predictions)
    frame["predicted_log_kH_mean"] = frame.mean(axis=1)
    member_columns = [member.name for member in members]
    frame["predicted_log_kH_std"] = frame[member_columns].std(axis=1, ddof=0)
    return frame


def main_for_model(model_key: str) -> None:
    parser = argparse.ArgumentParser(description=f"Predict log(k_H) with the released {MODEL_NAMES[model_key]} model.")
    parser.add_argument("--feature_csv", required=True, help="51D RD matrix with feature labels in the first row.")
    parser.add_argument("--pretrained_dir", default=str(Path(__file__).resolve().parents[1] / MODEL_NAMES[model_key] / "pretrained"))
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seeds", default="1,2,3,4,5", help="Comma-separated released members (subset of 1-5).")
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    result = predict_ensemble(model_key, args.feature_csv, args.pretrained_dir, device=args.device, seeds=seeds)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Saved {len(result)} predictions to {output}")
