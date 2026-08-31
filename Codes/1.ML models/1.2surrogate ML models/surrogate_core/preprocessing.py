"""Portable compatibility helpers for the original no-op input scaler artifact."""

from __future__ import annotations

import pickle
from pathlib import Path


class IdentityScaler:
    """No-op transformer used because Morgan count features require no input scaling."""

    def fit(self, values):
        return self

    def transform(self, values):
        return values

    def fit_transform(self, values):
        return values

    def inverse_transform(self, values):
        return values


class _LegacyIdentityScalerUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "__main__" and name == "IdentityScaler":
            return IdentityScaler
        return super().find_class(module, name)


def load_identity_scaler(path: str | Path) -> IdentityScaler:
    """Load either the original ``__main__`` pickle or a portable retrained scaler."""
    with Path(path).open("rb") as handle:
        scaler = _LegacyIdentityScalerUnpickler(handle).load()
    if not isinstance(scaler, IdentityScaler):
        raise TypeError(f"Expected an IdentityScaler artifact, received {type(scaler).__name__}.")
    return scaler
