"""PyTorch implementation of the DNN3 architecture used in the paper."""

from __future__ import annotations

from typing import Sequence

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


if nn is not None:

    class DNN3(nn.Module):
        def __init__(self, input_dim: int = 51, hidden: Sequence[int] = (128, 64, 32, 16)):
            super().__init__()
            layers: list[nn.Module] = []
            current = int(input_dim)
            for width in hidden:
                layers.extend([nn.Linear(current, int(width)), nn.ReLU()])
                current = int(width)
            layers.append(nn.Linear(current, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, values):
            return self.net(values)

else:

    class DNN3:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("Install torch to train or use the DNN model.")


def require_torch():
    if torch is None or nn is None:  # pragma: no cover
        raise ModuleNotFoundError("Install torch to train or use the DNN model.")
    return torch

