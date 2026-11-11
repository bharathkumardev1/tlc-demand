from __future__ import annotations

import numpy as np


def wape(y: np.ndarray, pred: np.ndarray) -> float:
    """Weighted absolute percentage error. Chosen over MAPE because most zones
    have many near-zero bins, where MAPE is undefined or dominated by noise."""
    denom = np.abs(y).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(y - pred).sum() / denom)


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.abs(y - pred).mean())
