"""Demand-spike labels and scoring.

A spike is a count well above what the zone usually sees in that hour of that weekday:
y > q-th quantile of the zone's counts in the same (weekday, hour) slot, and y >= min_count so
near-empty zones cannot spike on one trip. The quantiles come from training rows only. Scoring
uses average precision (PR-AUC) because spikes are rare."""
from __future__ import annotations

import numpy as np
import pandas as pd


def slot_thresholds(train: pd.DataFrame, q: float = 0.9) -> pd.Series:
    """Per (zone, weekday, hour) quantile of the target count, from training rows."""
    key = [train["zone"], train["target_bin"].dt.day_of_week, train["target_bin"].dt.hour]
    return train["y"].groupby(key).quantile(q).rename("slot_q")


def attach_threshold(df: pd.DataFrame, thr: pd.Series) -> np.ndarray:
    """Threshold for each row; NaN where the (zone, weekday, hour) slot was never seen in training."""
    idx = pd.MultiIndex.from_arrays([df["zone"], df["target_bin"].dt.day_of_week, df["target_bin"].dt.hour])
    return thr.reindex(idx).to_numpy()


def label_spikes(y: np.ndarray, threshold: np.ndarray, min_count: int = 5) -> np.ndarray:
    return ((y > threshold) & (y >= min_count)).astype(np.int8)


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    """Step-wise area under the precision-recall curve, ties broken by treating tied scores as one block."""
    y_true = np.asarray(y_true)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    s, y = score[order], y_true[order]
    tp = np.cumsum(y)
    k = np.arange(1, len(y) + 1)
    last_of_block = np.r_[s[1:] != s[:-1], True]
    tp_b, k_b = tp[last_of_block], k[last_of_block]
    precision, recall = tp_b / k_b, tp_b / n_pos
    prev = np.r_[0.0, recall[:-1]]
    return float(((recall - prev) * precision).sum())


def day_bootstrap_ap(y_true: np.ndarray, score: np.ndarray, day: np.ndarray, n: int = 300, seed: int = 0) -> tuple[float, float]:
    """95% interval for average precision, resampling whole days so within-day correlation is kept."""
    rng = np.random.default_rng(seed)
    days = np.unique(day)
    groups = {d: np.flatnonzero(day == d) for d in days}
    vals = []
    for _ in range(n):
        pick = rng.choice(days, size=len(days), replace=True)
        idx = np.concatenate([groups[d] for d in pick])
        vals.append(average_precision(y_true[idx], score[idx]))
    return float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))
