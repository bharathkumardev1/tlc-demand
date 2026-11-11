"""Leakage-safe features for a forecast made at bin t about bin t + horizon.

Every lag is measured from the forecast origin t, so a lag of L bins is only
allowed when L >= 0 relative to the origin. Calendar features describe the
target bin, which is known in advance."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .panel import BINS_PER_DAY, BINS_PER_WEEK

# Offsets (in bins) back from the forecast origin.
ORIGIN_LAGS = (0, 1, 2, 3, 4, 8, 16)
ROLL_WINDOWS = (4, 16, 96)


def make_features(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """One row per (zone, origin) with target y = count at origin + horizon."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    parts = []
    for zone, g in panel.groupby("zone", sort=False):
        g = g.sort_values("bin").reset_index(drop=True)
        y = g["y"].to_numpy(dtype="float64")
        f = pd.DataFrame({"zone": zone, "origin": g["bin"]})
        for lag in ORIGIN_LAGS:
            f[f"lag_{lag}"] = g["y"].shift(lag)
        # same slot one day / one week before the *target* bin
        f["same_slot_1d"] = g["y"].shift(BINS_PER_DAY - horizon)
        f["same_slot_1w"] = g["y"].shift(BINS_PER_WEEK - horizon)
        for w in ROLL_WINDOWS:
            f[f"roll_mean_{w}"] = g["y"].rolling(w, min_periods=w).mean()
        f["target_bin"] = g["bin"].shift(-horizon)
        f["y"] = pd.Series(y).shift(-horizon)
        parts.append(f)
    out = pd.concat(parts, ignore_index=True)
    tb = out["target_bin"]
    out["tod"] = tb.dt.hour * 4 + tb.dt.minute // 15
    out["dow"] = tb.dt.day_of_week
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")
    angle = 2 * np.pi * out["tod"] / BINS_PER_DAY
    out["tod_sin"], out["tod_cos"] = np.sin(angle), np.cos(angle)
    return out.dropna(subset=["target_bin", "y"]).reset_index(drop=True)


FEATURE_COLUMNS = (
    [f"lag_{l}" for l in ORIGIN_LAGS]
    + ["same_slot_1d", "same_slot_1w"]
    + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
    + ["tod", "dow", "is_weekend", "tod_sin", "tod_cos", "zone"]
)
