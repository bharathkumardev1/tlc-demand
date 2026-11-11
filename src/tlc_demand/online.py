"""Feature computation for a single forecast origin at serving time.

Must produce exactly what features.make_features produces for the same origin.
tests/test_online.py enforces that; any change to one side fails the other."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, ORIGIN_LAGS, ROLL_WINDOWS
from .panel import BINS_PER_DAY, BINS_PER_WEEK

# History length (bins, including the origin bin) a request must carry.
MIN_HISTORY = BINS_PER_WEEK


def features_at_origin(history: np.ndarray, origin: pd.Timestamp, zone: int, horizon: int) -> dict[str, float]:
    """`history` holds counts for consecutive 15-minute bins ending at `origin`."""
    if len(history) < MIN_HISTORY:
        raise ValueError(f"need at least {MIN_HISTORY} bins of history, got {len(history)}")
    y = np.asarray(history, dtype="float64")
    last = len(y) - 1
    f: dict[str, float] = {f"lag_{l}": y[last - l] for l in ORIGIN_LAGS}
    f["same_slot_1d"] = y[last - (BINS_PER_DAY - horizon)]
    f["same_slot_1w"] = y[last - (BINS_PER_WEEK - horizon)]
    for w in ROLL_WINDOWS:
        f[f"roll_mean_{w}"] = y[last - w + 1 : last + 1].mean()
    target = origin + pd.Timedelta(minutes=15 * horizon)
    tod = target.hour * 4 + target.minute // 15
    f["tod"], f["dow"] = tod, target.day_of_week
    f["is_weekend"] = int(target.day_of_week >= 5)
    f["tod_sin"], f["tod_cos"] = np.sin(2 * np.pi * tod / BINS_PER_DAY), np.cos(2 * np.pi * tod / BINS_PER_DAY)
    f["zone"] = zone
    return {c: f[c] for c in FEATURE_COLUMNS}
