from __future__ import annotations

import numpy as np
import pandas as pd


def seasonal_naive(features: pd.DataFrame) -> np.ndarray:
    """Forecast the target bin as the same slot one week earlier.

    Falls back to the same slot one day earlier when the week-ago value is
    missing, which happens only in the first week of a zone's history."""
    week = features["same_slot_1w"]
    day = features["same_slot_1d"]
    return week.fillna(day).to_numpy(dtype="float64")
