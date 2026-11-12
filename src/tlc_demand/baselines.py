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


def last_value(features: pd.DataFrame) -> np.ndarray:
    """Forecast the target bin as the count in the origin bin, the latest one observed."""
    return features["lag_0"].to_numpy(dtype="float64")


def same_slot_yesterday(features: pd.DataFrame) -> np.ndarray:
    """Forecast the target bin as the same slot one day earlier."""
    return features["same_slot_1d"].to_numpy(dtype="float64")
