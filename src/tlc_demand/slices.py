"""Error broken out by where the model is likely to be weak."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import wape


def _row(g: pd.DataFrame) -> pd.Series:
    ok = g["naive"].notna() & g["last"].notna() & g["yday"].notna()
    return pd.Series({
        "n": int(ok.sum()),
        "share_of_trips": g["y"].sum(),
        "wape_model": wape(g.loc[ok, "y"].to_numpy(), g.loc[ok, "pred"].to_numpy()),
        "wape_naive": wape(g.loc[ok, "y"].to_numpy(), g.loc[ok, "naive"].to_numpy()),
        "wape_last": wape(g.loc[ok, "y"].to_numpy(), g.loc[ok, "last"].to_numpy()),
        "wape_yesterday": wape(g.loc[ok, "y"].to_numpy(), g.loc[ok, "yday"].to_numpy()),
    })


def slice_table(pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Slices: zone volume tier (by trips in the test window), part of day, weekday vs weekend."""
    d = pred.copy()
    vol = d.groupby("zone")["y"].sum()
    # ranked first so qcut ties from many tiny zones don't collapse the bins
    tier = pd.qcut(vol.rank(method="first"), 4, labels=["q1_lowest", "q2", "q3", "q4_highest"])
    d["volume_tier"] = d["zone"].map(tier)
    hour = d["tod"] // 4
    d["part_of_day"] = pd.cut(hour, [-1, 5, 9, 15, 19, 23],
                              labels=["00-05", "06-09", "10-15", "16-19", "20-23"])
    d["weekend"] = np.where(d["target_bin"].dt.day_of_week >= 5, "weekend", "weekday")
    out = {}
    for col in ("volume_tier", "part_of_day", "weekend"):
        t = d.groupby(col, observed=True).apply(_row, include_groups=False)
        t["share_of_trips"] = t["share_of_trips"] / t["share_of_trips"].sum()
        out[col] = t
    return out
