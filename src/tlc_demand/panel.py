"""Turn trip rows into a dense zone x 15-minute count panel."""
from __future__ import annotations

import pandas as pd

BIN = "15min"
BINS_PER_DAY = 96
BINS_PER_WEEK = 7 * BINS_PER_DAY


def build_panel(trips: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Counts per (zone, bin) over [start, end), with explicit zeros.

    Zeros matter: a zone with no pickups in a quarter hour is a real
    observation, and lag features computed over a sparse panel would
    silently read the wrong row."""
    idx = pd.date_range(start, end, freq=BIN, inclusive="left")
    bins = trips["ts"].dt.floor(BIN)
    counts = (
        trips.assign(bin=bins)
        .groupby(["zone", "bin"], observed=True)
        .size()
        .rename("y")
    )
    zones = sorted(trips["zone"].unique())
    full = pd.MultiIndex.from_product([zones, idx], names=["zone", "bin"])
    panel = counts.reindex(full, fill_value=0).reset_index()
    panel["zone"] = panel["zone"].astype("int16")
    return panel.sort_values(["zone", "bin"]).reset_index(drop=True)
