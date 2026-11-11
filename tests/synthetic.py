"""Synthetic trips used only by the test suite. Not TLC data and never used
for any reported number."""
import numpy as np
import pandas as pd


def synthetic_trips(days: int = 35, zones=(10, 20, 30), seed: int = 0, start="2023-01-02"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 96, freq="15min")
    rows = []
    for z in zones:
        tod = (idx.hour * 4 + idx.minute // 15).to_numpy()
        base = 6 + 4 * np.sin(2 * np.pi * tod / 96 - z) + (idx.day_of_week.to_numpy() >= 5) * 3
        lam = np.clip(base * (z / 20), 0.2, None)
        counts = rng.poisson(lam)
        for t, c in zip(idx, counts):
            if c:
                offs = rng.integers(0, 15 * 60, size=c)
                rows.append(pd.DataFrame({"ts": t + pd.to_timedelta(offs, unit="s"), "zone": np.int16(z)}))
    return pd.concat(rows, ignore_index=True)
