"""Validation for raw TLC trip files. Rejects a month early instead of letting
a silent schema change turn into a bad feature column three stages later."""
from __future__ import annotations

import pandas as pd

# The pickup timestamp column is spelled differently across TLC vendors and years.
PICKUP_COLUMNS = ("tpep_pickup_datetime", "pickup_datetime")
ZONE_COLUMN = "PULocationID"
MAX_ZONE_ID = 265  # TLC taxi zone lookup ids run 1..265


class SchemaError(ValueError):
    pass


def pickup_column(df: pd.DataFrame) -> str:
    for name in PICKUP_COLUMNS:
        if name in df.columns:
            return name
    raise SchemaError(f"no pickup timestamp column; have {sorted(df.columns)[:10]}")


def validate_trips(df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """Return only rows that fall inside the file's own month and a valid zone.

    TLC monthly files contain stray timestamps from other years. We drop them
    and fail if too much is dropped, since that points at a wrong file."""
    ts_col = pickup_column(df)
    if ZONE_COLUMN not in df.columns:
        raise SchemaError(f"missing {ZONE_COLUMN}")
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    zone = pd.to_numeric(df[ZONE_COLUMN], errors="coerce")
    keep = (
        ts.notna()
        & (ts.dt.year == year)
        & (ts.dt.month == month)
        & zone.between(1, MAX_ZONE_ID)
    )
    dropped = 1.0 - keep.mean() if len(df) else 0.0
    if dropped > 0.05:
        raise SchemaError(
            f"{dropped:.1%} of rows fall outside {year}-{month:02d} or a valid zone"
        )
    out = pd.DataFrame({"ts": ts[keep], "zone": zone[keep].astype("int16")})
    return out.reset_index(drop=True)
