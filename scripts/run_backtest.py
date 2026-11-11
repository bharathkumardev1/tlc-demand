"""Backtest on real TLC yellow-taxi months. Usage:
python scripts/run_backtest.py data/raw 2023 1 6 --horizon 4"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from tlc_demand.backtest import run_backtest
from tlc_demand.features import make_features
from tlc_demand.panel import build_panel
from tlc_demand.schema import PICKUP_COLUMNS, ZONE_COLUMN, validate_trips


def load(raw: Path, year: int, m0: int, m1: int) -> pd.DataFrame:
    frames, total = [], 0
    for m in range(m0, m1 + 1):
        path = raw / f"yellow_tripdata_{year}-{m:02d}.parquet"
        names = set(pq.ParquetFile(path).schema.names)
        ts = next(c for c in PICKUP_COLUMNS if c in names)
        df = pq.read_table(path, columns=[ts, ZONE_COLUMN]).to_pandas()
        total += len(df)
        frames.append(validate_trips(df, year, m))
    out = pd.concat(frames, ignore_index=True)
    print(f"raw rows {total:,}, kept {len(out):,}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("year", type=int)
    ap.add_argument("m0", type=int)
    ap.add_argument("m1", type=int)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/backtest.json"))
    a = ap.parse_args()
    t0 = time.time()
    trips = load(a.raw, a.year, a.m0, a.m1)
    end = (pd.Timestamp(a.year, a.m1, 1) + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")
    panel = build_panel(trips, f"{a.year}-{a.m0:02d}-01", end)
    feats = make_features(panel, a.horizon)
    print(f"panel {len(panel):,} rows, features {len(feats):,} rows, zones {panel.zone.nunique()}", flush=True)
    res = run_backtest(feats, a.horizon, test_days=7, n_folds=a.folds, seed=a.seed)
    rows = [
        {"test_start": str(r.test_start), "n_test": r.n_test, "wape_model": r.wape_model,
         "wape_naive": r.wape_naive, "mae_model": r.mae_model, "mae_naive": r.mae_naive}
        for r in res
    ]
    for r in rows:
        print(r, flush=True)
    a.out.write_text(json.dumps({"horizon_bins": a.horizon, "seed": a.seed, "folds": rows,
                                 "seconds": round(time.time() - t0)}, indent=2))
