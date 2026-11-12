"""Volume-tier table with zones ranked by trips before the first test fold, instead of by the
test window's own totals. Usage:
python scripts/slices_by_training_volume.py data/raw 2023 1 6 --horizon 4
Runs the same three-fold backtest as run_backtest.py (about five minutes on 2 vCPU)."""
import argparse
import json
from pathlib import Path

import pandas as pd

from run_backtest import load
from tlc_demand.backtest import folds, run_backtest
from tlc_demand.features import make_features
from tlc_demand.panel import build_panel
from tlc_demand.slices import slice_table

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("year", type=int)
    ap.add_argument("m0", type=int)
    ap.add_argument("m1", type=int)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("results/slices_training_volume.json"))
    a = ap.parse_args()
    trips = load(a.raw, a.year, a.m0, a.m1)
    end = (pd.Timestamp(a.year, a.m1, 1) + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")
    feats = make_features(build_panel(trips, f"{a.year}-{a.m0:02d}-01", end), a.horizon)
    first_start = folds(feats, 7, a.folds)[0][0]
    prior = feats[feats["target_bin"] < first_start].groupby("zone")["y"].sum()
    collected = []
    run_backtest(feats, a.horizon, test_days=7, n_folds=a.folds, collect=collected)
    pred = pd.concat(collected)
    by_prior = slice_table(pred, volume=prior)["volume_tier"]
    by_realized = slice_table(pred)["volume_tier"]
    # how many zones land in a different tier under the two rankings
    def tiers(v):
        return pd.qcut(v.rank(method="first"), 4, labels=False)
    realized = pred.groupby("zone")["y"].sum()
    z = realized.index
    moved = int((tiers(prior.reindex(z).fillna(0)) != tiers(realized)).sum())
    a.out.write_text(json.dumps({
        "horizon_bins": a.horizon,
        "ranked_by": f"trips with target before {first_start}",
        "zones": int(len(z)),
        "zones_in_a_different_tier_than_realized_ranking": moved,
        "volume_tier_by_prior_volume": by_prior.round(4).reset_index().to_dict(orient="records"),
        "volume_tier_by_test_window_volume": by_realized.round(4).reset_index().to_dict(orient="records"),
    }, indent=2))
    print(by_prior.round(4))
    print("zones in a different tier:", moved, "of", len(z))
