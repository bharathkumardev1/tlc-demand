"""Does routing low-volume zones to the seasonal-naive forecast help?

The rule uses only information available at forecast time: a zone's mean trips per
15-minute bin over the 28 days before the fold starts. The threshold is picked on an
extra, earlier validation fold and then frozen before the three June test folds are scored.
Usage: python scripts/sparse_zone_rule.py data/raw 2023 1 6"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_backtest import load  # noqa: E402

from tlc_demand.backtest import run_backtest  # noqa: E402
from tlc_demand.features import make_features  # noqa: E402
from tlc_demand.metrics import mae, wape  # noqa: E402
from tlc_demand.panel import build_panel  # noqa: E402

raw, year, m0, m1 = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
H = 4
trips = load(raw, year, m0, m1)
end = (pd.Timestamp(year, m1, 1) + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")
panel = build_panel(trips, f"{year}-{m0:02d}-01", end)
feats = make_features(panel, H)
coll = []
run_backtest(feats, H, test_days=7, n_folds=4, collect=coll)
pred = pd.concat(coll)
pred = pred[pred["naive"].notna()].copy()


def trailing_volume(fold_start: pd.Timestamp) -> pd.Series:
    w = panel[(panel["bin"] >= fold_start - pd.Timedelta(days=28)) & (panel["bin"] < fold_start)]
    return w.groupby("zone")["y"].mean()


def with_vol(d: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for f, g in d.groupby("fold"):
        v = trailing_volume(pd.Timestamp(f))
        parts.append(g.assign(vol=g["zone"].map(v)))
    return pd.concat(parts)


pred = with_vol(pred)
folds = sorted(pred["fold"].unique())
val, test = pred[pred["fold"] == folds[0]], pred[pred["fold"] != folds[0]]


def routed(d: pd.DataFrame, thr: float) -> np.ndarray:
    return np.where(d["vol"] < thr, d["naive"], d["pred"])


def score(d, p):
    return {"wape": wape(d["y"].to_numpy(), p), "mae": mae(d["y"].to_numpy(), p)}


# candidate thresholds are quantiles of zone volume, so the rule is comparable across data
cands = [0.0] + [float(x) for x in np.quantile(val.groupby("zone")["vol"].first(), [.1, .2, .3, .4, .5, .6, .7, .8, .9])]
val_mae = {t: score(val, routed(val, t))["mae"] for t in cands}
best = min(val_mae, key=val_mae.get)
print("validation MAE by threshold:", {round(k, 3): round(v, 4) for k, v in val_mae.items()}, "chosen", round(best, 3))

report = {"validation_fold": folds[0], "chosen_threshold_trips_per_bin": best,
          "validation_mae": {str(round(k, 4)): v for k, v in val_mae.items()},
          "test_folds": folds[1:], "per_fold": {}}
for f in folds[1:]:
    d = test[test["fold"] == f]
    report["per_fold"][f] = {"model": score(d, d["pred"].to_numpy()), "naive": score(d, d["naive"].to_numpy()),
                             "routed": score(d, routed(d, best))}
d = test
report["pooled"] = {"model": score(d, d["pred"].to_numpy()), "naive": score(d, d["naive"].to_numpy()),
                    "routed": score(d, routed(d, best)),
                    "share_of_trips_routed": float(d.loc[d["vol"] < best, "y"].sum() / d["y"].sum()),
                    "zones_routed": int(d.loc[d["vol"] < best, "zone"].nunique())}
low = d[d["vol"] < best]
if len(low):
    report["routed_zones_only"] = {"model": score(low, low["pred"].to_numpy()),
                                   "naive": score(low, low["naive"].to_numpy())}
Path("results/sparse_zone_rule.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=1))
