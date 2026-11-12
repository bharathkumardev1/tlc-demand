"""Feature-group ablation on the same three June folds as the main backtest.

Each run drops one group of columns from the model input. The rows scored and the seasonal-naive
baseline are identical across runs, so differences come from the model only. Three seeds of the
full model give a noise floor for judging the differences.

Memory: the feature matrix is held once as float32 and training rows are a contiguous prefix of it
(sorted by target bin, as the backtest's gap rule makes them), so no run copies the data. Dropped
columns are removed by slicing a column subset per run (about 200 MB). LightGBM's ignore_column looks
like the cheaper way but only applies when data is loaded from a text file; with an in-memory array it
silently does nothing. Counts are exact integers in float32; rolling means round to float32, so the
full-model rows are the reference rather than the float64 backtest.
Usage: python scripts/make_features_cache.py data/raw 2023 1 6 artifacts/features_h4.parquet
       python scripts/run_ablation.py   (writes results/ablation.json; needs about 4 GB of memory)"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from tlc_demand.backtest import folds
from tlc_demand.features import FEATURE_COLUMNS
from tlc_demand.metrics import mae, wape

H = 4
GROUPS = {
    "lags": [c for c in FEATURE_COLUMNS if c.startswith("lag_")],
    "rolling_means": [c for c in FEATURE_COLUMNS if c.startswith("roll_mean_")],
    "same_slot": ["same_slot_1d", "same_slot_1w"],
    "calendar": ["tod", "dow", "is_weekend", "tod_sin", "tod_cos"],
    "zone": ["zone"],
}
cache = Path("artifacts/features_h4.parquet")
need = ["target_bin", "y", *FEATURE_COLUMNS]
df = pd.read_parquet(cache, columns=need)
df = df.sort_values("target_bin", kind="stable").reset_index(drop=True)
target_bin = df["target_bin"].to_numpy()
y = df["y"].to_numpy(np.float64)
X = np.ascontiguousarray(df[FEATURE_COLUMNS].to_numpy(np.float32))
naive_all = np.where(np.isnan(df["same_slot_1w"].to_numpy()), df["same_slot_1d"].to_numpy(), df["same_slot_1w"].to_numpy()).astype(np.float64)
last_bin = df["target_bin"].max()
fold_windows = folds(df[["target_bin"]], 7, 3)
del df
print(f"features {X.shape[0]:,} x {X.shape[1]}", flush=True)

P = {"objective": "poisson", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.9,
     "verbose": -1, "deterministic": True, "num_threads": 1}
gap = np.timedelta64(15 * H, "m")
col = {c: i for i, c in enumerate(FEATURE_COLUMNS)}


def run(drop: list[str], seed: int):
    keep = [c for c in FEATURE_COLUMNS if c not in drop]
    Xk = np.ascontiguousarray(X[:, [col[c] for c in keep]])
    per_fold = []
    for start, end in fold_windows:
        n_train = int(np.searchsorted(target_bin, (start - pd.Timedelta(minutes=15 * H)).to_datetime64(), side="left"))
        lo, hi = (int(np.searchsorted(target_bin, t.to_datetime64(), side="left")) for t in (start, end))
        cat = ["zone"] if "zone" in keep else []
        ds = lgb.Dataset(Xk[:n_train], y[:n_train], feature_name=keep, categorical_feature=cat)
        model = lgb.train({**P, "seed": seed}, ds, num_boost_round=300)
        assert model.feature_name() == keep, "model did not train on the requested columns"
        pred = np.clip(model.predict(Xk[lo:hi]), 0, None)
        yt, nv = y[lo:hi], naive_all[lo:hi]
        ok = ~np.isnan(nv)
        per_fold.append({"n": int(ok.sum()), "wape": wape(yt[ok], pred[ok]), "mae": mae(yt[ok], pred[ok]),
                         "naive_wape": wape(yt[ok], nv[ok])})
        del ds, model
    return per_fold


runs = [("full_seed0", [], 0), ("full_seed1", [], 1), ("full_seed2", [], 2)]
runs += [(f"minus_{g}", cols, 0) for g, cols in GROUPS.items()]
out = {"horizon_bins": H, "folds": 3, "groups": GROUPS, "runs": {}}
for name, drop, seed in runs:
    t0 = time.time()
    r = run(drop, seed)
    n = sum(f["n"] for f in r)
    out["runs"][name] = {
        "n_features": len(FEATURE_COLUMNS) - len(drop), "dropped": drop, "seed": seed,
        "wape_per_fold": [round(f["wape"], 5) for f in r],
        "wape_mean": round(sum(f["wape"] * f["n"] for f in r) / n, 5),
        "mae_mean": round(sum(f["mae"] * f["n"] for f in r) / n, 5),
        "naive_wape_mean": round(sum(f["naive_wape"] * f["n"] for f in r) / n, 5), "seconds": round(time.time() - t0)}
    print(name, out["runs"][name], flush=True)
    Path("results/ablation.json").write_text(json.dumps(out, indent=2))
