"""Zone / borough / city forecasts and their reconciliation, scored on the three June test folds.

Each level has its own LightGBM model trained on that level's series (independent base forecasts).
Reconciliation weights come only from the forecast errors of earlier folds (an expanding window), so
no fold is scored with weights fitted on its own errors. A fourth, earlier fold supplies the first
residuals. Usage: python scripts/run_hierarchy.py data/raw 2023 1 6 data/raw/taxi_zone_lookup.csv"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_backtest import load  # noqa: E402

from tlc_demand.backtest import run_backtest  # noqa: E402
from tlc_demand.features import make_features  # noqa: E402
from tlc_demand.hierarchy import reconcile, shrunk_covariance, summing_matrix  # noqa: E402
from tlc_demand.panel import build_panel  # noqa: E402

raw, year, m0, m1, lookup = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
H = 4
t0 = time.time()
trips = load(raw, year, m0, m1)
end = (pd.Timestamp(year, m1, 1) + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")
panel = build_panel(trips, f"{year}-{m0:02d}-01", end)
zone_borough = pd.read_csv(lookup, keep_default_na=False).set_index("LocationID")["Borough"]
zones = sorted(panel["zone"].unique())
missing = [z for z in zones if z not in zone_borough.index]
assert not missing, f"zones without a borough: {missing}"
names = sorted(set(zone_borough.loc[zones]))
b_of_z = [names.index(zone_borough[z]) for z in zones]
S, b_ids = summing_matrix(b_of_z)
print("boroughs:", names, "zones:", len(zones), flush=True)

CITY_ID, BOROUGH_BASE = 2000, 1000
panel["borough"] = panel["zone"].map(lambda z: BOROUGH_BASE + names.index(zone_borough[z]))
b_panel = panel.groupby(["borough", "bin"], as_index=False)["y"].sum().rename(columns={"borough": "zone"})
c_panel = panel.groupby("bin", as_index=False)["y"].sum().assign(zone=CITY_ID)
for p in (b_panel, c_panel):
    p["zone"] = p["zone"].astype("int16")
    p.sort_values(["zone", "bin"], inplace=True)


def forecasts(p, label):
    coll = []
    run_backtest(make_features(p, H), H, test_days=7, n_folds=4, collect=coll)
    print(label, "done", round(time.time() - t0), "s", flush=True)
    return pd.concat(coll)


zone_pred = forecasts(panel[["zone", "bin", "y"]], "zone")
b_pred = forecasts(b_panel, "borough")
c_pred = forecasts(c_panel, "city")
folds = sorted(zone_pred["fold"].unique())


def wide(df, col, nodes):
    return df.pivot(index="target_bin", columns="zone", values=col)[nodes]


zone_nodes = zones
b_nodes = [BOROUGH_BASE + i for i in b_ids]
order = [CITY_ID] + b_nodes + zone_nodes
levels = {"city": slice(0, 1), "borough": slice(1, 1 + len(b_nodes)), "zone": slice(1 + len(b_nodes), len(order))}
all_pred = pd.concat([c_pred, b_pred, zone_pred])


def fold_mats(fold):
    d = all_pred[all_pred["fold"] == fold]
    P, Y, N = wide(d, "pred", order), wide(d, "y", order), wide(d, "naive", order)
    assert P.index.equals(Y.index) and not P.isna().any().any()
    return P.to_numpy(), Y.to_numpy(), N.to_numpy()


def wape(y, p):
    return float(np.abs(y - p).sum() / np.abs(y).sum())


methods = ["base", "naive", "bottom_up", "ols", "wls_variance", "mint_shrink"]
acc = {m: {lv: [0.0, 0.0] for lv in levels} for m in methods}  # abs error sum, actual sum
per_fold, lams = {}, {}
for k in range(1, len(folds)):
    P, Y, N = fold_mats(folds[k])
    E = np.vstack([fold_mats(f)[1] - fold_mats(f)[0] for f in folds[:k]])  # errors of earlier folds only
    W_var = np.diag(E.var(axis=0, ddof=1) + 1e-6)
    W_mint, lam = shrunk_covariance(E)
    lams[folds[k]] = {"mint_shrink_intensity": lam, "residual_rows": int(len(E))}
    out = {"base": P, "naive": N, "bottom_up": P[:, levels["zone"]] @ S.T,
           "ols": reconcile(P, S), "wls_variance": reconcile(P, S, W_var), "mint_shrink": reconcile(P, S, W_mint)}
    per_fold[folds[k]] = {}
    for m, M in out.items():
        M = np.clip(M, 0, None) if m not in ("base", "naive") else M
        per_fold[folds[k]][m] = {}
        for lv, sl in levels.items():
            e, a = np.abs(Y[:, sl] - M[:, sl]).sum(), np.abs(Y[:, sl]).sum()
            acc[m][lv][0] += e; acc[m][lv][1] += a
            per_fold[folds[k]][m][lv] = round(e / a, 5)
res = {m: {lv: round(v[0] / v[1], 5) for lv, v in d.items()} for m, d in acc.items()}
Path("results/hierarchy.json").write_text(json.dumps({
    "horizon_bins": H, "test_folds": folds[1:], "weights_from": "forecast errors on earlier folds only",
    "boroughs": names, "n_zones": len(zones), "pooled_wape": res, "per_fold_wape": per_fold, "shrinkage": lams,
    "negative_values_clipped_to_zero": "after reconciliation", "seconds": round(time.time() - t0)}, indent=2))
print(json.dumps(res, indent=1))
