"""Spike classifier on the three June test folds, same train/test split and horizon gap as the
backtest. Thresholds and labels use training rows only. A dedicated classifier is compared with
two cheaper scorers: the regression forecast divided by the slot threshold, and last week's value
for the same slot divided by the threshold.
Usage: python scripts/run_spikes.py   (needs artifacts/features_h4.parquet from run_ablation.py)"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from tlc_demand.backtest import folds
from tlc_demand.features import FEATURE_COLUMNS
from tlc_demand.spikes import attach_threshold, average_precision, day_bootstrap_ap, label_spikes, slot_thresholds

H, Q, MIN_COUNT = 4, 0.9, 5
t0 = time.time()
feats = pd.read_parquet("artifacts/features_h4.parquet")
gap = pd.Timedelta(minutes=15 * H)
P = {"learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.9, "verbose": -1,
     "seed": 0, "deterministic": True, "num_threads": 1}
EXTRA = ["slot_q", "lag0_over_q", "roll16_over_q"]


def add_extra(d, thr):
    d = d.copy()
    d["slot_q"] = thr
    d["lag0_over_q"] = d["lag_0"] / (d["slot_q"] + 1)
    d["roll16_over_q"] = d["roll_mean_16"] / (d["slot_q"] + 1)
    return d


rows, per_fold, scorers = [], {}, ("classifier", "regression_ratio", "same_slot_1w_ratio", "lag0_ratio")
for start, end in folds(feats, 7, 3):
    tb = feats["target_bin"]
    train = feats[tb < start - gap]
    test = feats[(tb >= start) & (tb < end)]
    thr = slot_thresholds(train, Q)
    tr_thr, te_thr = attach_threshold(train, thr), attach_threshold(test, thr)
    keep_tr, keep_te = ~np.isnan(tr_thr), ~np.isnan(te_thr)
    train, tr_thr = train[keep_tr], tr_thr[keep_tr]
    test, te_thr = test[keep_te], te_thr[keep_te]
    ytr = label_spikes(train["y"].to_numpy(), tr_thr, MIN_COUNT)
    yte = label_spikes(test["y"].to_numpy(), te_thr, MIN_COUNT)
    tr, te = add_extra(train, tr_thr), add_extra(test, te_thr)
    cols = FEATURE_COLUMNS + EXTRA
    clf = lgb.train({**P, "objective": "binary"}, lgb.Dataset(tr[cols], ytr, categorical_feature=["zone"]), 300)
    reg = lgb.train({**P, "objective": "poisson"}, lgb.Dataset(train[FEATURE_COLUMNS], train["y"], categorical_feature=["zone"]), 300)
    score = {
        "classifier": clf.predict(te[cols]),
        "regression_ratio": np.clip(reg.predict(test[FEATURE_COLUMNS]), 0, None) / (te["slot_q"].to_numpy() + 1),
        "same_slot_1w_ratio": (test["same_slot_1w"].fillna(test["same_slot_1d"]) / (te["slot_q"] + 1)).to_numpy(),
        "lag0_ratio": te["lag0_over_q"].to_numpy(),
    }
    ok = ~np.isnan(score["same_slot_1w_ratio"])
    day = test["target_bin"].dt.normalize().to_numpy()
    rows.append(pd.DataFrame({"y": yte[ok], "day": day[ok], **{k: v[ok] for k, v in score.items()}, "fold": str(start)}))
    base = float(yte[ok].mean())
    per_fold[str(start)] = {"n_test": int(ok.sum()), "n_spikes": int(yte[ok].sum()), "base_rate": round(base, 5),
                            "train_base_rate": round(float(ytr.mean()), 5), "rows_dropped_unseen_slot": int((~keep_te).sum()),
                            **{f"ap_{k}": round(average_precision(yte[ok], v[ok]), 5) for k, v in score.items()}}
    print(start, per_fold[str(start)], round(time.time() - t0), "s", flush=True)

d = pd.concat(rows)
base = float(d["y"].mean())
pooled = {"base_rate": round(base, 5), "n_test": int(len(d)), "n_spikes": int(d["y"].sum())}
for k in scorers:
    ap = average_precision(d["y"].to_numpy(), d[k].to_numpy())
    lo, hi = day_bootstrap_ap(d["y"].to_numpy(), d[k].to_numpy(), d["day"].to_numpy())
    pooled[k] = {"ap": round(ap, 5), "ci95_day_bootstrap": [round(lo, 5), round(hi, 5)], "lift_over_base_rate": round(ap / base, 2)}
Path("results/spikes.json").write_text(json.dumps({
    "definition": f"y > {Q} quantile of the zone's training counts in the same weekday and hour, and y >= {MIN_COUNT}",
    "horizon_bins": H, "per_fold": per_fold, "pooled": pooled, "seconds": round(time.time() - t0)}, indent=2))
print(json.dumps(pooled, indent=1))
