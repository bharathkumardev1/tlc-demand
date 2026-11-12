"""Rolling-origin backtest. Each fold trains on everything strictly before the
fold's test window, with a gap of `horizon` bins so no training target overlaps
the test period."""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from .baselines import last_value, same_slot_yesterday, seasonal_naive
from .features import FEATURE_COLUMNS
from .metrics import mae, wape


@dataclass(frozen=True)
class FoldResult:
    test_start: pd.Timestamp
    n_test: int
    wape_model: float
    wape_naive: float
    mae_model: float
    mae_naive: float
    wape_last: float = float("nan")
    mae_last: float = float("nan")
    wape_yesterday: float = float("nan")
    mae_yesterday: float = float("nan")


def folds(features: pd.DataFrame, test_days: int, n_folds: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    last = features["target_bin"].max().normalize() + pd.Timedelta(days=1)
    out = []
    for i in range(n_folds, 0, -1):
        end = last - pd.Timedelta(days=test_days * (i - 1))
        out.append((end - pd.Timedelta(days=test_days), end))
    return out


def run_backtest(
    features: pd.DataFrame,
    horizon: int,
    test_days: int = 7,
    n_folds: int = 3,
    seed: int = 0,
    params: dict | None = None,
    collect: list | None = None,
) -> list[FoldResult]:
    p = {
        "objective": "poisson",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "verbose": -1,
        "seed": seed,
        "deterministic": True,
        "num_threads": 1,
    }
    p.update(params or {})
    gap = pd.Timedelta(minutes=15 * horizon)
    results = []
    for start, end in folds(features, test_days, n_folds):
        tb = features["target_bin"]
        test = features[(tb >= start) & (tb < end)]
        # train origins must have their target before the test window begins
        train = features[features["target_bin"] < start - gap]
        if train.empty or test.empty:
            continue
        model = lgb.train(
            p,
            lgb.Dataset(train[FEATURE_COLUMNS], train["y"], categorical_feature=["zone"]),
            num_boost_round=300,
        )
        pred = np.clip(model.predict(test[FEATURE_COLUMNS]), 0, None)
        naive = seasonal_naive(test)
        last = last_value(test)
        yday = same_slot_yesterday(test)
        # every baseline is scored on the same rows, so the comparison is like for like
        ok = ~(np.isnan(naive) | np.isnan(last) | np.isnan(yday))
        y = test["y"].to_numpy()
        if collect is not None:
            collect.append(test.assign(pred=pred, naive=naive, last=last, yday=yday, fold=str(start))[
                ["zone", "target_bin", "tod", "y", "pred", "naive", "last", "yday", "fold"]
            ])
        results.append(
            FoldResult(
                test_start=start,
                n_test=int(ok.sum()),
                wape_model=wape(y[ok], pred[ok]),
                wape_naive=wape(y[ok], naive[ok]),
                mae_model=mae(y[ok], pred[ok]),
                mae_naive=mae(y[ok], naive[ok]),
                wape_last=wape(y[ok], last[ok]),
                mae_last=mae(y[ok], last[ok]),
                wape_yesterday=wape(y[ok], yday[ok]),
                mae_yesterday=mae(y[ok], yday[ok]),
            )
        )
    return results
