import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from tlc_demand.spikes import attach_threshold, average_precision, day_bootstrap_ap, label_spikes, slot_thresholds


def frame(zone, start, n, y):
    return pd.DataFrame({"zone": zone, "target_bin": pd.date_range(start, periods=n, freq="15min"), "y": y})


def test_average_precision_matches_reference_implementation():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.1).astype(int)
    s = rng.normal(size=2000) + 1.5 * y
    assert abs(average_precision(y, s) - average_precision_score(y, s)) < 1e-9
    tied = np.round(s)  # many ties
    assert abs(average_precision(y, tied) - average_precision_score(y, tied)) < 1e-9


def test_average_precision_extremes():
    y = np.array([0, 0, 1, 1])
    assert average_precision(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert np.isnan(average_precision(np.zeros(4), np.arange(4.0)))
    rng = np.random.default_rng(1)
    big = (rng.random(20000) < 0.05).astype(int)
    assert abs(average_precision(big, rng.random(20000)) - big.mean()) < 0.01  # random scores: AP ~ base rate


def test_thresholds_come_from_training_rows_only():
    train = frame(1, "2023-01-02", 96 * 14, np.tile(np.arange(96), 14) % 7)  # same pattern every day
    test = frame(1, "2023-02-06", 96, np.full(96, 1000))  # absurd test counts must not move the thresholds
    thr = slot_thresholds(train)
    assert (attach_threshold(test, thr) == attach_threshold(test, slot_thresholds(train))).all()
    assert thr.max() <= 6


def test_label_requires_min_count_and_unseen_slots_are_not_spikes():
    y = np.array([3, 6, 6, 9])
    thr = np.array([2.0, 5.0, 6.0, np.nan])
    assert label_spikes(y, thr, min_count=5).tolist() == [0, 1, 0, 0]


def test_day_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(2)
    n = 4000
    y = (rng.random(n) < 0.1).astype(int)
    s = rng.normal(size=n) + 1.0 * y
    day = np.repeat(np.arange(20), n // 20)
    lo, hi = day_bootstrap_ap(y, s, day, n=100)
    assert lo <= average_precision(y, s) <= hi
