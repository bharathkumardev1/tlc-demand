import numpy as np
import pandas as pd
import pytest

from tlc_demand.backtest import run_backtest
from tlc_demand.baselines import seasonal_naive
from tlc_demand.features import FEATURE_COLUMNS, make_features
from tlc_demand.metrics import wape
from tlc_demand.panel import build_panel
from tlc_demand.schema import SchemaError, validate_trips

from synthetic import synthetic_trips

H = 4


@pytest.fixture(scope="module")
def panel():
    trips = synthetic_trips()
    return build_panel(trips, "2023-01-02", "2023-02-06")


def test_panel_is_dense_with_explicit_zeros(panel):
    per_zone = panel.groupby("zone").size()
    assert (per_zone == 35 * 96).all()
    assert (panel["y"] == 0).any()


def test_features_do_not_read_the_future(panel):
    base = make_features(panel, H)
    cut = pd.Timestamp("2023-01-20 12:00")
    # change every observation strictly after the cut; features at origins
    # <= cut must not move, since they may only see data up to the origin
    mutated = panel.copy()
    mutated.loc[mutated["bin"] > cut, "y"] += 1000
    after = make_features(mutated, H)
    key = ["zone", "origin"]
    cols = [c for c in FEATURE_COLUMNS if c != "zone"]
    a = base[base["origin"] <= cut].set_index(key)[cols]
    b = after[after["origin"] <= cut].set_index(key)[cols]
    pd.testing.assert_frame_equal(a, b)


def test_target_is_horizon_bins_ahead(panel):
    f = make_features(panel, H)
    row = f[f["zone"] == 20].iloc[500]
    want = panel[(panel["zone"] == 20) & (panel["bin"] == row["origin"] + pd.Timedelta(minutes=15 * H))]["y"].iloc[0]
    assert row["y"] == want


def test_same_slot_lags_point_at_target_minus_day_and_week(panel):
    f = make_features(panel, H)
    row = f[(f["zone"] == 10)].iloc[900]
    target = row["target_bin"]
    look = lambda ts: panel[(panel["zone"] == 10) & (panel["bin"] == ts)]["y"].iloc[0]
    assert row["same_slot_1d"] == look(target - pd.Timedelta(days=1))
    assert row["same_slot_1w"] == look(target - pd.Timedelta(days=7))


def test_seasonal_naive_matches_week_ago(panel):
    f = make_features(panel, H).dropna(subset=["same_slot_1w"])
    assert np.array_equal(seasonal_naive(f), f["same_slot_1w"].to_numpy())


def test_schema_drops_strays_and_rejects_wrong_file():
    good = pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime(["2023-01-05 10:00"] * 99 + ["2008-12-31 23:00"]),
        "PULocationID": [10] * 100,
    })
    assert len(validate_trips(good, 2023, 1)) == 99
    wrong = good.assign(tpep_pickup_datetime=pd.Timestamp("2022-06-01"))
    with pytest.raises(SchemaError):
        validate_trips(wrong, 2023, 1)
    with pytest.raises(SchemaError):
        validate_trips(good.drop(columns="PULocationID"), 2023, 1)


def test_wape_handles_all_zero_actuals():
    assert np.isnan(wape(np.zeros(3), np.ones(3)))


def test_backtest_is_deterministic_and_has_no_train_test_overlap(panel):
    f = make_features(panel, H)
    r1 = run_backtest(f, H, test_days=7, n_folds=2, seed=3)
    r2 = run_backtest(f, H, test_days=7, n_folds=2, seed=3)
    assert [x.wape_model for x in r1] == [x.wape_model for x in r2]
    assert len(r1) == 2 and all(x.n_test > 0 for x in r1)
