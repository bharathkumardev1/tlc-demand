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

    def look(ts):
        return panel[(panel["zone"] == 10) & (panel["bin"] == ts)]["y"].iloc[0]

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


def test_backtest_is_deterministic(panel):
    f = make_features(panel, H)
    r1 = run_backtest(f, H, test_days=7, n_folds=2, seed=3)
    r2 = run_backtest(f, H, test_days=7, n_folds=2, seed=3)
    assert [x.wape_model for x in r1] == [x.wape_model for x in r2]
    assert len(r1) == 2 and all(x.n_test > 0 for x in r1)


def test_no_training_target_reaches_the_test_origins(panel):
    from tlc_demand.backtest import folds, split_fold

    f = make_features(panel, H)
    seen = 0
    for start, end in folds(f, 7, 2):
        train, test = split_fold(f, start, end, H)
        if train.empty or test.empty:
            continue
        seen += 1
        # the earliest test origin is start minus the horizon gap
        assert test["origin"].min() == start - pd.Timedelta(minutes=15 * H)
        assert train["target_bin"].max() < test["origin"].min()
        assert not set(train.index) & set(test.index)
    assert seen == 2


def test_slices_cover_every_row_once(panel):
    from tlc_demand.slices import slice_table

    f = make_features(panel, H)
    f = f[f["target_bin"] >= "2023-01-30"]
    pred = f.assign(pred=f["lag_0"], naive=f["same_slot_1w"], last=f["lag_0"], yday=f["same_slot_1d"])[
        ["zone", "target_bin", "tod", "y", "pred", "naive", "last", "yday"]
    ]
    tables = slice_table(pred)
    ok = (pred["naive"].notna() & pred["last"].notna() & pred["yday"].notna()).sum()
    for t in tables.values():
        assert t["n"].sum() == ok
        assert abs(t["share_of_trips"].sum() - 1) < 1e-9


def test_online_features_match_batch_features(panel):
    from tlc_demand.online import MIN_HISTORY, features_at_origin

    batch = make_features(panel, H)
    zone = 20
    g = panel[panel["zone"] == zone].reset_index(drop=True)
    rows = batch[batch["zone"] == zone].sample(25, random_state=0)
    checked = 0
    for _, row in rows.iterrows():
        pos = g.index[g["bin"] == row["origin"]][0]
        if pos + 1 < MIN_HISTORY:
            continue
        hist = g["y"].to_numpy()[pos + 1 - MIN_HISTORY : pos + 1]
        got = features_at_origin(hist, row["origin"], zone, H)
        for col, val in got.items():
            assert val == pytest.approx(row[col]), col
        checked += 1
    assert checked >= 10
