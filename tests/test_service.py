import lightgbm as lgb
import numpy as np
import pytest
from fastapi.testclient import TestClient
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

from tlc_demand.features import FEATURE_COLUMNS, make_features
from tlc_demand.online import MIN_HISTORY, features_at_origin
from tlc_demand.panel import build_panel
from tlc_demand.service import create_app

from synthetic import synthetic_trips

H = 4


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("svc")
    panel = build_panel(synthetic_trips(days=21), "2023-01-02", "2023-01-23")
    feats = make_features(panel, H)
    model = lgb.train({"objective": "poisson", "verbose": -1, "num_leaves": 7, "seed": 0, "deterministic": True,
                       "num_threads": 1}, lgb.Dataset(feats[FEATURE_COLUMNS], feats["y"]), 30)
    onx = convert_lightgbm(model, initial_types=[("features", FloatTensorType([None, len(FEATURE_COLUMNS)]))],
                           target_opset=15)
    (d / "model.onnx").write_bytes(onx.SerializeToString())
    last = panel["bin"].max()
    arrays = {f"zone_{z}": g["y"].to_numpy(np.int32)[-MIN_HISTORY:] for z, g in panel.groupby("zone")}
    np.savez(d / "history.npz", last_origin=str(last), **arrays)
    app = create_app(d / "model.onnx", d / "history.npz", horizon=H, version="test")
    return TestClient(app), model, arrays, last


def test_forecast_matches_direct_model_call(setup):
    client, model, arrays, last = setup
    r = client.post("/v1/forecast", json={"zone": 20})
    assert r.status_code == 200
    body = r.json()
    f = features_at_origin(arrays["zone_20"], last, 20, H)
    want = max(float(model.predict(np.array([[f[c] for c in FEATURE_COLUMNS]]))[0]), 0)
    assert body["trips"] == pytest.approx(want, abs=0.05 * max(want, 1))
    assert body["model_version"] == "test" and body["horizon_bins"] == H


def test_unknown_zone_is_404_and_bad_zone_is_422(setup):
    client = setup[0]
    assert client.post("/v1/forecast", json={"zone": 99}).status_code == 404
    assert client.post("/v1/forecast", json={"zone": 0}).status_code == 422


def test_metrics_expose_requests_and_latency(setup):
    client = setup[0]
    client.post("/v1/forecast", json={"zone": 10})
    text = client.get("/metrics").text
    assert 'forecast_requests_total{outcome="ok"}' in text
    assert "forecast_latency_seconds_bucket" in text and "history_snapshot_age_seconds" in text


def test_onnx_agrees_with_lightgbm_in_aggregate_and_at_p99(tmp_path):
    """Bounds from docs/adr/001: float32 trees are not bit-identical, so compare distributions, not rows."""
    import onnxruntime as ort

    panel = build_panel(synthetic_trips(days=21), "2023-01-02", "2023-01-23")
    feats = make_features(panel, H)
    X = feats[FEATURE_COLUMNS]
    model = lgb.train({"objective": "poisson", "verbose": -1, "num_leaves": 7, "seed": 0, "deterministic": True,
                       "num_threads": 1}, lgb.Dataset(X, feats["y"]), 30)
    onx = convert_lightgbm(model, initial_types=[("features", FloatTensorType([None, len(FEATURE_COLUMNS)]))],
                           target_opset=15)
    sess = ort.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
    ref = model.predict(X.to_numpy(np.float64))
    got = sess.run(None, {"features": X.to_numpy(np.float32)})[0].ravel()
    diff = np.abs(ref - got)
    assert np.quantile(diff, 0.99) < 0.1
    assert diff.sum() / max(ref.sum(), 1.0) < 0.001


def test_drift_gauges_are_exported_and_alert_follows_the_threshold(tmp_path):
    """Wiring only: a single snapshot gives requests with three distinct feature vectors, so PSI against the
    reference is large whatever the history. Whether PSI separates drift from no drift is tested in test_drift.py
    and measured in results/drift.json."""
    from tlc_demand.drift import MONITORED, Reference

    panel = build_panel(synthetic_trips(days=21), "2023-01-02", "2023-01-23")
    feats = make_features(panel, H)
    ref = Reference.fit(feats["zone"].to_numpy(), feats["tod"].to_numpy(), feats["dow"].to_numpy(), feats["y"].to_numpy(),
                        {n: feats[n].to_numpy() for n in MONITORED})
    model = lgb.train({"objective": "poisson", "verbose": -1, "num_leaves": 7, "seed": 0, "num_threads": 1},
                      lgb.Dataset(feats[FEATURE_COLUMNS], feats["y"]), 10)
    onx = convert_lightgbm(model, initial_types=[("features", FloatTensorType([None, len(FEATURE_COLUMNS)]))],
                           target_opset=15)
    (tmp_path / "m.onnx").write_bytes(onx.SerializeToString())
    arrays = {f"zone_{z}": g["y"].to_numpy().astype(np.int32)[-MIN_HISTORY:] for z, g in panel.groupby("zone")}
    np.savez(tmp_path / "h.npz", last_origin=str(panel["bin"].max()), **arrays)

    def run(threshold, requests):
        ref.alert_threshold = threshold
        ref.save(tmp_path / "ref.json")
        c = TestClient(create_app(tmp_path / "m.onnx", tmp_path / "h.npz", horizon=H, version="t",
                                  drift_reference=tmp_path / "ref.json", drift_window=200))
        for i in range(requests):
            assert c.post("/v1/forecast", json={"zone": [10, 20, 30][i % 3]}).status_code == 200
        text = c.get("/metrics").text
        val = lambda name: float([ln for ln in text.splitlines() if ln.startswith(name)][-1].split()[-1])  # noqa: E731
        return val("drift_window_fill"), val("feature_drift_alert"), val('feature_psi{feature="lag_0"}')

    fill, alert, psi_lag0 = run(1e9, 300)
    assert fill == 1.0 and alert == 0.0 and psi_lag0 > 0
    assert run(0.0, 300)[1] == 1.0
    fill, alert, psi_lag0 = run(0.0, 10)  # window not filled enough: no scores, no alert
    assert fill == 0.05 and alert == 0.0 and psi_lag0 != psi_lag0
