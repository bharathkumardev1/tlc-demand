"""Forecast service: features from a per-zone history store, ONNX model, Prometheus metrics.

The history store is a snapshot (one array of recent counts per zone). A real
deployment would fill it from a stream; here it is loaded at startup, and its
age is exported so a stale snapshot is visible instead of silently served."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from .drift import MONITORED, DriftMonitor, Reference
from .features import FEATURE_COLUMNS
from .online import MIN_HISTORY, features_at_origin

LATENCY = Histogram("forecast_latency_seconds", "Model path latency", buckets=(.001, .0025, .005, .01, .025, .05, .1, .25))
PREDICTION = Histogram("forecast_prediction", "Predicted trips per 15 min", buckets=(0, 1, 2, 5, 10, 20, 50, 100, 200, 500))
REQUESTS = Counter("forecast_requests_total", "Requests", ["outcome"])
SNAPSHOT_AGE = Gauge("history_snapshot_age_seconds", "Seconds since the snapshot's last bin")
FEATURE_PSI = Gauge("feature_psi", "PSI of recent requests against the training reference", ["feature"])
DRIFT_ALERT = Gauge("feature_drift_alert", "1 when the largest feature PSI exceeds the calibrated threshold")
DRIFT_FILL = Gauge("drift_window_fill", "Fraction of the drift window filled with recent requests")


@dataclass
class History:
    last_origin: pd.Timestamp
    counts: dict[int, np.ndarray]  # zone -> counts ending at last_origin

    @classmethod
    def load(cls, path: Path) -> "History":
        z = np.load(path)
        origin = pd.Timestamp(str(z["last_origin"]))
        return cls(origin, {int(k.removeprefix("zone_")): z[k] for k in z.files if k.startswith("zone_")})


class ForecastIn(BaseModel):
    zone: int = Field(ge=1, le=265)


class ForecastOut(BaseModel):
    zone: int
    origin: str
    target: str
    horizon_bins: int
    trips: float
    model_version: str


def create_app(model_path: Path | None = None, history_path: Path | None = None, horizon: int = 4,
               version: str = "dev", drift_reference: Path | None = None, drift_window: int = 2000) -> FastAPI:
    model_path = model_path or Path(os.environ["MODEL_PATH"])
    history_path = history_path or Path(os.environ["HISTORY_PATH"])
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    hist = History.load(history_path)
    drift_reference = drift_reference or (Path(os.environ["DRIFT_REFERENCE"]) if os.environ.get("DRIFT_REFERENCE") else None)
    monitor = DriftMonitor(Reference.load(drift_reference), window=drift_window) if drift_reference else None
    app = FastAPI(title="tlc-demand")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "zones": len(hist.counts), "last_origin": str(hist.last_origin), "version": version}

    @app.post("/v1/forecast", response_model=ForecastOut)
    def forecast(req: ForecastIn):
        t0 = time.perf_counter()
        y = hist.counts.get(req.zone)
        if y is None or len(y) < MIN_HISTORY:
            REQUESTS.labels("no_history").inc()
            raise HTTPException(404, f"no history for zone {req.zone}")
        f = features_at_origin(y[-MIN_HISTORY:], hist.last_origin, req.zone, horizon)
        if monitor is not None:
            monitor.update(req.zone, f)
        x = np.array([[f[c] for c in FEATURE_COLUMNS]], dtype=np.float32)
        trips = max(float(sess.run(None, {"features": x})[0].ravel()[0]), 0.0)
        LATENCY.observe(time.perf_counter() - t0)
        PREDICTION.observe(trips)
        REQUESTS.labels("ok").inc()
        return ForecastOut(zone=req.zone, origin=str(hist.last_origin),
                           target=str(hist.last_origin + pd.Timedelta(minutes=15 * horizon)),
                           horizon_bins=horizon, trips=round(trips, 3), model_version=version)

    @app.get("/metrics")
    def metrics():
        SNAPSHOT_AGE.set(max((pd.Timestamp.now() - hist.last_origin).total_seconds(), 0))
        if monitor is not None:
            DRIFT_FILL.set(monitor.fill)
            scores = monitor.scores()
            for name in MONITORED:
                FEATURE_PSI.labels(name).set(scores[name] if scores and name in scores else float("nan"))
            DRIFT_ALERT.set(1.0 if monitor.alert() else 0.0)
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
