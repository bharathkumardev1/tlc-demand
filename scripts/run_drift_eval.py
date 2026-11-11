"""Calibrate and test the PSI drift monitor.

  python scripts/run_drift_eval.py 2023 data/raw      injected drift on 2023 data, false alarms on clean weeks
  python scripts/run_drift_eval.py 2020 data/raw2020  the real 2020 demand collapse

Reference: January (first week skipped, lags need history). Alert threshold: the largest PSI seen in
hourly windows of a clean calibration period that follows the reference (training windows only). Everything
after that period is test data. Windows trail the evaluation time and hold every zone's features for each
15-minute origin in the window, scored every hour."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_backtest import load  # noqa: E402

from tlc_demand.drift import MONITORED, Reference  # noqa: E402
from tlc_demand.features import make_features  # noqa: E402
from tlc_demand.panel import build_panel  # noqa: E402

H = 4
WINDOWS = (24, 6)  # hours
COLS = ["zone", "origin", "target_bin", "tod", "dow", "y", *MONITORED]


def prep(feats: pd.DataFrame):
    f = feats[COLS].dropna(subset=list(MONITORED)).sort_values("origin", kind="stable").reset_index(drop=True)
    return f


def lean_features(panel: pd.DataFrame, chunk: int = 40) -> pd.DataFrame:
    """make_features over groups of zones, keeping only the monitored columns as float32 so the full
    half-year history does not have to sit in memory at once."""
    zones = sorted(panel["zone"].unique())
    parts = []
    for i in range(0, len(zones), chunk):
        f = make_features(panel[panel["zone"].isin(zones[i:i + chunk])], H)[[c for c in COLS if c != "dow"]]
        f["dow"] = f["target_bin"].dt.day_of_week.astype("int8")
        f = f.astype({c: "float32" for c in [*MONITORED, "y"]})
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def fit_reference(f: pd.DataFrame, start: str, end: str) -> Reference:
    m = (f["target_bin"] >= start) & (f["target_bin"] < end)
    r = f[m]
    return Reference.fit(r["zone"].to_numpy(), r["tod"].to_numpy(), r["dow"].to_numpy(), r["y"].to_numpy(),
                         {n: r[n].to_numpy(np.float64) for n in MONITORED})


def window_series(ref: Reference, f: pd.DataFrame, start: str, end: str, hours: int) -> pd.Series:
    """Largest-feature PSI at each hourly evaluation time in [start, end)."""
    tier, rel = ref.prepare(f["zone"].to_numpy(), f["tod"].to_numpy(), f["dow"].to_numpy(),
                            {n: f[n].to_numpy(np.float64) for n in MONITORED})
    origin = f["origin"].to_numpy()
    times = pd.date_range(pd.Timestamp(start) + pd.Timedelta(hours=hours), end, freq="h", inclusive="left")
    out = []
    for t in times:
        lo = np.searchsorted(origin, np.datetime64(t - pd.Timedelta(hours=hours)), side="right")
        hi = np.searchsorted(origin, np.datetime64(t), side="right")
        s = ref.score_prepared(tier[lo:hi], {n: rel[n][lo:hi] for n in MONITORED})
        out.append(max(s.values()) if s else np.nan)
    return pd.Series(out, index=times)


def calibrate(ref, f, start, end):
    thr, stats = {}, {}
    for w in WINDOWS:
        s = window_series(ref, f, start, end, w)
        thr[w] = float(s.max())
        stats[w] = {"n_windows": int(s.notna().sum()), "max": round(float(s.max()), 5), "p99": round(float(s.quantile(.99)), 5),
                    "median": round(float(s.median()), 5)}
    return thr, stats


def alerts(series: pd.Series, thr: float) -> pd.Series:
    return series[series > thr]


def run_2023(raw: Path):
    t0 = time.time()
    feats = pd.read_parquet("artifacts/features_h4.parquet", columns=[c for c in COLS if c not in ("dow",)])
    feats["dow"] = feats["target_bin"].dt.day_of_week
    f = prep(feats)
    del feats
    ref = fit_reference(f, "2023-01-08", "2023-02-01")
    thr, cal = calibrate(ref, f, "2023-02-01", "2023-04-01")
    print("calibration", cal, thr, round(time.time() - t0), "s", flush=True)
    ref.alert_threshold = thr[24]
    ref.save(Path("artifacts/drift_reference.json"))
    clean = {}
    for w in WINDOWS:
        s = window_series(ref, f, "2023-04-01", "2023-06-15", w)
        a = alerts(s, thr[w])
        clean[w] = {"evaluations": int(s.notna().sum()), "alerts": int(len(a)), "days_with_alerts": sorted({str(t.date()) for t in a.index}),
                    "max": round(float(s.max()), 5)}
    print("clean", clean, flush=True)

    trips = load(raw, 2023, 5, 6)
    panel = build_panel(trips, "2023-05-01", "2023-07-01")
    del trips
    inject_at = pd.Timestamp("2023-06-15")
    mean_by_zone = panel[panel["bin"] < inject_at].groupby("zone")["y"].mean()
    rng = np.random.default_rng(0)
    group = sorted(int(z) for z in rng.choice(mean_by_zone[mean_by_zone >= 2].index.to_numpy(), size=20, replace=False))

    def variant(kind):
        p = panel.copy()
        after = p["bin"] >= inject_at
        if kind == "zones_x2":
            m = after & p["zone"].isin(group)
            p.loc[m, "y"] = np.rint(p.loc[m, "y"] * 2.0).astype(p["y"].dtype)
        elif kind == "all_x1.3":
            p.loc[after, "y"] = np.rint(p.loc[after, "y"] * 1.3).astype(p["y"].dtype)
        elif kind == "clock_shift_1h":
            p = p.sort_values(["zone", "bin"]).reset_index(drop=True)
            shifted = p.groupby("zone")["y"].shift(4)
            p["y"] = np.where(p["bin"] >= inject_at, shifted, p["y"]).astype(p["y"].dtype)
        return p

    res = {}
    for kind in ("control", "zones_x2", "all_x1.3", "clock_shift_1h"):
        p = panel if kind == "control" else variant(kind)
        g = prep(make_features(p, H))
        res[kind] = {}
        for w in WINDOWS:
            s = window_series(ref, g, "2023-06-08", "2023-07-01", w)
            a = alerts(s, thr[w])
            before, after = a[a.index < inject_at], a[a.index >= inject_at]
            post = s[s.index >= inject_at]
            first = after.index.min() if len(after) else None
            res[kind][w] = {"alerts_before_injection_jun8_to_jun15": int(len(before)),
                            "first_alert_after_injection": None if first is None else str(first),
                            "detection_delay_hours": None if first is None else (first - inject_at) / pd.Timedelta(hours=1),
                            "share_of_post_injection_windows_alerting": round(float((post > thr[w]).mean()), 4),
                            "max_psi_after_injection": round(float(post.max()), 5)}
        print(kind, res[kind], flush=True)
    Path("results/drift_injected_2023.json").write_text(json.dumps({
        "reference": "2023-01-08 to 2023-01-31", "calibration_period": "2023-02-01 to 2023-03-31 (hourly windows)",
        "threshold_is_max_psi_in_calibration": {str(k): v for k, v in thr.items()}, "calibration": {str(k): v for k, v in cal.items()},
        "false_alarms_clean_2023-04-01_to_2023-06-14": {str(k): v for k, v in clean.items()},
        "injection_time": str(inject_at), "zones_x2_group": group, "scenarios": res, "windows_hours": list(WINDOWS),
        "seconds": round(time.time() - t0)}, indent=2, default=str))


def run_2020(raw: Path):
    t0 = time.time()
    trips = load(raw, 2020, 1, 6)
    panel = build_panel(trips, "2020-01-01", "2020-07-01")
    del trips
    daily = panel.assign(day=panel["bin"].dt.normalize()).groupby("day")["y"].sum()
    f = prep(lean_features(panel))
    ref = fit_reference(f, "2020-01-08", "2020-02-01")
    thr, cal = calibrate(ref, f, "2020-02-01", "2020-03-01")
    print("calibration", cal, thr, flush=True)
    feb = daily["2020-02-03":"2020-02-29"]
    wk_mean = feb.groupby(feb.index.dayofweek).mean()
    ratio = (daily["2020-03-01":] / daily["2020-03-01":].index.dayofweek.map(wk_mean).to_numpy())
    drop = ratio[ratio < 0.8]
    out = {"reference": "2020-01-08 to 2020-01-31", "calibration_period": "2020-02-01 to 2020-02-29",
           "threshold_is_max_psi_in_calibration": {str(k): v for k, v in thr.items()}, "calibration": {str(k): v for k, v in cal.items()},
           "first_day_total_trips_below_80pct_of_feb_same_weekday": None if drop.empty else str(drop.index[0].date()),
           "windows": {}}
    for w in WINDOWS:
        s = window_series(ref, f, "2020-03-01", "2020-07-01", w)
        a = alerts(s, thr[w])
        d = s.groupby(s.index.normalize()).max()
        out["windows"][str(w)] = {
            "first_alert": None if a.empty else str(a.index[0]),
            "share_of_windows_alerting": round(float((s > thr[w]).mean()), 4),
            "daily_max_psi_and_trip_ratio": {str(k.date()): [round(float(v), 4), round(float(ratio.get(k, np.nan)), 3)] for k, v in d.items()}}
        print(w, "first alert", out["windows"][str(w)]["first_alert"], flush=True)
    out["seconds"] = round(time.time() - t0)
    Path("results/drift_2020.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    {"2023": run_2023, "2020": run_2020}[sys.argv[1]](Path(sys.argv[2]))
