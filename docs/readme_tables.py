"""Render the README result tables from results/*.json.

    python docs/readme_tables.py --write   # fill the marked blocks in README.md
    python docs/readme_tables.py --check   # exit 1 if README.md differs from the results files
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
README = ROOT / "README.md"


def load(name):
    return json.loads((R / name).read_text())


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def backtest():
    b = load("backtest.json")
    keys = ["wape_model", "wape_naive", "wape_last", "wape_yesterday"]
    rows = [[f["test_start"][:10]] + [f"{f[k]:.4f}" for k in keys] + [f"{f['mae_model']:.3f}", f"{f['mae_naive']:.3f}"]
            for f in b["folds"]]
    n = len(b["folds"])
    avg = lambda k: sum(f[k] for f in b["folds"]) / n
    rows.append(["mean of folds"] + [f"{avg(k):.4f}" for k in keys] + [f"{avg('mae_model'):.3f}", f"{avg('mae_naive'):.3f}"])
    return table(["Test week starting", "WAPE model", "WAPE seasonal-naive (same slot last week)", "WAPE last count",
                  "WAPE same slot yesterday", "MAE model", "MAE seasonal-naive"], rows)


def volume_tiers():
    b = load("backtest.json")["slices"]["volume_tier"]
    rows = []
    for t in b:
        base = min(t["wape_naive"], t["wape_last"], t["wape_yesterday"])
        rows.append([t["volume_tier"], f"{t['share_of_trips']:.2%}", f"{t['wape_model']:.3f}", f"{t['wape_naive']:.3f}",
                     f"{t['wape_last']:.3f}", f"{t['wape_yesterday']:.3f}",
                     "model" if t["wape_model"] < base else "a baseline"])
    return table(["Zone volume quartile", "Share of trips", "WAPE model", "WAPE seasonal-naive", "WAPE last count",
                  "WAPE same slot yesterday", "Lowest WAPE"], rows)


def ablation():
    a = load("ablation.json")["runs"]
    base = a["full_seed0"]["wape_mean"]
    rows = []
    for k, v in a.items():
        d = "" if k == "full_seed0" else f"{v['wape_mean'] - base:+.5f}"
        rows.append([k, v["n_features"], f"{v['wape_mean']:.5f}", d])
    return table(["Run (seed 0 unless noted)", "Features", "WAPE", "Change vs full"], rows)


def hierarchy():
    h = load("hierarchy.json")["pooled_wape"]
    names = [("naive", "Seasonal-naive"), ("base", "Independent LightGBM per level"), ("bottom_up", "Bottom-up"),
             ("ols", "OLS reconciliation"), ("wls_variance", "WLS (variance)"), ("mint_shrink", "MinT, shrunk covariance")]
    rows = [[label, h[k]["zone"], h[k]["borough"], h[k]["city"]] for k, label in names]
    return table(["Pooled WAPE, three June folds", "Zone", "Borough", "City"], rows)


def sparse():
    s = load("sparse_zone_rule.json")
    p = s["pooled"]
    rows = [
        ["Seasonal-naive", f"{p['naive']['wape']:.4f}", f"{p['naive']['mae']:.4f}"],
        ["LightGBM everywhere", f"{p['model']['wape']:.4f}", f"{p['model']['mae']:.4f}"],
        [f"Routed: seasonal-naive below {s['chosen_threshold_trips_per_bin']:.3f} trips/bin",
         f"{p['routed']['wape']:.4f}", f"{p['routed']['mae']:.4f}"],
    ]
    return table(["Pooled, three June folds", "WAPE", "MAE"], rows)


def spikes():
    p = load("spikes.json")["pooled"]
    names = [("classifier", "LightGBM classifier"), ("regression_ratio", "Forecast / slot threshold"),
             ("lag0_ratio", "Latest count / slot threshold"), ("same_slot_1w_ratio", "Same slot last week / threshold")]
    rows = [[label, f"{p[k]['ap']:.3f}", f"{p[k]['ci95_day_bootstrap'][0]:.3f} to {p[k]['ci95_day_bootstrap'][1]:.3f}",
             f"{p[k]['lift_over_base_rate']:.1f}x"] for k, label in names]
    return table([f"Average precision, base rate {p['base_rate']:.4f}", "AP", "95% interval (bootstrap over days)", "Lift"], rows)


def drift():
    d = load("drift_injected_2023.json")
    fa = d["false_alarms_clean_2023-04-01_to_2023-06-14"]["24"]
    rows = []
    for name, label in [("control", "none (control)"), ("zones_x2", "2x demand in 20 busy zones"),
                        ("all_x1.3", "1.3x demand everywhere"), ("clock_shift_1h", "clock shifted +1 h")]:
        s = d["scenarios"][name]["24"]
        delay = "never" if s["detection_delay_hours"] is None else f"{s['detection_delay_hours']:.0f} h"
        rows.append([label, delay, f"{s['share_of_post_injection_windows_alerting']:.1%}", s["max_psi_after_injection"]])
    t = table(["24 h window, injection from 2023-06-15", "Hours to first alert", "Share of windows alerting", "Max PSI"], rows)
    thr = d["threshold_is_max_psi_in_calibration"]["24"]
    return t + f"\n\nThreshold {thr:.3f}. False alarms on clean April to mid-June data: {fa['alerts']} of {fa['evaluations']:,} hourly evaluations."


def parity():
    p = load("onnx_parity.json")
    q = p["abs_diff_quantiles_50_99_999_max"]
    rows = [["WAPE, LightGBM / ONNX", f"{p['wape_lightgbm']} / {p['wape_onnx']}"],
            ["Absolute difference, trips per zone-interval: median / p99 / p99.9 / max", " / ".join(str(x) for x in q)],
            ["Rows with absolute difference over 0.01", f"{p['rows_abs_diff_over_0.01']} of {p['rows']}"]]
    return table(["ONNX against LightGBM, June held-out week", "Value"], rows)


TABLES = {"backtest": backtest, "volume_tiers": volume_tiers, "ablation": ablation, "hierarchy": hierarchy,
          "sparse": sparse, "spikes": spikes, "drift": drift, "parity": parity}
PAT = re.compile(r"(<!-- table:(\w+) -->\n)(.*?)(<!-- /table:\2 -->)", re.S)


def main(mode):
    text = README.read_text()
    seen = set()

    def sub(mo):
        seen.add(mo.group(2))
        return mo.group(1) + TABLES[mo.group(2)]() + "\n" + mo.group(4)

    new = PAT.sub(sub, text)
    missing = set(TABLES) - seen
    if mode == "--write":
        README.write_text(new)
        print("wrote", sorted(seen), "missing blocks:", sorted(missing))
        return 0
    if new != text or missing:
        print("README tables are stale or missing:", sorted(missing) or "content differs")
        return 1
    print("README tables match results/*.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "--check"))
