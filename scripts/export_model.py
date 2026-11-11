"""Train the serving model on all data before the last week, export to ONNX,
and verify the ONNX predictions against LightGBM on the held-out week.

python scripts/export_model.py data/raw 2023 1 6 artifacts/"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import onnxruntime as ort
import pandas as pd
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

from tlc_demand.features import FEATURE_COLUMNS, make_features
from tlc_demand.panel import build_panel
from tlc_demand.metrics import wape
from run_backtest import load  # noqa: E402 (script dir is on sys.path when run as a script)

ap = argparse.ArgumentParser()
ap.add_argument("raw", type=Path)
ap.add_argument("year", type=int)
ap.add_argument("m0", type=int)
ap.add_argument("m1", type=int)
ap.add_argument("out", type=Path)
ap.add_argument("--horizon", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

trips = load(a.raw, a.year, a.m0, a.m1)
end = (pd.Timestamp(a.year, a.m1, 1) + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")
panel = build_panel(trips, f"{a.year}-{a.m0:02d}-01", end)
feats = make_features(panel, a.horizon)
cut = feats["target_bin"].max().normalize() - pd.Timedelta(days=6)
gap = pd.Timedelta(minutes=15 * a.horizon)
train, test = feats[feats["target_bin"] < cut - gap], feats[feats["target_bin"] >= cut]
params = {"objective": "poisson", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50,
          "feature_fraction": 0.9, "verbose": -1, "seed": a.seed, "deterministic": True, "num_threads": 2}
t0 = time.time()
# zone is numeric here: the ONNX converter does not carry categorical splits
model = lgb.train(params, lgb.Dataset(train[FEATURE_COLUMNS], train["y"]), num_boost_round=300)
print(f"trained in {time.time()-t0:.0f}s", flush=True)
a.out.mkdir(parents=True, exist_ok=True)
model.save_model(str(a.out / "model.txt"))
onx = convert_lightgbm(model, initial_types=[("features", FloatTensorType([None, len(FEATURE_COLUMNS)]))],
                       target_opset=15)
(a.out / "model.onnx").write_bytes(onx.SerializeToString())

X = test[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
ref = model.predict(X)
sess = ort.InferenceSession(str(a.out / "model.onnx"), providers=["CPUExecutionProvider"])
got = sess.run(None, {"features": X})[0].ravel()
report = {
    "n_test": int(len(X)), "max_abs_diff_onnx_vs_lightgbm": float(np.abs(ref - got).max()),
    "mean_abs_diff": float(np.abs(ref - got).mean()),
    "wape_lightgbm": wape(test["y"].to_numpy(), np.clip(ref, 0, None)),
    "wape_onnx": wape(test["y"].to_numpy(), np.clip(got, 0, None)),
    "feature_columns": FEATURE_COLUMNS, "horizon_bins": a.horizon,
}
(a.out / "export_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "feature_columns"}, indent=1))
