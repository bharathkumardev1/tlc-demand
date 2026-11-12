"""Compare ONNX and LightGBM predictions on the last week of real features.
Tolerance is stated in the ADR: float32 trees are not bit-identical to float64.

python scripts/check_onnx_parity.py artifacts/ data/raw 2023 6
Arguments: directory holding model.txt and model.onnx, directory of raw parquet files, year, month.
Scores the last seven days of that month and writes results/onnx_parity.json."""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent))
from run_backtest import load  # noqa: E402

from tlc_demand.features import FEATURE_COLUMNS, make_features  # noqa: E402
from tlc_demand.metrics import wape  # noqa: E402
from tlc_demand.panel import build_panel  # noqa: E402

art = Path(sys.argv[1])
raw, year, month = Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
f = make_features(build_panel(load(raw, year, month, month), f"{year}-{month:02d}-01",
                              (f"{year}-{month+1:02d}-01" if month < 12 else f"{year+1}-01-01")), 4)
f = f[f["target_bin"] >= f["target_bin"].max().normalize() - np.timedelta64(6, "D")]
X64 = f[FEATURE_COLUMNS].to_numpy(np.float64)
X32 = X64.astype(np.float32)
m = lgb.Booster(model_file=str(art / "model.txt"))
sess = ort.InferenceSession(str(art / "model.onnx"), providers=["CPUExecutionProvider"])
ref = m.predict(X64)
onnx = sess.run(None, {"features": X32})[0].ravel()
d = np.abs(ref - onnx)
rel = d / np.maximum(ref, 1.0)
y = f["y"].to_numpy()
report = {
    "rows": int(len(f)),
    "abs_diff_quantiles_50_99_999_max": [round(float(x), 5) for x in np.quantile(d, [.5, .99, .999, 1])],
    "rel_diff_quantiles_99_999_max": [round(float(x), 5) for x in np.quantile(rel, [.99, .999, 1])],
    "rows_abs_diff_over_0.01": int((d > 0.01).sum()),
    "lightgbm_float64_vs_float32_rounded_inputs_max_abs": round(float(np.abs(ref - m.predict(X32.astype(np.float64))).max()), 5),
    "wape_lightgbm": round(wape(y, np.clip(ref, 0, None)), 5),
    "wape_onnx": round(wape(y, np.clip(onnx, 0, None)), 5),
}
Path("results/onnx_parity.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=1))
