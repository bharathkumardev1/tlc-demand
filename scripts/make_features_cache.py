"""Write the one-hour-horizon feature table that run_ablation.py, run_spikes.py and run_drift_eval.py read.

python scripts/make_features_cache.py data/raw 2023 1 6 artifacts/features_h4.parquet"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_backtest import load  # noqa: E402

from tlc_demand.features import make_features  # noqa: E402
from tlc_demand.panel import build_panel  # noqa: E402

raw, year, m0, m1, out = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
end = (pd.Timestamp(year, m1, 1) + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")
feats = make_features(build_panel(load(raw, year, m0, m1), f"{year}-{m0:02d}-01", end), 4)
out.parent.mkdir(parents=True, exist_ok=True)
feats.to_parquet(out, index=False)
print(f"{len(feats):,} rows written to {out}")
