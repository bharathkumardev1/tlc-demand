"""Write the per-zone history snapshot the service loads.
python scripts/make_history.py data/raw 2023 6 artifacts/history.npz"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_backtest import load  # noqa: E402

from tlc_demand.online import MIN_HISTORY  # noqa: E402
from tlc_demand.panel import build_panel  # noqa: E402

raw, year, month, out = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
end = (pd.Timestamp(year, month, 1) + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")
panel = build_panel(load(raw, year, month, month), f"{year}-{month:02d}-01", end)
last = panel["bin"].max()
arrays = {f"zone_{z}": g.sort_values("bin")["y"].to_numpy(np.int32)[-MIN_HISTORY:] for z, g in panel.groupby("zone")}
np.savez_compressed(out, last_origin=str(last), **arrays)
# seasonal-naive value for the next target bin, for the gateway's fallback path
HORIZON = 4
from tlc_demand.panel import BINS_PER_WEEK  # noqa: E402
fallback = {z: int(a[-1 - (BINS_PER_WEEK - HORIZON)]) for z, a in ((int(k[5:]), v) for k, v in arrays.items())}
out.with_name("fallback.csv").write_text("zone,trips\n" + "".join(f"{z},{v}\n" for z, v in sorted(fallback.items())))
print(f"{len(arrays)} zones, last origin {last}")
