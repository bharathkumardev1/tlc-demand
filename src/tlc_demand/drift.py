"""Population stability index (PSI) monitors for the model's input features.

Raw lag and rolling counts differ by orders of magnitude between zones, so a window of requests
that happens to hit different zones, or a window taken at 3 am, would look like drift. Each monitored value
is therefore divided by the zone's mean count for the target weekday and hour in the reference period,
and PSI is computed separately inside four tiers of that expected count (a ratio built from small counts
is noisier than one built from large counts, so the tiers keep like with like) and averaged by the tier's
share of the window. The statistic then responds
to demand shifts, not to which zones were asked or what time it is.

PSI = sum_i (a_i - e_i) * ln((a_i + eps) / (e_i + eps)) over bins fixed from the reference
(deciles of the reference values). A common rule of thumb reads 0.1 as a small shift and 0.25 as
a large one; here the alert threshold is calibrated on clean periods instead (see scripts/run_drift_eval.py)."""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MONITORED = ("lag_0", "roll_mean_16", "roll_mean_96", "same_slot_1w")
EPS = 1e-4


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = EPS) -> float:
    e, a = np.asarray(expected, float), np.asarray(actual, float)
    return float(((a - e) * np.log((a + eps) / (e + eps))).sum())


N_TIERS = 4
MIN_TIER_ROWS = 50


def slot_key(zone, dow, hour):
    return np.asarray(zone) * 1000 + np.asarray(dow) * 24 + np.asarray(hour)


@dataclass
class Reference:
    slot_mean: dict[int, float]  # slot_key(zone, weekday, hour) -> mean count
    tier_cuts: list[float]  # boundaries between tiers of the expected count for the slot
    edges: dict[str, list[list[float]]]  # feature -> tier -> interior bin edges
    expected: dict[str, list[list[float]]]  # feature -> tier -> reference proportions per bin
    alert_threshold: float | None = None

    @classmethod
    def fit(cls, zone: np.ndarray, tod: np.ndarray, dow: np.ndarray, y: np.ndarray, values: dict[str, np.ndarray],
            n_bins: int = 10) -> "Reference":
        """One row per (zone, target bin) of the reference period: the target count `y`, the target's
        time of day `tod` (0..95) and weekday `dow`, and the monitored feature columns in `values`."""
        zone, tod, dow = np.asarray(zone), np.asarray(tod), np.asarray(dow)
        keys, slot_means = _group_mean(slot_key(zone, dow, tod // 4), np.asarray(y, float))
        ref = cls({int(k): float(m) for k, m in zip(keys, slot_means)}, [], {}, {})
        row_mean = ref.expected_count(zone, tod, dow)
        ref.tier_cuts = np.unique(np.quantile(row_mean, np.linspace(0, 1, N_TIERS + 1)[1:-1])).tolist()
        cuts = ref.tier_cuts
        tier = ref.tiers(zone, tod, dow)
        for name in MONITORED:
            rel = ref.relative(zone, tod, dow, values[name])
            ref.edges[name], ref.expected[name] = [], []
            for t in range(len(cuts) + 1):
                r = rel[(tier == t) & ~np.isnan(rel)]
                if len(r) == 0:  # no reference zone fell in this tier (very few zones)
                    ref.edges[name].append([])
                    ref.expected[name].append([1.0])
                    continue
                edges = np.unique(np.quantile(r, np.linspace(0, 1, n_bins + 1)[1:-1]))
                ref.edges[name].append(edges.tolist())
                ref.expected[name].append(_proportions(r, edges).tolist())
        return ref

    def expected_count(self, zone, tod, dow) -> np.ndarray:
        if not hasattr(self, "_lookup"):
            ks = np.array(sorted(self.slot_mean))
            self._lookup = (ks, np.array([self.slot_mean[int(k)] for k in ks]))
        ks, vs = self._lookup
        keys = np.asarray(slot_key(zone, dow, np.asarray(tod) // 4))
        pos = np.clip(np.searchsorted(ks, keys), 0, len(ks) - 1)
        return np.where(ks[pos] == keys, vs[pos], np.nan)

    def tiers(self, zone, tod, dow) -> np.ndarray:
        m = self.expected_count(zone, tod, dow)
        t = np.searchsorted(np.array(self.tier_cuts), m, side="right").astype(float)
        t[np.isnan(m)] = -1
        return t.astype(int)

    def relative(self, zone: np.ndarray, tod: np.ndarray, dow: np.ndarray, value: np.ndarray) -> np.ndarray:
        return np.asarray(value, float) / (self.expected_count(zone, tod, dow) + 1.0)

    def prepare(self, zone, tod, dow, values: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Tier and relative value of every row. Split out so long histories can be prepared once and scored in windows."""
        zone = np.asarray(zone)
        return self.tiers(zone, tod, dow), {n: self.relative(zone, tod, dow, values[n]) for n in MONITORED}

    def score(self, zone: np.ndarray, tod: np.ndarray, dow: np.ndarray, values: dict[str, np.ndarray]) -> dict[str, float]:
        return self.score_prepared(*self.prepare(zone, tod, dow, values))

    def score_prepared(self, tier: np.ndarray, rel_by_feature: dict[str, np.ndarray]) -> dict[str, float]:
        """PSI per feature, computed inside each tier of expected count and averaged with weights equal to the
        tier's share of the window. Which zones or hours were asked about changes the weights, not the
        reference distribution each tier is compared with."""
        out = {}
        for name in MONITORED:
            rel = rel_by_feature[name]
            num = den = 0.0
            for t in range(len(self.tier_cuts) + 1):
                r = rel[(tier == t) & ~np.isnan(rel)]
                if len(r) < MIN_TIER_ROWS:
                    continue
                p = psi(np.array(self.expected[name][t]), _proportions(r, np.array(self.edges[name][t])))
                num, den = num + len(r) * p, den + len(r)
            if den:
                out[name] = num / den
        return out

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"slot_mean": self.slot_mean, "tier_cuts": self.tier_cuts, "edges": self.edges,
                                          "expected": self.expected, "alert_threshold": self.alert_threshold}))

    @classmethod
    def load(cls, path: Path) -> "Reference":
        d = json.loads(Path(path).read_text())
        return cls({int(k): v for k, v in d["slot_mean"].items()}, d["tier_cuts"], d["edges"], d["expected"],
                   d.get("alert_threshold"))


def _group_mean(zone: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u, inv = np.unique(zone, return_inverse=True)
    return u, np.bincount(inv, weights=y) / np.bincount(inv)


def _proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges, values, side="right")
    return np.bincount(idx, minlength=len(edges) + 1) / len(values)


class DriftMonitor:
    """Keeps the most recent `window` requests and scores them against the reference."""

    def __init__(self, reference: Reference, window: int = 2000, min_fill: float = 0.5) -> None:
        self.ref, self.window, self.min_fill = reference, window, min_fill
        self._rows: deque[tuple[int, int, int, dict[str, float]]] = deque(maxlen=window)

    def update(self, zone: int, features: dict[str, float]) -> None:
        """`features` is the dict the model sees; it carries the target's tod and dow."""
        self._rows.append((zone, int(features["tod"]), int(features["dow"]), {k: features[k] for k in MONITORED}))

    @property
    def fill(self) -> float:
        return len(self._rows) / self.window

    def scores(self) -> dict[str, float] | None:
        """PSI per feature, or None until the window is at least `min_fill` full."""
        if self.fill < self.min_fill:
            return None
        zones, tod, dow = (np.array([r[i] for r in self._rows]) for i in range(3))
        vals = {n: np.array([r[3][n] for r in self._rows]) for n in MONITORED}
        return self.ref.score(zones, tod, dow, vals)

    def alert(self) -> bool:
        s = self.scores()
        thr = self.ref.alert_threshold
        return bool(s is not None and thr is not None and max(s.values()) > thr)
