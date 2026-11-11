import numpy as np
import pytest

from tlc_demand.drift import MONITORED, DriftMonitor, Reference, psi

RNG = np.random.default_rng(5)


def make_frame(n=20000, scale=1.0, zones=None, hours=None):
    """Counts with a zone level and a day/night pattern: mean = zone * (1 + (hour >= 12))."""
    zone = RNG.integers(1, 21, size=n) if zones is None else RNG.choice(zones, size=n)
    hour = RNG.integers(0, 24, size=n) if hours is None else RNG.choice(hours, size=n)
    dow = RNG.integers(0, 7, size=n)
    level = zone * (1.0 + (hour >= 12)) * scale
    y = RNG.poisson(level)
    vals = {name: RNG.poisson(level).astype(float) for name in MONITORED}
    return dict(zone=zone, tod=hour * 4, dow=dow, y=y, values=vals)


def fit(f, **kw):
    return Reference.fit(f["zone"], f["tod"], f["dow"], f["y"], f["values"], **kw)


def score(ref, f):
    return ref.score(f["zone"], f["tod"], f["dow"], f["values"])


def test_psi_basics():
    e = np.array([0.1, 0.2, 0.3, 0.4])
    assert psi(e, e) == pytest.approx(0.0, abs=1e-12)
    assert psi(e, np.array([0.4, 0.3, 0.2, 0.1])) > 0.2
    assert np.isfinite(psi(e, np.array([0.0, 0.0, 0.0, 1.0])))  # empty bins do not blow up
    assert psi(e, np.array([0.1, 0.2, 0.35, 0.35])) < psi(e, np.array([0.1, 0.1, 0.4, 0.4]))  # grows with the shift


def test_same_distribution_scores_low_and_grows_with_demand_shift():
    ref = fit(make_frame())
    assert max(score(ref, make_frame()).values()) < 0.02
    scores = [max(score(ref, make_frame(scale=s)).values()) for s in (1.1, 1.5, 3.0)]
    assert scores == sorted(scores) and scores[-1] > 0.5


def test_request_mix_and_time_of_day_do_not_look_like_drift():
    ref = fit(make_frame())
    assert max(score(ref, make_frame(6000, zones=np.arange(15, 21))).values()) < 0.05  # only high-volume zones
    assert max(score(ref, make_frame(6000, hours=np.arange(0, 6))).values()) < 0.05  # only night hours
    assert max(score(ref, make_frame(6000, hours=np.arange(12, 24))).values()) < 0.05  # only the busy half of the day


def test_reference_roundtrip(tmp_path):
    ref = fit(make_frame(4000))
    ref.alert_threshold = 0.12
    ref.save(tmp_path / "r.json")
    back = Reference.load(tmp_path / "r.json")
    f2 = make_frame(1000)
    assert score(back, f2) == pytest.approx(score(ref, f2)) and back.alert_threshold == 0.12


def feed(mon, f):
    for i in range(len(f["zone"])):
        feats = {n: f["values"][n][i] for n in MONITORED} | {"tod": f["tod"][i], "dow": f["dow"][i]}
        mon.update(int(f["zone"][i]), feats)


def test_monitor_waits_for_window_then_alerts_on_shift():
    ref = fit(make_frame())
    ref.alert_threshold = 0.1
    mon = DriftMonitor(ref, window=4000)
    feed(mon, make_frame(1000))
    assert mon.scores() is None and not mon.alert()  # under min_fill
    for _ in range(3):
        feed(mon, make_frame(1000))
    assert mon.scores() is not None and not mon.alert()
    feed(mon, make_frame(4000, scale=3.0))
    assert mon.alert()
