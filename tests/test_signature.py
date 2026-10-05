import datetime as dt

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

import eventclock
from eventclock import as_event_prices, ec_signature, event_clock, load_dataset
from helpers import record_warnings

needs_simulate = pytest.mark.skipif(
    not hasattr(eventclock, "ec_simulate_path"), reason="needs ec_simulate_path (Step 10)"
)


def dated(q, start: dt.date = dt.date(2020, 1, 1)) -> pd.DataFrame:
    times = [start + dt.timedelta(days=i) for i in range(len(q))]
    return pd.DataFrame({"time": pd.Series(times, dtype=object), "q": list(q)})


def simulated_ep(seed: int, noise_sd: float = 0.0):
    rng = np.random.default_rng(seed)
    p = eventclock.ec_simulate_path(1, 400, q=0.4, A=1, rng=rng)
    q = expit(p["L"].to_numpy() + rng.normal(0, noise_sd, len(p))) if noise_sd else p["q"]
    return as_event_prices(dated(q), clip=(1e-8, 1 - 1e-8))


def test_hand_example():
    ep = as_event_prices(dated(expit(0.1 * np.arange(7))))
    sig = ec_signature(ep, max_every=3)
    assert list(sig.columns) == ["sample_every", "spacing_days", "n_incr", "A", "A_min", "A_max"]
    assert sig["sample_every"].tolist() == [1, 2, 3]
    assert sig["sample_every"].dtype == np.int64
    assert sig["spacing_days"].tolist() == [1.0, 2.0, 3.0]
    np.testing.assert_allclose(sig["n_incr"], [6, 2.5, 4 / 3], rtol=1e-15)
    np.testing.assert_allclose(sig["A"], [0.06, 0.10, 0.12], rtol=1e-12)
    np.testing.assert_allclose(sig["A_min"], [0.06, 0.08, 0.09], rtol=1e-12)
    np.testing.assert_allclose(sig["A_max"], [0.06, 0.12, 0.18], rtol=1e-12)


def test_k1_equals_event_clock(ep_brexit):
    sig = ec_signature(ep_brexit, max_every=10)
    assert sig["A"].iloc[0] == pytest.approx(event_clock(ep_brexit, methods="rv")["A"].item(), rel=1e-12)
    assert (sig["A_min"] <= sig["A"]).all()
    assert (sig["A"] <= sig["A_max"]).all()
    assert sig.attrs == {"market_id": "Brexit: Leave"}


def test_offsets_with_a_single_point_are_skipped():
    ep = as_event_prices(dated(expit(0.1 * np.arange(4))))
    sig = ec_signature(ep, max_every=3)
    np.testing.assert_allclose(sig["A"].iloc[2], 0.09, rtol=1e-12)
    np.testing.assert_allclose(sig["n_incr"].iloc[2], 1 / 3, rtol=1e-15)


def test_spacing_uses_the_median_step():
    ep = as_event_prices(load_dataset("polymarket2024"))
    sig = ec_signature(ep, max_every=3)
    np.testing.assert_allclose(sig["spacing_days"], np.array([1, 2, 3]) / 24, rtol=1e-6)


def test_window_bounds_and_na_warning():
    ep = as_event_prices(dated([0.40, 0.45, np.nan, 0.50, 0.55, 0.60, 0.65]))
    sig, warns = record_warnings(ec_signature, ep, max_every=2, from_=dt.date(2020, 1, 2))
    assert len(warns) == 1
    assert warns[0].startswith("1 missing observation")
    assert sig["n_incr"].iloc[0] == 4


def test_signature_input_validation():
    ep = as_event_prices(dated([0.5] * 6))
    with pytest.raises(ValueError, match="observations"):
        ec_signature(ep, max_every=10)
    with pytest.raises(ValueError, match="`max_every` must"):
        ec_signature(ep, max_every=0)
    assert len(ec_signature(ep, max_every=5.9)) == 5


@needs_simulate
def test_signature_is_flat_without_noise():
    ep = simulated_ep(3)
    sig = ec_signature(ep, max_every=8)
    assert sig["sample_every"].tolist() == list(range(1, 9))
    assert sig["spacing_days"].tolist() == [float(k) for k in range(1, 9)]
    assert ((sig["A_min"] <= sig["A"]) & (sig["A"] <= sig["A_max"])).all()
    assert (np.abs(sig["A"] / sig["A"].iloc[0] - 1) < 0.25).all()
    assert sig["A"].iloc[0] == pytest.approx(event_clock(ep, methods="rv")["A"].item(), rel=1e-12)


@needs_simulate
def test_signature_detects_microstructure_noise():
    sig = ec_signature(simulated_ep(4, noise_sd=0.05), max_every=10)
    assert sig["A"].iloc[0] > 1.5 * sig["A"].iloc[9]
