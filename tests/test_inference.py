import datetime as dt
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from scipy.stats import norm

import eventclock
from eventclock import as_event_prices, event_clock
from eventclock.kernels import kernel_rv_boot

L_HAND = np.array([0, 0.1, 0.3, 0.35, 0.5])


def dated(q: list, start: dt.date = dt.date(2020, 1, 1)) -> pd.DataFrame:
    times = [start + dt.timedelta(days=i) for i in range(len(q))]
    return pd.DataFrame({"time": pd.Series(times, dtype=object), "q": q})


def hand_ep():
    return as_event_prices(dated(list(expit(L_HAND))))


def test_quarticity_se_matches_the_closed_form():
    res = event_clock(hand_ep(), methods="rv", se=True)
    dL = np.diff(L_HAND)
    assert res["se"].item() == pytest.approx(np.sqrt((2 / 3) * np.sum(dL**4)), rel=1e-12)
    assert res["ci_lo"].item() < res["A"].item()
    assert res["ci_hi"].item() > res["A"].item()
    assert res["ci_lo"].item() > 0
    res2 = event_clock(hand_ep(), methods=["rv", "bipower"], se=True)
    assert np.isnan(res2.loc[res2["method"] == "bipower", "se"].item())
    assert not np.isnan(res2.loc[res2["method"] == "rv", "se"].item())


def test_quarticity_ci_is_log_based():
    res = event_clock(hand_ep(), methods="rv", se=True, conf=0.9)
    A, s = res["A"].item(), res["se"].item()
    z = norm.ppf(0.95)
    assert res["ci_lo"].item() == pytest.approx(np.exp(np.log(A) - z * s / A), rel=1e-14)
    assert res["ci_hi"].item() == pytest.approx(np.exp(np.log(A) + z * s / A), rel=1e-14)
    wide = event_clock(hand_ep(), methods="rv", se=True, conf=0.99)
    assert wide["ci_lo"].item() < res["ci_lo"].item()
    assert wide["ci_hi"].item() > res["ci_hi"].item()


def test_bootstrap_se_agrees_with_quarticity(ep_brexit):
    rq = event_clock(ep_brexit, from_=dt.date(2016, 5, 24), methods="rv", se=True)
    rb = event_clock(
        ep_brexit, from_=dt.date(2016, 5, 24), methods="rv",
        se=True, se_method="bootstrap", boot_reps=4000, rng=7,
    )
    assert abs(rb["se"].item() - rq["se"].item()) / rq["se"].item() < 0.1
    assert rb["ci_lo"].item() < rb["A"].item()
    assert rb["ci_hi"].item() > rb["A"].item()


def test_bootstrap_uses_sd_and_type7_quantiles(ep_brexit):
    rb = event_clock(ep_brexit, methods="rv", se=True, se_method="bootstrap", boot_reps=500, rng=3, conf=0.8)
    dL = np.diff(logit(np.clip(ep_brexit.q.to_numpy(), 0.01, 0.99)))
    star = kernel_rv_boot(dL, reps=500, rng=3)
    assert rb["se"].item() == np.std(star, ddof=1)
    lo, hi = np.quantile(star, [0.1, 0.9], method="linear")
    assert (rb["ci_lo"].item(), rb["ci_hi"].item()) == (lo, hi)


def test_bootstrap_rng_is_reproducible(ep_brexit):
    kw = dict(methods="rv", se=True, se_method="bootstrap", boot_reps=200)
    a = event_clock(ep_brexit, rng=5, **kw)
    assert a.equals(event_clock(ep_brexit, rng=np.random.default_rng(5), **kw))
    assert not a.equals(event_clock(ep_brexit, rng=6, **kw))


def test_bootstrap_draws_differ_across_horizons(ep_brexit, brexit_horizons):
    same = {"a": dt.date(2016, 6, 23), "b": dt.date(2016, 6, 23)}
    res = event_clock(ep_brexit, to=same, methods="rv", se=True, se_method="bootstrap", boot_reps=200, rng=1)
    assert res["A"].iloc[0] == res["A"].iloc[1]
    assert res["se"].iloc[0] != res["se"].iloc[1]


def test_se_is_nan_when_not_estimable():
    flat = as_event_prices(dated([0.5, 0.5, 0.5]))
    res = event_clock(flat, methods="rv", se=True)
    assert res["A"].item() == 0
    assert res[["se", "ci_lo", "ci_hi"]].isna().all(axis=None)
    one = event_clock(as_event_prices(dated([0.4, 0.5])), methods="rv", se=True)
    assert one["n_incr"].item() == 1
    assert np.isnan(one["se"].item())
    dup = event_clock(hand_ep(), methods=["rv", "rv"], se=True)
    assert dup["se"].isna().all()
    no_rv = event_clock(hand_ep(), methods="bipower", se=True)
    assert np.isnan(no_rv["se"].item())


def test_single_bootstrap_rep_gives_nan_se_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = event_clock(hand_ep(), methods="rv", se=True, se_method="bootstrap", boot_reps=1, rng=0)
    assert np.isnan(res["se"].item())
    assert res["ci_lo"].item() == res["ci_hi"].item()


def test_se_columns():
    res = event_clock(hand_ep(), methods=["bipower", "rv"], se=True)
    assert list(res.columns)[-4:] == ["A", "se", "ci_lo", "ci_hi"]
    assert "se" not in event_clock(hand_ep(), methods="rv").columns
    for c in ("se", "ci_lo", "ci_hi"):
        assert res[c].dtype == np.float64


def test_event_clock_validates_se_arguments(ep_brexit):
    with pytest.raises(ValueError, match="`conf` must"):
        event_clock(ep_brexit, methods="rv", se=True, conf=1.2)
    with pytest.raises(ValueError, match="`se_method` must"):
        event_clock(ep_brexit, methods="rv", se=True, se_method="nope")


@pytest.mark.skipif(
    not hasattr(eventclock, "ec_simulate_path"), reason="needs ec_simulate_path (Step 10)"
)
def test_ci_coverage_is_close_to_nominal():
    rng = np.random.default_rng(11)
    A = 0.5
    hits = []
    for _ in range(300):
        p = eventclock.ec_simulate_path(1, 60, q=0.4, A=A, rng=rng)
        times = [dt.date(2020, 1, 1) + dt.timedelta(days=int(s)) for s in p["step"]]
        ep = as_event_prices(
            pd.DataFrame({"time": pd.Series(times, dtype=object), "q": p["q"]}),
            clip=(1e-8, 1 - 1e-8),
        )
        r = event_clock(ep, methods="rv", se=True, conf=0.90)
        hits.append(r["ci_lo"].item() <= A <= r["ci_hi"].item())
    assert 0.82 < np.mean(hits) < 0.97
