import datetime as dt
import logging

import numpy as np
import pandas as pd
import pytest
from scipy.stats import median_abs_deviation

from eventclock import EventClockWarning, ec_default_params, ec_ilogit, ec_logit
from eventclock._utils import (
    align_bound,
    check_nonneg,
    check_prob,
    clip_q,
    logger,
    r_mad,
)

NY = "America/New_York"


def ny(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=NY)


def test_default_params_are_complete():
    p = ec_default_params()
    assert list(p) == ["clip", "methods", "sample_every", "trunc_sd", "trailing"]
    assert p["clip"] == (0.01, 0.99)
    assert p["methods"] == ("rv", "truncated", "bipower", "largest1", "largest2")
    assert p["sample_every"] == 1
    assert p["trunc_sd"] == 3
    assert p["trailing"] == 40


def test_default_params_are_modifiable_copies():
    p = ec_default_params()
    p2 = {**p, "trunc_sd": 4}
    assert p2["trunc_sd"] == 4
    assert p2["clip"] == p["clip"]
    p["trailing"] = 99
    assert ec_default_params()["trailing"] == 40


def test_clip_q_clips():
    np.testing.assert_array_equal(
        clip_q([0.005, 0.5, 0.995], (0.01, 0.99)), [0.01, 0.5, 0.99]
    )


def test_clip_q_keeps_nan_and_scalars():
    out = clip_q([np.nan, 0.2], (0.01, 0.99))
    assert np.isnan(out[0])
    assert out[1] == 0.2
    assert clip_q(0.001, (0.01, 0.99)) == 0.01
    assert isinstance(clip_q(0.5, (0.01, 0.99)), float)


@pytest.mark.parametrize(
    "clip",
    [(0.9, 0.1), (0, 0.99), (0.01, 1), (0.5, 0.5), (0.01,), (0.01, 0.5, 0.99), (np.nan, 0.99), 0.5],
)
def test_clip_q_validates_bounds(clip):
    with pytest.raises(ValueError, match="must"):
        clip_q(0.5, clip)


def test_logit_matches_r():
    assert ec_logit(0.3) == pytest.approx(-0.84729786038720356, rel=1e-15)
    assert ec_ilogit(-1.25) == pytest.approx(0.22270013882530884, rel=1e-15)
    assert ec_logit(0.5) == 0.0


def test_logit_edges_and_round_trip():
    q = np.array([0.01, 0.2, 0.5, 0.77, 0.99])
    np.testing.assert_allclose(ec_ilogit(ec_logit(q)), q, rtol=1e-14)
    out = ec_logit([0.0, 1.0, np.nan])
    assert out[0] == -np.inf
    assert out[1] == np.inf
    assert np.isnan(out[2])


def test_check_prob():
    q = [0.2, np.nan, 0.8]
    assert check_prob(q) is q
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="strictly inside"):
            check_prob([0.5, bad])
    with pytest.raises(ValueError, match="`target`"):
        check_prob(1.0, "target")


def test_check_nonneg():
    A = [0.0, np.nan, 2.0]
    assert check_nonneg(A) is A
    with pytest.raises(ValueError, match="non-negative"):
        check_nonneg([1.0, -1e-12])
    with pytest.raises(ValueError, match="`A_t`"):
        check_nonneg(-1.0, "A_t")


def test_r_mad_matches_r():
    dL = [0.01, 0.012, -0.011, 0.009, -0.013, 0.014, -0.01, 0.011, 0.5]
    assert r_mad(dL) == 0.0059303999999999997
    assert r_mad([0.3, 0.1, 0.7, 0.2]) == 0.14825999999999998


def test_r_mad_uses_the_r_constant():
    assert r_mad([1.0, 2.0, 3.0, 4.0, 100.0]) == 1.4826
    assert r_mad([1.0, 2.0, 3.0, 4.0, 100.0]) != median_abs_deviation(
        [1.0, 2.0, 3.0, 4.0, 100.0], scale="normal"
    )


def test_r_mad_degenerate():
    assert np.isnan(r_mad([]))
    assert np.isnan(r_mad([0.1, np.nan, 0.3]))
    assert r_mad([0.1] * 5) == 0.0


def test_align_bound_none_and_same_kind():
    assert align_bound(None, "date") is None
    assert align_bound(None, "instant", NY, "end") is None
    d = dt.date(2016, 5, 24)
    assert align_bound(d, "date", side="end") is d
    assert align_bound(ny("2020-01-02 12:00"), "instant", "UTC") == ny("2020-01-02 12:00")


def test_align_bound_date_on_ny_series():
    d = dt.date(2020, 1, 2)
    assert align_bound(d, "instant", NY, "start") == ny("2020-01-02 00:00:00")
    assert align_bound(d, "instant", NY, "end") == ny("2020-01-02 23:59:59")
    assert str(align_bound(d, "instant", NY).tz) == NY


def test_align_bound_date_on_utc_series():
    d = dt.date(2024, 11, 5)
    utc = pd.Timestamp("2024-11-05 00:00:00", tz="UTC")
    assert align_bound(d, "instant", "UTC") == utc
    assert align_bound(d, "instant", None) == utc
    assert align_bound(d, "instant", "UTC", "end") == pd.Timestamp("2024-11-05 23:59:59", tz="UTC")


def test_align_bound_date_on_dst_days_matches_r():
    assert align_bound(dt.date(2020, 3, 8), "instant", NY, "end") == ny("2020-03-09 00:59:59")
    assert align_bound(dt.date(2020, 11, 1), "instant", NY, "end") == ny("2020-11-01 22:59:59")


def test_align_bound_instant_on_date_series_uses_bound_tz():
    late_ny = ny("2016-05-24 23:30")
    assert late_ny.tz_convert("UTC").date() == dt.date(2016, 5, 25)
    assert align_bound(late_ny, "date") == dt.date(2016, 5, 24)
    assert align_bound(late_ny.tz_convert("UTC"), "date") == dt.date(2016, 5, 25)
    assert align_bound(dt.datetime(2016, 5, 24, 23, 30), "date") == dt.date(2016, 5, 24)


def test_align_bound_naive_instant_is_utc():
    out = align_bound(dt.datetime(2020, 1, 2, 12), "instant", NY)
    assert out == pd.Timestamp("2020-01-02 12:00", tz="UTC")
    assert align_bound(np.datetime64("2020-01-02T12:00"), "instant", NY) == out


def test_align_bound_ny_noon_window():
    times = pd.Series([ny("2020-01-01 12:00") + pd.Timedelta(days=i) for i in range(5)])
    lo = align_bound(dt.date(2020, 1, 2), "instant", NY, "start")
    hi = align_bound(dt.date(2020, 1, 4), "instant", NY, "end")
    assert int(((times >= lo) & (times <= hi)).sum()) == 3


def test_align_bound_rejects_bad_input():
    with pytest.raises(ValueError, match="must be"):
        align_bound("2020-01-02", "date")
    with pytest.raises(ValueError, match="must be"):
        align_bound(dt.date(2020, 1, 2), "date", side="middle")
    with pytest.raises(ValueError, match="must be"):
        align_bound(dt.date(2020, 1, 2), "posix")


def test_warning_class_and_logger():
    assert issubclass(EventClockWarning, UserWarning)
    assert logger is logging.getLogger("eventclock")
