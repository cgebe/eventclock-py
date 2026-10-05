import datetime as dt

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from eventclock import (
    EventClockWarning,
    as_event_prices,
    event_clock,
    event_clock_forecast,
    event_clock_path,
    load_dataset,
)
from eventclock.clock import gap_stats, window_increments, window_rows
from helpers import record_warnings

NY = "America/New_York"


def dated(start: dt.date, q: list) -> pd.DataFrame:
    times = [start + dt.timedelta(days=i) for i in range(len(q))]
    return pd.DataFrame({"time": pd.Series(times, dtype=object), "q": q})


def test_brexit_reproduces_the_paper_table(ep_brexit, brexit_horizons, clock_value):
    res = event_clock(ep_brexit, from_=dt.date(2016, 5, 24), to=brexit_horizons)
    assert clock_value(res, "1W", "rv") == pytest.approx(0.06391206, abs=1e-6)
    assert clock_value(res, "2W", "rv") == pytest.approx(0.16574907, abs=1e-6)
    assert clock_value(res, "1M", "rv") == pytest.approx(0.51046899, abs=1e-6)
    assert clock_value(res, "1M", "bipower") == pytest.approx(0.37418353, abs=1e-6)
    assert clock_value(res, "1M", "largest1") == pytest.approx(0.42547325, abs=1e-6)
    assert clock_value(res, "1M", "largest2") == pytest.approx(0.34076037, abs=1e-6)

    assert abs(clock_value(res, "1W", "rv") - 0.064) < 1.1e-3
    assert abs(clock_value(res, "2W", "rv") - 0.166) < 1.1e-3
    assert abs(clock_value(res, "1M", "rv") - 0.511) < 1.1e-3
    assert abs(clock_value(res, "1M", "bipower") - 0.375) < 1.1e-3
    assert abs(clock_value(res, "1M", "largest1") - 0.426) < 1.1e-3

    assert res.loc[res["horizon"] == "1W", "n_obs"].iloc[0] == 8
    assert res.loc[res["horizon"] == "2W", "n_obs"].iloc[0] == 15
    assert res.loc[res["horizon"] == "1M", "n_obs"].iloc[0] == 31


def test_us_reproduces_the_paper_table(ep_us, us_horizons, clock_value):
    res = event_clock(ep_us, from_=dt.date(2016, 10, 10), to=us_horizons)
    assert clock_value(res, "1W", "rv") == pytest.approx(0.01520781, abs=1e-6)
    assert clock_value(res, "2W", "rv") == pytest.approx(0.04600595, abs=1e-6)
    assert clock_value(res, "1M", "rv") == pytest.approx(0.37046375, abs=1e-6)
    assert clock_value(res, "1M", "bipower") == pytest.approx(0.38639860, abs=1e-6)
    assert clock_value(res, "1M", "truncated") == pytest.approx(0.25877271, abs=1e-6)
    assert clock_value(res, "1M", "largest1") == pytest.approx(0.25877271, abs=1e-6)

    assert abs(clock_value(res, "1W", "rv") - 0.015) < 1.1e-3
    assert abs(clock_value(res, "2W", "rv") - 0.046) < 1.1e-3
    assert abs(clock_value(res, "1M", "rv") - 0.370) < 1.1e-3
    assert abs(clock_value(res, "1M", "bipower") - 0.386) < 1.1e-3
    assert abs(clock_value(res, "1M", "truncated") - 0.259) < 1.1e-3


def test_polymarket2024_full_sample_hourly():
    ep = as_event_prices(load_dataset("polymarket2024"))
    assert event_clock(ep, methods="rv")["A"].item() == pytest.approx(1.263125, abs=1e-4)


def test_wedge_invariance(ep_brexit, brexit_horizons):
    shifted = load_dataset("brexit2016")
    shifted["q_leave"] = expit(logit(shifted["q_leave"]) + 0.5)
    ep2 = as_event_prices(shifted, time="date", price="q_leave")
    a1 = event_clock(ep_brexit, from_=dt.date(2016, 5, 24), to=brexit_horizons, methods="rv")
    a2 = event_clock(ep2, from_=dt.date(2016, 5, 24), to=brexit_horizons, methods="rv")
    np.testing.assert_allclose(a1["A"], a2["A"], rtol=1e-12)


def test_sparse_sampling_and_window_logic():
    ep = as_event_prices(dated(dt.date(2020, 1, 1), list(expit(np.arange(5) * 0.1))))
    assert event_clock(ep, methods="rv")["A"].item() == pytest.approx(4 * 0.01, rel=1e-12)
    r2 = event_clock(ep, methods="rv", sample_every=2)
    assert r2["A"].item() == pytest.approx(2 * 0.04, rel=1e-12)
    assert r2["n_incr"].item() == 2
    r3 = event_clock(ep, from_=dt.date(2020, 1, 2), to=dt.date(2020, 1, 4), methods="rv")
    assert r3["A"].item() == pytest.approx(2 * 0.01, rel=1e-12)


def test_na_values_are_skipped_with_a_warning():
    ep = as_event_prices(dated(dt.date(2020, 1, 1), [0.5, np.nan, 0.55, 0.6, np.nan]))
    res, warns = record_warnings(event_clock, ep, methods="rv")
    assert warns == [
        "2 missing observations inside the window skipped; adjacent increments bridge "
        "these gaps (see the Missing observations section of `?event_clock`)."
    ]
    assert res["n_obs"].item() == 3
    assert not np.isnan(res["A"].item())


def test_gap_diagnostics(ep_brexit):
    ep = as_event_prices(dated(dt.date(2020, 1, 1), [0.50, 0.52, np.nan, 0.55, 0.57, 0.60]))
    with pytest.warns(EventClockWarning, match="missing observation"):
        res = event_clock(ep, methods=["rv", "largest1"])
    assert res["n_gaps"].unique().tolist() == [1]
    assert res["max_gap_days"].unique().tolist() == [2.0]
    res2 = event_clock(ep_brexit, methods="rv")
    assert res2["n_gaps"].item() == 0
    assert res2["max_gap_days"].item() == 1.0


def test_date_bounds_on_ny_series():
    t = pd.Series([pd.Timestamp("2020-01-01 12:00", tz=NY) + pd.Timedelta(days=i) for i in range(5)])
    ep = as_event_prices(pd.DataFrame({"time": t, "q": [0.40, 0.45, 0.50, 0.55, 0.60]}))
    res = event_clock(ep, from_=dt.date(2020, 1, 2), to=dt.date(2020, 1, 4), methods="rv")
    assert res["n_obs"].item() == 3


def test_summary_a_full_matches_event_clock(ep_brexit):
    s = ep_brexit.summary()
    assert s.loc[0, "n"] == 119
    assert s.loc[0, "market_id"] == "Brexit: Leave"
    assert s.loc[0, "n_na"] == 0
    full = event_clock(ep_brexit, methods="rv")["A"].item()
    assert s.loc[0, "A_full"] == pytest.approx(full, rel=1e-12)


def test_output_columns_and_types(ep_brexit):
    res = event_clock(ep_brexit, to={"a": dt.date(2016, 5, 31)}, methods=("rv", "bipower"))
    assert list(res.columns) == [
        "market_id", "from", "to", "horizon", "n_obs", "n_incr",
        "n_gaps", "max_gap_days", "method", "A",
    ]
    assert res["method"].tolist() == ["rv", "bipower"]
    assert res["from"].iloc[0] == pd.Timestamp("2016-02-26")
    assert res["to"].iloc[0] == pd.Timestamp("2016-05-31")
    for c in ("n_obs", "n_incr", "n_gaps"):
        assert res[c].dtype == np.int64
    assert res["n_incr"].iloc[0] == res["n_obs"].iloc[0] - 1


def test_horizon_labels_follow_r_format():
    ep = as_event_prices(load_dataset("polymarket2024"))
    mid = [pd.Timestamp("2024-10-01", tz="UTC"), pd.Timestamp("2024-10-02", tz="UTC")]
    assert event_clock(ep, to=mid, methods="rv")["horizon"].tolist() == ["2024-10-01", "2024-10-02"]
    mixed = [mid[0], pd.Timestamp("2024-10-02 13:00", tz="UTC")]
    assert event_clock(ep, to=mixed, methods="rv")["horizon"].tolist() == [
        "2024-10-01 00:00:00",
        "2024-10-02 13:00:00",
    ]
    named = {"A": mid[0], "": mid[1]}
    assert event_clock(ep, to=named, methods="rv")["horizon"].tolist() == ["A", "2024-10-02"]
    assert event_clock(ep, methods="rv")["horizon"].item() == "2024-11-05 23:00:02"


def test_empty_windows_give_nan():
    ep = as_event_prices(dated(dt.date(2020, 1, 1), [0.4, 0.5, 0.6]))
    res = event_clock(ep, from_=dt.date(2020, 1, 3))
    assert res["n_obs"].unique().tolist() == [1]
    assert res["n_incr"].unique().tolist() == [0]
    assert res["A"].isna().all()
    assert res["max_gap_days"].isna().all()
    res = event_clock(ep, sample_every=5)
    assert res["n_obs"].unique().tolist() == [3]
    assert res["A"].isna().all()


def test_sample_every_is_truncated_like_as_integer():
    ep = as_event_prices(dated(dt.date(2020, 1, 1), list(expit(np.arange(5) * 0.1))))
    a = event_clock(ep, methods="rv", sample_every=2.9)
    b = event_clock(ep, methods="rv", sample_every=2)
    assert a.equals(b)


def test_clip_argument_overrides_the_object():
    ep = as_event_prices(dated(dt.date(2020, 1, 1), [0.001, 0.5]))
    a = event_clock(ep, methods="rv")["A"].item()
    b = event_clock(ep, methods="rv", clip=(0.0001, 0.9999))["A"].item()
    assert a == pytest.approx((logit(0.5) - logit(0.01)) ** 2)
    assert b == pytest.approx((logit(0.5) - logit(0.001)) ** 2)


def test_accepts_a_data_frame():
    res = event_clock(dated(dt.date(2020, 1, 1), [0.4, 0.5]), methods="rv")
    assert res["n_obs"].item() == 2


def test_input_validation():
    ep = as_event_prices(dated(dt.date(2020, 1, 1), [0.4, 0.5, 0.6]))
    with pytest.raises(ValueError, match="`methods` must"):
        event_clock(ep, methods="bip")
    with pytest.raises(ValueError, match="`methods` must"):
        event_clock(ep, methods=[])
    with pytest.raises(ValueError, match="`sample_every` must"):
        event_clock(ep, sample_every=0.5)
    for bad in (0, 1, 1.5, np.nan):
        with pytest.raises(ValueError, match="`conf` must"):
            event_clock(ep, conf=bad)
    with pytest.raises(ValueError, match="`se_method` must"):
        event_clock(ep, se_method="boot")
    with pytest.raises(ValueError, match="`to` must"):
        event_clock(ep, to=[dt.date(2020, 1, 2), pd.Timestamp("2020-01-03", tz="UTC")])
    with pytest.raises(ValueError, match="must be a date"):
        event_clock(ep, from_="2020-01-01")


def test_duplicate_methods_are_kept(ep_brexit):
    res = event_clock(ep_brexit, methods=["rv", "rv"])
    assert res["method"].tolist() == ["rv", "rv"]
    assert res["A"].iloc[0] == res["A"].iloc[1]


def test_empty_to_list(ep_brexit):
    res = event_clock(ep_brexit, to=[])
    assert len(res) == 0
    assert "A" in res.columns


def test_window_helpers(ep_brexit):
    rows = window_rows(ep_brexit, dt.date(2016, 5, 24), dt.date(2016, 5, 31))
    assert len(rows) == 8
    assert gap_stats(rows["time"]) == (0, 1.0)
    n_gaps, max_gap = gap_stats(rows["time"].iloc[:1])
    assert n_gaps == 0
    assert np.isnan(max_gap)
    w = window_increments(ep_brexit, dt.date(2016, 5, 24), dt.date(2016, 5, 31), 3, ep_brexit.clip)
    assert w.n_obs == 8
    assert len(w.dL) == 2
    assert w.max_gap_days == 3.0


def test_forecast_reproduces_the_paper_numbers(ep_us, ep_brexit):
    fus = event_clock_forecast(ep_us, at=dt.date(2016, 10, 10), horizon={"1W": 7, "2W": 14})
    a1w = fus.loc[fus["horizon"] == "1W", "A_forecast"].item()
    a2w = fus.loc[fus["horizon"] == "2W", "A_forecast"].item()
    assert a1w == pytest.approx(0.06809, abs=1e-4)
    assert a2w == pytest.approx(0.13618, abs=1e-4)
    assert abs(a1w - 0.068) < 1.1e-3
    assert abs(a2w - 0.135) < 1.5e-3
    assert fus["n_incr"].unique().tolist() == [39]

    fgb = event_clock_forecast(ep_brexit, at=dt.date(2016, 5, 24), horizon={"1W": 7, "2W": 14})
    assert fgb.loc[fgb["horizon"] == "1W", "A_forecast"].item() == pytest.approx(0.0439927, abs=1e-4)
    assert fgb.loc[fgb["horizon"] == "2W", "A_forecast"].item() == pytest.approx(0.0879854, abs=1e-4)
    assert fgb["A_forecast"].iloc[1] / fgb["A_forecast"].iloc[0] == pytest.approx(2, rel=1e-15)

    fd = event_clock_forecast(ep_us, at=dt.date(2016, 10, 10), horizon=dt.date(2016, 10, 17))
    assert fd["horizon_days"].item() == 7
    assert fd["A_forecast"].item() == a1w
    assert fd["horizon"].item() == "2016-10-17"


def test_forecast_on_the_constructed_series():
    ep = as_event_prices(dated(dt.date(2020, 1, 1), list(expit(np.arange(5) * 0.1))))
    f = event_clock_forecast(ep, horizon=8, trailing=5)
    assert f["A_forecast"].item() == pytest.approx(0.04 / 4 * 8, rel=1e-12)
    assert f["horizon"].item() == "8d"
    assert f["at"].item() == pd.Timestamp("2020-01-05")


def test_path_is_a_consistent_cumulative_clock(ep_brexit):
    path = event_clock_path(ep_brexit)
    assert list(path.columns) == ["time", "q", "L", "dL", "dA", "A", "A_frac", "cal_frac"]
    assert len(path) == 119
    assert np.isnan(path["dL"].iloc[0])
    assert path["A"].iloc[0] == 0
    full = event_clock(ep_brexit, methods="rv")["A"].item()
    assert path["A"].iloc[-1] == pytest.approx(full, rel=1e-12)
    assert path["A_frac"].iloc[-1] == 1
    assert path["cal_frac"].iloc[0] == 0
    assert path["cal_frac"].iloc[-1] == 1
    assert (np.diff(path["A"]) >= 0).all()
    assert path.attrs == {"market_id": "Brexit: Leave", "event_date": dt.date(2016, 6, 23)}


def test_clock_functions_validate_inputs():
    ep = as_event_prices(dated(dt.date(2020, 1, 1), [0.5]))
    with pytest.raises(ValueError, match="at least 2"):
        event_clock_path(ep)
    with pytest.raises(ValueError, match="at least 2") as e:
        event_clock_forecast(ep, horizon=7)
    assert str(e.value) == 'Need at least 2 non-missing observations up to "2020-01-01".'
    ep2 = as_event_prices(dated(dt.date(2020, 1, 1), [0.5] * 5))
    with pytest.raises(ValueError, match="strictly after"):
        event_clock_forecast(ep2, horizon=-1)
    with pytest.raises(ValueError, match="strictly after"):
        event_clock_forecast(ep2, horizon=dt.date(2020, 1, 5))


def test_forecast_gap_diagnostics(ep_us):
    f = event_clock_forecast(ep_us, at=dt.date(2016, 10, 10), horizon=7)
    assert f["n_gaps"].item() == 0
    assert f["max_gap_days"].item() == 1.0


def test_path_respects_date_bounds_on_ny_series():
    t = pd.Series([pd.Timestamp("2020-01-01 12:00", tz=NY) + pd.Timedelta(days=i) for i in range(5)])
    ep = as_event_prices(pd.DataFrame({"time": t, "q": [0.40, 0.45, 0.50, 0.55, 0.60]}))
    path = event_clock_path(ep, from_=dt.date(2020, 1, 2), to=dt.date(2020, 1, 4))
    assert len(path) == 3
    assert str(path["time"].dt.tz) == NY


def test_path_options():
    ep = as_event_prices(dated(dt.date(2020, 1, 1), [0.5] * 4))
    path = event_clock_path(ep)
    assert path["A"].iloc[-1] == 0
    assert path["A_frac"].isna().all()
    assert list(event_clock_path(ep, normalize=False).columns) == ["time", "q", "L", "dL", "dA", "A"]
    assert len(event_clock_path(ep, sample_every=2)) == 2
    with pytest.raises(ValueError, match="at least 2"):
        event_clock_path(ep, sample_every=4)
    with pytest.raises(ValueError, match="single value"):
        event_clock_path(ep, to=[dt.date(2020, 1, 2), dt.date(2020, 1, 3)])
    assert len(event_clock_path(ep, to=[dt.date(2020, 1, 2)])) == 2


def test_path_skips_na_with_a_warning():
    ep = as_event_prices(dated(dt.date(2020, 1, 1), [0.5, np.nan, 0.6]))
    path, warns = record_warnings(event_clock_path, ep)
    assert len(path) == 2
    assert len(warns) == 1
    assert warns[0].startswith("1 missing observation inside the window skipped")


def test_forecast_horizon_labels():
    ep = as_event_prices(dated(dt.date(2020, 1, 1), list(expit(np.arange(5) * 0.1))))
    f = event_clock_forecast(ep, horizon=[7, 7.5, 100000])
    assert f["horizon"].tolist() == ["7d", "7.5d", "1e+05d"]
    f = event_clock_forecast(ep, horizon={"a": 7, "": 3})
    assert f["horizon"].tolist() == ["a", "3d"]
    f = event_clock_forecast(ep, horizon={"a": dt.date(2020, 1, 12), "": dt.date(2020, 1, 8)})
    assert f["horizon"].tolist() == ["a", "3d"]
    assert f["horizon_days"].tolist() == [7.0, 3.0]
    f = event_clock_forecast(ep, horizon=[dt.date(2020, 1, 12), dt.date(2020, 1, 8)])
    assert f["horizon"].tolist() == ["2020-01-12", "2020-01-08"]


def test_forecast_trailing_and_subsampling():
    ep = as_event_prices(dated(dt.date(2020, 1, 1), list(expit(np.arange(9) * 0.1))))
    f = event_clock_forecast(ep, horizon=1, trailing=3)
    assert f["n_incr"].item() == 2
    assert f["trailing"].item() == 3
    assert f["A_forecast"].item() == pytest.approx(0.02 / 2, rel=1e-12)
    f = event_clock_forecast(ep, horizon=1, trailing=9, sample_every=4)
    assert f["n_incr"].item() == 2
    assert f["A_forecast"].item() == pytest.approx(2 * 0.16 / 8, rel=1e-12)
    with pytest.raises(ValueError, match="zero calendar span"):
        event_clock_forecast(ep, horizon=1, trailing=1)
    with pytest.raises(ValueError, match="`trailing` must"):
        event_clock_forecast(ep, horizon=1, trailing=0)


def test_forecast_instant_series_and_horizons():
    ep = as_event_prices(load_dataset("polymarket2024"))
    at = pd.Timestamp("2024-10-01 12:00", tz="UTC")
    f = event_clock_forecast(ep, at=at, horizon=pd.Timestamp("2024-10-02 00:00", tz="UTC"))
    assert f["horizon_days"].item() == 0.5
    assert f["horizon"].item() == "2024-10-02"
    assert f["at"].item() == at
    f2 = event_clock_forecast(ep, at=at, horizon=dt.date(2024, 10, 2))
    assert f2["horizon_days"].item() == 0.5
