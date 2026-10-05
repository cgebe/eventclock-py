import datetime as dt

import numpy as np
import pandas as pd
import pytest

from eventclock import as_event_prices, event_clock, load_dataset, pm_daily, pm_stitch

NY = "America/New_York"


def ny_ep(stamps: list[str], q: list[float]):
    t = pd.Series([pd.Timestamp(s, tz=NY) for s in stamps])
    return as_event_prices(pd.DataFrame({"time": t, "q": q}))


def test_pm_stitch_dedupes_and_sorts():
    c1 = pd.DataFrame({"t": [100, 200, 300], "p": [0.4, 0.5, 0.6]})
    c2 = pd.DataFrame({"t": [300, 400], "p": [0.99, 0.7]})
    out = pm_stitch([c1, c2])
    assert out["t"].tolist() == [100, 200, 300, 400]
    assert out["p"].tolist() == [0.4, 0.5, 0.6, 0.7]
    out2 = pm_stitch([None, c1, None])
    assert out2["t"].tolist() == c1["t"].tolist()
    assert len(pm_stitch([None, None])) == 0
    assert len(pm_stitch([])) == 0


def test_pm_stitch_keeps_the_first_chunk_before_sorting():
    c1 = pd.DataFrame({"t": [300, 100], "p": [0.6, 0.4]})
    c2 = pd.DataFrame({"t": [100, 50], "p": [0.99, 0.3]})
    out = pm_stitch([c1, c2])
    assert out["t"].tolist() == [50, 100, 300]
    assert out["p"].tolist() == [0.3, 0.4, 0.6]
    assert out.index.tolist() == [0, 1, 2]


def test_pm_daily_coerces_an_instant_event_date():
    ep = as_event_prices(
        load_dataset("polymarket2024"),
        market_id="Polymarket: Trump wins 2024",
        event_date=pd.Timestamp("2024-11-05", tz="UTC"),
    )
    daily = pm_daily(ep)
    assert daily.event_date == dt.date(2024, 11, 5)
    assert type(daily.event_date) is dt.date
    assert daily.market_id == "Polymarket: Trump wins 2024"
    late_ny = pd.Timestamp("2024-11-05 23:00", tz=NY)
    ep_ny = as_event_prices(load_dataset("polymarket2024"), event_date=late_ny)
    assert pm_daily(ep_ny).event_date == dt.date(2024, 11, 5)


def test_pm_daily_collapses_to_one_snapshot_per_day(ep_brexit):
    ep = ny_ep(
        [
            "2024-01-01 10:00", "2024-01-01 15:59", "2024-01-01 20:00",
            "2024-01-02 09:00", "2024-01-02 16:00",
            "2024-01-03 08:00",
        ],
        [0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
    )
    daily = pm_daily(ep, tz=NY, snapshot_hour=16)
    assert len(daily) == 3
    assert daily.time_kind == "date"
    assert daily.time.tolist() == [pd.Timestamp(f"2024-01-0{i}") for i in (1, 2, 3)]
    assert daily.q.tolist() == [0.45, 0.60, 0.65]
    with pytest.raises(ValueError, match="POSIXct"):
        pm_daily(ep_brexit)


def test_pm_daily_ignores_seconds_like_r():
    ep = ny_ep(["2024-01-01 15:00", "2024-01-01 16:00:30"], [0.4, 0.5])
    assert pm_daily(ep, tz=NY).q.tolist() == [0.5]


def test_pm_daily_ties_keep_the_first_row():
    ep = ny_ep(["2024-01-01 15:00:00", "2024-01-01 15:00:40"], [0.4, 0.5])
    assert pm_daily(ep, tz=NY).q.tolist() == [0.4]


def test_pm_daily_uses_the_local_date():
    t = pd.Series(
        [pd.Timestamp("2024-01-02 03:00", tz="UTC"), pd.Timestamp("2024-01-02 15:00", tz="UTC")]
    )
    ep = as_event_prices(pd.DataFrame({"time": t, "q": [0.4, 0.5]}))
    daily = pm_daily(ep, tz=NY, snapshot_hour=23)
    assert daily.time.tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    assert daily.q.tolist() == [0.4, 0.5]


def test_pm_daily_keeps_na_and_clip():
    ep = as_event_prices(
        pd.DataFrame({"time": pd.Series([pd.Timestamp("2024-01-01 12:00", tz=NY)]), "q": [np.nan]}),
        clip=(0.05, 0.95),
    )
    daily = pm_daily(ep)
    assert daily.clip == (0.05, 0.95)
    assert daily.data["flag_na"].tolist() == [True]


def test_pm_daily_with_nothing_before_the_snapshot():
    ep = ny_ep(["2024-01-01 20:00", "2024-01-02 21:00"], [0.4, 0.5])
    daily = pm_daily(ep, snapshot_hour=16)
    assert len(daily) == 0
    assert daily.time_kind == "date"
    assert list(daily.data.columns) == ["time", "q_raw", "q", "flag_na", "flag_clip"]


def test_pm_daily_on_the_shipped_data_is_stable():
    ep = as_event_prices(load_dataset("polymarket2024"))
    daily = pm_daily(ep)
    assert len(daily) == 158
    assert not daily.time.duplicated().any()
    assert ((daily.q > 0) & (daily.q < 1)).all()
    assert daily.q[daily.time == pd.Timestamp("2024-11-04")].item() == pytest.approx(0.578, abs=1e-6)


def test_polymarket2024_daily_vs_hourly_clock():
    ep = as_event_prices(load_dataset("polymarket2024"))
    daily = pm_daily(ep)
    a_daily = event_clock(daily, methods="rv")["A"].item()
    a_hourly = event_clock(ep, methods="rv")["A"].item()
    assert a_daily == pytest.approx(0.7756132, abs=1e-4)
    assert a_daily < a_hourly
