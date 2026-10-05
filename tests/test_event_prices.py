import datetime as dt

import numpy as np
import pandas as pd
import pytest

from eventclock import EventPrices, as_event_prices
from helpers import record_warnings

NY = "America/New_York"


def d(y: int, m: int, day: int) -> dt.date:
    return dt.date(y, m, day)


def dated(times: list, q: list) -> pd.DataFrame:
    return pd.DataFrame({"time": pd.Series(times, dtype=object), "q": q})


def test_r_example_drops_sorts_and_warns():
    x = dated(
        [d(2020, 1, 3), d(2020, 1, 1), None, d(2020, 1, 2), d(2020, 1, 2), None],
        [0.3, 0.1, 0.9, 0.2, 0.25, 0.8],
    )
    ep, warns = record_warnings(as_event_prices, x)
    assert warns == [
        "2 rows with missing timestamp removed.",
        "1 duplicated timestamp removed (first occurrence kept).",
    ]
    assert ep.time_kind == "date"
    assert list(ep.data.columns) == ["time", "q_raw", "q", "flag_na", "flag_clip"]
    assert str(ep.time.dtype) == "datetime64[ns]"
    assert ep.time.tolist() == [pd.Timestamp(f"2020-01-0{i}") for i in (1, 2, 3)]
    assert ep.q.tolist() == [0.1, 0.2, 0.3]
    assert ep.data.index.tolist() == [0, 1, 2]


def test_singular_missing_timestamp_warning():
    x = dated([d(2020, 1, 3), d(2020, 1, 1), None, d(2020, 1, 2)], [0.3, 0.1, 0.9, 0.2])
    _, warns = record_warnings(as_event_prices, x)
    assert warns == ["1 row with missing timestamp removed."]


def test_flags():
    x = dated([d(2020, 1, i) for i in range(1, 6)], [0.5, np.nan, 0.005, 0.995, 0.01])
    ep = as_event_prices(x)
    assert ep.data["flag_na"].tolist() == [False, True, False, False, False]
    assert ep.data["flag_clip"].tolist() == [False, False, True, True, False]
    ep2 = as_event_prices(x, clip=(0.001, 0.999))
    assert ep2.data["flag_clip"].sum() == 0
    assert ep2.clip == (0.001, 0.999)


def test_idempotent():
    ep = as_event_prices(dated([d(2020, 1, 1), d(2020, 1, 2)], [0.4, 0.5]))
    assert as_event_prices(ep) is ep
    assert as_event_prices(ep, market_id="other") is ep


def test_column_guessing_uses_candidate_order():
    x = pd.DataFrame(
        {
            "Date": pd.Series([d(2020, 1, 1), d(2020, 1, 2)], dtype=object),
            "TIME": pd.Series([d(2021, 1, 1), d(2021, 1, 2)], dtype=object),
            "value": [0.9, 0.9],
            "P": [0.3, 0.4],
        }
    )
    ep = as_event_prices(x)
    assert ep.time.dt.year.tolist() == [2021, 2021]
    assert ep.q.tolist() == [0.3, 0.4]


def test_column_errors_match_r():
    x = dated([d(2020, 1, 1)], [0.5])
    with pytest.raises(ValueError) as e:
        as_event_prices(x, time="zz")
    assert str(e.value) == 'Column "zz" not found in `x`.'
    with pytest.raises(ValueError) as e:
        as_event_prices(pd.DataFrame({"a": [1], "q": [2]}))
    assert str(e.value) == (
        "Cannot guess the time column.\n"
        'i Pass `time` explicitly; tried "time", "date", "timestamp", "datetime", and "t".'
    )
    with pytest.raises(ValueError, match="Cannot guess the price column"):
        as_event_prices(pd.DataFrame({"time": x["time"], "a": [0.5]}))
    with pytest.raises(ValueError) as e:
        as_event_prices(x, bid="b", ask="a")
    assert str(e.value) == 'Columns "b" / "a" not found in `x`.'


@pytest.mark.parametrize("bad", [[1, 2], [1.5, 2.5], [True, False], [None, None]])
def test_time_must_be_date_or_instant(bad):
    with pytest.raises(ValueError) as e:
        as_event_prices(pd.DataFrame({"time": bad, "q": [0.5, 0.5]}))
    assert str(e.value) == 'The time column "time" must be <Date> or <POSIXct>.'


def test_bid_ask_mid():
    x = pd.DataFrame(
        {"time": pd.Series([d(2020, 1, 1), d(2020, 1, 2)], dtype=object),
         "b": [0.40, 0.50], "a": [0.44, 0.52], "q": [0.9, 0.9]}
    )
    np.testing.assert_allclose(as_event_prices(x, bid="b", ask="a").q, [0.42, 0.51])
    assert as_event_prices(x, bid="b").q.tolist() == [0.9, 0.9]


def test_scale_discount_and_overround():
    x = dated([d(2020, 1, 1), d(2020, 1, 2)], [45.0, 60.0])
    ep = as_event_prices(x, scale=100)
    np.testing.assert_allclose(ep.data["q_raw"], [0.45, 0.60])
    np.testing.assert_allclose(ep.q, [0.45, 0.60])
    ep = as_event_prices(x, scale=100, discount=0.9)
    np.testing.assert_allclose(ep.data["q_raw"], [0.45, 0.60])
    np.testing.assert_allclose(ep.q, [0.5, 0.6 / 0.9])
    ep = as_event_prices(x, scale=100, book=1.05, method="overround")
    np.testing.assert_allclose(ep.q, [0.45 / 1.05, 0.60 / 1.05])
    for bad in (0, -1, np.nan, "1", True, [1]):
        with pytest.raises(ValueError, match="`scale` must be"):
            as_event_prices(x, scale=bad)
    with pytest.raises(ValueError, match="`method` must be one of"):
        as_event_prices(x, method="x")


def test_price_coercion_warns_like_r():
    x = pd.DataFrame({"time": ["2020-01-01", "2020-01-02"], "q": ["0.5", "abc"]})
    ep, warns = record_warnings(as_event_prices, x)
    assert warns == ["NAs introduced by coercion"]
    assert ep.q.iloc[0] == 0.5
    assert np.isnan(ep.q.iloc[1])
    assert ep.data["flag_na"].tolist() == [False, True]


def test_naive_datetimes_become_utc_instants():
    x = pd.DataFrame({"time": pd.to_datetime(["2020-01-01 10:00", "2020-01-01 11:00"]), "q": [0.4, 0.5]})
    ep = as_event_prices(x)
    assert ep.time_kind == "instant"
    assert ep.tz == "UTC"
    assert ep.time.iloc[0] == pd.Timestamp("2020-01-01 10:00", tz="UTC")


def test_aware_instants_keep_their_tz():
    t = pd.Series([pd.Timestamp("2020-01-01 12:00", tz=NY) + pd.Timedelta(days=i) for i in range(3)])
    ep = as_event_prices(pd.DataFrame({"time": t, "q": [0.4, 0.45, 0.5]}))
    assert ep.time_kind == "instant"
    assert ep.tz == NY
    assert str(ep.time.dtype) == f"datetime64[ns, {NY}]"
    obj = pd.DataFrame({"time": pd.Series(list(t), dtype=object), "q": [0.4, 0.45, 0.5]})
    assert as_event_prices(obj).tz == NY


def test_mixed_tz_objects_go_to_utc():
    t = [pd.Timestamp("2020-01-01 12:00", tz=NY), pd.Timestamp("2020-01-02 12:00", tz="Europe/Zurich")]
    ep = as_event_prices(pd.DataFrame({"time": pd.Series(t, dtype=object), "q": [0.4, 0.5]}))
    assert ep.tz == "UTC"
    assert ep.time.tolist() == [v.tz_convert("UTC") for v in t]


STRING_CASES = [
    (["2020-01-05 10:20:30", "2020-01-05 10:20"], ["2020-01-05 10:20:00", "2020-01-05 10:20:00"]),
    (["2020-01-05 10:20:30.25", None, "2020-01-06 00:00:00"], ["2020-01-05 10:20:30.25", None, "2020-01-06 00:00:00"]),
    (["2020-01-05T10:20:30", "2020-01-06T11:00:00"], ["2020-01-05 00:00:00", "2020-01-06 00:00:00"]),
    (["2020-01-05", "2020-01-06 12:00"], ["2020-01-05 00:00:00", "2020-01-06 00:00:00"]),
    (["05.01.2020", "6.1.2020"], ["2020-01-05 00:00:00", "2020-01-06 00:00:00"]),
    (["2020-01-05 24:00:00", "2020-01-05 23:59:60"], ["2020-01-06 00:00:00", "2020-01-06 00:00:00"]),
    (["2020-1-5 9:5:3", " 2020-01-06 10:00:00"], ["2020-01-05 09:05:03", "2020-01-06 10:00:00"]),
    (["2020-02-29 00:00:00", "2020-01-05abc"], ["2020-02-29 00:00:00", "2020-01-05 00:00:00"]),
]


@pytest.mark.parametrize(("raw", "expected"), STRING_CASES)
def test_string_times_parse_like_r(raw, expected):
    from eventclock.event_prices import _parse_time_strings

    got = _parse_time_strings(pd.Series(raw, dtype=object), "time")
    want = [pd.NaT if e is None else pd.Timestamp(e, tz="UTC") for e in expected]
    assert str(got.dtype) == "datetime64[ns, UTC]"
    assert [None if pd.isna(v) else v for v in got] == [None if pd.isna(v) else v for v in want]


@pytest.mark.parametrize(
    "raw",
    [["2020-02-29", "2020-02-30"], ["2020-01-05", "05.01.2020"], ["20200105"], [""], ["2020-13-01"]],
)
def test_string_times_that_r_rejects(raw):
    with pytest.raises(ValueError, match="Could not parse the time column"):
        as_event_prices(pd.DataFrame({"time": raw, "q": [0.5] * len(raw)}))


def test_string_times_in_as_event_prices():
    x = pd.DataFrame({"time": ["2020-01-02 10:00", "2020-01-01 10:00"], "q": [0.6, 0.5]})
    ep = as_event_prices(x)
    assert ep.time_kind == "instant"
    assert ep.tz == "UTC"
    assert ep.q.tolist() == [0.5, 0.6]
    y = pd.DataFrame({"time": pd.Series(["2020-01-01", "2020-01-02"], dtype="string"), "q": [0.5, 0.6]})
    assert as_event_prices(y).time.iloc[1] == pd.Timestamp("2020-01-02", tz="UTC")


def test_all_missing_string_times():
    x = pd.DataFrame({"time": pd.Series([None, None], dtype="string"), "q": [0.5, 0.6]})
    ep, warns = record_warnings(as_event_prices, x)
    assert warns == ["2 rows with missing timestamp removed."]
    assert len(ep) == 0
    assert ep.time_kind == "instant"


def test_event_date():
    x = dated([d(2020, 1, 1), d(2020, 1, 2)], [0.4, 0.5])
    assert as_event_prices(x, event_date=d(2020, 2, 1)).event_date == d(2020, 2, 1)
    ts = as_event_prices(x, event_date=dt.datetime(2020, 2, 1, 15)).event_date
    assert ts == pd.Timestamp("2020-02-01 15:00", tz="UTC")
    with pytest.raises(ValueError, match="`event_date` must be"):
        as_event_prices(x, event_date="2020-02-01")


def test_repr_matches_r():
    x = dated([d(2020, 1, 3), d(2020, 1, 1), d(2020, 1, 2)], [0.3, 0.1, 0.2])
    lines = repr(as_event_prices(x, market_id="M", event_date=d(2020, 2, 1))).splitlines()
    assert lines[:3] == [
        "-- Event prices: M",
        "3 observations, 2020-01-01 to 2020-01-03",
        "Scheduled event: 2020-02-01",
    ]
    t = pd.Series([pd.Timestamp("2020-01-01", tz=NY), pd.Timestamp("2020-01-02 00:00:30", tz=NY)])
    ep = as_event_prices(
        pd.DataFrame({"time": t, "q": [0.5, np.nan]}),
        event_date=pd.Timestamp("2020-01-02 01:00", tz=NY),
    )
    assert repr(ep).splitlines()[:4] == [
        "-- Event prices",
        "2 observations, 2020-01-01 to 2020-01-02 00:00:30",
        "Scheduled event: 2020-01-02 01:00:00",
        "1 flagged observations (NA or outside clipping bounds)",
    ]


def test_summary_small_case():
    x = dated([d(2020, 1, i) for i in range(1, 6)], [0.1, np.nan, 0.2, 0.005, 0.3])
    s = as_event_prices(x, market_id="M").summary()
    assert list(s.columns) == [
        "market_id", "n", "start", "end", "q_start", "q_end",
        "q_min", "q_max", "n_na", "n_clip", "A_full",
    ]
    row = s.iloc[0]
    assert row["market_id"] == "M"
    assert row["n"] == 5
    assert row["start"] == pd.Timestamp("2020-01-01")
    assert row["end"] == pd.Timestamp("2020-01-05")
    assert (row["q_start"], row["q_end"], row["q_min"], row["q_max"]) == (0.1, 0.3, 0.005, 0.3)
    assert (row["n_na"], row["n_clip"]) == (1, 1)
    L = np.log(np.array([0.1, 0.2, 0.01, 0.3]) / (1 - np.array([0.1, 0.2, 0.01, 0.3])))
    assert row["A_full"] == pytest.approx(np.sum(np.diff(L) ** 2), rel=1e-14)


def test_summary_needs_two_values_for_a_full():
    s = as_event_prices(dated([d(2020, 1, 1), d(2020, 1, 2)], [0.5, np.nan])).summary()
    assert np.isnan(s.loc[0, "A_full"])
    assert s.loc[0, "market_id"] is None


def test_eventprices_is_frozen():
    ep = as_event_prices(dated([d(2020, 1, 1), d(2020, 1, 2)], [0.4, 0.5]))
    assert isinstance(ep, EventPrices)
    assert len(ep) == 2
    with pytest.raises(AttributeError):
        ep.market_id = "x"


def test_rejects_non_frames():
    with pytest.raises(ValueError, match="must be a DataFrame"):
        as_event_prices({"time": [1], "q": [0.5]})


def test_r_port_shipped_dataset(ep_brexit):
    assert list(ep_brexit.data.columns) == ["time", "q_raw", "q", "flag_na", "flag_clip"]
    assert len(ep_brexit) == 119
    assert ep_brexit.time_kind == "date"
    assert ep_brexit.market_id == "Brexit: Leave"
    assert ep_brexit.event_date == d(2016, 6, 23)
    assert ep_brexit.clip == (0.01, 0.99)
    q = ep_brexit.q[ep_brexit.time == pd.Timestamp("2016-05-24")]
    assert q.item() == pytest.approx(0.202, abs=1e-6)


def test_r_port_column_guessing():
    x = pd.DataFrame({"date": pd.Series([d(2020, 1, i) for i in (1, 2, 3)], dtype=object),
                      "price": [0.4, 0.5, 0.6]})
    assert as_event_prices(x).q.tolist() == [0.4, 0.5, 0.6]
    x2 = pd.DataFrame({"when": x["date"], "prob": [0.4, 0.5, 0.6]})
    assert as_event_prices(x2, time="when", price="prob").q.tolist() == [0.4, 0.5, 0.6]
    with pytest.raises(ValueError, match="Cannot guess"):
        as_event_prices(x2)
    with pytest.raises(ValueError, match="not found"):
        as_event_prices(x, time="nope")


def test_r_port_scale_discount_mid():
    x = pd.DataFrame({"date": pd.Series([d(2020, 1, 1), d(2020, 1, 2)], dtype=object),
                      "price": [40.0, 60.0]})
    ep = as_event_prices(x, scale=100)
    np.testing.assert_allclose(ep.q, [0.4, 0.6])
    np.testing.assert_allclose(ep.data["q_raw"], [0.4, 0.6])
    np.testing.assert_allclose(as_event_prices(x, scale=100, discount=0.99).q, np.array([0.4, 0.6]) / 0.99)
    x2 = pd.DataFrame({"date": x["date"], "b": [0.39, 0.59], "a": [0.41, 0.61]})
    np.testing.assert_allclose(as_event_prices(x2, bid="b", ask="a").q, [0.4, 0.6])


def test_r_port_flag_dont_drop():
    x = pd.DataFrame({"date": pd.Series([d(2020, 1, i) for i in range(1, 5)], dtype=object),
                      "q": [0.005, 0.5, np.nan, 0.995]})
    ep = as_event_prices(x)
    assert len(ep) == 4
    assert ep.data["flag_na"].tolist() == [False, False, True, False]
    assert ep.data["flag_clip"].tolist() == [True, False, False, True]
    assert as_event_prices(x, clip=(0.001, 0.999)).data["flag_clip"].sum() == 0


def test_r_port_sort_and_dedupe():
    x = pd.DataFrame({"date": pd.Series([d(2020, 1, 3), d(2020, 1, 1), d(2020, 1, 2)], dtype=object),
                      "q": [0.6, 0.4, 0.5]})
    assert as_event_prices(x).q.tolist() == [0.4, 0.5, 0.6]
    x2 = pd.DataFrame({"date": pd.Series([d(2020, 1, 1), d(2020, 1, 1), d(2020, 1, 2)], dtype=object),
                       "q": [0.4, 0.45, 0.5]})
    ep2, warns = record_warnings(as_event_prices, x2)
    assert any("duplicated timestamp" in w for w in warns)
    assert len(ep2) == 2
    assert ep2.q.tolist() == [0.4, 0.5]


def test_r_port_character_dates_and_bad_time():
    x = pd.DataFrame({"date": ["2020-01-01", "2020-01-02"], "q": [0.4, 0.5]})
    assert len(as_event_prices(x)) == 2
    with pytest.raises(ValueError, match="must be"):
        as_event_prices(pd.DataFrame({"date": [1.0, 2.0], "q": [0.4, 0.5]}))


def test_r_port_missing_timestamps():
    x = pd.DataFrame({"date": pd.Series([d(2020, 1, 1), None, d(2020, 1, 3)], dtype=object),
                      "q": [0.4, 0.5, 0.6]})
    ep, warns = record_warnings(as_event_prices, x)
    assert any("missing timestamp" in w for w in warns)
    assert ep.q.tolist() == [0.4, 0.6]


def test_r_port_idempotent_and_print(ep_brexit):
    assert as_event_prices(ep_brexit) is ep_brexit
    assert "Event prices" in repr(ep_brexit)
    s = ep_brexit.summary()
    assert len(s) == 1
