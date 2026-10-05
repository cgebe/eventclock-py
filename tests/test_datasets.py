import datetime as dt

import numpy as np
import pytest

from eventclock import load_dataset
from eventclock.datasets import dataset_names

SHAPES = {
    "brexit2016": (119, ["date", "q_remain", "q_leave"]),
    "djt2024": (126, ["date", "close", "adjusted", "volume"]),
    "fomc_meetings": (56, ["decision_date", "year", "sep"]),
    "pipr_bins": (
        79084,
        ["time", "q_m50", "q_m25", "q_0", "q_p25", "q_raw_sum", "staleness_sec", "quality_flags"],
    ),
    "polymarket2024": (3791, ["time", "q"]),
    "us2016": (
        237,
        ["date", "trump", "clinton", "sanders", "cruz", "kasich", "biden", "mcmullin", "volume"],
    ),
}

DATE_COLUMNS = [
    ("brexit2016", "date"),
    ("us2016", "date"),
    ("djt2024", "date"),
    ("fomc_meetings", "decision_date"),
]


def test_dataset_names():
    assert dataset_names() == sorted(SHAPES)


def test_fomc_calendar_is_consistent():
    df = load_dataset("fomc_meetings")
    assert len(df) == 56
    assert df["year"].value_counts().unique().tolist() == [8]
    assert df["sep"].sum() == 28
    dates = set(df["decision_date"])
    assert dt.date(2024, 9, 18) in dates
    assert dt.date(2024, 11, 7) in dates
    assert dt.date(2026, 9, 16) in dates
    sep24 = df.loc[(df["year"] == 2024) & df["sep"], "decision_date"]
    assert [d.strftime("%m") for d in sep24] == ["03", "06", "09", "12"]

@pytest.mark.parametrize("name", sorted(SHAPES))
def test_shape_and_columns(name):
    n, cols = SHAPES[name]
    df = load_dataset(name)
    assert len(df) == n
    assert list(df.columns) == cols


@pytest.mark.parametrize(("name", "col"), DATE_COLUMNS)
def test_date_columns_are_calendar_dates(name, col):
    s = load_dataset(name)[col]
    assert s.dtype == object
    assert all(type(v) is dt.date for v in s)


def test_polymarket_time_is_utc_instant():
    s = load_dataset("polymarket2024")["time"]
    assert str(s.dtype) == "datetime64[ns, UTC]"
    assert s.is_monotonic_increasing
    assert s.iloc[0] == dt.datetime(2024, 6, 1, 0, 0, 3, tzinfo=dt.UTC)


def test_fomc_column_types():
    df = load_dataset("fomc_meetings")
    assert df["year"].dtype == np.int32
    assert df["sep"].dtype == bool


def test_fomc_calendar_is_consistent():
    df = load_dataset("fomc_meetings")
    assert len(df) == 56
    assert df["year"].value_counts().unique().tolist() == [8]
    assert df["sep"].sum() == 28
    dates = set(df["decision_date"])
    assert dt.date(2024, 9, 18) in dates
    assert dt.date(2024, 11, 7) in dates
    assert dt.date(2026, 9, 16) in dates
    sep24 = df.loc[(df["year"] == 2024) & df["sep"], "decision_date"]
    assert [d.strftime("%m") for d in sep24] == ["03", "06", "09", "12"]


@pytest.mark.parametrize("name", ["brexit2016", "us2016", "djt2024", "polymarket2024"])
def test_value_columns_are_float64(name):
    df = load_dataset(name)
    for col in SHAPES[name][1][1:]:
        assert df[col].dtype == np.float64, col


def test_unknown_dataset():
    with pytest.raises(ValueError, match="not found"):
        load_dataset("brexit2017")


def test_each_call_returns_a_new_frame():
    df = load_dataset("brexit2016")
    df.loc[0, "q_leave"] = -1.0
    assert load_dataset("brexit2016").loc[0, "q_leave"] != -1.0


@pytest.mark.parametrize(
    ("fixture", "dataset", "time_col", "price_col"),
    [
        ("ep_brexit", "brexit2016", "date", "q_leave"),
        ("ep_us", "us2016", "date", "trump"),
        ("ep_pm2024", "polymarket2024", "time", "q"),
    ],
)
def test_parquet_matches_r_fixture(load_fixture, fixture, dataset, time_col, price_col):
    fx = load_fixture(fixture)
    assert fx["input"] == {"kind": "dataset", "name": dataset}
    ep = fx["output"]["event_prices"]
    df = load_dataset(dataset)
    assert ep["time"].tolist() == df[time_col].tolist()
    np.testing.assert_array_equal(ep["q_raw"].to_numpy(), df[price_col].to_numpy())