import datetime as dt
from collections.abc import Callable

import pandas as pd
import pytest

from eventclock import EventPrices, as_event_prices, load_dataset
from helpers import read_fixture


@pytest.fixture
def load_fixture() -> Callable[[str], dict]:
    return read_fixture


@pytest.fixture
def ep_brexit() -> EventPrices:
    return as_event_prices(
        load_dataset("brexit2016"),
        time="date",
        price="q_leave",
        market_id="Brexit: Leave",
        event_date=dt.date(2016, 6, 23),
    )


@pytest.fixture
def ep_us() -> EventPrices:
    return as_event_prices(
        load_dataset("us2016"),
        time="date",
        price="trump",
        market_id="US 2016: Trump",
        event_date=dt.date(2016, 11, 8),
    )


@pytest.fixture
def brexit_horizons() -> dict[str, dt.date]:
    return {"1W": dt.date(2016, 5, 31), "2W": dt.date(2016, 6, 7), "1M": dt.date(2016, 6, 23)}


@pytest.fixture
def us_horizons() -> dict[str, dt.date]:
    return {"1W": dt.date(2016, 10, 17), "2W": dt.date(2016, 10, 24), "1M": dt.date(2016, 11, 8)}


@pytest.fixture
def clock_value() -> Callable[[pd.DataFrame, str, str], float]:
    def _value(res: pd.DataFrame, horizon: str, method: str) -> float:
        sel = res.loc[(res["horizon"] == horizon) & (res["method"] == method), "A"]
        return float(sel.to_numpy().item())

    return _value
