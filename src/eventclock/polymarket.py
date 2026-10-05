from typing import Any

import pandas as pd

from ._utils import is_instant
from .event_prices import EventPrices, as_event_prices


def pm_stitch(chunks: list[pd.DataFrame | None]) -> pd.DataFrame:
    parts = [c for c in chunks if c is not None]
    if not parts:
        return pd.DataFrame()
    raw = pd.concat(parts, ignore_index=True)
    if len(raw) == 0:
        return raw
    raw = raw[~raw["t"].duplicated()]
    return raw.sort_values("t", kind="stable").reset_index(drop=True)


def _empty_daily(x: EventPrices, event_date: Any) -> EventPrices:
    data = pd.DataFrame(
        {
            "time": pd.Series(dtype="datetime64[ns]"),
            "q_raw": pd.Series(dtype="float64"),
            "q": pd.Series(dtype="float64"),
            "flag_na": pd.Series(dtype="bool"),
            "flag_clip": pd.Series(dtype="bool"),
        }
    )
    return EventPrices(data, x.market_id, event_date, x.clip, "date")


def pm_daily(x: Any, tz: str = "America/New_York", snapshot_hour: float = 16) -> EventPrices:
    x = as_event_prices(x)
    if x.time_kind != "instant":
        raise ValueError("`pm_daily()` expects intraday <POSIXct> timestamps.")

    local = x.time.dt.tz_convert(tz)
    d = pd.DataFrame(
        {
            "date": local.dt.date,
            "hr": local.dt.hour + local.dt.minute / 60,
            "q": x.q.to_numpy(dtype="float64"),
        }
    )
    d = d[d["hr"] <= snapshot_hour]
    last = d.groupby("date", sort=True)["hr"].idxmax()
    daily = d.loc[last.to_numpy(), ["date", "q"]].rename(columns={"date": "time"})

    ed = x.event_date
    if ed is not None and is_instant(ed):
        ed = pd.Timestamp(ed).date()

    if len(daily) == 0:
        return _empty_daily(x, ed)
    return as_event_prices(
        daily.reset_index(drop=True),
        time="time",
        price="q",
        clip=x.clip,
        market_id=x.market_id,
        event_date=ed,
    )
