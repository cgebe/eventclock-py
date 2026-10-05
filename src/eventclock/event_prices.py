import datetime as dt
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._utils import (
    TimeKind,
    as_instant,
    clip_q,
    ec_logit,
    ec_warn,
    is_date,
    is_instant,
    r_format_time,
)
from .normalize import as_numeric, check_method, q_from_price
from .params import ec_default_params

DEFAULT_CLIP: tuple[float, float] = ec_default_params()["clip"]
TIME_CANDIDATES = ("time", "date", "timestamp", "datetime", "t")
PRICE_CANDIDATES = ("q", "price", "p", "q_t", "value")

_YMD = r" *(\d{1,4}+)- *(\d{1,2}+)- *(\d{1,2}+)"
_HM = r"\s* *(\d{1,2}+): *(\d{1,2}+)"
_TIME_FORMATS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("%Y-%m-%d %H:%M:%OS", re.compile(_YMD + _HM + r":\s*(\d+(?:\.\d*)?)")),
    ("%Y-%m-%d %H:%M", re.compile(_YMD + _HM)),
    ("%Y-%m-%d", re.compile(_YMD)),
    ("%d.%m.%Y", re.compile(r" *(\d{1,2}+)\. *(\d{1,2}+)\. *(\d{1,4}+)")),
)


@dataclass(frozen=True, eq=False, repr=False)
class EventPrices:
    data: pd.DataFrame
    market_id: str | None = None
    event_date: dt.date | pd.Timestamp | None = None
    clip: tuple[float, float] = DEFAULT_CLIP
    time_kind: TimeKind = "date"

    @property
    def time(self) -> pd.Series:
        return self.data["time"]

    @property
    def q(self) -> pd.Series:
        return self.data["q"]

    @property
    def tz(self) -> str | None:
        return None if self.time_kind == "date" else str(self.time.dt.tz)

    def __len__(self) -> int:
        return len(self.data)

    def summary(self) -> pd.DataFrame:
        d = self.data
        q_ok = d.loc[~d["flag_na"], "q"].to_numpy(dtype="float64")
        a_full = (
            float(np.sum(np.diff(ec_logit(clip_q(q_ok, self.clip))) ** 2))
            if len(q_ok) >= 2
            else np.nan
        )
        row = {
            "market_id": self.market_id,
            "n": len(d),
            "start": d["time"].min(),
            "end": d["time"].max(),
            "q_start": q_ok[0] if len(q_ok) else np.nan,
            "q_end": q_ok[-1] if len(q_ok) else np.nan,
            "q_min": q_ok.min() if len(q_ok) else np.inf,
            "q_max": q_ok.max() if len(q_ok) else -np.inf,
            "n_na": int(d["flag_na"].sum()),
            "n_clip": int(d["flag_clip"].sum()),
            "A_full": a_full,
        }
        return pd.DataFrame([row])

    def __repr__(self) -> str:
        head = "-- Event prices" + (f": {self.market_id}" if self.market_id is not None else "")
        lines = [head]
        if len(self) > 0:
            start = r_format_time([self.time.min()], self.time_kind)[0]
            end = r_format_time([self.time.max()], self.time_kind)[0]
            lines.append(f"{len(self)} observations, {start} to {end}")
        else:
            lines.append("0 observations")
        if self.event_date is not None:
            kind = "date" if is_date(self.event_date) else "instant"
            lines.append(f"Scheduled event: {r_format_time([self.event_date], kind)[0]}")
        n_flag = int((self.data["flag_na"] | self.data["flag_clip"]).sum())
        if n_flag > 0:
            lines.append(f"{n_flag} flagged observations (NA or outside clipping bounds)")
        lines.append(repr(self.data))
        return "\n".join(lines)


def _pick_col(x: pd.DataFrame, given: Any, candidates: tuple[str, ...], what: str) -> Any:
    cols = list(x.columns)
    if given is not None:
        if given not in cols:
            raise ValueError(f'Column "{given}" not found in `x`.')
        return given
    lower = [str(c).lower() for c in cols]
    for cand in candidates:
        if cand in lower:
            return cols[lower.index(cand)]
    tried = ", ".join(f'"{c}"' for c in candidates[:-1]) + f', and "{candidates[-1]}"'
    raise ValueError(f"Cannot guess the {what} column.\ni Pass `{what}` explicitly; tried {tried}.")


def _parse_one(s: str, pattern: re.Pattern[str], fmt: str) -> pd.Timestamp | None:
    m = pattern.match(s)
    if m is None:
        return None
    g = m.groups()
    if fmt == "%d.%m.%Y":
        year, month, day = int(g[2]), int(g[1]), int(g[0])
    else:
        year, month, day = int(g[0]), int(g[1]), int(g[2])
    hour = int(g[3]) if len(g) > 3 else 0
    minute = int(g[4]) if len(g) > 4 else 0
    second = float(g[5]) if len(g) > 5 else 0.0
    if hour > 24 or minute > 59 or second >= 61:
        return None
    try:
        day0 = dt.date(year, month, day)
    except ValueError:
        return None
    try:
        return pd.Timestamp(day0).tz_localize("UTC").as_unit("ns") + pd.Timedelta(
            hours=hour, minutes=minute, seconds=second
        )
    except (OverflowError, pd.errors.OutOfBoundsDatetime):
        raise ValueError(f"Time value {s!r} is outside the supported range (1677 to 2262).") from None


def _parse_time_strings(s: pd.Series, col: Any) -> pd.Series:
    missing = s.isna().to_numpy()
    present = [v for v, miss in zip(s, missing) if not miss]
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns, UTC]")
    if not present:
        return out
    for fmt, pattern in _TIME_FORMATS:
        parsed = [_parse_one(v, pattern, fmt) for v in present]
        if all(p is not None for p in parsed):
            out[~missing] = parsed
            return out
    raise ValueError(
        f'Could not parse the time column "{col}": '
        "character string is not in a standard unambiguous format."
    )


def _coerce_time(s: pd.Series, col: Any) -> tuple[pd.Series, TimeKind]:
    must_be = f'The time column "{col}" must be <Date> or <POSIXct>.'
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.as_unit("ns"), "instant"
    if pd.api.types.is_datetime64_dtype(s.dtype):
        return s.dt.tz_localize("UTC").dt.as_unit("ns"), "instant"
    if isinstance(s.dtype, pd.StringDtype):
        return _parse_time_strings(s, col), "instant"
    if s.dtype != object:
        raise ValueError(must_be)
    present = [v for v in s if not pd.isna(v)]
    if present and all(isinstance(v, str) for v in present):
        return _parse_time_strings(s, col), "instant"
    if present and all(is_date(v) for v in present):
        dates = pd.Series([None if pd.isna(v) else v for v in s], index=s.index, dtype=object)
        return pd.to_datetime(dates).dt.as_unit("ns"), "date"
    if present and all(is_instant(v) for v in present):
        tzs = {str(pd.Timestamp(v).tz or "UTC") for v in present}
        tz = tzs.pop() if len(tzs) == 1 else "UTC"
        return pd.to_datetime(s, utc=True).dt.tz_convert(tz).dt.as_unit("ns"), "instant"
    raise ValueError(must_be)


def _to_numeric(s: pd.Series) -> np.ndarray:
    num = pd.to_numeric(s, errors="coerce")
    out = pd.Series(num).to_numpy(dtype="float64", na_value=np.nan)
    if (np.isnan(out) & s.notna().to_numpy()).any():
        ec_warn("NAs introduced by coercion")
    return out


def _check_event_date(v: Any) -> dt.date | pd.Timestamp | None:
    if v is None:
        return None
    if is_instant(v):
        return as_instant(v)
    if is_date(v):
        return v
    raise ValueError("`event_date` must be a date or a datetime.")


def as_event_prices(
    x: Any,
    time: Any = None,
    price: Any = None,
    bid: Any = None,
    ask: Any = None,
    scale: float = 1,
    discount: Any = 1,
    book: Any = None,
    method: str = "discount",
    clip: tuple[float, float] = DEFAULT_CLIP,
    market_id: str | None = None,
    event_date: Any = None,
) -> EventPrices:
    if isinstance(x, EventPrices):
        return x
    if not isinstance(x, pd.DataFrame):
        raise ValueError("`x` must be a DataFrame or EventPrices.")
    check_method(method)

    time_col = _pick_col(x, time, TIME_CANDIDATES, "time")
    tt, time_kind = _coerce_time(x[time_col], time_col)

    if bid is not None and ask is not None:
        if bid not in x.columns or ask not in x.columns:
            raise ValueError(f'Columns "{bid}" / "{ask}" not found in `x`.')
        pp = (_to_numeric(x[bid]) + _to_numeric(x[ask])) / 2
    else:
        price_col = _pick_col(x, price, PRICE_CANDIDATES, "price")
        pp = _to_numeric(x[price_col])

    if (
        isinstance(scale, (bool, np.bool_))
        or not isinstance(scale, (int, float, np.integer, np.floating))
        or not scale > 0
    ):
        raise ValueError("`scale` must be a single positive number.")
    q_raw = pp / scale
    q = as_numeric(q_from_price(q_raw, discount=discount, book=book, method=method), "q")

    out = pd.DataFrame({"time": tt.reset_index(drop=True), "q_raw": q_raw, "q": q})

    na_time = out["time"].isna()
    n_na = int(na_time.sum())
    if n_na > 0:
        ec_warn(f"{n_na} row{'s' if n_na != 1 else ''} with missing timestamp removed.")
        out = out[~na_time]

    out = out.sort_values("time", kind="stable")
    dup = out["time"].duplicated()
    n_dup = int(dup.sum())
    if n_dup > 0:
        ec_warn(
            f"{n_dup} duplicated timestamp{'s' if n_dup != 1 else ''} removed (first occurrence kept)."
        )
        out = out[~dup]
    out = out.reset_index(drop=True)

    lo, hi = (float(v) for v in clip)
    out["flag_na"] = out["q"].isna()
    out["flag_clip"] = ~out["flag_na"] & ((out["q"] < lo) | (out["q"] > hi))

    return EventPrices(
        data=out,
        market_id=market_id,
        event_date=_check_event_date(event_date),
        clip=(lo, hi),
        time_kind=time_kind,
    )
