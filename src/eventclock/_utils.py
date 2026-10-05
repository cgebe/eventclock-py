import datetime as dt
import logging
import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.special import expit, logit

logger = logging.getLogger("eventclock")

MAD_CONSTANT = 1.4826
END_OF_DAY = pd.Timedelta(seconds=86400 - 1)

TimeKind = Literal["date", "instant"]
Side = Literal["start", "end"]


class EventClockWarning(UserWarning):
    pass


def as_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def ec_warn(msg: str) -> None:
    warnings.warn(msg, EventClockWarning, stacklevel=3)


def ec_logit(q: Any) -> Any:
    return logit(q)


def ec_ilogit(l: Any) -> Any:
    return expit(l)


def _check_clip(clip: Any) -> tuple[float, float]:
    try:
        lo, hi = (float(v) for v in clip)
    except (TypeError, ValueError):
        raise ValueError("`clip` must be a pair (lo, hi).") from None
    if not (lo < hi and lo > 0 and hi < 1):
        raise ValueError(f"`clip` must satisfy 0 < lo < hi < 1, got ({lo}, {hi}).")
    return lo, hi


def clip_q(q: Any, clip: Any) -> Any:
    lo, hi = _check_clip(clip)
    out = np.clip(np.asarray(q, dtype=float), lo, hi)
    return float(out) if out.ndim == 0 else out


def check_prob(q: Any, name: str = "q") -> Any:
    a = np.asarray(q, dtype=float)
    if np.any(~np.isnan(a) & ((a <= 0) | (a >= 1))):
        raise ValueError(f"`{name}` must lie strictly inside (0, 1).")
    return q


def check_nonneg(A: Any, name: str = "A") -> Any:
    a = np.asarray(A, dtype=float)
    if np.any(~np.isnan(a) & (a < 0)):
        raise ValueError(f"`{name}` must be non-negative.")
    return A


def r_mad(x: Any) -> float:
    a = np.asarray(x, dtype=float).ravel()
    if a.size == 0 or np.isnan(a).any():
        return np.nan
    return MAD_CONSTANT * float(np.median(np.abs(a - np.median(a))))


def is_date(x: Any) -> bool:
    return isinstance(x, dt.date) and not isinstance(x, dt.datetime)


def is_instant(x: Any) -> bool:
    return isinstance(x, (dt.datetime, np.datetime64))


def as_instant(x: Any) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.as_unit("ns")


def align_bound(
    bound: Any,
    time_kind: TimeKind,
    tz: str | dt.tzinfo | None = None,
    side: Side = "start",
) -> Any:
    if side not in ("start", "end"):
        raise ValueError(f'`side` must be "start" or "end", got {side!r}.')
    if time_kind not in ("date", "instant"):
        raise ValueError(f'`time_kind` must be "date" or "instant", got {time_kind!r}.')
    if bound is None:
        return None
    if is_instant(bound):
        ts = as_instant(bound)
        return ts.date() if time_kind == "date" else ts
    if is_date(bound):
        if time_kind == "date":
            return bound
        out = pd.Timestamp(bound).tz_localize(
            tz or "UTC", ambiguous=True, nonexistent="shift_forward"
        ).as_unit("ns")
        return out + END_OF_DAY if side == "end" else out
    raise ValueError(f"A window bound must be a date or a datetime, got {type(bound).__name__}.")


def r_format_time(values: Any, time_kind: TimeKind) -> list[str | None]:
    s = pd.to_datetime(pd.Series(values))
    fmt = "%Y-%m-%d"
    if time_kind == "instant":
        ok = s.dropna()
        midnight = (
            (ok.dt.hour == 0)
            & (ok.dt.minute == 0)
            & (ok.dt.second == 0)
            & (ok.dt.microsecond == 0)
            & (ok.dt.nanosecond == 0)
        ).all()
        if not midnight:
            fmt = "%Y-%m-%d %H:%M:%S"
    return [None if pd.isna(v) else v.strftime(fmt) for v in s]


def r_num_str(x: float) -> str:
    if np.isnan(x):
        return "NaN"
    if np.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    target = float(f"{x:.15g}")
    n = next(k for k in range(1, 16) if float(f"{x:.{k}g}") == target)
    sci = f"{target:.{n - 1}e}"
    fixed = np.format_float_positional(target, trim="-")
    return sci if len(fixed) > len(sci) else fixed
