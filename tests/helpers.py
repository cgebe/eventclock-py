import datetime as dt
import json
import logging
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eventclock import EventClockWarning, as_event_prices, load_dataset

FIXTURE_DIR = Path(__file__).parent / "fixtures"
TABLE_KEYS = ("table", "event_prices", "summary")


def _parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def _map(v: Any, fn: Callable[[Any], Any]) -> Any:
    if isinstance(v, dict):
        return {k: None if x is None else fn(x) for k, x in v.items()}
    if isinstance(v, list):
        return [None if x is None else fn(x) for x in v]
    return None if v is None else fn(v)


def decode_value(v: Any) -> Any:
    if isinstance(v, dict):
        if set(v) == {"date"}:
            return _map(v["date"], _parse_date)
        if "instant" in v and set(v) <= {"instant", "tz"}:
            tz = v.get("tz") or "UTC"
            return _map(v["instant"], lambda s: pd.Timestamp(s).tz_convert(tz))
        return {k: decode_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [decode_value(x) for x in v]
    return v


def decode_column(values: list, kind: str, tz: str | None = None) -> pd.Series:
    has_null = any(v is None for v in values)
    match kind:
        case "date":
            return pd.Series([None if v is None else _parse_date(v) for v in values], dtype=object)
        case "instant":
            s = pd.to_datetime(pd.Series(values, dtype=object), utc=True, format="ISO8601")
            return s.dt.as_unit("ns").dt.tz_convert(tz or "UTC")
        case "double":
            return pd.Series([np.nan if v is None else v for v in values], dtype="float64")
        case "integer":
            if has_null:
                return pd.Series([np.nan if v is None else v for v in values], dtype="float64")
            return pd.Series(values, dtype="int64")
        case "logical":
            return pd.Series(values, dtype=object if has_null else "bool")
        case "character":
            return pd.Series(values, dtype=object)
    raise ValueError(f"Unknown column type {kind!r}.")


def decode_table(enc: dict) -> pd.DataFrame:
    cols = {
        name: decode_column(values, enc["types"][name], enc["tz"].get(name))
        for name, values in enc["columns"].items()
    }
    df = pd.DataFrame(cols, index=pd.RangeIndex(enc["nrow"]))
    if len(df) != enc["nrow"]:
        raise ValueError(f"Decoded {len(df)} rows, fixture says {enc['nrow']}.")
    df.attrs = {k: decode_value(v) for k, v in enc["attrs"].items()}
    df.attrs["r_class"] = enc["class"]
    return df


def _as_kwargs(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def read_fixture(name: str) -> dict:
    raw = json.loads((FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))
    fx = dict(raw)
    fx["args"] = _as_kwargs(decode_value(raw["args"]))
    inp = dict(raw["input"])
    if inp["kind"] == "rows":
        inp["data"] = decode_table(inp["data"])
        inp["as_event_prices_args"] = _as_kwargs(decode_value(inp["as_event_prices_args"]))
    fx["input"] = inp
    fx["output"] = {
        k: decode_table(v) if k in TABLE_KEYS else v for k, v in raw["output"].items()
    }
    return fx


def fixture_names() -> list[str]:
    return sorted(p.stem for p in FIXTURE_DIR.glob("*.json"))


def build_input(fx: dict) -> Any:
    inp = fx["input"]
    match inp["kind"]:
        case "dataset":
            return load_dataset(inp["name"])
        case "event_prices":
            src = read_fixture(inp["name"])
            return as_event_prices(load_dataset(src["input"]["name"]), **src["args"])
        case "rows":
            return as_event_prices(inp["data"], **inp["as_event_prices_args"])
    raise ValueError(f"Unknown input kind {inp['kind']!r}.")


def record_warnings(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, [str(w.message) for w in caught if issubclass(w.category, EventClockWarning)]


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def record_all(
    fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> tuple[Any, list[str], list[str]]:
    logger = logging.getLogger("eventclock")
    handler = _ListHandler()
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        result, warns = record_warnings(fn, *args, **kwargs)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    return result, warns, handler.messages


def r_messages(fx: dict) -> list[str]:
    return [m.removeprefix("i ").removeprefix("\u2139 ") for m in fx["messages"]]


R_ARG_NAMES = {"from": "from_"}


def py_args(args: dict[str, Any]) -> dict[str, Any]:
    return {R_ARG_NAMES.get(k, k): v for k, v in args.items()}


def _is_time(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return True
    present = s.dropna()
    return s.dtype == object and len(present) > 0 and all(isinstance(v, dt.date) for v in present)


def _as_time(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return s.dt.as_unit("ns")
    return pd.to_datetime(s).dt.as_unit("ns")


def assert_frame_parity(
    actual: pd.DataFrame, expected: pd.DataFrame, rtol: float = 1e-12, atol: float = 1e-14
) -> None:
    assert list(actual.columns) == list(expected.columns)
    assert len(actual) == len(expected)
    for col in expected.columns:
        a = actual[col].reset_index(drop=True)
        e = expected[col].reset_index(drop=True)
        if pd.api.types.is_float_dtype(a.dtype) or pd.api.types.is_float_dtype(e.dtype):
            a_ = a.to_numpy(dtype="float64", na_value=np.nan)
            e_ = e.to_numpy(dtype="float64", na_value=np.nan)
            np.testing.assert_array_equal(np.isnan(a_), np.isnan(e_), err_msg=f"NaN positions in {col}")
            ok = ~np.isnan(e_)
            np.testing.assert_allclose(a_[ok], e_[ok], rtol=rtol, atol=atol, err_msg=col)
        elif _is_time(a) or _is_time(e):
            a_, e_ = _as_time(a), _as_time(e)
            assert str(getattr(a_.dt, "tz", None)) == str(getattr(e_.dt, "tz", None)), col
            assert a_.isna().tolist() == e_.isna().tolist(), col
            assert a_.dropna().tolist() == e_.dropna().tolist(), col
        else:
            a_ = [None if pd.isna(v) else v for v in a]
            e_ = [None if pd.isna(v) else v for v in e]
            assert a_ == e_, col
