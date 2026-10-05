from typing import Any

import numpy as np
import pandas as pd

from ._utils import ec_warn

METHODS = ("discount", "overround")


def check_method(method: str) -> None:
    if method not in METHODS:
        raise ValueError(f'`method` must be one of "discount" or "overround", not "{method}".')


def as_numeric(v: Any, name: str) -> np.ndarray:
    if (
        isinstance(v, (pd.Series, pd.Index))
        and pd.api.types.is_numeric_dtype(v.dtype)
        and not pd.api.types.is_bool_dtype(v.dtype)
    ):
        return v.to_numpy(dtype="float64", na_value=np.nan)
    a = np.asarray(v)
    if a.dtype.kind not in "iuf":
        raise ValueError(f"`{name}` must be numeric.")
    return a.astype("float64")


def q_from_price(
    price: Any,
    discount: Any = 1,
    book: Any = None,
    method: str = "discount",
) -> Any:
    check_method(method)
    p = as_numeric(price, "price")
    with np.errstate(divide="ignore", invalid="ignore"):
        if method == "discount":
            d = as_numeric(discount, "discount")
            if np.any(~np.isnan(d) & ((d <= 0) | (d > 1))):
                raise ValueError("`discount` must lie in (0, 1].")
            q = p / d
        else:
            if book is None:
                raise ValueError(
                    '`book` (sum of all outcome prices) is required for `method = "overround"`.'
                )
            q = p / as_numeric(book, "book")
        n_out = int(np.sum(~np.isnan(q) & ((q < 0) | (q > 1))))
    if n_out == 1:
        ec_warn("1 normalized probability falls outside [0, 1]; check `discount`/`book`.")
    elif n_out > 1:
        ec_warn(f"{n_out} normalized probabilities fall outside [0, 1]; check `discount`/`book`.")
    return float(q) if q.ndim == 0 else q
