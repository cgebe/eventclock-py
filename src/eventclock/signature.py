from typing import Any

import numpy as np
import pandas as pd

from ._utils import clip_q, ec_logit
from .clock import native_time, window_rows
from .event_prices import as_event_prices


def ec_signature(
    x: Any,
    max_every: int = 10,
    from_: Any = None,
    to: Any = None,
    clip: tuple[float, float] | None = None,
) -> pd.DataFrame:
    x = as_event_prices(x)
    clip = clip if clip is not None else x.clip
    if from_ is None:
        from_ = native_time(x.time.min(), x.time_kind)
    if to is None:
        to = native_time(x.time.max(), x.time_kind)
    if not max_every >= 1:
        raise ValueError("`max_every` must be at least 1.")
    max_every = int(max_every)

    d = window_rows(x, from_, to)
    n = len(d)
    if n < max_every + 1:
        raise ValueError("Need more than `max_every` observations in the window.")
    L = ec_logit(clip_q(d["q"].to_numpy(dtype="float64"), clip))
    spacing = (d["time"].diff().iloc[1:] / pd.Timedelta(days=1)).to_numpy(dtype="float64")
    base_dt = float(np.median(spacing))

    rows = []
    for k in range(1, max_every + 1):
        grids = [L[o::k] for o in range(k)]
        rv = np.array([np.sum(np.diff(g) ** 2) if len(g) >= 2 else np.nan for g in grids])
        n_incr = [max(0, len(g) - 1) for g in grids]
        rows.append(
            {
                "sample_every": k,
                "spacing_days": k * base_dt,
                "n_incr": float(np.mean(n_incr)),
                "A": float(np.nanmean(rv)),
                "A_min": float(np.nanmin(rv)),
                "A_max": float(np.nanmax(rv)),
            }
        )
    out = pd.DataFrame(
        rows, columns=["sample_every", "spacing_days", "n_incr", "A", "A_min", "A_max"]
    )
    out["sample_every"] = out["sample_every"].astype("int64")
    out.attrs = {"market_id": x.market_id}
    return out
