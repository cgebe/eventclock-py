from typing import Any

import numpy as np
import pandas as pd

from ._utils import clip_q, ec_logit, r_format_time, r_mad
from .clock import gap_stats
from .event_prices import as_event_prices

VALIDATION_COLUMNS = [
    "market_id", "n_obs", "start", "end", "spacing_days", "n_na", "share_clip",
    "share_zero_incr", "max_stale_run", "tick", "n_gaps", "max_gap_days",
    "max_abs_dL", "outlier_ratio",
]


def _longest_stale_run(dq: np.ndarray) -> int:
    longest = run = 0
    for moved in dq != 0:
        run = 0 if moved else run + 1
        longest = max(longest, run)
    return longest


def ec_validate(x: Any) -> pd.DataFrame:
    x = as_event_prices(x)
    ok = ~x.data["flag_na"]
    q = x.data.loc[ok, "q"].to_numpy(dtype="float64")
    tt = x.data.loc[ok, "time"].reset_index(drop=True)
    if len(q) < 2:
        raise ValueError("Need at least 2 non-missing observations.")

    dq = np.diff(q)
    dL = np.diff(ec_logit(clip_q(q, x.clip)))
    s = r_mad(dL)
    nz = np.abs(dq)[np.abs(dq) > 0]
    n_gaps, max_gap_days = gap_stats(tt)
    max_abs_dL = float(np.max(np.abs(dL)))
    spacing = (tt.diff().iloc[1:] / pd.Timedelta(days=1)).to_numpy(dtype="float64")

    row = {
        "market_id": x.market_id,
        "n_obs": len(x.data),
        "start": x.time.min(),
        "end": x.time.max(),
        "spacing_days": float(np.median(spacing)),
        "n_na": int(x.data["flag_na"].sum()),
        "share_clip": float(x.data.loc[ok, "flag_clip"].mean()),
        "share_zero_incr": float(np.mean(dq == 0)),
        "max_stale_run": _longest_stale_run(dq),
        "tick": float(nz.min()) if len(nz) else np.nan,
        "n_gaps": n_gaps,
        "max_gap_days": max_gap_days,
        "max_abs_dL": max_abs_dL,
        "outlier_ratio": max_abs_dL / s if np.isfinite(s) and s > 0 else np.nan,
    }
    return pd.DataFrame([row], columns=VALIDATION_COLUMNS)


def _r_sprintf(spec: str, v: Any) -> str:
    if pd.isna(v):
        return "NA"
    if np.isinf(v):
        return "Inf" if v > 0 else "-Inf"
    return spec % v


def format_validation(v: pd.DataFrame) -> str:
    r = v.iloc[0]
    start = r_format_time([r["start"]], "instant")[0]
    end = r_format_time([r["end"]], "instant")[0]
    head = "-- Event-price data-quality report"
    if not pd.isna(r["market_id"]):
        head += f": {r['market_id']}"
    lines = [
        head,
        f"Coverage:  {_r_sprintf('%d', r['n_obs'])} obs, {start} to {end}, "
        f"median spacing {_r_sprintf('%.2g', r['spacing_days'])} days, "
        f"{_r_sprintf('%d', r['n_na'])} NA",
        f"Clipping:  {_r_sprintf('%.1f', 100 * r['share_clip'])}% of observations "
        "outside the clipping bounds",
        f"Staleness: {_r_sprintf('%.1f', 100 * r['share_zero_incr'])}% zero increments, "
        f"longest stale run {_r_sprintf('%d', r['max_stale_run'])} obs",
        f"Tick:      smallest non-zero move {_r_sprintf('%.4g', r['tick'])}",
        f"Gaps:      {_r_sprintf('%d', r['n_gaps'])} gap increment(s), "
        f"largest {_r_sprintf('%.3g', r['max_gap_days'])} days",
        f"Outliers:  max |dL| = {_r_sprintf('%.3g', r['max_abs_dL'])} "
        f"({_r_sprintf('%.1f', r['outlier_ratio'])} robust SDs)",
    ]
    return "\n".join(lines)
