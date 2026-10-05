import datetime as dt

import numpy as np
import pandas as pd
import pytest

from eventclock import as_event_prices, ec_validate, format_validation, load_dataset
from eventclock.validate import VALIDATION_COLUMNS, _longest_stale_run


def dated(offsets: list[int], q: list) -> pd.DataFrame:
    times = [dt.date(2020, 1, 1) + dt.timedelta(days=o) for o in offsets]
    return pd.DataFrame({"time": pd.Series(times, dtype=object), "q": q})


CONSTRUCTED = dated(
    [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12],
    [0.40, 0.40, 0.40, 0.41, np.nan, 0.415, 0.42, 0.42, 0.005, 0.42, 0.43],
)


def test_clean_series_is_reported_clean(ep_brexit):
    v = ec_validate(ep_brexit)
    assert list(v.columns) == VALIDATION_COLUMNS
    assert v["n_obs"].item() == 119
    assert v["n_na"].item() == 0
    assert v["n_gaps"].item() == 0
    assert v["spacing_days"].item() == 1
    assert v["share_clip"].item() == 0
    assert "data-quality report" in format_validation(v)


def test_flags_staleness_gaps_clipping_and_outliers():
    v = ec_validate(as_event_prices(CONSTRUCTED))
    assert v["n_na"].item() == 1
    assert v["max_stale_run"].item() >= 2
    assert v["n_gaps"].item() > 0
    assert v["share_clip"].item() > 0
    assert v["outlier_ratio"].item() > 3
    assert v["tick"].item() == pytest.approx(0.005, abs=1e-12)


def test_constructed_values():
    v = ec_validate(as_event_prices(CONSTRUCTED)).iloc[0]
    assert v["n_obs"] == 11
    assert v["start"] == pd.Timestamp("2020-01-01")
    assert v["end"] == pd.Timestamp("2020-01-13")
    assert v["share_clip"] == pytest.approx(0.1)
    assert v["share_zero_incr"] == pytest.approx(1 / 3)
    assert v["max_stale_run"] == 2
    assert v["n_gaps"] == 2
    assert v["max_gap_days"] == 3


def test_needs_at_least_two_clean_observations():
    with pytest.raises(ValueError, match="at least 2"):
        ec_validate(as_event_prices(dated([0, 1], [0.5, np.nan])))


def test_stale_run_counts_zero_increments_like_r_rle():
    assert _longest_stale_run(np.array([0.0, 0.0, 0.1, 0.0])) == 2
    assert _longest_stale_run(np.array([0.1, 0.2])) == 0
    assert _longest_stale_run(np.array([0.0])) == 1


def test_no_nonzero_move_and_degenerate_scale():
    v = ec_validate(as_event_prices(dated([0, 1, 2], [0.5, 0.5, 0.5])))
    assert np.isnan(v["tick"].item())
    assert np.isnan(v["outlier_ratio"].item())
    assert v["max_stale_run"].item() == 2
    text = format_validation(v).splitlines()
    assert text[4] == "Tick:      smallest non-zero move NA"
    assert text[6] == "Outliers:  max |dL| = 0 (NA robust SDs)"


def test_format_matches_r_print():
    v = ec_validate(as_event_prices(CONSTRUCTED))
    assert format_validation(v).splitlines() == [
        "-- Event-price data-quality report",
        "Coverage:  11 obs, 2020-01-01 to 2020-01-13, median spacing 1 days, 1 NA",
        "Clipping:  10.0% of observations outside the clipping bounds",
        "Staleness: 33.3% zero increments, longest stale run 2 obs",
        "Tick:      smallest non-zero move 0.005",
        "Gaps:      2 gap increment(s), largest 3 days",
        "Outliers:  max |dL| = 4.27 (140.2 robust SDs)",
    ]


def test_format_for_instants_and_market_id():
    v = ec_validate(as_event_prices(load_dataset("polymarket2024"), market_id="PM"))
    lines = format_validation(v).splitlines()
    assert lines[0] == "-- Event-price data-quality report: PM"
    assert lines[1].startswith("Coverage:  3791 obs, 2024-06-01 00:00:03 to 2024-11-05 23:00:02,")
