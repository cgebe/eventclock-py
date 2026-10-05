# eventclock-py

[![ci](https://github.com/cgebe/eventclock-py/actions/workflows/ci.yaml/badge.svg)](https://github.com/cgebe/eventclock-py/actions/workflows/ci.yaml)

Python port of the R package [sstoeckl/eventclock](https://github.com/sstoeckl/eventclock)
by Sebastian Stöckl, at commit `30781dd48a691dbdb253b6c10463e3b2d1731aed`.

The R package is the original. This repo holds only the Python code.
For the method, the paper and the R docs, go to the R repo.

The package measures information time ("the event clock") from prediction-market prices.
The R code is the reference. Where the R code and the R docs differ, the port follows the R code.
Tests compare the Python results with R outputs that we saved as JSON fixtures.

## Install

```
git clone https://github.com/cgebe/eventclock-py
cd eventclock-py
python -m pip install -e ".[test]"
```

The import name is `eventclock`.

You need Python 3.13 or newer. The dependencies are numpy, pandas, scipy and pyarrow.

## Quick start

```python
import datetime as dt

from eventclock import as_event_prices, event_clock, load_dataset

ep = as_event_prices(
    load_dataset("brexit2016"),
    time="date",
    price="q_leave",
    market_id="Brexit: Leave",
    event_date=dt.date(2016, 6, 23),
)
horizons = {"1W": dt.date(2016, 5, 31), "2W": dt.date(2016, 6, 7), "1M": dt.date(2016, 6, 23)}
res = event_clock(ep, from_=dt.date(2016, 5, 24), to=horizons)
```

## What is in the port

| R | Python | Module |
|---|---|---|
| `ec_default_params` | `ec_default_params` | `params` |
| `ec_logit`, `ec_ilogit` | `ec_logit`, `ec_ilogit` | `_utils` |
| `q_from_price` | `q_from_price` | `normalize` |
| `as_event_prices`, `summary`, `print` | `as_event_prices`, `EventPrices.summary()`, `repr()` | `event_prices` |
| `event_clock`, `event_clock_path`, `event_clock_forecast` | same names | `clock` |
| `ec_signature` | `ec_signature` | `signature` |
| `ec_validate`, `print` | `ec_validate`, `format_validation` | `validate` |
| `pm_daily`, `pm_stitch` (internal) | `pm_daily`, `pm_stitch` | `polymarket` |
| `ec_simulate_path` | `ec_simulate_path` | `transition` |
| datasets | `load_dataset(name)` | `datasets` |

Datasets: `brexit2016`, `us2016`, `polymarket2024`, `djt2024`, `fomc_meetings`.
All five come from the R package. `tools/export_data.R` converts them to parquet without changes.

Not in the port yet:

- Plots: `plot_q`, `plot_clock`, `plot_clock_vs_calendar`, `plot_signature`.
- Polymarket API: `pm_markets`, `pm_prices`, `pm_search`.
- Formulas and converters: `ec_moments`, `ec_transition_density`, `ec_simulate`,
  `ec_exceedance`, `ec_atm_event_call`, `ec_iv_rule`, `ec_relevance`, `ec_revision`,
  `ec_sigma_eff`, `ec_target_clock`, `ec_variance_share`, `event_beta`,
  `q_from_deal_spread`, `q_from_ffutures`.

## Time values

R has two time classes. The port maps them like this:

| R | Python input | `time_kind` | Storage in `EventPrices.data["time"]` |
|---|---|---|---|
| `Date` | `datetime.date` objects | `"date"` | naive `datetime64[ns]` at midnight |
| `POSIXct` | `datetime64` or `pd.Timestamp` | `"instant"` | tz-aware `datetime64[ns, tz]` |

- A time value without a time zone becomes UTC.
- For daily data, give `datetime.date` objects:
  `df["date"] = pd.to_datetime(df["date"]).dt.date`.
- Strings become UTC instants, as in R. They do not become dates.
- For strings, the port uses R's rules. The first format that parses *all* non-missing
  values wins. The formats are `%Y-%m-%d %H:%M:%OS`, `%Y-%m-%d %H:%M`, `%Y-%m-%d` and `%d.%m.%Y`.
  Text after a match is ignored. Thus `"2020-01-05T10:20:30"` becomes midnight.
  To keep the time, convert strings first with `pd.to_datetime(..., utc=True)`.
- Window bounds (`from_`, `to`, `at`) can be dates or datetimes.
  A date bound on an intraday series covers the whole local day, as in R.

## Differences to R

Names and arguments:

- `from` is a Python keyword. The argument is `from_`.
- `methods` needs exact names. R accepts partial names (`"bip"`) and silently drops unknown names.
  The port gives an error.
- Functions with random draws have an `rng` argument (a `numpy.random.Generator`, a seed, or `None`):
  `event_clock(se_method="bootstrap")` and `ec_simulate_path`.
  The random numbers are different from R. Tests check properties, not values.
- `event_clock_forecast` needs `horizon` as a keyword argument.

Errors, warnings and messages:

- R errors (`cli_abort`, `stopifnot`) are `ValueError`. The key words are the same.
- R warnings are `EventClockWarning` (a `UserWarning`). The texts are the same as in R.
- R messages (`cli_inform`) go to the `"eventclock"` logger at level INFO.
  Python does not show them by default. To see them:
  `import logging; logging.basicConfig(level=logging.INFO)`.

Output objects:

- `as_event_prices` returns a frozen `EventPrices` dataclass with `data`, `market_id`,
  `event_date`, `clip` and `time_kind`. All other functions return plain DataFrames.
- `event_clock_path` puts `market_id` and `event_date` into `DataFrame.attrs`.
  `ec_signature` puts `market_id` there.
- `format_validation(v)` returns the text of the R `print` method for `ec_validate`.
- In the `from`, `to` and `at` columns, date bounds are naive `datetime64`.
- Horizon labels follow R: `format()` rules for times, `as.character()` rules for numbers
  (`7` gives `"7d"`, `100000` gives `"1e+05d"`).

Stricter input checks:

- `event_date` must be a date or a datetime.
- Window bounds must be dates or datetimes. R also accepts strings.
- All values of `to` must use one time zone.
- `trailing` must be at least 1.
- Years must be in 1677 to 2262, the range of pandas timestamps.

Edge cases:

- On a DST change day, local midnight can be missing or occur twice in some time zones.
  A date bound then moves a missing midnight forward and uses the first of two midnights.
  New York and UTC do not have this case.
- `summary()` calculates `A_full` directly. The value is the same as
  `event_clock(methods="rv")`. A test checks this.

## Tests

```
TZ=America/New_York python -m pytest
```

`TZ=America/New_York` finds time-zone bugs, as in the R CI.

The fixtures in `tests/fixtures/*.json` come from R. The parity tests in `tests/test_parity.py`
compare every fixture with the Python result: data at 1e-12 relative, the same NaN positions,
the same warnings and the same messages.

Two seeded property tests (`test_lumpy_information` and the terminal-q check in
`test_composes_the_transition_and_realizes_the_clock`) have tight bounds, as in R.
They pass with their fixed seeds. A new NumPy random stream can make them fail.
Then use a new seed. Do not change the code.

## Regenerate the data and the fixtures

You need R with `arrow`, `jsonlite` and `pkgload`, and a clone of the R repo at the pinned commit.
Run from the root of this repo:

```
git clone https://github.com/sstoeckl/eventclock ../eventclock-r
git -C ../eventclock-r checkout 30781dd48a691dbdb253b6c10463e3b2d1731aed
Rscript tools/export_data.R ../eventclock-r
Rscript tools/make_fixtures.R ../eventclock-r
```

Both scripts stop if the R repo is not at the pinned commit, or if its `R/`, `data/` or
`DESCRIPTION` has local changes.

To move the port to a newer R version, change the commit in `tools/export_data.R`,
`tools/make_fixtures.R`, `tests/test_parity.py` and this file. Then run the two scripts and the tests.

## Citation

Cite the paper and the R package. See `CITATION.cff`.

## License

MIT. See `LICENSE`. The R package is MIT too, copyright Sebastian Stöckl.
