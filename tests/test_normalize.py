import numpy as np
import pandas as pd
import pytest
from scipy.special import logit

from eventclock import EventClockWarning, ec_ilogit, ec_logit, q_from_price
from helpers import record_warnings


def test_discount_divides():
    assert q_from_price(0.45, discount=0.9) == pytest.approx(0.5, rel=1e-15)
    np.testing.assert_allclose(q_from_price([0.2, 0.4], discount=[0.8, 0.5]), [0.25, 0.8])
    np.testing.assert_array_equal(q_from_price(pd.Series([0.1, 0.7])), [0.1, 0.7])


def test_scalar_in_scalar_out():
    assert isinstance(q_from_price(0.5), float)
    assert isinstance(q_from_price([0.5]), np.ndarray)


def test_discount_must_lie_in_unit_interval():
    for bad in (0.0, -0.2, 1.5):
        with pytest.raises(ValueError, match=r"`discount` must lie in \(0, 1\]\."):
            q_from_price(0.5, discount=bad)
    assert q_from_price(0.5, discount=1.0) == 0.5


def test_na_discount_gives_na():
    out = q_from_price([0.5, 0.5], discount=[np.nan, 0.5])
    assert np.isnan(out[0])
    assert out[1] == 1.0


def test_overround():
    np.testing.assert_allclose(
        q_from_price([0.55, 0.50], book=1.05, method="overround"), [0.55 / 1.05, 0.50 / 1.05]
    )
    with pytest.raises(ValueError) as e:
        q_from_price(0.5, method="overround")
    assert str(e.value) == (
        '`book` (sum of all outcome prices) is required for `method = "overround"`.'
    )


def test_bad_method():
    with pytest.raises(ValueError) as e:
        q_from_price(0.5, method="x")
    assert str(e.value) == '`method` must be one of "discount" or "overround", not "x".'
    with pytest.raises(ValueError):
        q_from_price(0.5, method="disc")


def test_price_must_be_numeric():
    for bad in (["0.5"], [True, False], "0.5"):
        with pytest.raises(ValueError, match="must be numeric"):
            q_from_price(bad)


def test_out_of_range_warning_matches_r():
    out, warns = record_warnings(q_from_price, [0.5, 1.2])
    np.testing.assert_array_equal(out, [0.5, 1.2])
    assert warns == ["1 normalized probability falls outside [0, 1]; check `discount`/`book`."]
    _, warns = record_warnings(q_from_price, [0.5, 1.2, -0.1])
    assert warns == ["2 normalized probabilities fall outside [0, 1]; check `discount`/`book`."]
    _, warns = record_warnings(q_from_price, [0.5, np.nan, 1.0, 0.0])
    assert warns == []


def test_zero_book_gives_inf_and_warns():
    with pytest.warns(EventClockWarning, match="1 normalized probability"):
        out = q_from_price([0.5], book=0.0, method="overround")
    assert out[0] == np.inf


def test_r_port_discount():
    assert q_from_price(0.5) == 0.5
    assert q_from_price(0.495, discount=0.99) == pytest.approx(0.5, rel=1e-15)
    np.testing.assert_allclose(q_from_price([0.49, 0.48], discount=[0.98, 0.96]), [0.5, 0.5])
    with pytest.raises(ValueError, match="must lie in"):
        q_from_price(0.5, discount=1.01)
    with pytest.raises(ValueError, match="must lie in"):
        q_from_price(0.5, discount=0)


def test_r_port_overround():
    assert q_from_price(0.52, book=1.04, method="overround") == pytest.approx(0.5, rel=1e-15)
    with pytest.raises(ValueError, match="book"):
        q_from_price(0.52, method="overround")


def test_r_port_out_of_bounds():
    with pytest.warns(EventClockWarning, match="outside"):
        res = q_from_price(1.05)
    assert res == 1.05
    _, warns = record_warnings(q_from_price, [0.2, np.nan, 0.8])
    assert warns == []


def test_r_port_logit():
    q = np.array([0.195, 0.5, 0.9])
    np.testing.assert_allclose(ec_logit(q), logit(q))
    np.testing.assert_allclose(ec_ilogit(ec_logit(q)), q)
    assert ec_logit(0.195) == pytest.approx(-1.4178431, abs=1e-6)
