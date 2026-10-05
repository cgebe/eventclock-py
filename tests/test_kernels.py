import logging

import numpy as np
import pytest

from eventclock._utils import r_mad
from eventclock.kernels import (
    DEGENERATE_SCALE_MSG,
    kernel_bipower,
    kernel_drop_largest,
    kernel_rv,
    kernel_rv_boot,
    kernel_rv_se,
    kernel_truncated,
)

DL = np.array([0.01, 0.012, -0.011, 0.009, -0.013, 0.014, -0.01, 0.011, 0.5])


def test_hand_examples_from_r():
    assert kernel_rv([1, 2, 3]) == 14
    assert kernel_bipower([1, 2, 3]) == pytest.approx((np.pi / 2) * (2 + 6), rel=1e-15)
    assert np.isnan(kernel_bipower([1]))
    assert kernel_drop_largest([1, -3, 2], 1) == 5
    assert kernel_drop_largest([1, -3, 2], 2) == 1
    assert np.isnan(kernel_drop_largest([1], 1))


def test_truncation_drops_the_outlier():
    expected = np.sum(DL[np.abs(DL) <= 3 * r_mad(DL)] ** 2)
    assert kernel_truncated(DL, trunc_sd=3) == expected
    assert kernel_truncated(DL, trunc_sd=3) < np.sum(DL**2)


def test_kernels_match_r_values():
    assert kernel_truncated(DL, 3) == pytest.approx(0.0010319999999999999, rel=1e-14)
    assert kernel_truncated(DL, 1) == 0.0
    assert kernel_rv(DL) == pytest.approx(0.25103199999999998, rel=1e-14)
    assert kernel_bipower(DL) == pytest.approx(0.010053096491487338, rel=1e-14)
    assert kernel_drop_largest(DL, 1) == pytest.approx(0.0010319999999999999, rel=1e-14)
    assert kernel_drop_largest(DL, 2) == pytest.approx(0.00083599999999999994, rel=1e-14)
    assert kernel_rv_se(DL) == pytest.approx(0.20412437965776323, rel=1e-14)


def test_degenerate_scale_falls_back_to_rv(caplog):
    with caplog.at_level(logging.INFO, logger="eventclock"):
        assert kernel_truncated(np.full(5, 0.1)) == pytest.approx(np.sum(np.full(5, 0.1) ** 2))
    assert [r.getMessage() for r in caplog.records] == [DEGENERATE_SCALE_MSG]


def test_truncated_on_empty_and_nan(caplog):
    with caplog.at_level(logging.INFO, logger="eventclock"):
        assert kernel_truncated(np.empty(0)) == 0.0
        assert np.isnan(kernel_truncated([0.1, np.nan, 0.2]))
    assert len(caplog.records) == 2


def test_truncated_uses_scale_fn():
    assert kernel_truncated(DL, trunc_sd=3, scale_fn=lambda x: 1.0) == kernel_rv(DL)
    assert kernel_truncated(DL, trunc_sd=1, scale_fn=lambda x: 0.0101) == pytest.approx(
        0.01**2 + 0.009**2 + 0.01**2
    )


def test_drop_largest_ties_and_nan():
    assert kernel_drop_largest([2, -2, 1], 1) == 5
    assert kernel_drop_largest([2, -2, 1], 2) == 1
    assert np.isnan(kernel_drop_largest([np.nan, 1.0, 2.0], 1))
    assert np.isnan(kernel_drop_largest([1.0, 2.0, 3.0], 3))


def test_nan_propagates():
    bad = [0.1, np.nan, 0.2]
    assert np.isnan(kernel_rv(bad))
    assert np.isnan(kernel_bipower(bad))
    assert np.isnan(kernel_drop_largest(bad, 1))
    assert np.isnan(kernel_rv_se(bad))


def test_empty_input():
    assert kernel_rv(np.empty(0)) == 0.0
    assert np.isnan(kernel_bipower(np.empty(0)))
    assert np.isnan(kernel_rv_se([0.1]))
    assert kernel_rv_se([1, 2]) == pytest.approx(np.sqrt((2 / 3) * 17), rel=1e-15)


def test_boot_shape_and_multipliers():
    assert kernel_rv_boot([0.1]).shape == (0,)
    out = kernel_rv_boot([1.0, 0.0, 0.0], reps=500, rng=1)
    assert out.shape == (500,)
    np.testing.assert_allclose(np.unique(out), [1 - np.sqrt(2 / 3), 1 + np.sqrt(2 / 3)])


def test_boot_rng_is_reproducible():
    a = kernel_rv_boot(DL, reps=50, rng=7)
    np.testing.assert_array_equal(a, kernel_rv_boot(DL, reps=50, rng=np.random.default_rng(7)))
    assert not np.array_equal(a, kernel_rv_boot(DL, reps=50, rng=8))
    assert kernel_rv_boot(DL, reps=5).shape == (5,)


def test_boot_moments_match_quarticity():
    dL = np.random.default_rng(3).normal(0, 0.05, size=200)
    out = kernel_rv_boot(dL, reps=20000, rng=11)
    assert out.mean() == pytest.approx(kernel_rv(dL), rel=0.01)
    assert out.std(ddof=1) == pytest.approx(kernel_rv_se(dL), rel=0.03)
