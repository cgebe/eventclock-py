import datetime as dt

import numpy as np
import pandas as pd
import pytest

from eventclock import as_event_prices, ec_logit, ec_simulate_path, event_clock
from eventclock.kernels import kernel_bipower, kernel_rv


def path_qv(paths: pd.DataFrame) -> pd.Series:
    return paths.groupby("path")["L"].apply(lambda L: kernel_rv(np.diff(L)))


def test_lumpy_information():
    paths = ec_simulate_path(200, 100, q=0.4, A=1, jump_share=0.6, n_jumps=2, rng=9)
    assert "is_jump" in paths.columns
    assert paths.loc[paths["path"] == 1, "is_jump"].sum() == 2
    a_base = 1 * (1 - 0.6) / 100
    a_jump = a_base + 1 * 0.6 / 2
    exp_qv = 1 + (2 * a_jump**2 + 98 * a_base**2) / 4
    assert abs(path_qv(paths).mean() - exp_qv) < 0.04
    assert abs(paths.loc[paths["step"] == 100, "q"].mean() - 0.4) < 0.07
    ratio = paths.groupby("path")["L"].apply(
        lambda L: kernel_bipower(np.diff(L)) / kernel_rv(np.diff(L))
    )
    assert ratio.mean() < 0.85
    with pytest.raises(ValueError, match="smaller than"):
        ec_simulate_path(1, 10, 0.4, 1, jump_share=0.5, n_jumps=10)
    with pytest.raises(ValueError, match="`jump_share` must"):
        ec_simulate_path(1, 10, 0.4, 1, jump_share=1)


def test_composes_the_transition_and_realizes_the_clock():
    paths = ec_simulate_path(n_paths=300, n_steps=100, q=0.3, A=1, rng=2)
    assert len(paths) == 300 * 101
    assert paths.loc[paths["step"] == 0, "L"].unique().tolist() == [ec_logit(0.3)]
    assert path_qv(paths).mean() == pytest.approx(1, rel=0.05)
    assert paths.loc[paths["step"] == 100, "q"].mean() == pytest.approx(0.3, rel=0.05)
    p1 = paths[paths["path"] == 1]
    times = [dt.date(2020, 1, 1) + dt.timedelta(days=int(s)) for s in p1["step"]]
    ep = as_event_prices(
        pd.DataFrame({"time": pd.Series(times, dtype=object), "q": p1["q"].to_numpy()}),
        clip=(1e-6, 1 - 1e-6),
    )
    assert event_clock(ep, methods="rv")["A"].item() == pytest.approx(
        kernel_rv(np.diff(p1["L"])), abs=1e-6
    )


def test_output_layout():
    p = ec_simulate_path(2, 4, q=0.6, A=0.5, jump_share=0.5, n_jumps=1, rng=0)
    assert list(p.columns) == ["path", "step", "t_frac", "J", "L", "q", "is_jump"]
    assert p["path"].tolist() == [1] * 5 + [2] * 5
    assert p["step"].tolist() == list(range(5)) * 2
    np.testing.assert_allclose(p["t_frac"], np.tile(np.arange(5) / 4, 2))
    assert p.groupby("path")["J"].nunique().eq(1).all()
    assert set(p["J"]) <= {0, 1}
    assert not p.loc[p["step"] == 0, "is_jump"].any()
    assert p.groupby("path")["is_jump"].sum().eq(1).all()
    np.testing.assert_allclose(p["q"], 1 / (1 + np.exp(-p["L"])), rtol=1e-14)
    assert p["path"].dtype == np.int64
    assert p["is_jump"].dtype == bool


def test_without_jumps_no_step_is_a_jump():
    p = ec_simulate_path(3, 10, q=0.5, A=1, rng=1)
    assert not p["is_jump"].any()


def test_zero_clock_stays_put():
    p = ec_simulate_path(1, 5, q=0.2, A=0, rng=1)
    assert (p["L"] == ec_logit(0.2)).all()


def test_zero_paths():
    p = ec_simulate_path(0, 5, q=0.2, A=1)
    assert len(p) == 0
    assert list(p.columns) == ["path", "step", "t_frac", "J", "L", "q", "is_jump"]


def test_rng_is_reproducible():
    a = ec_simulate_path(2, 20, q=0.4, A=1, jump_share=0.3, n_jumps=2, rng=5)
    b = ec_simulate_path(2, 20, q=0.4, A=1, jump_share=0.3, n_jumps=2, rng=np.random.default_rng(5))
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(ec_simulate_path(2, 20, q=0.4, A=1, jump_share=0.3, n_jumps=2, rng=6))


def test_input_validation():
    with pytest.raises(ValueError, match="`q` must be a single"):
        ec_simulate_path(1, 5, q=[0.2, 0.3], A=1)
    with pytest.raises(ValueError, match="`A` must be a single"):
        ec_simulate_path(1, 5, q=0.2, A=[1, 2])
    with pytest.raises(ValueError, match="`n_steps` must"):
        ec_simulate_path(1, 0, q=0.2, A=1)
    with pytest.raises(ValueError, match="`jump_share` must"):
        ec_simulate_path(1, 5, q=0.2, A=1, jump_share=-0.1)
    with pytest.raises(ValueError, match="`n_jumps` must"):
        ec_simulate_path(1, 5, q=0.2, A=1, n_jumps=0)
    with pytest.raises(ValueError, match="strictly inside"):
        ec_simulate_path(1, 5, q=1.0, A=1)
    with pytest.raises(ValueError, match="non-negative"):
        ec_simulate_path(1, 5, q=0.2, A=-1)
    assert len(ec_simulate_path(1, 10, 0.4, 1, jump_share=0, n_jumps=10, rng=0)) == 11
