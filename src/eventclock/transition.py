import numpy as np
import pandas as pd

from ._utils import as_rng, check_nonneg, check_prob, ec_ilogit, ec_logit


def _is_scalar_number(v: object) -> bool:
    return np.ndim(v) == 0 and isinstance(np.asarray(v).item(), (int, float)) and not isinstance(v, bool)


def ec_simulate_path(
    n_paths: int,
    n_steps: int,
    q: float,
    A: float,
    jump_share: float = 0,
    n_jumps: int = 1,
    rng: np.random.Generator | int | None = None,
) -> pd.DataFrame:
    if not _is_scalar_number(q):
        raise ValueError("`q` must be a single number.")
    if not _is_scalar_number(A):
        raise ValueError("`A` must be a single number.")
    if not n_steps >= 1:
        raise ValueError("`n_steps` must be at least 1.")
    if not (jump_share >= 0 and jump_share < 1):
        raise ValueError("`jump_share` must lie in [0, 1).")
    if not n_jumps >= 1:
        raise ValueError("`n_jumps` must be at least 1.")
    check_prob(q)
    check_nonneg(A)
    n_steps = int(n_steps)
    n_jumps = int(n_jumps)
    if jump_share > 0 and n_jumps >= n_steps:
        raise ValueError("`n_jumps` must be smaller than `n_steps`.")

    gen = as_rng(rng)
    q, A = float(q), float(A)
    L0 = float(ec_logit(q))
    steps = np.arange(n_steps + 1)
    frames = []
    for i in range(1, int(n_paths) + 1):
        a = np.full(n_steps, A * (1 - jump_share) / n_steps)
        is_jump = np.zeros(n_steps, dtype=bool)
        if jump_share > 0:
            pos = gen.choice(n_steps, size=n_jumps, replace=False)
            a[pos] += A * jump_share / n_jumps
            is_jump[pos] = True
        J = int(gen.binomial(1, q))
        incr = (J - 0.5) * a + np.sqrt(a) * gen.standard_normal(n_steps)
        L = np.concatenate([[L0], L0 + np.cumsum(incr)])
        frames.append(
            pd.DataFrame(
                {
                    "path": i,
                    "step": steps,
                    "t_frac": steps / n_steps,
                    "J": J,
                    "L": L,
                    "q": ec_ilogit(L),
                    "is_jump": np.concatenate([[False], is_jump]),
                }
            )
        )
    if not frames:
        return pd.DataFrame(
            {
                "path": pd.Series(dtype="int64"),
                "step": pd.Series(dtype="int64"),
                "t_frac": pd.Series(dtype="float64"),
                "J": pd.Series(dtype="int64"),
                "L": pd.Series(dtype="float64"),
                "q": pd.Series(dtype="float64"),
                "is_jump": pd.Series(dtype="bool"),
            }
        )
    out = pd.concat(frames, ignore_index=True)
    return out.astype({"path": "int64", "step": "int64", "J": "int64"})
