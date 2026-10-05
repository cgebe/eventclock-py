from collections.abc import Callable

import numpy as np

from ._utils import as_rng, logger, r_mad

DEGENERATE_SCALE_MSG = (
    "Robust scale of the increments is degenerate (0 or non-finite); "
    "truncation is disabled and plain realized variation returned."
)


def kernel_rv(dL: np.ndarray) -> float:
    dL = np.asarray(dL, dtype=float)
    return float(np.sum(dL**2))


def kernel_bipower(dL: np.ndarray) -> float:
    dL = np.asarray(dL, dtype=float)
    if len(dL) < 2:
        return np.nan
    return float((np.pi / 2) * np.sum(np.abs(dL[1:]) * np.abs(dL[:-1])))


def kernel_truncated(
    dL: np.ndarray,
    trunc_sd: float = 3,
    scale_fn: Callable[[np.ndarray], float] = r_mad,
) -> float:
    dL = np.asarray(dL, dtype=float)
    s = scale_fn(dL)
    if not np.isfinite(s) or s <= 0:
        logger.info(DEGENERATE_SCALE_MSG)
        return kernel_rv(dL)
    return float(np.sum(dL[np.abs(dL) <= trunc_sd * s] ** 2))


def kernel_drop_largest(dL: np.ndarray, k: int = 1) -> float:
    dL = np.asarray(dL, dtype=float)
    if len(dL) <= k:
        return np.nan
    order = np.argsort(-np.abs(dL), kind="stable")
    return float(np.sum(dL[np.sort(order[k:])] ** 2))


def kernel_rv_se(dL: np.ndarray) -> float:
    dL = np.asarray(dL, dtype=float)
    if len(dL) < 2:
        return np.nan
    return float(np.sqrt((2 / 3) * np.sum(dL**4)))


def kernel_rv_boot(
    dL: np.ndarray,
    reps: int = 999,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    dL = np.asarray(dL, dtype=float)
    if len(dL) < 2:
        return np.empty(0)
    d2 = dL**2
    m = 1 + np.sqrt(2 / 3) * (2 * as_rng(rng).integers(0, 2, size=(reps, len(d2))) - 1)
    return m @ d2
