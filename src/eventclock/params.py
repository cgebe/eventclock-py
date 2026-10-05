from typing import Any


def ec_default_params() -> dict[str, Any]:
    return {
        "clip": (0.01, 0.99),
        "methods": ("rv", "truncated", "bipower", "largest1", "largest2"),
        "sample_every": 1,
        "trunc_sd": 3,
        "trailing": 40,
    }
