"""Paired summaries over simulated users."""
from __future__ import annotations

import numpy as np


def mean_se(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def paired(a, b):
    """Mean and standard error of the per-user difference a - b."""
    return mean_se(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


def fmt(m_se, digits: int = 3) -> str:
    m, se = m_se
    return f"{m:+.{digits}f} ± {se:.{digits}f}"


def best_fixed_k(V_by_k: np.ndarray):
    """V_by_k: (users, K) value of stopping after k visits. Returns (k*, column).

    One K shared by all users, chosen in hindsight on the same users (optimistic
    for the baseline).
    """
    k = int(np.argmax(V_by_k.mean(0)))
    return k + 1, V_by_k[:, k]
