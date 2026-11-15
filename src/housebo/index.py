"""Expected improvement and the Pandora's-box Gittins index on a finite pool.

For an unvisited listing with posterior mean mu, a scale s, and visit cost c,
the index g solves

    s * h((mu - g) / s) = lam * c,     h(z) = phi(z) + z * Phi(z).

Two choices of the scale s:

    "latent": s = sigma                       (posterior sd of the latent utility;
                                               treats one visit as revealing the truth)
    "kg":     s = sigma^2 / sqrt(sigma^2 + noise_var)
                                              (sd of the change in this listing's
                                               posterior mean after one noisy visit;
                                               self-component of sigma_tilde in
                                               Frazier, Powell & Dayanik 2009, Eq. 7)

With noise_var = 0 the two coincide.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

_SQRT2PI = np.sqrt(2.0 * np.pi)
SCALES = ("latent", "kg")


def h(z):
    """E[max(Z + z, 0)] for Z ~ N(0, 1)."""
    z = np.asarray(z, dtype=float)
    return np.exp(-0.5 * z * z) / _SQRT2PI + z * ndtr(z)


def scale(var, noise_var: float, kind: str):
    var = np.maximum(np.asarray(var, dtype=float), 1e-24)
    if kind == "latent":
        return np.sqrt(var)
    if kind == "kg":
        return var / np.sqrt(var + noise_var)
    raise ValueError(kind)


def ei(mu, s, threshold):
    """Expected improvement of N(mu, s^2) over a threshold."""
    return s * h((mu - threshold) / s)


def gittins(mu, s, lam_c, iters: int = 80):
    """Solve s * h((mu - g)/s) = lam_c for g, vectorised bisection on z = (mu-g)/s."""
    mu = np.asarray(mu, dtype=float)
    target = np.asarray(lam_c, dtype=float) / s
    lo = np.full_like(target, -40.0)
    hi = target + 1.0  # h(z) >= z, so h(hi) > target
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        above = h(mid) > target
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    return mu - s * 0.5 * (lo + hi)
