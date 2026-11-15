"""Gaussian-process posterior on a finite candidate pool.

The pool is finite, so the GP is just a multivariate normal over N latent
utilities. We keep the full posterior (mean vector, covariance matrix) and
update it one observation at a time with the rank-one formulas of
Frazier, Powell & Dayanik (2009), Eqs. (5)-(8):

    sigma_tilde(Sigma, x) = Sigma e_x / sqrt(noise_var + Sigma_xx)
    mu'    = mu + sigma_tilde * Z,   Z = (y - mu_x) / sqrt(noise_var + Sigma_xx)
    Sigma' = Sigma - sigma_tilde sigma_tilde^T
"""
from __future__ import annotations

import numpy as np


def sigma_tilde(Sigma: np.ndarray, x: int, noise_var: float) -> np.ndarray:
    """Std. dev. of the change in every posterior mean caused by observing x once."""
    return Sigma[:, x] / np.sqrt(noise_var + Sigma[x, x])


def update(mu: np.ndarray, Sigma: np.ndarray, x: int, y: float, noise_var: float):
    """Condition on one noisy observation y of listing x. Returns new (mu, Sigma)."""
    st = sigma_tilde(Sigma, x, noise_var)
    z = (y - mu[x]) / np.sqrt(noise_var + Sigma[x, x])
    return mu + st * z, Sigma - np.outer(st, st)


def batch_posterior(m: np.ndarray, K: np.ndarray, idx, y, noise_var: float):
    """Posterior from scratch given all observations at once (used for testing)."""
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return m.copy(), K.copy()
    Kss = K[np.ix_(idx, idx)] + noise_var * np.eye(idx.size)
    Kas = K[:, idx]
    mu = m + Kas @ np.linalg.solve(Kss, np.asarray(y) - m[idx])
    Sigma = K - Kas @ np.linalg.solve(Kss, Kas.T)
    return mu, Sigma
