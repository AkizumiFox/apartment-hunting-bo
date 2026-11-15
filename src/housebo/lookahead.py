"""Exact one-step value and two-step look-ahead under "must visit to choose".

State: posterior mean mu, posterior covariance Sigma, visited set S (boolean mask).
v(S) = max_{i in S} mu_i is what the user gets by signing now.

Visiting x moves ALL posterior means along one direction (Frazier et al. 2009):
    mu' = mu + sigma_tilde(Sigma, x) * Z,   Z ~ N(0,1),   Sigma' = Sigma - st st^T.

One-step value (restricted knowledge gradient minus cost):
    Q1(x) = E_Z[ max_{i in S+x} mu'_i ] - lam*c_x - v(S)
Two-step value (visit x, observe, then either sign or make the best second visit):
    Q2(x) = E_Z[ v' + max(0, max_{x'} Q1'(x')) ] - lam*c_x - v(S)

Because only visited listings can be chosen, information about OTHER unvisited
listings pays off only through a later visit; Q1 cannot see it, Q2 is the
shortest horizon that can.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import ndtr

from . import gp
from .index import gittins, scale

_SQRT2PI = np.sqrt(2.0 * np.pi)


def _phi(z):
    return np.exp(-0.5 * z * z) / _SQRT2PI


def gh(n: int):
    """Nodes and weights for E[g(Z)], Z ~ N(0,1)."""
    z, w = hermegauss(n)
    return z, w / w.sum()


def grid(n: int = 201, lim: float = 6.0):
    """Trapezoid nodes/weights for E[g(Z)] on a fine uniform grid.

    Used for integrands with kinks (max of lines), where Gauss-Hermite converges slowly.
    """
    z = np.linspace(-lim, lim, n)
    w = _phi(z) * (z[1] - z[0])
    w[[0, -1]] *= 0.5
    return z, w / w.sum()


def emax_lines_exact(a, b) -> float:
    """E[max_i (a_i + b_i Z)], Z ~ N(0,1), exactly via the upper envelope of the lines."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    order = np.lexsort((a, b))            # by slope, ties by intercept
    a, b = a[order], b[order]
    keep = np.ones(len(a), bool)          # among equal slopes keep the highest intercept
    keep[:-1] = b[1:] != b[:-1]
    a, b = a[keep], b[keep]
    hull, starts = [0], [-np.inf]         # line index, z from which it is on top
    for j in range(1, len(a)):
        while True:
            i = hull[-1]
            z = (a[i] - a[j]) / (b[j] - b[i])
            if z <= starts[-1]:
                hull.pop(); starts.pop()
            else:
                break
        hull.append(j); starts.append(z)
    lo = np.array(starts)
    hi = np.append(lo[1:], np.inf)
    ah, bh = a[hull], b[hull]
    return float(np.sum(ah * (ndtr(hi) - ndtr(lo)) + bh * (_phi(lo) - _phi(hi))))


def emax_lines_quad(a, b, nodes: int = 64) -> float:
    z, w = gh(nodes)
    return float(w @ np.max(np.asarray(a)[:, None] + np.asarray(b)[:, None] * z, axis=0))


def q1_exact(mu, Sigma, visited, cost, lam, noise_var, x: int) -> float:
    st = gp.sigma_tilde(Sigma, x, noise_var)
    idx = np.append(np.flatnonzero(visited), x)
    v = mu[visited].max() if visited.any() else -np.inf
    return emax_lines_exact(mu[idx], st[idx]) - lam * cost[x] - v


def q1_all(mu, Sigma, visited, cost, lam, noise_var, nodes: int = 201):
    """Q1 for every unvisited listing (grid quadrature, vectorised). Returns (cand, Q1)."""
    z, w = grid(nodes)
    cand = np.flatnonzero(~visited)
    vis = np.flatnonzero(visited)
    ST = Sigma[:, cand] / np.sqrt(noise_var + np.diag(Sigma)[cand])       # (N, C)
    own = mu[cand][:, None] + ST[cand, np.arange(len(cand))][:, None] * z  # (C, n)
    if vis.size:
        others = (mu[vis][None, :, None] + ST[vis].T[:, :, None] * z).max(1)  # (C, n)
        val = np.maximum(own, others) @ w
        v = mu[vis].max()
    else:
        val, v = own @ w, -np.inf
    return cand, val - lam * cost[cand] - v


def q2_all(mu, Sigma, visited, cost, lam, noise_var, outer: int = 121, inner: int = 201):
    """Q2 for every unvisited listing. Returns (cand, Q2).

    Requires at least one visited listing (the first visit is forced anyway).
    """
    zo, wo = grid(outer)
    cand = np.flatnonzero(~visited)
    v = mu[visited].max()
    out = np.empty(len(cand))
    for n_x, x in enumerate(cand):
        st = gp.sigma_tilde(Sigma, x, noise_var)
        Sigma2 = Sigma - np.outer(st, st)
        vis2 = visited.copy(); vis2[x] = True
        total = 0.0
        for zk, wk in zip(zo, wo):
            mu2 = mu + st * zk
            v2 = mu2[vis2].max()
            cont = 0.0
            if not vis2.all():
                _, q1 = q1_all(mu2, Sigma2, vis2, cost, lam, noise_var, inner)
                cont = max(0.0, q1.max())
            total += wk * (v2 + cont)
        out[n_x] = total - lam * cost[x] - v
    return cand, out


def run_lookahead_policy(prob, depth: int, max_visits: int | None = None):
    """Greedy policy on Q_depth: visit argmax Q while max Q > 0. Returns (order, V, hours)."""
    n = len(prob.p)
    mu, Sigma = prob.m.copy(), prob.K_model.copy()
    nv, lam, cost = prob.noise_var_model, prob.lam, prob.cost
    visited = np.zeros(n, bool)
    order, total = [], 0.0
    qfun = q1_all if depth == 1 else q2_all
    while not visited.all() and (max_visits is None or len(order) < max_visits):
        if not visited.any():
            # Forced first visit: same rule for every policy (highest kg-scaled index),
            # so that policies differ only in what they do afterwards.
            cand = np.arange(n)
            q = gittins(mu, scale(np.diag(Sigma), nv, "kg"), lam * cost)
        else:
            cand, q = qfun(mu, Sigma, visited, cost, lam, nv)
            if q.max() <= 0:
                break
        x = int(cand[np.argmax(q)])
        mu, Sigma = gp.update(mu, Sigma, x, prob.observe(x), nv)
        visited[x] = True
        order.append(x); total += cost[x]
    vis = np.flatnonzero(visited)
    rec = vis[np.argmax(mu[vis])]
    return np.array(order), float(prob.f[rec] - lam * total), total
