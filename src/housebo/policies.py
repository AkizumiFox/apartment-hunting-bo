"""Selection rules. Each produces a full visiting trajectory over the pool.

A selection rule never depends on when we stop, so we run every rule to the end
of the pool and record, after each visit t = 1..N,

    rec_t    the listing we would sign if we stopped now
             (highest posterior mean among VISITED listings: you must visit to choose)
    V_t      f[rec_t] - lam * hours_t          (uses the hidden truth f; evaluation only)
    stop_t   whether the cost-aware stopping rule fires, for both index scales

Any stopping rule is then just a choice of t on this trajectory, which makes
paired comparisons and the hindsight bound exact.

There is no outside option: the user must rent something, so the first visit
always happens.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import gp
from .index import SCALES, ei, gittins, scale
from .simulator import Problem

SELECTORS = ("ei", "pbgi:latent", "pbgi:kg", "eipc:kg", "prior", "random")


@dataclass
class Trajectory:
    order: np.ndarray    # (T,) listing visited at step t
    hours: np.ndarray    # (T,) cumulative hours after t visits
    rec: np.ndarray      # (T,) recommended listing after t visits
    V: np.ndarray        # (T,) cost-adjusted true utility if we stop after t visits
    stop: dict           # scale -> (T,) bool, cost-aware rule fires after t visits
    snapshots: list | None = None   # optional (mu, sd, g) per step for the demo figure


def stop_flag(mu, var, visited_mask, cost, lam, noise_var, kind) -> bool:
    """Cost-aware stopping rule, evaluated on the updated posterior.

    Stop iff every unvisited listing's index is no higher than the best visited
    posterior mean, equivalently EI_s(x; best) <= lam * c(x) for all unvisited x.
    """
    un = ~visited_mask
    if not un.any():
        return True
    best = mu[visited_mask].max()
    s = scale(var[un], noise_var, kind)
    return bool(np.all(ei(mu[un], s, best) <= lam * cost[un]))


def select(name, mu, var, visited_mask, prob: Problem, step: int) -> int:
    cand = np.flatnonzero(~visited_mask)
    if name == "prior":
        return int(cand[np.argmax(prob.p[cand])])
    if name == "random":
        return int(prob.order_random[step])
    nv = prob.noise_var_model
    # With nothing visited yet, use the best prior mean as the EI threshold.
    best = mu[visited_mask].max() if visited_mask.any() else mu.max()
    if name == "ei":
        return int(cand[np.argmax(ei(mu[cand], scale(var[cand], nv, "latent"), best))])
    kind_name, kind = name.split(":")
    s = scale(var[cand], nv, kind)
    if kind_name == "pbgi":
        return int(cand[np.argmax(gittins(mu[cand], s, prob.lam * prob.cost[cand]))])
    if kind_name == "eipc":
        return int(cand[np.argmax(np.log(ei(mu[cand], s, best) + 1e-300)
                                  - np.log(prob.cost[cand]))])
    raise ValueError(name)


def run_trajectory(prob: Problem, selector: str, max_visits: int | None = None,
                   snapshots: bool = False) -> Trajectory:
    n = len(prob.p)
    T = n if max_visits is None else min(n, max_visits)
    mu, Sigma = prob.m.copy(), prob.K_model.copy()
    nv = prob.noise_var_model
    visited = np.zeros(n, bool)
    order = np.empty(T, int)
    rec = np.empty(T, int)
    hours = np.empty(T)
    V = np.empty(T)
    stop = {k: np.zeros(T, bool) for k in SCALES}
    snaps = [] if snapshots else None
    total = 0.0
    for t in range(T):
        var = np.diag(Sigma)
        i = select(selector, mu, var, visited, prob, t)
        if snapshots:
            s = scale(var, nv, "kg")
            snaps.append(dict(mu=mu.copy(), sd=np.sqrt(np.maximum(var, 0)),
                              g=gittins(mu, s, prob.lam * prob.cost),
                              visited=visited.copy(), next=i))
        mu, Sigma = gp.update(mu, Sigma, i, prob.observe(i), nv)
        visited[i] = True
        total += prob.cost[i]
        order[t], hours[t] = i, total
        vis = np.flatnonzero(visited)
        rec[t] = vis[np.argmax(mu[vis])]
        V[t] = prob.f[rec[t]] - prob.lam * total
        var = np.diag(Sigma)
        for k in SCALES:
            stop[k][t] = stop_flag(mu, var, visited, prob.cost, prob.lam, nv, k)
    if snapshots:
        var = np.diag(Sigma)
        snaps.append(dict(mu=mu.copy(), sd=np.sqrt(np.maximum(var, 0)),
                          g=gittins(mu, scale(var, nv, "kg"), prob.lam * prob.cost),
                          visited=visited.copy(), next=None))
    return Trajectory(order, hours, rec, V, stop, snaps)
