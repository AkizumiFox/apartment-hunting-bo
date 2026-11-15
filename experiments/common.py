"""Shared runner: simulate many users, every policy on the same pool, truth and noise."""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from housebo.policies import run_trajectory
from housebo.simulator import Config, make_problem

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
N_USERS = 1000
SEED = 20261115


TRAIN_SEED = SEED + 1     # users used only to calibrate lambda and to tune fixed-K baselines
BUDGETS = (6, 9, 15)      # expected viewing hours the user might set; 9 h is the main setting


def budgets() -> dict:
    """Written by experiments/budgets.py before any method comparison."""
    return json.loads((RESULTS / "budgets.json").read_text())


def frozen_lambda(hours: int = 9) -> float:
    """lambda whose simulated searches spend `hours` on average (calibrated on training users)."""
    return budgets()["budgets"][str(hours)]["lam"]


def base_config(**kw) -> Config:
    return Config(lam=frozen_lambda()).with_(**kw)


def _one(args):
    cfg, seed, user, selectors, max_visits = args
    prob = make_problem(cfg, np.random.default_rng([seed, user]))
    out = {s: run_trajectory(prob, s, max_visits) for s in selectors}
    return out, prob.f.max() - 0  # trajectories + best achievable utility (for reference)


def run_users(cfg: Config, selectors, n_users: int = N_USERS, seed: int = SEED,
              max_visits: int | None = None):
    """Returns {selector: [Trajectory per user]}. User u is identical across selectors."""
    jobs = [(cfg, seed, u, tuple(selectors), max_visits) for u in range(n_users)]
    with ProcessPoolExecutor(max_workers=min(20, os.cpu_count() or 1)) as ex:
        res = list(ex.map(_one, jobs, chunksize=8))
    return {s: [r[0][s] for r in res] for s in selectors}


def at(trajs, taus, field="V"):
    """Value of `field` at each user's stopping time."""
    return np.array([getattr(t, field)[tau - 1] for t, tau in zip(trajs, taus)])


def save(name: str, obj) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(obj, indent=2, default=float))
