"""Stopping rules expressed as a number of visits tau on a recorded trajectory.

All functions return tau in 1..T (number of visits made before signing).
"""
from __future__ import annotations

import numpy as np

from .policies import Trajectory


def cap_by_budget(traj: Trajectory, tau: int, budget_hours: float | None) -> int:
    """Hard cap B: never start a visit that would push total hours above B.

    This is a user-facing safeguard added on top of the rule; it carries no
    guarantee from the literature. The first visit always happens.
    """
    if budget_hours is None:
        return tau
    within = int(np.searchsorted(traj.hours, budget_hours, side="right"))
    return max(1, min(tau, within))


def cost_aware(traj: Trajectory, kind: str = "kg", min_visits: int = 1,
               budget_hours: float | None = None) -> int:
    """First t >= min_visits at which the cost-aware rule fires (else: whole pool)."""
    flags = traj.stop[kind]
    hits = np.flatnonzero(flags[min_visits - 1:])
    tau = int(hits[0]) + min_visits if hits.size else len(flags)
    return cap_by_budget(traj, tau, budget_hours)


def fixed(traj: Trajectory, k: int) -> int:
    return min(k, len(traj.V))


def immediate(traj: Trajectory) -> int:
    return 1


def hindsight(traj: Trajectory) -> int:
    """Per-trial best stopping time in hindsight (uses the hidden truth; an upper bound)."""
    return int(np.argmax(traj.V)) + 1
