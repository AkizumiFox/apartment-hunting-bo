"""Simulated apartment pools and simulated users.

Everything here is synthetic. Feature ranges are assumptions chosen to look
like Taipei studio listings; nothing is scraped from any rental platform.

Generative model for one user and one pool of N listings
---------------------------------------------------------
    z_i      listing features (rent, size, commute, age, light) and a map location
    p_i      listing rating: the 0-10 score the user gives from the listing page
    delta_i  on-site surprise, delta ~ N(shift, K_true)
    f_i      latent utility, f_i = p_i + delta_i
    y_i      rating after a visit, y_i = f_i + eps_i, eps_i ~ N(0, noise_sd^2)
    c_i      visit cost in hours, known before the visit

K = linear kernel on features (systematic misjudgement of a feature)
  + RBF kernel on location (neighbourhood quality only seen on site)
  + diagonal (unit-specific quality only seen on site)

The optimiser sees z, p, c and the model kernel; it never sees f.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

FEATURES = ("rent", "size", "commute", "age", "light")
# Population-average taste on standardised features (rent hurts, size helps, ...)
BETA_BAR = np.array([-0.7, 0.6, -0.5, -0.3, 0.4])


@dataclass(frozen=True)
class Kernel:
    # Defaults are the values fitted on the LLM raters (experiments/e9_llm.py): total
    # on-site surprise sd ~1.6. The length scale is OUR assumption: that fit hit its bound.
    s_lin: float = 0.28   # per-feature sd of systematic misjudgement
    s_sp: float = 0.8     # sd of neighbourhood effect
    ell: float = 1.5      # km, length scale of neighbourhood effect
    s_idio: float = 1.2   # sd of unit-specific effect

    def scaled(self, a: float) -> "Kernel":
        """Same correlation structure, surprise sd multiplied by a."""
        return Kernel(self.s_lin * a, self.s_sp * a, self.ell, self.s_idio * a)

    def matrix(self, Z: np.ndarray, loc: np.ndarray) -> np.ndarray:
        d2 = ((loc[:, None, :] - loc[None, :, :]) ** 2).sum(-1)
        return (self.s_lin**2 * Z @ Z.T
                + self.s_sp**2 * np.exp(-0.5 * d2 / self.ell**2)
                + self.s_idio**2 * np.eye(len(Z)))


def decorrelate(K: np.ndarray) -> np.ndarray:
    """Same marginal variances, no cross-listing correlation."""
    return np.diag(np.diag(K))


@dataclass(frozen=True)
class Config:
    n: int = 60
    pool: str = "moi"              # "moi": real Taipei studio leases; "synthetic": generated
    n_districts: int = 6
    noise_sd: float = 0.65         # true rating noise (repeat-rating sd of the LLM raters)
    lam: float = 0.15              # utility points the user gives up per hour
    cost_spread: float = 1.0       # multiplies the spread of travel time
    kernel: Kernel = field(default_factory=Kernel)
    # --- what the optimiser believes (None = same as truth) ---
    model_kernel: Kernel | None = None
    model_noise_sd: float | None = None
    model_independent: bool = False   # optimiser ignores cross-listing correlation
    # --- departures of the world from the optimiser's prior ---
    world_independent: bool = False   # world has no cross-listing correlation
    prior_shift: float = 0.0          # delta mean (negative = listings look better than reality)
    prior_shrink: float = 0.0         # delta -= shrink * (p - mean p): glamorous listings disappoint
    model_knows_bias: bool = False    # reference: optimiser's prior mean already corrects shift/shrink
    # --- stress test: a few far, expensive, highly uncertain listings ---
    stress: bool = False

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)


@dataclass
class Problem:
    Z: np.ndarray        # (N, 5) standardised features
    raw: dict            # features in natural units, for display
    loc: np.ndarray      # (N, 2) km
    cost: np.ndarray     # (N,) hours
    p: np.ndarray        # (N,) listing rating given by the user before any visit
    m: np.ndarray        # (N,) optimiser's prior mean (= p unless model_knows_bias)
    f: np.ndarray        # (N,) latent utility (hidden from the optimiser)
    eps: np.ndarray      # (N,) pre-drawn rating noise, shared by all policies
    K_model: np.ndarray  # (N, N) optimiser's prior covariance of delta
    noise_var_model: float
    lam: float
    order_random: np.ndarray  # a pre-drawn random visiting order

    def observe(self, i: int) -> float:
        return float(self.f[i] + self.eps[i])


def make_problem(cfg: Config, rng: np.random.Generator) -> Problem:
    n = cfg.n
    if cfg.pool == "moi":
        Z, raw, loc = _moi_pool(n, rng)
        beta_bar = BETA_MOI
    else:
        Z, raw, loc = _synthetic_pool(cfg, rng)
        beta_bar = BETA_BAR
    dist = np.linalg.norm(loc, axis=1)  # user starts every trip from the origin

    # Visit cost: round trip at ~12 km/h door to door + 0.5 h viewing + 0.25 h scheduling.
    travel = 2 * dist / 12.0
    travel = travel.mean() + cfg.cost_spread * (travel - travel.mean())
    cost = np.clip(travel, 0.1, None) + 0.75

    beta = beta_bar + rng.normal(0, 0.25, size=beta_bar.size)  # this user's taste
    p = 5.0 + Z @ beta

    K_true = cfg.kernel.matrix(Z, loc)
    extra_var = np.zeros(n)
    if cfg.stress:
        # Stress test only: the 5 farthest listings cost ~10x more and are far less certain.
        far = np.argsort(-dist)[:5]
        cost = np.full(n, 0.6) + 0.1 * rng.random(n)
        cost[far] = 6.0
        extra_var[far] = 4.0 * np.diag(K_true)[far]
        K_true = K_true + np.diag(extra_var)
    if cfg.world_independent:
        K_true = decorrelate(K_true)

    L = np.linalg.cholesky(K_true + 1e-10 * np.eye(n))
    delta = L @ rng.normal(size=n)
    bias = cfg.prior_shift - cfg.prior_shrink * (p - p.mean())
    delta = delta + bias
    m = p + bias if cfg.model_knows_bias else p.copy()
    f = p + delta
    eps = cfg.noise_sd * rng.normal(size=n)

    mk = cfg.model_kernel or cfg.kernel
    K_model = mk.matrix(Z, loc) + np.diag(extra_var)
    if cfg.model_independent:
        K_model = decorrelate(K_model)
    model_noise = cfg.noise_sd if cfg.model_noise_sd is None else cfg.model_noise_sd

    return Problem(Z=Z, raw=raw, loc=loc, cost=cost, p=p, m=m, f=f, eps=eps,
                   K_model=K_model, noise_var_model=model_noise**2, lam=cfg.lam,
                   order_random=rng.permutation(n))


# ---------------------------------------------------------------------------------------
# Real pool: registered studio leases in Taipei City (Ministry of the Interior open data,
# prepared by experiments/prepare_moi.py). Real: district, rent, size, building age,
# elevator, furnished. ASSUMED: district centroids (approximate), the user's base (NTU
# main campus), commute = straight-line distance at 4 min/km + 10 min.
# ---------------------------------------------------------------------------------------
MOI_CSV = Path(__file__).resolve().parents[2] / "data" / "taipei_studios.csv"
HOME = (25.0174, 121.5397)
CENTROID = {  # approximate (lat, lon) of the populated part of each district
    "中正區": (25.0324, 121.5199), "大同區": (25.0634, 121.5130), "中山區": (25.0640, 121.5330),
    "松山區": (25.0580, 121.5576), "大安區": (25.0268, 121.5435), "萬華區": (25.0286, 121.4979),
    "信義區": (25.0306, 121.5716), "士林區": (25.0950, 121.5250), "北投區": (25.1320, 121.5010),
    "內湖區": (25.0790, 121.5890), "南港區": (25.0520, 121.6070), "文山區": (24.9920, 121.5600),
}
MOI_FEATURES = ("rent", "size", "commute", "age", "elevator", "furnished")
BETA_MOI = np.array([-0.7, 0.6, -0.5, -0.3, 0.3, 0.2])


@lru_cache(maxsize=1)
def _moi_table():
    with MOI_CSV.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["district"] in CENTROID]
    km = np.array([[(CENTROID[r["district"]][1] - HOME[1]) * 111.32 * np.cos(np.radians(HOME[0])),
                    (CENTROID[r["district"]][0] - HOME[0]) * 110.57] for r in rows])
    num = {k: np.array([float(r[k]) for r in rows]) for k in
           ("rent", "size_ping", "age_years", "elevator", "furnished", "floor")}
    return rows, km, num


def _moi_pool(n, rng):
    rows, km, num = _moi_table()
    idx = rng.choice(len(rows), size=n, replace=False)
    loc = km[idx]
    commute = 10 + 4.0 * np.linalg.norm(loc, axis=1)
    raw = {"rent": num["rent"][idx], "size": num["size_ping"][idx], "commute": commute,
           "age": num["age_years"][idx], "elevator": num["elevator"][idx],
           "furnished": num["furnished"][idx], "floor": num["floor"][idx],
           "district": [rows[i]["district"] for i in idx]}
    Z = np.column_stack([raw[k] for k in MOI_FEATURES])
    sd = Z.std(0)
    Z = (Z - Z.mean(0)) / np.where(sd > 0, sd, 1.0)
    return Z, raw, loc


def _synthetic_pool(cfg, rng):
    n = cfg.n
    centers = rng.uniform(-6, 6, size=(cfg.n_districts, 2))
    district = rng.integers(0, cfg.n_districts, size=n)
    loc = centers[district] + rng.normal(0, 0.8, size=(n, 2))
    dist = np.linalg.norm(loc, axis=1)
    premium = rng.normal(0, 0.6, size=cfg.n_districts)
    size = rng.normal(0, 1, n)
    age = rng.normal(0, 1, n)
    light = rng.normal(0, 1, n)
    commute = (dist - dist.mean()) / dist.std() + rng.normal(0, 0.3, n)
    rent = 0.6 * size - 0.3 * age + premium[district] + rng.normal(0, 0.5, n)
    Z = np.column_stack([rent, size, commute, age, light])
    Z = (Z - Z.mean(0)) / Z.std(0)
    raw = {  # natural units for display (assumed ranges)
        "rent": np.clip(16000 + 3500 * Z[:, 0], 9000, 26000),
        "size": np.clip(9 + 2.2 * Z[:, 1], 5, 16),
        "commute": np.clip(32 + 11 * Z[:, 2], 8, 60),
        "age": np.clip(25 + 12 * Z[:, 3], 0, 50),
        "light": np.clip(np.round(1 + 0.8 * Z[:, 4]), 0, 2),
    }
    return Z, raw, loc
