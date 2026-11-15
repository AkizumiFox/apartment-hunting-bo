import numpy as np
import pytest

from housebo import gp, lookahead as la, stopping
from housebo.index import ei, gittins, h, scale
from housebo.policies import SELECTORS, run_trajectory
from housebo.simulator import Config, make_problem


def problem(seed=0, **kw):
    return make_problem(Config(**{"pool": "synthetic", **kw}), np.random.default_rng(seed))


# ---------- index ----------
def test_gittins_satisfies_definition():
    rng = np.random.default_rng(1)
    mu, s = rng.normal(6, 1, 500), rng.uniform(0.01, 3, 500)
    lam_c = rng.uniform(0.005, 1.5, 500)
    g = gittins(mu, s, lam_c)
    assert np.max(np.abs(s * h((mu - g) / s) - lam_c)) < 1e-8


def test_gittins_decreases_with_cost_and_matches_stop_rule():
    c = np.linspace(0.05, 2, 50)
    g = gittins(np.full(50, 6.0), np.full(50, 1.0), c)
    assert np.all(np.diff(g) < 0)
    # g <= best  <=>  EI(best) <= lam*c
    for best in (5.0, 6.0, 6.5, 7.5):
        assert np.array_equal(g <= best, ei(6.0, 1.0, best) <= c)


def test_kg_scale_is_self_component_of_sigma_tilde():
    p = problem()
    nv = 0.4**2
    for x in (0, 7, 31):
        st = gp.sigma_tilde(p.K_model, x, nv)
        assert st[x] == pytest.approx(scale(p.K_model[x, x], nv, "kg"))
    assert scale(2.0, 0.0, "kg") == pytest.approx(scale(2.0, 0.0, "latent"))


# ---------- GP ----------
def test_incremental_update_matches_batch_and_brute_force():
    p = problem(n=25)
    nv = p.noise_var_model
    idx = [3, 11, 4, 20, 8]
    y = [p.observe(i) for i in idx]
    mu, S = p.p.copy(), p.K_model.copy()
    for i, yi in zip(idx, y):
        mu, S = gp.update(mu, S, i, yi, nv)
    mu_b, S_b = gp.batch_posterior(p.p, p.K_model, idx, y, nv)
    assert np.allclose(mu, mu_b, atol=1e-10) and np.allclose(S, S_b, atol=1e-10)
    # brute force: condition the joint Gaussian of (f, y)
    n = len(p.p)
    A = np.zeros((len(idx), n)); A[np.arange(len(idx)), idx] = 1
    J = np.block([[p.K_model, p.K_model @ A.T],
                  [A @ p.K_model, A @ p.K_model @ A.T + nv * np.eye(len(idx))]])
    G = J[:n, n:] @ np.linalg.inv(J[n:, n:])
    assert np.allclose(mu, p.p + G @ (np.array(y) - p.p[idx]), atol=1e-9)
    assert np.allclose(S, J[:n, :n] - G @ J[n:, :n], atol=1e-9)


# ---------- look-ahead ----------
def test_emax_lines_exact_vs_quadrature_vs_monte_carlo():
    rng = np.random.default_rng(2)
    for _ in range(20):
        k = rng.integers(1, 9)
        a, b = rng.normal(0, 1, k), rng.normal(0, 1, k)
        if k > 2:
            b[1] = b[0]  # duplicate slope
        exact = la.emax_lines_exact(a, b)
        assert exact == pytest.approx(la.emax_lines_quad(a, b, 200), abs=2e-3)
        z = rng.normal(size=400_000)
        mc = np.max(a[:, None] + b[:, None] * z, axis=0).mean()
        assert exact == pytest.approx(mc, abs=1e-2)


def _state(p, k=4):
    mu, S = p.p.copy(), p.K_model.copy()
    vis = np.zeros(len(mu), bool)
    for i in range(k):
        mu, S = gp.update(mu, S, i, p.observe(i), p.noise_var_model)
        vis[i] = True
    return mu, S, vis


def test_q1_quadrature_matches_exact():
    p = problem(n=20)
    mu, S, vis = _state(p)
    cand, q = la.q1_all(mu, S, vis, p.cost, p.lam, p.noise_var_model)
    ex = [la.q1_exact(mu, S, vis, p.cost, p.lam, p.noise_var_model, x) for x in cand]
    assert np.allclose(q, ex, atol=5e-4)


def test_noiseless_q1_equals_ei_minus_cost():
    """No noise: visited means do not move, so Q1 > 0 <=> index rule says continue."""
    p = problem(n=20, noise_sd=0.0)
    mu, S, vis = _state(p)
    best = mu[vis].max()
    for x in np.flatnonzero(~vis):
        q = la.q1_exact(mu, S, vis, p.cost, p.lam, 0.0, x)
        assert q == pytest.approx(ei(mu[x], np.sqrt(S[x, x]), best) - p.lam * p.cost[x], abs=1e-9)


def test_q2_at_least_q1_and_equal_decision_without_correlation():
    p = problem(n=12)
    mu, S, vis = _state(p, 3)
    _, q1 = la.q1_all(mu, S, vis, p.cost, p.lam, p.noise_var_model)
    _, q2 = la.q2_all(mu, S, vis, p.cost, p.lam, p.noise_var_model)
    assert np.all(q2 >= q1 - 5e-3)   # an extra option cannot hurt
    # independent listings, no noise: two-step and one-step agree on stop/continue
    for seed in range(15):
        p = problem(seed, n=10, noise_sd=0.0, world_independent=True, model_independent=True)
        mu, S, vis = _state(p, 3)
        _, q1 = la.q1_all(mu, S, vis, p.cost, p.lam, 0.0)
        _, q2 = la.q2_all(mu, S, vis, p.cost, p.lam, 0.0)
        if abs(q1.max()) > 2e-2 and abs(q2.max()) > 2e-2:
            assert (q1.max() > 0) == (q2.max() > 0)


# ---------- trajectories ----------
@pytest.mark.parametrize("sel", SELECTORS)
def test_hindsight_bounds_every_stopping_rule(sel):
    for seed in range(5):
        tr = run_trajectory(problem(seed, n=30), sel)
        top = tr.V[stopping.hindsight(tr) - 1]
        taus = [stopping.cost_aware(tr, k, m) for k in ("latent", "kg") for m in (1, 3)]
        taus += [stopping.immediate(tr), stopping.fixed(tr, 5),
                 stopping.cost_aware(tr, "kg", budget_hours=6.0)]
        for tau in taus:
            assert 1 <= tau <= 30 and tr.V[tau - 1] <= top + 1e-12
        assert sorted(tr.order) == list(range(30))
        assert np.allclose(tr.hours, np.cumsum(problem(seed, n=30).cost[tr.order]))


def test_budget_cap_and_determinism():
    tr = run_trajectory(problem(3), "pbgi:kg")
    tau = stopping.cost_aware(tr, "kg", budget_hours=4.0)
    assert tau == 1 or tr.hours[tau - 1] <= 4.0
    tr2 = run_trajectory(problem(3), "pbgi:kg")
    assert np.array_equal(tr.order, tr2.order) and np.array_equal(tr.V, tr2.V)


def test_recommendation_is_a_visited_listing():
    tr = run_trajectory(problem(4, n=20), "ei")
    for t in range(20):
        assert tr.rec[t] in tr.order[: t + 1]


def test_real_pool_loads_and_runs():
    p = make_problem(Config(pool="moi", n=40), np.random.default_rng(0))
    assert 6000 <= p.raw["rent"].min() and p.raw["rent"].max() <= 35000
    assert len(set(p.raw["district"])) > 3 and np.all(p.cost > 0.7)
    assert np.all(np.linalg.eigvalsh(p.K_model) > 0)
    tr = run_trajectory(p, "pbgi:kg")
    assert tr.V[stopping.cost_aware(tr, "kg") - 1] <= tr.V.max()
