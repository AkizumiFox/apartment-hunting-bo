"""E4 prior misspecification, E5 role of correlation, E6 two-step look-ahead,
E7 cost stress test, E8 delisting sensitivity.

Main method throughout: PBGI selection with the noise-adjusted (kg) scale and the
matching cost-aware stopping rule.
"""
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import N_USERS, SEED, at, base_config, run_users, save
from housebo import gp, lookahead as la, stopping as st
from housebo.metrics import best_fixed_k, fmt, mean_se, paired
from housebo.simulator import Kernel, make_problem

MAIN = "pbgi:kg"


def main_stats(cfg, kind="kg", sel=MAIN):
    tr = run_users(cfg, [sel])[sel]
    tau = np.array([st.cost_aware(t, kind) for t in tr])
    tau_h = np.array([st.hindsight(t) for t in tr])
    _, Vk = best_fixed_k(np.array([t.V[:25] for t in tr]))
    return dict(V=at(tr, tau), n=tau, hours=at(tr, tau, "hours"), n_h=tau_h,
                V_h=at(tr, tau_h), Vk=Vk)


# ---------------------------------------------------------------- E4
def e4():
    k = Kernel()
    big = Kernel(k.s_lin * 1.5, k.s_sp * 1.5, k.ell, k.s_idio * 1.5)
    small = Kernel(k.s_lin / 1.5, k.s_sp / 1.5, k.ell, k.s_idio / 1.5)
    # (label, world+misspecified model, same world with a correctly specified model)
    cases = [
        ("well specified", {}, {}),
        ("listings look 1 pt better than reality", dict(prior_shift=-1.0), dict(prior_shift=-1.0, model_knows_bias=True)),
        ("listings look 1 pt worse than reality", dict(prior_shift=1.0), dict(prior_shift=1.0, model_knows_bias=True)),
        ("glamorous listings disappoint (shrink 0.3)", dict(prior_shrink=0.3), dict(prior_shrink=0.3, model_knows_bias=True)),
        ("glamorous listings disappoint (shrink 0.6)", dict(prior_shrink=0.6), dict(prior_shrink=0.6, model_knows_bias=True)),
        ("surprises 1.5x larger than believed", dict(kernel=big, model_kernel=k), dict(kernel=big)),
        ("surprises 1.5x smaller than believed", dict(kernel=small, model_kernel=k), dict(kernel=small)),
        ("length scale believed 0.5 km (true 1.5)", dict(model_kernel=Kernel(ell=0.5)), {}),
        ("length scale believed 4.5 km (true 1.5)", dict(model_kernel=Kernel(ell=4.5)), {}),
        ("rating noise believed half its true value", dict(model_noise_sd=0.325), {}),
        ("rating noise believed twice its true value", dict(model_noise_sd=1.3), {}),
    ]
    print("\n== E4: prior misspecification (main method) ==")
    print(f"{'case':44s} {'visits':>7s} {'ref visits':>10s} {'visits - ref':>15s} "
          f"{'V - V(ref model)':>18s} {'V - best fixed K':>18s}")
    out, cache = {}, {}
    for label, mis, ref in cases:
        key = repr(sorted(ref.items()))
        if key not in cache:
            cache[key] = main_stats(base_config(**ref))
        a, b = main_stats(base_config(**mis)), cache[key]
        dn, dV, dk = paired(a["n"], b["n"]), paired(a["V"], b["V"]), paired(a["V"], a["Vk"])
        print(f"{label:44s} {a['n'].mean():7.2f} {b['n'].mean():10.2f} {fmt(dn, 2):>15s} "
              f"{fmt(dV):>18s} {fmt(dk):>18s}")
        out[label] = dict(visits=mean_se(a["n"]), ref_visits=mean_se(b["n"]),
                          visits_minus_ref=dn, V=mean_se(a["V"]), V_minus_ref=dV,
                          V_minus_best_fixed_k=dk, hindsight_visits=mean_se(a["n_h"]))
    save("e4", out)


# ---------------------------------------------------------------- E5
def e5():
    print("\n== E5: role of cross-listing correlation (same marginal variances) ==")
    out = {}
    for world_ind in (False, True):
        res = {mi: main_stats(base_config(world_independent=world_ind, model_independent=mi))
               for mi in (False, True)}
        w = "independent" if world_ind else "correlated"
        for mi, r in res.items():
            m = "independent" if mi else "correlated (GP)"
            print(f"world {w:12s} model {m:16s} V={r['V'].mean():.3f} ± {mean_se(r['V'])[1]:.3f} "
                  f"visits={r['n'].mean():.2f} hours={r['hours'].mean():.2f}")
        d = paired(res[False]["V"], res[True]["V"])
        dn = paired(res[False]["n"], res[True]["n"])
        print(f"  GP model - independent model | world {w}: V {fmt(d)}, visits {fmt(dn, 2)}")
        out[w] = dict(V_gp=mean_se(res[False]["V"]), V_indep=mean_se(res[True]["V"]),
                      visits_gp=mean_se(res[False]["n"]), visits_indep=mean_se(res[True]["n"]),
                      V_gp_minus_indep=d, visits_gp_minus_indep=dn)
    save("e5", out)


# ---------------------------------------------------------------- E6
def _e6_user(args):
    cfg, seed, u = args
    prob = make_problem(cfg, np.random.default_rng([seed, u]))
    n, nv, lam, cost = cfg.n, prob.noise_var_model, prob.lam, prob.cost
    from housebo.policies import run_trajectory
    tr = run_trajectory(prob, MAIN)
    tau = st.cost_aware(tr, "kg")
    # Replay the index policy's path; at every decision point ask Q1 and Q2.
    mu, Sigma, vis = prob.m.copy(), prob.K_model.copy(), np.zeros(n, bool)
    rec = dict(points=0, q1_stop_before=0, q2_stop_before=0)
    q1_at_stop = q2_at_stop = None
    for t in range(1, tau + 1):
        x = tr.order[t - 1]
        mu, Sigma = gp.update(mu, Sigma, x, prob.observe(x), nv)
        vis[x] = True
        if vis.all():
            break
        q1 = la.q1_all(mu, Sigma, vis, cost, lam, nv)[1].max()
        q2 = la.q2_all(mu, Sigma, vis, cost, lam, nv)[1].max()
        if t < tau:   # index rule says continue here
            rec["points"] += 1
            rec["q1_stop_before"] += q1 <= 0
            rec["q2_stop_before"] += q2 <= 0
        else:         # index rule says stop here
            q1_at_stop, q2_at_stop = q1, q2
    _, V1, h1 = la.run_lookahead_policy(prob, 1)
    o2, V2, h2 = la.run_lookahead_policy(prob, 2)
    o1 = la.run_lookahead_policy(prob, 1)[0]
    return dict(V0=tr.V[tau - 1], n0=tau, V1=V1, n1=len(o1), V2=V2, n2=len(o2),
                stopped=q1_at_stop is not None,
                q1_cont=bool(q1_at_stop is not None and q1_at_stop > 0),
                q2_cont=bool(q2_at_stop is not None and q2_at_stop > 0),
                q2_gain=float(q2_at_stop) if q2_at_stop is not None else np.nan, **rec)


def e6():
    settings = [("ell=0.5 km", dict(kernel=Kernel(ell=0.5))),
                ("ell=1.5 km (default)", {}),
                ("ell=4 km", dict(kernel=Kernel(ell=4.0))),
                ("ell=10 km", dict(kernel=Kernel(ell=10.0))),
                ("strong: s_sp=1.2, s_idio=0.4, ell=4", dict(kernel=Kernel(s_sp=1.2, s_idio=0.4, ell=4.0))),
                ("no correlation", dict(world_independent=True, model_independent=True)),
                ("default, no rating noise", dict(noise_sd=0.0)),
                ("default, N=24", dict(n=24))]
    print("\n== E6: index stop vs exact one-step (Q1) vs two-step look-ahead (Q2), N=16 ==")
    print(f"{'setting':38s} {'visits idx/Q1/Q2':>18s} | {'at index stop: Q1 / Q2 say continue':>36s} | "
          f"{'V(Q1)-V(idx)':>15s} {'V(Q2)-V(idx)':>15s} {'V(Q2)-V(Q1)':>15s}")
    out = {}
    for label, kw in settings:
        cfg = base_config(**{"n": 16, **kw})
        with ProcessPoolExecutor(max_workers=20) as ex:
            R = list(ex.map(_e6_user, [(cfg, SEED, u) for u in range(400)], chunksize=4))
        g = lambda k: np.array([r[k] for r in R], dtype=float)
        stopped = g("stopped").astype(bool)
        r = dict(visits_index=mean_se(g("n0")), visits_q1=mean_se(g("n1")), visits_q2=mean_se(g("n2")),
                 frac_q1_continue_at_index_stop=float(g("q1_cont")[stopped].mean()),
                 frac_q2_continue_at_index_stop=float(g("q2_cont")[stopped].mean()),
                 mean_q2_value_at_index_stop_when_continue=float(
                     np.nanmean(np.where(g("q2_cont") > 0, g("q2_gain"), np.nan))) if g("q2_cont").any() else 0.0,
                 frac_points_q1_would_stop_earlier=float(g("q1_stop_before").sum() / max(1, g("points").sum())),
                 frac_points_q2_would_stop_earlier=float(g("q2_stop_before").sum() / max(1, g("points").sum())),
                 V_q1_minus_index=paired(g("V1"), g("V0")), V_q2_minus_index=paired(g("V2"), g("V0")),
                 V_q2_minus_q1=paired(g("V2"), g("V1")))
        out[label] = r
        print(f"{label:38s} {g('n0').mean():5.2f}/{g('n1').mean():5.2f}/{g('n2').mean():5.2f} | "
              f"{r['frac_q1_continue_at_index_stop']:16.1%} / {r['frac_q2_continue_at_index_stop']:.1%} | "
              f"{fmt(r['V_q1_minus_index']):>15s} {fmt(r['V_q2_minus_index']):>15s} {fmt(r['V_q2_minus_q1']):>15s}")
    save("e6", out)


# ---------------------------------------------------------------- E7
def e7():
    print("\n== E7 (STRESS TEST, not a main result): 5 far listings cost ~10x and are far less certain ==")
    out = {}
    for lam in (0.02, 0.1):
        cfg = base_config(stress=True, lam=lam)
        sels = ["ei", "eipc:kg", "pbgi:kg"]
        T = run_users(cfg, sels)
        V = {s: at(T[s], [st.cost_aware(t, "kg") for t in T[s]]) for s in sels}
        H = {s: at(T[s], [st.cost_aware(t, "kg") for t in T[s]], "hours") for s in sels}
        print(f"lambda={lam}: " + "  ".join(f"{s}: V={V[s].mean():.3f} h={H[s].mean():.1f}" for s in sels))
        d = {f"pbgi - {s}": paired(V["pbgi:kg"], V[s]) for s in ("ei", "eipc:kg")}
        for k_, v in d.items():
            print(f"   {k_:16s} {fmt(v)}")
        out[str(lam)] = dict(V={s: mean_se(V[s]) for s in sels}, hours={s: mean_se(H[s]) for s in sels}, paired=d)
    save("e7", out)


# ---------------------------------------------------------------- E8
def e8():
    """Sensitivity only. After each further visit, a visited listing is taken by someone
    else with probability q. The stopping rule is not told. On stopping, the user signs the
    recommended listing if it is still available; otherwise the best surviving visited
    listing (picked by true utility, which is optimistic, and equally so for every rule).
    If nothing survives, the user keeps visiting along the same trajectory."""
    print("\n== E8 (sensitivity): visited listings get taken with probability q per further visit ==")
    cfg = base_config()
    tr = run_users(cfg, [MAIN])[MAIN]
    rng = np.random.default_rng(SEED + 8)
    out = {}
    for q in (0.0, 0.05, 0.15):
        probs = [make_problem(cfg, np.random.default_rng([SEED, u])) for u in range(len(tr))]
        life = [rng.geometric(q, size=cfg.n) if q > 0 else np.full(cfg.n, 10**9) for _ in tr]

        def value(t, prob, L, tau):
            while True:
                idx = np.arange(tau)
                alive = (tau - 1 - idx) < L[t.order[:tau]]   # visits made since listing was seen
                if alive.any() or tau == len(t.order):
                    break
                tau += 1
            seen = t.order[:tau][alive] if alive.any() else t.order[:tau]
            pick = t.rec[tau - 1] if t.rec[tau - 1] in seen else seen[np.argmax(prob.f[seen])]
            return prob.f[pick] - prob.lam * t.hours[tau - 1]

        Vs = np.array([value(t, p, L, st.cost_aware(t, "kg")) for t, p, L in zip(tr, probs, life)])
        Vk = np.array([[value(t, p, L, k) for k in range(1, 21)] for t, p, L in zip(tr, probs, life)])
        kb, Vkb = best_fixed_k(Vk)
        d = paired(Vs, Vkb)
        print(f"q={q:4.2f}: cost-aware stop V={Vs.mean():.3f}; best fixed K={kb} V={Vkb.mean():.3f}; "
              f"difference {fmt(d)}")
        out[str(q)] = dict(V_stop=mean_se(Vs), best_k=kb, V_best_fixed=mean_se(Vkb), diff=d)
    save("e8", out)


if __name__ == "__main__":
    for w in sys.argv[1:] or ["e4", "e5", "e6", "e7", "e8"]:
        globals()[w]()
