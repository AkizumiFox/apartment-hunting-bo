"""E1 selection x stopping, E2 reference points, E3 rating noise.

All comparisons are paired: every policy sees the same pool, the same hidden
utilities and the same pre-drawn rating noise for each simulated user.
"""
import sys

import numpy as np

from common import BUDGETS, at, base_config, budgets, run_users, save
from housebo import stopping as st
from housebo.metrics import best_fixed_k, fmt, mean_se, paired

KMAX = 25


def fixed_matrix(trajs):
    return np.array([t.V[:KMAX] for t in trajs])


def row(name, V, n, hrs):
    return dict(policy=name, V=mean_se(V), visits=mean_se(n), hours=mean_se(hrs))


def show(rows):
    print(f"{'policy':44s} {'V (mean ± se)':>18s} {'visits':>8s} {'hours':>7s}")
    for r in rows:
        print(f"{r['policy']:44s} {r['V'][0]:9.3f} ± {r['V'][1]:.3f} "
              f"{r['visits'][0]:8.2f} {r['hours'][0]:7.2f}")


def stop_stats(trajs, taus):
    taus = np.asarray(taus)
    return at(trajs, taus), taus, at(trajs, taus, "hours")


def e1(tag="e1", **kw):
    """Selection x stopping at the three budgets. Fixed K is tuned on training users."""
    out = {}
    for b in BUDGETS:
        info = budgets()["budgets"][str(b)]
        cfg = base_config(lam=info["lam"], **kw)
        T = run_users(cfg, ["ei", "pbgi:kg"])
        rows, V = [], {}
        k = info["k_gp"]
        for sel in T:
            tau = [st.cost_aware(t, "kg") for t in T[sel]]
            V[sel, "stop"], n, h = stop_stats(T[sel], tau)
            rows.append(row(f"{sel} + cost-aware stop", V[sel, "stop"], n, h))
            V[sel, "fixed"], n, h = stop_stats(T[sel], [k] * len(tau))
            rows.append(row(f"{sel} + fixed K={k} (tuned on other users)", V[sel, "fixed"], n, h))
        print(f"\n== E1: selection x stopping, budget {b} h (lambda={cfg.lam:.4f}) ==")
        show(rows)
        d = {
            "stop effect | EI selection": paired(V["ei", "stop"], V["ei", "fixed"]),
            "stop effect | PBGI selection": paired(V["pbgi:kg", "stop"], V["pbgi:kg", "fixed"]),
            "selection effect (PBGI - EI) | cost-aware stop": paired(V["pbgi:kg", "stop"], V["ei", "stop"]),
            "selection effect (PBGI - EI) | fixed K": paired(V["pbgi:kg", "fixed"], V["ei", "fixed"]),
        }
        for kk, v in d.items():
            print(f"  {kk:50s} {fmt(v)}")
        out[str(b)] = dict(lam=cfg.lam, rows=rows, paired=d)
    save(tag, out)


def synthetic_check():
    """Same comparison on fully synthetic pools (robustness to the pool generator)."""
    e1(tag="e1_synthetic", pool="synthetic")


def e2():
    cfg = base_config()
    T = run_users(cfg, ["pbgi:kg", "prior", "random"])
    main = T["pbgi:kg"]
    rows, V = [], {}

    def add(name, trajs, taus):
        V[name], n, h = stop_stats(trajs, taus)
        rows.append(row(name, V[name], n, h))

    add("PBGI + hindsight (per-trial upper bound)", main, [st.hindsight(t) for t in main])
    for m in (1, 2, 3):
        add(f"PBGI + cost-aware stop (min visits {m})", main, [st.cost_aware(t, "kg", m) for t in main])
    k, _ = best_fixed_k(fixed_matrix(main))
    add(f"PBGI + best fixed K={k} in hindsight (one K for all)", main, [k] * len(main))
    add("PBGI + immediate (1 visit)", main, [1] * len(main))
    for sel, label in (("prior", "listing-rating order"), ("random", "random order")):
        kb, _ = best_fixed_k(fixed_matrix(T[sel]))
        for kk in sorted({3, 5, kb}):
            tag = " (best K)" if kk == kb else ""
            add(f"{label}, fixed {kk}{tag}", T[sel], [kk] * len(main))
    # Hindsight assertion: no stopping time on the same trajectory can beat it.
    top = V["PBGI + hindsight (per-trial upper bound)"]
    for name in V:
        if name.startswith("PBGI"):
            assert np.all(V[name] <= top + 1e-12), name
    print("\n== E2: reference points ==")
    show(rows)
    ref = "PBGI + cost-aware stop (min visits 1)"
    d = {f"[main] - [{name}]": paired(V[ref], V[name]) for name in V if name != ref}
    for k, v in d.items():
        print(f"  {k:95s} {fmt(v)}")
    save("e2", dict(rows=rows, paired=d))


def e3():
    out = {}
    print("\n== E3: rating noise, original index (s = sigma) vs noise-adjusted (s = sigma~) ==")
    print(f"{'noise sd':>8s} | {'visits lat':>10s} {'visits kg':>10s} | {'V lat':>8s} {'V kg':>8s} | "
          f"{'V kg - V lat':>16s} | {'gap to hindsight lat / kg':>26s} | {'kg stop - best fixed K':>22s}")
    for sd in (0.0, 0.3, 0.65, 1.0, 1.4):
        cfg = base_config(noise_sd=sd)
        T = run_users(cfg, ["pbgi:latent", "pbgi:kg"])
        res = {}
        for kind in ("latent", "kg"):
            tr = T[f"pbgi:{kind}"]
            V, n, h = stop_stats(tr, [st.cost_aware(t, kind) for t in tr])
            hs = at(tr, [st.hindsight(t) for t in tr])
            k, Vk = best_fixed_k(fixed_matrix(tr))
            res[kind] = dict(V=V, n=n, hours=h, gap=hs - V, Vk=Vk, k=k)
        d = paired(res["kg"]["V"], res["latent"]["V"])
        dn = paired(res["kg"]["n"], res["latent"]["n"])
        dk = paired(res["kg"]["V"], res["kg"]["Vk"])
        print(f"{sd:8.2f} | {res['latent']['n'].mean():10.2f} {res['kg']['n'].mean():10.2f} | "
              f"{res['latent']['V'].mean():8.3f} {res['kg']['V'].mean():8.3f} | {fmt(d):>16s} | "
              f"{res['latent']['gap'].mean():12.3f} / {res['kg']['gap'].mean():.3f} | "
              f"{fmt(dk):>16s} (K={res['kg']['k']})")
        out[str(sd)] = dict(
            V_latent=mean_se(res["latent"]["V"]), V_kg=mean_se(res["kg"]["V"]),
            visits_latent=mean_se(res["latent"]["n"]), visits_kg=mean_se(res["kg"]["n"]),
            hours_latent=mean_se(res["latent"]["hours"]), hours_kg=mean_se(res["kg"]["hours"]),
            V_kg_minus_latent=d, visits_kg_minus_latent=dn,
            gap_to_hindsight_latent=mean_se(res["latent"]["gap"]),
            gap_to_hindsight_kg=mean_se(res["kg"]["gap"]),
            kg_stop_minus_best_fixed_k=dk, best_k=res["kg"]["k"])
    save("e3", out)


if __name__ == "__main__":
    which = sys.argv[1:] or ["e1", "e2", "e3", "synthetic_check"]
    for w in which:
        globals()[w]()
