"""Main result in the user's units: hours spent viewing vs. quality of the flat signed.

Quality = true (hidden) satisfaction with the signed flat. Fixed-K policies trace a curve in
this plane; the cost-aware rule is one point per budget. "Same quality, fewer hours" is read
off by linear interpolation of the fixed-K curve of the SAME visiting order at the rule's
quality; "same hours, better flat" by interpolating it at the rule's hours. Confidence
intervals: paired bootstrap over simulated users (2000 resamples). Lambda and the tuned
fixed K come from training users (experiments/budgets.py); everything here is test users.
"""
import numpy as np

from common import BUDGETS, at, base_config, budgets, run_users, save
from housebo import stopping as st
from housebo.metrics import fmt, mean_se, paired

KMAX = 25


def arrays(trajs, lam):
    H = np.array([t.hours[:KMAX] for t in trajs])
    Q = np.array([t.V[:KMAX] for t in trajs]) + lam * H
    return H, Q


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


if __name__ == "__main__":
    B = budgets()["budgets"]
    rng = np.random.default_rng(7)
    out = {}
    print(f"{'budget':>6s} {'lambda':>7s} | {'rule: hours':>11s} {'quality':>8s} {'viewings':>8s} | "
          f"{'same quality: hours saved [95% CI]':>36s} | {'same hours: quality gained [95% CI]':>36s} | "
          f"{'vs tuned fixed K: V':>20s} {'hours':>14s}")
    for b in BUDGETS:
        lam, k = B[str(b)]["lam"], B[str(b)]["k_gp"]
        cfg = base_config(lam=lam)
        T = run_users(cfg, ["pbgi:kg", "prior"])
        Ti = run_users(cfg.with_(model_independent=True), ["pbgi:kg"])["pbgi:kg"]
        main = T["pbgi:kg"]
        tau = np.array([st.cost_aware(t, "kg") for t in main])
        h = at(main, tau, "hours"); q = at(main, tau) + lam * h
        H, Q = arrays(main, lam)
        Hp, Qp = arrays(T["prior"], lam)
        taui = np.array([st.cost_aware(t, "kg") for t in Ti])
        hi = at(Ti, taui, "hours"); qi = at(Ti, taui) + lam * hi

        def gaps(idx):
            ch, cq = H[idx].mean(0), Q[idx].mean(0)            # fixed-K curve (quality increases in K)
            cq = np.maximum.accumulate(cq)
            hours_same_q = np.interp(q[idx].mean(), cq, ch)
            return hours_same_q - h[idx].mean(), q[idx].mean() - np.interp(h[idx].mean(), ch, cq)

        n = len(main)
        boot = np.array([gaps(rng.integers(0, n, n)) for _ in range(2000)])
        saved, gained = gaps(np.arange(n))
        dV = paired(at(main, tau), np.array([t.V[k - 1] for t in main]))
        dH = paired(h, np.array([t.hours[k - 1] for t in main]))
        dQ = paired(q, np.array([t.V[k - 1] + lam * t.hours[k - 1] for t in main]))
        d5 = dict(quality=paired(q, Qp[:, 4]), hours=paired(h, Hp[:, 4]))
        di = dict(quality=paired(q, qi), hours=paired(h, hi))
        print(f"{b:6d} {lam:7.4f} | {h.mean():11.2f} {q.mean():8.3f} {tau.mean():8.2f} | "
              f"{saved:10.2f} h [{ci(boot[:, 0])[0]:.2f}, {ci(boot[:, 0])[1]:.2f}]{'':10s} | "
              f"{gained:+10.3f} [{ci(boot[:, 1])[0]:+.3f}, {ci(boot[:, 1])[1]:+.3f}]{'':8s} | "
              f"{fmt(dV):>20s} {fmt(dH, 2):>14s}")
        out[str(b)] = dict(
            lam=lam, rule=dict(hours=mean_se(h), quality=mean_se(q), viewings=mean_se(tau)),
            hours_saved_same_quality=dict(est=float(saved), ci95=ci(boot[:, 0])),
            quality_gained_same_hours=dict(est=float(gained), ci95=ci(boot[:, 1])),
            vs_tuned_fixed_k=dict(k=k, V=dV, quality=dQ, hours=dH),
            vs_five_best_looking=d5, vs_no_cross_learning=di,
            no_cross_learning=dict(hours=mean_se(hi), quality=mean_se(qi), viewings=mean_se(taui)),
            curve_gp=dict(hours=H.mean(0).tolist(), quality=Q.mean(0).tolist()),
            curve_listing=dict(hours=Hp.mean(0).tolist(), quality=Qp.mean(0).tolist()))
    print("\nrule minus alternative (paired, mean ± s.e.):")
    for b in BUDGETS:
        r = out[str(b)]
        print(f"  budget {b:2d} h | 5 best-looking: quality {fmt(r['vs_five_best_looking']['quality'])}, hours "
              f"{fmt(r['vs_five_best_looking']['hours'], 2)} | no cross-listing learning: quality "
              f"{fmt(r['vs_no_cross_learning']['quality'])}, hours {fmt(r['vs_no_cross_learning']['hours'], 2)}")
    save("frontier", out)
