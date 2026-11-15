"""Turn an expected viewing budget (hours) into the cost weight lambda.

The user never states an exchange rate. They set an expected budget B; we search for the
lambda whose simulated searches (main method, the user's prior) spend B hours on average.
This is the expected-budget setting of Xie et al. (2024), where lambda is a Lagrange
multiplier. Calibration uses TRAINING users only; the same users tune the fixed-K baselines.

Also reports how many viewings follow from lambda = 0.05, 0.1, 0.2 as the size of the
on-site surprise varies (is "one or two viewings" an artefact of a small surprise scale?).
"""
import numpy as np

from common import BUDGETS, TRAIN_SEED, run_users, save
from housebo import stopping as st
from housebo.simulator import Config, Kernel

N_TRAIN = 600


def spend(lam, cfg=Config()):
    tr = run_users(cfg.with_(lam=lam), ["pbgi:kg"], n_users=N_TRAIN, seed=TRAIN_SEED)["pbgi:kg"]
    tau = np.array([st.cost_aware(t, "kg") for t in tr])
    return float(np.mean([t.hours[k - 1] for t, k in zip(tr, tau)])), float(tau.mean()), tr


def calibrate(hours):
    lo, hi = np.log(1e-4), np.log(1.0)          # mean hours decrease in lambda
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if spend(float(np.exp(mid)))[0] > hours:
            lo = mid
        else:
            hi = mid
    lam = float(np.exp(0.5 * (lo + hi)))
    h, n, tr = spend(lam)
    # fixed-K baselines tuned on the same training users
    def best_k(sel):
        T = run_users(Config(lam=lam), [sel], n_users=N_TRAIN, seed=TRAIN_SEED)[sel]
        return int(np.argmax(np.mean([t.V[:25] for t in T], axis=0))) + 1
    V = np.array([t.V[st.cost_aware(t, "kg") - 1] for t in tr])
    q = V + lam * np.array([t.hours[st.cost_aware(t, "kg") - 1] for t in tr])
    return dict(lam=lam, train_hours=h, train_viewings=n, train_quality=float(q.mean()),
                hours_per_point=1 / lam, k_gp=best_k("pbgi:kg"), k_listing=best_k("prior"))


if __name__ == "__main__":
    k = Kernel()
    sd = float(np.sqrt(k.s_lin**2 * 6 + k.s_sp**2 + k.s_idio**2))
    out = {"surprise_sd_main": sd, "budgets": {}}
    print(f"simulator on-site surprise sd = {sd:.2f}, rating noise sd = {Config().noise_sd}")
    for B in BUDGETS:
        out["budgets"][str(B)] = r = calibrate(B)
        print(f"budget {B:2d} h -> lambda {r['lam']:.4f} ({r['hours_per_point']:.0f} h per point as a shadow "
              f"price), {r['train_viewings']:.1f} viewings, quality {r['train_quality']:.2f}; "
              f"tuned fixed K: GP order {r['k_gp']}, listing order {r['k_listing']}")
    print("\nmean viewings / hours by surprise sd and lambda (training users):")
    sweep = {}
    for target in (0.5, 1.0, sd, 2.5):
        row = {}
        for lam in (0.05, 0.1, 0.2):
            h, n, _ = spend(lam, Config(kernel=k.scaled(target / sd)))
            row[str(lam)] = dict(viewings=n, hours=h)
        sweep[f"{target:.2f}"] = row
        print(f"  surprise sd {target:.2f}: " + "   ".join(
            f"lam={l}: {v['viewings']:.1f} viewings, {v['hours']:.1f} h" for l, v in row.items()))
    out["scale_sweep"] = sweep
    save("budgets", out)
