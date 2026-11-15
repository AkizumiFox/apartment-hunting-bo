"""Use the tool yourself: a real person in the loop, on real Taipei listings.

    uv run python experiments/interactive.py                 # 12 leases sampled from the MOI data
    uv run python experiments/interactive.py --csv mine.csv  # your own shortlist
    uv run python experiments/interactive.py --auto          # scripted answers (smoke test)

Your own CSV needs the columns: name,district,rent,size_ping,age_years,elevator,furnished
(district in Chinese, e.g. 大安區; elevator/furnished are 0 or 1).

Flow: (1) rate every listing page 0-10, (2) say how you trade hours for quality,
(3) the tool proposes the next viewing and shows each listing's fair value,
(4) after each real viewing you enter one rating, (5) it tells you when to stop.
The session is saved to results/sessions/ so it can be shown as a real run.
"""
import argparse
import csv
import json
import time

import numpy as np

from common import RESULTS
from housebo import gp
from housebo.index import ei, gittins, scale
from housebo.simulator import CENTROID, HOME, MOI_CSV, MOI_FEATURES, Kernel

NOISE_SD = 0.65  # assumed sd of one person's rating noise


def load(path, n, seed):
    with open(path or MOI_CSV, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["district"] in CENTROID]
    if path is None:
        rng = np.random.default_rng(seed)
        rows = [rows[i] for i in rng.choice(len(rows), size=n, replace=False)]
    return rows


def build(rows):
    loc = np.array([[(CENTROID[r["district"]][1] - HOME[1]) * 111.32 * np.cos(np.radians(HOME[0])),
                     (CENTROID[r["district"]][0] - HOME[0]) * 110.57] for r in rows])
    dist = np.linalg.norm(loc, axis=1)
    raw = {"rent": [float(r["rent"]) for r in rows], "size": [float(r["size_ping"]) for r in rows],
           "commute": 10 + 4 * dist, "age": [float(r["age_years"]) for r in rows],
           "elevator": [float(r["elevator"]) for r in rows],
           "furnished": [float(r["furnished"]) for r in rows]}
    Z = np.column_stack([raw[k] for k in MOI_FEATURES])
    sd = Z.std(0)
    Z = (Z - Z.mean(0)) / np.where(sd > 0, sd, 1.0)
    cost = 2 * dist / 12.0 + 0.75
    return Kernel().matrix(Z, loc), cost


def describe(i, r, cost):
    name = r.get("name") or f"listing {i + 1}"
    return (f"[{i + 1:2d}] {name}: {r['district']}, NT${int(float(r['rent'])):,}/mo, "
            f"{float(r['size_ping']):.1f} ping, {int(float(r['age_years']))} yr old, "
            f"{'elevator' if int(float(r['elevator'])) else 'walk-up'}, "
            f"{'furnished' if int(float(r['furnished'])) else 'unfurnished'}, "
            f"viewing ~{cost[i]:.1f} h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--auto", action="store_true")
    a = ap.parse_args()
    rows = load(a.csv, a.n, a.seed)
    K, cost = build(rows)
    n = len(rows)
    auto = np.random.default_rng(1)

    def ask(prompt, lo, hi, fake):
        if a.auto:
            print(f"{prompt}{fake:.1f}  (auto)")
            return float(fake)
        while True:
            s = input(prompt).strip()
            try:
                v = float(s)
                if lo <= v <= hi:
                    return v
            except ValueError:
                pass
            print(f"  please enter a number between {lo} and {hi}")

    print("\nStep 1. Rate each listing from its page alone (0 = no way, 10 = perfect).\n")
    p = np.empty(n)
    for i, r in enumerate(rows):
        print(describe(i, r, cost))
        p[i] = ask("     your rating: ", 0, 10, np.clip(auto.normal(6, 1.2), 0, 10))
    hrs = ask("\nStep 2. To end up ONE point happier with the flat you sign, how many extra hours "
              "of viewing would you accept? ", 0.5, 500, 50)
    lam = 1.0 / hrs
    print(f"   -> one hour costs you {lam:.3f} rating points.\n")

    mu, Sigma, nv = p.copy(), K.copy(), NOISE_SD**2
    visited = np.zeros(n, bool)
    log, total = [], 0.0
    while not visited.all():
        un = np.flatnonzero(~visited)
        s = scale(np.diag(Sigma)[un], nv, "kg")
        g = gittins(mu[un], s, lam * cost[un])
        best = mu[visited].max() if visited.any() else None
        print("Unvisited listings: predicted rating ± uncertainty, fair value")
        for j in np.argsort(-g):
            i = un[j]
            print(f"   [{i + 1:2d}] {mu[i]:4.1f} ± {np.sqrt(Sigma[i, i]):.1f}   fair value {g[j]:4.1f}"
                  f"   ({cost[i]:.1f} h)")
        nxt = int(un[np.argmax(g)])
        if best is not None and g.max() <= best:
            b = int(np.flatnonzero(visited)[np.argmax(mu[visited])])
            print(f"\nSTOP. Your best visited flat is [{b + 1}] (estimated {best:.1f}); no remaining "
                  f"viewing is worth its hours (highest fair value {g.max():.1f}).")
            if a.auto or input("Stop here? [Y/n] ").strip().lower() != "n":
                break
        elif best is not None:
            gain = ei(mu[nxt], scale(Sigma[nxt, nxt], nv, "kg"), best)
            print(f"\nBest so far {best:.1f}. Suggest viewing [{nxt + 1}] next: expected gain "
                  f"{gain:.2f} points for {cost[nxt]:.1f} h (worth {lam * cost[nxt]:.2f} to you).")
        else:
            print(f"\nSuggest viewing [{nxt + 1}] first.")
        print(describe(nxt, rows[nxt], cost))
        y = ask(f"   after viewing [{nxt + 1}], your rating (0-10): ", 0, 10,
                np.clip(p[nxt] + auto.normal(0, 1.1), 0, 10))
        before = mu.copy()
        mu, Sigma = gp.update(mu, Sigma, nxt, y, nv)
        visited[nxt] = True
        total += cost[nxt]
        moved = [(i, mu[i] - before[i]) for i in np.flatnonzero(~visited) if abs(mu[i] - before[i]) >= 0.1]
        if moved:
            print("   similar listings re-estimated: " +
                  ", ".join(f"[{i + 1}] {d:+.1f}" for i, d in sorted(moved, key=lambda t: -abs(t[1]))[:6]))
        log.append(dict(listing=nxt + 1, rating=y, hours=float(cost[nxt])))
        print()
    vis = np.flatnonzero(visited)
    b = int(vis[np.argmax(mu[vis])])
    print(f"\nDone: {len(log)} viewings, {total:.1f} h. Sign [{b + 1}].")
    out = RESULTS / "sessions"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"session_{'auto' if a.auto else time.strftime('%Y%m%d_%H%M%S')}.json"
    f.write_text(json.dumps(dict(listings=rows, listing_ratings=p.tolist(), lam=lam, visits=log,
                                 signed=b + 1, total_hours=total, human=not a.auto),
                            ensure_ascii=False, indent=2))
    print(f"session saved to {f}")


if __name__ == "__main__":
    main()
