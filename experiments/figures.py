"""Figures for the proposal: results overview (fig_main) and one run of the loop (fig_demo)."""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from common import FIGURES, RESULTS, SEED, base_config, budgets, run_users
from housebo import stopping as st
from housebo.policies import run_trajectory
from housebo.simulator import make_problem

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#5598e7", "#0d366b"])

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID,
    "grid.linewidth": 0.5, "font.family": "DejaVu Sans", "pdf.fonttype": 42,
})


def fig_main():
    fr = json.loads((RESULTS / "frontier.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.35), gridspec_kw=dict(width_ratios=[1.75, 1]))

    # (a) hours vs quality frontier
    a = ax[0]
    mid = fr["9"]
    K = 16
    a.plot(mid["curve_listing"]["hours"][:K], mid["curve_listing"]["quality"][:K], "-o", color=MUTED,
           lw=1.1, ms=2.2, label="best-looking first, fixed number $K$ of viewings")
    a.plot(mid["curve_gp"]["hours"][:K], mid["curve_gp"]["quality"][:K], "-o", color=BLUE, lw=1.1, ms=2.2,
           label="GP order, fixed number $K$")
    for i, (b, r) in enumerate(fr.items()):
        for key, col, mk, ms in (("no_cross_learning", VIOLET, "s", 4.5), ("rule", ORANGE, "o", 6)):
            (h, hs), (q, qs) = r[key]["hours"], r[key]["quality"]
            a.errorbar(h, q, xerr=1.96 * hs, yerr=1.96 * qs, fmt=mk, color=col, ms=ms, mec="white", mew=0.9,
                       elinewidth=0.8, zorder=5,
                       label=None if i else ("cost-aware stop (ours), 95% CI" if key == "rule"
                                             else "same rule, no cross-listing learning"))
        (h, _), (q, _) = r["rule"]["hours"], r["rule"]["quality"]
        a.annotate(f"budget {b} h", (h, q), (h - 0.3, q + 0.05), ha="right", fontsize=6.3, color=INK)
    (h, _), (q, _) = mid["rule"]["hours"], mid["rule"]["quality"]
    hx = h + mid["hours_saved_same_quality"]["est"]
    a.annotate("", (hx, q), (h + 0.35, q), arrowprops=dict(arrowstyle="->", color=INK, lw=0.6, ls="--"))
    a.set_xlabel("mean hours spent viewing"); a.set_ylabel("mean rating of the flat signed")
    a.set_xlim(0, 21); a.set_ylim(8.25, 9.25)
    a.grid(axis="y")
    hnd, lab = a.get_legend_handles_labels()
    order = [2, 3, 1, 0] if len(hnd) == 4 else range(len(hnd))
    a.legend([hnd[i] for i in order], [lab[i] for i in order], frameon=False, loc="lower right", handlelength=1.5)
    a.set_title("(a) Hours vs. quality of the flat signed (1000 simulated users)", loc="left")

    # (b) noise
    e3 = json.loads((RESULTS / "e3.json").read_text())
    sds = sorted(float(k) for k in e3)
    b = ax[1]
    for key, col, lab in (("visits_latent", ORANGE, "original index"),
                          ("visits_kg", BLUE, "noise-adjusted index")):
        m = np.array([e3[str(s_)][key][0] for s_ in sds]); se = np.array([e3[str(s_)][key][1] for s_ in sds])
        b.plot(sds, m, "-o", color=col, lw=1.4, ms=3)
        b.fill_between(sds, m - 1.96 * se, m + 1.96 * se, color=col, alpha=0.15, lw=0)
        b.text(sds[-1], m[-1] + (0.5 if key == "visits_latent" else -1.0), lab, color=INK, ha="right", fontsize=6.5)
    b.set_xlabel("rating noise (sd, points)"); b.set_ylabel("viewings before stopping")
    b.set_ylim(4, 12.5); b.grid(axis="y")
    b.set_title("(b) Noisy ratings", loc="left")
    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig(FIGURES / "fig_main.pdf"); fig.savefig(FIGURES / "fig_main.png", dpi=220)


ROMAN = {"中正區": "Zhongzheng", "大同區": "Datong", "中山區": "Zhongshan", "松山區": "Songshan",
         "大安區": "Da'an", "萬華區": "Wanhua", "信義區": "Xinyi", "士林區": "Shilin",
         "北投區": "Beitou", "內湖區": "Neihu", "南港區": "Nangang", "文山區": "Wenshan"}


def fig_demo():
    cfg = base_config(n=40)
    # Selection rule (stated in the caption): first simulated user whose search lasts 4-6
    # viewings and who does not sign the first flat viewed.
    for u in range(500):
        prob = make_problem(cfg, np.random.default_rng([SEED, u]))
        tr = run_trajectory(prob, "pbgi:kg", snapshots=True)
        tau = st.cost_aware(tr, "kg")
        if 4 <= tau <= 6 and tr.rec[tau - 1] != tr.order[0]:
            break
    S = tr.snapshots
    moved = lambda k: (np.abs(S[k]["mu"] - S[k - 1]["mu"]) >= 0.25) & ~S[k]["visited"]
    k = max(range(1, tau), key=lambda j: moved(j).sum())      # the viewing that taught the most
    rent = prob.raw["rent"] / 1000
    dist = np.array(prob.raw["district"])
    names = sorted(set(dist), key=lambda d: prob.cost[dist == d][0])
    row = np.array([names.index(d) for d in dist], float)
    size = prob.raw["size"]
    y = row + 0.62 * (size - size.min()) / (size.max() - size.min()) - 0.31
    vmin, vmax = 4.0, 9.5
    fig = plt.figure(figsize=(7.1, 2.55))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.3, 0.36, 1, 1.08], wspace=0.05, left=0.005, right=0.995,
                          top=0.9, bottom=0.2)

    # ---- (a) what the user sees
    a = fig.add_subplot(gs[0]); a.set_xlim(0, 1); a.set_ylim(0, 1); a.axis("off")
    s = S[k]; un = np.flatnonzero(~s["visited"]); last = tr.order[k - 1]
    best_i = np.flatnonzero(s["visited"])[np.argmax(s["mu"][s["visited"]])]
    best = s["mu"][best_i]
    bud = budgets()["budgets"]["9"]
    a.add_patch(plt.Rectangle((0.0, 0.0), 1.0, 1.0, fc="#fcfcfb", ec=MUTED, lw=0.6))
    a.text(0.03, 0.94, "Viewing budget", fontsize=6, color=MUTED, va="center")
    a.plot([0.37, 0.66], [0.94, 0.94], color=GRID, lw=2, solid_capstyle="round")
    a.plot([0.37, 0.49], [0.94, 0.94], color=BLUE, lw=2, solid_capstyle="round")
    a.plot(0.49, 0.94, "o", color=BLUE, ms=4.5, mec="white", mew=0.8)
    a.text(0.70, 0.94, "about 9 h", fontsize=6.5, weight="bold", va="center")
    a.text(0.03, 0.875, f"expect \u2248{bud['train_viewings']:.0f} viewings, final flat \u2248{bud['train_quality']:.1f}",
           fontsize=6, color=MUTED, va="center")
    a.plot([0.03, 0.97], [0.835, 0.835], color=GRID, lw=0.6)
    a.text(0.03, 0.78, f"You rated flat #{last + 1} ({ROMAN[dist[last]]}) {prob.observe(last):.1f}", fontsize=7, weight="bold", va="center")
    a.text(0.03, 0.715, f"page score was {prob.p[last]:.1f}; {moved(k).sum()} similar listings re-estimated", fontsize=6, color=MUTED, va="center")
    a.text(0.03, 0.65, f"Best so far: flat #{best_i + 1}, estimated {best:.1f}", fontsize=6.5, va="center")
    cols = [0.03, 0.50, 0.72, 0.88]
    for cx, h in zip(cols, ["Not yet viewed", "estimate", "fair v.", "trip"]):
        a.text(cx, 0.575, h, fontsize=6, color=MUTED, va="center")
    a.plot([0.03, 0.97], [0.545, 0.545], color=GRID, lw=0.6)
    top = un[np.argsort(-s["g"][un])][:4]
    for r_, i in enumerate(top):
        yy = 0.485 - 0.08 * r_
        if r_ == 0:
            a.add_patch(plt.Rectangle((0.015, yy - 0.038), 0.97, 0.076, fc="#fdebe3", ec="none"))
        d_ = s["mu"][i] - S[k - 1]["mu"][i]
        arrow = "" if abs(d_) < 0.05 else (f"  \u25b2{d_:.1f}" if d_ > 0 else f"  \u25bc{-d_:.1f}")
        a.text(cols[0], yy, f"#{i + 1} {ROMAN[dist[i]]} {rent[i]:.1f}k", fontsize=6.3, va="center")
        a.text(cols[1], yy, f"{s['mu'][i]:.1f}{arrow}", fontsize=6.3, va="center")
        a.text(cols[2], yy, f"{s['g'][i]:.1f}", fontsize=6.3, va="center", weight="bold" if r_ == 0 else "normal")
        a.text(cols[3], yy, f"{prob.cost[i]:.1f}h", fontsize=6.3, va="center")
    nxt = top[0]
    a.add_patch(plt.Rectangle((0.015, 0.03), 0.97, 0.13, fc="#eef4fd", ec="none"))
    a.text(0.03, 0.12, f"Keep looking: view #{nxt + 1} next.", fontsize=6.8, weight="bold", va="center")
    a.text(0.03, 0.06, f"Fair value {s['g'][nxt]:.1f} > best so far {best:.1f}.", fontsize=6.2, va="center")
    a.set_title(f"(a) What the user sees after viewing {k}", loc="left")

    # ---- (b), (c) the pool
    axes = [fig.add_subplot(gs[2])]
    axes.append(fig.add_subplot(gs[3], sharey=axes[0]))
    for a, kk in zip(axes, (k, tau)):
        s = S[kk]; un = ~s["visited"]
        sc = a.scatter(rent[un], y[un], c=s["mu"][un], s=5 + 22 * s["sd"][un] ** 2, cmap=SEQ,
                       vmin=vmin, vmax=vmax, edgecolors="white", linewidths=0.4, zorder=3)
        if kk == k:
            d_ = s["mu"] - S[k - 1]["mu"]
            for i in np.flatnonzero(moved(k)):
                a.text(rent[i] + 0.9, y[i], "\u25b2" if d_[i] > 0 else "\u25bc", fontsize=5.5,
                       color=INK, ha="left", va="center", zorder=7)
        for j, i in enumerate(tr.order[:kk]):
            a.scatter(rent[i], y[i], s=40, facecolors="white", edgecolors=INK, linewidths=0.8, zorder=4)
            a.text(rent[i], y[i], str(j + 1), ha="center", va="center", fontsize=5.5, color=INK, zorder=5)
        best = s["mu"][s["visited"]].max(); gmax = s["g"][un].max()
        if kk < tau:
            nx = np.flatnonzero(un)[np.argmax(s["g"][un])]
            a.scatter(rent[nx], y[nx], s=90, marker="*", color=ORANGE, edgecolors="white", linewidths=0.6, zorder=6)
            a.set_title(f"(b) After viewing {kk}", loc="left")
            a.text(0.5, -0.24, "\u25b2\u25bc moved by this viewing; \u2605 next", transform=a.transAxes,
                   ha="center", va="top", fontsize=6, color=INK)
        else:
            r = tr.rec[kk - 1]
            a.scatter(rent[r], y[r], s=115, facecolors="none", edgecolors=ORANGE, linewidths=1.4, zorder=6)
            a.set_title(f"(c) After {kk} viewings: stop", loc="left")
            a.text(0.5, -0.24, f"sign no. {list(tr.order[:kk]).index(r) + 1}: best so far {best:.1f} $\\geq$ top fair value {gmax:.1f}", transform=a.transAxes,
                   ha="center", va="top", fontsize=6, color=INK)
        a.grid(True, axis="x"); a.set_axisbelow(True)
        for r_ in range(len(names) - 1):
            a.axhline(r_ + 0.5, color=GRID, lw=0.5)
        a.set_xlabel("rent (kNT\\$ / month)")
        a.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels([f"{ROMAN[d]} ({prob.cost[dist == d][0]:.1f} h)" for d in names], fontsize=5.6)
    axes[0].set_ylim(len(names) - 0.5, -0.5)
    plt.setp(axes[1].get_yticklabels(), visible=False)
    cb = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.015)
    cb.set_label("estimated rating (0-10)", fontsize=6.5); cb.outline.set_visible(False)
    fig.savefig(FIGURES / "fig_demo.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGURES / "fig_demo.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    info = dict(user=u, tau=int(tau), shown_viewing=int(k), n_reestimated=int(moved(k).sum()),
                order=[int(i) + 1 for i in tr.order[:tau]],
                districts=[ROMAN[dist[i]] for i in tr.order[:tau]],
                ratings=[round(prob.observe(int(i)), 2) for i in tr.order[:tau]],
                hours=float(tr.hours[tau - 1]), signed=int(tr.rec[tau - 1]) + 1)
    (RESULTS / "demo.json").write_text(json.dumps(info, indent=2))
    print(info)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    for w in sys.argv[1:] or ["fig_main", "fig_demo"]:
        globals()[w]()
