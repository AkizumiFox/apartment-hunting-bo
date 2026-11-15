# One more viewing, or sign? Apartment hunting as cost-aware BO over a finite pool

Course assignment "Bayesian Optimization in Everyday Life". Proposal: `proposal/main.pdf`.
Honest summary of every experiment: `FINDINGS.md`.

The optimiser chooses which listing to view next from a finite pool and advises when to stop.
A Gaussian process over the "on-site surprise" (visit rating minus listing-page rating) shares
what one viewing reveals with similar listings; a Pandora's-box Gittins index gives each
listing a fair value in rating points; the search stops when the best visited listing is at
least as good as every remaining fair value. Listings are real (1,332 registered Taipei studio
leases from the Ministry of the Interior's open rental records); people's ratings are simulated.
No rental site is scraped.

## Run

```
uv sync
uv run pytest                                   # 17 tests
uv run python experiments/prepare_moi.py        # rebuild data/taipei_studios.csv from the MOI files
uv run python experiments/interactive.py        # try it yourself: you rate, it recommends and says when to stop
uv run python experiments/budgets.py            # viewing budgets -> lambda (results/budgets.json)
uv run python experiments/e1_e2_e3.py           # selection x stopping, references, noise
uv run python experiments/frontier.py           # main result: hours vs. quality at three budgets
uv run python experiments/e4_to_e8.py           # misspecification, correlation, look-ahead (slow, ~10 min), stress, delisting
uv run python experiments/llm_raters.py         # optional: re-collect LLM ratings (1,200 `claude -p` calls, ~$8); data/llm_ratings.json is included
uv run python experiments/e9_llm.py             # the loop on 300 LLM raters (first 80, new 220, pooled)
uv run python experiments/figures.py            # figures/fig_main.*, figures/fig_demo.*
cd proposal && latexmk -pdf main.tex
```

Seeds are fixed; reruns reproduce `results/*.json`.

## Layout

| Path | What |
|---|---|
| `src/housebo/gp.py` | finite-pool GP posterior, rank-one updates |
| `src/housebo/index.py` | EI, Gittins index, original and noise-adjusted scale |
| `src/housebo/policies.py` | selection rules; full trajectories with stop flags |
| `src/housebo/stopping.py` | cost-aware stop, fixed K, immediate, hindsight, hour cap |
| `src/housebo/lookahead.py` | exact one-step value and two-step look-ahead |
| `src/housebo/simulator.py` | real or synthetic pools, simulated users, misspecification switches |
| `data/taipei_studios.csv` | cleaned real leases (no addresses); source: plvr.land.moi.gov.tw, Open Government Data License v1.0 |
| `experiments/` | one entry point per experiment group, plus figures |
| `figures/fig_demo.*` | one search, step by step, over real leases (district × rent) |

## Conventions worth knowing

- You must visit a listing to sign it; the recommendation is the visited listing with the
  highest posterior mean. The first visit always happens.
- `lam` is points per hour. Rating-noise variance is `noise_var` (Frazier et al. use lambda for
  noise; we do not).
- Hindsight = per-user best stopping time on the same trajectory, using the hidden truth.
  "Best fixed K" = one K for all users, chosen in hindsight.

The paper PDFs in `0_lec/` … `5_extra/` are reading material and are git-ignored.
