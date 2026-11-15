# Findings (honest summary)

**What is real and what is simulated.** Listings are real: pools of 60 are drawn from 1,332
registered studio leases in Taipei City (Ministry of the Interior open rental records, 2026 Q2–Q3
files; `data/taipei_studios.csv`). Real fields: district, rent, size, building age, elevator,
furnished, and (used only by the LLM raters) floor, rooms and baths in the unit, equipment,
remarks. People are simulated, or LLM personas in E9. Also assumed: district centroids
(approximate), the user's base (NTU main campus), travel at 12 km/h door to door, commute time.
No real person's ratings are in any number below.

**Simulator scale.** On-site surprise sd 1.60 and rating noise sd 0.65 are the values fitted on
the LLM raters, our only empirical anchor (no published listing-vs-visit gap for rentals was
found). An earlier version used surprise sd 1.23 and noise 0.4; all numbers here are from the
rerun. The neighbourhood length scale (1.5 km) is our assumption: the LLM fit hit its bound.

**λ comes from a budget, not from a question.** The user sets an expected viewing budget;
`experiments/budgets.py` finds, on 600 training users, the λ whose simulated searches spend it
on average (expected-budget setting of Xie et al. 2024, where λ is a Lagrange multiplier).

| Budget | λ (pt/h) | Viewings | Shadow price |
|---|---|---|---|
| 6 h | 0.066 | 4.6 (4.7 on test users) | 15 h per point |
| 9 h (main) | 0.032 | 6.7 | 31 h per point |
| 15 h | 0.012 | 10.4 | 86 h per point |

The shadow price is not a measured value of time and nobody was asked for it. Fixed-K baselines
are tuned on the same training users; all results below are on 1000 separate test users (400 for
E6), paired, mean ± standard error unless a 95% CI is stated.

**Does the surprise scale drive search length? Yes.** Mean viewings (training users):

| Surprise sd | λ = 0.05 | λ = 0.1 | λ = 0.2 |
|---|---|---|---|
| 0.5 | 1.5 | 1.2 | 1.1 |
| 1.0 | 3.0 | 2.2 | 1.6 |
| 1.6 (main) | 5.4 | 3.7 | 2.5 |
| 2.5 | 8.6 | 6.0 | 3.8 |

If viewings rarely change one's mind (sd 0.5) the rule stops after one or two and the tool has
little to offer; the earlier "reasonable λ means one or two viewings" was partly a consequence of
the smaller scale.

## Main result: hours vs. quality (`experiments/frontier.py`, Fig. 2a)

Quality = true rating of the flat signed. The fixed-K curve of the same visiting order is linearly
interpolated; CIs are a paired bootstrap over users.

| Budget | Rule: hours, quality | Same hours: quality gained [95% CI] | Same quality: hours a fixed K needs in addition [95% CI] | vs. tuned fixed K, V |
|---|---|---|---|---|
| 6 h | 6.1 h, 8.88 | +0.17 [0.13, 0.22] | 2.9 h [2.2, 3.7] | +0.160 ± 0.021 (K = 5) |
| 9 h | 9.0 h, 9.02 | +0.13 [0.10, 0.17] | 6.6 h [3.9, 8.8] | +0.122 ± 0.019 (K = 7) |
| 15 h | 14.5 h, 9.12 | +0.12 [0.09, 0.15] | 11.8 h [7.8, 16.4] | +0.118 ± 0.015 (K = 13) |

The hours-saved figures are large and wide because the fixed-K curve is nearly flat there; the
quality-at-equal-hours column is the sturdier statement.

- vs. "view the 5 best-looking" (6.8 h): the 9 h rule is +0.33 ± 0.03 better for 2.2 h more; the
  6 h rule is +0.19 ± 0.03 better and 0.75 h shorter.
- Without cross-listing learning the same rule spends 0.9, 1.3 and 2.7 h more at the three
  budgets for a flat that is 0.03–0.04 worse (E5 at 9 h: V +0.078 ± 0.017, 1.0 fewer viewings).
- On fully synthetic pools the stop effect is +0.07 to +0.14 (`e1_synthetic`).

## What did not matter much

- **Ordering by fair value vs. EI:** +0.015 ± 0.013, +0.026 ± 0.011, +0.007 ± 0.007 with the
  cost-aware stop at 6, 9, 15 h. Small; nominally nonzero only at 9 h. In the 10× cost stress test
  PBGI beats cost-blind EI (+0.79 ± 0.06 at λ = 0.1) and ties EI-per-cost (+0.02 ± 0.02) (E7).
- **Two-step look-ahead (E6):** it disagrees with the index stop in 0.3–7% of cases and changes V
  by −0.008 to +0.020; only the strongest-correlation setting is distinguishable from zero
  (+0.020 ± 0.006).
- Minimum visits of 1, 2, 3: V changes by ≤ 0.004 (E2).

## Noise (E3, Fig. 2b)

| Noise sd | Viewings: original / adjusted index | V adjusted − original |
|---|---|---|
| 0.3 | 7.2 / 6.9 | +0.002 ± 0.005 |
| 0.65 (main) | 8.0 / 6.7 | +0.012 ± 0.012 |
| 1.0 | 9.2 / 6.3 | +0.052 ± 0.018 |
| 1.4 | 10.9 / 5.7 | +0.118 ± 0.025 |

At the main noise level the adjustment saves 1.4 viewings with no measurable change in V; it
matters in V from sd ≈ 1.0 up.

## Where it breaks (E4)

Relative to a correctly specified model in the same world (9 h budget):

| Wrong belief | Viewings | V | vs. best fixed K |
|---|---|---|---|
| listings look 1 pt better than reality | +3.7 | −0.09 ± 0.02 | +0.06 ± 0.02 |
| listings look 1 pt worse than reality | −2.3 | −0.07 ± 0.02 | +0.07 ± 0.02 |
| surprises 1.5× larger than believed | −5.4 | **−0.27 ± 0.04** | −0.05 ± 0.03 (no better) |
| surprises 1.5× smaller than believed | +5.1 | −0.15 ± 0.02 | −0.06 ± 0.02 (worse) |
| length scale 3× too short / long | +0.1 / −0.6 | 0.00 / −0.02 | +0.12 |
| rating noise believed half / twice | −0.1 / 0.0 | 0.00 / −0.04 | +0.12 / +0.11 |
| glamorous listings disappoint (0.3 / 0.6) | +0.3 / +0.2 | −0.01 / −0.05 | +0.13 / +0.16 |

Getting the size of the surprise wrong in either direction removes the advantage over a fixed
budget. Assuming correlation that is not there costs 0.06 ± 0.02 (E5).

## E9: LLM raters (stress test, not evidence about people)

300 personas played by Claude Haiku 5.5 (1,200 calls, $7.94, no failures), 40 real leases each;
p from the listing page, f = mean of two post-"viewing" ratings, the optimiser observes a third.
Surprise mean −0.86, sd 1.79. Everything is fitted on other raters (two-fold cross-fitting). LLM
respondents are known to vary less than humans (Bisbee et al. 2024), so this checks robustness to
a non-Gaussian, listing-dependent surprise and nothing more.

**Sample size was increased once, after looking.** The first 80 raters showed two non-significant
comparisons; 220 were added. All three views:

| Main method minus tuned fixed K | V | Quality | Hours |
|---|---|---|---|
| first 80 | +0.201 ± 0.083 | +0.155 ± 0.086 | −4.7 ± 0.9 |
| new 220 only | +0.017 ± 0.031 | −0.057 ± 0.032 | −6.5 ± 0.4 |
| pooled 300 | +0.042 ± 0.032 | +0.006 ± 0.033 | −2.4 ± 0.3 |

The first-batch V advantage does not replicate. What is stable is time: an equally good flat in
fewer hours. Pooled, vs. "5 best-looking": quality +0.11 ± 0.05 for 1.2 h more; vs. no
cross-listing learning: quality −0.01 ± 0.03, 1.8 ± 0.3 h less. The fitted spatial length scale
sits at its bound (7.4 km): the LLM's surprises are only weakly tied to district.

## Sensitivity only

- Delisting (E8): with viewed flats taken with probability 5% / 15% per further viewing the edge
  over the best fixed budget is +0.24 ± 0.03 / +0.37 ± 0.04. The fallback pick is optimistic for
  every rule; indicative only.
- Gap to the per-user hindsight bound (uses the hidden truth): 0.32 ± 0.02.

## Not done / not verified

- No person has used the tool. `experiments/interactive.py` runs end to end with scripted input;
  a run by the author would be an interface walkthrough, not evidence, unless flats are actually
  viewed.
- OpenBO was not used. Its documentation (README, methods page, single-function guide; source not
  read) describes designs in a continuous unit cube with a fixed evaluation budget and no
  cost-aware acquisition or stopping rule. Whether it accepts a user-supplied candidate list is
  not established.
- Demo figure: first simulated user (by seed) whose search lasts 4–6 viewings and who does not
  sign the first flat viewed; panels (a)–(b) show the viewing that moved the most estimates. An
  illustration, not a typical-case statistic.
- The charting guideline's palette validator could not be run (Node 18 on this machine).

## Literature checked against sources

Read in the local PDFs: Xie et al. 2026 (Thm 3.2; noiseless setting, App. B), Xie et al. 2024
(expected-budget setting and Thm 2; §3.2 "fair value"; §3.3 on correlations), Frazier et al. 2009
(Eq. 7–8), Hodgson & Lewis NBER w31697, Seale & Rapoport 1997 (no recall, no search cost, stop too
early), Zwick et al. (2001 working-paper version: recall succeeds with a known probability; too
little search without cost, too much with cost; the 2003 citation is bibliographic), Chan et al.
2022. Abstract pages fetched: Bisbee et al. 2024 (less variation than real surveys), Krueger &
Schkade (test–retest r = 0.59; not cited in the final PDF for space).
