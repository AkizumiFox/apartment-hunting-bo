"""LLM raters: a language model plays the person looking for a flat.

Each rater (a persona) scores 40 real Taipei leases in four separate calls:

    P   listing page only: district, rent, size, type, furnished          -> p_i
    A,B after a "viewing": every recorded fact about the flat (building type and
        age, floor, elevator, manager, how many rooms share the unit, bathrooms,
        equipment, remarks)                                               -> f_i = mean(A, B)
    C   a third independent viewing rating                                -> y_i (what the
                                                                             optimiser observes)

So the on-site surprise f - p comes from real facts that the listing page withholds,
and rating noise is the model's own call-to-call variation. These are LLM ratings,
not human ratings.

    uv run python experiments/llm_raters.py            # 300 raters, 4 calls each to `claude -p`; resumes
"""
import csv
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import ROOT
from housebo.simulator import CENTROID, HOME, MOI_CSV

OUT = ROOT / "data" / "llm_ratings.json"
MODEL = "haiku"
N_RATERS, N_LISTINGS = 300, 40
ROMAN = {"中正區": "Zhongzheng", "大同區": "Datong", "中山區": "Zhongshan", "松山區": "Songshan",
         "大安區": "Da'an", "萬華區": "Wanhua", "信義區": "Xinyi", "士林區": "Shilin",
         "北投區": "Beitou", "內湖區": "Neihu", "南港區": "Nangang", "文山區": "Wenshan"}
TRAITS = [
    ["You are on a tight budget and every NT$1,000 of rent hurts.",
     "You have a comfortable allowance; rent matters little below NT$25,000.",
     "You are moderately price-sensitive."],
    ["You need room for a desk and bookshelves and hate cramped rooms.",
     "You are out all day and are fine with a small room.",
     "You want enough space to have a friend over."],
    ["You refuse to climb more than two flights of stairs.",
     "You do not mind walk-ups at all.",
     "You strongly prefer newer buildings and dislike anything over 40 years old.",
     "You find old buildings charming as long as they are maintained."],
    ["You want privacy and dislike units carved into many rented rooms.",
     "You do not mind many housemates.",
     "You are noise-sensitive and wary of crowded subdivided flats."],
    ["You cook often and need a fridge and gas or a stove.",
     "You do laundry at home and need a washing machine.",
     "You work from home and need reliable internet.",
     "You care that utilities and fees are included in the rent."],
    ["You hate long commutes.",
     "You tolerate a long commute for a better flat.",
     "You cycle everywhere and want to be close to campus."],
]
ROLES = ["a graduate student", "an undergraduate", "a first-year software engineer",
         "a research assistant", "an exchange student"]


def persona(rng):
    picks = [opts[rng.integers(len(opts))] for opts in TRAITS]
    return (f"You are {ROLES[rng.integers(len(ROLES))]} based at the NTU main campus (Gongguan), "
            "looking for a studio. " + " ".join(picks))


def minutes(row):
    lat, lon = CENTROID[row["district"]]
    km = np.hypot((lon - HOME[1]) * 111.32 * np.cos(np.radians(HOME[0])), (lat - HOME[0]) * 110.57)
    return int(round(10 + 4 * km))


def page(i, r):
    return (f"{i + 1}. {ROMAN[r['district']]} district (about {minutes(r)} min from campus), NT${int(r['rent']):,}/month, "
            f"{r['size_ping']} ping, {'self-contained studio' if r['rental_type'] == '獨立套房' else 'studio room in a shared flat'}, "
            f"{'furnished' if r['furnished'] == '1' else 'unfurnished'}")


def viewing(i, r):
    return (page(i, r) + f". Seen on site: {r['building_type']} building, {r['age_years']} years old, floor {r['floor']} of "
            f"{r['total_floors']}, {'elevator' if r['elevator'] == '1' else 'no elevator'}, "
            f"{'has a building manager' if r['manager'] == '1' else 'no manager'}, the unit contains {r['rooms']} rooms and "
            f"{r['baths']} bathrooms, equipment: {r['equipment'] or 'none listed'}"
            + (f", notes: {r['remarks']}" if r["remarks"] else ""))


def prompt(who, lines, stage):
    lead = ("You are browsing listing pages. Judging ONLY from what each page says, score how satisfied you expect to be "
            "living there." if stage == "page" else
            "You have now visited every flat below and know the facts listed. Score how satisfied you would be living there.")
    return (f"{who}\n\n{lead} Use 0-10 with one decimal (0 = would never live there, 10 = ideal). Judge each flat on its own "
            f"merits for this person and use as much of the scale as the flats deserve.\n\n" + "\n".join(lines) +
            f"\n\nReply with ONLY a JSON array of {len(lines)} numbers, in order. No other text.")


def call(text, n):
    for _ in range(4):
        with tempfile.TemporaryDirectory() as d:   # empty cwd: no project context leaks in
            r = subprocess.run(["claude", "-p", "--model", MODEL, "--output-format", "json"], input=text,
                               capture_output=True, text=True, cwd=d, timeout=300)
        try:
            body = json.loads(r.stdout)
            m = re.search(r"\[[^\[\]]*\]", body["result"])
            vals = [float(v) for v in json.loads(m.group(0))]
            if len(vals) == n and all(0 <= v <= 10 for v in vals):
                return vals, next(iter(body.get("modelUsage", {})), MODEL), body.get("total_cost_usd", 0.0)
        except Exception:
            continue
    raise RuntimeError("no valid rating array after 4 attempts")


def rate(u, rows):
    rng = np.random.default_rng([2026, u])
    who = persona(rng)
    idx = rng.choice(len(rows), size=N_LISTINGS, replace=False)
    mine = [rows[i] for i in idx]
    pg = prompt(who, [page(i, r) for i, r in enumerate(mine)], "page")
    vw = prompt(who, [viewing(i, r) for i, r in enumerate(mine)], "viewing")
    res = [call(pg, N_LISTINGS)] + [call(vw, N_LISTINGS) for _ in range(3)]
    return dict(rater=u, persona=who, listings=[int(i) for i in idx], p=res[0][0], A=res[1][0], B=res[2][0],
                C=res[3][0], model=res[0][1], cost_usd=sum(r[2] for r in res))


if __name__ == "__main__":
    with MOI_CSV.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["district"] in CENTROID]
    if "--show" in sys.argv:
        rng = np.random.default_rng([2026, 0])
        who = persona(rng)
        mine = [rows[i] for i in rng.choice(len(rows), size=3, replace=False)]
        print(prompt(who, [page(i, r) for i, r in enumerate(mine)], "page"), "\n\n---\n")
        print(prompt(who, [viewing(i, r) for i, r in enumerate(mine)], "viewing"))
        sys.exit()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_RATERS
    done = json.loads(OUT.read_text())["raters"] if OUT.exists() else []   # resume: keep what we have
    have = {d["rater"] for d in done}
    todo = [u for u in range(n) if u not in have]
    failed = 0
    with ThreadPoolExecutor(16) as ex:
        futs = [ex.submit(rate, u, rows) for u in todo]
        for k, fu in enumerate(futs):
            try:
                done.append(fu.result())
            except Exception as e:   # report, never fabricate
                failed += 1
                print(f"rater {todo[k]} failed: {e}", flush=True)
            if (k + 1) % 20 == 0:
                print(f"{k + 1}/{len(todo)} new raters", flush=True)
                OUT.write_text(json.dumps(dict(model=done[0]["model"], n_listings=N_LISTINGS, failed=failed,
                               total_cost_usd=sum(d["cost_usd"] for d in done),
                               raters=sorted(done, key=lambda d: d["rater"])), ensure_ascii=False))
    done.sort(key=lambda d: d["rater"])
    OUT.write_text(json.dumps(dict(model=done[0]["model"], n_listings=N_LISTINGS, failed=failed,
                                   total_cost_usd=sum(d["cost_usd"] for d in done), raters=done),
                              ensure_ascii=False))
    print(f"{len(done)} raters saved ({failed} failed this run), total cost ${sum(d['cost_usd'] for d in done):.2f} -> {OUT}")
