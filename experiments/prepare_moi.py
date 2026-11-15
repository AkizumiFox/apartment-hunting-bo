"""Build data/taipei_studios.csv from the Ministry of the Interior's open rental records.

Source: 內政部不動產交易實價查詢服務網, 不動產租賃 (plvr.land.moi.gov.tw), Open Government
Data License v1.0. We keep registered leases in Taipei City whose rental type is a studio
(獨立套房 / 分租套房) and keep only listing-level attributes; street addresses and serial
numbers are dropped.

    uv run python experiments/prepare_moi.py 115S2 115S3

Not in the data and therefore ASSUMED (see simulator.py): district centroids, the user's
home base, travel speed, and everything about how a person rates a flat.
"""
import csv
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "taipei_studios.csv"
URL = "https://plvr.land.moi.gov.tw/DownloadSeason?season={s}&type=zip&fileName=lvr_landcsv.zip"
PING = 3.305785  # square metres per ping
FIELDS = ["season", "district", "rental_type", "building_type", "rent", "size_ping", "age_years",
          "floor", "total_floors", "elevator", "furnished", "manager",
          "rooms", "baths", "equipment", "remarks"]
FLOOR = {c: i + 1 for i, c in enumerate("一二三四五六七八九")}


def floor_number(s: str):
    """'三層' -> 3, '十二層' -> 12; None for basements, multi-floor or whole-building leases."""
    s = s.strip()
    if not s.endswith("層") or "，" in s or "地下" in s:
        return None
    s = s[:-1]
    if s == "十":
        return 10
    if s.startswith("十"):
        return 10 + FLOOR.get(s[1:], 0)
    if "十" in s:
        a, b = s.split("十")
        return FLOOR.get(a, 0) * 10 + (FLOOR.get(b, 0) if b else 0)
    return FLOOR.get(s)


def load(season: str):
    z = RAW / f"lvr_{season}.zip"
    if not z.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(URL.format(s=season), headers={"User-Agent": "Mozilla/5.0"})
        z.write_bytes(urllib.request.urlopen(req, timeout=180).read())
    with zipfile.ZipFile(z) as f:
        text = f.read("a_lvr_land_c.csv").decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))[1:]  # second line is an English header
    lease_year = int(season[:3])
    for r in rows:
        try:
            if r["出租型態"] not in ("獨立套房", "分租套房"):
                continue
            if "社會住宅" in r["租賃住宅服務"]:        # subsidised rents, not market listings
                continue
            if r["主要用途"] not in ("住家用", "集合住宅", "住商用", ""):
                continue
            rent, sqm = float(r["總額元"]), float(r["建物總面積平方公尺"])
            built = r["建築完成年月"].strip()
            floor = floor_number(r["租賃層次"])
            if not (6000 <= rent <= 35000) or not (4 * PING <= sqm <= 20 * PING):
                continue
            if len(built) < 6 or floor is None:
                continue
            age = lease_year - int(built[:-4])
            if not 0 <= age <= 70:
                continue
            yield dict(season=season, district=r["鄉鎮市區"], rental_type=r["出租型態"],
                       building_type=r["建物型態"].split("(")[0], rent=int(rent),
                       size_ping=round(sqm / PING, 1), age_years=age, floor=floor,
                       total_floors=int(r["總樓層數"] or 0), elevator=int(r["有無電梯"] == "有"),
                       furnished=int(r["有無附傢俱"] == "有"), manager=int(r["有無管理員"] == "有"),
                       rooms=int(r["建物現況格局-房"] or 0), baths=int(r["建物現況格局-衛"] or 0),
                       equipment=r["附屬設備"].strip(), remarks=r["備註"].strip()[:120])
        except (ValueError, KeyError):
            continue


if __name__ == "__main__":
    seasons = sys.argv[1:] or ["115S2", "115S3"]
    rows = [r for s in seasons for r in load(s)]
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} studio leases -> {OUT}")
