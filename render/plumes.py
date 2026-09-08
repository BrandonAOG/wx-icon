#!/usr/bin/env python3
"""
Plumes: every ensemble's spread of wind gusts and rain for a list of cities,
via Open-Meteo's ensemble API (free for non-commercial use, no key).

    python render/plumes.py             # live
    python render/plumes.py --synthetic # fake members, for layout work

Output:
    site/images/plumes/<city>.png
    site/plumes.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("plumes")

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
OUT = SITE / "images" / "plumes"

API = "https://ensemble-api.open-meteo.com/v1/ensemble"
DAYS = 10

CITIES = [
    ("miami",        "Miami",            25.76, -80.19),
    ("ftlauderdale", "Fort Lauderdale",  26.12, -80.14),
    ("westpalm",     "West Palm Beach",  26.71, -80.05),
    ("keywest",      "Key West",         24.56, -81.78),
    ("marcoisland",  "Marco Island",     25.94, -81.72),
    ("naples",       "Naples",           26.14, -81.79),
    ("bonita",       "Bonita Springs",   26.34, -81.78),
    ("estero",       "Estero",           26.44, -81.81),
    ("fmbeach",      "Fort Myers Beach", 26.45, -81.95),
    ("sanibel",      "Sanibel",          26.45, -82.08),
    ("fortmyers",    "Fort Myers",       26.64, -81.87),
    ("capecoral",    "Cape Coral",       26.56, -81.95),
    ("pineisland",   "Pine Island",      26.60, -82.12),
    ("puntagorda",   "Punta Gorda",      26.93, -82.05),
    ("portcharlotte","Port Charlotte",   26.98, -82.09),
    ("englewood",    "Englewood",        26.96, -82.35),
    ("venice",       "Venice",           27.10, -82.45),
    ("siestakey",    "Siesta Key",       27.27, -82.55),
    ("sarasota",     "Sarasota",         27.34, -82.53),
    ("tampa",        "Tampa",            27.95, -82.46),
    ("orlando",      "Orlando",          28.54, -81.38),
    ("daytona",      "Daytona Beach",    29.21, -81.02),
    ("jacksonville", "Jacksonville",     30.33, -81.66),
    ("tallahassee",  "Tallahassee",      30.44, -84.28),
    ("panamacity",   "Panama City",      30.16, -85.66),
    ("pensacola",    "Pensacola",        30.42, -87.22),
]

# Open-Meteo model ids -> display. Requested one at a time so an unknown id
# (they rename occasionally) only loses that model; the API's error message
# lists the valid ids and is logged.
ENSEMBLES = [
    ("gfs025",                     "GEFS",                 "#d62728"),
    ("ecmwf_ifs025",               "ECMWF ENS",            "#1f4ed8"),
    ("ecmwf_aifs025_ensemble",     "ECMWF AIFS ENS",       "#4fa3ff"),
    ("gem_global",                 "GEPS",                 "#ff7f0e"),
    ("icon_global",                "ICON EPS",             "#17becf"),
    ("ukmo_global_ensemble_20km",  "UKMO MOGREPS-G",       "#2ca02c"),
    ("ncep_aigefs025",             "AI-GEFS",              "#e8590c"),
    ("google_weathernext2_ensemble", "Google WeatherNext 2", "#00a86b"),
]
VARS = ["wind_gusts_10m", "wind_speed_10m", "precipitation", "pressure_msl"]


import time as _time
PACE_S = 0.6            # seconds between requests: keeps well under Open-Meteo's per-minute limit


def fetch(lat, lon, model, session):
    """-> {"time": [datetime], var: 2-D array (member, time)} or None"""
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(VARS), "models": model,
              "forecast_days": DAYS, "wind_speed_unit": "kn", "precipitation_unit": "inch", "timezone": "GMT"}
    r = None
    for attempt in range(4):
        _time.sleep(PACE_S)
        try:
            r = session.get(API, params=params, timeout=60)
        except requests.RequestException as e:
            log.warning("%s: %s", model, e); r = None
        if r is not None and r.status_code == 200:
            break
        if r is not None and r.status_code == 429:           # rate limited: back off and retry
            wait = 15 * (attempt + 1)
            log.warning("%s: rate limited, waiting %ds", model, wait); _time.sleep(wait); continue
        if r is not None:
            log.warning("%s -> HTTP %s: %s", model, r.status_code, r.text[:300]); return None
    if r is None or r.status_code != 200:
        return None
    h = r.json().get("hourly", {})
    times = [dt.datetime.fromisoformat(t) for t in h.get("time", [])]
    out = {"time": times}
    for v in VARS:
        cols = [k for k in h if k == v or k.startswith(v + "_member")]
        if not cols:
            continue
        arr = np.array([[np.nan if x is None else x for x in h[k]] for k in cols], dtype=float)
        out[v] = arr
    return out if any(v in out for v in VARS) else None


def synthetic(lat, lon, model_i, rng, n=30):
    t0 = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    times = [t0 + dt.timedelta(hours=i) for i in range(DAYS * 24)]
    h = np.arange(len(times)); shift = rng.normal() * 8 + model_i * 3; amp = 40 + rng.normal() * 10
    g = np.array([12 + amp * np.exp(-((h - 110 - shift - rng.normal() * 6) / 20) ** 2) + rng.normal(0, 2, len(h)) for _ in range(n)])
    p = np.array([np.clip(0.35 * np.exp(-((h - 110 - shift - rng.normal() * 6) / 16) ** 2) + 0.02 * rng.random(len(h)), 0, None) for _ in range(n)])
    m = np.array([101300 - 3500 * np.exp(-((h - 110 - shift - rng.normal() * 6) / 20) ** 2) + rng.normal(0, 150, len(h)) for _ in range(n)]) / 100
    out = {"time": times, "wind_gusts_10m": g, "precipitation": p, "pressure_msl": m}
    if model_i % 3 == 1:                                  # mimic models without gusts
        out["wind_speed_10m"] = out.pop("wind_gusts_10m") * 0.78
    return out


def six_hourly(times, arr):
    """Sum hourly precip into 6-h bins ending at each 6th hour."""
    n = (arr.shape[1] // 6) * 6
    return [times[i] for i in range(5, n, 6)], arr[:, :n].reshape(arr.shape[0], -1, 6).sum(axis=2)


def plot_city(cid, name, lat, lon, data, init, dest: Path):
    fig = plt.figure(figsize=(14, 10), dpi=100); fig.patch.set_facecolor("#f1f4f7")
    fig.text(0.04, 0.955, f"Plumes — {name}, FL  ({lat:.2f}°N, {abs(lon):.2f}°W)", fontsize=18, fontweight="bold", color="#17212b")
    fig.text(0.04, 0.925, f"{len(data)} ensembles · shading = 10th–90th percentile of members · line = ensemble mean · updated {init:%a %d %b %Y %H:%M}Z",
             fontsize=10.5, color="#5d6c7b")
    ax_g = fig.add_axes([0.06, 0.53, 0.66, 0.36]); ax_r = fig.add_axes([0.06, 0.09, 0.66, 0.36])
    rows = []
    for (mid, label, color), d in data:
        t = d["time"]
        # gusts where the model provides them; otherwise sustained wind, drawn dashed
        has_gust = "wind_gusts_10m" in d and np.isfinite(d["wind_gusts_10m"]).any()
        wkey = "wind_gusts_10m" if has_gust else ("wind_speed_10m" if "wind_speed_10m" in d and np.isfinite(d["wind_speed_10m"]).any() else None)
        if wkey:
            g = d[wkey]
            g = g[np.isfinite(g).any(axis=1)]                 # drop members that are entirely empty
            ax_g.fill_between(t, np.nanpercentile(g, 10, axis=0), np.nanpercentile(g, 90, axis=0), color=color, alpha=0.10, lw=0)
            ax_g.plot(t, np.nanmean(g, axis=0), color=color, lw=2, ls="-" if has_gust else (0, (4, 2)),
                      label=f"{label} ({g.shape[0]})" + ("" if has_gust else " — sustained"))
            mean_g = np.nanmean(g, axis=0)
            if not np.isfinite(mean_g).any():
                continue
            peak = np.nanmax(mean_g); when = t[int(np.nanargmax(np.nan_to_num(mean_g, nan=-1)))]
            p64 = 100 * np.nanmean(np.nanmax(g, axis=1) >= 64); p34 = 100 * np.nanmean(np.nanmax(g, axis=1) >= 34)
            rows.append((label + ("" if has_gust else "*"), peak, when, p34, p64, g.shape[0]))
        if "precipitation" in d and np.isfinite(d["precipitation"]).any():
            t6, p6 = six_hourly(t, np.nan_to_num(d["precipitation"], nan=0.0))
            ax_r.fill_between(t6, np.nanpercentile(p6, 10, axis=0), np.nanpercentile(p6, 90, axis=0), color=color, alpha=0.10, lw=0)
            ax_r.plot(t6, np.nanmean(p6, axis=0), color=color, lw=2)
    top = min(160, max(60, 10 * int(np.ceil((max([r[1] for r in rows] + [40]) * 1.25) / 10))))
    ax_g.set_ylim(0, top); ax_g.set_ylabel("10 m wind (kt): gusts solid, sustained dashed")
    for y, lab in [(34, "TS"), (64, "Cat 1"), (83, "Cat 2"), (96, "Cat 3"), (113, "Cat 4"), (137, "Cat 5")]:
        if y < top:
            ax_g.axhline(y, color="#bbb", lw=0.7, ls=":")
            ax_g.text(1.0, y, " " + lab, transform=ax_g.get_yaxis_transform(), fontsize=8, color="#888", va="center")
    ax_g.set_title("Wind", loc="left", fontsize=12, fontweight="bold", color="#17212b")
    ax_r.set_ylabel("6-hr precipitation (in)"); ax_r.set_title("Rainfall", loc="left", fontsize=12, fontweight="bold", color="#17212b")
    ax_r.set_ylim(0, None)
    for ax in (ax_g, ax_r):
        ax.set_facecolor("white"); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color="#eee")
        ax.xaxis.set_major_locator(mdates.DayLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
        ax.tick_params(labelsize=8.5)
    ax_g.legend(loc="upper left", bbox_to_anchor=(1.03, 1.0), frameon=False, fontsize=9.5, title="Ensembles (members)", title_fontsize=10)
    # summary table
    tx = fig.add_axes([0.745, 0.09, 0.24, 0.36]); tx.axis("off")
    tx.text(0, 1, "Peak wind by ensemble", fontsize=12, fontweight="bold", va="top", color="#17212b")
    for x, h in [(0, "model"), (0.52, "mean peak"), (0.78, "P(≥34)"), (0.92, "P(≥64)")]:
        tx.text(x, 0.88, h, fontsize=8.5, color="#5d6c7b", va="top")
    for i, (label, peak, when, p34, p64, n) in enumerate(sorted(rows, key=lambda r: -r[1])):
        y = 0.79 - i * 0.095
        tx.text(0, y, label, fontsize=9.5, va="top", color="#17212b")
        tx.text(0.52, y, f"{peak:.0f} kt", fontsize=9.5, va="top", fontweight="bold", color="#17212b")
        tx.text(0.78, y, f"{p34:.0f}%", fontsize=9.5, va="top", color="#c81e1e" if p34 >= 50 else "#17212b")
        tx.text(0.92, y, f"{p64:.0f}%", fontsize=9.5, va="top", color="#c81e1e" if p64 >= 30 else "#17212b")
        tx.text(0, y - 0.04, f"{when:%a %d %b %HZ}", fontsize=7.5, va="top", color="#8a97a5")
    fig.text(0.04, 0.03, "WxModels · point forecasts via Open-Meteo (NOAA, ECMWF, ECCC, DWD, UKMO, Google) · model output, not an official forecast · P(≥) = share of members reaching that wind at any time · * = sustained wind (model gives no gusts)",
             fontsize=8.5, color="#8a97a5")
    fig.savefig(dest, facecolor=fig.get_facecolor()); plt.close(fig)
    return rows


SW_CLUSTER = ["siestakey", "venice", "englewood", "portcharlotte", "puntagorda", "pineisland", "capecoral", "fortmyers",
              "sanibel", "fmbeach", "estero", "bonita", "naples", "marcoisland"]


def plot_overview(summary, init, dest: Path):
    """Florida map: each city coloured by all-ensemble mean peak wind, labelled
    with peak wind (kt) and 10-day rain total (in). The crowded southwest coast
    gets an inset zoom over the Gulf."""
    import matplotlib.colors as mcolors
    import matplotlib.patheffects as pe
    from plots import PC, add_basemap
    fig = plt.figure(figsize=(12, 9), dpi=100); fig.patch.set_facecolor("#f1f4f7")
    bounds = [0, 20, 34, 50, 64, 83, 96, 113, 140]
    cmap = mcolors.ListedColormap(["#9ecae1", "#41ab5d", "#f7e530", "#f5a623", "#f05a28", "#d0021b", "#9b0c3d", "#5e0a5e"])
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    def draw(ax, extent, cities, offsets, fs):
        ax.set_extent(extent, crs=PC); ax.set_facecolor("#dfe9f1")
        try:
            import cartopy.feature as cfeature
            land = cfeature.LAND.with_scale("50m"); next(iter(land.geometries()))
            ax.add_feature(land, facecolor="#f7f4ea", zorder=1)
        except Exception:  # noqa: BLE001
            pass
        add_basemap(ax)
        for c in cities:
            col = cmap(norm(c["peak_kt"] or 0))
            ax.plot(c["lon"], c["lat"], "o", ms=10, color=col, mec="white", mew=1.2, transform=PC, zorder=6)
            dx, dy = offsets.get(c["id"], (0.12, 0.05))
            ax.text(c["lon"] + dx, c["lat"] + dy, f"{c['name']}  {c['peak_kt']} kt · {c['rain_in']:.1f} in", fontsize=fs, fontweight="bold",
                    ha="left" if dx > 0 else "right", va="center", transform=PC, zorder=7, color="#17212b",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    main_cities = [c for c in summary if c["id"] not in SW_CLUSTER]
    sw = [c for c in summary if c["id"] in SW_CLUSTER]
    ax = fig.add_axes([0.02, 0.06, 0.96, 0.84], projection=PC)
    draw(ax, (-88.5, -78.5, 24.0, 31.5), main_cities,
         {"panamacity": (0.1, -0.3), "pensacola": (0.1, -0.3), "keywest": (0.15, -0.15), "tampa": (-0.15, 0.1), "sarasota": (-0.15, -0.05),
          "miami": (0.15, -0.05), "ftlauderdale": (0.15, 0.05), "westpalm": (0.15, 0.05)}, 9)
    # box marking the inset area on the main map
    x0, x1, y0, y1 = -83.7, -81.3, 25.75, 27.5
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="#0e7c86", lw=1.4, transform=PC, zorder=8)
    if sw:
        ins = fig.add_axes([0.05, 0.10, 0.40, 0.50], projection=PC)
        ins.spines["geo"].set_edgecolor("#0e7c86"); ins.spines["geo"].set_linewidth(1.6)
        left = {k: (-0.05, 0.0) for k in ["siestakey", "venice", "englewood", "portcharlotte", "pineisland", "sanibel", "fmbeach"]}
        right = {k: (0.05, 0.0) for k in ["puntagorda", "fortmyers", "capecoral", "estero", "bonita", "naples", "marcoisland"]}
        left["englewood"] = (-0.05, -0.06); right["puntagorda"] = (0.05, 0.06)
        left["pineisland"] = (-0.05, 0.08); right["capecoral"] = (0.05, -0.05); right["fortmyers"] = (0.05, 0.07)
        left["sanibel"] = (-0.05, 0.03); left["fmbeach"] = (-0.05, -0.10)
        draw(ins, (x0, x1, y0, y1), sw, {**left, **right}, 8)
        ins.set_title("Southwest coast", fontsize=9, fontweight="bold", color="#0e7c86", loc="left", pad=3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cax = fig.add_axes([0.30, 0.058, 0.45, 0.014]); cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=bounds)
    cb.ax.tick_params(labelsize=8); cb.set_label("Peak 10 m wind, all-ensemble mean (kt) — TS 34 · Cat 1 64 · Cat 2 83 · Cat 3 96", fontsize=8.5)
    fig.text(0.02, 0.955, "Florida ensemble outlook — next 10 days", fontsize=16, fontweight="bold", color="#17212b")
    fig.text(0.02, 0.925, f"Label: peak wind (kt) · total rainfall (in), averaged across all ensembles · updated {init:%a %d %b %Y %H:%M}Z",
             fontsize=10, color="#5d6c7b")
    fig.text(0.02, 0.006, "WxModels · via Open-Meteo · model output, not an official forecast", fontsize=8.5, color="#8a97a5")
    fig.savefig(dest, facecolor=fig.get_facecolor()); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--synthetic", action="store_true"); args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    session = requests.Session(); session.headers["User-Agent"] = "wxmodels-plumes (github actions)"
    now = dt.datetime.now(dt.timezone.utc)
    rng = np.random.default_rng(1)
    result = {"generated": now.isoformat(), "cities": [], "models": []}
    seen_models = set()
    for cid, name, lat, lon in CITIES:
        data = []
        for i, (mid, label, color) in enumerate(ENSEMBLES):
            d = synthetic(lat, lon, i, rng) if args.synthetic else fetch(lat, lon, mid, session)
            if d and any(v in d and np.isfinite(d[v]).any() for v in ("wind_gusts_10m", "wind_speed_10m", "precipitation")):
                data.append(((mid, label, color), d)); seen_models.add(label)
        if not data:
            log.warning("%s: no ensemble data", name); continue
        try:
            rows = plot_city(cid, name, lat, lon, data, now, OUT / f"{cid}.png")
        except Exception as e:  # noqa: BLE001
            plt.close("all"); log.error("%s: plot failed: %s", name, str(e)[:160]); continue
        best = max(rows, key=lambda r: r[1]) if rows else None
        rains = [float(np.nansum(np.nanmean(np.nan_to_num(d["precipitation"], nan=0.0), axis=0))) for _, d in data if "precipitation" in d]
        entry = {"id": cid, "name": name, "lat": lat, "lon": lon, "image": f"images/plumes/{cid}.png",
                 "peak_kt": round(float(np.mean([r[1] for r in rows]))) if rows else None,     # all-ensemble mean of peaks
                 "max_kt": round(best[1]) if best else None, "peak_model": best[0] if best else None,
                 "peak_time": best[2].isoformat() + "Z" if best else None,
                 "rain_in": round(float(np.mean(rains)), 1) if rains else 0.0,
                 "p34": round(max(r[3] for r in rows)) if rows else None, "p64": round(max(r[4] for r in rows)) if rows else None}
        result["cities"].append(entry)
        log.info("%s: %d ensembles", name, len(data))
    result["models"] = sorted(seen_models)
    if result["cities"]:
        try:
            plot_overview(result["cities"], now, OUT / "overview.png"); result["overview"] = "images/plumes/overview.png"
        except Exception as e:  # noqa: BLE001
            plt.close("all"); log.error("overview failed: %s", str(e)[:160])
    (SITE / "plumes.json").write_text(json.dumps(result, indent=1))
    log.info("done: %d cities", len(result["cities"]))


if __name__ == "__main__":
    main()
