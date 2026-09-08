#!/usr/bin/env python3
"""
Tropical page renderer.

Sources (all NOAA/NHC, public domain):
  * https://www.nhc.noaa.gov/CurrentStorms.json         active storms (Atlantic, E/C Pacific)
  * https://ftp.nhc.noaa.gov/atcf/aid_public/a<id>.dat.gz  ATCF a-deck: every model's forecast track
  * https://ftp.nhc.noaa.gov/atcf/btk/b<id>.dat            ATCF b-deck: observed best track so far
  * https://www.nhc.noaa.gov/gis/forecast/archive/<id>_5day_latest.zip  official cone shapefile

Output:
  site/images/tropical/<storm>/track.png       spaghetti + cone + best track
  site/images/tropical/<storm>/intensity.png   wind-speed guidance
  site/images/tropical/overview_<basin>.png    every active storm in the basin
  site/tropical.json                           what the frontend reads

    python render/tropical.py             # live
    python render/tropical.py --synthetic # fake storm for testing, no network
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import io
import json
import logging
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cartopy.crs as ccrs  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402

import storage  # noqa: E402
from plots import PC, add_basemap  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("tropical")

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
OUT = SITE / "images" / "tropical"

CURRENT_STORMS = "https://www.nhc.noaa.gov/CurrentStorms.json"
ADECK = "https://ftp.nhc.noaa.gov/atcf/aid_public/a{sid}.dat.gz"
BDECK = "https://ftp.nhc.noaa.gov/atcf/btk/b{sid}.dat"
CONE = "https://www.nhc.noaa.gov/gis/forecast/archive/{sid}_5day_latest.zip"
GTWO = "https://www.nhc.noaa.gov/xgtwo/gtwo_shapefiles.zip"     # outlook areas with 2/7-day probabilities
INVEST_MAX_AGE_H = 30                                           # a-deck must have been touched this recently

# ATCF "tech" codes -> display. Order here = legend order. Colours chosen so
# the deterministic globals stand apart from hurricane models and consensus.
# GEFS members (AP01..AP30) are handled separately.
MODELS = {
    "OFCL": ("NHC official",      "#000000", 2.6),
    # --- global models (parent code, then the 6-h "I" and 12-h "2" interpolated versions
    #     that NHC times to the current cycle; same label so they dedupe)
    "AVNO": ("GFS",               "#d62728", 1.8), "AVNI": ("GFS", "#d62728", 1.8), "AVN2": ("GFS", "#d62728", 1.8),
    "ECMF": ("ECMWF",             "#1f4ed8", 1.8), "EMXI": ("ECMWF", "#1f4ed8", 1.8), "EMX2": ("ECMWF", "#1f4ed8", 1.8),
    "UKX":  ("UKMET",             "#2ca02c", 1.6), "UKM":  ("UKMET", "#2ca02c", 1.6), "UKXI": ("UKMET", "#2ca02c", 1.6), "UKX2": ("UKMET", "#2ca02c", 1.6), "EGRI": ("UKMET", "#2ca02c", 1.6),
    "CMC":  ("Canadian",          "#ff7f0e", 1.6), "CMCI": ("Canadian", "#ff7f0e", 1.6), "CMC2": ("Canadian", "#ff7f0e", 1.6),
    "NVGM": ("NAVGEM",            "#8c564b", 1.4), "NVGI": ("NAVGEM", "#8c564b", 1.4), "NVG2": ("NAVGEM", "#8c564b", 1.4),
    "ICON": ("ICON",              "#17becf", 1.4), "ICNI": ("ICON", "#17becf", 1.4),
    # --- AI models
    "GDMN": ("Google DeepMind",   "#00a86b", 1.8), "GDMI": ("Google DeepMind", "#00a86b", 1.8), "GDM2": ("Google DeepMind", "#00a86b", 1.8),
    "FNV3": ("Google FNV3",       "#00a86b", 1.8), "GNCS": ("Google GenCast", "#2e8b57", 1.4),
    "ECAI": ("ECMWF AIFS",        "#4fa3ff", 1.4), "EAII": ("ECMWF AIFS", "#4fa3ff", 1.4),
    "NGX":  ("NCEP AI-GFS",       "#e8590c", 1.4), "NGX2": ("NCEP AI-GFS", "#e8590c", 1.4), "NGXI": ("NCEP AI-GFS", "#e8590c", 1.4),
    # --- hurricane models
    "HFSA": ("HAFS-A",            "#e377c2", 1.6), "HFAI": ("HAFS-A", "#e377c2", 1.6), "HFA2": ("HAFS-A", "#e377c2", 1.6),
    "HFSB": ("HAFS-B",            "#bc5090", 1.6), "HFBI": ("HAFS-B", "#bc5090", 1.6), "HFB2": ("HAFS-B", "#bc5090", 1.6),
    "HWRF": ("HWRF",              "#9467bd", 1.6), "HWFI": ("HWRF", "#9467bd", 1.6), "HWF2": ("HWRF", "#9467bd", 1.6),
    "HMON": ("HMON",              "#7f3fbf", 1.4), "HMNI": ("HMON", "#7f3fbf", 1.4), "HMN2": ("HMON", "#7f3fbf", 1.4),
    "CTCX": ("COAMPS-TC",         "#a05195", 1.4), "CTCI": ("COAMPS-TC", "#a05195", 1.4), "CTC2": ("COAMPS-TC", "#a05195", 1.4),
    # --- ensemble means
    "AEMN": ("GEFS mean",         "#b22222", 1.4), "AEMI": ("GEFS mean", "#b22222", 1.4), "AEM2": ("GEFS mean", "#b22222", 1.4),
    "EEMN": ("ECMWF ens mean",    "#0b2a8a", 1.4), "EEMI": ("ECMWF ens mean", "#0b2a8a", 1.4),
    "UEMN": ("UKMET ens mean",    "#1b6b1b", 1.4), "UEMI": ("UKMET ens mean", "#1b6b1b", 1.4),
    "CEMN": ("Canadian ens mean", "#c26a00", 1.4), "CEMI": ("Canadian ens mean", "#c26a00", 1.4), "CEM2": ("Canadian ens mean", "#c26a00", 1.4),
    # --- consensus
    "TVCN": ("Consensus TVCN",    "#555555", 2.0), "TVCA": ("Consensus TVCA", "#666666", 1.6), "TVCX": ("Consensus TVCX", "#777777", 1.4),
    "HCCA": ("HCCA",              "#777777", 1.8),
    # --- intensity-only guidance (charted, not mapped)
    "IVCN": ("Intensity cons.",   "#999999", 1.6), "RVCN": ("RI consensus", "#aaaaaa", 1.4),
    "SHIP": ("SHIPS",             "#c49c00", 1.2), "DSHP": ("Decay-SHIPS", "#e0b400", 1.2), "LGEM": ("LGEM", "#aa8800", 1.2),
    "NNIC": ("NHC neural-net",    "#5c7cfa", 1.2), "NNIB": ("NHC neural-net B", "#7a94fa", 1.0),
}
# codes we deliberately skip on both plots: climatology/persistence baselines and NHC's own copies
SKIP = {"CARQ", "CLP5", "OCD5", "SHF5", "TCLP", "XTRP", "DRCL", "TABD", "TABM", "TABS", "OFCI", "RI25", "RI30", "RI35", "RI40"}
INTENSITY_ONLY = {"SHIP", "DSHP", "LGEM", "IVCN", "RVCN", "NNIC", "NNIB"}
GEFS_PREFIX = "AP"      # AP01..AP30
ECENS_PREFIX = "EE"     # EE01..EE50 (present when ECMWF ensemble tracks are fed)


# ---------------------------------------------------------------- ATCF -----

def atcf_latlon(lat_s: str, lon_s: str):
    lat = int(lat_s[:-1]) / 10 * (1 if lat_s[-1] == "N" else -1)
    lon = int(lon_s[:-1]) / 10 * (-1 if lon_s[-1] == "W" else 1)
    return lat, lon


def parse_atcf(text: str):
    """Return {tech: {cycle: [(tau, lat, lon, vmax, mslp), ...]}}"""
    out = defaultdict(lambda: defaultdict(dict))
    for line in text.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 11 or not f[6] or not f[7]:
            continue
        try:
            cycle = f[2]
            tech = f[4]
            tau = int(f[5])
            lat, lon = atcf_latlon(f[6], f[7])
            vmax = int(f[8]) if f[8] else None
            mslp = int(f[9]) if len(f) > 9 and f[9] else None
        except ValueError:
            continue
        if lat == 0 and lon == 0:
            continue
        # ATCF repeats a tau per wind radius (34/50/64 kt); keep one
        out[tech][cycle].setdefault(tau, (tau, lat, lon, vmax, mslp))
    return {t: {c: sorted(v.values()) for c, v in cyc.items()} for t, cyc in out.items()}


def pick_tracks(adeck, max_lag_h=12):
    """Newest cycle in the deck, then for each tech its newest cycle within
    max_lag_h of that (ECMWF etc. arrive later than the GFS)."""
    all_cycles = sorted({c for cyc in adeck.values() for c in cyc})
    if not all_cycles:
        return None, {}
    newest = dt.datetime.strptime(all_cycles[-1], "%Y%m%d%H")
    tracks = {}
    for tech, cyc in adeck.items():
        if tech in SKIP:
            continue
        best = max(cyc)
        t = dt.datetime.strptime(best, "%Y%m%d%H")
        if (newest - t).total_seconds() / 3600 <= max_lag_h and len(cyc[best]) >= 2:
            tracks[tech] = {"cycle": best, "pts": cyc[best], "lag": int((newest - t).total_seconds() // 3600)}
    return newest, tracks


def read_cone(zbytes: bytes):
    """Polygon(s) of the NHC forecast cone from the GIS zip. Returns list of
    (lons, lats) or [] if anything goes wrong — the cone is a nice-to-have."""
    try:
        import shapefile  # pyshp
        z = zipfile.ZipFile(io.BytesIO(zbytes))
        base = next(n[:-4] for n in z.namelist() if n.endswith("_pgn.shp") or n.endswith("pgn.shp"))
        r = shapefile.Reader(shp=io.BytesIO(z.read(base + ".shp")), dbf=io.BytesIO(z.read(base + ".dbf")),
                             shx=io.BytesIO(z.read(base + ".shx")))
        polys = []
        for shp in r.shapes():
            parts = list(shp.parts) + [len(shp.points)]
            for a, b in zip(parts, parts[1:]):
                pts = np.array(shp.points[a:b])
                polys.append((pts[:, 0], pts[:, 1]))
        return polys
    except Exception as e:  # noqa: BLE001
        log.warning("cone unavailable: %s", e)
        return []


# ---------------------------------------------------------------- plots ----

def extent_for(tracks, btrack, pad=4):
    lats, lons = [], []
    for tr in tracks.values():
        for _, la, lo, *_ in tr["pts"]:
            lats.append(la); lons.append(lo)
    for _, la, lo, *_ in btrack[-20:]:
        lats.append(la); lons.append(lo)
    if not lats:
        return (-100, -40, 5, 45)
    lo0, lo1 = np.percentile(lons, [2, 98]); la0, la1 = np.percentile(lats, [2, 98])
    w = max(lo1 - lo0, 12); h = max(la1 - la0, 8)
    cx, cy = (lo0 + lo1) / 2, (la0 + la1) / 2
    w = max(w, h * 1.5); h = max(h, w / 1.5)
    return (cx - w / 2 - pad, cx + w / 2 + pad, max(cy - h / 2 - pad, -5), min(cy + h / 2 + pad, 70))


def plot_track(storm, newest, tracks, btrack, cone, dest: Path):
    fig = plt.figure(figsize=(12, 8.5), dpi=100)
    ax = fig.add_axes([0.01, 0.05, 0.98, 0.86], projection=PC)
    ax.set_extent(extent_for(tracks, btrack), crs=PC)
    ax.set_facecolor("#eef4f8")
    add_basemap(ax)
    for lons, lats in cone:
        ax.fill(lons, lats, color="#ffffff", alpha=0.55, transform=PC, zorder=3)
        ax.plot(lons, lats, color="#000", lw=0.6, transform=PC, zorder=3)

    n_gefs = n_ecens = 0
    for tech, tr in tracks.items():
        pts = np.array([(lo, la) for _, la, lo, *_ in tr["pts"]])
        if (tech.startswith(GEFS_PREFIX) and tech[2:].isdigit()) or tech == "AC00":
            ax.plot(pts[:, 0], pts[:, 1], color="#d62728", lw=0.7, alpha=0.35, transform=PC, zorder=4); n_gefs += 1
        elif tech.startswith(ECENS_PREFIX) and tech[2:].isdigit():
            ax.plot(pts[:, 0], pts[:, 1], color="#1f4ed8", lw=0.7, alpha=0.35, transform=PC, zorder=4); n_ecens += 1
    handles = []
    if n_gefs:
        handles.append(plt.Line2D([], [], color="#d62728", lw=0.8, alpha=0.5, label=f"GEFS members ({n_gefs})"))
    if n_ecens:
        handles.append(plt.Line2D([], [], color="#1f4ed8", lw=0.8, alpha=0.5, label=f"ECMWF ens members ({n_ecens})"))

    seen = set()
    for tech, (label, color, lw) in MODELS.items():
        if tech not in tracks or tech in INTENSITY_ONLY or label in seen:
            continue
        seen.add(label)
        tr = tracks[tech]
        pts = np.array([(lo, la) for _, la, lo, *_ in tr["pts"]])
        ls = "-" if tr["lag"] == 0 else "--"
        z = 8 if tech == "OFCL" else 6
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, ls=ls, transform=PC, zorder=z,
                path_effects=[pe.withStroke(linewidth=lw + 1.2, foreground="white")])
        for tau, la, lo, *_ in tr["pts"]:
            if tau % 24 == 0 and tau > 0:
                ax.plot(lo, la, "o", ms=5 if tech == "OFCL" else 3.5, color=color, mec="white", mew=0.5, transform=PC, zorder=z + 1)
        lag = f"  ({tr['cycle'][-2:]}Z)" if tr["lag"] else ""
        handles.append(plt.Line2D([], [], color=color, lw=lw, ls=ls, label=label + lag))

    if btrack:
        b = np.array([(lo, la) for _, la, lo, *_ in btrack])
        ax.plot(b[:, 0], b[:, 1], color="#222", lw=2.2, transform=PC, zorder=9)
        ax.plot(b[:, 0], b[:, 1], "o", ms=3, color="#222", mec="white", mew=0.6, transform=PC, zorder=9)
        ax.plot(b[-1, 0], b[-1, 1], marker=(8, 2, 0), ms=14, color="#000", transform=PC, zorder=10)
        handles.insert(0, plt.Line2D([], [], color="#222", lw=2.2, marker="o", ms=3, label="Observed track"))

    ax.legend(handles=handles, loc="best", fontsize=8, ncol=2, framealpha=0.92, borderpad=0.6)
    fig.text(0.01, 0.965, f"{storm['title']}  —  model track guidance", fontsize=14, fontweight="bold", va="center")
    fig.text(0.01, 0.93, f"Latest cycle {newest:%a %d %b %Y %HZ}   ·   dots every 24 h   ·   dashed = model from an earlier cycle   ·   white shading = NHC forecast cone",
             fontsize=9.5, color="#333", va="center")
    fig.text(0.01, 0.015, "WxModels · data: NOAA/NHC ATCF a-deck & b-deck", fontsize=8.5, color="#666", va="center")
    fig.savefig(dest, facecolor="white"); plt.close(fig)


def plot_intensity(storm, newest, tracks, btrack, dest: Path):
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=100)
    handles = []
    for tech, tr in tracks.items():
        if (tech.startswith(GEFS_PREFIX) and tech[2:].isdigit()) or tech == "AC00":
            xs = [p[0] for p in tr["pts"] if p[3]]; ys = [p[3] for p in tr["pts"] if p[3]]
            ax.plot(xs, ys, color="#d62728", lw=0.6, alpha=0.3)
    seen = set()
    for tech, (label, color, lw) in MODELS.items():
        if tech not in tracks or label in seen:
            continue
        pts = [p for p in tracks[tech]["pts"] if p[3]]
        if len(pts) < 2:
            continue
        seen.add(label)
        ls = "-" if tracks[tech]["lag"] == 0 else "--"
        ax.plot([p[0] for p in pts], [p[3] for p in pts], color=color, lw=lw, ls=ls, marker="o", ms=2.5)
        handles.append(plt.Line2D([], [], color=color, lw=lw, ls=ls, label=label))
    for y, name in [(34, "TS"), (64, "Cat 1"), (83, "Cat 2"), (96, "Cat 3"), (113, "Cat 4"), (137, "Cat 5")]:
        ax.axhline(y, color="#bbb", lw=0.6, ls=":", zorder=0)
        ax.text(ax.get_xlim()[1] if False else 168, y + 1, name, fontsize=7.5, color="#888", ha="right", va="bottom")
    ax.set_xlim(0, 168); ax.set_xticks(range(0, 169, 12))
    ax.set_ylim(0, max(150, ax.get_ylim()[1]))
    ax.set_xlabel("Forecast hour"); ax.set_ylabel("Max sustained wind (kt)")
    ax.grid(axis="x", color="#eee")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.text(0.01, 0.97, f"{storm['title']}  —  intensity guidance", fontsize=14, fontweight="bold", va="top")
    fig.text(0.01, 0.905, f"Latest cycle {newest:%a %d %b %Y %HZ}", fontsize=9.5, color="#333", va="top")
    fig.subplots_adjust(top=0.85, bottom=0.11, left=0.06, right=0.84)
    fig.savefig(dest, facecolor="white"); plt.close(fig)


def plot_overview(basin, storms, dest: Path, areas=()):
    bbox = {"al": (-100, -15, 5, 50), "ep": (-150, -85, 3, 35), "cp": (-180, -140, 3, 35)}.get(basin, (-100, -15, 5, 50))
    fig = plt.figure(figsize=(12, 7), dpi=100)
    ax = fig.add_axes([0.01, 0.05, 0.98, 0.86], projection=PC)
    ax.set_extent(bbox, crs=PC); ax.set_facecolor("#eef4f8")
    add_basemap(ax)
    for a in areas:
        if a["basin"] not in (basin, ""):
            continue
        try:
            p7 = int(str(a["prob7"]).rstrip("%"))
        except ValueError:
            p7 = 0
        col = "#e6c200" if p7 < 40 else "#f28c28" if p7 < 60 else "#d0021b"
        ax.fill(a["lons"], a["lats"], color=col, alpha=0.25, transform=PC, zorder=2)
        ax.plot(a["lons"], a["lats"], color=col, lw=1.2, transform=PC, zorder=3)
        ax.text(np.mean(a["lons"]), np.mean(a["lats"]), f"{a['prob2']} / {a['prob7']}", fontsize=8.5, fontweight="bold",
                ha="center", va="center", transform=PC, zorder=8, path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    for s in storms:
        for lons, lats in s.get("_cone", []):
            ax.fill(lons, lats, color="#ffffff", alpha=0.5, transform=PC, zorder=3)
            ax.plot(lons, lats, color="#000", lw=0.5, transform=PC, zorder=3)
        tr = s.get("_tracks", {}).get("OFCL")
        if tr:
            pts = np.array([(lo, la) for _, la, lo, *_ in tr["pts"]])
            ax.plot(pts[:, 0], pts[:, 1], color="#000", lw=2, transform=PC, zorder=6)
        b = s.get("_btrack", [])
        if b:
            bb = np.array([(lo, la) for _, la, lo, *_ in b])
            ax.plot(bb[:, 0], bb[:, 1], color="#444", lw=1.6, transform=PC, zorder=5)
        if s.get("lat") is None:
            continue
        sym = "x" if s.get("invest") else (8, 2, 0)
        ax.plot(s["lon"], s["lat"], marker=sym, ms=11 if s.get("invest") else 13, color="#000", mew=2, transform=PC, zorder=7)
        lab = s["name"] + (f"\n{s['intensity']} kt" if s.get("intensity") else "")
        ax.text(s["lon"] + 0.8, s["lat"] + 0.8, lab, fontsize=9, fontweight="bold",
                transform=PC, zorder=8, path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    names = {"al": "Atlantic", "ep": "East Pacific", "cp": "Central Pacific"}
    fig.text(0.01, 0.965, f"Active systems — {names.get(basin, basin)}", fontsize=14, fontweight="bold", va="center")
    fig.text(0.01, 0.93, f"Updated {dt.datetime.now(dt.timezone.utc):%a %d %b %Y %H:%M}Z   ·   black line = NHC forecast, shading = cone   ·   X = invest   ·   outlook areas labelled 2-day / 7-day %",
             fontsize=9.5, color="#333", va="center")
    fig.text(0.01, 0.015, "WxModels · data: NOAA/NHC", fontsize=8.5, color="#666", va="center")
    fig.savefig(dest, facecolor="white"); plt.close(fig)


# ---------------------------------------------------------------- data -----

def storm_title(s):
    return s["name"] if s.get("invest") else f"{s['classification']} {s['name']}"


def load_storms(session):
    r = session.get(CURRENT_STORMS, timeout=30); r.raise_for_status()
    out = []
    for s in r.json().get("activeStorms", []):
        sid = s["id"].lower()  # e.g. al092026
        out.append({
            "id": sid, "basin": sid[:2], "name": s["name"], "classification": s["classification"],
            "lat": float(s["latitudeNumeric"]), "lon": float(s["longitudeNumeric"]),
            "intensity": int(s["intensity"]), "pressure": int(s["pressure"]) if s.get("pressure") else None,
            "movement": f"{s.get('movementDir', '')}° at {s.get('movementSpeed', '')} kt",
            "advisory": s.get("lastUpdate"),
            "nhc_url": f"https://www.nhc.noaa.gov/graphics_{'at' if sid[:2]=='al' else 'ep'}{sid[3]}.shtml",
        })
    return out


def find_invests(session, year: int, known_ids: set[str]) -> list[dict]:
    """Invests (90–99) in the Atlantic / E. Pacific with a recently updated
    a-deck. NHC's storm feed doesn't list them; the ATCF directory does."""
    out = []
    now = dt.datetime.now(dt.timezone.utc)
    for basin in ("al", "ep"):
        for n in range(90, 100):
            sid = f"{basin}{n}{year}"
            if sid in known_ids:
                continue
            try:
                r = session.head(ADECK.format(sid=sid), timeout=20)
                if r.status_code != 200 or "Last-Modified" not in r.headers:
                    continue
                mod = dt.datetime.strptime(r.headers["Last-Modified"], "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=dt.timezone.utc)
                if (now - mod).total_seconds() / 3600 > INVEST_MAX_AGE_H:
                    continue
            except Exception as e:  # noqa: BLE001
                log.info("invest probe %s: %s", sid, e); continue
            out.append({"id": sid, "basin": basin, "name": f"Invest {n}{'L' if basin == 'al' else 'E'}",
                        "classification": "Invest", "lat": None, "lon": None, "intensity": None, "pressure": None,
                        "movement": "", "advisory": mod.isoformat(), "invest": True,
                        "nhc_url": "https://www.nhc.noaa.gov/gtwo.php?basin=" + ("atlc" if basin == "al" else "epac")})
    return out


def fill_from_bdeck(storm, btrack, bdeck_rows):
    """Invests have no advisory: take position/intensity from the last best-track fix."""
    if btrack:
        _, la, lo, vm, mp = btrack[-1]
        storm["lat"], storm["lon"] = la, lo
        storm["intensity"] = vm; storm["pressure"] = mp


def read_outlook_areas(session):
    """NHC graphical outlook areas: [{lons, lats, prob2, prob7, basin}]"""
    try:
        import shapefile
        z = zipfile.ZipFile(io.BytesIO(session.get(GTWO, timeout=60).content))
        names = [n[:-4] for n in z.namelist() if n.endswith(".shp") and "areas" in n.lower()]
        out = []
        for base in names:
            r = shapefile.Reader(shp=io.BytesIO(z.read(base + ".shp")), dbf=io.BytesIO(z.read(base + ".dbf")),
                                 shx=io.BytesIO(z.read(base + ".shx")))
            fields = [f[0] for f in r.fields[1:]]
            for sr in r.shapeRecords():
                rec = dict(zip(fields, sr.record))
                pts = np.array(sr.shape.points)
                if len(pts) < 3:
                    continue
                p2 = rec.get("PROB2DAY") or rec.get("prob2day") or ""
                p7 = rec.get("PROB7DAY") or rec.get("prob7day") or ""
                raw = str(rec.get("BASIN", rec.get("basin", ""))).strip().lower()
                basin = "al" if raw.startswith(("al", "atl")) else "ep" if raw.startswith(("ep", "pac", "east")) else "cp" if raw.startswith(("cp", "cent")) else None
                if basin is None:                     # fall back to geography
                    basin = "al" if pts[:, 0].mean() > -100 else "ep"
                out.append({"lons": pts[:, 0], "lats": pts[:, 1], "prob2": str(p2).strip(), "prob7": str(p7).strip(),
                            "basin": basin, "area": str(rec.get("AREA", rec.get("area", ""))).strip()})
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("outlook areas unavailable: %s", e)
        return []


BOM_WAVES = "https://www.bom.gov.au/clim_data/IDCK000080/{wave}.tropical_waves.daily.glb_tropics.{ymd}.hr.png"
BOM_WAVE_TYPES = [("mjo", "Madden-Julian Oscillation"), ("kelvin", "Kelvin wave"), ("eq_rossby", "Equatorial Rossby wave"), ("gravity", "Mixed Rossby-gravity wave")]


def fetch_bom_waves(session, out_dir: Path, back_days: int = 30, ahead_days: int = 45) -> list:
    """BoM 'tropical atmospheric waves' daily frames — one image per wave type
    (MJO, Kelvin, equatorial Rossby, MRG) per day, observed and forecast days.
    Returns [{date, images: {type: path}}]. Cached on disk between runs. CC BY (BoM)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # BoM refuses non-browser clients; present browser-like headers for these requests only
    bom = requests.Session()
    bom.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36",
                        "Accept": "image/avif,image/webp,image/png,image/*,*/*;q=0.8", "Referer": "https://www.bom.gov.au/climate/mjo/",
                        "Accept-Language": "en-US,en;q=0.9"})
    today = dt.datetime.now(dt.timezone.utc).date()
    frames, misses, seen_types, statuses = [], 0, set(), {}
    for k in range(-back_days, ahead_days + 1):
        d = today + dt.timedelta(days=k); ymd = d.strftime("%Y%m%d")
        images = {}
        for wave, _ in BOM_WAVE_TYPES:
            dest = out_dir / f"{wave}_{ymd}.png"
            if not dest.exists():
                try:
                    r = bom.get(BOM_WAVES.format(wave=wave, ymd=ymd), timeout=30)
                    statuses[r.status_code] = statuses.get(r.status_code, 0) + 1
                    if r.status_code == 200 and len(r.content) > 5000:
                        dest.write_bytes(r.content)
                    else:
                        continue
                except requests.RequestException as e:
                    statuses["error"] = statuses.get("error", 0) + 1; continue
            images[wave] = f"images/mjo/{dest.name}"; seen_types.add(wave)
        if not images:
            misses += 1
            if k > 0 and misses > 6:                  # past the end of the forecast frames
                break
            continue
        misses = 0
        frames.append({"date": d.isoformat(), "images": images})
    # BoM's RMM charts can't be hot-linked (they block other sites), so copy them too:
    # the phase-space diagram (last 40 days) and the daily RMM index series (dated; take the newest that exists)
    try:
        r = bom.get("https://www.bom.gov.au/clim_data/IDCKGEM000/rmm.phase.Last40days.gif", timeout=30)
        if r.status_code == 200 and len(r.content) > 5000:
            (out_dir / "rmm_phase.gif").write_bytes(r.content)
        else:
            log.info("BoM RMM phase diagram -> HTTP %s", r.status_code)
        for k in range(0, 6):
            d = today - dt.timedelta(days=k)
            r = bom.get(f"https://www.bom.gov.au/clim_data/IDCK000080/mjo_rmm.daily.{d:%Y%m%d}.png", timeout=30)
            if r.status_code == 200 and len(r.content) > 5000:
                (out_dir / "rmm_daily.png").write_bytes(r.content); break
        # BoM Hovmöllers (OLR and 850 hPa wind anomalies, 15°S–15°N) replace CPC's, whose page image is years stale
        for name, url in [("hov_olr.png", "https://www.bom.gov.au/clim_data/IDCKGEM000/olr_hovs_183_-15_15.ps.png"),
                          ("hov_u850.png", "https://www.bom.gov.au/clim_data/IDCKGEM000/winds_hovs_u850_183_-15_15.ps.png")]:
            r = bom.get(url, timeout=30)
            if r.status_code == 200 and len(r.content) > 5000:
                (out_dir / name).write_bytes(r.content)
            else:
                log.info("BoM %s -> HTTP %s", name, r.status_code)
    except requests.RequestException as e:
        log.info("BoM RMM charts failed: %s", e)
    missing_types = [w for w, _ in BOM_WAVE_TYPES if w not in seen_types]
    log.info("BoM tropical waves: %d frames; types found %s%s; HTTP responses %s", len(frames), sorted(seen_types),
             f"; NOT found (name guess wrong?): {missing_types}" if missing_types else "", statuses)
    return frames


def synthetic_storm():
    """One fake Atlantic hurricane with a plausible a-deck for testing."""
    rng = np.random.default_rng(7)
    cyc = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    cyc = cyc.replace(hour=(cyc.hour // 6) * 6, tzinfo=None)
    lines = []
    def track(tech, cycle, spread, curve, vmax0, dv, taus=range(0, 169, 6)):
        e_lat, e_lon, e_v = rng.normal(size=3)
        for tau in taus:
            t = tau / 24
            lat = 16 + 2.2 * t + curve * t**2 + spread * e_lat * t * 0.4
            lon = -55 - 6.5 * t + 0.5 * t**2 + spread * e_lon * t * 0.5
            v = max(25, vmax0 + dv * t - 3 * t**2 + e_v * 4 * t)
            lines.append(f"AL, 09, {cycle:%Y%m%d%H}, 03, {tech}, {tau:3d}, {abs(lat)*10:.0f}N, {abs(lon)*10:.0f}W, {v:.0f}, {1010-v:.0f}, XX")
    track("OFCL", cyc, 0, 0.35, 75, 18, range(0, 121, 12))
    for tech in ["AVNO", "EMXI", "UKX", "CMC", "HFSA", "HFSB", "HWRF", "NVGM", "TVCN", "HCCA", "AEMN", "ICON", "FNV3"]:
        track(tech, cyc, 1.5, 0.35 + rng.normal() * 0.15, 75, 18 + rng.normal() * 6)
    track("ECAI", cyc - dt.timedelta(hours=6), 1.5, 0.3, 75, 15)
    for i in range(1, 31):
        track(f"AP{i:02d}", cyc, 2.5, 0.35 + rng.normal() * 0.2, 75, 18 + rng.normal() * 10)
    for tech in ["SHIP", "DSHP", "LGEM", "IVCN"]:
        track(tech, cyc, 0, 0, 75, 20 + rng.normal() * 4)
    b = []
    for k in range(8, 0, -1):
        c = cyc - dt.timedelta(hours=6 * k)
        b.append(f"AL, 09, {c:%Y%m%d%H},   , BEST,   0, {(16 - 0.5*k)*10:.0f}N, {(55 - 1.6*k)*10:.0f}W, {75-6*k:.0f}, {1010-75+6*k:.0f}, HU")
    storm = {"id": "al092026", "basin": "al", "name": "Testorm", "classification": "Hurricane", "lat": 16.0, "lon": -55.0,
             "intensity": 75, "pressure": 985, "movement": "290° at 12 kt", "advisory": cyc.isoformat() + "Z",
             "nhc_url": "https://www.nhc.noaa.gov/"}
    return storm, "\n".join(lines), "\n".join(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    session = requests.Session(); session.headers["User-Agent"] = "wxmodels-tropical (github actions)"
    OUT.mkdir(parents=True, exist_ok=True)

    areas = []
    if args.synthetic:
        s, a_text, b_text = synthetic_storm()
        storms = [s]; decks = {s["id"]: (a_text, b_text, b"")}
        areas = [{"lons": np.array([-45, -35, -33, -42, -48]), "lats": np.array([10, 11, 16, 18, 14]),
                  "prob2": "20%", "prob7": "60%", "basin": "al", "area": ""}]
    else:
        storms = load_storms(session)
        year = dt.datetime.now(dt.timezone.utc).year
        storms += find_invests(session, year, {s["id"] for s in storms})
        areas = read_outlook_areas(session)
        decks = {}
        for s in storms:
            try:
                a_text = gzip.decompress(session.get(ADECK.format(sid=s["id"]), timeout=60).content).decode("latin-1")
            except Exception as e:  # noqa: BLE001
                log.warning("no a-deck for %s: %s", s["id"], e); a_text = ""
            try:
                b_text = session.get(BDECK.format(sid=s["id"]), timeout=60).text
            except Exception as e:  # noqa: BLE001
                log.warning("no b-deck for %s: %s", s["id"], e); b_text = ""
            try:
                cone_bytes = session.get(CONE.format(sid=s["id"]), timeout=60).content
            except Exception:  # noqa: BLE001
                cone_bytes = b""
            decks[s["id"]] = (a_text, b_text, cone_bytes)

    from plots import _basemap_layers
    _basemap_layers()
    result = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(), "storms": [], "overviews": {}}
    for s in storms:
        a_text, b_text, cone_bytes = decks[s["id"]]
        s["title"] = storm_title(s)
        adeck = parse_atcf(a_text)
        newest, tracks = pick_tracks(adeck)
        bdeck = parse_atcf(b_text)
        btrack = sorted({c: v[0] for c, v in bdeck.get("BEST", {}).items()}.items())
        btrack = [(0, la, lo, vm, mp) for _, (_, la, lo, vm, mp) in btrack]
        cone = read_cone(cone_bytes) if cone_bytes else []
        if s.get("invest"):
            fill_from_bdeck(s, btrack, bdeck)
            if s["lat"] is None and tracks:                   # no best track yet: use any model's t=0
                _, la, lo, *_ = next(iter(tracks.values()))["pts"][0]
                s["lat"], s["lon"] = la, lo
        s["_tracks"], s["_btrack"], s["_cone"] = tracks, btrack, cone
        sdir = OUT / s["id"]; sdir.mkdir(exist_ok=True)
        entry = {k: v for k, v in s.items() if not k.startswith("_")}
        if newest:
            plot_track(s, newest, tracks, btrack, cone, sdir / "track.png")
            plot_intensity(s, newest, tracks, btrack, sdir / "intensity.png")
            entry["cycle"] = newest.strftime("%Y-%m-%dT%H:00Z")
            entry["models"] = sorted({MODELS[t][0] for t in tracks if t in MODELS})
            entry["gefs_members"] = sum(1 for t in tracks if t.startswith(GEFS_PREFIX) and t[2:].isdigit())
            entry["ecens_members"] = sum(1 for t in tracks if t.startswith(ECENS_PREFIX) and t[2:].isdigit())
            entry["images"] = {"track": f"images/tropical/{s['id']}/track.png",
                               "intensity": f"images/tropical/{s['id']}/intensity.png"}
            unknown = sorted(t for t in tracks if t not in MODELS and t != "AC00" and not t.startswith((GEFS_PREFIX, ECENS_PREFIX)))
            if unknown:
                log.info("%s: techs in a-deck not in MODELS table: %s", s["id"], " ".join(unknown))
        else:
            log.warning("%s: no usable a-deck yet", s["id"])
        result["storms"].append(entry)
        log.info("%s %s: %d model tracks", s["title"], s["id"], len(tracks))

    for basin in sorted({s["basin"] for s in storms} | {a["basin"] for a in areas} | {"al"}):
        dest = OUT / f"overview_{basin}.png"
        plot_overview(basin, [s for s in storms if s["basin"] == basin], dest, areas)
        result["overviews"][basin] = f"images/tropical/overview_{basin}.png"
    if not args.synthetic:
        try:
            result["mjo_frames"] = fetch_bom_waves(session, SITE / "images" / "mjo")
        except Exception as e:  # noqa: BLE001
            log.warning("BoM waves unavailable: %s", e)
    result["areas"] = [{"basin": a["basin"], "prob2": a["prob2"], "prob7": a["prob7"],
                        "lat": round(float(np.mean(a["lats"])), 1), "lon": round(float(np.mean(a["lons"])), 1)} for a in areas]

    if storage.enabled():
        storage.delete_prefix("images/tropical/")
        storage.upload_dir(OUT, "images/tropical")
        storage.put_json(result, "tropical.json")
        import shutil; shutil.rmtree(OUT, ignore_errors=True)
    else:
        (SITE / "tropical.json").write_text(json.dumps(result, indent=1))
    log.info("done: %d storms", len(storms))


if __name__ == "__main__":
    main()
