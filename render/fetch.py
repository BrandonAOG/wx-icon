"""
Download GFS subsets from NOAA NOMADS and load them into plain numpy arrays.

The grib_filter CGI lets us request only the variables/levels/bbox we need, so
each forecast hour is a few MB. GRIB decoding uses cfgrib (needs the eccodes
system library: `apt install libeccodes-dev` or `conda install eccodes`).
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import re
import time

os.environ.setdefault("TQDM_DISABLE", "1")          # no per-file progress bars from the ECMWF client
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import requests

from config import (MODEL, NOMADS_DIR, NOMADS_FILE, NOMADS_FILTER, NOMADS_IDX,
                    PARAMS)

# ECMWF open data file layout (one GRIB2 per step, all params)
ECMWF_FILE = "https://data.ecmwf.int/forecasts/{ymd}/{hh}z/ifs/0p25/oper/{ymd}{hh}0000-{step}h-oper-fc.grib2"

log = logging.getLogger("fetch")
logging.getLogger("multiurl").setLevel(logging.WARNING)      # ECMWF client: no per-file progress bars
logging.getLogger("ecmwf.opendata").setLevel(logging.ERROR)

# cfgrib short names for each (VAR, LEVEL) pair we ask NOMADS for.
CFGRIB_NAMES = {
    ("HGT", "500_mb"): "gh",
    ("HGT", "850_mb"): "gh",
    ("HGT", "1000_mb"): "gh",
    ("ABSV", "500_mb"): "absv",
    ("PRMSL", "mean_sea_level"): "prmsl",
    ("APCP", "surface"): "tp",
    ("TMP", "850_mb"): "t",
    ("TMP", "2_m_above_ground"): "t2m",
    ("UGRD", "850_mb"): "u",
    ("VGRD", "850_mb"): "v",
    ("UGRD", "10_m_above_ground"): "u10",
    ("VGRD", "10_m_above_ground"): "v10",
    ("PWAT", "entire_atmosphere_\\(considered_as_a_single_layer\\)"): "pwat",
    ("CAPE", "surface"): "cape",
}


def _candidate_cycles(now: dt.datetime):
    """Cycles in MODEL['cycles'], newest first, that are old enough to be complete."""
    start = (now - dt.timedelta(hours=MODEL["min_age_hours"])).replace(minute=0, second=0, microsecond=0)
    c = start
    for _ in range(48):
        if c.hour in MODEL["cycles"]:
            yield c
        c -= dt.timedelta(hours=1)


def latest_available_run(now: dt.datetime | None = None,
                         session: requests.Session | None = None) -> dt.datetime:
    """Newest cycle that's actually on the server."""
    now = now or dt.datetime.now(dt.timezone.utc)
    session = session or requests.Session()
    if MODEL["source"] not in ("nomads", "nomads_grid"):
        # A run is complete when its last step's file exists. Some cycles are
        # published to a shorter range, so probe possible final steps longest-first.
        for cand in _candidate_cycles(now):
            if run_max_hour(cand, session) is not None:
                return cand
            log.info("%s %s not complete yet", MODEL["name"], cand.strftime("%Y%m%d %HZ"))
        if MODEL["source"].startswith("ecmwf"):
            ecmwf_explain(now, session)
        raise RuntimeError(f"No complete {MODEL['name']} run found in the last 48 h")
    # NOMADS models: usable once the probe hour's index exists
    for cand in _candidate_cycles(now):
        if run_max_hour(cand, session) is not None:
            return cand
    raise RuntimeError("No GFS run found on NOMADS in the last 48 h")


def ecmwf_explain(now, session):
    """When an ECMWF model can't be found, list what the open-data server
    actually has for the most recent cycle so the layout can be corrected."""
    cand = next(_candidate_cycles(now))
    ymd, hh = cand.strftime("%Y%m%d"), cand.strftime("%H")
    for url in [f"https://data.ecmwf.int/forecasts/{ymd}/{hh}z/",
                f"https://data.ecmwf.int/forecasts/{ymd}/{hh}z/{ecmwf_model_name()}/",
                f"https://data.ecmwf.int/forecasts/{ymd}/{hh}z/{ecmwf_model_name()}/0p25/",
                f"https://data.ecmwf.int/forecasts/{ymd}/{hh}z/{ecmwf_model_name()}/0p25/enfo/"]:
        entries = _listing(session, url, retries=1)
        files = [e for e in entries if not e.endswith("/")]
        log.info("ECMWF listing %s -> dirs %s, %d files%s", url, [e for e in entries if e.endswith("/")][:12], len(files),
                 (" e.g. " + " ".join(files[:4])) if files else "")
    for cand in _candidate_cycles(now):
        url = NOMADS_IDX.format(ymd=cand.strftime("%Y%m%d"), hh=cand.strftime("%H"))
        try:
            if session.head(url, timeout=20).status_code == 200:
                return cand
        except requests.RequestException as e:
            log.warning("HEAD %s failed: %s", url, e)
    raise RuntimeError("No GFS run found on NOMADS in the last 48 h")


def _probe_url(run: dt.datetime, step: int, session=None) -> str | None:
    """A file whose presence means `step` of this run is published."""
    src = MODEL["source"]
    if src == "ecmwf_opendata":
        return ECMWF_FILE.format(ymd=run.strftime("%Y%m%d"), hh=run.strftime("%H"), step=step)
    if src == "cmc":
        return None            # handled in run_max_hour via cmc_step_complete
    if src == "icon":
        return icon_urls(run, step, {("msl", None)})[0]
    if src == "gefs":
        return GEFS_IDX.format(ymd=run.strftime("%Y%m%d"), hh=run.strftime("%H"), mem=MODEL["members"][-1], fhr=step)
    if src in ("ecmwf_ens", "ecmwf_aifs_ens"):
        url = ECMWF_ENS_FILE.format(ymd=run.strftime("%Y%m%d"), hh=run.strftime("%H"), step=step, model=ecmwf_model_name())
        # AIFS-ENS publishes control and perturbed members as separate files (-cf / -pf); IFS ENS combines them (-ef)
        return url.replace("-enfo-ef.grib2", "-enfo-cf.grib2") if src == "ecmwf_aifs_ens" else url
    if src == "aigefs":
        return aigefs_url(run, step, MODEL["members"][-1]) + ".idx"
    return None


def run_max_hour(run: dt.datetime, session: requests.Session | None = None) -> int | None:
    """Furthest forecast hour available for this run, or None if the run isn't
    complete at any known range. GFS is always the full range."""
    session = session or requests.Session()
    if MODEL["source"] == "nomads_grid":
        for last in MODEL.get("probe_max_hours", [MODEL["hours"][-1]]):
            url = MODEL["idx"].format(ymd=run.strftime("%Y%m%d"), hh=run.strftime("%H"), fhr=last)
            try:
                if session.head(url, timeout=20).status_code == 200:
                    return last
            except requests.RequestException as e:
                log.warning("HEAD %s failed: %s", url, e)
        return None
    if MODEL["source"] == "nomads":
        for last in MODEL.get("probe_max_hours", [MODEL["hours"][-1]]):
            url = NOMADS_IDX.format(ymd=run.strftime("%Y%m%d"), hh=run.strftime("%H")).replace("f000.idx", f"f{last:03d}.idx")
            try:
                if session.head(url, timeout=20).status_code == 200:
                    return last
            except requests.RequestException as e:
                log.warning("HEAD %s failed: %s", url, e)
        return None
    for last in MODEL.get("probe_max_hours", [MODEL["hours"][-1]]):
        if MODEL["source"] == "cmc":
            # also require the last 6-hourly step before the end, so a run whose
            # tail happens to be up first isn't mistaken for complete
            if cmc_step_complete(run, last, session) and cmc_step_complete(run, last - 6, session):
                return last
            continue
        if MODEL["source"] == "geps":
            if geps_step_complete(run, last, session) and geps_step_complete(run, last - 6, session):
                return last
            continue
        url = _probe_url(run, last, session)
        try:
            r = session.get(url, timeout=30, allow_redirects=True, stream=True); r.close()
            if r.status_code == 200:
                return last
            log.info("probe %s -> HTTP %s", url, r.status_code)
        except requests.RequestException as e:
            log.warning("probe %s failed: %s", url, e)
    return None


def all_fetch_pairs(param_ids: list[str]) -> set[tuple[str, str]]:
    from config import products
    table = products()
    pairs: set[tuple[str, str]] = set()
    for pid in param_ids:
        pairs.update(table[pid]["fetch"])
    return pairs


def spec_pairs(param_ids: list[str]) -> set[tuple]:
    """Generic (field, level) pairs for non-GFS sources, minus fields the
    current source can't supply (e.g. vorticity, computed from u/v instead)."""
    from config import SOURCE_FIELDS
    have = SOURCE_FIELDS.get(MODEL["source"], set())
    pairs: set[tuple] = set()
    for pid in param_ids:
        for name, lev in (PARAMS[pid].get("spec") or []):
            if name in have:
                pairs.add((name, lev))
            elif name == "vo":
                pairs.update({("u", lev), ("v", lev)})
    return pairs


ecmwf_pairs = spec_pairs   # backwards-compatible name


def prev_steps(param_ids: list[str]) -> dict:
    """{offset: {"fetch": set(pairs), "ecmwf": set(pairs)}} merged across products.
    offset is an int (hours back) or "f0"."""
    out: dict = {}
    from config import products
    for pid in param_ids:
        spec = products()[pid].get("prev")
        if not spec:
            continue
        for off in spec["offsets"]:
            slot = out.setdefault(off, {"fetch": set(), "ecmwf": set()})
            slot["fetch"].update(spec.get("fetch", []))
            slot["ecmwf"].update(spec.get("spec", spec.get("ecmwf", [])))
    return out


def step_for(fhr: int, offset) -> int | None:
    """Forecast step to fetch for a previous-step offset, or None if n/a."""
    step = 0 if offset == "f0" else fhr - int(offset)
    return step if 0 <= step < fhr else None


def ecmwf_requests(pairs: set[tuple], step: int) -> list[dict]:
    """Open-data requests for one step: one per pressure level, one for
    single-level fields. tp doesn't exist at step 0."""
    pl, sfc = {}, set()
    for name, lev in pairs:
        if lev is None:
            sfc.add(name)
        else:
            pl.setdefault(lev, set()).add(name)
    if step == 0:
        sfc.discard("tp")
    reqs = [{"type": "fc", "stream": "oper", "step": step, "levtype": "pl", "levelist": lev, "param": sorted(n)}
            for lev, n in pl.items()]
    if sfc:
        reqs.append({"type": "fc", "stream": "oper", "step": step, "levtype": "sfc", "param": sorted(sfc)})
    return reqs


def download_ecmwf(run: dt.datetime, step: int, pairs: set[tuple], dest: Path, retries: int = 4) -> Path:
    """Fetch all messages for one hour into a single GRIB file. The client uses
    the .index files to byte-range only the requested fields."""
    from ecmwf.opendata import Client
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    client = Client(source="ecmwf", model="ifs", resol="0p25")
    tmp = dest.with_suffix(".part")
    for attempt in range(retries):
        try:
            reqs = ecmwf_requests(pairs, step)
            if not reqs:
                raise RuntimeError("nothing to fetch")
            with open(tmp, "wb") as out:
                for req in reqs:
                    part = dest.with_suffix(f".{len(req['param'])}_{req.get('levelist', 'sfc')}_{req['step']}.grib2")
                    client.retrieve(date=run.strftime("%Y%m%d"), time=run.hour, target=str(part), **req)
                    out.write(part.read_bytes()); part.unlink()
            tmp.rename(dest)
            return dest
        except Exception as e:  # noqa: BLE001
            log.warning("ECMWF step %d attempt %d failed: %s", step, attempt, e)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"Failed to download ECMWF step {step}")


# grib_filter level names -> the wording NOAA uses inside .idx files
_IDX_LEVEL = {
    "surface": "surface", "mean_sea_level": "mean sea level", "2_m_above_ground": "2 m above ground",
    "10_m_above_ground": "10 m above ground", "entire_atmosphere": "entire atmosphere",
    "entire_atmosphere_\\(considered_as_a_single_layer\\)": "entire atmosphere (considered as a single layer)",
    "top_of_atmosphere": "top of atmosphere", "PV=2e-06_(Km^2/kg/s)_surface": "PV=2e-06 (Km^2/kg/s) surface",
}
_IDX_CACHE: dict = {}
_SKIP_LOGGED: set = set()


def idx_url_for(run: dt.datetime, fhr: int) -> str | None:
    ymd, hh = run.strftime("%Y%m%d"), run.strftime("%H")
    if MODEL["source"] == "nomads_grid":
        return MODEL["idx"].format(ymd=ymd, hh=hh, fhr=fhr)
    if MODEL["source"] == "nomads":
        return NOMADS_IDX.format(ymd=ymd, hh=hh).replace("f000.idx", f"f{fhr:03d}.idx")
    return None


def available_pairs(run: dt.datetime, fhr: int, pairs, session) -> set:
    """Keep only (VAR, level) pairs that this hour's .idx says are in the file.
    grib_filter answers HTTP 500 to a request naming anything absent, so this
    is what makes mixed requests survive across models. Falls back to all pairs
    if the .idx can't be read."""
    url = idx_url_for(run, fhr)
    if not url:
        return set(pairs)
    if url not in _IDX_CACHE:
        try:
            r = session.get(url, timeout=60)
            if r.status_code != 200:
                log.info("idx %s -> HTTP %s; requesting all fields", url.rsplit("/", 1)[-1], r.status_code)
                return set(pairs)
            present = set()
            for line in r.text.splitlines():
                parts = line.split(":")
                if len(parts) > 4:
                    present.add((parts[3], parts[4]))
            _IDX_CACHE[url] = present
        except requests.RequestException as e:
            log.info("idx fetch failed (%s); requesting all fields", str(e)[:60]); return set(pairs)
    present = _IDX_CACHE[url]
    norm = lambda t: t.replace(" (considered as a single layer)", "").strip()
    present_n = {(v, norm(l)) for v, l in present}
    keep, dropped = set(), []
    for var, lev in pairs:
        lev_txt = norm(_IDX_LEVEL.get(lev, lev.replace("_", " ")))
        if (var, lev_txt) in present_n:
            keep.add((var, lev))
        else:
            dropped.append(f"{var}@{lev}")
    if dropped and (fhr, tuple(sorted(dropped))) not in _SKIP_LOGGED:
        _SKIP_LOGGED.add((fhr, tuple(sorted(dropped))))
        if len(_SKIP_LOGGED) <= 6:
            log.info("f%03d: not in this model's file, skipped: %s", fhr, " ".join(sorted(dropped)))
    return keep


def grid_filter_url(run: dt.datetime, fhr: int, pairs) -> str:
    """grib_filter URL for the CONUS mesoscale models (HRRR/NAM/NBM): whole grid,
    selected fields only. These grids are Lambert, so no lat/lon subregion."""
    ymd, hh = run.strftime("%Y%m%d"), run.strftime("%H")
    q = {"dir": MODEL["dir"].format(ymd=ymd, hh=hh), "file": MODEL["file"].format(ymd=ymd, hh=hh, fhr=fhr)}
    for var, lev in pairs:
        q[f"var_{var}"] = "on"; q[f"lev_{lev}"] = "on"
    return "https://nomads.ncep.noaa.gov/cgi-bin/" + MODEL["filter"] + "?" + urlencode(q, safe="\\()")


def build_filter_url(run: dt.datetime, fhr: int, pairs: set[tuple[str, str]],
                     bbox: tuple[float, float, float, float]) -> str:
    """grib_filter URL for one forecast hour, all variables, one bounding box."""
    if MODEL["source"] == "nomads_grid":
        return grid_filter_url(run, fhr, pairs)
    lon0, lon1, lat0, lat1 = bbox
    # grib_filter wants 0..360 longitudes
    left = lon0 % 360
    right = lon1 % 360
    q = {
        "dir": NOMADS_DIR.format(ymd=run.strftime("%Y%m%d"), hh=run.strftime("%H")),
        "file": NOMADS_FILE.format(hh=run.strftime("%H"), fhr=fhr),
        "subregion": "",
        "leftlon": f"{left:g}",
        "rightlon": f"{right:g}",
        "toplat": f"{lat1:g}",
        "bottomlat": f"{lat0:g}",
    }
    for var, lev in pairs:
        q[f"var_{var}"] = "on"
        q[f"lev_{lev}"] = "on"
    return NOMADS_FILTER + "?" + urlencode(q, safe="\\()")


BACKOFF = [5, 10, 20, 30, 45, 60]

# Circuit breaker: hosts that refuse connections repeatedly (per-IP rate limiting)
# get marked dead for this process so a slice fails in minutes, not hours.
_HOST_FAILS: dict = {}
_DEAD_HOSTS: set = set()


def _host(url: str) -> str:
    return url.split("/")[2] if "//" in url else url


def note_conn_failure(url: str):
    h = _host(url); _HOST_FAILS[h] = _HOST_FAILS.get(h, 0) + 1
    if _HOST_FAILS[h] >= 3 and h not in _DEAD_HOSTS:
        _DEAD_HOSTS.add(h); log.error("%s refused %d connections in a row; giving up on it for this job", h, _HOST_FAILS[h])


def note_conn_ok(url: str):
    _HOST_FAILS[_host(url)] = 0


def host_dead(url: str) -> bool:
    return _host(url) in _DEAD_HOSTS


def download(url: str, dest: Path, session: requests.Session, retries: int = 6) -> Path:
    """NOMADS returns 500/503 freely when busy; back off progressively."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    for attempt in range(retries):
        if host_dead(url):
            raise RuntimeError(f"host unreachable: {url}")
        try:
            r = session.get(url, timeout=120)
            note_conn_ok(url)
            if r.status_code == 200 and len(r.content) > 1000:
                dest.write_bytes(r.content)
                return dest
            if r.status_code == 404:                       # not there: retrying won't help
                raise RuntimeError(f"404 {url}")
            log.warning("GET %s -> %s (%d bytes), attempt %d", url[:80], r.status_code, len(r.content), attempt + 1)
        except requests.RequestException as e:
            note_conn_failure(url)
            log.warning("GET failed (attempt %d): %s", attempt + 1, str(e)[:100])
        time.sleep(BACKOFF[min(attempt, len(BACKOFF) - 1)])
    raise RuntimeError(f"Failed to download {url}")


def _group_of(lev: str) -> str:
    if lev.endswith("_mb"):
        return "iso"
    if lev.startswith("PV"):
        return "pv"
    if lev.startswith("top_of_atmosphere"):
        return "toa"          # SBT brightness temps live in the pgrb2b file
    return "sfc"


_DEAD_GROUPS: set = set()     # groups that failed with a server error this process; skip, don't keep retrying
_SBT_FILE: str | None = None  # "a" (pgrb2) or "b" (pgrb2b), discovered from the .idx listings


def sbt_file(run: dt.datetime, session) -> str | None:
    """Which GFS file carries the SBT124 brightness temperature? Read NOAA's
    .idx listings for both and log what they say, so the answer is in the log."""
    global _SBT_FILE
    if _SBT_FILE is not None:
        return _SBT_FILE or None
    ymd, hh = run.strftime("%Y%m%d"), run.strftime("%H")
    base = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{ymd}/{hh}/atmos/"
    for tag, fname in (("a", f"gfs.t{hh}z.pgrb2.0p25.f006.idx"), ("b", f"gfs.t{hh}z.pgrb2b.0p25.f006.idx")):
        try:
            r = session.get(base + fname, timeout=60)
            if r.status_code != 200:
                log.info("idx %s -> HTTP %s", fname, r.status_code); continue
            hits = [ln for ln in r.text.splitlines() if "SBT" in ln]
            log.info("idx %s: %d SBT entries%s", fname, len(hits), (": " + " | ".join(h.split(":", 2)[-1][:40] for h in hits[:4])) if hits else "")
            if any(":SBT124:" in ln for ln in hits):
                _SBT_FILE = tag
                return tag
        except requests.RequestException as e:
            log.info("idx %s failed: %s", fname, str(e)[:80])
    _SBT_FILE = ""
    log.warning("SBT124 not found in either GFS file listing; simulated IR disabled for this job")
    return None


def download_grouped(run: dt.datetime, fhr: int, pairs: set, bbox, dest: Path,
                     session: requests.Session, retries: int = 4) -> Path:
    """GFS: NOMADS grib_filter chokes on one huge var×level request, so fetch in
    groups (isobaric / surface-ish / PV / top-of-atmosphere) and concatenate.
    A failing group is logged and skipped; the frame still renders what it can."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    pairs = available_pairs(run, fhr, pairs, session)
    groups: dict[str, set] = {}
    for var, lev in pairs:
        groups.setdefault(_group_of(lev), set()).add((var, lev))
    parts = []
    for name, grp in sorted(groups.items()):
        if name in _DEAD_GROUPS:
            continue
        part = dest.with_suffix(f".{name}.grb2")
        url = build_filter_url(run, fhr, grp, bbox)
        if name == "toa":
            which = sbt_file(run, session)
            if which is None:
                _DEAD_GROUPS.add("toa"); continue
            if which == "b":
                url = url.replace("filter_gfs_0p25.pl", "filter_gfs_0p25b.pl").replace("pgrb2.0p25", "pgrb2b.0p25")
        try:
            download(url, part, session, retries=2 if name in ("toa", "pv") else retries)
            parts.append(part)
        except RuntimeError as e:
            msg = str(e)
            log.warning("f%03d group %s failed (%s): %s", fhr, name, sorted(grp)[:3], msg[:120])
            if "404" not in msg and name in ("toa", "pv"):
                _DEAD_GROUPS.add(name)
                log.warning("group %s disabled for the rest of this job", name)
    if not parts:
        raise RuntimeError(f"All download groups failed for f{fhr:03d}")
    with open(dest, "wb") as out:
        for part in parts:
            out.write(part.read_bytes()); part.unlink()
    return dest


# ------------------------------------------------------------- ECMWF ENS ----
ECMWF_ENS_FILE = "https://data.ecmwf.int/forecasts/{ymd}/{hh}z/{model}/0p25/enfo/{ymd}{hh}0000-{step}h-enfo-ef.grib2"
# ECMWF replicates open data to public cloud buckets with the same layout; used when data.ecmwf.int errors
ECMWF_MIRRORS = ["https://data.ecmwf.int/forecasts/", "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/"]


def _mirrored(url: str):
    """The same path on each mirror, primary first."""
    for m in ECMWF_MIRRORS:
        yield url.replace(ECMWF_MIRRORS[0], m)


def ecmwf_model_name() -> str:
    return "aifs-ens" if MODEL["source"] == "ecmwf_aifs_ens" else "ifs"


def _index_select(index_url: str, session, want, retries: int = 3):
    """Parse an ECMWF open-data .index (JSON lines) and return (offset, length)
    for entries matching any of `want` = [(param, levelist or None)]. Logs the
    parameters present when a wanted one is missing. Falls back to the mirrors."""
    import json as _json
    text = None
    for attempt in range(retries):
        for url in _mirrored(index_url):
            try:
                r = session.get(url, timeout=120)
                if r.status_code == 200:
                    text = r.text; break
                log.info("index %s -> HTTP %s", url.split("/forecasts/")[-1] if "/forecasts/" in url else url.rsplit("/", 1)[-1], r.status_code)
            except requests.RequestException as e:
                log.info("index fetch failed: %s", str(e)[:80])
        if text is not None:
            break
        time.sleep(BACKOFF[min(attempt, len(BACKOFF) - 1)])
    if text is None:
        raise RuntimeError(f"index unavailable: {index_url}")
    entries = [_json.loads(line) for line in text.splitlines() if line.strip()]
    # some models publish equivalents under other names: geopotential z (m²/s²) for height gh,
    # total column water tcw for tcwv. normalise() converts them after loading.
    ALT = {"gh": ["gh", "z"], "tcwv": ["tcwv", "tcw"], "msl": ["msl", "prmsl"]}
    found, ranges = set(), []
    for wp, wl in want:
        for cand in ALT.get(wp, [wp]):
            hits = [e for e in entries if e.get("param") == cand and (wl is None or str(e.get("levelist")) == str(wl))]
            if hits:
                ranges += [(int(e["_offset"]), int(e["_length"])) for e in hits]; found.add((wp, wl)); break
    missing = [w for w in want if w not in found]
    if missing:
        present = sorted({f"{e.get('param')}@{e.get('levelist', e.get('levtype'))}" for e in entries})
        log.warning("index %s lacks %s; has: %s", index_url.rsplit("/", 1)[-1], missing, " ".join(present))
    return ranges


def _range_download(url: str, ranges, out, session, retries: int = 4):
    """Fetch byte ranges from url and append to file object `out`. Ranges are
    merged into contiguous blocks to keep the request count low."""
    ranges = sorted(ranges)
    blocks = []
    for off, ln in ranges:
        if blocks and off <= blocks[-1][1]:
            blocks[-1][1] = max(blocks[-1][1], off + ln)
        else:
            blocks.append([off, off + ln])
    for a, b in blocks:
        ok = False
        for attempt in range(retries):
            for u in _mirrored(url):                 # primary, then the cloud mirror
                try:
                    r = session.get(u, headers={"Range": f"bytes={a}-{b - 1}"}, timeout=300)
                    if r.status_code in (200, 206):
                        out.write(r.content); ok = True; break
                    log.info("range %s -> HTTP %s", u.rsplit("/", 1)[-1] + (" (mirror)" if "amazonaws" in u else ""), r.status_code)
                except requests.RequestException as e:
                    log.info("range fetch failed: %s", str(e)[:80])
            if ok:
                break
            time.sleep(BACKOFF[min(attempt, len(BACKOFF) - 1)])
        if not ok:
            raise RuntimeError(f"range download failed: {url}")


def download_ecmwf_ens_direct(run: dt.datetime, step: int, fields, dest: Path, session: requests.Session | None = None) -> Path:
    """AIFS-ENS layout: separate -enfo-cf (control) and -enfo-pf (perturbed)
    files per step. Select fields via the .index files and range-download."""
    session = session or requests.Session()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    base = ECMWF_ENS_FILE.format(ymd=run.strftime("%Y%m%d"), hh=run.strftime("%H"), step=step, model=ecmwf_model_name())
    want = [(p, lev) for p, lev in fields if not (p == "tp" and step == 0)]
    tmp = dest.with_suffix(".part")
    total = 0
    with open(tmp, "wb") as out:
        for kind in ("cf", "pf"):
            grib = base.replace("-enfo-ef.grib2", f"-enfo-{kind}.grib2")
            ranges = _index_select(grib[:-6] + ".index", session, want)
            if not ranges:
                continue
            _range_download(grib, ranges, out, session)
            total += len(ranges)
    if total == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"no matching fields in AIFS-ENS index for step {step}")
    tmp.rename(dest)
    log.info("AIFS-ENS step %d: %d fields via range requests", step, total)
    return dest


def download_ecmwf_ens(run: dt.datetime, step: int, fields, dest: Path, retries: int = 4) -> Path:
    """All 51 members (control + perturbed) of the listed fields for one step,
    byte-ranged out of the enfo file via the .index."""
    if MODEL["source"] == "ecmwf_aifs_ens":
        return download_ecmwf_ens_direct(run, step, fields, dest)
    from ecmwf.opendata import Client
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    pl, sfc = {}, set()
    for name, lev in fields:
        (pl.setdefault(lev, set()).add(name) if lev is not None else sfc.add(name))
    if step == 0:
        sfc.discard("tp")
    reqs = [{"stream": "enfo", "type": ["cf", "pf"], "step": step, "levtype": "pl", "levelist": lev, "param": sorted(n)} for lev, n in pl.items()]
    if sfc:
        reqs.append({"stream": "enfo", "type": ["cf", "pf"], "step": step, "levtype": "sfc", "param": sorted(sfc)})
    tmp = dest.with_suffix(".part")
    for attempt in range(retries):
        source = "ecmwf" if attempt % 2 == 0 else "aws"      # alternate primary / cloud mirror
        client = Client(source=source, model=ecmwf_model_name(), resol="0p25")
        try:
            with open(tmp, "wb") as out:
                for req in reqs:
                    part = dest.with_suffix(f".{req.get('levelist', 'sfc')}.grib2")
                    client.retrieve(date=run.strftime("%Y%m%d"), time=run.hour, target=str(part), **req)
                    out.write(part.read_bytes()); part.unlink()
            tmp.rename(dest)
            return dest
        except Exception as e:  # noqa: BLE001
            log.warning("ECMWF ENS step %d attempt %d (%s) failed: %s", step, attempt + 1, source, str(e)[:120])
            time.sleep(BACKOFF[min(attempt, len(BACKOFF) - 1)])
    raise RuntimeError(f"Failed to download ECMWF ENS step {step}")


def load_grib_members(path: Path, tag: str = "", bbox=None) -> dict:
    """Like load_grib, but splits messages by ensemble member:
    {"c00": Fields, "p01": Fields, ...}. Control = perturbationNumber 0.
    With bbox, every field is cropped as it's read (global 51-member files
    would otherwise need ~3 GB of memory) and stored as float32."""
    import eccodes as ec
    out: dict = {}
    coords = {}
    sel = None
    with open(path, "rb") as fh:
        while True:
            h = ec.codes_grib_new_from_file(fh)
            if h is None:
                break
            try:
                try:
                    num = int(ec.codes_get(h, "perturbationNumber"))
                except Exception:  # noqa: BLE001
                    num = 0
                mem = "c00" if num == 0 else f"p{num:02d}"
                name = ec.codes_get(h, "shortName")
                if name in ("unknown", "~", ""):
                    # unambiguous WMO identity: discipline / category / number (see WMO_NAMES)
                    name = f"d{ec.codes_get(h, 'discipline')}c{ec.codes_get(h, 'parameterCategory')}n{ec.codes_get(h, 'parameterNumber')}"
                name = WMO_NAMES.get(name, name)
                tol = ec.codes_get(h, "typeOfLevel"); lev = ec.codes_get(h, "level")
                if tol == "isobaricInhPa":
                    key = f"{name}{int(lev)}"
                elif tol in ("heightAboveGround", "heightAboveGroundLayer"):
                    key = height_key(name, lev)
                else:
                    key = name
                if ec.codes_get(h, "stepType") == "accum":
                    start = int(ec.codes_get(h, "startStep")); endstep = int(ec.codes_get(h, "endStep"))
                    key += "_acc" if start == 0 else f"_{endstep - start}"
                key += tag
                ni, nj = ec.codes_get(h, "Ni"), ec.codes_get(h, "Nj")
                vals = ec.codes_get_values(h).reshape(nj, ni)
                if not coords:
                    lats = ec.codes_get_array(h, "latitudes").reshape(nj, ni); lons = ec.codes_get_array(h, "longitudes").reshape(nj, ni)
                    lat0, lon0 = lats[:, 0].copy(), lons[0, :].copy()
                    lon180 = np.where(lon0 > 180, lon0 - 360, lon0)
                    if bbox is not None:
                        blon0, blon1, blat0, blat1 = bbox
                        li = np.where((lon180 >= blon0) & (lon180 <= blon1))[0]
                        la = np.where((lat0 >= blat0) & (lat0 <= blat1))[0]
                        sel = (la, li)
                        coords = {"lat": lat0[la], "lon": lon180[li]}
                    else:
                        coords = {"lat": lat0, "lon": lon180}
                if sel is not None:
                    vals = vals[np.ix_(*sel)]
                f = out.setdefault(mem, Fields())
                if key not in f:
                    f[key] = np.asarray(vals, dtype=np.float32)
            finally:
                ec.codes_release(h)
    if not out:
        raise RuntimeError(f"No data in {path}")
    lon = coords["lon"]; order = np.argsort(lon); lon = lon[order]
    lat = coords["lat"]; flip = lat[0] < lat[-1]
    for f in out.values():
        for k in list(f):
            f[k] = f[k][:, order]
            if flip:
                f[k] = f[k][::-1, :]
        f.lon, f.lat = lon, (lat[::-1] if flip else lat)
    return out


# ------------------------------------------------------------- GEFS ---------
GEFS_FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p50a.pl"
GEFS_DIR = "/gefs.{ymd}/{hh}/atmos/pgrb2ap5"
GEFS_FILE = "ge{mem}.t{hh}z.pgrb2a.0p50.f{fhr:03d}"
GEFS_IDX = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod/gefs.{ymd}/{hh}/atmos/pgrb2ap5/ge{mem}.t{hh}z.pgrb2a.0p50.f{fhr:03d}.idx"


def gefs_member_url(run: dt.datetime, fhr: int, member: str, pairs, bbox) -> str:
    lon0, lon1, lat0, lat1 = bbox
    d = GEFS_DIR.format(ymd=run.strftime("%Y%m%d"), hh=run.strftime("%H"))
    if MODEL["source"] == "aigefs":
        d = d.replace("/gefs.", "/aigefs.")
    q = {"dir": d,
         "file": GEFS_FILE.format(mem=member, hh=run.strftime("%H"), fhr=fhr),
         "subregion": "", "leftlon": f"{lon0 % 360:g}", "rightlon": f"{lon1 % 360:g}",
         "toplat": f"{lat1:g}", "bottomlat": f"{lat0:g}"}
    for var, lev in pairs:
        q[f"var_{var}"] = "on"; q[f"lev_{lev}"] = "on"
    return GEFS_FILTER + "?" + urlencode(q, safe="\\()")


# ------------------------------------------------------------- AI-GEFS ------

_AIGEFS_TYPES: list | None = None      # file types (pres, sfc, ...) found in a member folder
_AIGEFS_FIELD_FILE: dict = {}          # (VAR, level) -> file type that carries it


def aigefs_url(run: dt.datetime, fhr: int, member: str, ftype: str = "pres") -> str:
    n = 0 if member == "c00" else int(member[1:])
    return MODEL["path"].format(ymd=run.strftime("%Y%m%d"), hh=run.strftime("%H"), mem=n, fhr=fhr).replace(".pres.", f".{ftype}.")


def aigefs_file_types(run: dt.datetime, session) -> list:
    """Distinct file types in a member's grib2 folder, e.g. ['pres', 'sfc']."""
    global _AIGEFS_TYPES
    if _AIGEFS_TYPES is not None:
        return _AIGEFS_TYPES
    folder = aigefs_url(run, 0, "c00").rsplit("/", 1)[0] + "/"
    names = [f for f in _listing(session, folder, retries=2, timeout=30) if f.endswith(".grib2")]
    types = sorted({f.split(".")[2] for f in names if f.count(".") >= 4})
    log.info("AI-GEFS file types in %s: %s (%d files)", folder, types or "listing unavailable", len(names))
    _AIGEFS_TYPES = types or ["pres"]
    return _AIGEFS_TYPES


def aigefs_locate_fields(run: dt.datetime, fhr: int, member: str, wanted, session) -> dict:
    """{file type: [wanted fields found in that type's index]}; logs anything not found anywhere."""
    if _AIGEFS_FIELD_FILE:
        out = {}
        for w in wanted:
            t = _AIGEFS_FIELD_FILE.get(w)
            if t:
                out.setdefault(t, []).append(w)
        return out
    remaining = list(wanted); out = {}
    for t in aigefs_file_types(run, session):
        if not remaining:
            break
        try:
            r = session.get(aigefs_url(run, fhr, member, t) + ".idx", timeout=60)
            if r.status_code != 200:
                continue
        except requests.RequestException:
            continue
        present = set()
        for line in r.text.splitlines():
            parts = line.split(":")
            if len(parts) > 4:
                present.add((parts[3], parts[4]))
        found = [w for w in remaining if w in present]
        for w in found:
            _AIGEFS_FIELD_FILE[w] = t
        if found:
            out[t] = found
        remaining = [w for w in remaining if w not in found]
    if remaining:
        log.warning("AI-GEFS: fields not found in any file type %s: %s", aigefs_file_types(run, session), remaining)
    return out


def nomads_idx_ranges(idx_url: str, wanted, session, retries: int = 3):
    """Byte ranges for wanted (VAR, level-text) fields from a NOMADS .idx
    (lines like  12:3456789:d=2026090718:TMP:850 mb:24 hour fcst:). The last
    field's length is unknown, so it's fetched to end-of-file."""
    text = None
    for attempt in range(retries):
        try:
            r = session.get(idx_url, timeout=60)
            if r.status_code == 200:
                text = r.text; break
            if r.status_code == 404:
                raise RuntimeError(f"404 {idx_url}")
        except requests.RequestException as e:
            log.info("idx fetch failed (%d): %s", attempt + 1, str(e)[:80])
        time.sleep(BACKOFF[min(attempt, len(BACKOFF) - 1)])
    if text is None:
        raise RuntimeError(f"idx unavailable: {idx_url}")
    rows = []
    for line in text.splitlines():
        parts = line.split(":")
        if len(parts) > 5:
            rows.append((int(parts[1]), parts[3], parts[4]))
    want = set(wanted); ranges = []
    for i, (off, var, lev) in enumerate(rows):
        if (var, lev) in want:
            end = rows[i + 1][0] if i + 1 < len(rows) else None
            ranges.append((off, (end - off) if end else None))
    return ranges


def download_aigefs_member(run: dt.datetime, fhr: int, member: str, dest: Path, session) -> Path:
    """Pull the wanted fields for one member/hour from whichever file types hold them."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    wanted = [(v, l) for v, l in MODEL["idx_fields"] if not (v == "APCP" and fhr == 0)]
    by_type = aigefs_locate_fields(run, fhr, member, wanted, session)
    if not by_type:
        raise RuntimeError(f"no wanted fields found for {member} f{fhr:03d}")
    tmp = dest.with_suffix(".part")
    with open(tmp, "wb") as out:
        for ftype, flds in by_type.items():
            url = aigefs_url(run, fhr, member, ftype)
            ranges = nomads_idx_ranges(url + ".idx", flds, session)
            for off, ln in sorted(ranges):
                hdr = {"Range": f"bytes={off}-{off + ln - 1}" if ln else f"bytes={off}-"}
                for attempt in range(4):
                    if host_dead(url):
                        raise RuntimeError(f"host unreachable: {url}")
                    try:
                        r = session.get(url, headers=hdr, timeout=120)
                        note_conn_ok(url)
                        if r.status_code in (200, 206):
                            out.write(r.content); break
                        if r.status_code == 404:
                            raise RuntimeError(f"404 {url}")
                    except requests.RequestException as e:
                        note_conn_failure(url)
                        log.info("range fetch failed (%d): %s", attempt + 1, str(e)[:80])
                    time.sleep(BACKOFF[min(attempt, len(BACKOFF) - 1)])
                else:
                    tmp.unlink(missing_ok=True); raise RuntimeError(f"range download failed: {url}")
    tmp.rename(dest)
    return dest


# ------------------------------------------------------------- CMC GDPS -----
# MSC Datamart (2025+ layout): one GRIB2 per field per step, global 0.15° lat-lon.
#   https://dd.weather.gc.ca/{ymd}/WXO-DD/model_gdps/15km/{hh}/{fhr}/
#   {ymd}T{hh}Z_MSC_GDPS_{Variable}_{LevelType}-{Level}_LatLon0.15_PT{fhr}H.grib2
# Variable names are descriptive (AirTemp, AbsoluteVorticity, ...). We resolve
# each generic field against the directory listing so renames don't break us.
CMC_DIR = "https://dd.weather.gc.ca/{ymd}/WXO-DD/model_gdps/15km/{hh}/{fhr:03d}/"
CMC_FILE = "{ymd}T{hh}Z_MSC_GDPS_{token}_LatLon0.15_PT{fhr:03d}H.grib2"

# generic field -> regex candidates for the "{Variable}_{LevelType}-{Level}" token.
# {lev} is the 4-digit isobaric level.
CMC_PATTERNS = {
    "gh":   [r"GeopotentialHeight_IsbL-{lev}", r"Geopotential.*_IsbL-{lev}", r"HGT_IsbL-{lev}"],
    "t":    [r"AirTemp_IsbL-{lev}", r"TMP_IsbL-{lev}"],
    "u":    [r"WindU_IsbL-{lev}", r"UGRD_IsbL-{lev}", r"WindComponentU_IsbL-{lev}", r"UWind_IsbL-{lev}"],
    "v":    [r"WindV_IsbL-{lev}", r"VGRD_IsbL-{lev}", r"WindComponentV_IsbL-{lev}", r"VWind_IsbL-{lev}"],
    "r":    [r"RelativeHumidity_IsbL-{lev}", r"RelHum_IsbL-{lev}", r"RH_IsbL-{lev}"],
    "vo":   [r"AbsoluteVorticity_IsbL-{lev}", r"ABSV_IsbL-{lev}"],
    "msl":  [r"Pressure_MSL", r"PressureMSL_MSL(-0)?", r"Pressure.*MSL.*"],
    "tp":   [r"Precip-Accum_Sfc", r"PrecipAccum_Sfc(-0)?", r"Precip.*Accum_Sfc(-0)?"],
    "2t":   [r"AirTemp_AGL-2m"],
    "10u":  [r"WindU_AGL-10m"],
    "10v":  [r"WindV_AGL-10m"],
    "cape": [r"CAPE_Sfc(-0)?"],
    "snod": [r"SnowDepth_Sfc(-0)?", r"Snow-?Depth_Sfc(-0)?"],
    "skt":  [r"RadiativeTemp_Sfc(-0)?", r"SurfaceTemp_Sfc(-0)?", r"SkinTemp_Sfc(-0)?", r"AirTemp_Sfc(-0)?"],
    "lsm":  [r"LandWaterProportion_Sfc(-0)?", r"LandCover_Sfc(-0)?", r"LandMask_Sfc(-0)?", r"Land.*_Sfc(-0)?"],
}
_CMC_TOKENS: dict | None = None


def _listing(session, url, retries: int = 4, timeout: int = 45):
    """href targets from an Apache-style directory index. The Datamart gets
    slow when many jobs hit it at once, so retry with backoff."""
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 200:
                return [h for h in re.findall(r'href="([^"?][^"]*)"', r.text) if not h.startswith("/")]
            if r.status_code == 404:
                return []
            log.info("listing %s -> HTTP %s", url, r.status_code)
        except requests.RequestException as e:
            log.info("listing %s failed (attempt %d): %s", url, attempt + 1, str(e)[:80])
        time.sleep(BACKOFF[min(attempt, len(BACKOFF) - 1)])
    return []


# Confirmed against the live Datamart (Sep 2026); used when the listing is unreachable.
CMC_DEFAULT_TOKENS = {
    "gh": "GeopotentialHeight_IsbL-{lev:04d}", "t": "AirTemp_IsbL-{lev:04d}", "u": "WindU_IsbL-{lev:04d}",
    "v": "WindV_IsbL-{lev:04d}", "r": "RelativeHumidity_IsbL-{lev:04d}", "vo": "AbsoluteVorticity_IsbL-{lev:04d}",
    "msl": "Pressure_MSL", "tp": "Precip-Accum_Sfc", "2t": "AirTemp_AGL-2m", "10u": "WindU_AGL-10m",
    "10v": "WindV_AGL-10m", "cape": "CAPE_Sfc", "snod": "SnowDepth_Sfc", "skt": "RadiativeTemp_Sfc",
    "lsm": "LandWaterProportion_Sfc",
}


def cmc_tokens(run: dt.datetime, session: requests.Session | None = None) -> dict:
    """Resolve generic fields to the Datamart's variable_level tokens by reading
    the listing for step 0 (and step 6 for accumulated precip, absent at 0)."""
    global _CMC_TOKENS
    if _CMC_TOKENS is not None:
        return _CMC_TOKENS
    session = session or requests.Session()
    ymd, hh = run.strftime("%Y%m%d"), run.strftime("%H")
    # One quick attempt at the listing (to catch renames); the Datamart is often
    # too busy to answer when 20 jobs start together, and we know the names anyway.
    names = set()
    for f in _listing(session, CMC_DIR.format(ymd=ymd, hh=hh, fhr=6), retries=1, timeout=20):
        m = re.match(r".*?_MSC_GDPS_(.+)_LatLon0\.15_PT\d{3}H\.grib2$", f)
        if m:
            names.add(m.group(1))
    tokens: dict = {}
    if not names:
        log.info("CMC: listing not available quickly; using known field names")
        _CMC_TOKENS = dict(CMC_DEFAULT_TOKENS)
        return _CMC_TOKENS
    for field, pats in CMC_PATTERNS.items():
        # isobaric fields: find the family once using level 0500, then template the level
        for pat in pats:
            probe = pat.format(lev="0500") if "{lev}" in pat else pat
            hit = next((n for n in sorted(names) if re.fullmatch(probe, n)), None)
            if hit:
                tokens[field] = hit.replace("0500", "{lev:04d}") if "{lev}" in pat else hit
                break
        if field not in tokens:
            if field in CMC_DEFAULT_TOKENS:
                tokens[field] = CMC_DEFAULT_TOKENS[field]
                log.warning("CMC: no listing match for '%s'; using default %s", field, tokens[field])
            else:
                log.warning("CMC: no match for '%s' (tried %s)", field, pats[0])
    log.info("CMC resolved %d/%d fields: %s", len(tokens), len(CMC_PATTERNS), tokens)
    unmatched = sorted(n for n in names if not any(n == t or re.fullmatch(t.replace("{lev:04d}", r"\d{4}"), n) for t in tokens.values()))
    log.info("CMC other variables present (%d): %s", len(unmatched), " ".join(unmatched[:80]))
    _CMC_TOKENS = tokens
    return tokens


def cmc_urls(run: dt.datetime, step: int, pairs: set, session: requests.Session | None = None) -> list[str]:
    tokens = cmc_tokens(run, session)
    ymd, hh = run.strftime("%Y%m%d"), run.strftime("%H")
    urls = []
    for name, lev in pairs:
        if name == "tp" and step == 0:
            continue
        tok = tokens.get(name)
        if not tok:
            continue
        token = tok.format(lev=int(lev)) if lev is not None else tok
        st = 0 if name == "lsm" else step          # land mask is a static field published at hour 0 only
        urls.append(CMC_DIR.format(ymd=ymd, hh=hh, fhr=st) + CMC_FILE.format(ymd=ymd, hh=hh, token=token, fhr=st))
    return urls


def cmc_step_complete(run: dt.datetime, step: int, session, min_files: int = 40) -> bool:
    """The Datamart creates step folders before all files arrive, so 'folder
    exists' isn't enough: require a populated listing including MSLP."""
    files = [f for f in _listing(session, CMC_DIR.format(ymd=run.strftime("%Y%m%d"), hh=run.strftime("%H"), fhr=step))
             if f.endswith(".grib2")]
    ok = len(files) >= min_files and any("MSL" in f for f in files)
    if not ok:
        log.info("CMC step %03d: %d files present, not complete", step, len(files))
    return ok


# ------------------------------------------------------------- CMC GEPS -----
# Canadian ensemble on the Datamart (legacy layout):
#   https://dd.weather.gc.ca/{ymd}/WXO-DD/ensemble/geps/grib2/raw/{hh}/{fhr}/
#   CMC_geps-raw_{VAR}_{LVLTYPE}_{LVL}_latlon0p5x0p5_{ymd}{hh}_P{fhr}_allmbrs.grib2
# Each file holds all 21 members. The filename template is derived from the
# PRMSL file actually present, so either classic or new-style names work.
GEPS_DIR = "https://dd.weather.gc.ca/{ymd}/WXO-DD/ensemble/geps/grib2/raw/{hh}/{fhr:03d}/"
_GEPS_TMPL: dict | None = None      # {"file": template with {token}/{fhr}, "style": "classic"|"new", "tokens": {...}}

GEPS_CLASSIC = {  # generic -> classic Datamart token (GEPS zero-pads levels: ISBL_0500)
    "gh": "HGT_ISBL_{lev:04d}", "t": "TMP_ISBL_{lev:04d}", "u": "UGRD_ISBL_{lev:04d}", "v": "VGRD_ISBL_{lev:04d}",
    "msl": "PRMSL_MSL_0", "tp": "APCP_SFC_0", "2t": "TMP_TGL_2m", "10u": "UGRD_TGL_10m", "10v": "VGRD_TGL_10m",
}
# alternative spellings to try if the first 404s (near-surface levels vary between MSC products)
GEPS_ALT = {"2t": ["TMP_TGL_2m", "TMP_TGL_2"], "10u": ["UGRD_TGL_10m", "UGRD_TGL_10"], "10v": ["VGRD_TGL_10m", "VGRD_TGL_10"]}


def geps_template(run: dt.datetime, session) -> dict:
    global _GEPS_TMPL
    if _GEPS_TMPL:
        return _GEPS_TMPL
    ymd, hh = run.strftime("%Y%m%d"), run.strftime("%H")
    files = []
    for step in (6, 24):
        files = [f for f in _listing(session, GEPS_DIR.format(ymd=ymd, hh=hh, fhr=step), retries=1, timeout=20) if f.endswith(".grib2")]
        if files:
            break
    log.info("GEPS listing sample (%d files): %s", len(files), " ".join(files[:6]))
    prm = next((f for f in files if "PRMSL" in f or "PressureMSL" in f or "Pressure_MSL" in f), None)
    if prm and "_MSC_GEPS_" in prm:                       # new-style names, like the GDPS
        m = re.match(r".*?_MSC_GEPS_(.+?)_(LatLon[\d.x]+)_PT(\d{3})H\.grib2$", prm)
        names = {re.match(r".*?_MSC_GEPS_(.+?)_LatLon", f).group(1) for f in files if "_MSC_GEPS_" in f}
        tokens = {}
        for field, pats in CMC_PATTERNS.items():
            for pat in pats:
                probe = pat.format(lev="0500") if "{lev}" in pat else pat
                hit = next((n for n in sorted(names) if re.fullmatch(probe, n)), None)
                if hit:
                    tokens[field] = hit.replace("0500", "{lev:04d}") if "{lev}" in pat else hit; break
        _GEPS_TMPL = {"style": "new", "tokens": tokens,
                      "file": f"{ymd}T{hh}Z_MSC_GEPS_{{token}}_{m.group(2)}_PT{{fhr:03d}}H.grib2"}
    else:                                                 # classic CMC_geps-raw_* names
        grid = "latlon0p5x0p5"
        if prm:
            m = re.search(r"_(latlon[\dp x]+?)_\d{10}_P\d{3}", prm)
            if m:
                grid = m.group(1)
        _GEPS_TMPL = {"style": "classic", "tokens": dict(GEPS_CLASSIC),
                      "file": f"CMC_geps-raw_{{token}}_{grid}_{ymd}{hh}_P{{fhr:03d}}_allmbrs.grib2"}
    log.info("GEPS template: %s (%s)", _GEPS_TMPL["file"], _GEPS_TMPL["style"])
    return _GEPS_TMPL


def download_geps(run: dt.datetime, step: int, fields, dest: Path, session: requests.Session | None = None) -> Path:
    """Fetch each field's all-member file for one step and concatenate."""
    session = session or requests.Session()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    tm = geps_template(run, session)
    ymd, hh = run.strftime("%Y%m%d"), run.strftime("%H")
    tmp = dest.with_suffix(".part"); got = 0
    with open(tmp, "wb") as out:
        for name, lev in fields:
            if name == "tp" and step == 0:
                continue
            t = tm["tokens"].get(name)
            if not t:
                log.warning("GEPS: no token for %s", name); continue
            candidates = GEPS_ALT.get(name, [t]) if tm["style"] == "classic" else [t]
            done = False
            for cand in candidates:
                token = cand.format(lev=int(lev)) if lev is not None else cand
                url = GEPS_DIR.format(ymd=ymd, hh=hh, fhr=step) + tm["file"].format(token=token, fhr=step)
                for attempt in range(3):
                    if host_dead(url):
                        break
                    try:
                        r = session.get(url, timeout=90)
                        note_conn_ok(url)
                        if r.status_code == 200 and len(r.content) > 500:
                            out.write(r.content); got += 1; done = True; break
                        if r.status_code == 404:
                            break
                    except requests.RequestException as e:
                        note_conn_failure(url)
                        log.info("GET failed (%d): %s", attempt + 1, str(e)[:80])
                    time.sleep(BACKOFF[min(attempt, len(BACKOFF) - 1)])
                if host_dead(url):
                    break
                if done:
                    break
            if not done:
                log.warning("missing: %s (tried %s)", name, " ".join(candidates))
    if got == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"No GEPS fields downloaded for step {step}")
    tmp.rename(dest)
    return dest


def geps_step_complete(run: dt.datetime, step: int, session, min_files: int = 10) -> bool:
    files = [f for f in _listing(session, GEPS_DIR.format(ymd=run.strftime("%Y%m%d"), hh=run.strftime("%H"), fhr=step)) if f.endswith(".grib2")]
    ok = len(files) >= min_files and any("PRMSL" in f or "MSL" in f for f in files)
    if not ok:
        log.info("GEPS step %03d: %d files present, not complete", step, len(files))
    return ok


# ------------------------------------------------------------- DWD ICON -----
# One bz2-compressed GRIB2 per field per step, on ICON's native triangular
# grid. Regridded to 0.125° lat-lon with cdo using DWD's own weights file.
ICON_BASE = "https://opendata.dwd.de/weather/nwp/icon/grib/{hh}/{vdir}/"
ICON_SL = "icon_global_icosahedral_single-level_{ymd}{hh}_{fhr:03d}_{VAR}.grib2.bz2"
ICON_PL = "icon_global_icosahedral_pressure-level_{ymd}{hh}_{fhr:03d}_{lev}_{VAR}.grib2.bz2"
ICON_TI = "icon_global_icosahedral_time-invariant_{ymd}{hh}_{VAR}.grib2.bz2"
ICON_NAMES = {  # generic -> (dir, VAR, kind)
    "gh": ("fi", "FI", "pl"), "t": ("t", "T", "pl"), "u": ("u", "U", "pl"), "v": ("v", "V", "pl"), "r": ("relhum", "RELHUM", "pl"),
    "msl": ("pmsl", "PMSL", "sl"), "tp": ("tot_prec", "TOT_PREC", "sl"), "2t": ("t_2m", "T_2M", "sl"),
    "10u": ("u_10m", "U_10M", "sl"), "10v": ("v_10m", "V_10M", "sl"), "tcwv": ("tqv", "TQV", "sl"),
    "cape": ("cape_ml", "CAPE_ML", "sl"), "snod": ("h_snow", "H_SNOW", "sl"), "skt": ("t_g", "T_G", "sl"),
    "lsm": ("fr_land", "FR_LAND", "ti"),
}
ICON_WEIGHTS_URL = "https://opendata.dwd.de/weather/lib/cdo/ICON_GLOBAL2WORLD_0125_EASY.tar.bz2"
ICON_WEIGHTS_DIR = Path(os.environ.get("ICON_WEIGHTS_DIR", str(Path.home() / ".cache" / "icon_weights")))


def icon_urls(run: dt.datetime, step: int, pairs: set) -> list[str]:
    ymd, hh = run.strftime("%Y%m%d"), run.strftime("%H")
    urls = []
    for name, lev in pairs:
        if name not in ICON_NAMES or (name == "tp" and step == 0):
            continue
        vdir, VAR, kind = ICON_NAMES[name]
        base = ICON_BASE.format(hh=hh, vdir=vdir)
        if kind == "pl":
            urls.append(base + ICON_PL.format(ymd=ymd, hh=hh, fhr=step, lev=lev, VAR=VAR))
        elif kind == "sl":
            urls.append(base + ICON_SL.format(ymd=ymd, hh=hh, fhr=step, VAR=VAR))
        else:
            urls.append(base + ICON_TI.format(ymd=ymd, hh=hh, VAR=VAR))
    return urls


def icon_weights() -> tuple[Path, Path]:
    """DWD's cdo grid description + remap weights (cached; ~60 MB download)."""
    import subprocess, tarfile
    ICON_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    grid = next(ICON_WEIGHTS_DIR.rglob("target_grid_world_0125.txt"), None)
    wts = next(ICON_WEIGHTS_DIR.rglob("weights_icogl2world_0125.nc"), None)
    if grid and wts:
        return grid, wts
    tb = ICON_WEIGHTS_DIR / "weights.tar.bz2"
    log.info("downloading ICON regrid weights")
    r = requests.get(ICON_WEIGHTS_URL, timeout=600); r.raise_for_status()
    tb.write_bytes(r.content)
    with tarfile.open(tb) as t:
        t.extractall(ICON_WEIGHTS_DIR)
    tb.unlink()
    grid = next(ICON_WEIGHTS_DIR.rglob("target_grid_world_0125.txt"))
    wts = next(ICON_WEIGHTS_DIR.rglob("weights_icogl2world_0125.nc"))
    return grid, wts


def icon_remap(src: Path, dest: Path):
    import subprocess
    grid, wts = icon_weights()
    cmd = ["cdo", "-s", "-f", "grb2", f"remap,{grid},{wts}", str(src), str(dest)]
    subprocess.run(cmd, check=True, capture_output=True)


def download_files(run: dt.datetime, step: int, pairs: set, dest: Path, session: requests.Session,
                   retries: int = 4) -> Path:
    """CMC / ICON: fetch each field's file, concatenate (decompressing bz2 for
    ICON), and for ICON regrid to lat-lon. Missing individual fields are logged
    and skipped so one absent variable doesn't kill the frame."""
    import bz2
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    urls = cmc_urls(run, step, pairs, session) if MODEL["source"] == "cmc" else icon_urls(run, step, pairs)
    if not urls:
        raise RuntimeError(f"nothing to fetch for step {step}")
    raw = dest.with_suffix(".raw.grib2")
    got = 0
    with open(raw, "wb") as out:
        for url in urls:
            for attempt in range(retries):
                if host_dead(url):
                    break
                try:
                    r = session.get(url, timeout=60)
                    note_conn_ok(url)
                    if r.status_code == 200 and len(r.content) > 500:
                        data = bz2.decompress(r.content) if url.endswith(".bz2") else r.content
                        out.write(data); got += 1
                        break
                    if r.status_code == 404:
                        log.warning("missing: %s", url.rsplit("/", 1)[-1]); break
                    log.warning("GET %s -> %s", url.rsplit("/", 1)[-1], r.status_code)
                except (requests.RequestException, OSError) as e:
                    note_conn_failure(url)
                    log.warning("GET failed (%d): %s", attempt + 1, str(e)[:100])
                time.sleep(BACKOFF[min(attempt, len(BACKOFF) - 1)])
            if host_dead(url):
                break
    if got == 0:
        raw.unlink(missing_ok=True)
        raise RuntimeError(f"No fields downloaded for step {step}")
    if MODEL["source"] == "icon":
        icon_remap(raw, dest); raw.unlink()
    else:
        raw.rename(dest)
    return dest


def pack_members(path: Path, prev_path, bbox, dest: Path) -> Path:
    """Read a multi-member GRIB (plus optional previous-step file) once, cropped
    to bbox, and save a compact .npz that workers can load cheaply."""
    per = load_grib_members(path, bbox=bbox)
    if prev_path:
        for m, pf in load_grib_members(Path(prev_path), "_m6", bbox=bbox).items():
            if m in per:
                per[m].update(pf)
    members = sorted(per)
    keys = sorted(set.intersection(*(set(per[m]) for m in members)))
    arrays = {k: np.stack([per[m][k] for m in members]).astype(np.float32) for k in keys}
    any_f = per[members[0]]
    np.savez(dest, lon=any_f.lon, lat=any_f.lat, members=np.array(members), keys=np.array(keys), **arrays)
    return dest


class Fields(dict):
    """A dict of name -> 2D numpy array, plus shared lon/lat 1-D coordinates."""
    lon: np.ndarray
    lat: np.ndarray


# WMO discipline/category/number -> our names, for fields eccodes labels "unknown"
# (Environment Canada's GRIB uses templates eccodes doesn't always resolve).
WMO_NAMES = {"d0c1n8": "tp", "d0c7n6": "cape", "d0c1n11": "snod", "d2c0n0": "lsm", "d0c0n17": "skt", "d0c2n22": "gust",
             "d0c2n1": "si10", "d0c2n0": "wdir10",
             "d0c3n1": "prmsl", "d0c1n3": "pwat", "d0c0n0": "t", "d0c2n2": "u", "d0c2n3": "v", "d0c3n5": "gh",
             "d0c1n1": "r", "d0c2n10": "absv", "d0c3n0": "pres", "d0c1n7": "prate", "d0c16n196": "refc"}

# Names eccodes gives GFS/ECMWF fields at fixed heights -> the names plots.py uses
HEIGHT_NAMES = {"2t": "t2m", "10u": "u10", "10v": "v10", "2r": "rh2m", "2d": "d2m", "10si": "si10", "10wdir": "wdir10",
                "gust": "gust", "10fg": "gust", "i10fg": "gust", "si10": "si10", "wdir10": "wdir10", "wdir": "wdir10", "ws": "si10"}
# generic names at a fixed height (how unnamed/WMO-mapped fields arrive): (shortName, level) -> key
HEIGHT_BY_LEVEL = {("u", 10): "u10", ("v", 10): "v10", ("t", 2): "t2m", ("r", 2): "rh2m", ("si", 10): "si10", ("wdir", 10): "wdir10",
                   ("gust", 10): "gust", ("si10", 10): "si10", ("wdir10", 10): "wdir10"}


def height_key(name: str, lev) -> str:
    try:
        lv = int(round(float(lev)))
    except (TypeError, ValueError):
        lv = None
    if name in HEIGHT_NAMES:
        return HEIGHT_NAMES[name]
    if (name, lv) in HEIGHT_BY_LEVEL:
        return HEIGHT_BY_LEVEL[(name, lv)]
    return f"{name}{lv}m" if lv is not None and name in ("u", "v", "t", "r", "q") else name


_REGRID_CACHE: dict = {}


def _regrid_index(lats2d, lons2d, res: float):
    """Nearest-neighbour mapping from an irregular (e.g. Lambert) grid to a regular
    lat/lon grid at `res` degrees covering the data. Cached per grid shape."""
    key = (lats2d.shape, round(float(lats2d[0, 0]), 3), round(float(lons2d[0, 0]), 3), res)
    if key in _REGRID_CACHE:
        return _REGRID_CACHE[key]
    from scipy.spatial import cKDTree
    lon0, lon1 = float(np.nanmin(lons2d)), float(np.nanmax(lons2d))
    lat0, lat1 = float(np.nanmin(lats2d)), float(np.nanmax(lats2d))
    tlon = np.arange(np.floor(lon0), np.ceil(lon1) + res / 2, res)
    tlat = np.arange(np.ceil(lat1), np.floor(lat0) - res / 2, -res)          # north to south
    TLON, TLAT = np.meshgrid(tlon, tlat)
    tree = cKDTree(np.column_stack([lons2d.ravel(), lats2d.ravel()]))
    dist, idx = tree.query(np.column_stack([TLON.ravel(), TLAT.ravel()]), k=1, distance_upper_bound=res * 2.5)
    mask = ~np.isfinite(dist)
    idx = np.where(mask, 0, idx)
    _REGRID_CACHE[key] = (tlon, tlat, idx, mask, TLON.shape)
    return _REGRID_CACHE[key]


def load_grib(path: Path, tag: str = "", bbox=None) -> Fields:
    """Read every message in a GRIB file into Fields keyed so that plots can
    tell fields apart unambiguously:

        t850, u250, gh500, r700     isobaric: shortName + level (hPa)
        t2m, u10, v10               fixed heights (renamed via HEIGHT_NAMES)
        pres_pv, u_pv, v_pv         2-PVU surface
        tp_acc                      accumulation from t=0
        tp_6                        6-hour bucket (GFS)
        refc, csnow, cape, ...      anything else: shortName
        unknown ids                 p<paramId>

    `tag` is appended to every key (e.g. "_m24" for fields fetched from
    forecast hour fhr-24) so previous-step fields can live alongside."""
    import eccodes as ec
    out = Fields()
    lat = lon = None
    regular = True
    with open(path, "rb") as fh:
        while True:
            h = ec.codes_grib_new_from_file(fh)
            if h is None:
                break
            try:
                name = ec.codes_get(h, "shortName")
                if name in ("unknown", "~", ""):
                    # unambiguous WMO identity: discipline / category / number (see WMO_NAMES)
                    name = f"d{ec.codes_get(h, 'discipline')}c{ec.codes_get(h, 'parameterCategory')}n{ec.codes_get(h, 'parameterNumber')}"
                name = WMO_NAMES.get(name, name)
                tol = ec.codes_get(h, "typeOfLevel")
                lev = ec.codes_get(h, "level")
                step_type = ec.codes_get(h, "stepType")
                if tol == "isobaricInhPa":
                    key = f"{name}{int(lev)}"
                elif tol == "potentialVorticity":
                    key = f"{name}_pv"
                elif tol in ("heightAboveGround", "heightAboveGroundLayer"):
                    key = height_key(name, lev)
                elif tol == "surface" and name in ("t", "u", "v", "q", "r"):
                    key = f"{name}_sfc"
                else:
                    key = name
                if step_type == "accum":
                    start = int(ec.codes_get(h, "startStep")); endstep = int(ec.codes_get(h, "endStep"))
                    key += "_acc" if start == 0 else f"_{endstep - start}"
                key += tag
                ni, nj = ec.codes_get(h, "Ni"), ec.codes_get(h, "Nj")
                vals = ec.codes_get_values(h).reshape(nj, ni)
                missing = ec.codes_get(h, "missingValue")
                vals = np.where(vals == missing, np.nan, vals)
                if lat is None:
                    lats = ec.codes_get_array(h, "latitudes").reshape(nj, ni)
                    lons = ec.codes_get_array(h, "longitudes").reshape(nj, ni)
                    lons = np.where(lons > 180, lons - 360, lons)
                    regular = ec.codes_get(h, "gridType") == "regular_ll"
                    if regular:
                        lat, lon = lats[:, 0].copy(), lons[0, :].copy()
                    else:                                   # Lambert etc.: regrid to lat/lon
                        tlon, tlat, ridx, rmask, rshape = _regrid_index(lats, lons, MODEL.get("grid_res", 0.05))
                        lat, lon = tlat, tlon
                        if bbox is not None:                # only keep the window this frame needs
                            b0, b1, c0, c1 = bbox
                            li = np.where((tlon >= b0) & (tlon <= b1))[0]; la = np.where((tlat >= c0) & (tlat <= c1))[0]
                            if len(li) > 4 and len(la) > 4:
                                sub = np.ix_(la, li)
                                ridx = ridx.reshape(rshape)[sub].ravel(); rmask = rmask.reshape(rshape)[sub].ravel()
                                rshape = (len(la), len(li)); lat, lon = tlat[la], tlon[li]
                if not regular:
                    v = vals.ravel()[ridx].astype(np.float32); v[rmask] = np.nan; vals = v.reshape(rshape)
                if key not in out:                 # first occurrence wins (e.g. duplicate tp records)
                    out[key] = np.asarray(vals, dtype=np.float32 if not regular else float)
            finally:
                ec.codes_release(h)
    if lat is None:
        raise RuntimeError(f"No data in {path}")
    order = np.argsort(lon)
    lon = lon[order]
    for k in list(out):
        out[k] = out[k][:, order]
    if lat[0] < lat[-1]:                           # plots assume north-to-south rows
        lat = lat[::-1]
        for k in list(out):
            out[k] = out[k][::-1, :]
    out.lon, out.lat = lon, lat
    return out


def merge(a: Fields, b: Fields) -> Fields:
    """Merge previous-step fields (already tagged) into the main Fields."""
    for k, v in b.items():
        if v.shape == next(iter(a.values())).shape:
            a[k] = v
    return a


def normalise(f: "Fields", fhr: int = 0) -> "Fields":
    """Map model-specific names/units onto what plots.py expects:
    prmsl [Pa], tp_6 [mm/6 h], tp_acc [mm since t0], absv500 [s^-1], pwat [mm],
    t2m, u10, v10, t850 ... GFS is the reference convention."""
    src = MODEL["source"]
    accum_from_zero = src in ("ecmwf_opendata", "cmc", "icon", "geps", "ecmwf_ens", "ecmwf_aifs_ens")
    # ---- name aliases (any tag suffix)
    alias = {"msl": "prmsl", "mslma": "prmsl", "mslet": "prmsl", "tcwv": "pwat", "tciwv": "pwat", "tcw": "pwat", "sde": "snod", "z": "gh",
             "gust": "gust", "i10fg": "gust", "10fg": "gust", "si10": "si10", "10si": "si10", "wdir10": "wdir10", "10wdir": "wdir10",
             # DWD local names that eccodes passes through verbatim
             "TQV": "pwat", "T_G": "t_sfc", "CAPE_ML": "cape", "H_SNOW": "snod", "FR_LAND": "lsm", "PMSL": "prmsl",
             "TOT_PREC": "tp", "T_2M": "t2m", "U_10M": "u10", "V_10M": "v10", "RELHUM": "r", "FI": "z"}
    for _pass in range(2):                                        # two passes so FI -> z -> gh resolves
        for k in list(f):
            base, tag = (k.split("_m", 1)[0], "_m" + k.split("_m", 1)[1]) if "_m" in k and k.split("_m", 1)[1].isdigit() else \
                        ((k[:-3], "_f0") if k.endswith("_f0") else (k, ""))
            for old, new in alias.items():
                if base == old or (base.startswith(old) and base[len(old):].isdigit()):
                    nk = new + base[len(old):] + tag
                    if nk not in f:
                        f[nk] = f.pop(k)
                        if old == "z":                               # ICON geopotential m²/s² -> gpm
                            f[nk] = f[nk] / 9.80665
                    break
    if "vo500" in f and "absv500" not in f:                      # relative -> absolute vorticity
        _, LAT = np.meshgrid(f.lon, f.lat)
        f["absv500"] = f["vo500"] + 2 * 7.2921e-5 * np.sin(np.radians(LAT))
    if "absv500" not in f and "u500" in f and "v500" in f:      # sources without vorticity: compute it
        from plots import rel_vort
        _, LAT = np.meshgrid(f.lon, f.lat)
        f["absv500"] = rel_vort(f["u500"], f["v500"], f.lon, f.lat) + 2 * 7.2921e-5 * np.sin(np.radians(LAT))
    if accum_from_zero:
        # Units: ECMWF IFS (deterministic and ENS) publish tp in metres; CMC, ICON and AIFS
        # in mm. Detect rather than assume: a run-total in mm exceeds 3 somewhere in any
        # domain this size, a total in metres never does.
        for k in [k for k in f if k.startswith("tp_acc")]:
            mx = np.nanmax(f[k]) if np.isfinite(f[k]).any() else 0.0
            if 0 < mx < 3.0:
                f[k] = f[k] * 1000.0
        if "tp_acc" in f:
            prev = f.get("tp_acc_m6", np.zeros_like(f["tp_acc"]))
            f["tp_6"] = np.clip(f["tp_acc"] - prev, 0, None)
    # hourly/3-hourly models: 6-h total from the run accumulation and the one 6 h earlier
    if "tp_6" not in f and "tp_acc" in f and "tp_acc_m6" in f:
        f["tp_6"] = np.clip(f["tp_acc"] - f["tp_acc_m6"], 0, None)
    # NBM gives 10 m wind as speed + direction: rebuild components for the barbs
    if "si10" in f and "wdir10" in f and "u10" not in f:
        d = np.radians(f["wdir10"]); f["u10"] = -f["si10"] * np.sin(d); f["v10"] = -f["si10"] * np.cos(d)
    # GFS: at f006 the only bucket is 0-6, keyed tp_acc. Same for tagged previous steps.
    for tag in ("", "_m6", "_m12", "_m18"):
        if f"tp_6{tag}" not in f and f"tp_acc{tag}" in f:
            f[f"tp_6{tag}"] = f[f"tp_acc{tag}"]
    for k in [k for k in f if k.startswith("tp_acc_m")]:         # 24-h totals
        pass
    if "tp_acc" in f and "tp_acc_m24" in f:
        f["tp_24"] = np.clip(f["tp_acc"] - f["tp_acc_m24"], 0, None)
    elif "tp_acc" in f and fhr <= 24:
        f["tp_24"] = f["tp_acc"]
    return f


def crop(f: "Fields", bbox) -> "Fields":
    """Cut a global grid down to a bbox (lon0, lon1, lat0, lat1)."""
    lon0, lon1, lat0, lat1 = bbox
    li = np.where((f.lon >= lon0) & (f.lon <= lon1))[0]
    la = np.where((f.lat >= lat0) & (f.lat <= lat1))[0]
    if len(li) < 4 or len(la) < 4:
        return f
    out = Fields()
    out.lon, out.lat = f.lon[li], f.lat[la]
    for k, v in f.items():
        out[k] = v[np.ix_(la, li)]
    return out


def synthetic_fields(fhr: int, bbox, n=(120, 200), tags=("", "_m6", "_m12", "_m18", "_m24", "_f0")) -> Fields:
    """Fake but plausible-looking fields (with the same key scheme as
    load_grib) for testing the plots without network access."""
    lon0, lon1, lat0, lat1 = bbox
    lat = np.linspace(lat1, lat0, n[0])
    lon = np.linspace(lon0, lon1, n[1])
    LON, LAT = np.meshgrid(lon, lat)
    rng = np.random.default_rng(fhr)
    out = Fields()
    out.lon, out.lat = lon, lat
    for tag in tags:
        t = (fhr - {"": 0, "_m6": 6, "_m12": 12, "_m18": 18, "_m24": 24, "_f0": fhr}[tag]) / 24.0
        wave = np.sin(np.radians(LON * 3 + t * 40)) * np.cos(np.radians((LAT - 35) * 4))
        cold = np.clip((LAT - 30) / 25, 0, 1)
        f = {
            "gh500": 5700 - 12 * (LAT - 25) + 120 * wave, "gh700": 3000 - 7 * (LAT - 25) + 70 * wave,
            "gh850": 1500 - 4 * (LAT - 25) + 40 * wave, "gh1000": 100 + 20 * wave, "gh250": 10600 - 22 * (LAT - 25) + 200 * wave, "gh200": 12000 - 24 * (LAT - 25) + 220 * wave,
            "absv500": 2e-5 + 1.5e-4 * np.clip(wave, 0, 1) ** 2 * np.sin(np.radians(LON * 6)) ** 2,
            "u500": 25 * wave + 15, "v500": 12 * np.cos(np.radians(LON * 3 + t * 40)),
            "u700": 15 * wave + 8, "v700": 9 * np.cos(np.radians(LON * 3 + t * 40)),
            "u850": 10 * wave + 5, "v850": 8 * np.cos(np.radians(LON * 3 + t * 40)),
            "u250": 45 * wave + 25 + 20 * np.exp(-((LAT - 40) / 6) ** 2), "v250": 20 * np.cos(np.radians(LON * 3 + t * 40)),
            "u200": 50 * wave + 28 + 22 * np.exp(-((LAT - 40) / 6) ** 2), "v200": 22 * np.cos(np.radians(LON * 3 + t * 40)),
            "u300": 35 * wave + 20 + 15 * np.exp(-((LAT - 40) / 6) ** 2), "v300": 16 * np.cos(np.radians(LON * 3 + t * 40)),
            "prmsl": 101300 - 1200 * wave + 200 * np.cos(np.radians(LAT * 5)),
            "tp_6": 15 * np.clip(-wave, 0, 1) ** 3 * (rng.random(LON.shape) * 0.5 + 0.5),
            "t850": 293 - 0.5 * (LAT - 10) + 5 * wave, "t700": 283 - 0.5 * (LAT - 10) + 5 * wave,
            "t2m": 303 - 0.7 * (LAT - 10) + 4 * wave, "u10": 6 * wave + 3, "v10": 5 * np.cos(np.radians(LON * 3 + t * 40)),
            "pwat": 45 - 0.8 * (LAT - 10) + 12 * -wave, "cape": 3000 * np.clip(-wave, 0, 1) ** 2 * np.clip((40 - LAT) / 30, 0, 1),
            "r700": np.clip(60 - 40 * wave, 0, 100), "r500": np.clip(50 - 40 * wave, 0, 100), "r300": np.clip(40 - 40 * wave, 0, 100),
            "refc": np.clip(55 * np.clip(-wave, 0, 1) ** 1.5 * (rng.random(LON.shape) * 0.6 + 0.4) - 5, -10, 70),
            "csnow": (cold * np.clip(-wave, 0, 1) > 0.45).astype(float), "cicep": np.zeros_like(LAT),
            "cfrzr": ((cold * np.clip(-wave, 0, 1) > 0.38) & (cold * np.clip(-wave, 0, 1) <= 0.45)).astype(float),
            "pres_pv": 25000 + 20000 * cold + 15000 * wave, "u_pv": 40 * wave + 30, "v_pv": 20 * np.cos(np.radians(LON * 3 + t * 40)),
            "sbt124": 290 - 70 * np.clip(-wave, 0, 1) ** 2 - 10 * cold, "snod": 0.05 * cold * (1 + t) * np.clip(-wave, 0, 1),
            "t_sfc": 303 - 0.35 * (LAT - 10) + 1.5 * wave, "land": (np.sin(np.radians(LON * 2)) * np.cos(np.radians(LAT * 3)) > 0.4).astype(float),
            "gust": (8 * wave + 6) * 1.4,
        }
        f["crain"] = ((f["tp_6"] > 0.2) & (f["csnow"] == 0) & (f["cfrzr"] == 0)).astype(float)
        f["tp_acc"] = f["tp_6"] * max(1, (fhr / 6) * 0.6)
        for k, v in f.items():
            out[k + tag] = v
    return out
