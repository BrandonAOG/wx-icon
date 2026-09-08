#!/usr/bin/env python3
"""
Render GFS maps for the site.

    python render/render.py                      # latest available run, all regions/params
    python render/render.py --run 2026090612     # specific run
    python render/render.py --hours 0-48/6 --regions conus natl --params z500_vort mslp_precip
    python render/render.py --synthetic          # no network: fake fields, for testing plots

CI helpers (used by the GitHub Actions workflow):
    python render/check.py                       # newest run vs. what's live; writes job outputs
    python render/render.py --manifest-only --run 2026090612   # write manifest for already-rendered images

Output:
    site/images/gfs/<run>/<region>/<param>/f<hhh>.png
    site/manifest.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402

import plots  # noqa: E402
import storage  # noqa: E402
from config import (FORECAST_HOURS, KEEP_RUNS, MANIFEST_NAME, MODEL, REGIONS, model_params, param_hours, products)  # noqa: E402
PARAMS = products()  # deterministic or ensemble product table for this model
ENSEMBLE = MODEL.get("kind") == "ensemble"
_LOGGED_FIELDS: set = set()
if ENSEMBLE:
    import ensemble  # noqa: E402
from fetch import Fields  # noqa: E402
from fetch import (all_fetch_pairs, available_pairs, build_filter_url, crop, download, download_aigefs_member, download_ecmwf, download_ecmwf_ens, download_files, download_geps, download_grouped, ecmwf_pairs, gefs_member_url, load_grib_members, pack_members,
                   latest_available_run, load_grib, merge, normalise, prev_steps, step_for,
                   synthetic_fields)  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("render")

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
PAD = 6  # degrees of extra data around each region so edge contours aren't clipped


def parse_hours(spec: str) -> list[int]:
    if "-" in spec:
        rng, _, step = spec.partition("/")
        a, b = (int(x) for x in rng.split("-"))
        return list(range(a, b + 1, int(step or 6)))
    return [int(x) for x in spec.split(",")]


def padded(bbox):
    lon0, lon1, lat0, lat1 = bbox
    return (lon0 - PAD, lon1 + PAD, max(lat0 - PAD, -89), min(lat1 + PAD, 89))


def render_frame(run_iso: str, fhr: int, region: str, param_ids: list[str],
                 grib_paths: dict | None, out_dir: str, synthetic: bool) -> list[str]:
    """Render every requested parameter for one (hour, region). Runs in a worker.
    grib_paths: {"": main file, "_m24": file for fhr-24, "_f0": file for hour 0, ...}"""
    run = dt.datetime.fromisoformat(run_iso)
    bbox = REGIONS[region]["bbox"]
    if ENSEMBLE:
        return render_ensemble_frame(run, fhr, region, param_ids, grib_paths, out_dir, synthetic)
    if synthetic:
        fields = synthetic_fields(fhr, padded(bbox))
    else:
        win = padded(bbox) if MODEL["source"] == "nomads_grid" else None
        try:
            fields = load_grib(Path(grib_paths[""]), bbox=win)
        except Exception as e:  # noqa: BLE001
            log.error("f%03d %s: data unreadable: %s", fhr, region, str(e)[:200]); return []
        for tag, path in grib_paths.items():
            if tag and path:
                try:
                    fields = merge(fields, load_grib(Path(path), tag, bbox=win))
                except Exception as e:  # noqa: BLE001
                    log.warning("f%03d %s: previous-step file %s unreadable: %s", fhr, region, tag, e)
        fields = crop(fields, padded(bbox))
    fields = normalise(fields, fhr)
    if region == list(REGIONS)[0] and not _LOGGED_FIELDS:
        _LOGGED_FIELDS.add(1)
        log.info("fields available at f%03d: %s", fhr, " ".join(sorted(fields)))
    meta = {"run": run, "fhr": fhr, "bbox": bbox, "region": region,
            "region_name": REGIONS[region]["name"]}
    written = []
    for pid in param_ids:
        if fhr not in param_hours(pid):
            continue
        fn = getattr(plots, PARAMS[pid]["plot"])
        dest = Path(out_dir) / region / pid / f"f{fhr:03d}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig = fn(fields, meta)
            fig.savefig(dest, dpi=fig.dpi, facecolor="white")
            plt.close(fig)
            compress_png(dest)
            written.append(str(dest))
        except Exception as e:  # noqa: BLE001
            plt.close("all")
            log.error("failed %s %s f%03d: %s", region, pid, fhr, str(e)[:160])
    return written


def compress_png(path: Path):
    """Palette-quantize the PNG: these maps have few distinct colours, so this
    roughly halves the file with no visible change."""
    try:
        from PIL import Image
        im = Image.open(path).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        im.save(path, optimize=True)
    except Exception as e:  # noqa: BLE001
        log.warning("compress failed for %s: %s", path.name, e)


def render_ensemble_frame(run, fhr, region, param_ids, grib_paths, out_dir, synthetic):
    """Load every member for this hour, stack, crop to the region, draw ensemble products."""
    from ensemble import Stack
    bbox = REGIONS[region]["bbox"]
    members, fields = [], []
    if synthetic:
        rng = np.random.default_rng(fhr)
        for i, m in enumerate(MODEL["members"]):
            f = synthetic_fields(fhr, padded(bbox), tags=("",))
            for k in ("prmsl", "gh500", "t850", "t2m", "u10", "v10", "tp_6"):
                f[k] = f[k] * (1 + 0.004 * rng.normal() * (1 + fhr / 48)) + (rng.normal() * (150 if k == "prmsl" else 0.4) * (1 + fhr / 48))
            members.append(m); fields.append(f)
    elif MODEL["source"] in ("ecmwf_ens", "ecmwf_aifs_ens", "geps"):
        # packed .npz prepared once per hour in the main process (see pack_members)
        try:
            z = np.load(grib_paths["npz"], allow_pickle=False)
            lon, lat = z["lon"], z["lat"]; mems = list(z["members"]); keys = list(z["keys"])
            for i, m in enumerate(mems):
                f = Fields(); f.lon, f.lat = lon, lat
                for k in keys:
                    f[k] = z[k][i]
                members.append(m); fields.append(normalise(crop(f, padded(bbox)), fhr))
        except Exception as e:  # noqa: BLE001
            log.error("f%03d: packed ENS data unreadable: %s", fhr, e); return []
    else:
        for m, path in (grib_paths or {}).items():
            try:
                f = normalise(crop(load_grib(Path(path)), padded(bbox)), fhr)
                members.append(m); fields.append(f)
            except Exception as e:  # noqa: BLE001
                log.warning("f%03d member %s unreadable: %s", fhr, m, e)
    if len(fields) < 3:
        log.error("f%03d %s: only %d members loaded, skipping", fhr, region, len(fields))
        return []
    keys = set.intersection(*(set(f) for f in fields))
    stack = Stack({k: np.stack([f[k] for f in fields]) for k in keys})
    stack.lon, stack.lat, stack.members = fields[0].lon, fields[0].lat, members
    if fhr == 0 and region == list(REGIONS)[0]:
        log.info("ensemble fields at f000 (%d members): %s", len(members), " ".join(sorted(keys)))
    meta = {"run": run, "fhr": fhr, "bbox": bbox, "region": region, "region_name": REGIONS[region]["name"], "members": members}
    written = []
    for pid in param_ids:
        if fhr not in param_hours(pid):
            continue
        fn = getattr(ensemble, PARAMS[pid]["plot"])
        dest = Path(out_dir) / region / pid / f"f{fhr:03d}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig = fn(stack, meta)
            fig.savefig(dest, dpi=fig.dpi, facecolor="white"); plt.close(fig)
            compress_png(dest); written.append(str(dest))
        except Exception as e:  # noqa: BLE001
            plt.close("all")
            log.error("failed %s %s f%03d: %s", region, pid, fhr, str(e)[:160])
    return written


def write_manifest(run_id: str, hours: list[int], regions: list[str], param_ids: list[str]):
    man_path = SITE / MANIFEST_NAME
    manifest = None
    if storage.enabled():
        manifest = storage.get_json(MANIFEST_NAME)   # merge with what's already published
    if manifest is None and man_path.exists():
        try:
            manifest = json.loads(man_path.read_text())
        except json.JSONDecodeError:
            manifest = None
    manifest = manifest or {"model": {}, "regions": {}, "params": {}}
    runs = [r for r in manifest.get("model", {}).get("runs", []) if r["id"] != run_id]
    limited = {pid: [h for h in hours if h in param_hours(pid)] for pid in param_ids
               if PARAMS[pid].get("max_hour") is not None}
    runs.append({
        "id": run_id,
        "init": dt.datetime.strptime(run_id, "%Y%m%d%H").replace(tzinfo=dt.timezone.utc).isoformat(),
        "hours": hours, "regions": regions, "params": param_ids,
        "param_hours": limited,           # products rendered over fewer hours than the run
    })
    runs.sort(key=lambda r: r["id"], reverse=True)
    manifest["model"] = {"id": MODEL["id"], "name": MODEL["name"], "resolution": MODEL["resolution"],
                         "credit": MODEL.get("credit", ""), "runs": runs[:KEEP_RUNS]}
    manifest["regions"] = {k: {"name": v["name"]} for k, v in REGIONS.items()}
    manifest["params"] = {k: {"name": v["name"], "group": v["group"]} for k, v in PARAMS.items()}
    manifest["path"] = "images/{model}/{run}/{region}/{param}/f{hour}.png"
    manifest["generated"] = dt.datetime.now(dt.timezone.utc).isoformat()
    man_path.write_text(json.dumps(manifest, indent=1))
    return manifest


def prune_runs(keep_ids: list[str]):
    if storage.enabled():
        for rid in storage.list_prefixes(f"images/{MODEL['id']}"):
            if rid not in keep_ids:
                storage.delete_prefix(f"images/{MODEL['id']}/{rid}/")
        return
    img_root = SITE / "images" / MODEL["id"]
    if not img_root.exists():
        return
    for d in img_root.iterdir():
        if d.is_dir() and d.name not in keep_ids:
            log.info("pruning old run %s", d.name)
            shutil.rmtree(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="YYYYMMDDHH; default = latest available on NOMADS")
    ap.add_argument("--hours", default=None, help="e.g. 0-120/6 or 0,6,12")
    ap.add_argument("--regions", nargs="*", default=MODEL.get("regions", list(REGIONS)))
    ap.add_argument("--params", nargs="*", default=model_params())
    ap.add_argument("--workers", type=int, default=MODEL.get("workers", 4))
    ap.add_argument("--synthetic", action="store_true", help="fake data, no network")
    ap.add_argument("--keep-grib", action="store_true")
    ap.add_argument("--manifest-only", action="store_true", help="write manifest for images already in site/")
    args = ap.parse_args()

    hours = parse_hours(args.hours) if args.hours else FORECAST_HOURS
    session = requests.Session()
    session.headers["User-Agent"] = "wxmodels-renderer (github actions)"

    if args.run:
        run = dt.datetime.strptime(args.run, "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)
    elif args.synthetic:
        run = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
        run = run.replace(hour=(run.hour // 6) * 6)
    else:
        run = latest_available_run(session=session)
    run_id = run.strftime("%Y%m%d%H")
    log.info("rendering %s run %s: %d hours × %d regions × %d params",
             MODEL["name"], run_id, len(hours), len(args.regions), len(args.params))

    out_dir = SITE / "images" / MODEL["id"] / run_id

    if args.manifest_only:
        # only list hours that actually have images, so a failed slice doesn't leave 404 frames
        have = sorted({int(p.stem[1:]) for p in out_dir.rglob("f*.png")}) if out_dir.exists() else []
        hours = [h for h in hours if h in have] or hours
        manifest = write_manifest(run_id, hours, args.regions, args.params)
        if storage.enabled():
            storage.put_json(manifest, MANIFEST_NAME)
        prune_runs([r["id"] for r in manifest["model"]["runs"]])
        log.info("%s written: %d hours", MANIFEST_NAME, len(hours))
        return

    grib_dir = Path(tempfile.mkdtemp(prefix="wx_grib_")) if not args.keep_grib else ROOT / "grib" / run_id
    pairs = all_fetch_pairs(args.params) if MODEL["source"] in ("nomads", "nomads_grid", "gefs") else set()

    # 1. download (sequential; both servers rate-limit aggressive parallel clients)
    #    GFS: one regional subset per (hour, region) plus small previous-step subsets.
    #    ECMWF: one global file per step, shared by every region.
    prev = prev_steps(args.params)
    grib_paths: dict[tuple[int, str], dict | None] = {}
    gfs_jobs: list[tuple[int, str]] = []
    for fhr in hours:
        if args.synthetic:
            for region in args.regions:
                grib_paths[(fhr, region)] = None
            continue
        if ENSEMBLE and MODEL["source"] in ("ecmwf_ens", "ecmwf_aifs_ens", "geps"):
            files = {}
            dl = (lambda r, st, fl, d: download_geps(r, st, fl, d, session)) if MODEL["source"] == "geps" else download_ecmwf_ens
            try:
                main = dl(run, fhr, MODEL["ens_fields"], grib_dir / f"ens_f{fhr:03d}.grib2")
                prev_f = dl(run, fhr - 6, [("tp", None)], grib_dir / f"ens_f{fhr-6:03d}_tp.grib2") if fhr >= 6 else None
                npz = pack_members(main, prev_f, MODEL["domain"], grib_dir / f"ens_f{fhr:03d}.npz")
                for pth in (main, prev_f):
                    if pth:
                        Path(pth).unlink(missing_ok=True)      # free disk: the .npz is all we need now
                files["npz"] = str(npz)
            except Exception as e:  # noqa: BLE001
                log.error("f%03d: %s", fhr, e); continue
            for region in args.regions:
                grib_paths[(fhr, region)] = files
            continue
        if ENSEMBLE:
            bbox = MODEL["domain"]
            files = {}
            def one(m):
                dest = grib_dir / f"{m}_f{fhr:03d}.grb2"
                try:
                    if MODEL["source"] == "aigefs":
                        download_aigefs_member(run, fhr, m, dest, session)
                    else:
                        download(gefs_member_url(run, fhr, m, pairs, bbox), dest, session, retries=5)
                    return m, str(dest)
                except RuntimeError as e:
                    log.warning("f%03d member %s: %s", fhr, m, e); return m, None
            from concurrent.futures import ThreadPoolExecutor as _TPE
            with _TPE(max_workers=4) as pool:
                for m, path in pool.map(one, MODEL["members"]):
                    if path:
                        files[m] = path
            for region in args.regions:
                grib_paths[(fhr, region)] = files
            continue
        if MODEL["source"] == "nomads_grid":
            files = {}
            dest = grib_dir / f"grid_f{fhr:03d}.grb2"
            try:
                have = available_pairs(run, fhr, pairs, session)
                if not have:
                    raise RuntimeError(f"f{fhr:03d}: none of the requested fields are in this file")
                download(build_filter_url(run, fhr, have, None), dest, session)
                files[""] = str(dest)
            except RuntimeError as e:
                log.error("f%03d: %s", fhr, e); continue
            for off, spec in prev.items():
                step = step_for(fhr, off)
                if step is None or not spec["fetch"]:
                    continue
                tag = "_f0" if off == "f0" else f"_m{off}"
                pdest = grib_dir / f"grid_f{step:03d}{tag}.grb2"
                try:
                    phave = available_pairs(run, step, spec["fetch"], session)
                    if not phave:
                        continue
                    download(build_filter_url(run, step, phave, None), pdest, session, retries=3)
                    files[tag] = str(pdest)
                except RuntimeError as e:
                    log.warning("%s", e)
            for region in args.regions:
                grib_paths[(fhr, region)] = files
            continue
        if MODEL["source"] not in ("nomads",):
            fetch = download_ecmwf if MODEL["source"] == "ecmwf_opendata" else \
                    (lambda r, st, prs, d: download_files(r, st, prs, d, session))
            files = {}
            try:
                files[""] = str(fetch(run, fhr, ecmwf_pairs(args.params), grib_dir / f"global_f{fhr:03d}.grib2"))
            except (RuntimeError, Exception) as e:  # noqa: BLE001
                log.error("f%03d: %s", fhr, e); continue
            for off, spec in prev.items():
                step = step_for(fhr, off)
                if step is None or not spec["ecmwf"]:
                    continue
                tag = "_f0" if off == "f0" else f"_m{off}"
                try:
                    files[tag] = str(fetch(run, step, spec["ecmwf"], grib_dir / f"global_f{step:03d}_{tag}.grib2"))
                except Exception as e:  # noqa: BLE001
                    log.warning("%s", e)
            for region in args.regions:
                grib_paths[(fhr, region)] = files
            continue
        for region in args.regions:
            gfs_jobs.append((fhr, region))

    def fetch_gfs(job):
        fhr, region = job
        bbox = padded(REGIONS[region]["bbox"])
        files = {}
        try:
            dest = grib_dir / f"{region}_f{fhr:03d}.grb2"
            download_grouped(run, fhr, pairs, bbox, dest, session)
            files[""] = str(dest)
        except RuntimeError as e:
            log.error("%s", e); return job, None
        for off, spec in prev.items():
            step = step_for(fhr, off)
            if step is None or not spec["fetch"]:
                continue
            tag = "_f0" if off == "f0" else f"_m{off}"
            dest = grib_dir / f"{region}_f{step:03d}{tag}.grb2"
            try:
                download(build_filter_url(run, step, spec["fetch"], bbox), dest, session, retries=3)
                files[tag] = str(dest)
            except RuntimeError as e:
                log.warning("%s", e)         # e.g. no APCP at step 0 — plots degrade gracefully
        return job, files

    if gfs_jobs:
        from concurrent.futures import ThreadPoolExecutor as _TPE
        with _TPE(max_workers=3) as pool:          # NOMADS tolerates a few concurrent clients
            for job, files in pool.map(fetch_gfs, gfs_jobs):
                if files:
                    grib_paths[job] = files

    # 1b. second pass: anything that failed gets one more try after the server has had a breather
    missing = [(fhr, region) for fhr in hours for region in args.regions
               if not args.synthetic and (fhr, region) not in grib_paths and MODEL["source"] == "nomads"]
    if missing:
        log.info("retrying %d failed frame downloads", len(missing))
        time.sleep(60)
        for fhr, region in missing:
            bbox = padded(REGIONS[region]["bbox"])
            dest = grib_dir / f"{region}_f{fhr:03d}.grb2"
            try:
                download_grouped(run, fhr, pairs, bbox, dest, session, retries=6)
                grib_paths[(fhr, region)] = {"": str(dest)}
            except RuntimeError as e:
                log.error("still failing: %s", e)

    # 1c. warm the basemap cache once here, so worker processes don't race to
    #     download the same Natural Earth zips and corrupt each other's copies
    layers = plots._basemap_layers()
    if len(layers) < 3:
        log.warning("only %d/3 basemap layers loaded; retrying once", len(layers))
        plots._BASEMAP = None
        import shutil as _sh
        _sh.rmtree(Path.home() / ".local/share/cartopy", ignore_errors=True)
        layers = plots._basemap_layers()
    log.info("basemap layers ready: %d/3", len(layers))

    # 2. render in parallel
    jobs = [(fhr, region, p) for (fhr, region), p in grib_paths.items()]
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(render_frame, run.isoformat(), fhr, region, args.params,
                          p, str(out_dir), args.synthetic) for fhr, region, p in jobs]
        for fut in as_completed(futs):
            try:
                n_done += len(fut.result())
            except Exception as e:  # noqa: BLE001
                log.error("frame failed: %s", str(e)[:200])
            if n_done % 25 == 0:
                log.info("%d images written", n_done)
    log.info("done: %d images", n_done)

    # 3. publish: R2 if configured, else leave in site/ for the Pages artifact
    if storage.enabled():
        storage.upload_dir(out_dir, f"images/{MODEL['id']}/{run_id}")
    manifest = write_manifest(run_id, hours, args.regions, args.params)
    if storage.enabled():
        storage.put_json(manifest, MANIFEST_NAME)
    prune_runs([r["id"] for r in manifest["model"]["runs"]])
    if storage.enabled():
        shutil.rmtree(out_dir, ignore_errors=True)   # don't ship images in the Pages artifact too
        (SITE / MANIFEST_NAME).unlink(missing_ok=True)
    if not args.keep_grib:
        shutil.rmtree(grib_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
