#!/usr/bin/env python3
"""
CI gate: is there a new run to render?

Compares the newest complete run on the data server with runs[0] in the
manifest.json already live on the site (SITE_URL env), then writes GitHub
Actions job outputs:  run, needs_render, chunks (parallel hour slices).

Deliberately imports only fetch/config (numpy + requests), so the 30-minute
poll stays a ~20 s job.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import requests  # noqa: E402

from config import FORECAST_HOURS, MANIFEST_NAME, MODEL  # noqa: E402
from fetch import latest_available_run, run_max_hour  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("check")


def gh_output(**kv):
    path = os.environ.get("GITHUB_OUTPUT")
    for k, v in kv.items():
        line = f"{k}={v if isinstance(v, str) else json.dumps(v)}"
        print(line)
        if path:
            with open(path, "a") as fh:
                fh.write(line + "\n")


def parse_hours(spec: str) -> list[int]:
    if "-" in spec:
        rng, _, step = spec.partition("/")
        a, b = (int(x) for x in rng.split("-"))
        return list(range(a, b + 1, int(step or 6)))
    return [int(x) for x in spec.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="YYYYMMDDHH to render instead of the latest")
    ap.add_argument("--hours", help="e.g. 0-120/6; default = model's full range")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--chunks", type=int, default=6)
    args = ap.parse_args()

    hours = parse_hours(args.hours) if args.hours else FORECAST_HOURS
    session = requests.Session()
    session.headers["User-Agent"] = "wxmodels-check (github actions)"

    if args.run:
        run = dt.datetime.strptime(args.run, "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)
    else:
        run = latest_available_run(session=session)
    run_id = run.strftime("%Y%m%d%H")
    cap = run_max_hour(run, session) or hours[-1]
    hours = [h for h in hours if h <= cap]          # 06/18Z ECMWF runs are shorter

    published, published_max = None, None
    site = os.environ.get("SITE_URL")
    if site:
        try:
            r = session.get(site.rstrip("/") + "/" + MANIFEST_NAME, timeout=30, headers={"Cache-Control": "no-cache"})
            if r.ok:
                runs = r.json().get("model", {}).get("runs", [])
                published = runs[0]["id"] if runs else None
                published_max = max(runs[0]["hours"]) if runs and runs[0].get("hours") else None
            else:
                log.info("live manifest: HTTP %s (first deploy?)", r.status_code)
        except Exception as e:  # noqa: BLE001
            log.warning("could not read live manifest: %s", e)

    extended = published == run_id and published_max is not None and hours and hours[-1] > published_max
    needs = args.force or args.run is not None or published != run_id or extended
    n = max(1, min(args.chunks, len(hours)))
    slices = [",".join(str(h) for h in hours[i::n]) for i in range(n)]  # interleaved so slices finish together
    log.info("%s: latest run %s to %dh, live %s to %sh -> render=%s%s (%d slices)", MODEL["name"], run_id, cap,
             published, published_max, needs, " (extending live run)" if extended else "", n)
    gh_output(run=run_id, needs_render="true" if needs else "false", chunks=slices)


if __name__ == "__main__":
    main()
