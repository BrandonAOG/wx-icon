#!/usr/bin/env python3
"""
CI gate for a repo that renders several models into one Pages site.

Runs check.py once per model in WX_MODELS (space-separated) and combines the
results. Because a Pages deploy replaces the whole site, if ANY model has a
new run we re-render ALL of them (ensemble runs are cheap).

Outputs: needs_render, jobs (JSON list of {model, run, chunk, manifest}),
runs (JSON {model: run}).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent


def gh_output(**kv):
    path = os.environ.get("GITHUB_OUTPUT")
    for k, v in kv.items():
        line = f"{k}={v if isinstance(v, str) else json.dumps(v)}"
        print(line)
        if path:
            with open(path, "a") as fh:
                fh.write(line + "\n")


def main():
    models = os.environ.get("WX_MODELS", "gefs ecens").split()
    extra = sys.argv[1:]
    results = {}
    for m in models:
        out = tempfile.NamedTemporaryFile("r+", delete=False)
        env = dict(os.environ, WX_MODEL=m, WX_MANIFEST=f"manifest-{m}.json", GITHUB_OUTPUT=out.name)
        r = subprocess.run([sys.executable, str(HERE / "check.py"), *extra], env=env, text=True, capture_output=True)
        print(r.stdout, r.stderr[-2000:], sep="")
        if r.returncode != 0:
            print(f"::warning::{m}: check failed (no run available?) — skipping this model")
            continue
        kv = dict(line.split("=", 1) for line in Path(out.name).read_text().splitlines() if "=" in line)
        results[m] = {"run": kv["run"], "needs": kv["needs_render"] == "true", "chunks": json.loads(kv["chunks"])}
    needs = any(v["needs"] for v in results.values())
    jobs = [{"model": m, "run": v["run"], "chunk": c, "manifest": f"manifest-{m}.json"}
            for m, v in results.items() for c in v["chunks"]] if needs else []
    gh_output(needs_render="true" if needs else "false", jobs=jobs, runs={m: v["run"] for m, v in results.items()})


if __name__ == "__main__":
    main()
