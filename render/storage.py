"""
Optional Cloudflare R2 storage. When the R2_* environment variables are set,
rendered images and manifests are uploaded to the bucket and the frontend reads
them from there. When they're absent, everything stays in site/ and ships with
the GitHub Pages artifact exactly as before.

Environment:
  R2_ACCOUNT_ID         Cloudflare account id
  R2_ACCESS_KEY_ID      from an R2 API token (Object Read & Write)
  R2_SECRET_ACCESS_KEY
  R2_BUCKET             bucket name, e.g. wxmodels
"""
from __future__ import annotations

import json
import logging
import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

log = logging.getLogger("storage")

_client = None


def enabled() -> bool:
    return all(os.environ.get(k) for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"))


def bucket() -> str:
    return os.environ["R2_BUCKET"]


def client():
    global _client
    if _client is None:
        import boto3
        from botocore.config import Config
        _client = boto3.client(
            "s3",
            endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
            config=Config(retries={"max_attempts": 6, "mode": "adaptive"}, max_pool_connections=32),
        )
    return _client


def _cache_control(key: str) -> str:
    # Images for a given run never change once written; manifests do.
    return "no-cache" if key.endswith(".json") else "public, max-age=2592000, immutable"


def upload_file(local: Path, key: str):
    ctype = mimetypes.guess_type(str(local))[0] or "application/octet-stream"
    client().upload_file(str(local), bucket(), key,
                         ExtraArgs={"ContentType": ctype, "CacheControl": _cache_control(key)})


def upload_dir(local_dir: Path, prefix: str, workers: int = 16) -> int:
    files = [p for p in local_dir.rglob("*") if p.is_file()]
    def one(p: Path):
        upload_file(p, f"{prefix}/{p.relative_to(local_dir).as_posix()}")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, files))
    log.info("uploaded %d files to %s/%s", len(files), bucket(), prefix)
    return len(files)


def put_json(obj, key: str):
    client().put_object(Bucket=bucket(), Key=key, Body=json.dumps(obj, indent=1).encode(),
                        ContentType="application/json", CacheControl="no-cache")


def get_json(key: str):
    try:
        r = client().get_object(Bucket=bucket(), Key=key)
        return json.loads(r["Body"].read())
    except Exception as e:  # noqa: BLE001  (NoSuchKey on first run is normal)
        log.info("no existing %s in bucket (%s)", key, type(e).__name__)
        return None


def list_prefixes(prefix: str) -> list[str]:
    """Immediate 'subdirectories' under prefix, e.g. run ids under images/gfs/."""
    out = []
    paginator = client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket(), Prefix=prefix.rstrip("/") + "/", Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            out.append(cp["Prefix"].rstrip("/").split("/")[-1])
    return out


def delete_prefix(prefix: str) -> int:
    paginator = client().get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=bucket(), Prefix=prefix):
        keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        for i in range(0, len(keys), 1000):
            client().delete_objects(Bucket=bucket(), Delete={"Objects": keys[i:i + 1000], "Quiet": True})
        n += len(keys)
    log.info("deleted %d objects under %s", n, prefix)
    return n
