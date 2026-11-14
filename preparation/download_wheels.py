#!/usr/bin/env python3
"""Prefetch large PyPI wheels with bounded parallel ranges and SHA256 validation."""
import concurrent.futures
import hashlib
import os
from pathlib import Path
import sys
import time

import requests
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

DEST = Path(__file__).resolve().parent / "wheels"
TAGS = set(sys_tags())


def download(spec):
    name, version = spec.split("==")
    r = requests.get(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30)
    r.raise_for_status()
    candidates = [x for x in r.json()["urls"] if x["filename"].endswith(".whl")
                  and TAGS.intersection(parse_wheel_filename(x["filename"])[3])]
    if not candidates:
        raise RuntimeError(f"No matching wheel: {spec}")
    item = candidates[0]
    target = DEST / item["filename"]
    if target.exists() and hashlib.file_digest(target.open("rb"), "sha256").hexdigest() == item["digests"]["sha256"]:
        print(f"Cached: {target.name}", flush=True)
        return
    partial = target.with_suffix(".partial")
    fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(fd, item["size"])
    block = 8 * 1024**2

    def chunk(start):
        end = min(start + block, item["size"]) - 1
        for attempt in range(5):
            try:
                response = requests.get(item["url"], headers={"Range": f"bytes={start}-{end}"}, timeout=60)
                response.raise_for_status()
                if response.status_code != 206 or len(response.content) != end - start + 1:
                    raise RuntimeError(f"Invalid range response: {response.status_code}")
                content = memoryview(response.content)
                offset = start
                while content:
                    count = os.pwrite(fd, content, offset)
                    content, offset = content[count:], offset + count
                return
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(attempt + 1)

    print(f"Downloading: {target.name} ({item['size']/1024**2:.1f} MiB)", flush=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            list(pool.map(chunk, range(0, item["size"], block)))
    finally:
        os.close(fd)
    with partial.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    assert actual == item["digests"]["sha256"], f"SHA256 mismatch: {spec}"
    partial.replace(target)
    print(f"Verified: {target.name}", flush=True)


if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, sys.argv[1:]))
