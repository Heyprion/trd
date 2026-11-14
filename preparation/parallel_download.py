"""Resumable HTTP ranges with exact range checks and SHA256 verification."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import requests


def download(url, target, size, sha256, workers=32):
    target = Path(target)
    if target.exists() and target.stat().st_size == size:
        with target.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() == sha256:
                return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    state_path = target.with_name(target.name + ".ranges.json")
    block = 8 * 1024**2
    identity = {"size": size, "sha256": sha256, "block": block}
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    completed = set(state.get("completed", [])) if partial.exists() and all(state.get(k) == v for k, v in identity.items()) else set()
    fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(fd, size)
    lock = threading.Lock()
    connections = threading.local()

    def chunk(start):
        end = min(start + block, size) - 1
        for attempt in range(8):
            try:
                if not hasattr(connections, "session"):
                    connections.session = requests.Session()
                with connections.session.get(url, headers={"Range": f"bytes={start}-{end}"},
                                             timeout=(30, 60), stream=True) as response:
                    response.raise_for_status()
                    expected = f"bytes {start}-{end}/{size}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                        raise RuntimeError(f"Invalid range: {response.headers.get('Content-Range')}; expected {expected}")
                    payload = response.raw.read(end - start + 1)
                    if len(payload) != end - start + 1:
                        raise RuntimeError(f"Incomplete range: {len(payload)} bytes")
                content, offset = memoryview(payload), start
                while content:
                    n = os.pwrite(fd, content, offset)
                    content, offset = content[n:], offset + n
                with lock:
                    completed.add(start)
                    temporary_state = state_path.with_suffix(".tmp")
                    temporary_state.write_text(json.dumps({**identity, "completed": sorted(completed)}))
                    temporary_state.replace(state_path)
                    if len(completed) % 64 == 0:
                        print(f"{target.name}: {min(len(completed)*block,size)/1024**3:.2f}/{size/1024**3:.2f} GiB", flush=True)
                return
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(min(attempt + 1, 5))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(chunk, (s for s in range(0, size, block) if s not in completed)))
    finally:
        os.close(fd)
    with partial.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != sha256:
        state_path.unlink(missing_ok=True)
        raise RuntimeError(f"SHA256 mismatch: {target}")
    partial.replace(target)
    state_path.unlink(missing_ok=True)
    print(f"SHA256 verified: {target.name}", flush=True)
