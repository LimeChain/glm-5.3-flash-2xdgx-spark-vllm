#!/usr/bin/env python3
"""Verify provisioned checkpoint sizes and advertised LFS hashes, without loading tensors."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
manifest = json.loads(args.manifest.read_text())
started = time.time()


def check(row):
    file = args.model / row["name"]
    if not file.is_file():
        return {"name": row["name"], "pass": False, "error": "missing"}
    size = file.stat().st_size
    digest = hashlib.sha256()
    with file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    sha = digest.hexdigest()
    return {"name": row["name"], "size": size, "sha256": sha, "pass": size == row["size"] and (row["sha256"] is None or sha == row["sha256"])}


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    rows = list(executor.map(check, manifest["files"]))
receipt = {"revision": manifest["revision"], "pass": all(r["pass"] for r in rows), "elapsed_seconds": time.time() - started, "files": rows}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({k: v for k, v in receipt.items() if k != "files"}))
raise SystemExit(0 if receipt["pass"] else 1)
