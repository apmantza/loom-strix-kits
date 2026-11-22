#!/usr/bin/env python3
"""Download a kit's pinned files (kits/<kit>/pins.json) at their pinned revisions with the Hugging Face CLI, into WORK/<dest>
(WORK: the machine env's work dir, else the kit directory), then verify every sha256. Exit 1 on any mismatch or failed download.
usage: fetch.py KIT [--work DIR] [--verify-only]      needs `hf` (pip install -U huggingface_hub)"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.realpath(__file__))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kit")
    ap.add_argument("--variant", default=os.environ.get("KIT_VARIANT"), help="kits/<kit>/variants/<name>/pins.json instead of the kit's")
    ap.add_argument("--work", default=os.environ.get("WORK"))
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    kd = os.path.join(HERE, "..", "kits", a.kit)
    work = a.work or kd
    pins = json.load(open(os.path.join(kd, *(["variants", a.variant] if a.variant else []), "pins.json")))
    bad = 0
    for f in pins["files"]:
        dest = os.path.join(work, f["dest"])
        out = os.path.join(dest, f["path"])
        if not a.verify_only and not (os.path.exists(out) and sha256(out) == f["sha256"]):
            os.makedirs(dest, exist_ok=True)
            r = subprocess.run(["hf", "download", f["repo"], f["path"], "--revision", f["revision"], "--local-dir", dest])
            if r.returncode != 0:
                print(f"FAILED download {f['repo']}/{f['path']}"); bad += 1; continue
        ok = os.path.exists(out) and sha256(out) == f["sha256"]
        print(f"{'OK      ' if ok else 'MISMATCH'} {f['dest']}/{f['path']}")
        bad += not ok
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
