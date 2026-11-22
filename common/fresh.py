#!/usr/bin/env python3
"""Is a kit still on the newest upstream pieces? Compares kits/<kit>/pins.json with upstream:
  - hrx-loom: installed (the venv's python, VENV or --venv), the version the kit was measured with (pins "hrx_loom"),
    and PyPI's latest release;
  - each pinned file: its sha256 at the pinned revision against the same path at the repository's main (Hugging Face API).
Prints one line per piece. Exit 1 when anything is behind or changed upstream, 2 when a check could not run (network, a file
missing upstream): never silently green. A model repository can re-upload a file under the same name (an unnoticed
re-upload once kept a server on old weights for weeks), and Loom ships releases between sessions (0.3.0 went unnoticed for a
day); run this at the start of a session and before blaming the SDK or the model.
usage: fresh.py KIT [--venv DIR]"""
import argparse
import json
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.realpath(__file__))


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "loom-strix-kits-fresh"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def check(pins, installed, get=get_json):
    """(lines, stale, broken) for a pins dict, the installed hrx-loom version (or None) and a JSON getter"""
    lines, stale, broken = [], [], []
    try:
        latest = get("https://pypi.org/pypi/hrx-loom/json")["info"]["version"]
        tested = pins.get("hrx_loom")
        lines.append(f"hrx-loom: installed {installed or 'unknown'}, kit measured on {tested}, PyPI latest {latest}")
        if installed is None:
            broken.append("hrx-loom: installed version unknown (set VENV or --venv)")
        elif installed != latest:
            stale.append(f"hrx-loom {installed} < {latest} (PyPI)")
        if tested and tested != latest:
            stale.append(f"kit measured on hrx-loom {tested}, PyPI has {latest}: re-measure before trusting its numbers")
    except Exception as e:   # noqa: BLE001 - reported, never green
        broken.append(f"hrx-loom check: {e}")
    trees = {}
    for f in pins["files"]:
        repo, path = f["repo"], f["path"]
        try:
            if repo not in trees:
                trees[repo] = {x["path"]: x for x in get(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true")}
            up = trees[repo].get(path)
            if up is None:
                broken.append(f"{repo}: {path} not at main")
                continue
            sha = (up.get("lfs") or {}).get("oid") or up.get("oid")
            same = sha == f["sha256"] or (not up.get("lfs") and up.get("size") is not None and f.get("size") == up.get("size"))
            lines.append(f"{repo}/{path}: pinned {f['revision'][:8]} {f['sha256'][:12]}, main {sha[:12] if sha else '?'} -> {'same' if same else 'CHANGED'}")
            if not same:
                stale.append(f"{repo}/{path} changed upstream since revision {f['revision'][:8]} (re-uploaded?)")
        except Exception as e:   # noqa: BLE001
            broken.append(f"{repo}/{path}: {e}")
    return lines, stale, broken


def installed_version(venv):
    py = os.path.join(venv, "bin", "python") if venv else sys.executable
    r = subprocess.run([py, "-c", "import importlib.metadata as m; print(m.version('hrx-loom'))"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kit")
    ap.add_argument("--variant", default=os.environ.get("KIT_VARIANT"), help="kits/<kit>/variants/<name>/pins.json instead of the kit's")
    ap.add_argument("--venv", default=os.environ.get("VENV"))
    a = ap.parse_args()
    pins = json.load(open(os.path.join(HERE, "..", "kits", a.kit, *(["variants", a.variant] if a.variant else []), "pins.json")))
    lines, stale, broken = check(pins, installed_version(a.venv))
    for x in lines:
        print(x)
    for s in stale:
        print("STALE:", s)
    for b in broken:
        print("CHECK FAILED:", b)
    return 2 if broken else 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
