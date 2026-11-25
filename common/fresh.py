#!/usr/bin/env python3
"""Is a kit still on the newest upstream pieces? Compares kits/<kit>/pins.json with upstream:
  - hrx-loom: installed (the venv's python, VENV or --venv), the version the kit was measured with (pins "hrx_loom"),
    and PyPI's latest release;
  - each pinned file: its sha256 at the pinned revision against the same path at the repository's main (Hugging Face API).
Prints one line per piece. Exit 1 when anything is behind or changed upstream, 2 when a check could not run (network, a file
missing upstream): never silently green. A model repository can re-upload a file under the same name (an unnoticed
re-upload once kept a server on old weights for weeks), and Loom ships releases between sessions (0.3.0 went unnoticed for a
day); run this at the start of a session and before blaming the SDK or the model.

Loom tool surface (reported, never changes the exit code): per installed Loom (--venv, and --venv-next for the one a port is
being prepared on), the `--help=all` text (`--help` where =all is refused) of every loom-* / iree-* / amdf-* binary in its bin/
and of each argparse subcommand, plus the docs/workflows file list at tag v<version> of an hrx-system checkout
(--hrx-system). Snapshots are cached per hrx-loom version (--surface-cache); each run prints the flags, binaries and workflows
that are new or removed against the cached snapshot of the same version (or, on a version's first run, the newest older one)
and between the installed and the next venv. A Loom release once shipped a sanitizer and a profiling filter nobody here used
for weeks. The diff sees help text only: an option accepted but not listed (loom-compile 0.2.0 accepts `--sanitizer=race`
while its help lists access / value / operation) is invisible to it; changed description lines are counted per tool.
usage: fresh.py KIT [--venv DIR] [--venv-next DIR] [--hrx-system CHECKOUT] [--surface-cache DIR]"""
import argparse
import concurrent.futures
import json
import os
import re
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


TOOL = re.compile(r"^(loom-|iree-|amdf-)")
FLAG = re.compile(r"(?<![\w-])(--[A-Za-z0-9][-A-Za-z0-9_.]*)")
SUBCOMMANDS = re.compile(r"^usage: \S+ .*?\{([\w,-]+)\}", re.M)   # argparse's `{show,diff,suggest}`


def help_text(argv, timeout=15):
    """(mode, text, error) for one tool: --help=all, else --help; error is None when one of them exits 0"""
    err = None
    for mode in ("--help=all", "--help"):
        try:
            r = subprocess.run(argv + [mode], capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=timeout)
        except (OSError, subprocess.SubprocessError) as e:
            err = f"{mode}: {e}"
            continue
        if r.returncode == 0:
            return mode, r.stdout + r.stderr, None
        err = f"{mode}: exit {r.returncode}: {((r.stderr or r.stdout).strip().splitlines() or [''])[0][:120]}"
    return None, "", err


def snapshot(bin_dir, version, hrx=None):
    """{version, tools: {name: {mode, flags, lines, error}}, workflows: [...] | None, workflows_note} for one venv's bin/"""
    names = sorted(n for n in os.listdir(bin_dir) if TOOL.match(n) and os.access(os.path.join(bin_dir, n), os.X_OK))
    tools = {}

    def one(argv, key):
        mode, text, err = help_text(argv)
        tools[key] = {"mode": mode, "flags": sorted({f.rstrip(".") for f in FLAG.findall(text)}), "lines": text.splitlines(), "error": err}
        return text

    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        texts = dict(zip(names, ex.map(lambda n: one([os.path.join(bin_dir, n)], n), names)))
        subs = [(n, s) for n in names for m in SUBCOMMANDS.findall(texts[n])[:1] for s in m.split(",")]
        list(ex.map(lambda ns: one([os.path.join(bin_dir, ns[0]), ns[1]], f"{ns[0]} {ns[1]}"), subs))
    wf, note = None, "not checked (no hrx-system checkout given)"
    if hrx:
        r = subprocess.run(["git", "-C", hrx, "ls-tree", "-r", "--name-only", f"v{version}", "--", "loom/docs/src/workflows"],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            wf, note = sorted(x.removeprefix("loom/docs/src/") for x in r.stdout.split()), f"{hrx} at tag v{version}"
        else:
            note = f"unknown (tried: git ls-tree v{version} in {hrx}: {(r.stderr.strip() or 'no files')[:120]})"
    return {"version": version, "bin": bin_dir, "tools": dict(sorted(tools.items())), "workflows": wf, "workflows_note": note}


def surface_diff(old, new):
    """lines naming new / removed binaries, flags, workflows, and changed help lines, old -> new"""
    out = []
    ot, nt = old["tools"], new["tools"]
    out += [f"+ binary {t}" for t in nt if t not in ot] + [f"- binary {t}" for t in ot if t not in nt]
    for t in nt:
        if t not in ot:
            continue
        a, b = set(ot[t]["flags"]), set(nt[t]["flags"])
        if b - a or a - b:
            out.append(f"{t}: " + "; ".join(x for x in (" ".join("+" + f for f in sorted(b - a)), " ".join("-" + f for f in sorted(a - b))) if x))
        la, lb = set(ot[t]["lines"]), set(nt[t]["lines"])
        if la != lb:
            out.append(f"{t}: help text +{len(lb - la)} / -{len(la - lb)} lines")
        if bool(ot[t]["error"]) != bool(nt[t]["error"]):
            out.append(f"{t}: help {'now fails: ' + nt[t]['error'] if nt[t]['error'] else 'works again'}")
    if old["workflows"] is not None and new["workflows"] is not None:
        a, b = set(old["workflows"]), set(new["workflows"])
        out += [f"+ workflow {w}" for w in sorted(b - a)] + [f"- workflow {w}" for w in sorted(a - b)]
    return out


def default_cache():
    return os.environ.get("LOOM_SURFACE_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "loom-strix-kits", "loom-surface")


def vkey(v):
    return tuple(int(x) if x.isdigit() else -1 for x in re.split(r"[.+-]", v))


def surface(venvs, cache, hrx=None):
    """report lines for [(label, venv)]; never raises: a failure is a line"""
    lines, snaps = [], []
    vers = {}
    for _, venv in venvs:
        try:
            vers[venv] = installed_version(venv)
        except OSError:   # no python in that venv: reported below as an unknown version
            vers[venv] = None
    for label, venv in venvs:
        try:
            ver = vers[venv]
            if ver is None:
                lines.append(f"LOOM SURFACE: {label} {venv}: hrx-loom version unknown, not snapshotted")
                continue
            snap = snapshot(os.path.join(venv, "bin"), ver, hrx)
            snaps.append((label, snap))
            for t, x in snap["tools"].items():
                if x["error"]:
                    lines.append(f"LOOM SURFACE: {ver} {t}: help failed ({x['error']})")
            os.makedirs(cache, exist_ok=True)
            mine = os.path.join(cache, f"{ver}.json")
            # an older version that another venv here holds is compared below (installed vs next), not twice
            older = sorted((f[:-5] for f in os.listdir(cache) if f.endswith(".json") and vkey(f[:-5]) < vkey(ver)
                            and f[:-5] not in vers.values()), key=vkey)
            base = mine if os.path.exists(mine) else os.path.join(cache, older[-1] + ".json") if older else None
            if base is None:
                lines.append(f"LOOM SURFACE: {label} {ver}: first snapshot ({len(snap['tools'])} tools; workflows: {snap['workflows_note']})")
            else:
                prev = json.load(open(base))
                d = surface_diff(prev, snap)
                what = "same version, earlier snapshot" if base == mine else f"previous release {prev['version']}"
                lines.append(f"LOOM SURFACE: {label} {ver} vs {what}: " + (f"{len(d)} changes" if d else "no change"))
                lines += [f"LOOM SURFACE:   {x}" for x in d]
            tmp = mine + ".tmp"
            with open(tmp, "w") as f:
                json.dump(snap, f)
            os.replace(tmp, mine)
        except Exception as e:   # noqa: BLE001 - the surface is a report; it never fails the freshness checks
            lines.append(f"LOOM SURFACE: {label} {venv}: check failed: {e}")
    if len(snaps) == 2 and snaps[0][1]["version"] != snaps[1][1]["version"]:
        (la, a), (lb, b) = snaps
        d = surface_diff(a, b)
        lines.append(f"LOOM SURFACE: {lb} {b['version']} vs {la} {a['version']}: {len(d)} changes")
        lines += [f"LOOM SURFACE:   {x}" for x in d]
    if snaps:
        lines.append(f"LOOM SURFACE: snapshots in {cache}/<version>.json (help text per tool under .tools[NAME].lines)")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kit")
    ap.add_argument("--variant", default=os.environ.get("KIT_VARIANT"), help="kits/<kit>/variants/<name>/pins.json instead of the kit's")
    ap.add_argument("--venv", default=os.environ.get("VENV"))
    ap.add_argument("--venv-next", default=os.environ.get("VENV_NEXT"), help="a venv with the next hrx-loom (a port in preparation)")
    ap.add_argument("--hrx-system", default=os.environ.get("HRX_SYSTEM"), help="an hrx-system git checkout (tags v<version>)")
    ap.add_argument("--surface-cache", default=default_cache())
    a = ap.parse_args()
    pins = json.load(open(os.path.join(HERE, "..", "kits", a.kit, *(["variants", a.variant] if a.variant else []), "pins.json")))
    lines, stale, broken = check(pins, installed_version(a.venv))
    for x in lines:
        print(x)
    for s in stale:
        print("STALE:", s)
    for b in broken:
        print("CHECK FAILED:", b)
    venvs = [(k, v) for k, v in (("installed", a.venv), ("next", a.venv_next)) if v]
    for x in surface(venvs, a.surface_cache, a.hrx_system) if venvs else ["LOOM SURFACE: not checked (no --venv)"]:
        print(x)
    return 2 if broken else 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
