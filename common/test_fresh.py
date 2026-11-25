#!/usr/bin/env python3
"""Tests of fresh.check() with a fake upstream (the network is the true boundary), and of the Loom tool-surface diff over real
processes: small executable scripts stand in for a venv's tools, and, when a venv with hrx-loom is given (VENV), its real tools.
Each case names the recurrence it prevents.
usage: [VENV=DIR] python3 test_fresh.py (exit 1 on a failure)"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import fresh  # noqa: E402

PINS = {"hrx_loom": "0.2.0", "files": [{"repo": "u/m", "revision": "r" * 40, "path": "m.gguf", "sha256": "a" * 64, "dest": "models"}]}


def upstream(version, sha, missing=False, down=False, pypi_down=False):
    def get(url):
        if down or (pypi_down and "pypi.org" in url):
            raise OSError("network down")
        if "pypi.org" in url:
            return {"info": {"version": version}}
        return [] if missing else [{"path": "m.gguf", "size": 1, "lfs": {"oid": sha}}]
    return get


fails = []


def case(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


_, s, b = fresh.check(PINS, "0.2.0", upstream("0.2.0", "a" * 64))
case("all current -> neither stale nor broken", not s and not b)
# an unnoticed re-upload under the same name kept a server on old weights for weeks
_, s, b = fresh.check(PINS, "0.2.0", upstream("0.2.0", "b" * 64))
case("file re-uploaded upstream -> stale", any("changed upstream" in x for x in s) and not b)
# a Loom release went unnoticed for a day
_, s, b = fresh.check(PINS, "0.2.0", upstream("0.3.0", "a" * 64))
case("newer hrx-loom on PyPI -> stale (installed and kit both named)", len(s) == 2 and not b)
case("never silently green: network down -> broken", bool(fresh.check(PINS, "0.2.0", upstream("", "", down=True))[2]))
case("never silently green: PyPI unreachable alone -> broken", bool(fresh.check(PINS, "0.2.0", upstream("0.2.0", "a" * 64, pypi_down=True))[2]))
case("never silently green: pinned file gone upstream -> broken", bool(fresh.check(PINS, "0.2.0", upstream("0.2.0", "", missing=True))[2]))
case("never silently green: unknown installed version -> broken", bool(fresh.check(PINS, None, upstream("0.2.0", "a" * 64))[2]))

# ---- Loom tool surface ----
TMP = tempfile.mkdtemp(prefix="test_fresh_")
CACHE = os.path.join(TMP, "cache")


def venv(name, version, tools):
    """a venv directory: bin/python prints VERSION (installed_version's probe), each tool a script with that body"""
    b = os.path.join(TMP, name, "bin")
    os.makedirs(b, exist_ok=True)
    for f in os.listdir(b):
        os.remove(os.path.join(b, f))
    for n, body in {"python": f"echo {version}", **tools}.items():
        with open(os.path.join(b, n), "w") as f:
            f.write("#!/bin/bash\n" + body + "\n")
        os.chmod(os.path.join(b, n), 0o755)
    return os.path.join(TMP, name)


ALL = 'case "$1" in --help=all) echo "--alpha=1"; echo "# --beta=..." ;; *) exit 2 ;; esac'
ARGP = ('[ "$1" = --help=all ] && { echo "error: ignored explicit argument" >&2; exit 2; }\n'
        'case "$1" in show) echo "  --function F";; *) echo "usage: loom-rep [-h] {show,diff} ..."; echo "  --agents_md";; esac')
v1 = venv("v1", "9.1.0", {"loom-a": ALL, "loom-rep": ARGP, "loom-gone": ALL, "not-a-tool": "exit 0"})
out = fresh.surface([("installed", v1)], CACHE)
case("first snapshot of a version is named, not silently empty", any("installed 9.1.0: first snapshot" in x for x in out))
snap = fresh.snapshot(os.path.join(v1, "bin"), "9.1.0")
# loom-compile-report and loom-export-cxx refuse --help=all: they must fall back to --help, not drop out
case("a tool refusing --help=all falls back to --help", snap["tools"]["loom-rep"]["mode"] == "--help" and "--agents_md" in snap["tools"]["loom-rep"]["flags"])
case("argparse subcommands are snapshotted (loom-compile-report suggest gained --expand in 0.3.0)",
     "--function" in snap["tools"].get("loom-rep show", {}).get("flags", []) and "loom-rep diff" in snap["tools"])
case("only loom-* / iree-* / amdf-* binaries", "not-a-tool" not in snap["tools"] and "python" not in snap["tools"])
case("flags from definitions and commented defaults", {"--alpha", "--beta"} <= set(snap["tools"]["loom-a"]["flags"]))

# the same version reinstalled: a binary gone, one new, a flag added, one tool whose help now errors
v1 = venv("v1", "9.1.0", {"loom-a": ALL.replace('echo "--alpha=1"', 'echo "--alpha=1"; echo "--sanitizer=race"'), "loom-rep": ARGP,
                          "loom-new": ALL, "loom-broken": "echo boom >&2; exit 3"})
out = fresh.surface([("installed", v1)], CACHE)
case("a removed binary is reported", any(x.endswith("- binary loom-gone") for x in out))
case("a new binary is reported", any(x.endswith("+ binary loom-new") for x in out))
case("a new flag is reported", any("loom-a: +--sanitizer" in x for x in out))
case("a tool whose --help errors is reported, not fatal", any("loom-broken: help failed" in x and "boom" in x for x in out)
     and any("+ binary loom-new" in x for x in out))

# a release: the next venv's first run is diffed against the installed one, once (not also as 'previous release')
v2 = venv("v2", "9.2.0", {"loom-a": ALL, "loom-rep": ARGP})
out = fresh.surface([("installed", v1), ("next", v2)], CACHE)
case("installed vs next venv: removed flag and binaries named", any("next 9.2.0 vs installed 9.1.0" in x for x in out)
     and any("loom-a: -" in x and "--sanitizer" in x for x in out) and any("- binary loom-new" in x for x in out))
case("a release diff is printed once", sum("- binary loom-new" in x for x in out) == 1)
# no-surprise path: a venv that does not exist is a report line, never an exception that hides the freshness verdict
out = fresh.surface([("installed", os.path.join(TMP, "nope"))], CACHE)
case("a missing venv is reported, surface never raises", any("version unknown" in x for x in out))
# a cache that cannot be written (a file in its place) is a report line, not an exception
blocked = os.path.join(TMP, "blocked")
open(blocked, "w").close()
case("an unwritable cache is reported, surface never raises", any("check failed" in x for x in fresh.surface([("installed", v1)], blocked)))
# docs workflows per tag of an hrx-system checkout (0.3.0 added workflows/oracles/pytorch.md)
hrx = os.path.join(TMP, "hrx")
os.makedirs(os.path.join(hrx, "loom/docs/src/workflows"))
G = ["git", "-C", hrx, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
import subprocess  # noqa: E402
subprocess.run(["git", "init", "-q", hrx], check=True)
for tag, name in (("v9.1.0", "a.md"), ("v9.2.0", "b.md")):
    open(os.path.join(hrx, "loom/docs/src/workflows", name), "w").close()
    subprocess.run(G + ["add", "-A"], check=True)
    subprocess.run(G + ["commit", "-qm", tag], check=True)
    subprocess.run(G + ["tag", tag], check=True)
wcache = os.path.join(TMP, "wcache")
out = fresh.surface([("installed", v1), ("next", v2)], wcache, hrx)
case("a workflow new at the next tag is reported", any(x.endswith("+ workflow workflows/b.md") for x in out)
     and not any("- workflow" in x for x in out))
case("a tag missing from the checkout is 'unknown (tried: ...)', not an empty list",
     "unknown (tried" in fresh.snapshot(os.path.join(v1, "bin"), "8.0.0", hrx)["workflows_note"])
# an upgrade in place: a venv at a version newer than every cached one, its tools all gone
v3 = venv("v3", "9.3.0", {})
shutil.rmtree(os.path.join(v3, "bin"))
os.makedirs(os.path.join(v3, "bin"))
with open(os.path.join(v3, "bin", "python"), "w") as f:
    f.write("#!/bin/bash\necho 9.3.0\n")
os.chmod(os.path.join(v3, "bin", "python"), 0o755)
out = fresh.surface([("x", v3)], CACHE)
case("a version's first run is diffed against the newest older cached release (an upgrade in place)",
     any("x 9.3.0 vs previous release 9.2.0" in x for x in out) and any("- binary loom-a" in x for x in out))

# the real red: 0.2.0's help with every --sanitizer line removed, then the real 0.2.0 -> --sanitizer reported as new
real = os.environ.get("VENV")
n_real = 0
if real and fresh.installed_version(real):
    n_real = 1
    rb = os.path.join(real, "bin")
    tools = {n: f'out=$("{rb}/{n}" "$@" 2>&1); rc=$?; printf "%s\\n" "$out" | grep -v -- --sanitizer; exit $rc'
             for n in os.listdir(rb) if fresh.TOOL.match(n)}
    fake = venv("real-minus-sanitizer", "x", tools)
    with open(os.path.join(fake, "bin", "python"), "w") as f:   # the real version: same cache key as the real venv
        f.write(f'#!/bin/bash\nexec "{rb}/python" "$@"\n')
    rc_cache = os.path.join(TMP, "real-cache")
    fresh.surface([("installed", fake)], rc_cache)
    out = fresh.surface([("installed", real)], rc_cache)
    got = sorted(x.split("LOOM SURFACE:")[1].split(":")[0].strip() for x in out if "+--sanitizer" in x)
    print("   real venv:", real, "->", got)
    case("real Loom: help without --sanitizer lines, then the real help -> --sanitizer new in loom-test and loom-compile",
         {"loom-test", "loom-compile"} <= set(got))
else:
    print("SKIP real-Loom red case (set VENV to a venv with hrx-loom)")
shutil.rmtree(TMP)
total = 22 + n_real
print(f"{total - len(fails)}/{total} passed")
sys.exit(1 if fails else 0)
