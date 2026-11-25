#!/usr/bin/env python3
"""Tests of fresh.check() with a fake upstream (the network is the true boundary), and of the Loom tool-surface diff over real
processes: small executable scripts stand in for a venv's tools, and, when a venv with hrx-loom is given (VENV), its real tools.
Each case names the recurrence it prevents.
usage: [VENV=DIR] python3 test_fresh.py (exit 1 on a failure)"""
import atexit
import os
import shutil
import subprocess
import sys
import tempfile
import time

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


fails, ran = [], []


def case(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    ran.append(name)
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
atexit.register(shutil.rmtree, TMP, True)   # a case that raises must not leak the directory (review r1)
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

out = fresh.surface([("installed", v2), ("next", v1)], os.path.join(TMP, "wcache-r"), hrx)
case("a workflow missing at the other tag is reported as removed", any(x.endswith("- workflow workflows/b.md") for x in out))


def run3(name, version, bodies, cache):
    """surface() once per tool body, same venv and version: the outputs"""
    return [fresh.surface([("i", venv(name, version, {"loom-x": b}))], cache) for b in bodies]


# review r1 F1: a failed help run overwrote the cached entry; the next run called unchanged flags new and lost a removal
o1, o2, o3 = run3("tf", "9.9.0", ['echo "--a"; echo "--b"', "exit 1", 'echo "--a"'], os.path.join(TMP, "tcache"))
case("help failing once keeps the cached entry: a removal made meanwhile is reported, unchanged flags are not new",
     any("help failed" in x for x in o2) and any("loom-x: -" in x and "--b" in x for x in o3) and not any("+--a" in x for x in o3))
# the help-state transitions: failing only in the next venv; a tool whose help never worked, working
out = fresh.surface([("installed", venv("wa", "8.1.0", {"loom-y": 'echo "--y"'})), ("next", venv("wb", "8.2.0", {"loom-y": "exit 1"}))],
                    os.path.join(TMP, "fcache"))
case("help failing only in the next venv is named in the cross diff, and its flags are not listed as removed",
     any("loom-y: help now fails" in x for x in out) and not any("loom-y: -" in x for x in out))
o1, o2 = run3("wr", "8.3.0", ["exit 1", 'echo "--z"'], os.path.join(TMP, "wcache2"))[:2]
case("a tool whose help never worked is named when it works again, its flags not listed as new",
     any("loom-x: help works again" in x for x in o2) and not any("+--z" in x for x in o2))

# review r1 F3: a changed option value (0.3.0 dropped parameter_mode 'preload') must be named, not only counted
PRE = "echo \"# A parameter I/O mode of ['preload', 'file', 'direct'].\"; echo --parameter_mode"
POST = "echo \"# Parameter file access mode, either 'file' or 'direct'.\"; echo --parameter_mode"
o1, o2 = run3("pv", "7.1.0", [PRE, POST], os.path.join(TMP, "pcache"))[:2]
case("a changed help line is printed, not only counted", any("- # A parameter I/O mode of ['preload'" in x for x in o2)
     and any("+ # Parameter file access mode" in x for x in o2))
many = 'echo --m; for i in $(seq 1 12); do echo "line $i"; done'
o1, o2 = run3("pm", "7.2.0", ["echo --m", many], os.path.join(TMP, "mcache"))[:2]
case("changed lines are capped per tool with the remainder counted", sum("+ line " in x for x in o2) == getattr(fresh, "SHOW_LINES", 8)
     and any("(+4 more)" in x for x in o2))
gcache = os.path.join(TMP, "gcache")
fresh.surface([("i", venv("pg", "7.3.0", {"loom-p": PRE, "loom-q": PRE}))], gcache)
out = fresh.surface([("i", venv("pg", "7.3.0", {"loom-p": POST, "loom-q": POST}))], gcache)
case("the same line change in several tools is printed once, naming them", sum("['preload'" in x for x in out) == 1
     and any("loom-p, loom-q" in x for x in out))

# review r1 F2: argparse wraps to the parent's COLUMNS: a narrow terminal lost the subcommands and changed every line
APY = f"""exec {sys.executable} - "$@" <<'PY'
import argparse
ap = argparse.ArgumentParser(prog="loom-ap", description="a description long enough to wrap " * 4)
ap.add_argument("--a-rather-long-option-name", help="help text that wraps at a narrow width " * 3)
sp = ap.add_subparsers()
for n in ("show", "diff", "suggest"):
    sp.add_parser(n).add_argument("--opt-" + n)
ap.parse_args()
PY"""
ab = os.path.join(venv("ap", "6.1.0", {"loom-ap": APY, "loom-loc": 'echo "--lang-${LC_ALL:-${LANG:-none}}"'}), "bin")
saved = dict(os.environ)
snaps = []
for env in ({"COLUMNS": "40", "LANG": "de_DE.UTF-8"}, {"COLUMNS": "250", "LANG": "C"}):
    os.environ.update(env)
    snaps.append(fresh.snapshot(ab, "6.1.0"))
    os.environ.clear()
    os.environ.update(saved)
case("COLUMNS=40 and COLUMNS=250 (and two locales) in the parent give identical snapshots", snaps[0]["tools"] == snaps[1]["tools"])
case("argparse subcommands are found at any parent width", all({"loom-ap show", "loom-ap suggest"} <= set(x["tools"]) for x in snaps))
ob = os.path.join(venv("oc", "6.1.1", {"loom-o": 'echo "usage: loom-o [-h] [--lock-scope {assets,all}]"; echo "  --lock-scope"'}), "bin")
case("an option's choices {a,b} are not subcommands (loom-fetch --lock-scope {assets,all})",
     list(fresh.snapshot(ob, "6.1.1")["tools"]) == ["loom-o"])
case("a subcommand is reported as a subcommand, not a binary",
     any(x.endswith("+ subcommand loom-ap show") for x in fresh.surface_diff({"tools": {}, "workflows": None}, snaps[0])))

# review r1 F4: a hanging tool is bounded and its whole process group is killed (a forked grandchild survived)
pidf = os.path.join(TMP, "grandchild.pid")
hb = os.path.join(venv("hang", "6.2.0", {"loom-h": f'sleep 60 & echo $! >> {pidf}; wait'}), "bin")
fresh.HELP_TIMEOUT, saved_t = 1, getattr(fresh, "HELP_TIMEOUT", 15)
t0 = time.time()
snap = fresh.snapshot(hb, "6.2.0")
dt = time.time() - t0
fresh.HELP_TIMEOUT = saved_t
time.sleep(0.2)


def alive(pid):
    try:
        os.kill(pid, 0)
        return open(f"/proc/{pid}/stat").read().split(")")[1].split()[0] != "Z"
    except OSError:
        return False


left = [p for p in map(int, open(pidf).read().split()) if alive(p)]   # one grandchild per help mode tried
case("a hanging tool times out (bounded) and is reported", dt < 10 and "timed out" in (snap["tools"]["loom-h"]["error"] or ""))
case("the timed-out tool's process group is killed, grandchild included", not left)
for p in left:
    os.kill(p, 9)

# a tool that reads stdin must see EOF, not whatever the caller's stdin holds
r, w = os.pipe()
os.write(w, b"x\n")
keep = os.dup(0)
os.dup2(r, 0)
try:
    snap = fresh.snapshot(os.path.join(venv("si", "6.3.0", {"loom-s": 'if read -t 2 x; then echo "--read-stdin"; fi; echo --ok'}), "bin"), "6.3.0")
finally:
    os.dup2(keep, 0)
    os.close(r), os.close(w), os.close(keep)
case("a tool's stdin is empty (never the caller's)", "--read-stdin" not in snap["tools"]["loom-s"]["flags"] and "--ok" in snap["tools"]["loom-s"]["flags"])

# versions in numeric order: 0.10 after 0.9; a dev build before its release
vc = os.path.join(TMP, "vcache")
for v in ("0.9.0", "0.10.0"):
    fresh.surface([("i", venv("v" + v, v, {"loom-v": f'echo "--v{v}"'}))], vc)
out = fresh.surface([("i", venv("v0.11.0", "0.11.0", {"loom-v": "echo --v"}))], vc)
case("0.10.0 is newer than 0.9.0", any("vs previous release 0.10.0" in x for x in out))
dc = os.path.join(TMP, "dcache")
for v in ("0.2.0", "0.3.0.dev1"):
    fresh.surface([("i", venv("d" + v, v, {"loom-v": "echo --v"}))], dc)
out = fresh.surface([("i", venv("d0.3.0", "0.3.0", {"loom-v": "echo --v"}))], dc)
case("0.3.0.dev1 is older than 0.3.0", any("vs previous release 0.3.0.dev1" in x for x in out))
dc2 = os.path.join(TMP, "dcache2")
fresh.surface([("i", venv("e0.3.0", "0.3.0", {"loom-v": "echo --v"}))], dc2)
out = fresh.surface([("i", venv("e0.3.0rc1", "0.3.0rc1", {"loom-v": "echo --v"}))], dc2)
case("a release candidate is not diffed against its own later release", any("0.3.0rc1: first snapshot" in x for x in out))

# two fresh.py runs share one cache: a leftover or concurrent '<version>.json.tmp' must not block the write
ac = os.path.join(TMP, "acache")
os.makedirs(os.path.join(ac, "5.1.0.json.tmp"))
out = fresh.surface([("i", venv("at", "5.1.0", {"loom-a": "echo --a"}))], ac)
case("the cache write does not use a fixed temporary name", not any("check failed" in x for x in out)
     and os.path.exists(os.path.join(ac, "5.1.0.json")))

# the real installed and next venvs and the real hrx-system tags, when the machine names them
rn, rx = os.environ.get("VENV"), os.environ.get("VENV_NEXT")
hs = os.environ.get("HRX_SYSTEM")
if rn and rx and hs:
    out = fresh.surface([("installed", rn), ("next", rx)], os.path.join(TMP, "rcache"), hs)
    print("   " + "\n   ".join(x for x in out if "vs" in x or "first" in x))
    case("real VENV vs VENV_NEXT with HRX_SYSTEM: both snapshotted at their tags, cross diff printed, no help failures",
         sum("at tag v" in x for x in out) == 2 and any("next" in x and " vs installed " in x for x in out)
         and not any("help failed" in x or "check failed" in x for x in out))
else:
    print("SKIP real VENV vs VENV_NEXT case (set VENV, VENV_NEXT and HRX_SYSTEM)")

# the real red: 0.2.0's help with every --sanitizer line removed, then the real 0.2.0 -> --sanitizer reported as new
real = os.environ.get("VENV")
if real and fresh.installed_version(real):
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
print(f"{len(ran) - len(fails)}/{len(ran)} passed")
sys.exit(1 if fails else 0)
