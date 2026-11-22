#!/usr/bin/env python3
"""Same kernels, two Loom compilers (HRX-like E177: 0.2.0 vs hrx-system main, compile-only). For each root of a program:
VGPRs, LDS, occupancy, and in the hot loop (the backward branch whose body holds the most WMMAs) the instruction, v_mov and
WMMA counts, s_waitcnt vmcnt count, and each global/buffer load's cover (instructions until the vmcnt wait that retires it;
E109 / scripts/load_cover.py). Fails closed: a root that does not compile under either compiler is reported and exits 1.
usage: compiler_ab.py PROGRAM.loom OUT_DIR --a NAME=LOOM_COMPILE --b NAME=LOOM_COMPILE ROOT [ROOT ...]"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop_census import cover, disasm, hot_loop, moves  # noqa: E402  (one census for every tool, retro 11-22)


def compile_root(lc, prog, root, out):
    hs = f"{out}/{root}.hsaco"
    r = subprocess.run([lc, prog, "--target=amdgpu:gfx1151", f"--root={root}", f"--output={hs}",
                        "--compile-report=summary", f"--compile-report-output={out}/{root}.json"], capture_output=True, text=True)
    open(f"{out}/{root}.cc.log", "w").write(r.stdout + r.stderr)
    return hs if r.returncode == 0 and os.path.exists(hs) else None


def notes(hs):
    t = subprocess.run(["llvm-readelf-21", "--notes", hs], capture_output=True, text=True).stdout
    g = lambda k: (re.search(r"\." + k + r":\s+(\d+)", t) or [None, None])[1]
    return {"vgpr": g("vgpr_count"), "sgpr": g("sgpr_count"), "lds": g("group_segment_fixed_size"), "scratch": g("private_segment_fixed_size")}


def occupancy(js):
    try:
        r = json.load(open(js))["entries"]["rows"][0]["target_resources"]
        return f'{r.get("resident_subgroups_per_simd")}/SIMD ({r.get("limiting_resource")})'
    except Exception:
        return "?"


def report(name, lc, prog, roots, out):
    os.makedirs(out, exist_ok=True)
    rows, bad = {}, []
    for k in roots:
        hs = compile_root(lc, prog, k, out)
        if not hs:
            bad.append(k)
            continue
        ins, addr = disasm(hs)
        r = {**notes(hs), "occ": occupancy(f"{out}/{k}.json"), "instr": len(ins)}
        hl = hot_loop(ins, addr)
        if hl:
            lo, hi, w = hl
            body = ins[lo:hi + 1]
            cv, pend = cover(ins, lo, hi)
            r.update(loop=hi - lo + 1, wmma=w, vmov=sum(moves(x) for x in body),
                     vmcnt_waits=sum(1 for x in body if x.startswith("s_waitcnt") and "vmcnt" in x),
                     loads=sum(1 for x in body if re.match(r"(global|buffer)_load_", x)),
                     cover_min=min(cv) if cv else None, cover_p50=statistics.median(cv) if cv else None, in_flight_at_backedge=pend)
        rows[k] = r
    return rows, bad


ap = argparse.ArgumentParser()
ap.add_argument("program")
ap.add_argument("out")
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("roots", nargs="+")
a = ap.parse_args()
res, failed = {}, []
for spec in (a.a, a.b):
    n, lc = spec.split("=", 1)
    res[n], bad = report(n, lc, a.program, a.roots, os.path.join(a.out, n))
    failed += [f"{n}:{k}" for k in bad]
cols = ["vgpr", "lds", "scratch", "occ", "instr", "loop", "wmma", "vmov", "loads", "vmcnt_waits", "cover_min", "cover_p50", "in_flight_at_backedge"]
na, nb = [s.split("=", 1)[0] for s in (a.a, a.b)]
print("| kernel | compiler | " + " | ".join(cols) + " |")
print("|" + "---|" * (len(cols) + 2))
for k in a.roots:
    for n in (na, nb):
        r = res[n].get(k)
        print(f"| {k} | {n} | " + (" | ".join(str(r.get(c, "-")) for c in cols) if r else "COMPILE FAILED") + " |")
json.dump(res, open(os.path.join(a.out, "compiler_ab.json"), "w"), indent=1)
if failed:
    sys.exit("compiler_ab: did not compile: " + ", ".join(failed))
