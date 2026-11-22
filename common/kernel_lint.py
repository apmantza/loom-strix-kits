#!/usr/bin/env python3
"""Static lint of compiled gfx11 kernels: per kernel, the hot loop's instruction mix and the compiler's residency, flagged
against thresholds that each caught a real defect. A flag is a candidate to read, not a verdict: read the loop before acting.
Flags (the recurrence each one names); shares are of the hot loop's common path (nested loops excluded: rare masked paths):
  select    compares + selects >= 25 % of the common path: a vector.table.lookup scalarised into a select chain because the
            index range was hidden (E229/E230: IQ4 masked as i32 words, 960 of 1,544 loop instructions).
  repack    packed 16-bit shifts (v_pk_lshrrev/lshlrev_b16) >= 8 and byte extracts (v_bfe_*) >= 32 in the hot loop: byte
            vectors shifted by a runtime amount, every byte extracted and repacked (E233: q6_k / q3_k codes, 616 vs 290).
  scratch   any scratch load/store in the hot loop: a spill; on gfx11 spills share vmcnt with the loads (E230: 2,464 B).
Reported, not flagged: non-WMMA instructions per WMMA (or per dot4), moves, VGPRs, waves/SIMD and the VGPRs to free for the
next residency tier (from Loom's summary report beside the code object, KERNEL.json, when present), and pf: 'drain' when a
vmcnt(0) sits between the loop's last load and its first WMMA. Not flags, calibrated on the 225 served kernels (E237):
v_mul_lo_u32 is in nearly every loop's address math; a drain is the normal shape of a kernel that does not prefetch across
iterations, and a defect only relative to another build of the same kernel (compiler_ab.py's question, E226).
Decoding goes through loop_census.disasm (ROCm 10.2 llvm-objdump; it refuses undecoded words).
usage: kernel_lint.py DIR_OR_HSACO ... [--flagged] [--strict]   (--strict: exit 1 when anything is flagged)"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop_census import disasm, loops, moves  # noqa: E402

FLAGS = ("select", "repack", "scratch")


MM = ("v_wmma", "v_dot4", "v_dot2", "v_dot8")


def common_path(ins, L, t, e):
    """indices of loop [t, e] outside every loop nested in it"""
    inner = set()
    for a, z in L:
        if t <= a and z <= e and (a, z) != (t, e):
            inner.update(range(a, z + 1))
    return [j for j in range(t, e + 1) if j not in inner]


def hot(ins, addr):
    """the loop whose own common path holds the most WMMA / dot instructions (ties: the longest), else None. Counting the
    common path, not the whole range, picks a nested inner loop over the outer shell around it (E237: the 16-token swiglu
    matvecs keep their dot4s in an inner loop)."""
    L = loops(ins, addr)
    best = None
    for t, e in L:
        mm = sum(1 for j in common_path(ins, L, t, e) if ins[j].startswith(MM))
        key = (mm, e - t)
        if best is None or key > best[0]:
            best = (key, t, e)
    return (best[1], best[2], L) if best else None


def residency(js):
    """(VGPRs, waves/SIMD, VGPRs to free for the next better tier) from a Loom summary report, or Nones"""
    if not os.path.exists(js):
        return None, None, None
    tr = json.load(open(js)).get("target_resources") or {}
    res = tr.get("residency") or {}
    lim = res.get("unique_limiting_resource") or {}
    return (((tr.get("vector") or {}).get("final") or {}).get("register_count"), res.get("current_tier"),
            lim.get("reduction_units_to_next_better_tier"))


def lint(path):
    ins, addr = disasm(path)
    return lint_listing(os.path.basename(path)[:-6], ins, addr, residency(path[:-6] + ".json"))


def lint_listing(name, ins, addr, res=(None, None, None)):
    """the lint of one kernel's decoded listing (instructions, byte addresses), as lint() reports it"""
    r = {"kernel": name, "instr": len(ins), "flags": []}
    r["vgpr"], r["waves"], r["to_next"] = res
    h = hot(ins, addr)
    if h is None or not any(ins[j].startswith(MM) for j in common_path(ins, h[2], h[0], h[1])):
        b = full = ins                                                # no multiply loop: the whole kernel (e.g. a dequant pass)
        r["where"] = "kernel"
    else:
        t, e, L = h
        b = [ins[j] for j in common_path(ins, L, t, e)]              # the hot loop's common path
        full = ins[t:e + 1]
        r["where"] = "loop"
    n = lambda *p: sum(1 for x in b if x.startswith(p))
    mm = n("v_wmma"); dot = n("v_dot4", "v_dot2", "v_dot8")
    r.update(loop=len(full), common=len(b), wmma=mm, dot=dot, moves=sum(moves(x) for x in b),
             select=n("v_cmp", "v_cndmask"), pk=n("v_pk_lshrrev_b16", "v_pk_lshlrev_b16"), bfe=n("v_bfe_"),
             scratch=sum(1 for x in full if "scratch_" in x))
    per = mm or dot
    r["other_per_op"] = round((len(b) - per) / per, 1) if per else None   # common path
    if r["select"] >= 0.25 * len(b):
        r["flags"].append("select")
    if r["pk"] >= 8 and r["bfe"] >= 32:
        r["flags"].append("repack")
    if r["scratch"]:
        r["flags"].append("scratch")
    r["pf"] = "-"
    loads = [j for j, x in enumerate(full) if re.match(r"(global|buffer)_load_", x)]
    w = [j for j, x in enumerate(full) if x.startswith("v_wmma")]
    if loads and w:
        before = [j for j in loads if j < w[0]]
        if before:
            r["pf"] = "drain" if any(re.match(r"s_waitcnt.*vmcnt\(0\)", full[j]) for j in range(before[-1], w[0])) else "kept"
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--flagged", action="store_true", help="print flagged kernels only")
    ap.add_argument("--strict", action="store_true", help="exit 1 when any kernel is flagged")
    a = ap.parse_args()
    files = []
    for p in a.paths:
        files += sorted(glob.glob(os.path.join(p, "*.hsaco"))) if os.path.isdir(p) else [p]
    if not files:
        sys.exit("kernel_lint: no code objects")
    rows = [lint(f) for f in files]
    rows.sort(key=lambda r: (-len(r["flags"]), r["kernel"]))
    print(f"{'kernel':42s} {'where':>6s} {'loop':>5s} {'comm':>5s} {'mm':>4s} {'other/op':>8s} {'sel':>4s} {'pk':>3s} {'bfe':>4s} {'scr':>4s} {'mov':>4s} "
          f"{'vgpr':>4s} {'w/S':>3s} {'next':>4s} {'pf':>5s}  flags")
    for r in rows:
        if a.flagged and not r["flags"]:
            continue
        f = lambda v: "-" if v is None else str(v)
        print(f"{r['kernel']:42s} {r['where']:>6s} {r['loop']:5d} {r['common']:5d} {r['wmma'] or r['dot']:4d} {f(r['other_per_op']):>8s} {r['select']:4d} {r['pk']:3d} "
              f"{r['bfe']:4d} {r['scratch']:4d} {r['moves']:4d} {f(r['vgpr']):>4s} {f(r['waves']):>3s} {f(r['to_next']):>4s} {r['pf']:>5s}  "
              f"{' '.join(r['flags'])}")
    flagged = sum(1 for r in rows if r["flags"])
    print(f"kernel_lint: {len(rows)} kernels, {flagged} flagged")
    sys.exit(1 if a.strict and flagged else 0)


if __name__ == "__main__":
    main()
