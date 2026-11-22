#!/usr/bin/env python3
"""Loop census of a gfx1151 code object (HRX-like retro 11-22: E168 named a mechanism from a hand-cut line range; E179 rewrote
this census six times by hand). Every loop is a backward branch [target, branch]; for each loop holding WMMAs:
length, WMMAs, v_mov (v_dual_mov halves counted), scratch ops, vmcnt waits, and the runs of >= RUN consecutive moves with their
offset in the loop and their first/last instruction, so a reader can place them (before the back-edge, at a branch join, in a
rare path). Also the shared helpers of kernel_lint.py, load_events.py and load_cover.py (disasm, loops, hot_loop, cover).
usage: loop_census.py HSACO [HSACO ...] [--run 4] [--all]   (--all: loops without WMMAs too)"""
import argparse
import os
import re
import subprocess


def objdump():
    """the disassembler: $OBJDUMP, else the ROCm SDK's (ROCM_SDK from the environment or the machine env file; ROCm 10.2's
    LLVM decodes every gfx1151 encoding Loom emits), else llvm-objdump-21 (disasm refuses its output where it cannot)"""
    if os.environ.get("OBJDUMP"):
        return os.environ["OBJDUMP"]
    sdk = os.environ.get("ROCM_SDK")
    if not sdk:
        f = os.environ.get("LSK_ENV") or os.path.expanduser("~/.config/loom-strix-kits/env")
        m = re.search(r"(?m)^ROCM_SDK=(\S+)", open(f).read()) if os.path.exists(f) else None
        sdk = m.group(1) if m else None
    p = os.path.join(sdk, "lib", "llvm", "bin", "llvm-objdump") if sdk else None
    return p if p and os.access(p, os.X_OK) else "llvm-objdump-21"


def disasm(hs, mcpu="gfx1151"):
    """(instructions, addresses) of a code object, instruction text without the encoding comment. Refuses a listing with
    undecoded words (`.long`): llvm-objdump-21 cannot decode some gfx1151 VOP3 forms Loom emits (v_cmp_*_e64 with an SGPR
    result), prints the first dword as `.long` and the second as a phantom instruction (mostly v_cndmask), so every count
    and every branch target behind it is wrong (E230: E229's '972 v_cndmask' was 480 v_cmp_eq_i32_e64 + 480 v_cndmask)."""
    tool = objdump()
    d = subprocess.run([tool, "-d", f"--mcpu={mcpu}", hs], capture_output=True, text=True, check=True).stdout
    bad = [l.strip() for l in d.split("\n") if re.match(r"^\s+\.(long|short|byte)\b", l)]
    if bad:
        raise SystemExit(f"loop_census: {tool} left {len(bad)} undecoded words in {hs} (first: {bad[0]}); "
                         "use a decoder that knows the target (OBJDUMP=..., or ROCM_SDK = a ROCm 10.2 _rocm_sdk_core)")
    ins, addr = [], []
    for line in d.split("\n"):
        m = re.match(r"^\s+([a-z]\S*)(.*?)//\s*([0-9A-F]+):", line)
        if m:
            ins.append((m.group(1) + " " + m.group(2).strip()).strip())
            addr.append(int(m.group(3), 16))
    return ins, addr


def loops(ins, addr):
    """every backward branch as (target index, branch index)"""
    a2i = {a: i for i, a in enumerate(addr)}
    out = []
    for i, l in enumerate(ins):
        m = re.match(r"s_(cbranch_\w+|branch)\s+(-?\d+)", l)
        if not m:
            continue
        off = int(m.group(2))
        if off > 32767:
            off -= 65536
        t = a2i.get(addr[i] + 4 + 4 * off)
        if t is not None and t < i:
            out.append((t, i))
    return out


def hot_loop(ins, addr):
    """the backward branch [target, branch] whose body holds the most WMMAs, as (target, branch, wmma count), or None"""
    best = None
    for t, i in loops(ins, addr):
        w = sum(1 for x in ins[t:i + 1] if x.startswith("v_wmma"))
        if best is None or w > best[2]:
            best = (t, i, w)
    return best


def cover(ins, lo, hi):
    """per global/buffer load in [lo, hi]: instructions until the vmcnt wait that retires it; and the loads still pending at hi"""
    q, out = [], []
    for i in range(lo, hi + 1):
        l = ins[i]
        if re.match(r"(global|buffer)_load_", l):
            q.append(i)
        w = re.search(r"vmcnt\((\d+)\)", l) if l.startswith("s_waitcnt") else None
        if w:
            while len(q) > int(w.group(1)):
                out.append(i - q.pop(0))
    return out, len(q)


def moves(x):
    """register moves an instruction performs: v_mov 1, each v_dual_mov half 1"""
    return (1 if x.startswith("v_mov") else 0) + x.count("v_dual_mov")


def census(ins, t, e, run=4):
    body = ins[t:e + 1]
    mv = [j for j in range(t, e + 1) if moves(ins[j])]
    runs, cur = [], []
    for j in mv:
        if cur and j - cur[-1] > 3:
            runs.append(cur)
            cur = []
        cur.append(j)
    if cur:
        runs.append(cur)
    return {"loop": [t, e], "len": len(body), "wmma": sum(1 for x in body if x.startswith("v_wmma")),
            "moves": sum(moves(x) for x in body), "scratch": sum(1 for x in body if x.startswith("scratch_")),
            "vmcnt_waits": sum(1 for x in body if x.startswith("s_waitcnt") and "vmcnt" in x),
            "runs": [{"at": r[0] - t, "to": r[-1] - t, "n": sum(moves(ins[j]) for j in r), "first": ins[r[0]], "last": ins[r[-1]]}
                     for r in runs if len(r) >= run]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hsaco", nargs="+")
    ap.add_argument("--run", type=int, default=4)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    for hs in a.hsaco:
        ins, addr = disasm(hs)
        print(f"{hs}: {len(ins)} instructions")
        for t, e in sorted(loops(ins, addr)):
            c = census(ins, t, e, a.run)
            if not c["wmma"] and not a.all:
                continue
            print(f"  loop [{t},{e}] len {c['len']} wmma {c['wmma']} moves {c['moves']} scratch {c['scratch']} vmcnt-waits {c['vmcnt_waits']}")
            for r in c["runs"]:
                print(f"    moves at +{r['at']}..+{r['to']} of {c['len']} ({r['n']}): {r['first']} ... {r['last']}")


if __name__ == "__main__":
    main()
