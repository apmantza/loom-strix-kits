#!/usr/bin/env python3
"""Static lint of compiled gfx11 kernels: per kernel, the hot loop's instruction mix and the compiler's residency, flagged
against thresholds that each caught a real defect. A flag is a candidate to read, not a verdict: read the loop before acting.
Flags (the recurrence each one names); shares are of the hot loop's common path (nested loops excluded: rare masked paths):
  select    compares + selects >= 25 % of the common path: a vector.table.lookup scalarised into a select chain because the
            index range was hidden (E229/E230: IQ4 masked as i32 words, 960 of 1,544 loop instructions).
  repack    packed 16-bit shifts (v_pk_lshrrev/lshlrev_b16) >= 8 and byte extracts (v_bfe_*) >= 32 in the hot loop: byte
            vectors shifted by a runtime amount, every byte extracted and repacked (E233: q6_k / q3_k codes, 616 vs 290).
  scratch   any scratch load/store in the hot loop: a spill; on gfx11 spills share vmcnt with the loads (E230: 2,464 B).
  copies    a run of >= COPY_RUN register moves in the hot loop (gaps of up to 3 other instructions): the allocator copying a
            carried value or a branch's results (E179 / ROCm/hrx-system#3542: flash2's 45-move run before its back-edge).
  outlier   non-MMA instructions per MMA (or dot) > OUTLIER x the median of its peers (same family, place, MMA kind and MMA
            count, >= 3 kernels): a kernel doing much more per multiply than its siblings (E233: q6_k's matvec 616 vs q5_k's 261).
  regress:  (with --baseline) worse than the same kernel in another build: a new spill, the prefetch now drained, fewer
            waves per SIMD, or the hot loop / its longest move run >= 10 % longer (E226: Loom 0.3.0's vmcnt(0) + 32-move shift;
            E230: a rewrite's spill; E236: a variant's residency drop).
Reported, not flagged: non-MMA instructions per MMA, moves, VGPRs, waves/SIMD, VGPRs to free for the next residency tier
(Loom's summary report beside the code object, KERNEL.json), pf ('drain' when a vmcnt(0) sits between the loop's last load and
its first WMMA), and with --package the kernel's launch sites in the package. Calibrated on 225 served kernels (E237 / E241):
v_mul_lo_u32 (address math) and a drain alone are not defects.
Decoding goes through loop_census.disasm (ROCm 10.2 llvm-objdump; it refuses undecoded words).
usage: kernel_lint.py DIR_OR_HSACO ... [--baseline DIR] [--flagged] [--strict]
       kernel_lint.py --package PKG --out DIR [--baseline DIR] [--jobs N] [--flagged] [--strict]
         (--package compiles every gfx11 kernel root of PKG/program/*.loom with the machine's Loom into DIR first;
          --strict: exit 1 when anything is flagged)"""
import argparse
import collections
import concurrent.futures
import glob
import json
import os
import re
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop_census import disasm, loops, moves  # noqa: E402

FLAGS = ("select", "repack", "scratch", "copies", "outlier")
MM = ("v_wmma", "v_dot4", "v_dot2", "v_dot8")
COPY_RUN = 16
OUTLIER = 2.0
GROW = 1.10


def loom_compile():
    """the machine's loom-compile: VENV from the environment or the machine env file; exits when there is none"""
    venv = os.environ.get("VENV")
    if not venv:
        f = os.environ.get("LSK_ENV") or os.path.expanduser("~/.config/loom-strix-kits/env")
        m = re.search(r"(?m)^VENV=(\S+)", open(f).read()) if os.path.exists(f) else None
        venv = m.group(1) if m else None
    p = os.path.join(venv, "bin", "loom-compile") if venv else None
    if not (p and os.access(p, os.X_OK)):
        sys.exit("kernel_lint: no loom-compile (set VENV)")
    return p


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


def longest_move_run(body):
    """the most register moves in one run (moves at most 3 instructions apart), as loop_census.census groups them"""
    best = cur = 0
    last = None
    for j, x in enumerate(body):
        m = moves(x)
        if not m:
            continue
        cur = cur + m if last is not None and j - last <= 3 else m
        last = j
        best = max(best, cur)
    return best


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
    r.update(loop=len(full), common=len(b), wmma=mm, dot=dot, moves=sum(moves(x) for x in b), run=longest_move_run(b),
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
    if r["where"] == "loop" and r["run"] >= COPY_RUN:
        r["flags"].append("copies")
    r["pf"] = "-"
    loads = [j for j, x in enumerate(full) if re.match(r"(global|buffer)_load_", x)]
    w = [j for j, x in enumerate(full) if x.startswith("v_wmma")]
    if loads and w:
        before = [j for j in loads if j < w[0]]
        if before:
            r["pf"] = "drain" if any(re.match(r"s_waitcnt.*vmcnt\(0\)", full[j]) for j in range(before[-1], w[0])) else "kept"
    return r


def family(name):
    """the kernel's family from its name's leading words (g_mv, g_gemm, g_attention, g_head, v_gemm ...)"""
    m = re.match(r"([a-z0-9]+_[a-z]+)", name)
    return m.group(1) if m else name


def peer_key(r):
    """kernels comparable per multiply: same family, place (loop / kernel), MMA kind and MMA count. The family keeps attention
    from being judged against GEMMs with the same WMMA count (E241 calibration)."""
    return (family(r["kernel"]), r["where"], "wmma" if r["wmma"] else "dot", r["wmma"] or r["dot"])


def outliers(rows):
    """flag 'outlier' on rows whose non-MMA instructions per MMA exceed OUTLIER x their peers' median (groups of >= 3);
    returns {kernel: (value, peer median, peers)} for the flagged ones"""
    groups = collections.defaultdict(list)
    for r in rows:
        if r.get("other_per_op") is not None:
            groups[peer_key(r)].append(r)
    hit = {}
    for g in groups.values():
        if len(g) < 3:
            continue
        med = statistics.median(x["other_per_op"] for x in g)
        for r in g:
            if med and r["other_per_op"] > OUTLIER * med:
                r["flags"].append("outlier")
                hit[r["kernel"]] = (r["other_per_op"], med, len(g))
    return hit


def regressions(r, base):
    """what got worse in row r against the same kernel's row in another build"""
    out = []
    if r["scratch"] and not base["scratch"]:
        out.append("spill")
    if r["pf"] == "drain" and base["pf"] == "kept":
        out.append("drain")
    if r["waves"] is not None and base["waves"] is not None and r["waves"] < base["waves"]:
        out.append(f"waves{base['waves']}->{r['waves']}")
    if r["loop"] >= GROW * base["loop"] and r["loop"] - base["loop"] >= 8:
        out.append(f"loop{base['loop']}->{r['loop']}")
    if r["run"] >= GROW * base["run"] and r["run"] - base["run"] >= 4:
        out.append(f"run{base['run']}->{r['run']}")
    return out


def launch_sites(texts):
    """kernel -> launch sites in a package's program texts: direct `kernel.launch @K[` outside template bodies, plus every
    `template.apply<@selN_K>` (the dual-target selectors launch K or its gfx12 twin)"""
    sites = collections.Counter()
    for t in texts:
        in_template = False
        for line in t.split("\n"):
            if line.startswith("template.def"):
                in_template = True
            elif in_template and line.startswith("}"):
                in_template = False
            m = re.search(r"kernel\.launch @([\w]+)[\[(]", line)
            if m and not in_template:
                sites[m.group(1)] += 1
            for m in re.finditer(r"template\.apply<@sel\d+_([\w]+)>", line):
                sites[m.group(1)] += 1
    return sites


def compile_package(pkg, out, jobs):
    """every gfx11 kernel root of PKG/program/*.loom compiled into OUT (code object + summary report); returns the program
    texts and the roots that failed to compile standalone (named, never hidden)"""
    lc = loom_compile()
    os.makedirs(out, exist_ok=True)
    texts, work = [], []
    for prog in sorted(glob.glob(os.path.join(pkg, "program", "*.loom"))):
        t = open(prog).read()
        texts.append(t)
        for k in sorted(set(re.findall(r'(?m)^kernel\.def (?:target\(@\w+\) )?export\("(\w+)"\)', t))):
            if not k.endswith("_rdna4"):
                work.append((prog, k))
    def one(pk):
        prog, k = pk
        r = subprocess.run([lc, prog, "--target=amdgpu:gfx1151", f"--root={k}", f"--output={out}/{k}.hsaco",
                            "--compile-report=summary", f"--compile-report-output={out}/{k}.json"], capture_output=True, text=True)
        return k, r.returncode, r.stderr
    failed = []
    with concurrent.futures.ThreadPoolExecutor(jobs) as ex:
        for k, rc, err in ex.map(one, work):
            if rc:
                failed.append((k, (err.strip().split("\n") or [""])[0][:160]))
    return texts, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--package", help="a package directory: compile its kernel roots into --out first")
    ap.add_argument("--out", help="with --package: where the code objects go")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--baseline", help="a directory of the same kernels from another build: flag what got worse")
    ap.add_argument("--flagged", action="store_true", help="print flagged kernels only")
    ap.add_argument("--strict", action="store_true", help="exit 1 when any kernel is flagged")
    a = ap.parse_args()
    sites, failed = None, []
    if a.package:
        if not a.out:
            sys.exit("kernel_lint: --package needs --out")
        texts, failed = compile_package(a.package, a.out, a.jobs)
        sites = launch_sites(texts)
        a.paths = [a.out] + a.paths
    files = []
    for p in a.paths:
        files += sorted(glob.glob(os.path.join(p, "*.hsaco"))) if os.path.isdir(p) else [p]
    if not files:
        sys.exit("kernel_lint: no code objects")
    rows = [lint(f) for f in files]
    outl = outliers(rows)
    if a.baseline:
        for r in rows:
            b = os.path.join(a.baseline, r["kernel"] + ".hsaco")
            if os.path.exists(b):
                r["flags"] += ["regress:" + x for x in regressions(r, lint(b))]
    rows.sort(key=lambda r: (-len(r["flags"]), -(sites or {}).get(r["kernel"], 0), r["kernel"]))
    print(f"{'kernel':42s} {'where':>6s} {'loop':>5s} {'comm':>5s} {'mm':>4s} {'other/op':>8s} {'sel':>4s} {'pk':>3s} {'bfe':>4s} {'scr':>4s} {'mov':>4s} "
          f"{'run':>4s} {'vgpr':>4s} {'w/S':>3s} {'next':>4s} {'pf':>5s} {'sites':>5s}  flags")
    for r in rows:
        if a.flagged and not r["flags"]:
            continue
        f = lambda v: "-" if v is None else str(v)
        s = "-" if sites is None else str(sites.get(r["kernel"], 0))
        print(f"{r['kernel']:42s} {r['where']:>6s} {r['loop']:5d} {r['common']:5d} {r['wmma'] or r['dot']:4d} {f(r['other_per_op']):>8s} {r['select']:4d} {r['pk']:3d} "
              f"{r['bfe']:4d} {r['scratch']:4d} {r['moves']:4d} {r['run']:4d} {f(r['vgpr']):>4s} {f(r['waves']):>3s} {f(r['to_next']):>4s} {r['pf']:>5s} {s:>5s}  "
              f"{' '.join(r['flags'])}")
    for k, (v, med, n) in sorted(outl.items()):
        print(f"outlier {k}: {v} non-MMA instructions per MMA vs a median of {med} over {n} peers")
    if sites is not None:
        unl = sorted(r["kernel"] for r in rows if not sites.get(r["kernel"]))
        print(f"not launched in the package ({len(unl)}): {' '.join(unl)}")
    for k, why in failed:
        print(f"not compiled standalone: {k}: {why}")
    flagged = sum(1 for r in rows if r["flags"])
    print(f"kernel_lint: {len(rows)} kernels, {flagged} flagged" + (f", {len(failed)} roots did not compile standalone" if failed else ""))
    sys.exit(1 if a.strict and flagged else 0)


if __name__ == "__main__":
    main()
