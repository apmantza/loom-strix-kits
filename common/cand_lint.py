#!/usr/bin/env python3
"""A candidate package's post-build lint against a base package (common/build.sh runs `check` when LINT_BASE is set).
Recurrence (HRX-like E520): FLASH2_STAGE_LIVE combined with FLASH2_FLATADDR put an s_waitcnt vmcnt(0) inside a scf.if that
retired the V-tile loads issued before it (flash2's V cover 166-203 -> 23-25 instructions); each switch alone kept the
prefetch, the binary gate passed, and it took three hand-built pair packages to find.

check BASE_PKG CAND_PKG --out DIR: common/binary_gate.py between the packages (its rows in DIR/bgate.json, the code objects in
  DIR/base and DIR/cand), then common/kernel_lint.py's lint of every kernel the gate reports CHANGED, judged against the same
  kernel of the base by kernel_lint.regressions. Fails (exit 1) on:
    - a FAIL_ON regression: 'drain', kernel_lint's prefetch check ('pf' kept in the base, drain in the candidate: a vmcnt(0)
      between the hot loop's last load before its first WMMA and that WMMA, i.e. the loads meant to be in flight across the
      multiplies are waited first);
    - a changed set the gate could not establish (a kernel the gate reports ERROR) or a CHANGED kernel whose code objects
      cannot be linted: fail closed;
    - with --allow: binary_gate's own failures (the untouched-kernel verdict, which needs the switch's kernel patterns).
      Without --allow its verdict is printed, not judged; its ERROR rows still fail (above).
  The minimum load cover (kernel_lint 'cov') and the loads in flight at the back-edge ('fly') are printed per changed kernel,
  base -> candidate, but do not decide: on E520's packages they cannot separate the clash from its fix (flash2_partial's cover
  8 in the clash, 10 in the fix pkg-cmb2, 86 in the base: E521's accepted residue), while 'drain' does (clash: all four flash2
  kernels drain; fix and every viable pair: kept). The other regressions (waves, loop or move-run growth) are printed as notes:
  E520's viable pairs grow their move runs.
  NEW kernels have no base: printed with their own lint, not judged.

pairs KIT BASE_PKG PREFIX SWITCH=V ...: build every single switch, every pair and (for three or more) the whole set on top of the
  kit's recipe with common/build.sh (packages WORK/PREFIX-<ids>, about 16 s and 0.9 GB each), each checked as above against
  BASE_PKG, and name the smallest switch sets that fail while none of their subsets do (where the collapse first appears).
  Exit 1 when any set fails, 0 when none does, 2 when a build fails.

usage: cand_lint.py check BASE_PKG CAND_PKG --out DIR [--allow REGEX ...] [--allow-missing REGEX ...] [--jobs N]
       cand_lint.py pairs KIT BASE_PKG PREFIX SWITCH=V SWITCH=V ... [--allow REGEX ...] [--jobs N]"""
import argparse
import itertools
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FAIL_ON = ("drain",)          # kernel_lint.regressions kinds that fail a candidate (the prefetch check)
CHANGED = ("CHANGED",)        # binary_gate verdicts whose kernels are linted against the base


def judge(rows, lint_base, lint_cand):
    """(report lines, failures) of the lint of the gate's changed kernels. ROWS: binary_gate's verdict rows; LINT_BASE / LINT_CAND:
    kernel -> kernel_lint row (lint() of that side's code object), absent when it could not be linted. Pure."""
    import kernel_lint as kl
    lines, fails = [], []
    f = lambda v: "-" if v is None else str(v)
    for r in rows:
        k, v = r["kernel"], r["verdict"]
        if v == "ERROR":
            fails.append(f"{k}: binary_gate ERROR ({'; '.join(r.get('fail') or []) or 'no reason'}): the changed set is unknown")
            continue
        if v == "NEW":
            c = lint_cand.get(k)
            lines.append(f"  NEW      {k}: pf {c['pf'] if c else '?'} cov {f(c and c['cover_min'])} fly {f(c and c['in_flight'])} "
                         "(no base kernel: not judged)")
            continue
        if v not in CHANGED:
            continue
        b, c = lint_base.get(k), lint_cand.get(k)
        if b is None or c is None:
            fails.append(f"{k}: CHANGED but its {'base' if b is None else 'candidate'} code object was not linted")
            continue
        reg = kl.regressions(c, b)
        hit = [x for x in reg if x in FAIL_ON]
        lines.append(f"  {'FAIL' if hit else 'ok':8s} {k}: pf {b['pf']}->{c['pf']} cov {f(b['cover_min'])}->{f(c['cover_min'])} "
                     f"fly {b['in_flight']}->{c['in_flight']}" + (f"  regress {' '.join(reg)}" if reg else ""))
        if hit:
            fails.append(f"{k}: {' '.join(hit)} (prefetch: pf {b['pf']} -> {c['pf']}, min load cover {f(b['cover_min'])} -> "
                         f"{f(c['cover_min'])})")
    return lines, fails


def lint_dir(d, kernels):
    """kernel -> kernel_lint row for each of KERNELS with a code object in D"""
    import kernel_lint as kl
    out = {}
    for k in kernels:
        p = os.path.join(d, k + ".hsaco")
        if os.path.exists(p):
            out[k] = kl.lint(p)
    return out


def check(base_pkg, cand_pkg, out, allow=(), allow_missing=(), jobs=2, lc=None):
    """run the gate and the lint; writes OUT/bgate.json and OUT/verdict.json; returns (lines, failures)"""
    import binary_gate as bg
    os.makedirs(out, exist_ok=True)
    rows, bfails, n = bg.gate(base_pkg, cand_pkg, list(allow), out, jobs, False, lc, list(allow_missing))
    json.dump(rows, open(os.path.join(out, "bgate.json"), "w"), indent=1)
    lines = [f"binary_gate {base_pkg} -> {cand_pkg}: " + ", ".join(
        f"{v} {sum(1 for r in rows if r['verdict'] == v)}" for v in sorted({r["verdict"] for r in rows})) + f"; {n} compiles"]
    lines += [f"  {r['verdict']:9s} {r['kernel']}" for r in rows if r["verdict"] not in ("same", "text-only", "CHANGED")]
    ks = [r["kernel"] for r in rows if r["verdict"] in CHANGED + ("NEW",)]
    lb, lcand = lint_dir(os.path.join(out, "base"), ks), lint_dir(os.path.join(out, "cand"), ks)
    ll, fails = judge(rows, lb, lcand)
    lines += ll
    if allow:
        fails += [f"binary_gate: {x}" for x in bfails]
    elif bfails:
        lines.append(f"  binary_gate: {len(bfails)} untouched-kernel failures not judged (no --allow / LINT_ALLOW): "
                     + "; ".join(bfails[:3]) + (" ..." if len(bfails) > 3 else ""))
    json.dump({"base": base_pkg, "cand": cand_pkg, "fails": fails, "lines": lines,
               "changed": sorted(r["kernel"] for r in rows if r["verdict"] in ("CHANGED", "NEW"))},
              open(os.path.join(out, "verdict.json"), "w"), indent=1)
    return lines, fails


def switch_sets(sw):
    """singles, pairs, and the whole set when it has three or more switches (subsets in increasing size)"""
    sets = [(s,) for s in sw] + list(itertools.combinations(sw, 2))
    return sets + ([tuple(sw)] if len(sw) > 2 else [])


def first_failures(results):
    """the failing switch sets none of whose proper subsets (among those built) fails. RESULTS: {tuple of switches: failed}"""
    return [s for s, bad in results.items() if bad and not any(results.get(t) for t in results if set(t) < set(s))]


def pairs(kit, base, prefix, sw, allow=(), jobs=2):
    """build and check every switch set; returns the exit code"""
    names = {s: f"s{i + 1}" for i, s in enumerate(sw)}
    results, report = {}, []
    for s in switch_sets(sw):
        pkg = prefix + "-" + "".join(names[x] for x in s)
        env = dict(os.environ, EXTRA=" ".join(s), LINT_BASE=base, LINT_ALLOW=" ".join(allow), LINT_JOBS=str(jobs))
        r = subprocess.run(["bash", os.path.join(HERE, "build.sh"), kit, pkg], env=env, capture_output=True, text=True)
        print(r.stdout.strip())
        if r.returncode not in (0, 3):
            print(f"cand_lint pairs: build of {pkg} failed (exit {r.returncode}): {r.stderr.strip()[-300:]}")
            return 2
        vf = os.path.join(os.environ.get("WORK") or kit_work(kit), pkg + ".lint", "verdict.json")
        v = json.load(open(vf))
        results[s] = bool(v["fails"]) or r.returncode == 3
        report.append((s, pkg, v))
    print("\nswitch sets (" + ", ".join(f"{n} = {s}" for s, n in names.items()) + f"), against {base}:")
    for s, pkg, v in report:
        print(f"{'FAIL' if results[s] else 'pass':4s} {' x '.join(x.split('=')[0] for x in s)}  ({pkg})")
        for line in v["lines"][1:]:
            print("    " + line.strip())
    first = first_failures(results)
    for s in first:
        print("COLLAPSE FIRST APPEARS: " + " x ".join(x.split("=")[0] for x in s))
    print("CAND_LINT PAIRS: " + ("FAIL" if first else "PASS"))
    return 1 if first else 0


def kit_work(kit):
    """WORK as common/env.sh lsk_load sets it for KIT"""
    r = subprocess.run(["bash", "-c", f'. "{HERE}/env.sh"; lsk_load "$1" >/dev/null && echo "$WORK"', "_", kit],
                       capture_output=True, text=True)
    if r.returncode or not r.stdout.strip():
        sys.exit(f"cand_lint: kit {kit}: lsk_load failed: {r.stderr.strip()[-200:]}")
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("base")
    c.add_argument("cand")
    c.add_argument("--out", required=True)
    p = sub.add_parser("pairs")
    p.add_argument("kit")
    p.add_argument("base")
    p.add_argument("prefix")
    p.add_argument("switches", nargs="+")
    for x in (c, p):
        x.add_argument("--allow", action="append", default=[])
        x.add_argument("--jobs", type=int, default=2)
    c.add_argument("--allow-missing", action="append", default=[])
    a = ap.parse_args()
    for x in a.allow + getattr(a, "allow_missing", []):
        try:
            re.compile(x)
        except re.error as e:
            ap.error(f"--allow {x!r}: {e}")
    if a.cmd == "pairs":
        bad = [s for s in a.switches if not re.match(r"[A-Za-z_]\w*=\S*$", s)]
        if bad or len(a.switches) < 2:
            ap.error(f"pairs: two or more SWITCH=V (got {a.switches})")
        sys.exit(pairs(a.kit, a.base, a.prefix, a.switches, a.allow, a.jobs))
    lines, fails = check(a.base, a.cand, a.out, a.allow, a.allow_missing, a.jobs)
    print("\n".join(lines))
    for x in fails:
        print("  FAIL " + x)
    print("CAND_LINT: " + ("FAIL" if fails else "PASS"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
