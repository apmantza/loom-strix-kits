#!/usr/bin/env python3
"""A candidate package's post-build lint against a base package (common/build.sh runs `check` when LINT_BASE is set).
Recurrence (HRX-like E520): FLASH2_STAGE_LIVE combined with FLASH2_FLATADDR put an s_waitcnt vmcnt(0) inside a scf.if that
retired the V-tile loads issued before it (flash2's V cover 166-203 -> 23-25 instructions); each switch alone kept the
prefetch, the binary gate passed, and it took three hand-built pair packages to find.

check BASE_PKG CAND_PKG --out DIR: common/binary_gate.py between the packages (its rows in DIR/bgate.json, the code objects in
  DIR/base and DIR/cand), then common/kernel_lint.py's lint of every kernel the gate reports CHANGED, judged against the same
  kernel of the base. A CHANGED kernel fails on any loss (losses()):
    - the base's prefetch is 'kept' (kernel_lint pf) and the candidate's is anything else: 'drain' (E520: a vmcnt(0) between the
      hot loop's last load and its first WMMA) or '-' (the load sunk past every MMA). Gate on the known-good form, not on one
      bad spelling (review r1 F2);
    - more exposed loads: per hot-loop load, the MMAs issued between it and the wait that retires it (prefetch_covers; the loop
      walked three times so the next tile's load is followed across the back-edge); a load is exposed when it covers fewer than
      FULL = min(the base's best cover, the candidate loop's MMAs). Catches the wait moved one WMMA later and the partial
      vmcnt(N) wait, which keep pf 'kept'. E520: flash2 exposed 1 -> 4 (clash), 1 -> 1 (fix); flash2_partial 1 -> 3 in the fix
      pkg-cmb2 too (E521's accepted residue: V(kt) loads retired at +31 / +59): it passes only with --accept and the ledger row;
    - a new spill in the hot loop (kernel_lint 'spill') or new scratch (binary_gate's row metrics), judged without --allow.
  Fail closed: a binary_gate ERROR row, a CHANGED kernel whose code objects cannot be linted, an --accept that matches no kernel
  with a loss. With --allow, binary_gate's own failures (the untouched-kernel verdict) count; without it they are printed.
  text-only kernels (closure changed, code identical: e.g. the launch grid) have nothing to lint and are printed; common/ktest.sh
  KTEST_CHANGED selects their cases. NEW kernels have no base: printed, not judged (scratch in one fails).
  Printed, not deciding: kernel_lint's minimum load cover ('cov', instructions) and loads in flight ('fly'): no threshold on them
  separates E520's clash from its fix (flash2_partial cov 8 vs 10).

pairs KIT BASE_PKG PREFIX SWITCH=V ...: build every single switch, every pair and (for three or more) the whole set on top of the
  kit's recipe with common/build.sh (packages WORK/PREFIX-<ids>, about 16 s and 0.9 GB each), each checked as above against
  BASE_PKG, and name the smallest switch sets that fail while none of their subsets do (where the collapse first appears).
  Exit 1 when any set fails, 0 when none does, 2 when a build fails or leaves no readable verdict.

usage: cand_lint.py check BASE_PKG CAND_PKG --out DIR [--allow REGEX ...] [--allow-missing REGEX ...] [--accept REGEX ...] [--jobs N]
       cand_lint.py pairs KIT BASE_PKG PREFIX SWITCH=V SWITCH=V ... [--allow REGEX ...] [--accept REGEX ...] [--jobs N]"""
import argparse
import itertools
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CHANGED = ("CHANGED",)        # binary_gate verdicts whose kernels are linted against the base


def prefetch_covers(ins, addr):
    """(per global/buffer load of the hot loop: the MMA / dot instructions issued between it and the s_waitcnt vmcnt that retires it;
    the MMAs of the loop's common path), or None without a multiply loop. The loop body is walked three times and the middle copy's
    loads are counted, so a load carried across the back-edge (the next tile's prefetch) is followed into the next iteration with
    the loads ahead of it in the in-order vmcnt queue."""
    import kernel_lint as kl
    h = kl.hot(ins, addr)
    if h is None:
        return None
    t, e, L = h
    body = ins[t:e + 1]
    n = len(body)
    q, out = [], []
    for i, x in enumerate(body * 3):
        if re.match(r"(global|buffer)_load_", x):
            q.append([0, n <= i < 2 * n])
        if x.startswith(kl.MM):
            for y in q:
                y[0] += 1
        w = re.search(r"vmcnt\((\d+)\)", x) if x.startswith("s_waitcnt") else None
        if w:
            while len(q) > int(w.group(1)):
                c, mid = q.pop(0)
                if mid:
                    out.append(c)
    out += [c for c, mid in q if mid]
    mm = sum(1 for j in kl.common_path(ins, L, t, e) if ins[j].startswith(kl.MM))
    return sorted(out), mm


def with_covers(row, ins, addr):
    pc = prefetch_covers(ins, addr)
    row["covers"], row["loop_mm"] = pc if pc else (None, None)
    return row


def lint_listing(name, ins, addr):
    """kernel_lint's row of a decoded listing plus prefetch_covers (the tests drive this)"""
    import kernel_lint as kl
    return with_covers(kl.lint_listing(name, ins, addr), ins, addr)


def lint_file(path):
    """kernel_lint's row of a code object plus prefetch_covers"""
    import kernel_lint as kl
    from loop_census import disasm
    ins, addr = disasm(path)
    return with_covers(kl.lint(path), ins, addr)


def exposed(b, c):
    """(base, candidate) count of hot-loop loads retired before FULL MMAs of their issue, FULL = min(the base's best cover, the
    candidate loop's MMAs); None when the base has no load in a multiply loop"""
    if not b.get("covers") or c.get("covers") is None:
        return None
    full = min(max(b["covers"]), c["loop_mm"] or 0)
    return sum(1 for x in b["covers"] if x < full), sum(1 for x in c["covers"] if x < full), full


def losses(gate_row, b, c):
    """what the candidate lost against the base: the failure texts for one CHANGED kernel (kernel_lint rows B, C with covers)"""
    import kernel_lint as kl
    why = []
    if b["pf"] == "kept" and c["pf"] != "kept":
        why.append(f"prefetch pf kept -> {c['pf']}")                     # E520 (drain); '-': the load sunk past every MMA
    ex = exposed(b, c)
    if ex and ex[1] > ex[0]:
        why.append(f"exposed loads {ex[0]} -> {ex[1]} (hot-loop loads retired before {ex[2]} MMAs of their issue)")
    if "spill" in kl.regressions(c, b):
        why.append("new spill in the hot loop (spills share vmcnt with the prefetch)")
    gb, gc = gate_row.get("base") or {}, gate_row.get("cand") or {}
    if (gc.get("scratch") or 0) > 0 and not (gb.get("scratch") or 0) > 0:
        why.append(f"new scratch {gc['scratch']} B")
    return why


def judge(rows, lint_base, lint_cand, accept=()):
    """(report lines, failures) of the lint of the gate's changed kernels. ROWS: binary_gate's verdict rows; LINT_BASE / LINT_CAND:
    kernel -> row (lint_file / lint_listing), absent when it could not be linted. ACCEPT: regexes of kernels whose losses a ledger
    decision accepted (each must match a kernel with a loss: a stale waiver fails). Pure.
    A CHANGED kernel fails on any loss: the base's 'kept' prefetch not 'kept' (review F2: not only 'drain'), more exposed loads,
    a new spill, new scratch (binary_gate's row metrics; judged without --allow, review F2b)."""
    lines, fails, accepted = [], [], set()
    f = lambda v: "-" if v is None else str(v)
    for r in rows:
        k, v = r["kernel"], r["verdict"]
        if v == "ERROR":
            fails.append(f"{k}: binary_gate ERROR ({'; '.join(r.get('fail') or []) or 'no reason'}): the changed set is unknown")
            continue
        if v == "text-only":
            lines.append(f"  text     {k}: source closure changed, code identical (launch region or text): nothing to lint")
            continue
        if v == "NEW":
            c = lint_cand.get(k)
            gc = r.get("cand") or {}
            if (gc.get("scratch") or 0) > 0:
                fails.append(f"{k}: NEW with scratch {gc['scratch']} B")
            lines.append(f"  NEW      {k}: pf {c['pf'] if c else '?'} cov {f(c and c['cover_min'])} fly {f(c and c['in_flight'])} "
                         "(no base kernel: not judged)")
            continue
        if v not in CHANGED:
            continue
        b, c = lint_base.get(k), lint_cand.get(k)
        if b is None or c is None:
            fails.append(f"{k}: CHANGED but its {'base' if b is None else 'candidate'} code object was not linted")
            continue
        why = losses(r, b, c)
        ex = exposed(b, c)
        acc = bool(why) and any(re.search(a, k) for a in accept)
        lines.append(f"  {'ACCEPTED' if acc else 'FAIL' if why else 'ok':8s} {k}: pf {b['pf']}->{c['pf']} cov {f(b['cover_min'])}->"
                     f"{f(c['cover_min'])} fly {b['in_flight']}->{c['in_flight']}"
                     + (f" exposed {ex[0]}->{ex[1]} of full {ex[2]}" if ex else "") + (f"  ({'; '.join(why)})" if why else ""))
        if acc:
            accepted.add(k)
        else:
            fails += [f"{k}: {w}" for w in why]
    for a in accept:
        if not any(re.search(a, k) for k in accepted):
            fails.append(f"--accept {a!r} matched no kernel with a loss: remove the waiver")
    return lines, fails


def verdict(rows, bfails, lint_base, lint_cand, allow=(), accept=()):
    """judge() plus binary_gate's own failures, judged when --allow names the switch's kernels (the untouched-kernel verdict needs
    them); its ERROR and scratch failures are judged by judge() either way. Pure."""
    lines, fails = judge(rows, lint_base, lint_cand, accept)
    if allow:
        fails += [f"binary_gate: {x}" for x in bfails]
    elif bfails:
        lines.append(f"  binary_gate: {len(bfails)} untouched-kernel failures not judged (no --allow / LINT_ALLOW): "
                     + "; ".join(bfails[:3]) + (" ..." if len(bfails) > 3 else ""))
    return lines, fails


def lint_dir(d, kernels):
    """kernel -> lint_file row for each of KERNELS with a code object in D"""
    out = {}
    for k in kernels:
        p = os.path.join(d, k + ".hsaco")
        if os.path.exists(p):
            out[k] = lint_file(p)
    return out


def check(base_pkg, cand_pkg, out, allow=(), allow_missing=(), jobs=2, lc=None, accept=()):
    """run the gate and the lint; writes OUT/bgate.json and OUT/verdict.json (with the packages' real paths: ktest_select.py
    changed checks them); returns (lines, failures)"""
    import binary_gate as bg
    os.makedirs(out, exist_ok=True)
    rows, bfails, n = bg.gate(base_pkg, cand_pkg, list(allow), out, jobs, False, lc, list(allow_missing))
    json.dump(rows, open(os.path.join(out, "bgate.json"), "w"), indent=1)
    lines = [f"binary_gate {base_pkg} -> {cand_pkg}: " + ", ".join(
        f"{v} {sum(1 for r in rows if r['verdict'] == v)}" for v in sorted({r["verdict"] for r in rows})) + f"; {n} compiles"]
    lines += [f"  {r['verdict']:9s} {r['kernel']}" for r in rows if r["verdict"] not in ("same", "text-only", "CHANGED", "NEW")]
    ks = [r["kernel"] for r in rows if r["verdict"] in CHANGED + ("NEW",)]
    lb, lcand = lint_dir(os.path.join(out, "base"), ks), lint_dir(os.path.join(out, "cand"), ks)
    ll, fails = verdict(rows, bfails, lb, lcand, allow, accept)
    lines += ll
    json.dump({"base": os.path.realpath(base_pkg), "cand": os.path.realpath(cand_pkg), "fails": fails, "lines": lines,
               "changed": sorted(r["kernel"] for r in rows if r["verdict"] in ("CHANGED", "NEW", "text-only"))},
              open(os.path.join(out, "verdict.json"), "w"), indent=1)
    return lines, fails


def switch_sets(sw):
    """singles, pairs, and the whole set when it has three or more switches (subsets in increasing size)"""
    sets = [(s,) for s in sw] + list(itertools.combinations(sw, 2))
    return sets + ([tuple(sw)] if len(sw) > 2 else [])


def first_failures(results):
    """the failing switch sets none of whose proper subsets (among those built) fails. RESULTS: {tuple of switches: failed}"""
    return [s for s, bad in results.items() if bad and not any(results.get(t) for t in results if set(t) < set(s))]


def run_build(kit, pkg, env):
    """common/build.sh KIT PKG under ENV -> (exit code, stdout, stderr)"""
    r = subprocess.run(["bash", os.path.join(HERE, "build.sh"), kit, pkg], env=env, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def pairs(kit, base, prefix, sw, allow=(), jobs=2, accept=(), build=run_build, work=None):
    """build and check every switch set; returns the exit code: 0 no set fails, 1 a set fails, 2 a build failed or left no verdict
    (BUILD: the builder, a test double in the tests; WORK: where build.sh puts the packages)"""
    names = {s: f"s{i + 1}" for i, s in enumerate(sw)}
    results, report = {}, []
    for s in switch_sets(sw):
        pkg = prefix + "-" + "".join(names[x] for x in s)
        env = dict(os.environ, EXTRA=" ".join(s), LINT_BASE=base, LINT_ALLOW=" ".join(allow), LINT_ACCEPT=" ".join(accept),
                   LINT_JOBS=str(jobs))
        rc, out, err = build(kit, pkg, env)
        print(out.strip())
        if rc not in (0, 3):
            print(f"cand_lint pairs: build of {pkg} failed (exit {rc}): {err.strip()[-300:]}")
            return 2
        vf = os.path.join(work or os.environ.get("WORK") or kit_work(kit), pkg + ".lint", "verdict.json")
        try:
            v = json.load(open(vf))
            bad = bool(v["fails"])
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(f"cand_lint pairs: {pkg} left no readable verdict ({vf}: {e})")
            return 2
        results[s] = bad or rc == 3
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
        x.add_argument("--accept", action="append", default=[], help="regex of a kernel whose loss a ledger decision accepted "
                       "(cite it); each must match a kernel with a loss")
        x.add_argument("--jobs", type=int, default=2)
    c.add_argument("--allow-missing", action="append", default=[])
    a = ap.parse_args()
    for x in a.allow + a.accept + getattr(a, "allow_missing", []):
        try:
            re.compile(x)
        except re.error as e:
            ap.error(f"--allow {x!r}: {e}")
    if a.cmd == "pairs":
        bad = [s for s in a.switches if not re.match(r"[A-Za-z_]\w*=\S*$", s)]
        if bad or len(a.switches) < 2:
            ap.error(f"pairs: two or more SWITCH=V (got {a.switches})")
        sys.exit(pairs(a.kit, a.base, a.prefix, a.switches, a.allow, a.jobs, a.accept))
    lines, fails = check(a.base, a.cand, a.out, a.allow, a.allow_missing, a.jobs, accept=a.accept)
    print("\n".join(lines))
    for x in fails:
        print("  FAIL " + x)
    print("CAND_LINT: " + ("FAIL" if fails else "PASS"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
