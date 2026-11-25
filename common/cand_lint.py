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
      followed across the back-edge into later iterations), capped at FULL = min(the base's best cover,
      the candidate loop's MMAs), sorted and compared rank by rank (tolerance 0 MMAs): a rank the base covered fully and the
      candidate does not is 'exposed a->b' (counts of ranks below FULL); a rank already below FULL in the base that drops further
      is 'coverR a->b' (verify r2 V2); a changed load count ('loads') or loop MMA count ('mma') cannot be paired and is a loss;
    - a new spill in the hot loop (kernel_lint scratch, 'spill') or new scratch bytes (binary_gate, 'scratch'; a NEW kernel's too).
  Pins (--accept KERNEL:KIND:FROM->TO, comma-separated; build.sh LINT_ACCEPT): a ledger-accepted loss. A kernel passes only when
  its pins equal its loss set exactly; any other or extra loss fails, a pin that does not equal its kernel's loss set fails, and a
  malformed pin (a regex, an unknown kind) is refused, exit 2 (verify r2 V1: '--accept flash2' passed E520's whole clash).
  E520: pkg-cmb-sf partial {pf kept->drain, exposed 1->4}; pkg-cmb2 partial {exposed 1->3} (E521's accepted residue, V(kt) loads
  retired at +31 / +59): it passes with g_attention_flash2_partial:exposed:1->3,g_attention_flash2_partial_f16o:exposed:1->3.
  Fail closed: a binary_gate ERROR row, a CHANGED kernel whose code objects cannot be linted (neither can be pinned). With
  --allow, binary_gate's own failures (the untouched-kernel verdict) count; without it they are printed.
  text-only kernels (closure changed, code identical: e.g. the launch grid) have nothing to lint and are printed; common/ktest.sh
  KTEST_CHANGED selects their cases. NEW kernels have no base: printed, not judged (scratch in one fails).
  Printed, not deciding: kernel_lint's minimum load cover ('cov', instructions) and loads in flight ('fly'): no threshold on them
  separates E520's clash from its fix (flash2_partial cov 8 vs 10).

pairs KIT BASE_PKG PREFIX SWITCH=V ...: build every single switch, every pair and (for three or more) the whole set on top of the
  kit's recipe with common/build.sh (packages WORK/PREFIX-<ids>, about 16 s and 0.9 GB each), each checked as above against
  BASE_PKG, and name the smallest switch sets that fail while none of their subsets do (where the collapse first appears).
  Exit 1 when any set fails, 0 when none does, 2 when a build fails or leaves no readable verdict.

usage: cand_lint.py check BASE_PKG CAND_PKG --out DIR [--allow REGEX ...] [--allow-missing REGEX ...] [--accept PINS ...] [--jobs N]
       cand_lint.py pairs KIT BASE_PKG PREFIX SWITCH=V SWITCH=V ... [--allow REGEX ...] [--accept PINS ...] [--jobs N]"""
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


WALK = 8   # loop copies walked: a load retired up to WALK - 1 iterations after its issue gets its exact cover


def prefetch_covers(ins, addr):
    """(per global/buffer load of the hot loop: the MMA / dot instructions issued between it and the s_waitcnt vmcnt that retires it;
    the MMAs of the loop's common path), or None without a multiply loop. The loads of one iteration are followed across the
    back-edge into the next iterations (the next tile's prefetch is retired there), up to WALK copies of the body. vmcnt retires in
    issue order and a wait vmcnt(N) retires a load exactly when more than N loads issued at or after it are pending, which older
    loads do not change: the first iteration's covers are the steady state's (verify r2 V6; r2 measured the middle of three copies
    and undercounted a load retired two iterations later)."""
    import kernel_lint as kl
    h = kl.hot(ins, addr)
    if h is None:
        return None
    t, e, L = h
    body = ins[t:e + 1]
    q, out, mine = [], [], 0
    for i in range(WALK * len(body)):
        x = body[i % len(body)]
        if re.match(r"(global|buffer)_load_", x):
            q.append([0, i < len(body)])
            mine += i < len(body)
        if x.startswith(kl.MM):
            for y in q:
                y[0] += 1
        w = re.search(r"vmcnt\((\d+)\)", x) if x.startswith("s_waitcnt") else None
        if w:
            while len(q) > int(w.group(1)):
                c, first = q.pop(0)
                if first:
                    out.append(c)
        if i >= len(body) - 1 and len(out) == mine:
            break
    out += [c for c, first in q if first]              # never retired within the walk: covers everything walked
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
    """(base, candidate) count of hot-loop loads retired before FULL MMAs of their issue, FULL; FULL = min(the base's best cover,
    the candidate loop's MMAs); None when the base has no load in a multiply loop or the candidate none to compare"""
    if not b.get("covers") or not c.get("covers"):
        return None
    full = min(max(b["covers"]), c["loop_mm"] or 0)
    return sum(1 for x in b["covers"] if x < full), sum(1 for x in c["covers"] if x < full), full


def val(x):
    return "none" if x in (None, "-", "") else str(x)


def losses(gate_row, b, c):
    """the candidate's losses against the base for one CHANGED kernel, as (kind, from, to) triples (the pin vocabulary; the state
    table in HANDOFF r3). B, C: lint rows with covers; GATE_ROW: binary_gate's row (scratch bytes)."""
    out = []
    if b["pf"] == "kept" and c["pf"] != "kept":
        out.append(("pf", "kept", val(c["pf"])))                         # E520 drain; '-': the load sunk past every MMA
    if b.get("covers"):
        bm, cm = b.get("loop_mm"), c.get("loop_mm") if c.get("covers") is not None else None
        if bm != cm:
            out.append(("mma", val(bm), val(cm)))
        if c.get("covers") is not None and len(c["covers"]) != len(b["covers"]):
            out.append(("loads", str(len(b["covers"])), str(len(c["covers"]))))
        elif c.get("covers"):
            ne, ce, full = exposed(b, c)
            # ranks of the sorted capped covers; a rank fully covered in the base and not in the candidate is exactly a rise in the
            # exposed count (both sorted, same length), so ranks only add the decay of loads the base already exposed (verify r2 V2)
            cap = lambda v: sorted(min(x, full) for x in v)
            for r, (x, y) in enumerate(zip(cap(b["covers"]), cap(c["covers"]))):
                if x < full and y < x:
                    out.append((f"cover{r}", str(x), str(y)))
            if ce > ne:
                out.append(("exposed", str(ne), str(ce)))
    if (c.get("scratch") or 0) and not (b.get("scratch") or 0):
        out.append(("spill", val(b.get("scratch") or 0), val(c["scratch"])))   # spills share vmcnt with the prefetch
    gb, gc = gate_row.get("base") or {}, gate_row.get("cand") or {}
    if (gc.get("scratch") or 0) > 0 and not (gb.get("scratch") or 0) > 0:
        out.append(("scratch", "0", str(gc["scratch"])))
    return out


PIN = re.compile(r"([A-Za-z0-9_]+):(pf|exposed|cover\d+|loads|mma|spill|scratch):([A-Za-z0-9.]+)->([A-Za-z0-9.]+)")


def parse_pins(specs):
    """KERNEL:KIND:FROM->TO pins (each SPEC may hold several, comma-separated) -> {kernel: set of (kind, from, to)}; ValueError on a
    malformed pin (a regex, a missing part, an unknown kind): a waiver is exact or it is refused (review r2 V1)"""
    pins = {}
    for spec in specs:
        for x in spec.split(","):
            x = x.strip()
            if not x:
                continue
            m = PIN.fullmatch(x)
            if not m:
                raise ValueError(f"malformed pin {x!r}: want KERNEL:KIND:FROM->TO with an exact kernel name, KIND one of pf, exposed, "
                                 "coverN, loads, mma, spill, scratch")
            pins.setdefault(m.group(1), set()).add(m.group(2, 3, 4))
    return pins


def fmt(loss):
    return f"{loss[0]} {loss[1]}->{loss[2]}"


def judge(rows, lint_base, lint_cand, pins=None):
    """(report lines, failures) of the lint of the gate's changed kernels. ROWS: binary_gate's verdict rows; LINT_BASE / LINT_CAND:
    kernel -> row (lint_file / lint_listing), absent when it could not be linted. PINS: parse_pins() of the losses a ledger row
    accepted. A kernel with losses passes only when its pins equal its loss set exactly; every pin must match. Pure."""
    pins = pins or {}
    lines, fails, used = [], [], set()
    f = lambda v: "-" if v is None else str(v)

    def settle(k, ls, head):
        exact = bool(ls) and pins.get(k) == set(ls)
        lines.append(f"  {'PINNED' if exact else 'FAIL' if ls else 'ok':8s} {k}: {head}" + (f"  ({'; '.join(map(fmt, ls))})" if ls else ""))
        if exact:
            used.add(k)
        else:
            fails.extend(f"{k}: {fmt(x)}" for x in ls)

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
            ls = [("scratch", "new", str(gc["scratch"]))] if (gc.get("scratch") or 0) > 0 else []
            settle(k, ls, f"NEW: pf {c['pf'] if c else '?'} cov {f(c and c['cover_min'])} fly {f(c and c['in_flight'])} "
                          "(no base kernel: only scratch judged)")
            continue
        if v not in CHANGED:
            continue
        b, c = lint_base.get(k), lint_cand.get(k)
        if b is None or c is None:
            fails.append(f"{k}: CHANGED but its {'base' if b is None else 'candidate'} code object was not linted")
            continue
        ex = exposed(b, c)
        settle(k, losses(r, b, c), f"pf {b['pf']}->{c['pf']} cov {f(b['cover_min'])}->{f(c['cover_min'])} fly {b['in_flight']}->"
                                   f"{c['in_flight']}" + (f" exposed {ex[0]}->{ex[1]} of full {ex[2]}" if ex else ""))
    for k, ps in sorted(pins.items()):
        if k not in used:
            fails.append(f"pin {k}:" + ",".join(f"{a}:{x}->{y}" for a, x, y in sorted(ps))
                         + " does not equal that kernel's loss set (it matched nothing, or the kernel has other or more losses)")
    return lines, fails


def verdict(rows, bfails, lint_base, lint_cand, allow=(), pins=None):
    """judge() plus binary_gate's own failures, judged when --allow names the switch's kernels (the untouched-kernel verdict needs
    them); its ERROR and scratch failures are judged by judge() either way. Pure."""
    lines, fails = judge(rows, lint_base, lint_cand, pins)
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


def cmd_sha(pkg):
    """sha256 of PKG.cmd (build.sh's record of the build command and switches), None without one; the same digest as
    kit/ktest_select.py cmd_sha: a gate dir outside PKG.lint stays tied to the build it judged (verify r2 F5 residual)"""
    import hashlib
    try:
        with open(os.path.realpath(pkg) + ".cmd", "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def check(base_pkg, cand_pkg, out, allow=(), allow_missing=(), jobs=2, lc=None, pins=None):
    """run the gate and the lint; writes OUT/bgate.json and OUT/verdict.json (with the packages' real paths and the candidate's
    cmd_sha: ktest_select.py changed checks them); returns (lines, failures)"""
    import binary_gate as bg
    os.makedirs(out, exist_ok=True)
    rows, bfails, n = bg.gate(base_pkg, cand_pkg, list(allow), out, jobs, False, lc, list(allow_missing))
    json.dump(rows, open(os.path.join(out, "bgate.json"), "w"), indent=1)
    lines = [f"binary_gate {base_pkg} -> {cand_pkg}: " + ", ".join(
        f"{v} {sum(1 for r in rows if r['verdict'] == v)}" for v in sorted({r["verdict"] for r in rows})) + f"; {n} compiles"]
    lines += [f"  {r['verdict']:9s} {r['kernel']}" for r in rows if r["verdict"] not in ("same", "text-only", "CHANGED", "NEW")]
    ks = [r["kernel"] for r in rows if r["verdict"] in CHANGED + ("NEW",)]
    lb, lcand = lint_dir(os.path.join(out, "base"), ks), lint_dir(os.path.join(out, "cand"), ks)
    ll, fails = verdict(rows, bfails, lb, lcand, allow, pins)
    lines += ll
    json.dump({"base": os.path.realpath(base_pkg), "cand": os.path.realpath(cand_pkg), "cand_cmd_sha": cmd_sha(cand_pkg),
               "fails": fails, "lines": lines,
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


def pairs(kit, base, prefix, sw, allow=(), jobs=2, pins=(), build=run_build, work=None):
    """build and check every switch set; returns the exit code: 0 no set fails, 1 a set fails, 2 a build failed or left no verdict
    (BUILD: the builder, a test double in the tests; WORK: where build.sh puts the packages)"""
    names = {s: f"s{i + 1}" for i, s in enumerate(sw)}
    results, report = {}, []
    for s in switch_sets(sw):
        pkg = prefix + "-" + "".join(names[x] for x in s)
        env = dict(os.environ, EXTRA=" ".join(s), LINT_BASE=base, LINT_ALLOW=" ".join(allow), LINT_ACCEPT=",".join(pins),
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
        x.add_argument("--accept", action="append", default=[], help="KERNEL:KIND:FROM->TO[,...]: a loss a ledger row accepted "
                       "(cite it); a kernel passes only when its pins equal its losses, and every pin must match")
        x.add_argument("--jobs", type=int, default=2)
    c.add_argument("--allow-missing", action="append", default=[])
    a = ap.parse_args()
    try:
        pins = parse_pins(a.accept)
    except ValueError as e:
        print(f"cand_lint: {e}", file=sys.stderr)
        sys.exit(2)
    for x in a.allow + getattr(a, "allow_missing", []):
        try:
            re.compile(x)
        except re.error as e:
            ap.error(f"--allow {x!r}: {e}")
    if a.cmd == "pairs":
        bad = [s for s in a.switches if not re.match(r"[A-Za-z_]\w*=\S*$", s)]
        if bad or len(a.switches) < 2:
            ap.error(f"pairs: two or more SWITCH=V (got {a.switches})")
        sys.exit(pairs(a.kit, a.base, a.prefix, a.switches, a.allow, a.jobs, a.accept))
    lines, fails = check(a.base, a.cand, a.out, a.allow, a.allow_missing, a.jobs, pins=pins)
    print("\n".join(lines))
    for x in fails:
        print("  FAIL " + x)
    print("CAND_LINT: " + ("FAIL" if fails else "PASS"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
