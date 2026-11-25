#!/usr/bin/env python3
"""The one reader of `loom-compile-report suggest` (Loom's own static advisor) for common/: run it once per compile report, parse
its JSON, and turn each finding into a short labelled line. Callers: lint_counters.py (a column beside each ranked lever and
(d2)'s occupancy lines) and kernel_lint.py (notes). A finding is ADVICE, not a measurement: suggest's own guidance says its order
and confidence do not rank benefit, and nothing here changes a price.

Fail closed:
  - a failed run is 'unknown (loom-compile-report suggest rc N)' (timed out, not found, output not JSON, a status other than
    'available', a kind or schema this reader does not know: each its own 'unknown (...)'), never an empty list;
  - a finding id outside KNOWN is shown as 'unrecognised suggest finding: <id>', never dropped;
  - a residency_cliff is shown only through common/lint_counters.occupancy()'s whole-workgroup tiers (cliff_verdict): Loom 0.2.0's
    report rounds VGPRs per wave, not per workgroup, so its next tier can be one the hardware cannot seat (HRX-like E533: flash2's
    'tier 7' at 3 waves per SIMD per workgroup, the 256-lane GEMMs' 'tier 5' at 2). Such a tier is 'phantom (E533)', never a lever;
    one occupancy() cannot verify is 'unverified', and only 'agrees' (the same reachable tier and the same cuts) is advice to act on.
`--include-experimental` is not used by default: on Loom 0.2.0 it exits 2 on g_attention_flash2_f16o ('argument_packets[9].
packet_count: subgroup access groups model 64, got 56'). Loom 0.3.0's default output keeps one example per investigation and adds
match_count (its `--expand all` lists every match); the kinds read here are the same.
usage: loom_suggest.py REPORT.json [--tool PATH] [--timeout S] [--include-experimental]   (prints the labelled findings)"""
import json
import os
import re
import subprocess
import sys

KIND, SCHEMA = "loom.compile_report.suggest", 0
CLIFF = "amdgpu.residency_cliff"
# id -> short label; every id seen on live pkg-e480's 261 reports (Loom 0.2.0, 11-25). Anything else is unrecognised.
KNOWN = {CLIFF: "cliff", "amdgpu.lds_bank_service": "lds bank", "amdgpu.lds_ssa_use_serialization": "lds drains",
         "math.inspect_native_fastmath": "fastmath", "vector.compare_componentwise_bank_state": "bank state",
         "lowering.reduce_expansion": "expand"}
ADVICE = "Loom suggest: advice from static evidence, not a measurement; never changes a price"


def tool(explicit=None):
    """the loom-compile-report to run: EXPLICIT, else $LOOM_COMPILE_REPORT, else bin/ of $VENV or of the machine env file's VENV
    (as kernel_lint finds loom-compile); None when there is none"""
    if explicit:
        return explicit
    if os.environ.get("LOOM_COMPILE_REPORT"):
        return os.environ["LOOM_COMPILE_REPORT"]
    venv = os.environ.get("VENV")
    if not venv:
        f = os.environ.get("LSK_ENV") or os.path.expanduser("~/.config/loom-strix-kits/env")
        m = re.search(r"(?m)^VENV=(\S+)", open(f).read()) if os.path.exists(f) else None
        venv = m.group(1) if m else None
    p = os.path.join(venv, "bin", "loom-compile-report") if venv else None
    return p if p and os.access(p, os.X_OK) else None


def parse(text):
    """{"findings": [...]} from suggest's JSON text, or {"error": "unknown (...)"}: the document must be suggest's kind and schema
    with status 'available'"""
    try:
        doc = json.loads(text)
    except ValueError:
        return {"error": "unknown (loom-compile-report suggest: output not JSON)"}
    if not isinstance(doc, dict) or doc.get("kind") != KIND or doc.get("schema_version") != SCHEMA:
        k = doc.get("kind") if isinstance(doc, dict) else type(doc).__name__
        v = doc.get("schema_version") if isinstance(doc, dict) else None
        return {"error": f"unknown (loom-compile-report suggest: kind {k!r} schema {v!r}, this reader knows {KIND!r} {SCHEMA})"}
    if doc.get("status") != "available":
        why = doc.get("target_unavailable_reason")
        return {"error": f"unknown (loom-compile-report suggest status {doc.get('status')!r}" + (f": {why}" if why else "") + ")"}
    f = doc.get("findings")
    if not isinstance(f, list) or not all(isinstance(x, dict) for x in f):
        return {"error": "unknown (loom-compile-report suggest: no findings list)"}
    tu = doc.get("target_unavailable_reason")
    return {"findings": f, "target_unavailable": str(tu) if tu else None}


def run(report, tool_path, timeout=60, experimental=False):
    """suggest on one compile report: {"findings": [...]} or {"error": "unknown (...)"}; never raises for a tool failure"""
    if not tool_path:
        return {"error": "unknown (loom-compile-report not found: pass its path, or set LOOM_COMPILE_REPORT or VENV)"}
    if not os.path.exists(report):
        return {"error": "no compile report"}
    cmd = [tool_path, "suggest", report, "--format=json"] + (["--include-experimental"] if experimental else [])
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"unknown (loom-compile-report suggest timed out after {timeout:g} s)"}
    except OSError as e:
        return {"error": f"unknown (loom-compile-report suggest could not start: {e.strerror})"}
    if p.returncode:
        last = ((p.stderr or p.stdout).strip().splitlines() or [""])[-1][:160]
        return {"error": f"unknown (loom-compile-report suggest rc {p.returncode})" + (f": {last}" if last else "")}
    return parse(p.stdout)


def cliff(x):
    """Loom's residency cliff from a finding's evidence: {"cur", "next", "needs": {resource: max units}} or None when the evidence
    is not the shape this reader knows (then the finding is unrecognised, not dropped)"""
    ev = x.get("evidence")
    if not isinstance(ev, dict):
        return None
    cur = next((v for k, v in ev.items() if k.endswith(".residency.current_tier")), None)
    nxt = next((v for k, v in ev.items() if k.endswith(".residency.next_better_tier")), None)
    rows = {}
    for k, v in ev.items():
        m = re.match(r"(residency_constraints\.rows\[\d+\])\.(name|units|reduction_units_to_next_better_tier)$", k)
        if m:
            rows.setdefault(m.group(1), {})[m.group(2)] = v
    needs = {}
    for r in rows.values():
        name, u, cut = r.get("name"), r.get("units"), r.get("reduction_units_to_next_better_tier")
        if not isinstance(name, str) or not isinstance(u, int) or not isinstance(cut, int):
            return None
        needs[name.split(".")[-1]] = u - cut
    if not isinstance(cur, int) or not isinstance(nxt, int) or not needs:
        return None
    return {"cur": cur, "next": nxt, "needs": needs}


def cliff_verdict(c, occ):
    """(verdict, text) for Loom's cliff C against OCC, common/lint_counters.occupancy()'s result for the same report (None: at the best
    reachable tier; {"why"}: not verified; else {"tier", "next", "q", "needs"}). Verdicts: 'agrees' (Loom's next tier is the reachable
    one and its cuts are occupancy()'s), 'phantom (E533)' (a tier whole workgroups cannot seat: not a multiple of the waves per SIMD
    per workgroup, or above the best reachable tier), 'unverified' (occupancy() cannot verify the tier: Loom's cannot be checked),
    'disagrees' (a seatable tier, but not the reachable next one or not the same cuts). Only 'agrees' is advice to act on."""
    lt = f"Loom next tier {c['next']}"
    if occ is None:
        return "phantom (E533)", f"{lt}: phantom (E533), the kernel is at its best reachable tier"
    if "why" in occ:
        if occ["why"].startswith("next tier") and "blocked" in occ["why"]:
            return "phantom (E533)", f"{lt}: phantom (E533), {occ['why']}"
        return "unverified", f"{lt}: unverified ({occ['why']})"
    q, on = occ.get("q"), occ.get("next")
    if not isinstance(q, int) or q <= 0 or not isinstance(on, int) or not isinstance(occ.get("needs"), list):
        return "unverified", f"{lt}: unverified (occupancy result without whole-workgroup steps: {sorted(occ)})"
    if occ.get("bound"):   # an occupancy() that gives a tier as a bound (lintloom r2's non-whole-wave workgroups) verifies nothing
        return "unverified", f"{lt}: unverified (the whole-workgroup tier is a bound, not exact)"
    if c["next"] % q:
        return "phantom (E533)", (f"{lt}: phantom (E533), {q} waves/SIMD per workgroup seat tiers in steps of {q}; "
                                  f"reachable next {on}")
    mine = {n: mx for n, _, _, mx, _ in occ["needs"]}
    # the same cuts land on one tier: occupancy() may name tier + q (lint_counters at 43293d4) or the tier the cuts reach
    # (Loom names that one: g_mv_q4_k_k4096_t8_q8, VGPR <= 96 lands on 16, not 13)
    if c["needs"] == mine and c["next"] >= on:
        return "agrees", f"{lt}: agrees with the whole-workgroup cuts" + (f" (they land above tier {on})" if c["next"] > on else "")
    return "disagrees", (f"{lt} ({', '.join(f'{n} <= {v}' for n, v in sorted(c['needs'].items()))}): disagrees with the "
                         f"whole-workgroup tier {occ['next']} ({', '.join(f'{n} <= {v}' for n, v in sorted(mine.items()))}); not a lever")


def _num(ev, suffix):
    return sum(v for k, v in (ev or {}).items() if k.endswith(suffix) and isinstance(v, (int, float)))


def describe(res, occ=None, occ_known=False):
    """(cell, lines) for one report's suggest result RES ({"findings"} or {"error"}): CELL a compact summary for a table column,
    LINES one labelled line per finding kind. OCC is occupancy()'s result for the same report (OCC_KNOWN False: occupancy was not
    computed, so a cliff is 'unverified')."""
    if "error" in res:
        return res["error"], [res["error"]]
    by = {}
    for x in res["findings"]:
        by.setdefault(x.get("id") if isinstance(x.get("id"), str) else repr(x.get("id")), []).append(x)
    cells, lines = [], []
    if res.get("target_unavailable"):   # live: g_silu_mul / _lv, 'compile_status_not_ok': source advice only, no residency evidence
        cells.append(f"target analysis unavailable ({res['target_unavailable']})")
        lines.append(f"target analysis unavailable ({res['target_unavailable']}): source findings only, no residency or LDS evidence")
    for i in sorted(by, key=lambda i: (i not in KNOWN, list(KNOWN).index(i) if i in KNOWN else 0, i)):
        xs = by[i]
        if i not in KNOWN:
            cells.append(f"unrecognised suggest finding: {i}")
            lines.append(f"unrecognised suggest finding: {i} (x{len(xs)}): {(xs[0].get('action') or '')[:160]}")
            continue
        if i == CLIFF:
            c = cliff(xs[0])
            if c is None or len(xs) != 1:
                cells.append(f"unrecognised suggest finding: {i} (evidence shape)")
                lines.append(f"unrecognised suggest finding: {i}: {len(xs)} finding(s), evidence not the shape this reader knows")
                continue
            v, t = cliff_verdict(c, occ) if occ_known else ("unverified", f"Loom next tier {c['next']}: unverified (occupancy not computed)")
            cells.append(f"cliff {c['cur']}->{c['next']} {v}")
            lines.append(f"residency cliff: {t}")
            continue
        ev = [x.get("evidence") for x in xs]
        if i == "amdgpu.lds_bank_service":
            n = sum(_num(e, "summary.structural.extra_round_count") for e in ev)
            cells.append(f"lds bank +{n:g} rounds")
            lines.append(f"lds bank service: {len(xs)} buffer(s), {n:g} extra static service rounds (static counts, not cycles)")
        elif i == "amdgpu.lds_ssa_use_serialization":
            n = sum(_num(e, "wait_plan.full_drain_count") for e in ev)
            cells.append(f"lds drains {n:g}")
            lines.append(f"lds SSA-use serialization: LDS results consumed after full counter drains ({n:g} full drains in the wait plan)")
        elif i == "math.inspect_native_fastmath":
            n = sum(_num(e, "native_alternative_op_count") for e in ev)
            cells.append(f"fastmath {n:g} ops")
            lines.append(f"native fast-math: {n:g} source math op(s) have native forms under more fast-math (an accuracy decision)")
        elif i == "vector.compare_componentwise_bank_state":
            cells.append(f"bank state x{len(xs)}")
            lines.append(f"componentwise bank state: {len(xs)} loop bank(s) kept whole by a whole-value use")
        else:   # lowering.reduce_expansion
            worst = max((_num(e, "maximum_emitted_low_op_count") for e in ev), default=0)
            cells.append(f"expand x{len(xs)}")
            lines.append(f"reduce expansion: {len(xs)} source op(s) expand past 5 Low ops (worst x{worst:g}; stage-local counts, not costs)")
    return ("; ".join(cells) or "no findings"), (lines or ["no findings"])


def survey(reports, tool_path, occ_of=None, jobs=4, timeout=60):
    """THE entry point for callers: {kernel: {"res", "cell", "lines", "notes"}} for REPORTS {kernel: compile report path}, suggest run
    once per report (JOBS at a time). OCC_OF(report path) -> the caller's occupancy() result for that report (None: best reachable
    tier); without it every cliff is 'unverified'. An OCC_OF that raises makes that kernel's cliff 'unverified', never the run."""
    import concurrent.futures

    def one(k):
        res = run(reports[k], tool_path, timeout)
        known, occ = False, None
        if occ_of is not None and "findings" in res and any(x.get("id") == CLIFF for x in res["findings"]):
            try:
                occ, known = occ_of(reports[k]), True
            except Exception as e:   # noqa: BLE001 -- a malformed report: the cliff stays unverified, the others print
                occ, known = {"why": f"occupancy failed: {type(e).__name__}"}, True
        cell, lines = describe(res, occ, known)
        return k, {"res": res, "cell": cell, "lines": lines, "notes": notes(res)}
    with concurrent.futures.ThreadPoolExecutor(max(1, jobs)) as ex:
        return dict(ex.map(one, sorted(reports)))


NOTELESS = (CLIFF, "lowering.reduce_expansion")   # a cliff is the 'next' column / occupancy(); expansion fires on 258 of 261 kernels


def notes(res):
    """kernel_lint's NOTES for one suggest result (never flags: suggest's order and confidence do not rank benefit): 'loom:<id
    without its family>' per known kind except NOTELESS, 'loom:unrecognised:<id>' per unknown kind, 'loom:unknown' when the run
    failed (its reason is in describe()'s lines)"""
    if "error" in res:
        return ["loom:unknown"] if res["error"] != "no compile report" else []
    out = []
    for x in res["findings"]:
        i = x.get("id") if isinstance(x.get("id"), str) else repr(x.get("id"))
        n = ("loom:" + i.split(".", 1)[-1]) if i in KNOWN else "loom:unrecognised:" + i
        if i not in NOTELESS and n not in out:
            out.append(n)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("report")
    ap.add_argument("--tool")
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--include-experimental", action="store_true")
    a = ap.parse_args()
    res = run(a.report, tool(a.tool), a.timeout, a.include_experimental)
    cell, lines = describe(res)
    print(ADVICE)
    print("\n".join(lines))
    sys.exit(2 if "error" in res else 0)


if __name__ == "__main__":
    main()
