#!/usr/bin/env python3
"""common/loom_suggest.py (the one reader of `loom-compile-report suggest`) on real suggest output and real compile reports from
live pkg-e480 (Loom 0.2.0; kit/fixtures/loom_suggest/: suggest JSON as the tool wrote it with `source` cut to a basename, and the
report fields lint_counters.report_limit reads). Fake tools stand only at the process boundary. Recurrences each case prevents:
  - Loom's residency cliff relayed as a lever at a tier the hardware cannot seat (HRX-like E533: flash2_f16o 'tier 7' at 3 waves
    per SIMD per workgroup; the 256-lane GEMMs 'tier 5' at 2): 'phantom (E533)', judged by lint_counters.occupancy();
  - a cliff occupancy() cannot verify (160 / 192-lane workgroups: the q5_k t8 matvec, vsplit) shown as agreeing: 'unverified';
  - a correct Loom cliff called phantom because occupancy() names tier + q while the same cuts land higher (g_mv_q4_k_k4096_t8_q8:
    VGPR <= 96 lands on 16, Loom says 16, occupancy() says 13);
  - a suggest failure killing the run or vanishing (lintloom-1125 review: a missing tool raised FileNotFoundError, exit 0 with
    non-JSON raised JSONDecodeError, a JSON list raised AttributeError, no timeout; 0.2.0 `--include-experimental` exits 2 on
    flash2_f16o): each is a printed 'unknown (...)';
  - a finding kind this reader does not know silently dropped: 'unrecognised suggest finding: <id>';
  - an occupancy failure on one report taking down the survey of the others."""
import json
import os
import stat
import sys
import tempfile

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import lint_counters as lc  # noqa: E402
import loom_suggest as ls  # noqa: E402

FIX = os.path.join(H, "..", "kit", "fixtures", "loom_suggest")
fails = 0


def ok(cond, what, got=""):
    global fails
    print(("ok  " if cond else "FAIL") + f" {what}" + (f": {got}" if not cond else ""))
    fails += not cond


def fixture(k):
    res = ls.parse(open(os.path.join(FIX, k + ".suggest.json")).read())
    occ = lc.occupancy(lc.report_limit(os.path.join(FIX, k + ".report.json"))[4], 4)
    return res, occ


def fake(t, name, body):
    p = os.path.join(t, name)
    open(p, "w").write("#!/bin/sh\n" + body + "\n")
    os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR)
    return p


def main():
    # ---------------------------------------------------------------- real live findings against whole-workgroup tiers
    res, occ = fixture("g_attention_flash2_f16o")
    cell, lines = ls.describe(res, occ, True)
    ok("cliff 6->7 phantom (E533)" in cell and any("reachable next 9" in x for x in lines),
       "flash2_f16o: Loom's tier 7 is phantom (3 waves/SIMD per workgroup: 3/6/9), reachable 9 named", (cell, lines))
    ok("agrees" not in cell, "flash2_f16o: a phantom cliff is never 'agrees'", cell)
    res, occ = fixture("g_gemm_f16_acc_m256")
    cell, lines = ls.describe(res, occ, True)
    ok("cliff 4->5 phantom (E533)" in cell and any("reachable next 6" in x for x in lines), "GEMM m256: Loom's tier 5 is phantom", (cell, lines))
    ok("lds bank +48 rounds" in cell and "expand x101" in cell, "GEMM m256: bank service rounds and expansion count from the evidence", cell)
    ok(ls.notes(res) == ["loom:lds_bank_service"], "GEMM m256 notes: lds_bank_service only (cliff and expansion are not notes)", ls.notes(res))
    for k, t in (("g_mv_q5_k_k5120_t8_q8", "12->15"), ("g_attention_vsplit", "3->5")):
        res, occ = fixture(k)
        cell, lines = ls.describe(res, occ, True)
        ok(f"cliff {t} unverified" in cell and any("not whole waves per SIMD" in x for x in lines),
           f"{k}: occupancy() cannot verify the tier, so Loom's cliff is 'unverified'", (cell, lines))
    res, occ = fixture("g_mv_q4_k_k4096_t8_q8")
    cell, lines = ls.describe(res, occ, True)
    ok("cliff 12->16 agrees" in cell, "q4_k k4096: the same cuts (VGPR <= 96) land on Loom's 16, above occupancy()'s 13: agrees", (cell, occ))
    cell, _ = ls.describe(res, occ, False)
    ok("cliff 12->16 unverified" in cell, "occupancy not computed: a cliff is unverified, never agrees", cell)

    # ---------------------------------------------------------------- cliff verdicts on synthetic occupancy results
    c = {"cur": 6, "next": 7, "needs": {"vgpr": 216, "lds": 43008}}
    ok(ls.cliff_verdict(c, None)[0] == "phantom (E533)", "a cliff past the best reachable tier is phantom", ls.cliff_verdict(c, None))
    blk = {"why": "next tier 9 blocked by a fixed limit: workgroup_slots (tier 7)"}
    ok(ls.cliff_verdict({"cur": 6, "next": 9, "needs": {"vgpr": 168}}, blk)[0] == "phantom (E533)", "a cliff behind a fixed limit is phantom",
       ls.cliff_verdict(c, blk))
    occ9 = {"tier": 6, "next": 9, "q": 3, "needs": [("vgpr", 240, "registers", 168, 72), ("lds", 61952, "bytes", 43008, 18944)]}
    ok(ls.cliff_verdict({"cur": 6, "next": 9, "needs": {"vgpr": 168, "lds": 43008}}, occ9)[0] == "agrees", "same tier, same cuts: agrees")
    v = ls.cliff_verdict({"cur": 6, "next": 9, "needs": {"vgpr": 168}}, occ9)
    ok(v[0] == "disagrees" and "not a lever" in v[1], "a seatable tier with fewer cuts than occupancy() needs: disagrees, not a lever", v)
    v = ls.cliff_verdict({"cur": 6, "next": 12, "needs": {"vgpr": 120, "lds": 43008}}, occ9)
    ok(v[0] == "disagrees", "a seatable tier with other cuts: disagrees", v)
    ok(ls.cliff_verdict({"cur": 6, "next": 9, "needs": {"vgpr": 168, "lds": 43008}}, dict(occ9, bound=True))[0] == "unverified",
       "an occupancy tier marked as a bound (lintloom r2's non-whole-wave form) verifies nothing")
    ok(ls.cliff_verdict({"cur": 6, "next": 9, "needs": {"vgpr": 168}}, {"tier": 6, "next": 9})[0] == "unverified",
       "an occupancy result without whole-workgroup steps: unverified")

    # ---------------------------------------------------------------- fail closed on what the reader does not know
    doc = json.load(open(os.path.join(FIX, "g_gemm_f16_acc_m256.suggest.json")))
    doc["findings"].append({"id": "amdgpu.new_advice", "action": "Try something new.", "evidence": {}})
    res = ls.parse(json.dumps(doc))
    cell, lines = ls.describe(res, None, False)
    ok("unrecognised suggest finding: amdgpu.new_advice" in cell and any(x.startswith("unrecognised suggest finding: amdgpu.new_advice") for x in lines),
       "an unknown finding kind is shown, not dropped", (cell, lines))
    ok("loom:unrecognised:amdgpu.new_advice" in ls.notes(res), "an unknown kind is a note too", ls.notes(res))
    bad = {"kind": ls.KIND, "schema_version": 0, "status": "available",
           "findings": [{"id": ls.CLIFF, "evidence": {"entries.rows[0].target_resources.residency.current_tier": 6}}]}
    cell, _ = ls.describe(ls.parse(json.dumps(bad)), None, True)
    ok("unrecognised suggest finding: amdgpu.residency_cliff (evidence shape)" in cell, "a cliff whose evidence is not the known shape", cell)
    for d, want in (([1, 2], "kind 'list'"), (dict(doc, kind="other"), "kind 'other'"), (dict(doc, schema_version=1), "schema 1"),
                    (dict(doc, status="unavailable", target_unavailable_reason="FAILED_PRECONDITION"), "status 'unavailable': FAILED_PRECONDITION"),
                    (dict(doc, findings="x"), "no findings list")):
        r = ls.parse(json.dumps(d))
        ok("error" in r and r["error"].startswith("unknown (") and want in r["error"], f"parse refuses {want}", r)
        ok(ls.notes(r) == ["loom:unknown"] and ls.describe(r)[0] == r["error"], f"{want}: note loom:unknown, cell the reason")
    r = ls.parse(json.dumps(dict(doc, provider="source", target_unavailable_reason="compile_status_not_ok", findings=[])))
    ok(ls.describe(r)[0] == "target analysis unavailable (compile_status_not_ok)",
       "a report whose target analysis failed (live g_silu_mul) says so, never 'no findings'", ls.describe(r))
    ok(ls.parse("not json")["error"] == "unknown (loom-compile-report suggest: output not JSON)", "non-JSON output", ls.parse("not json"))

    # ---------------------------------------------------------------- the process boundary: crash, missing, timeout, rc 0 garbage
    with tempfile.TemporaryDirectory() as t:
        rep = os.path.join(FIX, "g_attention_flash2_f16o.report.json")
        crash = fake(t, "crash", "echo 'loom-compile-report: error: argument_packets[9].packet_count: subgroup access groups model 64, got 56' >&2; exit 2")
        r = ls.run(rep, crash)
        ok(r.get("error", "").startswith("unknown (loom-compile-report suggest rc 2)") and "packet_count" in r["error"],
           "a crashing suggest (0.2.0 --include-experimental on flash2_f16o) is 'unknown (... rc 2)' with its message", r)
        r = ls.run(rep, os.path.join(t, "absent"))
        ok(r.get("error", "").startswith("unknown (loom-compile-report suggest could not start"), "a missing tool path: unknown, no exception", r)
        ok(ls.run(rep, None)["error"].startswith("unknown (loom-compile-report not found"), "no tool at all: unknown")
        r = ls.run(rep, fake(t, "slow", "sleep 5"), timeout=0.5)
        ok(r.get("error") == "unknown (loom-compile-report suggest timed out after 0.5 s)", "a hung suggest times out", r)
        r = ls.run(rep, fake(t, "garbage", "echo hello; exit 0"))
        ok(r.get("error") == "unknown (loom-compile-report suggest: output not JSON)", "exit 0 with non-JSON output", r)
        r = ls.run(rep, fake(t, "list", "echo '[1, 2]'; exit 0"))
        ok(r.get("error", "").startswith("unknown (") and "list" in r["error"], "exit 0 with a JSON list", r)
        good = fake(t, "good", f"cat {os.path.join(FIX, 'g_gemm_f16_acc_m256.suggest.json')}")
        r = ls.run(rep, good)
        ok(len(r.get("findings", [])) == 103, "a working tool's findings are read", r.get("error"))
        ok(ls.run(os.path.join(t, "nope.json"), good) == {"error": "no compile report"}, "no compile report: said")
        args = fake(t, "args", f"echo \"$@\" > {t}/argv; cat {os.path.join(FIX, 'g_gemm_f16_acc_m256.suggest.json')}")
        ls.run(rep, args)
        ok("--include-experimental" not in open(os.path.join(t, "argv")).read() and "--format=json" in open(os.path.join(t, "argv")).read(),
           "default run: --format=json, never --include-experimental", open(os.path.join(t, "argv")).read())

        # survey: once per report, an occupancy failure on one report leaves the others
        reps = {"a": rep, "b": rep, "c": os.path.join(t, "missing.json")}
        calls = []

        def occ_of(js):
            calls.append(js)
            if len(calls) == 1:
                raise KeyError("pool_units")
            return {"tier": 6, "next": 9, "q": 3, "needs": []}
        s = ls.survey(reps, good, occ_of, jobs=1)
        ok(sorted(s) == ["a", "b", "c"] and "cliff 4->5 unverified" in s["a"]["cell"] and "cliff 4->5 phantom (E533)" not in s["a"]["cell"],
           "an occupancy() that raises on one report: that cliff unverified", s["a"]["cell"])
        ok("phantom" in s["b"]["cell"] or "disagrees" in s["b"]["cell"], "the next report is still judged", s["b"]["cell"])
        ok(s["c"]["cell"] == "no compile report" and s["c"]["notes"] == [], "a kernel without a report: said, no note", s["c"])
        s = ls.survey({"a": rep}, crash, occ_of)
        ok(s["a"]["cell"].startswith("unknown (loom-compile-report suggest rc 2)") and s["a"]["notes"] == ["loom:unknown"],
           "a crash in the survey: unknown cell and note", s["a"])

    print(f"{'FAILED ' + str(fails) if fails else 'all ok'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
