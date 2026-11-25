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


def edited(k, **ev):
    """fixture K's suggest document with its residency_cliff evidence fields (by suffix) replaced"""
    d = json.load(open(os.path.join(FIX, k + ".suggest.json")))
    for x in d["findings"]:
        if x["id"] == ls.CLIFF:
            for key in list(x["evidence"]):
                for suf, v in ev.items():
                    if key.endswith("." + suf):   # dotted: 'next_better_tier' must not match 'reduction_units_to_next_better_tier'
                        x["evidence"][key] = v
    return d


def r2():
    """review r1 of suggest-1125 (7e9caf2): F1 fail closed past the process boundary, F2 'agrees' = Loom's tier IS the tier the cuts
    land on, F3 Loom 0.3.0's per-investigation grouping, F4 a summary of failed runs, F5 the two-cliff and empty-needs guards"""
    # F2: kernel_lint's occupancy() names the tier the cuts land on (q4_k k4096: VGPR <= 96 lands on 16); Loom's 14, 15 or 40 with the
    # same cut, or flash2's 9-cuts called 12, is not that tier: 'disagrees' (7e9caf2 said 'agrees' for every tier >= occupancy's next)
    for k, nt, cuts in (("g_mv_q4_k_k4096_t8_q8", 14, None), ("g_mv_q4_k_k4096_t8_q8", 15, None), ("g_mv_q4_k_k4096_t8_q8", 40, None),
                        ("g_attention_flash2_f16o", 12, None)):
        res = ls.parse(json.dumps(edited(k, next_better_tier=nt)))
        occ = lc.occupancy(lc.report_limit(os.path.join(FIX, k + ".report.json"))[4], 4)
        if k == "g_attention_flash2_f16o":   # give Loom exactly occupancy()'s cuts (VGPR <= 168, LDS <= 43008: they land on 9)
            res = ls.parse(json.dumps(edited(k, next_better_tier=nt, **{"rows[1].reduction_units_to_next_better_tier": 72})))
        cell, lines = ls.describe(res, occ, True)
        ok(f"->{nt} disagrees" in cell, f"F2: {k} with Loom's tier edited to {nt} (same cuts as occupancy()): disagrees", (cell, occ and occ.get("next")))
    occ = lc.occupancy(lc.report_limit(os.path.join(FIX, "g_mv_q4_k_k4096_t8_q8.report.json"))[4], 4)
    ok(occ.get("next") == 16, "F2: occupancy()'s next is the tier the cuts land on (kernel_lint tier_state)", occ)
    # F5: two cliffs in one report, a cliff without a constraint row: unrecognised, never judged
    d = json.load(open(os.path.join(FIX, "g_attention_flash2_f16o.suggest.json")))
    d["findings"] += [x for x in d["findings"] if x["id"] == ls.CLIFF]
    ok("residency_cliff (evidence shape)" in ls.describe(ls.parse(json.dumps(d)), None, True)[0], "F5 (R4): two cliffs in one report are unrecognised")
    d = json.load(open(os.path.join(FIX, "g_attention_flash2_f16o.suggest.json")))
    for x in d["findings"]:
        if x["id"] == ls.CLIFF:
            x["evidence"] = {k: v for k, v in x["evidence"].items() if not k.startswith("residency_constraints")}
    ok("residency_cliff (evidence shape)" in ls.describe(ls.parse(json.dumps(d)), None, True)[0],
       "F5 (R7): a cliff naming no resource to cut is unrecognised, not 'phantom' or 'agrees' on empty cuts")
    # F3: Loom 0.3.0 keeps one example per investigation with match_count, under the same kind and schema 0
    for k, want in (("g_attention_flash2_f16o", "expand x21"), ("g_gemm_f16_acc_m256", "expand x101")):
        res = ls.parse(open(os.path.join(FIX, k + ".v030.suggest.json")).read())
        cell, lines = ls.describe(res, None, False)
        ok(want in cell, f"F3: 0.3.0 {k}: counts weighted by match_count ({want}, as 0.2.0 lists them)", cell)
        ok(any("examples" in x for x in lines if x.startswith("reduce expansion")),
           f"F3: 0.3.0 {k}: a figure summed over examples says it covers examples, not every match", lines)
    with tempfile.TemporaryDirectory() as t:
        # F1: past the process boundary (review probes): non-UTF-8 output, lds_bank_service evidence as a list, an unknown id whose
        # action is a dict. Each was an exception that ended kernel_lint (7e9caf2); now 'unknown (...)' for that kernel only.
        rep = os.path.join(FIX, "g_gemm_f16_acc_m256.report.json")
        g = json.load(open(os.path.join(FIX, "g_gemm_f16_acc_m256.suggest.json")))
        ev = json.loads(json.dumps(g)); [x.update(evidence=list(x["evidence"].items())) for x in ev["findings"] if x["id"] == "amdgpu.lds_bank_service"]
        ad = json.loads(json.dumps(g)); ad["findings"].append({"id": "amdgpu.new", "action": {"text": "x"}, "evidence": {}})
        json.dump(ev, open(os.path.join(t, "ev.json"), "w")); json.dump(ad, open(os.path.join(t, "ad.json"), "w"))
        tools = {"nonutf8": fake(t, "nonutf8", r'printf "\377\376 not utf8"'), "evlist": fake(t, "evlist", f"cat {t}/ev.json"),
                 "actiondict": fake(t, "actiondict", f"cat {t}/ad.json"), "good": fake(t, "good", f"cat {os.path.join(FIX, 'g_gemm_f16_acc_m256.suggest.json')}")}
        for name in ("nonutf8", "evlist", "actiondict"):
            try:
                s = ls.survey({"a": rep, "b": rep}, tools[name], None, jobs=1)
            except Exception as e:   # noqa: BLE001
                s = {"raised": f"{type(e).__name__}: {e}"}
            want = "unrecognised suggest finding: amdgpu.new" if name == "actiondict" else "unknown ("
            ok("raised" not in s and s["a"]["cell"].startswith(want) if name != "actiondict" else "raised" not in s and want in s["a"]["cell"],
               f"F1: a {name} document gives '{want}' for that kernel and the survey completes", s.get("raised") or s["a"]["cell"])
        ok(ls.run(rep, tools["nonutf8"]).get("error") == "unknown (loom-compile-report suggest: output not UTF-8)", "F1: non-UTF-8 named as such",
           ls.run(rep, tools["nonutf8"]))
        # F4: one summary line; the tool and its version on it
        s = ls.survey({"a": rep, "b": rep, "c": os.path.join(t, "none.json")}, tools["good"], None)
        ok(ls.summary(s, tools["good"]).startswith("loom suggest (") and ls.summary(s, tools["good"]).endswith("0 of 2 runs unknown"),
           "F4: summary counts runs (a kernel without a report is not a run)", ls.summary(s, tools["good"]))
        s = ls.survey({"a": rep, "b": rep}, tools["nonutf8"], None)
        ok(ls.summary(s, tools["nonutf8"]).endswith("every run failed (unknown (loom-compile-report suggest: output not UTF-8))"),
           "F4: every run failed: said, with the first reason", ls.summary(s, tools["nonutf8"]))
        s["b"] = ls.survey({"b": rep}, tools["good"], None)["b"]
        ok(ls.summary(s, tools["good"]).endswith("1 of 2 runs unknown"), "F4: N of M runs unknown", ls.summary(s, tools["good"]))
        os.makedirs(os.path.join(t, "bin")); os.makedirs(os.path.join(t, "lib", "python3.12", "site-packages", "hrx_loom-0.3.0.dist-info"))
        tp = fake(os.path.join(t, "bin"), "loom-compile-report", "exit 0")
        ok(ls.version(tp) == "hrx-loom 0.3.0", "F3: the tool's hrx-loom version from its venv", ls.version(tp))
        ok(ls.version(fake(t, "other", "exit 0")) == "version unknown", "F3: a tool outside a venv: version unknown", ls.version(fake(t, "other", "exit 0")))


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
        ok(f"cliff {t} unverified" in cell and any("a bound, not a measured tier" in x for x in lines),
           f"{k}: a workgroup whose waves are not a multiple of the SIMDs has a bound tier (kernel_lint's ceil rule): Loom's cliff "
           "against it is 'unverified', never 'agrees'", (cell, lines))
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

    r2()
    print(f"{'FAILED ' + str(fails) if fails else 'all ok'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
