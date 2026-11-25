#!/usr/bin/env python3
"""common/cand_lint.py (a candidate's post-build lint against a base package) on kernel_lint rows of synthetic listings and on
binary_gate-shaped rows. Recurrences it prevents:
  - HRX-like E520: STAGE_LIVE x FLATADDR put an s_waitcnt vmcnt(0) before flash2's first WMMA that retired the V-tile loads meant
    to be in flight (kernel_lint pf kept -> drain); the binary gate passed and only hand-built pair packages found it: a CHANGED
    kernel whose prefetch now drains must fail the candidate;
  - the same lint judging kernels the gate did not report changed (a base-vs-base drift or a stale code object) or skipping one it
    did: only CHANGED rows are judged, and every one of them must have been linted on both sides (fail closed);
  - a gate that could not establish the changed set (a kernel that did not compile: binary_gate ERROR) passing: fail closed;
  - E521's fix failing: the fixed candidate keeps its prefetch but flash2_partial's minimum load cover stays low (86 -> 10, an
    accepted residue): a cover drop without a drain is printed, not failed;
  - the pairwise mode naming a pair whose failure a single switch already explains, or missing the pair: the first failing set is
    the smallest one none of whose subsets fails.
usage: python3 kit/test_cand_lint.py   (no compiler, no GPU)"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import cand_lint as cl  # noqa: E402
import kernel_lint as kl  # noqa: E402

fails = 0


def ok(cond, what):
    global fails
    print(("ok  " if cond else "FAIL") + " " + what)
    fails += not cond


WMMA = "v_wmma_f32_16x16x16_f16 v[0:7], v[8:15], v[16:23], v[0:7]"


def listing(body):
    """a loop over BODY (s_mov before it, the back-edge after it); addresses 4 bytes apart"""
    ins = ["s_mov_b32 s0, 0"] + body + ["s_add_u32 s0, s0, 1"]
    ins.append(f"s_branch {(1 - len(ins) - 1) & 0xFFFF}")
    return ins, [4 * i for i in range(len(ins))]


def row(name, body):
    return kl.lint_listing(name, *listing(body))


LOAD = "global_load_b128 v[40:43], v[1:2], off"
# kept: the next tile's load is issued and the WMMAs run while it is in flight; its wait comes after them
KEPT = [LOAD] + [WMMA] * 4 + ["s_waitcnt vmcnt(0)", "v_mov_b32 v50, v40"]
# drain (E520's shape): a full wait between the load and the first WMMA retires it before the multiplies
DRAIN = [LOAD, "s_waitcnt vmcnt(0)", "v_mov_b32 v50, v40"] + [WMMA] * 4


def gate_row(k, verdict, fail=()):
    return {"kernel": k, "verdict": verdict, "fail": list(fail)}


def main():
    kept, drain = row("k", KEPT), row("k", DRAIN)
    ok(kept["pf"] == "kept" and drain["pf"] == "drain", f"the synthetic listings read pf kept / drain ({kept['pf']}, {drain['pf']})")

    lines, f = cl.judge([gate_row("k", "CHANGED")], {"k": kept}, {"k": drain})
    ok(len(f) == 1 and "drain" in f[0], f"E520: a CHANGED kernel whose prefetch drains against the base fails ({f})")
    lines, f = cl.judge([gate_row("k", "CHANGED")], {"k": kept}, {"k": kept})
    ok(f == [] and any("ok" in x for x in lines), f"a CHANGED kernel that keeps its prefetch passes ({f})")
    lines, f = cl.judge([gate_row("k", "CHANGED")], {"k": drain}, {"k": drain})
    ok(f == [], f"a drain the base already had is inherited, not the candidate's delta ({f})")

    low = dict(kept, cover_min=10)
    lines, f = cl.judge([gate_row("k", "CHANGED")], {"k": dict(kept, cover_min=86)}, {"k": low})
    ok(f == [] and any("86->10" in x for x in lines), f"E521: a cover drop without a drain is printed, not failed ({f}; {lines})")

    rows = [gate_row("a", "same"), gate_row("b", "text-only"), gate_row("c", "CHANGED"), gate_row("d", "MISSING")]
    lines, f = cl.judge(rows, {"a": kept, "b": kept, "c": kept}, {"a": drain, "b": drain, "c": kept})
    ok(f == [] and not any(" a:" in x or " b:" in x for x in lines),
       f"only the gate's CHANGED kernels are judged: same / text-only / missing ones are not, whatever their code objects say ({f})")
    lines, f = cl.judge([gate_row("c", "CHANGED")], {"c": kept}, {})
    ok(len(f) == 1 and "not linted" in f[0], f"a CHANGED kernel without a candidate code object fails closed ({f})")
    lines, f = cl.judge([gate_row("c", "CHANGED")], {}, {"c": kept})
    ok(len(f) == 1 and "not linted" in f[0], f"a CHANGED kernel without a base code object fails closed ({f})")
    lines, f = cl.judge([gate_row("e", "ERROR", ["did not compile: x"])], {}, {})
    ok(len(f) == 1 and "unknown" in f[0], f"a binary_gate ERROR (the changed set is unknown) fails closed ({f})")
    lines, f = cl.judge([gate_row("n", "NEW")], {}, {"n": drain})
    ok(f == [] and any("NEW" in x and "not judged" in x for x in lines), f"a NEW kernel has no base: printed, not judged ({lines})")

    S, N, F = "FLASH2_STAGE_LIVE=1", "FLASH2_NOB3=1", "FLASH2_FLATADDR=1"
    sets = cl.switch_sets([S, N, F])
    ok(sets == [(S,), (N,), (F,), (S, N), (S, F), (N, F), (S, N, F)], f"three switches: singles, pairs, the whole set ({len(sets)} sets)")
    ok(cl.switch_sets([S, F]) == [(S,), (F,), (S, F)], "two switches: the pair is the whole set (built once)")
    e520 = {(S,): False, (N,): False, (F,): False, (S, N): False, (S, F): True, (N, F): False, (S, N, F): True}
    ok(cl.first_failures(e520) == [(S, F)], f"E520: the collapse first appears in STAGE_LIVE x FLATADDR ({cl.first_failures(e520)})")
    single = {**e520, (F,): True, (N, F): True}
    ok(cl.first_failures(single) == [(F,)], f"a single switch that fails alone is named, not its pairs ({cl.first_failures(single)})")
    ok(cl.first_failures({k: False for k in e520}) == [], "nothing fails: no set is named")
    print(f"{'FAILED' if fails else 'all passed'} ({fails} failures)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
