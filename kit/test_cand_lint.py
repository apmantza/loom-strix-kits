#!/usr/bin/env python3
"""common/cand_lint.py (a candidate's post-build lint against a base package) on kernel_lint rows of synthetic listings and on
binary_gate-shaped rows. Recurrences it prevents:
  - HRX-like E520: STAGE_LIVE x FLATADDR put an s_waitcnt vmcnt(0) before flash2's first WMMA that retired the V-tile loads meant
    to be in flight (kernel_lint pf kept -> drain); the binary gate passed and only hand-built pair packages found it: a CHANGED
    kernel whose prefetch now drains must fail the candidate;
  - the same lint judging kernels the gate did not report changed (a base-vs-base drift or a stale code object) or skipping one it
    did: only CHANGED rows are judged, and every one of them must have been linted on both sides (fail closed);
  - a gate that could not establish the changed set (a kernel that did not compile: binary_gate ERROR) passing: fail closed;
  - review r1 F2: a loss with any other shape passing (the wait one WMMA later, a partial vmcnt(N) wait, the load sunk past the
    WMMAs): every loss of the base's prefetch fails; a new spill or new scratch fails without --allow (F2b);
  - E521: a kernel_lint 'cov' drop alone (instructions, no lost MMA cover) is printed, not failed;
  - verify r2 V1: a waiver wider than the accepted loss ('--accept flash2' passed E520's whole clash): a pin passes a kernel only
    when the pins equal its loss set, every pin must match, malformed pins are refused (the state table in HANDOFF r3);
  - verify r2 V2: an already-exposed load decaying further with the exposed count unchanged ([3, 4] -> [0, 4]);
  - verify r2 V6: the cover walk not following a load across the back-edge, or stopping before a deep pipeline's wait;
  - verify r2 V13: a NEW kernel with scratch passing;
  - verify r3 A: a 'loads' pin hiding an exposure (the exposed count is compared at any load count); B: a spill / flat load or a
    returning atomic missing from the vmcnt queue, so a vmcnt(1) after it looked like it kept the prefetch;
  - pairs taking pins (coordinator r3): the residue's sets pass and every other set fails on the unmatched pin, so pairs names the
    singles; pairs refuses --accept / LINT_ACCEPT and clears LINT_ACCEPT for its builds;
  - binary_gate's own failures not judged with --allow (R6); pairs taking a failed build or a missing verdict as built (R8, F8);
  - the pairwise mode naming a pair whose failure a single switch already explains, or missing the pair: the first failing set is
    the smallest one none of whose subsets fails.
usage: python3 kit/test_cand_lint.py   (no compiler, no GPU)"""
import json
import os
import sys
import tempfile

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
    return cl.lint_listing(name, *listing(body))


LOAD = "global_load_b128 v[40:43], v[1:2], off"
# kept: the next tile's load is issued and the WMMAs run while it is in flight; its wait comes after them
KEPT = [LOAD] + [WMMA] * 4 + ["s_waitcnt vmcnt(0)", "v_mov_b32 v50, v40"]
# drain (E520's shape): a full wait between the load and the first WMMA retires it before the multiplies
DRAIN = [LOAD, "s_waitcnt vmcnt(0)", "v_mov_b32 v50, v40"] + [WMMA] * 4
# review r1 F2: prefetch losses that keep pf 'kept' or read '-' (the reviewer's drain_moved.py shapes)
LOSSES = {
    "wait after the 1st WMMA": [LOAD, WMMA, "s_waitcnt vmcnt(0)", "v_mov_b32 v50, v40"] + [WMMA] * 3,
    "vmcnt(1) partial wait": [LOAD, "global_load_b128 v[44:47], v[1:2], off", "s_waitcnt vmcnt(1)", "v_mov_b32 v50, v40"] + [WMMA] * 4,
    "load moved after the WMMAs": [WMMA] * 4 + [LOAD, "s_waitcnt vmcnt(0)", "v_mov_b32 v50, v40"],
}
SPILL = [LOAD, "scratch_load_b128 v[60:63], off, s0 offset:16"] + [WMMA] * 4 + ["s_waitcnt vmcnt(0)", "v_mov_b32 v50, v40"]
# verify r2 V6: the next tile's load is issued late in the iteration and retired after the next iteration's first WMMA (cover
# 2 + 1 = 3: only a walk across the back-edge sees the 1); DEEP: retired two iterations later by vmcnt(2) (cover 2 + 2 + 2 = 6)
BACKEDGE = [WMMA, "s_waitcnt vmcnt(0)", "v_mov_b32 v50, v40", WMMA, LOAD, WMMA, WMMA]
DEEP = [LOAD, WMMA, WMMA, "s_waitcnt vmcnt(2)", "v_mov_b32 v50, v40"]

# the gate's state table (HANDOFF r3): a base kernel row and one candidate per loss kind, with the loss it must report
BASE = {"pf": "kept", "covers": [0, 4, 4], "loop_mm": 4, "scratch": 0, "cover_min": 5, "in_flight": 0}
KINDS = {
    "pf": (dict(BASE, pf="drain"), {}, {("pf", "kept", "drain")}),
    "pf '-'": (dict(BASE, pf="-"), {}, {("pf", "kept", "none")}),
    "exposed": (dict(BASE, covers=[0, 0, 4]), {}, {("exposed", "1", "2")}),
    "cover rank": (dict(BASE, covers=[0, 4]), {"covers": [3, 4]}, {("cover0", "3", "0")}),     # verify r2 V2
    "loads": (dict(BASE, covers=[0, 4, 4, 4]), {}, {("loads", "3", "4")}),
    "mma": (dict(BASE, loop_mm=8), {}, {("mma", "4", "8")}),
    "lost loop": (dict(BASE, covers=None, loop_mm=None), {}, {("mma", "4", "none")}),
    "spill": (dict(BASE, scratch=2), {}, {("spill", "0", "2")}),
}


# verify r3 B: a spill load (or a returning atomic) issued after the prefetch load takes a vmcnt entry, so a vmcnt(1) right after
# it retires the prefetch before the WMMAs (cover 0); a walk that does not queue it sees cover 4 and only the spill
def queued(op):
    return [LOAD, op, "s_waitcnt vmcnt(1)", "v_mov_b32 v50, v40"] + [WMMA] * 4


def pin(k, ls):
    return cl.parse_pins([",".join(f"{k}:{a}:{x}->{y}" for a, x, y in sorted(ls))])




def gate_row(k, verdict, fail=()):
    return {"kernel": k, "verdict": verdict, "fail": list(fail)}


def main():
    kept, drain = row("k", KEPT), row("k", DRAIN)
    ok(kept["pf"] == "kept" and drain["pf"] == "drain", f"the synthetic listings read pf kept / drain ({kept['pf']}, {drain['pf']})")

    lines, f = cl.judge([gate_row("k", "CHANGED")], {"k": kept}, {"k": drain})
    ok(any("pf kept->drain" in x for x in f), f"E520: a CHANGED kernel whose prefetch drains against the base fails ({f})")
    lines, f = cl.judge([gate_row("k", "CHANGED")], {"k": kept}, {"k": kept})
    ok(f == [] and any("ok" in x for x in lines), f"a CHANGED kernel that keeps its prefetch passes ({f})")
    lines, f = cl.judge([gate_row("k", "CHANGED")], {"k": drain}, {"k": drain})
    ok(f == [], f"a drain the base already had is inherited, not the candidate's delta ({f})")

    low = dict(kept, cover_min=10)
    lines, f = cl.judge([gate_row("k", "CHANGED")], {"k": dict(kept, cover_min=86)}, {"k": low})
    ok(f == [] and any("86->10" in x for x in lines), f"E521: a cover drop without a drain is printed, not failed ({f}; {lines})")

    rows = [gate_row("a", "same"), gate_row("b", "text-only"), gate_row("c", "CHANGED"), gate_row("d", "MISSING")]
    lines, f = cl.judge(rows, {"a": kept, "b": kept, "c": kept}, {"a": drain, "b": drain, "c": kept})
    ok(f == [] and not any(" a:" in x or (" b:" in x and "pf" in x) for x in lines),
       f"only the gate's CHANGED kernels are judged: same / text-only / missing ones are not, whatever their code objects say ({f})")
    lines, f = cl.judge([gate_row("c", "CHANGED")], {"c": kept}, {})
    ok(len(f) == 1 and "not linted" in f[0], f"a CHANGED kernel without a candidate code object fails closed ({f})")
    lines, f = cl.judge([gate_row("c", "CHANGED")], {}, {"c": kept})
    ok(len(f) == 1 and "not linted" in f[0], f"a CHANGED kernel without a base code object fails closed ({f})")
    lines, f = cl.judge([gate_row("e", "ERROR", ["did not compile: x"])], {}, {})
    ok(len(f) == 1 and "unknown" in f[0], f"a binary_gate ERROR (the changed set is unknown) fails closed ({f})")
    lines, f = cl.judge([gate_row("n", "NEW")], {}, {"n": drain})
    ok(f == [] and any("NEW" in x and "only scratch judged" in x for x in lines), f"a NEW kernel has no base: printed, not judged ({lines})")

    for n, body in LOSSES.items():
        c = row("k", body)
        lines, f = cl.judge([gate_row("k", "CHANGED")], {"k": kept}, {"k": c})
        ok(len(f) >= 1, f"review F2: {n} (pf {c['pf']}, covers {c['covers']}) fails against a kept base ({f})")
    nocov = lambda r: dict(r, covers=None, loop_mm=None)   # the exposure rule silent: pf alone must decide
    for pf in ("drain", "-"):
        f = cl.losses(gate_row("k", "CHANGED"), nocov(kept), dict(nocov(kept), pf=pf))
        ok(("pf", "kept", {"-": "none"}.get(pf, pf)) in f, f"review F2: pf kept -> {pf} fails on its own ({f})")
    lines, f = cl.judge([gate_row("k", "CHANGED")], {"k": kept}, {"k": row("k", SPILL)})
    ok(any("spill" in x for x in f), f"review F2b: a new spill in the hot loop fails without --allow ({f})")
    g = dict(gate_row("k", "CHANGED"), base={"scratch": 0}, cand={"scratch": 64})
    lines, f = cl.judge([g], {"k": kept}, {"k": kept})
    ok(any("scratch 0->64" in x for x in f), f"review F2b: binary_gate's new scratch fails without --allow ({f})")
    g = dict(gate_row("k", "CHANGED"), base={"scratch": 64}, cand={"scratch": 64})
    ok(cl.judge([g], {"k": kept}, {"k": kept})[1] == [], "scratch the base already had is not the candidate's")

    ok(row("k", BACKEDGE)["covers"] == [3], f"V6: a load retired in the next iteration counts that iteration's MMAs "
       f"({row('k', BACKEDGE)['covers']})")
    ok(row("k", DEEP)["covers"] == [6], f"V6: a load retired two iterations later by vmcnt(2) gets its full cover ({row('k', DEEP)['covers']})")

    # the state table: rows = loss kinds, columns = no pin / exact pin / pin for another loss / pin matching nothing
    other = {("spill", "0", "9")}
    for name, (c, bx, want) in KINDS.items():
        b = dict(BASE, **bx)
        got = set(cl.losses(gate_row("k", "CHANGED"), b, c))
        ok(got == want, f"state table {name}: the loss is {want} ({got})")
        run = lambda pins: cl.judge([gate_row("k", "CHANGED"), gate_row("z", "CHANGED")], {"k": b, "z": BASE}, {"k": c, "z": BASE}, pins)[1]
        ok(run(None) != [], f"state table {name} x no pin: FAIL")
        ok(run(pin("k", want)) == [], f"state table {name} x exact pin: PASS ({run(pin('k', want))})")
        ok(run(pin("k", other)) != [], f"state table {name} x pin for another loss: FAIL")
        ok(run(pin("k", want | other)) != [], f"state table {name} x its pin plus another: FAIL")
        ok(run({**pin("k", want), **pin("z", want)}) != [], f"state table {name} x a pin matching nothing (kernel z, no loss): FAIL")
    g = dict(gate_row("k", "CHANGED"), base={"scratch": 0}, cand={"scratch": 64})
    ok(cl.judge([g], {"k": BASE}, {"k": BASE}, pin("k", {("scratch", "0", "64")}))[1] == [], "state table scratch x exact pin: PASS")
    ok(cl.judge([g], {"k": BASE}, {"k": BASE}, pin("k", {("scratch", "0", "32")}))[1] != [], "state table scratch x another value: FAIL")
    n = dict(gate_row("n", "NEW"), cand={"scratch": 64})
    ok(any("scratch new->64" in x for x in cl.judge([n], {}, {"n": BASE})[1]), "V13: a NEW kernel with scratch fails")
    ok(cl.judge([n], {}, {"n": BASE}, pin("n", {("scratch", "new", "64")}))[1] == [], "state table NEW scratch x exact pin: PASS")
    ok(cl.judge([dict(gate_row("n", "NEW"), cand={"scratch": 0})], {}, {"n": BASE})[1] == [], "a NEW kernel without scratch: not judged")
    two = dict(BASE, pf="drain", covers=[0, 0, 4])
    both = {("pf", "kept", "drain"), ("exposed", "1", "2")}
    ok(cl.judge([gate_row("k", "CHANGED")], {"k": BASE}, {"k": two}, pin("k", {("exposed", "1", "2")}))[1] != [],
       "V1: two losses, one pinned (E520's clash with E521's pin): FAIL")
    ok(cl.judge([gate_row("k", "CHANGED")], {"k": BASE}, {"k": two}, pin("k", both))[1] == [], "two losses, both pinned: PASS")
    ok(cl.judge([gate_row("k", "CHANGED")], {"k": BASE}, {"k": dict(BASE, covers=[0, 0, 0])}, pin("k", {("exposed", "1", "2")}))[1] != [],
       "a pin of exposed 1->2 does not pass exposed 1->3")
    ok(any("PINNED" in x for x in cl.judge([gate_row("k", "CHANGED")], {"k": BASE}, {"k": two}, pin("k", both))[0]), "a pinned kernel is printed PINNED")
    for bad in (".", "flash2", "^g_attention_flash2_partial(_f16o)?$", "k:exposed:1", "k:foo:1->2", "k.*:exposed:1->3", "k:exposed:1->3 x"):
        try:
            cl.parse_pins([bad])
            ok(False, f"V1: malformed pin {bad!r} refused")
        except ValueError:
            ok(True, f"V1: malformed pin {bad!r} refused")
    # verify r3 A: a changed load count still compares the exposed count, so a 'loads' pin cannot hide an exposure
    a_b, a_c = dict(BASE, covers=[4], loop_mm=4), dict(BASE, covers=[1, 4], loop_mm=4)
    la = set(cl.losses(gate_row("k", "CHANGED"), a_b, a_c))
    ok(la == {("loads", "1", "2"), ("exposed", "0", "1")}, f"verify r3 A: +1 load and L1 exposed 4->1 report loads and exposed ({la})")
    ok(cl.judge([gate_row("k", "CHANGED")], {"k": a_b}, {"k": a_c}, pin("k", {("loads", "1", "2")}))[1] != [],
       "verify r3 A: pinning k:loads:1->2 alone FAILS")
    ok(cl.judge([gate_row("k", "CHANGED")], {"k": a_b}, {"k": a_c}, pin("k", la))[1] == [], "both losses pinned: PASS")
    # coordinator r4 decision (fail closed): sorted covers pair from the lowest up whatever the load counts, so an already-exposed
    # load's decay beside an added load is a loss of its own (cover0 3->0), not hidden behind a 'loads 2->3' pin
    lc = set(cl.losses(gate_row("k", "CHANGED"), dict(BASE, covers=[3, 4]), dict(BASE, covers=[0, 4, 4])))
    ok(lc == {("loads", "2", "3"), ("cover0", "3", "0")}, f"ranks pair from the lowest up across a load-count change ({lc})")
    ok(cl.judge([gate_row("k", "CHANGED")], {"k": dict(BASE, covers=[3, 4])}, {"k": dict(BASE, covers=[0, 4, 4])},
                pin("k", {("loads", "2", "3")}))[1] != [], "a loads pin alone no longer accepts the decay")
    for name, op, want in (("spill load", "scratch_load_b32 v60, off, s0 offset:16", [0]),
                           ("flat load", "flat_load_b32 v60, v[2:3]", [0]),
                           ("returning atomic", "global_atomic_add_u32 v60, v2, v3, s[0:1] glc", [0]),
                           ("non-returning atomic (vscnt, not vmcnt)", "global_atomic_add_u32 v2, v3, s[0:1]", [4])):
        r_ = row("k", queued(op))
        ok(r_["covers"] == want, f"verify r3 B: a {name} before vmcnt(1) is {'' if want == [0] else 'not '}a vmcnt entry (covers {r_['covers']})")
    sp = set(cl.losses(gate_row("k", "CHANGED"), kept, row("k", queued("scratch_load_b32 v60, off, s0 offset:16"))))
    ok(("exposed", "0", "1") in sp and ("spill", "0", "1") in sp, f"verify r3 B: the spill case reports the exposure, not only the spill ({sp})")
    ok(cl.parse_pins(["a:exposed:1:3,b:pf:kept:drain"]) == {"a": {("exposed", "1", "3")}, "b": {("pf", "kept", "drain")}},
       "the colon form KERNEL:KIND:FROM:TO needs no shell quoting")
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(HERE, "..", "common", "cand_lint.py"), "check", "/nonexistent/a", "/nonexistent/b",
                        "--out", os.devnull, "--accept", "."], capture_output=True, text=True)
    ok(r.returncode == 2 and "malformed pin" in r.stderr, f"V1: the CLI refuses a malformed pin before the gate runs, exit 2 ({r.returncode})")
    # coordinator r3: pairs is a diagnostic; a pin there passes the residue's sets and fails every other (the singles were named)
    with tempfile.TemporaryDirectory() as t:
        envf = os.path.join(t, "env")
        with open(envf, "w") as fh:
            fh.write(f"WORK={t}/work\nSTATE={t}/state\nLOOM_KB_ROOT={t}\n")
        cli = [sys.executable, os.path.join(HERE, "..", "common", "cand_lint.py"), "pairs", "nokit", "base", "zz", "A=1", "B=1"]
        env = {k: v for k, v in os.environ.items() if k not in ("LINT_ACCEPT", "WORK", "STATE", "KIT_SRC")}
        env.update(LSK_ENV=envf, HOME=t)
        for how, extra, e in (("--accept", ["--accept", "k:exposed:1->3"], env), ("LINT_ACCEPT", [], dict(env, LINT_ACCEPT="k:exposed:1->3"))):
            r = subprocess.run(cli + extra, capture_output=True, text=True, env=e)
            ok(r.returncode == 2 and "pins judge one candidate" in r.stderr, f"pairs refuses pins given by {how}, exit 2 ({r.returncode})")
    seen = []
    old_env = os.environ.get("LINT_ACCEPT")
    os.environ["LINT_ACCEPT"] = "k:exposed:1->3"
    try:
        with tempfile.TemporaryDirectory() as w, open(os.devnull, "w") as quiet:
            def spy(kit, pkg, env):
                seen.append(env.get("LINT_ACCEPT"))
                return 1, "", ""
            old, sys.stdout = sys.stdout, quiet
            try:
                cl.pairs("kit", "base", "p", ["A=1", "B=1"], build=spy, work=w)
            finally:
                sys.stdout = old
    finally:
        if old_env is None:
            del os.environ["LINT_ACCEPT"]
        else:
            os.environ["LINT_ACCEPT"] = old_env
    ok(seen and all(x == "" for x in seen), f"pairs' builds run with LINT_ACCEPT cleared, not an inherited one ({seen})")
    ok(cl.parse_pins(["a:exposed:1->3,b:pf:kept->drain"]) == {"a": {("exposed", "1", "3")}, "b": {("pf", "kept", "drain")}},
       "pins are comma-separated")
    lines, f = cl.judge([gate_row("t", "text-only")], {}, {})
    ok(f == [] and any("text" in x and " t:" in x for x in lines), f"review F3: a text-only kernel is printed (nothing to lint) ({lines})")

    bf = ["x: changed and no --allow matches"]
    ok(any("binary_gate:" in x for x in cl.verdict([gate_row("k", "CHANGED")], bf, {"k": kept}, {"k": kept}, allow=["^k$"])[1]),
       "R6: with --allow, binary_gate's own failures are judged")
    ok(cl.verdict([gate_row("k", "CHANGED")], bf, {"k": kept}, {"k": kept})[1] == [],
       "without --allow, binary_gate's untouched-kernel failures are printed, not judged")

    with tempfile.TemporaryDirectory() as w:
        def fake(rc, fails=None):
            def build(kit, pkg, env):
                if fails is not None:
                    os.makedirs(os.path.join(w, pkg + ".lint"), exist_ok=True)
                    json.dump({"fails": fails(env["EXTRA"]), "lines": ["head"]}, open(os.path.join(w, pkg + ".lint", "verdict.json"), "w"))
                return rc(env["EXTRA"]), "", "boom"
            return build
        A, B = "A=1", "B=1"
        quiet = open(os.devnull, "w")
        old, sys.stdout = sys.stdout, quiet
        try:
            r_fail = cl.pairs("kit", "base", "p", [A, B], build=fake(lambda e: 1, lambda e: []), work=w)   # a stale verdict is left
            r_nov = cl.pairs("kit", "base", "q", [A, B], build=fake(lambda e: 0), work=w)
            r_pair = cl.pairs("kit", "base", "r", [A, B], build=fake(lambda e: 3 if e == "A=1 B=1" else 0,
                                                                     lambda e: ["x"] if e == "A=1 B=1" else []), work=w)
            r_ok = cl.pairs("kit", "base", "s", [A, B], build=fake(lambda e: 0, lambda e: []), work=w)
        finally:
            sys.stdout = old
        ok(r_fail == 2, f"R8: pairs stops with exit 2 on a build that failed ({r_fail})")
        ok(r_nov == 2, f"F8: pairs exits 2 (not 1) when a build left no verdict.json ({r_nov})")
        ok(r_pair == 1, f"pairs exits 1 when the pair fails and neither single does ({r_pair})")
        ok(r_ok == 0, f"pairs exits 0 when nothing fails ({r_ok})")

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
