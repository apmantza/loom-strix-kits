#!/usr/bin/env python3
"""common/binary_gate.py on real compiles of a fixture (fixtures/binary_gate.loom: bg_store and bg_swiglu share the inline
helper @bg_load, bg_other calls nothing) and on synthetic metrics. Recurrences it prevents:
  - a rewrite meant for one kernel reaching another through a shared helper unseen (E372: GEMM256_FLATADDR at kit 718c339 meant
    for the store GEMM also rewrote the swiglu GEMM's loads): a changed helper with --allow bg_store must flag bg_swiglu, although
    bg_swiglu's own text did not change (a kernel's text is not its code: Loom inlines helpers);
  - a switch that did not apply passing its step (the per-step "rewrites present" greps this replaces): an --allow that matches no
    changed kernel fails;
  - a kernel dropped by a switch, a new spill, a kernel that no longer compiles passing (fail closed);
  - an authored Low helper (`low.func.def target<core>(@rdna3) @f`, called through low.invoke) filed under the target @rdna3
    instead of @f (FLASH2_SMX2, 11-24): a changed helper then left its untargeted caller "same" (never compiled) and marked every
    @rdna3 kernel changed;
  - renames / comment edits that do not change code flagged (false reds teach people to route around a gate): text-only passes,
    and an unchanged closure is not compiled at all.
usage: python3 kit/test_binary_gate.py   (needs loom-compile: VENV from the environment or the machine env file)"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import binary_gate as bg  # noqa: E402

fails = 0


def ok(cond, what):
    global fails
    print(("ok  " if cond else "FAIL") + " " + what)
    fails += not cond


LOW = """amdgpu.target<gfx11-generic> @rdna3

low.func.def target<amdgpu.gfx11.generic.core>(@rdna3) @lowh(%a: reg<amdgpu.vgpr>) -> (reg<amdgpu.vgpr>) asm {
  %b = v_mov_b32 %a
  return %b
}

kernel.def export("k_uses") @k_uses(%n: index) {
  %y = low.invoke @lowh(%x) : (i32) -> (i32)
} launch(%n: index) {
}

kernel.def target(@rdna3) export("k_other") @k_other(%n: index) {
} launch(%n: index) {
}
"""


def low_helper_closure():
    """compile-free: a low.func.def helper is its own item, in its invoker's closure only"""
    a, b = bg.closures(LOW), bg.closures(LOW.replace("v_mov_b32 %a", "v_not_b32 %a"))
    ok("lowh" in bg.items(LOW) and a["k_uses"] != b["k_uses"] and a["k_other"] == b["k_other"],
       f"a changed low.func.def helper changes its low.invoke caller's closure only (items {sorted(bg.items(LOW))})")


def main():
    low_helper_closure()
    src = open(os.path.join(HERE, "fixtures", "binary_gate.loom")).read()
    helper_load = "  %x = view.load %v[%c0] : view<1xf32> -> f32\n  func.return %x : f32\n"
    assert src.count(helper_load) == 1 and src.count("%two = scalar.constant 2.0 : f32") == 1, "fixture text changed: update this test"
    with tempfile.TemporaryDirectory() as t:
        def run(cand_text, allow, base_text=src, all_=False):
            b, c = os.path.join(t, "base.loom"), os.path.join(t, "cand.loom")
            open(b, "w").write(base_text)
            open(c, "w").write(cand_text)
            rows, fl, n = bg.gate(b, c, allow, os.path.join(t, "out"), jobs=2, all_=all_)
            return {r["kernel"]: r["verdict"] for r in rows}, fl, n
        v, fl, n = run(src + "\n// a trailing comment\n", [])
        ok(not fl and n == 0 and set(v.values()) == {"same"}, f"identical programs (plus a comment): pass, nothing compiled ({n} compiles, {fl})")
        # E372 shape: the helper both GEMMs inline is rewritten (here: the load scaled by 2); only bg_store was meant
        rew = src.replace(helper_load, "  %x0 = view.load %v[%c0] : view<1xf32> -> f32\n  %x = scalar.addf %x0, %x0 : f32\n"
                          "  func.return %x : f32\n")
        v, fl, n = run(rew, ["^bg_store$"])
        ok(v.get("bg_swiglu") == "CHANGED" and any(f.startswith("bg_swiglu:") for f in fl) and v.get("bg_other") == "same",
           f"a rewritten shared helper with --allow bg_store flags bg_swiglu, whose own text is unchanged ({v}, {fl})")
        v, fl, n = run(rew, ["^bg_(store|swiglu)$"])
        ok(not fl, f"both callers allowed: pass ({fl})")
        own = src.replace("%two = scalar.constant 2.0 : f32", "%two = scalar.constant 3.0 : f32")
        v, fl, n = run(own, ["^bg_store$"])
        ok(not fl and v["bg_store"] == "CHANGED" and v["bg_swiglu"] == "same" and n == 2,
           f"a change to the allowed kernel's own text: pass, only that kernel compiled ({v}, {n} compiles)")
        v, fl, n = run(own, ["^bg_store$", "^bg_swiglu$"])
        ok(any("'^bg_swiglu$' matched no changed kernel" in f for f in fl), f"an --allow matching no changed kernel fails: the switch did not apply ({fl})")
        v, fl, n = run(src, ["^bg_store$"])
        ok(any("matched no changed kernel" in f for f in fl), f"a switch that changed nothing fails ({fl})")
        ren = src.replace("%one = scalar.constant 1.0 : f32", "%unit = scalar.constant 1.0 : f32").replace("view.store %one,", "view.store %unit,")
        v, fl, n = run(ren, [])
        ok(not fl and v["bg_other"] == "text-only", f"an SSA rename: text-only, pass ({v}, {fl})")
        a, z = src.index('kernel.def target(@rdna3) export("bg_other")'), len(src)
        v, fl, n = run(src[:a], [])
        ok(v.get("bg_other") == "MISSING" and any("bg_other: missing" in f for f in fl), f"a dropped kernel fails ({fl})")
        bad = own.replace("%r = scalar.mulf %x, %two : f32", "%r = scalar.mulf %x, %nothere : f32")
        v, fl, n = run(bad, ["^bg_store$"])
        ok(v["bg_store"] == "ERROR" and any("did not compile" in f for f in fl), f"a kernel that no longer compiles fails closed ({fl})")
        v, fl, n = run(rew, ["^bg_(store|swiglu)$"], all_=True)
        ok(not fl and v["bg_other"] == "text-only" and n == 6, f"--all compiles every kernel and finds the unchanged closure's code identical ({v}, {n})")
    # new scratch (no fixture kernel spills): synthetic metrics through the same judge
    base, cand = {"k": ("f", "a")}, {"k": ("f", "b")}
    m0 = {"isa": "x", "vgpr": 10, "sgpr": 10, "scratch": 0, "lds": 0}
    rows, fl = bg.judge(base, cand, {("base", "k"): m0, ("cand", "k"): dict(m0, isa="y", scratch=64)}, ["^k$"])
    ok(any("new scratch 64" in f for f in fl), f"new scratch in an allowed kernel fails ({fl})")
    rows, fl = bg.judge(base, cand, {("base", "k"): dict(m0, scratch=64), ("cand", "k"): dict(m0, isa="y", scratch=64)}, ["^k$"])
    ok(not fl, f"scratch the base already had is not new ({fl})")
    rows, fl = bg.judge({}, {"n": ("f", "a")}, {("cand", "n"): dict(m0, scratch=16)}, ["^n$"])
    ok(any("new scratch 16" in f for f in fl), f"a new kernel with scratch fails ({fl})")
    # --allow-missing (HRX-like q199: a drafter of another shape drops the z-lab drafter's k4096 matvecs, called only by
    # llm_dflash_propose): a declared removal passes, an undeclared one still fails, a declaration that matches nothing fails
    base2 = {"keep": ("f", "a"), "gone": ("f", "a"), "other": ("f", "a")}
    rows, fl = bg.judge(base2, {"keep": ("f", "a")}, {}, [], ["^gone$"])
    ok(any("other: missing" in f for f in fl) and not any(f.startswith("gone:") for f in fl), f"declared missing passes, undeclared fails ({fl})")
    rows, fl = bg.judge(base2, {"keep": ("f", "a"), "other": ("f", "a")}, {}, [], ["^gone$"])
    ok(not fl, f"only the declared kernel missing: pass ({fl})")
    rows, fl = bg.judge(base2, dict(base2), {}, [], ["^gone$"])
    ok(any("--allow-missing" in f for f in fl), f"a declaration matching no missing kernel fails closed ({fl})")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
