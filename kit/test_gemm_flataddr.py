#!/usr/bin/env python3
"""GEMM256_FLATADDR's kernel choice on a whole program (gemm_flataddr.apply).
Recurrence (HRX-like E512): the hook's pattern ended at `_m256"`, so F16O11's served g_gemm_f16_swiglu_m256_f16o (25 % of served
prefill) kept its 64-bit per-lane addresses (loop 248 instructions vs 209) while the build log listed the unserved
g_gemm_f16_swiglu_m256 as rewritten. Fixture: fixtures/gemm_f16o_m256.loom, the _f16o kernel and its gfx12 twin as live served
them (raw gen_gemm.py text is not the rewrite's input: later gen_llm passes set its anchors, E350).
With a program path (a built package's program/llm.loom): checks every served gfx11 f16 m256 GEMM there holds the flat views."""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def program():
    return open(os.path.join(HERE, "fixtures", "gemm_f16o_m256.loom")).read()


def body(text, name):
    m = re.search(r'(?ms)^kernel\.def (?:target\(@\w+\) )?export\("' + name + r'"\).*?\n\} launch\(.*?\n\}\n', text)
    return m.group(0)


def main():
    import gemm_flataddr
    fails, n = 0, 0
    def case(desc, ok):
        nonlocal fails, n
        n += 1; fails += not ok; print(("ok  " if ok else "FAIL") + "  " + desc)
    text = program()
    out, done = gemm_flataddr.apply(text)
    case(f"F16O11's served _f16o kernel is rewritten ({done})", done == ["g_gemm_f16_swiglu_m256_f16o"])
    f = body(out, "g_gemm_f16_swiglu_m256_f16o")
    case("the _f16o kernel's in-loop loads use the flat views (no 2-D x / dq origin loads left)",
         "%w_sv = buffer.view" in f and "%x_sv = buffer.view" in f and "%dqan_origin" not in f and "%xv = vector.load %x_view[" not in f)
    case("its gfx12 twin is left byte-identical", body(out, "g_gemm_f16_swiglu_m256_f16o_rdna4") == body(text, "g_gemm_f16_swiglu_m256_f16o_rdna4"))
    odd = text + body(text, "g_gemm_f16_swiglu_m256_f16o").replace("g_gemm_f16_swiglu_m256_f16o", "g_gemm_f16_store_m256_x2")
    try:
        gemm_flataddr.apply(odd); ok = False
    except SystemExit as e:
        ok = "g_gemm_f16_store_m256_x2" in str(e)
    case("an unknown g_gemm_f16_*m256* kernel stops the build (fail closed, not skipped)", ok)
    try:
        gemm_flataddr.apply("kernel.def export(\"g_other\") @g_other() {\n} launch() {\n}\n"); ok = False
    except SystemExit as e:
        ok = "no g_gemm_f16" in str(e)
    case("a program with no f16 m256 GEMM refuses", ok)
    if len(sys.argv) > 1:
        prog = open(sys.argv[1]).read()
        names = [m.group(2) for m in gemm_flataddr.KERNEL.finditer(prog) if m.group(1) != "rdna4"]
        flat = [k for k in names if "%w_sv = buffer.view" in body(prog, k) and "%x_sv = buffer.view" in body(prog, k)]
        case(f"{os.path.basename(sys.argv[1])}: every gfx11 f16 m256 GEMM holds the flat views ({len(flat)} of {len(names)}: {sorted(set(names) - set(flat))} not)",
             bool(names) and flat == names)
    print(f"{n - fails}/{n} passed"); return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
