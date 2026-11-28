#!/usr/bin/env python3
"""GEMM256_FLATADDR's kernel choice on a whole program (gemm_flataddr.apply).
Recurrence (HRX-like E512): the hook's pattern ended at `_m256"`, so F16O11's served g_gemm_f16_swiglu_m256_f16o (25 % of served
prefill) kept its 64-bit per-lane addresses (loop 248 instructions vs 209) while the build log listed the unserved
g_gemm_f16_swiglu_m256 as rewritten. Fixture: fixtures/gemm_f16o_m256.loom, the _f16o kernel and its gfx12 twin as live served
them (raw gen_gemm.py text is not the rewrite's input: later gen_llm passes set its anchors, E350).
Recurrence (HRX-like E744): the rewrite on the served swiglu GEMM (E514) cost +0.93 % of its served cycles (q258, 2 x ABBA, 4 starts
per arm), so g_gemm_f16_swiglu_m256_f16o leaves the rewrite set (gemm_flataddr.SKIP) and must stay as live pkg-e480 serves it; the
cases below fail if it is rewritten again, or if the skip widens to the kernels that keep the rewrite.
With a program path (a built package's program/llm.loom): checks every gfx11 f16 m256 GEMM there except KEEP_2D holds the flat views and
KEEP_2D's kernels hold none; with a second path (the live package's llm.loom), that KEEP_2D's kernels are byte-identical to it."""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


KEEP_2D = frozenset({"g_gemm_f16_swiglu_m256_f16o"})   # E744: served as live pkg-e480 serves it, without the rewrite


def program():
    return open(os.path.join(HERE, "fixtures", "gemm_f16o_m256.loom")).read()


def body(text, name):
    m = re.search(r'(?ms)^kernel\.def (?:target\(@\w+\) )?export\("' + name + r'"\).*?\n\} launch\(.*?\n\}\n', text)
    return m.group(0)


def refuses(fn, needle):
    try:
        fn(); return False
    except SystemExit as e:
        return needle in str(e)


def main():
    import gemm_flataddr
    fails, n = 0, 0
    def case(desc, ok):
        nonlocal fails, n
        n += 1; fails += not ok; print(("ok  " if ok else "FAIL") + "  " + desc)
    served = program()
    # the unserved g_gemm_f16_swiglu_m256 holds the same anchors as the _f16o copy (live pkg-e480 serves it rewritten): a copy under its
    # name stands in for a kernel that keeps the rewrite, beside the served _f16o kernel that leaves it (E744)
    text = served + body(served, "g_gemm_f16_swiglu_m256_f16o").replace("g_gemm_f16_swiglu_m256_f16o", "g_gemm_f16_swiglu_m256")
    try:
        out, done = gemm_flataddr.apply(text)
    except SystemExit as e:   # a skip widened past the served kernel refuses here: a failed case, not a test abort
        print(f"FAIL  apply refused the two-kernel program: {e}"); print("0/1 passed"); return 1
    case(f"the served _f16o kernel is not rewritten, g_gemm_f16_swiglu_m256 is ({done})", done == ["g_gemm_f16_swiglu_m256"])
    case("the served _f16o kernel is left byte-identical (as live pkg-e480 serves it, E744)",
         body(out, "g_gemm_f16_swiglu_m256_f16o") == body(text, "g_gemm_f16_swiglu_m256_f16o"))
    f = body(out, "g_gemm_f16_swiglu_m256")
    case("a kernel that keeps the rewrite has its in-loop loads on the flat views (no 2-D x / dq origin loads left)",
         "%w_sv = buffer.view" in f and "%x_sv = buffer.view" in f and "%dqan_origin" not in f and "%xv = vector.load %x_view[" not in f)
    case("a program whose only f16 m256 GEMM is skipped refuses (nothing rewritten)",
         refuses(lambda: gemm_flataddr.apply(served), "no g_gemm_f16"))
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
        want = [k for k in names if k not in KEEP_2D]
        flat = [k for k in names if "%w_sv = buffer.view" in body(prog, k) or "%x_sv = buffer.view" in body(prog, k)]
        full = [k for k in names if "%w_sv = buffer.view" in body(prog, k) and "%x_sv = buffer.view" in body(prog, k)]
        case(f"{os.path.basename(sys.argv[1])}: every gfx11 f16 m256 GEMM but KEEP_2D holds the flat views, KEEP_2D none "
             f"({len(flat)} of {len(names)} flat: {sorted(flat)})", bool(want) and full == want and flat == want)
        skipped = [k for k in names if k in KEEP_2D]
        case(f"{os.path.basename(sys.argv[1])}: the served swiglu kernel is present ({skipped})", skipped == sorted(KEEP_2D))
        if len(sys.argv) > 2:
            ref = open(sys.argv[2]).read()
            case(f"{os.path.basename(sys.argv[1])}: KEEP_2D's kernels are byte-identical to {sys.argv[2]}'s",
                 all(body(prog, k) == body(ref, k) for k in skipped))
    print(f"{n - fails}/{n} passed"); return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
