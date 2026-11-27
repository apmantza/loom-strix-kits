#!/usr/bin/env python3
"""GEMM_WAVE_SKIP's rewrite on a whole program (gemm_wave_skip.apply), static checks; the numbers are the GPU kernel test's
(make_kernel_tests.py 3a'''', @wave_skip_*: every row equal at atol 0 with and without padding rows; mutants in HRX-like q236).
Recurrences named per case: the pass is a text rewrite of generated code (rules section 4: a regex pass finds code by its exact shape
and silently skips new shapes), so each structural guard below is one way a later generator change could make it unsafe:
a barrier or a store pulled into the skipped region (a wave that skips would miss a workgroup barrier: a hang, or an output row not
written), an MMA left outside the region (half the work skipped), a region value read after the region (undefined on the skip path),
or a new m256 kernel name that is neither rewritten nor skipped.
Fixture: fixtures/gemm_f16o_m256.loom (F16O11's served swiglu kernel and its gfx12 twin), run through GEMM256_FLATADDR first, the
order gen_llm applies them. With a program path (a built package's program/llm.loom): checks every served gfx11 f16 m256 GEMM there
holds the wave skip."""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def body(text, name):
    m = re.search(r'(?ms)^kernel\.def (?:target\(@\w+\) )?export\("' + name + r'"\).*?\n\} launch\(.*?\n\}\n', text)
    return m.group(0)


def refuses(fn, needle):
    try:
        fn(); return False
    except SystemExit as e:
        return needle in str(e)


def main():
    import gemm_flataddr, gemm_wave_skip
    fails, n = 0, 0
    def case(desc, ok):
        nonlocal fails, n
        n += 1; fails += not ok; print(("ok  " if ok else "FAIL") + "  " + desc)
    text, _ = gemm_flataddr.apply(open(os.path.join(HERE, "fixtures", "gemm_f16o_m256.loom")).read())
    out, done = gemm_wave_skip.apply(text)
    case(f"the served _f16o kernel is rewritten ({done})", done == ["g_gemm_f16_swiglu_m256_f16o"])
    k0, k1 = body(text, "g_gemm_f16_swiglu_m256_f16o"), body(out, "g_gemm_f16_swiglu_m256_f16o")
    case("its gfx12 twin is left byte-identical", body(out, "g_gemm_f16_swiglu_m256_f16o_rdna4") == body(text, "g_gemm_f16_swiglu_m256_f16o_rdna4"))
    loop = k1[k1.index("= scf.for %step"):]
    r0 = loop.index("scf.if %wsk_live"); region = loop[r0:loop.index("    } else {", r0)]
    case("every MMA is inside the skipped region", region.count("= vector.mma ") == k0.count("= vector.mma ") > 0)
    case("no barrier, store or global view inside the region",
         not any(f in region for f in ("kernel.barrier", "vector.store", "view.store", "buffer.view")))
    case("the K loop keeps both barriers outside the region", loop[:loop.index("\n  }\n")].count("kernel.barrier<workgroup>") == 2)
    case("the predicate is a subgroup reduce over both of the lane's rows (wave_row + lane, + 32)",
         "kernel.subgroup.reduce<maxnumf> %wsk_m" in k1 and "%wsk_r0 = index.add %row_origin, %wave_row" in k1 and "%wsk_rb_v = index.add %wsk_ra_v, %wsk_c32" in k1
         and "%wsk_m = scalar.maxnumf %wsk_aa, %wsk_ab : f32" in k1)
    # removing the inserted lines gives back the original kernel's ops: the rewrite only wraps (same ops, same order)
    def ops(t):
        return [ln.strip() for ln in t.split("\n") if ln.strip() and "%wsk_" not in ln and ln.strip() not in ("} else {", "}")]
    case("outside the inserted lines the kernel's ops are unchanged, in order (live waves run the same ops)",
         [o for o in ops(k1) if not o.startswith("scf.yield")] == [o for o in ops(k0) if not o.startswith("scf.yield")])
    case("a rewritten kernel refuses a second rewrite", refuses(lambda: gemm_wave_skip.wave_skip(k1, "x"), "already rewritten"))
    # structural guards: each mutation of the input must stop the build
    bar = k0.replace("    %rv00_1 = ", "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n    %rv00_1 = ", 1)
    case("a barrier inside the region stops the build", bar != k0 and refuses(lambda: gemm_wave_skip.wave_skip(bar, "x"), "not pure fragment"))
    j = k0.index("\n", k0.index("    kernel.barrier<workgroup>", k0.index("= vector.mma "))) + 1
    late = k0[:j] + "    %late = vector.mulf %l0_0, %l0_0 : vector<16xf16>\n" + k0[j:]
    case("a region value read after the region stops the build", refuses(lambda: gemm_wave_skip.wave_skip(late, "x"), "used after it"))
    j2 = k0.index("\n", k0.index("= scf.for %step")) + 1
    early = k0[:j2] + "    %early = vector.mma %init16a, %init16b, %init : vector<16xf16>, vector<16xf16>, vector<8xf32>\n" + k0[j2:]
    case("an MMA outside the region stops the build", refuses(lambda: gemm_wave_skip.wave_skip(early, "x"), "outside the region"))
    nowave = k0.replace("  %wave_row = index.assume %wave_row_value [range(%wave_row_value, 0, 192)] : index\n",
                        "  %wave_row = index.assume %wave_row_value [range(%wave_row_value, 0, 96)] : index\n")
    case("a kernel whose waves are not 64-row waves of a 256-row tile stops the build", refuses(lambda: gemm_wave_skip.wave_skip(nowave, "x"), "anchor"))
    odd = text + k0.replace("g_gemm_f16_swiglu_m256_f16o", "g_gemm_f16_store_m256_x2")
    case("an unknown g_gemm_f16_*m256* kernel stops the build (fail closed, not skipped)",
         refuses(lambda: gemm_wave_skip.apply(odd), "g_gemm_f16_store_m256_x2"))
    case("a program with no f16 m256 GEMM refuses",
         refuses(lambda: gemm_wave_skip.apply("kernel.def export(\"g_other\") @g_other() {\n} launch() {\n}\n"), "no g_gemm_f16"))
    if len(sys.argv) > 1:
        prog = open(sys.argv[1]).read()
        names = [m.group(2) for m in gemm_wave_skip.KERNEL.finditer(prog) if m.group(1) != "rdna4"]
        have = [k for k in names if "scf.if %wsk_live" in body(prog, k)]
        case(f"{os.path.basename(sys.argv[1])}: every gfx11 f16 m256 GEMM holds the wave skip ({len(have)} of {len(names)})", bool(names) and have == names)
    print(f"{n - fails}/{n} passed"); return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
