#!/usr/bin/env python3
"""GEMM_WAVE_SKIP's rewrite on a whole program (gemm_wave_skip.apply), static checks; the numbers are the GPU kernel test's
(make_kernel_tests.py 3a'''', @wave_skip_*: every row equal at atol 0 with and without padding rows; mutants in HRX-like q236).
Recurrences named per case: the pass is a text rewrite of generated code (rules section 4: a regex pass finds code by its exact shape
and silently skips new shapes), so each structural guard below is one way a later generator change could make it unsafe:
a barrier or a store pulled into the skipped region (a wave that skips would miss a workgroup barrier: a hang, or an output row not
written), an MMA left outside the region (half the work skipped), a region value read after the region (undefined on the skip path),
or a new m256 kernel name that is neither rewritten nor skipped.
Fixture: fixtures/gemm_f16o_m256.loom (F16O11's served swiglu kernel and its gfx12 twin) as generated: GEMM256_FLATADDR, which
gen_llm applies first, leaves this kernel alone since HRX-like E744 (gemm_flataddr.SKIP); the flat-lane x prefetch form is covered by
the _qgs fixture below (store_m256 and _qgs, both rewritten). With a program path (a built package's program/llm.loom): checks every served gfx11 f16 m256 GEMM there
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
    except Exception:   # a guard that crashes instead of refusing is a failed case, not a test abort
        return False


def main():
    import gemm_flataddr, gemm_wave_skip
    fails, n = 0, 0
    def case(desc, ok):
        nonlocal fails, n
        n += 1; fails += not ok; print(("ok  " if ok else "FAIL") + "  " + desc)
    text = open(os.path.join(HERE, "fixtures", "gemm_f16o_m256.loom")).read()
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
    # HRX-like E690 (review): every guard of the pass reds a case here when neutered (static-1127/wsk-fail/guard_mutants.py). Each case
    # changes one thing in the fixture kernel and names the refusal it must get; the recurrence is the guard's own comment in the pass.
    def refused(desc, txt, needle, fn=None):
        case(desc, txt != k0 and refuses(fn or (lambda: gemm_wave_skip.wave_skip(txt, "x")), needle))
    # F1: the region admits five op forms only (the deny-list accepted these; review failopen.py)
    rv = "    %rv00_1 = "
    for what, line in (("an LDS fragment.store", "    vector.fragment.store<result> %init, %x_stage_view[%wave_row, %c0] shape [%c16, %c16] : vector<8xf32>, view<256x72xf16>"),
                       ("a global atomic", "    %ax = buffer.atomic.add %y[%origin], %one : f32"),
                       ("a view.store", "    view.store %one, %so_view[%c0, %c0] : f32, view<[%rows]x2xf32>"),
                       ("a vector.store", "    vector.store %rv00_0, %x_stage_view[%wave_row, %c0] : vector<16xf16>, view<256x72xf16>"),
                       ("a nested scf op", "    %sx = scf.select %tile_live, %c0, %c1 : index"),
                       ("a buffer.view", "    %bv = buffer.view %x_stage[%origin] : buffer -> view<256x72xf16>"),
                       ("a workgroup op", "    %wg = kernel.workgroup.reduce<maxnumf> %no_scale : f32"),
                       ("a line at 2-space indent", "  %zz = index.constant 0 : index"),
                       ("an lhs load from another view", "    %lq = vector.fragment.load<lhs> %x_other_view[%wave_row, %c0] shape [%c16, %c16] : view<256x72xf16> -> vector<16xf16>"),
                       ("a vector.load from another view", "    %vq = vector.load %x_stage_view[%wave_row, %c0] : view<256x72xf16> -> vector<16xf16>")):
        refused(f"{what} in the region stops the build", k0.replace(rv, line + "\n" + rv, 1), "not pure fragment")
    # F2: the lhs rows are the wave's rows (the predicate reads row_origin + wave_row + lane, + 32)
    refused("lhs rows off the wave (%wave_row -> %wave_col) stop the build",
            k0.replace("vector.fragment.load<lhs> %x_stage_view[%wave_row, ", "vector.fragment.load<lhs> %x_stage_view[%wave_col, ", 1), "not pure fragment")
    refused("lhs loads that skip one of the wave's 16-row groups stop the build",
            k0.replace("vector.fragment.load<lhs> %x_stage_view[%wave_row3, ", "vector.fragment.load<lhs> %x_stage_view[%wave_row2, "), "lhs loads read rows")
    # the lines the pass must pin, listed here and not read from the pass (a line dropped from its ANCHORS must red a case)
    for a in ("  %lane = kernel.subgroup.lane.id : index\n", "  %wave_row = index.assume %wave_row_value [range(%wave_row_value, 0, 192)] : index\n",
              "  %wave_row1_c = index.constant 16 : index\n", "  %wave_row2_c = index.constant 32 : index\n", "  %wave_row3_c = index.constant 48 : index\n",
              "  %wave_row1 = index.add %wave_row, %wave_row1_c : index\n", "  %wave_row2 = index.add %wave_row, %wave_row2_c : index\n",
              "  %wave_row3 = index.add %wave_row, %wave_row3_c : index\n", "  %s_view = buffer.view %scales[%origin] : buffer -> view<[%rows]x2xf32>\n",
              "  %row_origin = index.mul %row_group, %c256 : index\n", "  %x_stage = buffer.alloca<workgroup> align(16) %x_bytes : buffer\n",
              "  %x_stage_view = buffer.view %x_stage[%origin] : buffer -> view<256x72xf16>\n", "  %w_stage = buffer.alloca<workgroup> align(16) %w_bytes : buffer\n",
              "  %x_stage_row = index.div %workitem, %c2 : index\n", "  %x_stage_row1 = index.add %x_stage_row, %c128 : index\n",
              "  %x_row_value = index.add %row_origin, %x_stage_row : index\n", "  %x_row = index.assume %x_row_value [lt(%x_row_value, %rows)] : index\n",
              "  %x_row1_value = index.add %x_row, %c128 : index\n", "  %x_row1 = index.assume %x_row1_value [lt(%x_row1_value, %rows)] : index\n",
              "    %x_rawp = vector.load %x_view[%x_row, %x_colp] : view<[%rows]x[%kx]xf16> -> vector<32xf16>\n",
              "    %x_rawp1 = vector.load %x_view[%x_row1, %x_colp] : view<[%rows]x[%kx]xf16> -> vector<32xf16>\n",
              "    vector.store %cr_x, %x_stage_view[%x_stage_row, %x_stage_col] : vector<32xf16>, view<256x72xf16>\n",
              "    vector.store %cr_x1, %x_stage_view[%x_stage_row1, %x_stage_col] : vector<32xf16>, view<256x72xf16>\n"):
        refused(f"a kernel without anchor {a.strip()[:60]!r} stops the build", k0.replace(a, "", 1), a.strip()[:40])
    refused("an in-loop x prefetch in neither known form stops the build (by-row form: second load off x_row1)",
            k0.replace("%xv1 = vector.load %x_view[%x_row1, %x_coln]", "%xv1 = vector.load %x_view[%x_row, %x_coln]", 1), "x prefetch")
    refused("a weight view not pinned to its workgroup alloca stops the build",
            re.sub(r"(?m)^  %w_wave_view = buffer\.view %w_stage\[", "  %w_wave_view = buffer.view %w[", k0, count=1), "w_wave_view")
    # loop shape: one K loop, its header, its yield last, a region, both barriers beside the region
    refused("a kernel with no K loop stops the build", k0.replace("= scf.for %step = ", "= scf.for %stp = ", 1), "K loops")
    h0 = next(ln for ln in k0.split("\n") if "= scf.for %step = " in ln)
    refused("a K loop header not in the expected form stops the build", k0.replace(h0, h0[:-2] + "  {", 1), "header not in the expected form")
    ylast = next(ln for ln in k0.split("\n") if ln.startswith("    scf.yield %m0r0c0_"))
    refused("a loop-carried value not in the expected form stops the build", k0.replace("%i0r0c0 = %init : ", "%i0r0c0 = %init: ", 1), "loop-carried value")
    refused("a K loop that never closes stops the build", k0[:k0.index(ylast) + len(ylast) + 1], "K loop end not found")
    refused("a line after the K loop's yield stops the build", k0.replace(ylast + "\n", ylast + "\n    %after = index.constant 0 : index\n", 1), "yield is not its last line")
    refused("a K loop without fragment loads stops the build", k0.replace("= vector.fragment.load<lhs> ", "= vector.fragment.load<rhs> "), "no fragment-load / MMA region")
    jl = k0.index("\n", k0.rindex("= vector.mma ")) + 1
    refused("a line between the last MMA and the closing barrier stops the build", k0[:jl] + "    %zz = index.constant 0 : index\n" + k0[jl:], "closing barrier")
    jf = k0.index("= vector.fragment.load<lhs>"); jb = k0.rindex("    kernel.barrier<workgroup>", 0, jf)
    refused("a K loop without the staging barrier before the region stops the build", k0[:jb] + k0[k0.index("\n", jb) + 1:], "closing barrier")
    refused("a yield with fewer values than the loop carries stops the build", k0.replace(ylast, ylast.replace("%m0r0c0_3, ", "", 1), 1), "yield has")
    hl = k0.split("\n").index(h0)
    refused("a yielded region value carried from another init stops the build", k0.replace("%i0r0c0 = %init :", "%i0r0c0 = %init2 :", 1), "is not an accumulator")
    extra = h0.replace(") -> (", ", %extra = %init : vector<8xf32>) -> (", 1).replace(") {", ", vector<8xf32>) {", 1) if h0.endswith(") {") else h0
    ytail = ylast.split(" : ", 1)
    refused("an accumulator the region does not produce stops the build",
            k0.replace(h0, extra, 1).replace(ylast, ytail[0] + ", %extra : " + ytail[1] + ", vector<8xf32>", 1), "an accumulator is not produced")
    ini = "  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>\n"
    refused("a second accumulator init stops the build", k0.replace(ini, ini + ini, 1), "accumulator init anchor")
    ye = k0.index("\n  }\n", k0.index(ylast)) + 5
    refused("an accumulator init after the K loop stops the build", (k0[:ye] + ini + k0[ye:]).replace(ini, "", 1), "accumulator init anchor")
    # which kernels apply() rewrites: gfx11 by target, gfx12 twins by name only
    other = text + k0.replace("target(@rdna3) ", "target(@rdna5) ", 1).replace("g_gemm_f16_swiglu_m256_f16o", "g_gemm_f16_store_m256")
    refused("a kernel for an unknown target stops the build", other, "target rdna5", lambda: gemm_wave_skip.apply(other))
    twin = text + k0.replace("target(@rdna3) ", "target(@rdna4) ", 1).replace("g_gemm_f16_swiglu_m256_f16o", "g_gemm_f16_store_m256")
    refused("a gfx12 kernel without the _rdna4 name stops the build (not skipped)", twin, "target rdna4", lambda: gemm_wave_skip.apply(twin))
    # QG_SPLIT's _qgs (HRX-like keeps-1127 sweep: GEMM_WAVE_SKIP stopped every build with QG_SPLIT=1 on its name): fixture cut from
    # QG_SPLIT=1's generated program. The rewrite must take it, skip its gfx12 twin, and insert in it exactly the lines it inserts
    # in store_m256 (same predicate, same region), so its K loop is store_m256's and only the epilogue differs.
    import difflib
    qt = open(os.path.join(HERE, "fixtures", "gemm_f16_store_m256_qgs.loom")).read()
    qout, qdone = gemm_wave_skip.apply(qt)
    # the flat-lane x prefetch (GEMM256_FLATADDR's form, served by store / acc / _qgs; the swiglu fixture above is by-row since E744)
    ks = body(qt, "g_gemm_f16_store_m256")
    ksm = ks.replace("  %x_lane1_v = index.madd %x_row1, %kx, %x_stage_col : index\n", "  %x_lane1_v = index.madd %x_row, %kx, %x_stage_col : index\n", 1)
    case("an in-loop x prefetch in neither known form stops the build (flat-lane form: lane1 off x_row)",
         ksm != ks and refuses(lambda: gemm_wave_skip.wave_skip(ksm, "x"), "x prefetch"))
    case(f"QG_SPLIT's _qgs is rewritten beside store_m256, its twin skipped ({qdone})",
         sorted(qdone) == ["g_gemm_f16_store_m256", "g_gemm_f16_store_m256_qgs"])
    def added(name):
        a, b = body(qt, name).split("\n"), body(qout, name).split("\n")
        return [ln for ln in difflib.ndiff(a, b) if ln[:2] in ("+ ", "- ")]
    qa, sa = added("g_gemm_f16_store_m256_qgs"), added("g_gemm_f16_store_m256")
    case(f"the rewrite changes the same {len(sa)} lines in _qgs as in store_m256 (predicate added, region wrapped)", bool(sa) and qa == sa)
    kl = lambda k: k[k.index("scf.for"):k.index("\n  }\n", k.index("scf.for"))]
    case("_qgs's generated K loop is store_m256's, character for character",
         kl(body(qt, "g_gemm_f16_store_m256_qgs")) == kl(body(qt, "g_gemm_f16_store_m256")))
    case("the _qgs gfx12 twin is left as generated", body(qout, "g_gemm_f16_store_m256_qgs_rdna4") == body(qt, "g_gemm_f16_store_m256_qgs_rdna4"))
    qodd = qt.replace("g_gemm_f16_store_m256_qgs", "g_gemm_f16_store_m256_qgt")
    case("a _qgs-like kernel under another name still stops the build (fail closed)",
         refuses(lambda: gemm_wave_skip.apply(qodd), "g_gemm_f16_store_m256_qgt"))
    if len(sys.argv) > 1:
        prog = open(sys.argv[1]).read()
        names = [m.group(2) for m in gemm_wave_skip.KERNEL.finditer(prog) if m.group(1) != "rdna4"]
        have = [k for k in names if "scf.if %wsk_live" in body(prog, k)]
        case(f"{os.path.basename(sys.argv[1])}: every gfx11 f16 m256 GEMM holds the wave skip ({len(have)} of {len(names)})", bool(names) and have == names)
    print(f"{n - fails}/{n} passed"); return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
