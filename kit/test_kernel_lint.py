#!/usr/bin/env python3
"""common/kernel_lint.py on real compiles and on listings. Recurrences it prevents:
  - an IQ4 lookup scalarised into a select chain (E229/E230: i32 nibble mask) not flagged: g_deq16_iq4_nl compiled with the
    i32 mask must flag `select`, with the byte mask it must not (both forms built from the generator's text, whichever it has);
  - byte-vector code assembly (E233) not flagged: a kernel calling the pre-E233 @g_q6_k_icodes (fixtures/q6_k_icodes_bytes.loom)
    must flag `repack`; with the word form (fixtures/q6_k_icodes_words.loom) it must not;
  - a spill in the hot loop not flagged (E230: 2,464 B): a listing with a scratch load inside the loop must flag `scratch`;
  - an allocator copy run not flagged (E179 / #3542: flash2's 45 moves before the back-edge): a loop with a 20-move run flags
    `copies`, one with 10 does not;
  - a kernel doing far more per multiply than its peers not flagged (E233: q6_k 616 vs q5_k 261), or attention judged against
    GEMMs with the same WMMA count (E241 calibration): `outlier` within a family only;
  - selects that are an algorithm flagged as a scalarised lookup (E244: llama.cpp's index scatter and Walsh-Hadamard kernels):
    a lookup chain compares against the table's entries;
  - another engine's code object (many kernels in one file, E244: llama.cpp's 147 objects hold 8,374 kernels) linted as one
    kernel or sliced wrongly: a two-kernel object must give two rows, each equal to that kernel compiled alone;
  - accumulator zero-inits counted as copies, or 16-bit half-register copies missed (E251, llama.cpp's blind review: 3 of 4
    copies flags were `v_mov vN, 0` runs; a real 36-move run was v_mov_b16 vN.l, vM.h): constants do not count, true16 moves do;
  - outlier presented as a defect (E251: 0 of 2 confirmed): it is a note, not a flag;
  - loads issued one at a time, each waited before the next independent one (E109: KV8 loads retired 2-26 instructions after
    issue; E254 scout: g_attention_split's 7 per pass, llama.cpp NVFP4 mul_mat_q's per-byte scales) not flagged `serial`: five
    independent lone vector loads in a loop flag it, so do five lone scalar loads; staggered waits, a pointer chase and the same
    loads in a prologue do not;
  - per-iteration 64-bit load addresses not noted `addr64` (E345: the f16 GEMMs rebuilt them every k step, 39 -> 15 VALU when hoisted):
    four such loads with carry adds get the note; the SGPR-base form and 64-bit addresses without carry adds do not;
  - the same bytes loaded twice not noted `reread` (E278: Strata's depth mix; llama.cpp's head-256 flash_attn_ext_vec repeats 24
    loads): two repeated loads with unchanged addresses get the note; a moved address, an intervening store or a pointer chase
    do not count;
  - launch-bound small kernels not noted `tiny` from a profile (E278): >= 1000 calls with a median under 10 us get it, fewer calls
    or longer calls do not;
  - a branch-join copy not flagged (E247: the norm's idle-lane zero branch, 61-64 moves after an exec restore, every pass):
    a run of 40 moves right after an exec restore flags `join`, one of 20 does not;
  - a regression against another build missed (E226 0.3.0's longer loop and move run, E230 a new spill, E236 fewer waves);
  - launch sites miscounted (E241: launches without a grid were missed): direct launches with and without a grid, through
    dual-target selectors, and none inside template bodies;
  - report columns read from the wrong place: LDS bank extra rounds from the summary report's gfx1151 model (E243: the 256-row
    GEMMs' activation staging store, 48 extra rounds), each kernel's served ms from a profile run, a load's distance to its wait
    and the loads still in flight at the back-edge (E109);
  - selects behind a branch read as every-iteration cost (E334: vsplit's masked path, 384 of 496 cndmask, runs only on edge
    blocks): loop_census.guarded reports the region a forward branch can skip, not the straight path or an early exit;
  - the outer shell of a nested loop taken as the hot loop (E237: the 16-token swiglu matvecs were flagged `select` for
    their outer loop's selects while the dot4s sat in the inner loop): the inner loop must be chosen, nothing flagged;
  - an LDS bank conflict in the hot loop not noted `ldsbank` (E424 / E435 / E456: flash2's SX11 exchange, 8.2 % measured), or an
    epilogue conflict noted (E249: the f16 GEMMs' once-per-tile store, 0.2 % measured): a 2-way store inside the loop gets the
    note, the same store after the loop does not, nor 1 % of the loop's rounds, nor rows that cannot be matched to the binary;
  - an attention kernel whose key-valid bit masks only the score not noted `maskedv` (E441 / E444: g_attention_split read V of
    unwritten keys, a NaN reached the output): the score-only mask fires, and so does zeroing only p; V's scale selected to 0
    (KV8_SPLIT_VZERO), a clamped key (g_dfl_ctx_split), no softmax (g_dfl_topk) or a -1.0 fill do not.
usage: python3 kit/test_kernel_lint.py   (needs loom-compile: VENV from the environment or the machine env file)"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import gen_gemm  # noqa: E402
import kernel_lint as kl  # noqa: E402
from kit_testlib import kernel, loom_compile  # noqa: E402

fails = 0


def ok(cond, what):
    global fails
    print(("ok  " if cond else "FAIL") + " " + what)
    fails += not cond


def compile_lint(lc, t, name, text, root):
    p = os.path.join(t, name + ".loom")
    open(p, "w").write("target.decl @device\namdgpu.target<gfx11-generic> @rdna3\n\n" + text)
    out = os.path.join(t, name + ".hsaco")
    r = subprocess.run([lc, p, "--target=amdgpu:gfx1151", f"--root={root}", f"--output={out}"], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"FAIL compile {name}: {r.stderr.strip()[:300]}")
    return kl.lint(out)


def main():
    lc = loom_compile()
    qv = open(os.path.join(HERE, "quant_values.loom")).read()
    gen = gen_gemm.deq16("iq4_nl")
    byte_mask = ("      %dq_sbytes = vector.bitcast %dq_shifted : vector<4xi32> to vector<16xi8>\n"
                 "      %dq_idx = vector.andi %dq_sbytes, %dq_nmask : vector<16xi8>\n")
    word_mask = ("      %dq_masked = vector.andi %dq_shifted, %dq_nmask : vector<4xi32>\n"
                 "      %dq_idx = vector.bitcast %dq_masked : vector<4xi32> to vector<16xi8>\n")
    nb, nw = "%dq_nmask = vector.splati 15 : vector<16xi8>", "%dq_nmask = vector.splati 252645135 : vector<4xi32>"
    if gen.count(byte_mask) == 1 and gen.count(nb) == 1:
        new, old = gen, gen.replace(byte_mask, word_mask).replace(nb, nw)
    elif gen.count(word_mask) == 1 and gen.count(nw) == 1:
        old, new = gen, gen.replace(word_mask, byte_mask).replace(nw, nb)
    else:
        sys.exit("FAIL: g_deq16_iq4_nl's nibble mask is in neither known form (generator text changed): update this test")
    q6_new = open(os.path.join(HERE, "fixtures", "q6_k_icodes_words.loom")).read()
    q6_old = open(os.path.join(HERE, "fixtures", "q6_k_icodes_bytes.loom")).read()
    with tempfile.TemporaryDirectory() as t:
        r = compile_lint(lc, t, "iq4_old", qv + old, "g_deq16_iq4_nl")
        ok("select" in r["flags"], f"pre-E230 IQ4 dequant flags select ({r['select']} of {r['common']})")
        two = os.path.join(t, "two.loom")
        open(two, "w").write("target.decl @device\namdgpu.target<gfx11-generic> @rdna3\n\n" + qv + old + old.replace("g_deq16_iq4_nl", "g_deq16_iq4_nl2"))
        subprocess.run([lc, two, "--target=amdgpu:gfx1151", f"--output={t}/two.hsaco"], capture_output=True, text=True, check=True)
        rows = kl.lint_file(os.path.join(t, "two.hsaco"))
        ok(len(rows) == 2 and all(x["select"] == r["select"] and x["loop"] == r["loop"] and "select" in x["flags"] for x in rows),
           f"a two-kernel object gives two rows equal to the kernel alone ({[(x['kernel'], x['select'], x['loop']) for x in rows]} vs {r['select']}, {r['loop']})")
        r = compile_lint(lc, t, "iq4_new", qv + new, "g_deq16_iq4_nl")
        ok(not r["flags"], f"IQ4 dequant with the byte mask: no flag ({r['select']} of {r['common']}, flags {r['flags']})")
        r = compile_lint(lc, t, "q6_old", q6_old + kernel("g_q6_k_icodes", False), "t_g_q6_k_icodes")
        ok("repack" in r["flags"], f"pre-E233 q6_k codes flag repack (pk {r['pk']}, bfe {r['bfe']})")
        r = compile_lint(lc, t, "q6_new", q6_new + kernel("g_q6_k_icodes", False), "t_g_q6_k_icodes")
        ok(not r["flags"], f"q6_k codes on words: no flag (pk {r['pk']}, bfe {r['bfe']}, flags {r['flags']})")
    # listings: addresses 4 bytes apart; s_branch's offset counts dwords from the next instruction
    def listing(body):
        return body, [4 * i for i in range(len(body))]
    def back(i, target):
        return f"s_branch {(target - (i + 1)) & 0xFFFF}"
    spill = ["s_mov_b32 s0, 0", "v_dot4_i32_iu8 v1, v2, v3, v1", "scratch_load_b32 v4, off, s0", "v_add_nc_u32_e32 v1, v1, v4"]
    spill.append(back(len(spill), 1))
    # guarded regions (E334): selects behind a forward conditional branch are reported as skippable; the same selects on the
    # straight path are not, and a branch out of the loop is not a region
    import loop_census as lcen
    def fwd(i, target, op="s_cbranch_scc0"):
        return f"{op} {(target - (i + 1)) & 0xFFFF}"
    gl = ["s_mov_b32 s0, 0"] + ["v_wmma_f32_16x16x16_f16 v[0:7], v[8:15], v[16:23], v[0:7]"] * 4 + ["s_cmp_eq_u32 s1, 0"]
    gb = len(gl); gl.append("")                                            # the forward branch, filled in below
    gl += ["v_cmp_eq_i32_e64 s2, v1, v2", "v_cndmask_b32_e64 v3, v4, v5, s2"] * 12
    gl[gb] = fwd(gb, len(gl))
    gl += ["s_add_u32 s0, s0, 1"]; gl.append(back(len(gl), 1))
    ins_g, addr_g = listing(gl)
    rg = lcen.guarded(ins_g, addr_g, 1, len(gl) - 1)
    inside = {j for _, lo, hi in rg for j in range(lo, hi + 1)}
    ok(len(rg) == 1 and sum(ins_g[j].startswith("v_cndmask") for j in inside) == 12 and not any(ins_g[j].startswith("v_wmma") for j in inside),
       f"selects behind a forward branch are a guarded region, the WMMAs before it are not ({rg})")
    straight = [x for k, x in enumerate(gl) if k != gb]
    straight[-1] = back(len(straight) - 1, 1)
    ins_s, addr_s = listing(straight)
    ok(lcen.guarded(ins_s, addr_s, 1, len(straight) - 1) == [], "the same selects on the straight path: no guarded region")
    ex = list(gl); ex[gb] = fwd(gb, len(gl) + 5)
    ins_x, addr_x = listing(ex)
    ok(lcen.guarded(ins_x, addr_x, 1, len(ex) - 1) == [], "a branch out of the loop (early exit) is not a region")
    ex2 = list(gl); ex2[gb] = fwd(gb, len(gl))                            # to the instruction after the back-edge: the loop's exit test
    ex2.append("s_endpgm")
    ins_y, addr_y = listing(ex2)
    ok(lcen.guarded(ins_y, addr_y, 1, len(gl) - 1) == [], "a branch to just past the back-edge (the loop's exit test) is not a region")
    r = kl.lint_listing("spill", *listing(spill))
    ok("scratch" in r["flags"], f"a scratch load in the hot loop flags scratch ({r['scratch']})")
    nest = ["s_mov_b32 s0, 0"] + ["v_cmp_eq_u32_e64 s1, v0, v1", "v_cndmask_b32_e64 v2, v3, v4, s1"] * 4   # outer shell: selects
    t_in = len(nest)
    nest += ["v_dot4_i32_iu8 v5, v6, v7, v5"] * 8 + ["s_add_u32 s0, s0, 1"]
    nest.append(back(len(nest), t_in))                                  # inner loop: the dot4s
    nest.append(back(len(nest), 1))                                     # outer loop
    r = kl.lint_listing("nested", *listing(nest))
    ok(r["where"] == "loop" and r["dot"] == 8 and not r["flags"], f"the inner dot4 loop is the hot loop, no flag (dot {r['dot']}, flags {r['flags']})")
    moves_loop = lambda nm: ["s_mov_b32 s0, 0", "v_dot4_i32_iu8 v1, v2, v3, v1"] + [f"v_mov_b32_e32 v{10 + i}, v{40 + i}" for i in range(nm)] + [back(2 + nm, 1)]
    r = kl.lint_listing("run20", *listing(moves_loop(20)))
    ok("copies" in r["flags"] and r["run"] == 20, f"a 20-move run in the hot loop flags copies (run {r['run']})")
    zi = ["s_mov_b32 s0, 0", "v_dot4_i32_iu8 v1, v2, v3, v1"] + [f"v_mov_b32_e32 v{10 + i}, 0" for i in range(20)] + [back(22, 1)]
    r = kl.lint_listing("zeroinit", *listing(zi))
    ok("copies" not in r["flags"] and r["run"] == 0, f"20 zero-init moves are not copies (run {r['run']})")
    h16 = ["s_mov_b32 s0, 0", "v_dot4_i32_iu8 v1, v2, v3, v1"] + [f"v_mov_b16_e64 v{10 + i}.l, v{60 + i}.h op_sel:[1,0]" for i in range(20)] + [back(22, 1)]
    r = kl.lint_listing("true16", *listing(h16))
    ok("copies" in r["flags"] and r["run"] == 20, f"20 half-register copies are copies (run {r['run']})")
    def loop_of(body):                                                  # body as one loop after a prologue instruction
        b = ["s_mov_b32 s0, 0"] + body
        return b + [back(len(b), 1)]
    vm_lone = sum(([f"global_load_b32 v{10 + k}, v[2:3], off offset:{4 * k}", "s_waitcnt vmcnt(0)", f"v_add_f32_e32 v1, v1, v{10 + k}"]
                   for k in range(5)), [])
    r = kl.lint_listing("serial_vm", *listing(loop_of(vm_lone)))
    ok("serial" in r["flags"] and r["serial"] == 4, f"five lone independent vector loads in a loop flag serial (links {r.get('serial')})")
    sm_lone = sum(([f"s_load_b32 s{10 + k}, s[4:5], 0x{4 * k:x}", "s_waitcnt lgkmcnt(0)", f"v_add_f32_e32 v1, s{10 + k}, v1"]
                   for k in range(5)), [])
    r = kl.lint_listing("serial_sm", *listing(loop_of(sm_lone)))
    ok("serial" in r["flags"], f"five lone scalar loads flag serial too (links {r.get('serial')})")
    stag = [f"global_load_b32 v{10 + k}, v[2:3], off offset:{4 * k}" for k in range(5)]
    stag += sum(([f"s_waitcnt vmcnt({4 - k})", f"v_add_f32_e32 v1, v1, v{10 + k}"] for k in range(5)), [])
    r = kl.lint_listing("staggered", *listing(loop_of(stag)))
    ok("serial" not in r["flags"], f"the same loads issued together with staggered waits do not (links {r.get('serial')})")
    chase = sum((["global_load_b64 v[2:3], v[2:3], off", "s_waitcnt vmcnt(0)", "v_add_f32_e32 v1, v1, v2"] for k in range(5)), [])
    r = kl.lint_listing("chase", *listing(loop_of(chase)))
    ok("serial" not in r["flags"], f"a pointer chase (each address the previous load's data) does not (links {r.get('serial')})")
    pro = vm_lone + ["s_mov_b32 s0, 0", "v_dot4_i32_iu8 v1, v2, v3, v1", "s_add_u32 s0, s0, 1"]
    pro.append(back(len(pro), len(vm_lone) + 1))
    r = kl.lint_listing("prologue", *listing(pro))
    ok("serial" not in r["flags"], f"lone loads in a prologue, outside every loop, do not (links {r.get('serial')})")
    def wm(k):
        return f"v_wmma_f32_16x16x16_f16 v[{40 + 8 * (k % 4)}:{47 + 8 * (k % 4)}], v[20:27], v[30:37], v[{40 + 8 * (k % 4)}:{47 + 8 * (k % 4)}]"
    # addr64 (E345): 64-bit per-lane load addresses rebuilt every iteration get the note; the SGPR-base form and 64-bit loads
    # without per-iteration carry adds do not
    a64 = sum(([f"v_add_co_u32 v{20 + 2 * k}, s18, s12, v12", f"v_add_co_ci_u32_e64 v{21 + 2 * k}, s18, s13, v13, s18",
                f"global_load_b128 v[{40 + 4 * k}:{43 + 4 * k}], v[{20 + 2 * k}:{21 + 2 * k}], off"] for k in range(4)), [])
    r = kl.lint_listing("addr64", *listing(loop_of(a64 + [wm(0)])))
    ok("addr64" in r.get("notes", []) and "addr64" not in r["flags"] and r["addr64"] == 4,
       f"four 64-bit-address loads rebuilt with carry adds each iteration get the addr64 note, not a flag ({r.get('addr64')})")
    sb = [f"global_load_b128 v[{40 + 4 * k}:{43 + 4 * k}], v2, s[8:9] offset:{16 * k}" for k in range(4)]
    r = kl.lint_listing("saddr", *listing(loop_of(sb + [wm(0)])))
    ok("addr64" not in r.get("notes", []), f"the SGPR-base form: no note ({r.get('addr64')})")
    nc = [f"global_load_b128 v[{40 + 4 * k}:{43 + 4 * k}], v[20:21], off offset:{16 * k}" for k in range(4)]
    r = kl.lint_listing("a64nocarry", *listing(loop_of(nc + [wm(0)])))
    ok("addr64" not in r.get("notes", []), f"64-bit addresses kept across iterations (no carry adds in the loop): no note ({r.get('addr64')})")
    rr = ["global_load_b128 v[0:3], v4, off", "global_load_b128 v[8:11], v5, off", "v_add_f32_e32 v10, v0, v1",
          "global_load_b128 v[20:23], v4, off", "global_load_b128 v[24:27], v5, off", "s_endpgm"]
    r = kl.lint_listing("reread", *listing(rr))
    ok("reread" in r.get("notes", []) and r["reread"] == 2, f"two loads repeated with unchanged addresses get the reread note ({r['reread']})")
    clean = ["global_load_b128 v[0:3], v4, off", "v_add_nc_u32_e32 v4, 16, v4", "global_load_b128 v[20:23], v4, off",
             "global_load_b64 v[8:9], v5, off", "global_store_b32 v6, v7, off", "global_load_b64 v[12:13], v5, off",
             "global_load_b64 v[30:31], v[30:31], off", "global_load_b64 v[30:31], v[30:31], off", "s_endpgm"]
    r = kl.lint_listing("noreread", *listing(clean))
    ok(r["reread"] == 0, f"a moved address, an intervening store and a pointer chase are not rereads ({r['reread']})")
    rows_t = [{"kernel": k} for k in ("a", "b", "c")]
    kl.tiny(rows_t, {"a": (50.0, 5000, 3.0), "b": (50.0, 500, 3.0), "c": (500.0, 5000, 40.0)})
    ok(["tiny" in r_.get("notes", []) for r_ in rows_t] == [True, False, False],
       "tiny: 5000 calls at 3 us noted; 500 calls, or 40 us calls, not")
    r = kl.lint_listing("run10", *listing(moves_loop(10)))
    ok("copies" not in r["flags"], f"a 10-move run does not (run {r['run']}, flags {r['flags']})")
    row = lambda k, op, wm=0, dt=8: {"kernel": k, "where": "loop", "wmma": wm, "dot": dt, "other_per_op": op, "flags": []}
    rows = [row("g_mv_a", 3.0), row("g_mv_b", 3.2), row("g_mv_c", 2.9), row("g_mv_d", 8.6),
            row("g_attention_x", 30.0, 32, 0), row("g_gemm_a", 10.0, 32, 0), row("g_gemm_b", 11.0, 32, 0), row("g_gemm_c", 10.5, 32, 0)]
    hit = kl.outliers(rows)
    ok(set(hit) == {"g_mv_d"} and not any("outlier" in x["flags"] for x in rows) and "outlier" in rows[3].get("notes", []),
       f"outlier: the matvec at 2.8x its peers, as a note not a flag, not the attention kernel beside GEMMs ({sorted(hit)})")
    base = {"scratch": 0, "pf": "kept", "waves": 12, "loop": 1544, "run": 4}
    ok(kl.regressions(dict(base, loop=2179, run=32), base) == ["loop1544->2179", "run4->32"], "regress: 0.3.0's longer loop and move run (E226)")
    ok(kl.regressions(dict(base, scratch=513, pf="drain"), base) == ["spill", "drain"], "regress: a new spill and a drained prefetch (E230)")
    ok(kl.regressions(dict(base, waves=10), base) == ["waves12->10"], "regress: fewer waves per SIMD (E236)")
    ok(kl.regressions(dict(base), base) == [], "regress: nothing when nothing changed")
    prog = ("template.def<@sel1_g_k1> target(@rdna3) @sel1_g_k1_rdna3(%g0: index) {\n  kernel.launch @g_k1[%g0](%g0) : [index](index)\n}\n"
            "command.def public @c() {\n  template.apply<@sel1_g_k1>(%t) : (index)\n  template.apply<@sel1_g_k1>(%u) : (index)\n"
            "  kernel.launch @g_k2[%t](%t) : [index](index)\n  kernel.launch @g_k3(%t, %u) : (index, index)\n}\n")
    s2 = kl.launch_sites([prog])
    ok((s2["g_k1"], s2["g_k2"], s2["g_k3"], s2.get("g_k4", 0)) == (2, 1, 1, 0),
       f"launch sites: selectors 2, grid launch 1, gridless launch 1, none 0 ({dict(s2)})")
    # maskedv (E444): the key-valid bit masks only the score while V is read unconditionally (p = +0 times a NaN V is NaN)
    def att(vdef, extra=""):
        return ('kernel.def export("g_att") @g_att(%tokens: index) {\n} launch(%q: buffer) {\n'
                "  %neg_big = scalar.constant -3.0e38 : f32\n  %zero_f = scalar.constant 0.0 : f32\n"
                "  %okl0 = index.cmp uge, %key0, %lo : index\n  %okh0 = index.cmp ult, %key0, %hi : index\n"
                "  %ok0 = scalar.andi %okl0, %okh0 : i1\n  %vs0 = view.load %vsv[%kv] : view<4xf32> -> f32\n"
                f"  {vdef}\n  %v0 = scalar.mulf %vq0, %vsz0 : f32\n  %sr0 = scalar.mulf %qv, %k0 : f32\n"
                "  %s0 = scf.select %ok0, %sr0, %neg_big : f32\n  %d0 = scalar.subf %s0, %m : f32\n  %p0 = scalar.expf<afn> %d0 : f32\n"
                f"{extra}  %a0 = scalar.fmaf %p0, %v0, %acc : f32\n}}\nfunc.def @g_after() {{\n}}\n")
    plain = "%vsz0 = scalar.maxnumf %vs0, %vs0 : f32"
    kt = kl.kernel_texts([att(plain)])
    ok(list(kt) == ["g_att"] and "func.def" not in kt["g_att"], f"kernel_texts: one kernel, cut at the next definition ({list(kt)})")
    ok(kl.masked_v(att(plain)) == ["%ok0"], f"maskedv: score masked, V scale used unconditionally fires (E444) ({kl.masked_v(att(plain))})")
    pz = "  %pw0 = scf.select %ok0, %p0, %zero_f : f32\n"
    ok(kl.masked_v(att(plain, pz)) == ["%ok0"], "maskedv: zeroing p is not a guard (p = +0 times NaN is NaN, E441): still fires")
    ok(kl.masked_v(att("%vsz0 = scf.select %ok0, %vs0, %zero_f : f32")) is None,
       "maskedv: the V scale selected to 0 under the bit (q179's KV8_SPLIT_VZERO) does not fire")
    ok(kl.masked_v(att(plain, "  %kc0 = index.min %key0, %last : index\n")) is None,
       "maskedv: the compared key clamped (g_dfl_ctx_split's index.min) does not fire")
    ok(kl.masked_v(att(plain).replace("scalar.expf<afn> %d0", "scalar.addf %d0, %d0")) is None,
       "maskedv: a masked score with no softmax (a top-k logit mask, g_dfl_topk) does not fire")
    vec = att(plain).replace("  %s0 = scf.select %ok0, %sr0, %neg_big : f32\n",
                             "  %neg_v = vector.splat %neg_big : vector<8xf32>\n  %s0 = vector.select %ok0, %sr0, %neg_v : vector<8xf32>\n")
    ok(kl.masked_v(vec) == ["%ok0"], "maskedv: a vector.select against a splat of the fill fires too (vsplit's form)")
    ok(kl.masked_v(att(plain).replace("-3.0e38", "-1.0")) is None, "maskedv: a select against -1.0 is not a score mask")
    # barrier check (user 11-25, cuda-oxide's no_data_race / uniform_barriers; HRX-like E504: one extra barrier per flash2 tile
    # cost +1.4 %). Recurrences: an LDS write read by another wave with no barrier between not reported; a barrier that orders
    # nothing kept silently; a barrier under a lane-dependent branch (some waves never arrive) passed; a construct the model cannot
    # read reported clean instead of 'not analysed'; a predecessor list doubled per scf.if (11-25: 2^k entries, the draft fire run
    # took 47.9 GB and was OOM-killed); report lines one off the program file's (11-25 draft: L4298 for the barrier at 4299)
    def bk(body, wg="%c64", alloc=None):
        alloc = alloc or "  %t = buffer.alloca<workgroup> align(16) %bytes : buffer\n  %tv = buffer.view %t[%c0] : buffer -> view<64xf32>\n"
        return ('kernel.def export("g_b") @g_b(%n: index) {\n  %c1 = index.constant 1 : index\n  %c64 = index.constant 64 : index\n'
                "  %c32 = index.constant 32 : index\n"
                f"  kernel.launch.config workgroups(%n, %c1, %c1) workgroup_size({wg}, %c1, %c1) : index\n"
                "} launch(%x: buffer, %y: buffer) {\n  %c0 = index.constant 0 : index\n  %c1 = index.constant 1 : index\n"
                "  %c32 = index.constant 32 : index\n  %tid = kernel.workitem.id x : index\n" + alloc +
                "  %xv = buffer.view %x[%c0] : buffer -> view<64xf32>\n  %v = view.load %xv[%tid] : view<64xf32> -> f32\n"
                + body + "  kernel.return\n}\n")
    st = "  view.store %v, %tv[%tid] : f32, view<64xf32>\n"
    nb = "  %nb = index.sub %c32, %tid : index\n  %r = view.load %tv[%nb] : view<64xf32> -> f32\n"
    bar = "  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n"
    b = kl.barrier_check(bk(st + nb))
    ok(b["status"] == "no barriers", f"barrier: a kernel without barriers is 'no barriers', not analysed further ({b['status']})")
    b = kl.barrier_check(bk(st + bar + bar + nb))
    ok(b["status"] == "analysed" and [x["verdict"] for x in b["barriers"]] == ["orders nothing"] * 2 and not b["races"],
       f"barrier: write, barrier, barrier, neighbour read: either barrier alone is redundant ({b['barriers']})")
    b = kl.barrier_check(bk(st + bar + nb))
    ok([x["verdict"] for x in b["barriers"]] == ["orders"] and not b["races"] and b["barriers"][0]["pairs"][0][2] == "RAW",
       f"barrier: write, barrier, neighbour read: the barrier orders the RAW ({b['barriers']})")
    b = kl.barrier_check(bk(st + bar + nb + "  view.store %v, %tv[%c1] : f32, view<64xf32>\n"), races=True)
    ok(len(b["races"]) == 1 and b["races"][0][2] == "WAR", f"barrier: a store after the read with no barrier is a WAR race ({b['races']})")
    b = kl.barrier_check(bk("  scf.for %i = [%c0 to %n step %c1] {\n" + st + bar + nb + "  }\n"), races=True)
    ok(any(r[2] == "WAR" for r in b["races"]), f"barrier: a tile loop without its end barrier races the next store with this "
       f"read across the back-edge (WAR) ({b['races']})")
    b = kl.barrier_check(bk("  scf.for %i = [%c0 to %n step %c1] {\n" + st + bar + nb + bar + "  }\n"), races=True)
    ok(not b["races"] and [x["verdict"] for x in b["barriers"]] == ["orders", "orders"],
       f"barrier: with the end barrier the loop is clean and both barriers order a pair ({b['barriers']}, {b['races']})")
    two = ("  %c128 = index.constant 128 : index\n  %lo = buffer.view %t[%c0] : buffer -> view<32xf32>\n"
           "  %hi = buffer.view %t[%c128] : buffer -> view<32xf32>\n  view.store %v, %lo[%tid] : f32, view<32xf32>\n"
           "  %r = view.load %hi[%tid] : view<32xf32> -> f32\n" + bar)
    b = kl.barrier_check(bk(two), races=True)
    ok(not b["races"], f"barrier: views at disjoint constant byte extents of one buffer do not race ({b['races']})")
    b = kl.barrier_check(bk(two.replace("%t[%c128]", "%t[%c64]")), races=True)
    ok(len(b["races"]) == 1, f"barrier: the same views overlapping (offset 64 < 128 bytes) race ({b['races']})")
    div = "  %lt = index.cmp ult, %tid, %c32 : index\n  scf.if %lt {\n" + st + bar + "  }\n" + nb
    b = kl.barrier_check(bk(div))
    ok(b["status"].startswith("divergent"), f"barrier: a barrier under a lane-dependent scf.if is reported divergent ({b['status']})")
    uni = div.replace("index.cmp ult, %tid, %c32", "index.cmp ult, %n, %c32")
    b = kl.barrier_check(bk(uni))
    ok(b["status"] == "analysed", f"barrier: the same under a uniform condition is analysed ({b['status']})")
    b = kl.barrier_check(bk(div.replace("index.cmp ult, %tid, %c32", "index.cmp ult, %w, %c32")
                            .replace("  %lt", "  %one = scalar.constant 1.0 : f32\n"
                                     "  %ct = view.atomic.rmw<add> %tv[%c0], %one : view<64xf32>, f32 -> f32\n"
                                     "  %w = scalar.fptosi %ct : f32 to index\n  %lt")))
    ok(b["status"].startswith("divergent"), f"barrier: a condition from an atomic's result (uniform operands) is non-uniform "
       f"({b['status']})")
    b = kl.barrier_check(bk(st + "  %pre = kernel.subgroup.scan<add> %n : index\n  scf.if %pre {\n" + bar + "  }\n"))
    ok(b["status"].startswith("divergent"), f"barrier: a subgroup scan of a uniform value (a per-lane prefix) is non-uniform "
       f"({b['status']})")
    red = div.replace("index.cmp ult, %tid, %c32", "scalar.cmpf one, %any, %no_scale").replace(
        "  %lt", "  %any = kernel.workgroup.reduce<maxnumf> %v : f32\n  %lt")
    b = kl.barrier_check(bk(red))
    ok(b["status"] == "analysed", f"barrier: a workgroup reduce of lane values is uniform (the GEMMs' tile_live skip; 42 GEMMs read "
       f"divergent while it was not) ({b['status']})")
    b = kl.barrier_check(bk(st + bar + "  scf.while %a {\n" + nb + "  }\n"))
    ok(b["status"].startswith("not analysed"), f"barrier: an unknown region op is 'not analysed', never clean ({b['status']})")
    b = kl.barrier_check(bk(st + "  scf.if %n {\n  kernel.return\n  }\n" + bar + nb))
    ok(b["status"].startswith("not analysed"), f"barrier: a return before the end of the launch body is 'not analysed' ({b['status']})")
    b = kl.barrier_check(bk(st + bar + "  %s = kernel.workgroup.scan<add> %v : f32\n" + nb))
    ok(b["status"].startswith("not analysed"), f"barrier: an unmodelled workgroup collective is 'not analysed' ({b['status']})")
    b = kl.barrier_check(bk(st + bar + nb, wg="%c32"))
    ok(b["status"] == "single wave", f"barrier: a one-wave workgroup has nothing to order ({b['status']})")
    chain = "".join(f"  %q{i} = index.cmp ult, %n, %c32 : index\n  scf.if %q{i} {{\n  }}\n" for i in range(28))
    probe = ("import resource, sys\nresource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))\n"
             f"sys.path.insert(0, {os.path.dirname(kl.__file__)!r})\nimport kernel_lint as kl\n"
             "print(kl.barrier_check(sys.stdin.read())['status'])\n")
    pr = subprocess.run([sys.executable, "-c", probe], input=bk(st + chain + bar + nb), capture_output=True, text=True, timeout=120)
    ok(pr.returncode == 0 and pr.stdout.strip() == "analysed", f"barrier: 28 empty ifs in a row analyse within 1 GiB of address "
       f"space (the 11-25 draft doubled a list per if: 2^28 entries; its fire run took 47.9 GB): rc {pr.returncode} "
       f"{(pr.stdout + pr.stderr).strip()[-80:]}")
    row = {"kernel": "g_b", "s_barrier": 1}
    kl.barriers([row], {"g_b": bk(st + bar + bar + nb)}, {})
    ok("barrier" not in row.get("notes", []) and row["barrier"]["status"].startswith("not analysed: the binary has 1"),
       f"barrier: a binary whose s_barrier count differs from the text's is not analysed ({row['barrier']['status']})")
    row = {"kernel": "g_b", "s_barrier": 2}
    kl.barriers([row], {"g_b": bk(st + bar + bar + nb)}, {})
    ok("barrier" in row.get("notes", []), f"barrier: matching counts give the note ({row.get('notes')})")
    row = {"kernel": "g_b", "s_barrier": 1}
    kl.barriers([row], {"g_b": bk(st + bar + nb)}, {})
    rep = kl.barrier_report(row, ("p.loom", 100), full=True)
    want = 100 + bk(st + bar + nb).split("\n").index(bar.rstrip("\n"))
    ok(any(x.startswith(f"  L{want}: orders") for x in rep), f"barrier: report lines are the program file's (kernel.def at 100, "
       f"barrier at {want}): {rep[1:3]}")
    row = {"kernel": "g_b", "s_barrier": 2}
    kl.barriers([row], {"g_b": bk(st + bar + bar + nb + "  view.store %v, %tv[%c1] : f32, view<64xf32>\n")}, {}, races=True)
    ok("race" not in row.get("notes", []) and row["barrier"]["races"] and kl.barrier_report(row)
       and not any("potential race" in x for x in kl.barrier_report(row))
       and any("potential race" in x for x in kl.barrier_report(row, full=True)),
       "barrier: a potential race is listed with --barriers only, never a note (0 of 10 sampled real, 11-25)")
    row = {"kernel": "g_other", "s_barrier": 3}
    kl.barriers([row], {"g_b": bk(st)}, {})
    ok(row.get("barrier", {}).get("status", "").startswith("not analysed"), "barrier: a binary with barriers and no text is 'not analysed'")
    # fail closed (review 11-25, static-1125/barrier-lint/review: B1 / B2 / B3 / N1 / N3). Recurrence: an LDS access, region,
    # footprint or name binding the check did not recognise was read as "no access" / "disjoint", so a barrier ordering a real
    # hazard was called removable (probes P1-P8, F3). Each row ends "orders" or "not analysed" / "divergent", never "orders nothing".
    wanted = lambda b: [x["verdict"] for x in b["barriers"]]
    rd = "  %nb = index.sub %c32, %tid : index\n"
    msk = "  %m = vector.splati 1 : vector<4xi1>\n  %old = vector.splatf 0.0 : vector<4xf32>\n"
    iota = "  %o = vector.iota %c0 step %c1 : vector<4xindex>\n"
    P = {
        "P1 a view.alloca<workgroup> is a region": (bk(st + bar + nb, alloc="  %tv = view.alloca<workgroup> align(16) : view<64xf32>\n"),),
        "P2 vector.load.mask reads the region": (bk(st + bar + rd + msk +
            "  %r = vector.load.mask %tv[%nb], %m, %old : view<64xf32>, vector<4xi1>, vector<4xf32>\n"),),
        "P2b vector.gather.mask reads it": (bk(st + bar + rd + msk + iota +
            "  %r = vector.gather.mask %tv[%nb][%o], %m, %old : view<64xf32>, vector<4xindex>, vector<4xi1>, vector<4xf32>\n"),),
        "P2c vector.load.expand reads it": (bk(st + bar + rd + msk +
            "  %r = vector.load.expand %tv[%nb], %m, %old : view<64xf32>, vector<4xi1>, vector<4xf32>\n"),),
        "P3 vector.atomic.rmw writes it": (bk(st + bar + rd + "  %w = vector.splatf 1.0 : vector<4xf32>\n" + iota +
            "  %old = vector.atomic.rmw<addf> %w, %tv[%nb][%o] {ordering = relaxed, scope = workgroup} : vector<4xf32>, view<64xf32>, "
            "vector<4xindex>\n"),),
        "P4 a 2-D vector<4x8> store overlaps a row-1 read": (bk("  %w = vector.splatf 1.0 : vector<4x8xf32>\n"
            "  vector.store %w, %tv[%c0, %c0] : vector<4x8xf32>, view<8x8xf32>\n" + bar + "  %r = view.load %tv[%c1, %c0] : view<8x8xf32> -> f32\n",
            alloc="  %t = buffer.alloca<workgroup> align(16) %bytes : buffer\n  %tv = buffer.view %t[%c0] : buffer -> view<8x8xf32>\n"),),
        "P5 a name reused in sibling if arms (then reads tv[tid], overlapping the store)": (bk(
            "  %c8 = index.constant 8 : index\n  %c40 = index.constant 40 : index\n"
            "  %k = index.rem %tid, %c8 : index\n  view.store %v, %tv[%k] : f32, view<64xf32>\n" + bar +
            "  %u = index.cmp ult, %n, %c32 : index\n  scf.if %u {\n  %j = index.add %tid, %c0 : index\n  %r = view.load %tv[%j] : view<64xf32> -> f32\n"
            "  } else {\n  %j = index.add %c40, %c1 : index\n  %r = view.load %tv[%j] : view<64xf32> -> f32\n  }\n"),),
        "P6 a name constant in one arm, computed in the other": (bk(
            "  %c8 = index.constant 8 : index\n  %k = index.rem %tid, %c8 : index\n  view.store %v, %tv[%k] : f32, view<64xf32>\n" + bar +
            "  %u = index.cmp ult, %n, %c32 : index\n  scf.if %u {\n  %j = scalar.fptosi %v : f32 to index\n  %r = view.load %tv[%j] : view<64xf32> -> f32\n"
            "  } else {\n  %j = index.constant 50 : index\n  %r = view.load %tv[%j] : view<64xf32> -> f32\n  }\n"),),
        "P5b an offset name constant 0 in one arm and 128 in the other (extents bind names in walk order)": (bk(
            "  %u = index.cmp ult, %n, %c32 : index\n"
            "  scf.if %u {\n  %off = index.constant 0 : index\n  %lo = buffer.view %t[%off] : buffer -> view<32xf32>\n"
            "  view.store %v, %lo[%c0] : f32, view<32xf32>\n  } else {\n  %off = index.constant 128 : index\n  }\n" + bar +
            "  %hi = buffer.view %t[%c0] : buffer -> view<32xf32>\n  %r = view.load %hi[%c0] : view<32xf32> -> f32\n"),),
        "F3 a callee writing the region and returning a view": (bk(st + bar + rd +
            "  %w2 = func.call @fill(%tv, %v) : (view<64xf32>, f32) -> view<64xf32>\n"),),
        "P5c an offset name constant in one arm, computed in the other (a rebinding drops the constant)": (bk(
            "  %u = index.cmp ult, %n, %c32 : index\n  scf.if %u {\n  %off = index.constant 128 : index\n  } else {\n"
            "  %off = index.add %tid, %c0 : index\n  %lo = buffer.view %t[%off] : buffer -> view<32xf32>\n"
            "  view.store %v, %lo[%c0] : f32, view<32xf32>\n  }\n" + bar +
            "  %hi = buffer.view %t[%c0] : buffer -> view<32xf32>\n  %r = view.load %hi[%c0] : view<32xf32> -> f32\n"),),
        "P5d a view name at a constant offset in one arm, a lane offset in the other (a rebinding drops the extent)": (bk(
            "  %u = index.cmp ult, %n, %c32 : index\n  scf.if %u {\n  %lo = buffer.view %t[%c0] : buffer -> view<32xf32>\n  } else {\n"
            "  %lo = buffer.view %t[%tid] : buffer -> view<32xf32>\n  view.store %v, %lo[%c0] : f32, view<32xf32>\n  }\n" + bar +
            "  %c128 = index.constant 128 : index\n  %hi = buffer.view %t[%c128] : buffer -> view<32xf32>\n"
            "  %r = view.load %hi[%c0] : view<32xf32> -> f32\n"),),
    }
    fill = {"fill": "func.def @fill(%w: view<64xf32>, %val: f32) -> view<64xf32> {\n  %c3 = index.constant 3 : index\n"
                    "  view.store %val, %w[%c3] : f32, view<64xf32>\n  func.return %w : view<64xf32>\n}\n"}
    for k, (t,) in P.items():
        b = kl.barrier_check(t, fill)
        ok(b["status"] == "analysed" and wanted(b) == ["orders"], f"barrier: {k}: orders ({b['status']}, {wanted(b)})")
    NA = {
        "P7 kernel.async.copy into the region (data lands at the wait, not at issue)": bk(
            "  %tok = kernel.async.copy %xv[%tid], %tv[%tid] : view<64xf32>, view<64xf32> -> kernel.async.token\n" + bar +
            "  kernel.async.wait %tok : kernel.async.token\n" + bar + nb),
        "buffer.alloca.init<workgroup> (a collective initialisation)": bk(st + bar + nb,
            alloc="  %t = buffer.alloca.init<workgroup> align(16) dense<0> : buffer\n  %tv = buffer.view %t[%c0] : buffer -> view<64xf32>\n"),
        "an alloca in a memory space the check does not know": bk(st + bar + nb,
            alloc="  %t = buffer.alloca<generic> align(16) %bytes : buffer\n  %tv = buffer.view %t[%c0] : buffer -> view<64xf32>\n"),
        "a barrier form other than <workgroup> scope(workgroup)": bk(st + "  kernel.barrier.arrive<workgroup> scope(workgroup)\n" + bar + nb),
        "a workgroup view through an scf.for": bk(st + bar + "  scf.for %i = [%c0 to %n step %c1](%w = %tv : view<64xf32>) {\n" + nb + "  }\n"),
        "a yielded workgroup view": bk(st + bar + "  %u = index.cmp ult, %n, %c32 : index\n  %w = scf.if %u -> (view<64xf32>) {\n"
                                      "  scf.yield %tv : view<64xf32>\n  } else {\n  scf.yield %tv : view<64xf32>\n  }\n" + nb),
        "a callee that synchronises": bk(st + bar + "  func.call @sync() : () -> ()\n" + nb),
        "a callee calling one that synchronises": bk(st + bar + "  func.call @outer() : () -> ()\n" + nb),
        "a callee with a workgroup collective": bk(st + bar + "  func.call @wred() : () -> ()\n" + nb),
        "a callee allocating workgroup memory (view.alloca)": bk(st + bar + "  func.call @lds() : () -> ()\n" + nb),
        "a callee copying asynchronously": bk(st + bar + "  func.call @acopy() : () -> ()\n" + nb),
        "an op line the parser does not read": bk(st + bar + '  "unknown.op"() : () -> ()\n' + nb),
        "an scf.if whose condition is not a name": bk(st + bar + "  scf.if true {\n  }\n" + nb),
        "a conditional kernel.exit (later barriers not reached by all)": bk(st + "  %u = index.cmp ult, %n, %c32 : index\n  kernel.exit %u\n" + bar + nb),
    }
    cal = {"sync": "func.def @sync() {\n" + bar + "  func.return\n}\n",
           "outer": "func.def @outer() {\n  func.call @sync() : () -> ()\n  func.return\n}\n",
           "wred": "func.def @wred() {\n  %a = scalar.constant 1.0 : f32\n  %r = kernel.workgroup.reduce<addf> %a : f32\n  func.return\n}\n",
           "lds": "func.def @lds() {\n  %s = view.alloca<workgroup> align(16) : view<8xf32>\n  func.return\n}\n",
           "acopy": "func.def @acopy(%a: view<8xf32>, %b: view<8xf32>) {\n  %c0 = index.constant 0 : index\n"
                    "  %tok = kernel.async.copy %a[%c0], %b[%c0] : view<8xf32>, view<8xf32> -> kernel.async.token\n  func.return\n}\n"}
    for k, t in NA.items():
        b = kl.barrier_check(t, cal)
        ok(b["status"].startswith("not analysed"), f"barrier: {k} is 'not analysed' ({b['status']})")
    DV = {
        "P8 a barrier under a fragment-derived condition": st + "  %c16 = index.constant 16 : index\n"
            "  %gv = buffer.view %x[%c0] : buffer -> view<16x16xf16>\n"
            "  %f = vector.fragment.load<lhs> %gv[%c0, %c0] shape [%c16, %c16] : view<16x16xf16> -> vector<16xf16>\n"
            "  %e = vector.extract %f[%c0] : vector<16xf16> -> f16\n  %z = scalar.constant 0.0 : f16\n  %p = scalar.cmpf ogt, %e, %z : f16\n"
            "  scf.if %p {\n" + bar + "  }\n" + nb,
        "P8b a barrier under a volatile-load flag (other workgroups write it)": st +
            "  %fl = view.load<volatile> %xv[%c0] : view<64xf32> -> f32\n  %z = scalar.constant 0.0 : f32\n  %p = scalar.cmpf ogt, %fl, %z : f32\n"
            "  scf.if %p {\n" + bar + "  }\n" + nb,
        "a condition from a callee that reads the invocation id": st + "  %w = func.call @lane() : () -> index\n"
            "  %lt = index.cmp ult, %w, %c32 : index\n  scf.if %lt {\n" + bar + "  }\n" + nb,
        "a loop result counting a lane-dependent trip count": st + "  %cnt = scf.for %i = [%c0 to %tid step %c1](%a = %c0 : index) -> (index) {\n"
            "  %a2 = index.add %a, %c1 : index\n  scf.yield %a2 : index\n  }\n  %lt = index.cmp ult, %cnt, %c32 : index\n"
            "  scf.if %lt {\n" + bar + "  }\n" + nb,
    }
    lane = {"lane": "func.def @lane() -> index {\n  %l = kernel.workitem.id x : index\n  func.return %l : index\n}\n"}
    for k, t in DV.items():
        b = kl.barrier_check(bk(t), lane)
        ok(b["status"].startswith("divergent"), f"barrier: {k} is divergent ({b['status']})")
    b = kl.barrier_check(bk(st + "  scf.for %i = [%c0 to %n step %c1] {\n" + bar + "  }\n" + bar + nb))
    ok(wanted(b) == ["orders nothing", "orders"], f"barrier: a barrier after a loop holding a barrier still orders the store before the loop "
       f"(the loop may run zero times) ({wanted(b)})")
    b = kl.barrier_check(bk(st + bar + "  %u = index.cmp ult, %n, %c32 : index\n  scf.if %u {\n  %w = buffer.view %t[%c0] : buffer -> "
                            "view<64xf32>\n  } else {\n  %w = view.load %xv[%c0] : view<64xf32> -> f32\n  %q = scalar.addf %w, %w : f32\n  }\n"))
    ok(wanted(b) == ["orders nothing"], f"barrier: a data value reusing a view's name in a sibling arm is not an access ({wanted(b)})")
    shift = {"shift": "func.def @shift(%b: buffer) -> buffer {\n  func.return %b : buffer\n}\n"}
    b = kl.barrier_check(bk("  %b2 = func.call @shift(%t) : (buffer) -> buffer\n" + bar + "  %lo = buffer.view %b2[%c0] : buffer -> "
                            "view<32xf32>\n  view.store %v, %lo[%c0] : f32, view<32xf32>\n" + bar + "  %c128 = index.constant 128 : index\n"
                            "  %hi = buffer.view %t[%c128] : buffer -> view<32xf32>\n  %r = view.load %hi[%c0] : view<32xf32> -> f32\n"), shift)
    ok(wanted(b) == ["orders", "orders"], f"barrier: a buffer a callee returns has no known base: a view of it overlaps every view "
       f"of the region ({wanted(b)})")
    # shadowing (verify r2 V-B1: Loom allows an inner name to shadow an outer one, hrx-system v0.2.0 parser_test.py
    # test_child_shadows_parent; r2's walk-order binding erased the outer binding for the rest of the walk -> 'orders nothing')
    u = "  %u = index.cmp ult, %n, %c32 : index\n"
    SH = {
        "S1 an if arm shadows the region view's name with an index": st + bar + u + "  scf.if %u {\n  %tv = index.constant 3 : index\n  }\n" + nb,
        "S1b an if arm shadows it with a view of a global buffer": st + bar + u + "  scf.if %u {\n  %tv = buffer.view %x[%c0] : buffer -> "
            "view<64xf32>\n  %q = view.load %tv[%c0] : view<64xf32> -> f32\n  }\n" + nb,
        "S1c a loop body shadows it": st + bar + "  scf.for %i = [%c0 to %n step %c1] {\n  %tv = index.add %i, %c1 : index\n  }\n" + nb,
        "S1d an if arm shadows the alloca root": st + bar + u + "  scf.if %u {\n  %t = index.constant 3 : index\n  }\n"
            "  %tv2 = buffer.view %t[%c0] : buffer -> view<64xf32>\n  %nb = index.sub %c32, %tid : index\n"
            "  %r = view.load %tv2[%nb] : view<64xf32> -> f32\n",
        "S2 an if arm shadows an offset constant (outer 0, arm 128)": "  %c128 = index.constant 128 : index\n  %off = index.constant 0 : index\n"
            "  %lo = buffer.view %t[%c0] : buffer -> view<32xf32>\n  view.store %v, %lo[%tid] : f32, view<32xf32>\n" + bar + u +
            "  scf.if %u {\n  %off = index.constant 128 : index\n  }\n  %hi = buffer.view %t[%off] : buffer -> view<32xf32>\n"
            "  %r = view.load %hi[%tid] : view<32xf32> -> f32\n",
        "S3 an if arm shadows a view by one at another offset": "  %c128 = index.constant 128 : index\n  %w = buffer.view %t[%c0] : buffer -> "
            "view<32xf32>\n  view.store %v, %w[%tid] : f32, view<32xf32>\n" + bar + u + "  scf.if %u {\n  %w = buffer.view %t[%c128] : "
            "buffer -> view<32xf32>\n  }\n  %r = view.load %w[%tid] : view<32xf32> -> f32\n",
    }
    for k, t in SH.items():
        b = kl.barrier_check(bk(t))
        ok(b["status"] == "analysed" and wanted(b) == ["orders"], f"barrier: {k}: the outer binding holds after the scope: orders "
           f"({b['status']}, {wanted(b)})")
    t = bk(st + bar + nb).replace("  %c0 = index.constant 0 : index\n", "  %c0 = index.constant 0 : index\n  %c64 = index.constant 8 : index\n", 1)
    b = kl.barrier_check(t)
    ok(b["status"] == "analysed", f"barrier: workgroup_size reads the launch config's constants, not a launch-body name reuse "
       f"(%c64 = 8 in the body is not 'single wave') ({b['status']})")
    b = kl.barrier_check(bk("  %c124 = index.constant 124 : index\n  %lo = buffer.view %t[%c0] : buffer -> view<32xf32>\n"
                            "  %hi = buffer.view %t[%c124] : buffer -> view<32xf32>\n  view.store %v, %lo[%tid] : f32, view<32xf32>\n"
                            + bar + "  %r = view.load %hi[%tid] : view<32xf32> -> f32\n"))
    ok(wanted(b) == ["orders"], f"barrier: views [0, 128) and [124, 252) overlap by one f32: orders ({wanted(b)})")
    row = {"kernel": "g_b", "s_barrier": 3}
    kl.barriers([row], {"g_b": bk(st + "  %s = kernel.workgroup.reduce<maxnumf> %v : f32\n" + bar + nb)}, {})
    ok(row["barrier"]["status"] == "analysed", f"barrier: a workgroup reduce is two s_barriers in the binary (GEMM: 3 + 2) "
       f"({row['barrier']['status']})")
    row = {"kernel": "g_b", "s_barrier": 2}
    kl.barriers([row], {"g_b": NA["P7 kernel.async.copy into the region (data lands at the wait, not at issue)"]}, {})
    ok(any("not analysed" in x for x in kl.barrier_report(row)), "barrier: a kernel the check cannot model is printed without "
       "--barriers (fail closed, never silent)")
    arms = "  %lt = index.cmp ult, %tid, %c32 : index\n  scf.if %lt {\n" + st + "  } else {\n" + nb + "  }\n" + bar
    b = kl.barrier_check(bk(arms), races=True)
    ok(len(b["races"]) == 1, f"barrier: the two arms of a lane-dependent if run at once: a store in one races a read in the other "
       f"({b['races']})")
    b = kl.barrier_check(bk(st + bar + nb + "  view.store %v, %tv[%c1] : f32, view<64xf32>\n"))
    ok(b["races"] == [], f"barrier: races are computed only when asked (--barriers): super-linear, detail only (review N2: "
       f"187 s / 2 GB at 4,000 accesses) ({b['races']})")
    import json, tempfile as tf
    with tf.TemporaryDirectory() as t:
        js = os.path.join(t, "k.json")
        json.dump({"source_low": {"memory": {"bank_service": {"structural": {"conflicted_packet_count": 4, "extra_round_count": 48}}}}}, open(js, "w"))
        ok(kl.bank(js) == (4, 48), f"bank: conflicted packets and extra rounds from the report ({kl.bank(js)})")
        ok(kl.bank(os.path.join(t, "none.json")) == (None, None), "bank: no report, no figure")
        # ldsbank (E424 / E435 / E456): conflict rounds of the hot loop's LDS packets, placed by matching the report's per-packet rows
        # to the binary's ds_ instructions; an epilogue conflict (E249: the f16 GEMMs' once-per-tile store, 0.2 % measured) is not
        bl = ["s_mov_b32 s0, 0", "ds_store_b128 v1, v[2:5]", "s_waitcnt lgkmcnt(0)", "ds_load_b128 v[6:9], v1", wm(0), "s_add_u32 s0, s0, 1"]
        bl.append(back(len(bl), 1))
        bl += ["ds_store_b128 v1, v[2:5] offset:16", "s_endpgm"]
        def lrow(pk, req, ext):
            return {"memory_space": "workgroup", "packet": "amdgpu." + pk,
                    "bank_service": {"proof": "exact", "required_rounds": req, "extra_rounds": ext}}
        def bank_note(rows_):
            jb = os.path.join(t, "b.json")
            json.dump({"source_low": {"memory_rows": [{"memory_space": "global", "packet": "amdgpu.global_load_b128"}] + rows_}}, open(jb, "w"))
            r_ = kl.lint_listing("banks", *listing(bl))
            kl.ldsbank(r_, jb, bl)
            return r_["bank_loop"], "ldsbank" in r_.get("notes", [])
        got = bank_note([lrow("ds_write_b128", 8, 4), lrow("ds_read_b128", 4, 0), lrow("ds_write_b128", 4, 0)])
        ok(got == ((4, 12), True), f"ldsbank: a 2-way store in the hot loop (4 of 12 rounds) gets the note ({got})")
        got = bank_note([lrow("ds_write_b128", 4, 0), lrow("ds_read_b128", 4, 0), lrow("ds_write_b128", 8, 4)])
        ok(got == ((0, 8), False), f"ldsbank: the same conflict in the epilogue, after the loop, does not ({got})")
        got = bank_note([lrow("ds_write_b128", 8, 4), lrow("ds_read_b128", 400, 0), lrow("ds_write_b128", 4, 0)])
        ok(got == ((4, 408), False), f"ldsbank: 4 conflict rounds of 408 (1.0 %, under LDSBANK) do not ({got})")
        got = bank_note([lrow("ds_write_b128", 8, 4), lrow("ds_read_b128", 4, 0), lrow("ds_write_b128", 4, 0), lrow("ds_write_b128", 4, 0)])
        ok(got == (None, False), f"ldsbank: rows that do not match the binary's instructions place nothing ({got})")
        run = os.path.join(t, "run"); os.makedirs(run)
        with open(os.path.join(run, "events.jsonl"), "w") as f:
            for k, ns in (("g_a", 2_000_000), ("g_a", 1_000_000), ("g_b", 500_000)):
                f.write(json.dumps({"type": "dispatch_event", "valid": True, "key": k, "duration_ns": ns}) + "\n")
            f.write(json.dumps({"type": "dispatch_event", "valid": False, "key": "g_a", "duration_ns": 9_000_000}) + "\n")
        open(os.path.join(run, "x.req.json"), "w").write(json.dumps({"requests": [{"prompt_tokens": 81430}]}))
        ms, prompt = kl.served_ms(run)
        ok(ms == {"g_a": 3.0, "g_b": 0.5} and prompt == 81430, f"served ms per kernel from valid events only ({dict(ms)}, {prompt})")
    jn = lambda nm: (["s_and_saveexec_b64 s[6:7], vcc", "v_mov_b32_e32 v1, 0", "s_mov_b64 exec, s[6:7]"]
                     + [f"v_mov_b32_e32 v{10 + i}, v{60 + i}" for i in range(nm)] + ["s_endpgm"])
    r = kl.lint_listing("join40", *listing(jn(40)))
    ok("join" in r["flags"] and r["join"] == 40, f"40 moves after an exec restore flag join ({r['join']})")
    r = kl.lint_listing("join20", *listing(jn(20)))
    ok("join" not in r["flags"], f"20 do not ({r['join']}, flags {r['flags']})")
    algo = ["s_mov_b32 s0, 0", "v_dot4_i32_iu8 v1, v2, v3, v1"] + ["v_cmp_lt_i32_e64 s1, v4, v5", "v_cndmask_b32_e64 v6, v7, v8, s1"] * 10
    algo.append(back(len(algo), 1))
    r = kl.lint_listing("algo", *listing(algo))
    ok("select" not in r["flags"], f"selects without equality constants (an algorithm, not a lookup) do not flag select (E244: {r['eq_imm']} constants)")
    pf = ["s_mov_b32 s0, 0", "global_load_b128 v[1:4], v0, s[2:3]", "global_load_b128 v[5:8], v0, s[2:3]"] + \
         ["v_dot4_i32_iu8 v9, v10, v11, v9"] * 6 + ["s_waitcnt vmcnt(1)", "v_add_nc_u32_e32 v1, v1, v2"]
    pf.append(back(len(pf), 1))
    r = kl.lint_listing("cover", *listing(pf))
    ok(r["cover_min"] == 8 and r["in_flight"] == 1, f"load cover: first load retired 8 instructions after issue, one in flight at the back-edge ({r['cover_min']}, {r['in_flight']})")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
