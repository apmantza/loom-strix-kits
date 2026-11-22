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
  - the 256-row GEMMs' LDS-to-WMMA serialization (E254: 48 of 64 WMMAs right after lgkmcnt(0), queue item 6 / E152) not
    reported: a loop whose WMMAs each wait lgkmcnt(0) on a fragment just loaded gets the `ldsser` note (a note until a counter
    run shows its cost), one that keeps a fragment in flight with lgkmcnt(2) does not;
  - reductions serialised one barrier pair at a time (E272: a depth-mix draft with 40 barriers) not noted `barriers`: ten
    straight-line barriers get the note, the same barriers inside a loop (a staged pipeline) do not;
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
  - the outer shell of a nested loop taken as the hot loop (E237: the 16-token swiglu matvecs were flagged `select` for
    their outer loop's selects while the dot4s sat in the inner loop): the inner loop must be chosen, nothing flagged.
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
    ser = sum((["ds_load_b128 v[30:33], v8 offset:" + str(32 * k), "ds_load_b128 v[34:37], v8 offset:" + str(32 * k + 16),
                "s_waitcnt lgkmcnt(0)", wm(k)] for k in range(8)), [])
    r = kl.lint_listing("ldsser", *listing(loop_of(ser)))
    ok("ldsser" in r.get("notes", []) and "ldsser" not in r["flags"] and r["ldsser"] == 1.0,
       f"WMMAs each waiting lgkmcnt(0) on a just-loaded fragment get the ldsser note, not a flag (share {r.get('ldsser')})")
    kept = ["ds_load_b128 v[30:33], v8", "ds_load_b128 v[34:37], v8 offset:16"]
    for k in range(8):
        kept += [f"ds_load_b128 v[50:53], v8 offset:{32 * k + 32}", f"ds_load_b128 v[54:57], v8 offset:{32 * k + 48}", "s_waitcnt lgkmcnt(2)", wm(k)]
    r = kl.lint_listing("ldskept", *listing(loop_of(kept)))
    ok("ldsser" not in r.get("notes", []), f"one fragment kept in flight with lgkmcnt(2): no note (share {r.get('ldsser')})")
    far = [f"ds_load_b128 v[{30 + 4 * (k % 2)}:{33 + 4 * (k % 2)}], v8 offset:{16 * k}" for k in range(8)] + ["v_add_f32_e32 v1, v1, v2"] * 14
    far += sum(([f"s_load_b32 s{10 + k}, s[4:5], 0x{4 * k:x}", "s_waitcnt lgkmcnt(0)", wm(k)] for k in range(8)), [])
    r = kl.lint_listing("ldsfar", *listing(loop_of(far)))
    ok("ldsser" not in r.get("notes", []), f"lgkmcnt(0) waits on scalar loads, the LDS loads long before: no note (share {r.get('ldsser')})")
    sr = ["s_mov_b32 s0, 0"] + sum((["ds_store_b32 v1, v2", "s_barrier", "ds_load_b32 v3, v1", "v_add_f32_e32 v4, v4, v3"] for _ in range(10)), [])
    r = kl.lint_listing("serial_reduce", *listing(sr + ["s_endpgm"]))
    ok("barriers" in r.get("notes", []) and r["bar"] == 10, f"ten straight-line barriers get the barriers note (bar {r.get('bar')})")
    r = kl.lint_listing("pipelined", *listing(loop_of(sum((["ds_store_b32 v1, v2", "s_barrier", "ds_load_b32 v3, v1"] for _ in range(10)), []))))
    ok("barriers" not in r.get("notes", []), f"barriers inside a loop (a staged pipeline) do not (bar {r.get('bar')})")
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
    import json, tempfile as tf
    with tf.TemporaryDirectory() as t:
        js = os.path.join(t, "k.json")
        json.dump({"source_low": {"memory": {"bank_service": {"structural": {"conflicted_packet_count": 4, "extra_round_count": 48}}}}}, open(js, "w"))
        ok(kl.bank(js) == (4, 48), f"bank: conflicted packets and extra rounds from the report ({kl.bank(js)})")
        ok(kl.bank(os.path.join(t, "none.json")) == (None, None), "bank: no report, no figure")
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
