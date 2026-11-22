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
    r = kl.lint_listing("run10", *listing(moves_loop(10)))
    ok("copies" not in r["flags"], f"a 10-move run does not (run {r['run']}, flags {r['flags']})")
    row = lambda k, op, wm=0, dt=8: {"kernel": k, "where": "loop", "wmma": wm, "dot": dt, "other_per_op": op, "flags": []}
    rows = [row("g_mv_a", 3.0), row("g_mv_b", 3.2), row("g_mv_c", 2.9), row("g_mv_d", 8.6),
            row("g_attention_x", 30.0, 32, 0), row("g_gemm_a", 10.0, 32, 0), row("g_gemm_b", 11.0, 32, 0), row("g_gemm_c", 10.5, 32, 0)]
    hit = kl.outliers(rows)
    ok(set(hit) == {"g_mv_d"}, f"outlier: the matvec at 2.8x its peers, not the attention kernel beside GEMMs ({sorted(hit)})")
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
    pf = ["s_mov_b32 s0, 0", "global_load_b128 v[1:4], v0, s[2:3]", "global_load_b128 v[5:8], v0, s[2:3]"] + \
         ["v_dot4_i32_iu8 v9, v10, v11, v9"] * 6 + ["s_waitcnt vmcnt(1)", "v_add_nc_u32_e32 v1, v1, v2"]
    pf.append(back(len(pf), 1))
    r = kl.lint_listing("cover", *listing(pf))
    ok(r["cover_min"] == 8 and r["in_flight"] == 1, f"load cover: first load retired 8 instructions after issue, one in flight at the back-edge ({r['cover_min']}, {r['in_flight']})")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
