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
    (KV8_SPLIT_VZERO), a clamped key (g_dfl_ctx_split), no softmax (g_dfl_topk) or a -1.0 fill do not;
  - two occupancy figures for one kernel (review r1 F1: lint_counters (a) printed 11.25, (d2) 'unverified'), a residency tier the
    hardware cannot reach (HRX-like E533: flash2 reported next tier 7; reachable is 9), a fraction shown as a figure (F3), or a
    next tier the cut does not land on (F2: g_dfl_ctx_split's -3 VGPRs reaches 12, not 11): occupancy() is the one seam; flash2's
    report gives 6 -> 9 at -72 VGPRs and -18,944 B together; a 160-lane matvec gives an integer tier marked bound; g_dfl_ctx_split
    and g_head_mv_t8_q8 (live rows) name the tier reached; a real compile's report is verified through the device file;
  - live registers read by hand from RGA's .live files (E527 / E531; E533: flash2's loop never drops below 159 live VGPRs, so 168
    needs a new organisation): flash2_f16o's RGA output gives peak 229, loop floor 159, 766 loop instructions over 168; a row
    missing or not ours, or RGA absent, gives no number.
  - C18's notes (11-25 lessons): a one-round grid not said (E539: a study to learn that g_gdn_recurrence_pf's 384 one-wave
    workgroups are all resident at spans = 1), LDS loads drained right after issue not said (E534: the m256 GEMMs keep none in
    flight across a wait, hipBLASLt up to 16), a guard's wait draining a prefetch issued before the branch not caught on a single
    build (E520: STAGE_LIVE x FLATADDR), VOPD pairs left unformed not counted (E551: flash2's back-edge 65 moves -> 31 pairs);
    each with near misses (two rounds, a runtime grid, an unmodelled op; loads kept in flight or issued early; loads issued after
    the join, read inside the block or issued inside it, an exit test; one-parity destinations, dependent moves, modifiers).
usage: python3 kit/test_kernel_lint.py   (needs loom-compile: VENV from the environment or the machine env file)"""
import json
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


def occ(wg, vgpr, vtier, lds, ltier, slots, cur, cap=16, extra=()):
    """occupancy() inputs as a Loom 0.2.0 gfx1151 report carries them (VGPR pool 1536 in granules of 24 per SIMD, LDS 131072 per
    WGP in granules of 1024, the slot row's tier)"""
    rows = [{"name": "amdgpu.sgpr", "kind": "unconstrained_resource", "allocation_scope": "subgroup", "independent_tier": 16},
            {"name": "amdgpu.vgpr", "kind": "pooled_resource", "unit": "registers", "allocation_scope": "subgroup", "pool_units": 1536,
             "allocation_granularity": 24, "units": vgpr, "independent_tier": vtier},
            {"name": "amdgpu.lds", "kind": "pooled_resource", "unit": "bytes", "allocation_scope": "workgroup", "pool_units": 131072,
             "allocation_granularity": 1024, "units": lds, "independent_tier": ltier},
            {"name": "amdgpu.workgroup_slots", "kind": "fixed_limit", "independent_tier": slots}] + list(extra)
    return {"wg": wg, "wave": 32, "cap": cap, "tier": cur, "rows": rows, "target": "gfx1151"}


def occupancy_seam():
    """kernel_lint.tier_state() and occupancy(), its lever view: the one occupancy seam (review r1 F1-F3); live rows from pkg-e480's reports"""
    o = kl.residency(os.path.join(HERE, "fixtures", "flash2_f16o_residency.json"))     # pkg-cmb2's flash2_f16o report, trimmed
    ok(o[0] == 237 and o[1].get("tier") == 6 and not o[1].get("bound") and o[1].get("next") == 9 and
       o[1].get("needs") == [("vgpr", 237, "registers", 168, 69), ("lds", 61952, "bytes", 43008, 18944)],
       f"occupancy: flash2_f16o (384 lanes) 6 -> 9 at -69 VGPRs and -18,944 B LDS together, not Loom's 7 at -21 ({o})")
    o = kl.tier_state(occ(160, 114, 12, 2560, 16, 15, 12), 4)                   # g_mv_q5_k_k5120_t8_q8
    ok(o.get("tier") == 12 and o.get("bound") is True and o.get("next") == 15 and o.get("needs") == [("vgpr", 114, "registers", 96, 18)],
       f"occupancy: a 160-lane matvec (5 waves over 4 SIMDs) is verified at tier 12 = ceil(9 x 5 / 4), a bound, not 'unverified' "
       f"or 11.25 ({o})")
    ok(kl.tier_text(o) == "12 bound" and kl.next_text({"occ": o}) == "15 bound:v18", f"occupancy: printed '12 bound', never a "
       f"fraction ({kl.tier_text(o)}, {kl.next_text({'occ': o})})")
    o = kl.tier_state(occ(128, 123, 10, 0, 16, 16, 10), 4)                       # g_dfl_ctx_split
    ok(o.get("tier") == 10 and not o.get("bound") and o.get("next") == 12 and o.get("needs") == [("vgpr", 123, "registers", 120, 3)],
       f"occupancy: g_dfl_ctx_split's -3 VGPRs (123 -> 120) lands on 12 waves, not 11 (review F2) ({o})")
    o = kl.tier_state(occ(160, 114, 12, 10240, 15, 15, 12), 4)                   # g_head_mv_t8_q8
    ok(o.get("tier") == 12 and o.get("next") == 15, f"occupancy: g_head_mv_t8_q8 12 -> 15 after -18 VGPRs (review F2: was 12.5) ({o})")
    o = kl.tier_state(occ(32, 240, 6, 9216, 4, 16, 4), 4)                       # v_attn_72x16_s3456_0_1152_2304 (1 wave)
    ok(o.get("tier") == 4 and o.get("next") == 5 and o.get("needs") == [("lds", 9216, "bytes", 7168, 2048)],
       f"occupancy: a 1-wave workgroup's next tier needs 17 workgroups (LDS <= 7,168 B), not 15 or 16 that stay at tier 4 "
       f"(verify r2 V-F1: next 4 at -1,024 B) ({o})")
    o = kl.tier_state(occ(384, 252, 5, 61952, 6, 15, 5), 4)                      # flash2_partial
    ok(o.get("tier") == 3 and o.get("report_tier") == 5 and o.get("next") == 6, f"occupancy: flash2_partial is 3, Loom says 5 ({o})")
    o = kl.tier_state(occ(544, 30, 16, 544, 16, 13, 13), 4)                      # g_mv_iq4_nl_k17408_q8_r8: 3 workgroups, the wave cap
    ok(o.get("tier") == 13 and o.get("next") is None and not o.get("blocked") and kl.next_text({"occ": o}) == "best",
       f"occupancy: a 17-wave matvec at 3 workgroups is at its best tier 13 ({o})")
    o = kl.tier_state(occ(128, 24, 16, 0, 16, 8, 8), 4)
    ok(o.get("tier") == 8 and o.get("next") is None and "workgroup_slots" in (o.get("blocked") or "") and
       kl.next_text({"occ": o}) == "blocked", f"occupancy: a slot tier below the cap blocks the next tier, named ({o})")
    o = kl.tier_state(occ(384, 100, 12, 43500, 6, 15, 6), 4)
    ok(o.get("tier") == 6 and o.get("next") == 9 and o.get("needs") == [("lds", 43500, "bytes", 43008, 492)], f"occupancy: LDS counts in 1,024-B granules: 3 x 43,500 B would fit unrounded ({o})")
    o = kl.tier_state(occ(160, 237, 6, 0, 16, 15, 6, extra=[{"name": "amdgpu.agpr", "kind": "pooled_resource", "allocation_scope": "wave"}]), 4)
    ok("unverified" in o.get("why", ""), f"occupancy: an unknown resource row is unverified, never a figure ({o})")
    o = kl.tier_state(occ(384, 240, 7, 61952, 6, 15, 6), 4)
    ok("unverified" in o.get("why", ""), f"occupancy: a resource tier that recomputes differently is unverified ({o})")
    o = kl.tier_state(occ(384, 240, 6, 61952, 6, 15, 7), 4)
    ok("unverified" in o.get("why", "") and "says 7" in o["why"], f"occupancy: every row agrees but the report's current tier does not: "
       f"unverified ({o})")
    ok("unverified" in kl.tier_state(occ(384, 240, 6, 61952, 6, 15, 6), None).get("why", ""), "occupancy: no SIMD count: unverified")
    # device_simds() reads common/devices/<target_key>.json (verify r2 V7: a constant 4 passed every gfx1151 test)
    ok(kl.device_simds("no_such_target") is None and kl.device_simds(None) is None, "device_simds: no device file: None (unverified)")
    import importlib.util, json, shutil, tempfile
    with tempfile.TemporaryDirectory() as t:
        shutil.copy(kl.__file__, t)
        os.makedirs(os.path.join(t, "devices"))
        json.dump({"simds_per_occupancy_domain": 2}, open(os.path.join(t, "devices", "fake2.json"), "w"))
        sp = importlib.util.spec_from_file_location("kl_fake2", os.path.join(t, "kernel_lint.py"))
        k2 = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(k2)
        o = k2.tier_state(dict(occ(64, 48, 16, 16384, 8, 16, 8), target="fake2"), k2.device_simds("fake2"))
        ok(k2.device_simds("fake2") == 2 and o.get("tier") == 8,
           f"device_simds: a device file with 2 SIMDs gives 2, and a 64-lane workgroup's tier 8 verifies with it ({o})")
    o = kl.tier_state(occ(384, 240, 6, 61952, 6, 15, 6, extra=[{"name": "amdgpu.sgpr", "kind": "pooled_resource", "allocation_scope":
                                                                 "subgroup", "pool_units": 800, "allocation_granularity": 16,
                                                                 "units": 106, "independent_tier": 7}]), 4)
    ok("unverified" in o.get("why", ""), f"occupancy: a pooled SGPR row (not one Loom 0.2.0 emits) is unverified, not modelled ({o})")
    # occupancy(): the lever contract loom_suggest (C6, suggest-1125) reads: None at the best tier, {"why"} unverified or blocked
    # ('next tier ... blocked', its phantom test), else the dict with "bound"
    ok(kl.occupancy(occ(544, 30, 16, 544, 16, 13, 13), 4) is None, "occupancy: None at the best tier (C6's contract)")
    o = kl.occupancy(occ(128, 24, 16, 0, 16, 8, 8), 4)
    ok(o.get("why", "").startswith("next tier") and "blocked" in o["why"] and o.get("tier") == 8,
       f"occupancy: blocked is {{'why': 'next tier ... blocked ...'}} with the tier kept ({o})")
    o = kl.occupancy(occ(160, 114, 12, 2560, 16, 15, 12), 4)
    ok(o.get("bound") is True and o.get("next") == 15 and o.get("q") == 1.25, f"occupancy: a ceil() tier carries bound: True ({o})")
    o = kl.occupancy(occ(384, 240, 6, 61952, 6, 15, 6), 4)
    ok(o.get("bound") is False and o.get("q") == 3 and o.get("next") == 9, f"occupancy: whole waves per SIMD: bound False, q an int ({o})")
    ok("why" in kl.occupancy(occ(384, 240, 7, 61952, 6, 15, 6), 4), "occupancy: unverified passes {'why'} through")
    f2 = {"vgpr": 240, "occ": kl.tier_state(occ(384, 240, 6, 61952, 6, 15, 6), 4)}
    ok(kl.live_budget(f2) == 168, f"livereg budget: flash2's next tier (9) allows 168 VGPRs, not today's 240 ({kl.live_budget(f2)})")
    vs = {"vgpr": 256, "occ": kl.tier_state(occ(192, 256, 5, 49664, 3, 15, 3), 4)}          # g_attention_vsplit: LDS alone holds it
    ok(kl.live_budget(vs) == 256, f"livereg budget: only LDS must move: today's VGPRs ({kl.live_budget(vs)})")
    ok(kl.live_budget({"vgpr": 30, "occ": kl.tier_state(occ(544, 30, 16, 544, 16, 13, 13), 4)}) is None,
       "livereg budget: none at the best tier")


def livereg():
    """RGA live registers (E527 / E531 / E533) on pkg-cmb2's flash2_f16o: its listing (address, instruction: loop_census.disasm of the
    code object) and RGA 2.15's --livereg output (register map blanked) in fixtures/ (text: the export check refuses binaries)"""
    rows = kl.live_rows(open(os.path.join(HERE, "fixtures", "flash2_f16o_cmb2.live")).read())
    lst = [x.split("\t", 1) for x in open(os.path.join(HERE, "fixtures", "flash2_f16o_cmb2.listing")).read().split("\n") if x]
    ins, addr = [i for _, i in lst], [int(a, 16) for a, _ in lst]
    x = kl.live(rows, ins, addr, 168)
    ok(len(rows) == len(ins) == 4692 and x.get("peak") == 229 and x.get("floor") == 159 and x.get("loop") == (617, 1519),
       f"livereg: flash2_f16o peaks at 229 live VGPRs (RGA's 'Maximum # VGPR used'), its hot loop never drops below 159 "
       f"({len(rows)} rows, {len(ins)} instructions, {dict((k, v) for k, v in x.items() if k != 'at')})")
    ok(x.get("loop_peak") == 229 and x.get("over") == 766, f"livereg: 766 of the loop's 903 instructions above the 168-VGPR budget "
       f"of 9 waves/SIMD ({x.get('over')}, peak at {x.get('at')})")
    ok(rows[81] == (5, "s_delay_alu instid0(VALU_DEP_1) | instid1(VALU_DEP_2)") and rows[617][1].startswith("s_cmp_lt_i32"),
       f"livereg: an instruction holding '|' and a labelled one parse whole ({rows[81]}, {rows[617]})")
    y = kl.live(rows[:-1], ins, addr, 168)                                # the last row missing: every other one still pairs
    ok("error" in y, f"livereg: a row missing (the last) gives no figure ({y})")
    z = kl.live(rows[:5] + [(rows[5][0], "v_nop")] + rows[6:], ins, addr, 168)
    ok("error" in z, f"livereg: a row that is not our instruction gives no figure ({z})")
    d = kl.live([(3, "v_add_f32 v1, v2, v3 row_shr:1 row_mask:0xf"), (2, "v_max_f32 v1, v2, v3")],
                ["v_add_f32_dpp v1, v2, v3 row_shr:1 row_mask:0xf", "v_max_f32_e64_dpp v1, v2, v3"], [0, 8], None)
    ok(d == {"peak": 3}, f"livereg: RGA writes v_add_f32 where we decode v_add_f32_dpp / _e64_dpp: the same instruction ({d})")
    ok(kl.live_text({"live": {"error": "rga: x"}}) == "unknown" and kl.live_text({}) == "-" and
       kl.live_text({"live": x}) == "229/159/766", f"livereg: the live column ({kl.live_text({'live': x})})")
    ok(kl.rga_live("/nonexistent.hsaco", "") == {"error": "rga: not found (--rga, or RGA in the environment / machine env file)"},
       "livereg: no RGA is 'unknown (rga: not found)', never a number")
    import tempfile
    with tempfile.TemporaryDirectory() as t:                 # RGA is another process: a stub stands in at that boundary only
        live = open(os.path.join(HERE, "fixtures", "flash2_f16o_cmb2.live")).read()
        open(os.path.join(t, "k.live.src"), "w").write(live)
        for name, extra in (("one", ""), ("two", 'cp "$src" "$(dirname "$out")/k_1.live"\n')):
            stub = os.path.join(t, name)
            open(stub, "w").write(f'#!/bin/sh\nsrc={t}/k.live.src\nout=$6\ncp "$src" "$out"\n' + extra)
            os.chmod(stub, 0o755)
        got = kl.rga_live("x.hsaco", os.path.join(t, "one"))
        ok(isinstance(got, list) and len(got) == 4692, f"livereg: one .live file from RGA is read ({type(got).__name__})")
        got = kl.rga_live("x.hsaco", os.path.join(t, "two"))
        ok(isinstance(got, dict) and "2 .live files" in got.get("error", ""), f"livereg: two .live files (a multi-kernel object) give "
           f"no figure, never the first one's ({got if isinstance(got, dict) else len(got)})")


def lesson_notes():
    """C18's notes (11-25 lessons; r2 after review E581), each with a fixture that fires, near misses that do not, and a test for
    every guard (a guard no test can red is deleted)"""
    def listing(body):
        return body, [4 * i for i in range(len(body))]
    def back(i, target):
        return f"s_branch {(target - (i + 1)) & 0xFFFF}"
    def fwd(i, target, op="s_cbranch_scc0"):
        return f"{op} {(target - (i + 1)) & 0xFFFF}"
    def loop_of(body):
        b = ["s_mov_b32 s0, 0"] + body
        return b + [back(len(b), 1)]
    wm = "v_wmma_f32_16x16x16_f16 v[200:207], v[192:199], v[184:191], v[200:207]"
    notes = lambda r: r.get("notes", [])
    # ---- (a) rounds (E539: a worker's study established that g_gdn_recurrence_pf runs one round at spans 1). Review r2, option
    # iii: information only (grid at size 1, resident, the one-round bound in the size variable), never a note, no served-size claim
    def prog(tiles=8, extra="", grid="%heads, %tiles, %spans", body=None, params="%spans: index"):
        body = body if body is not None else ("  %spans = config.get @llm.spans : index\n"
                                              "  kernel.launch @g_r[%spans](%spans) : [index](index)\n")
        return (f'kernel.def export("g_r") @g_r({params}) {{\n  %c1 = index.constant 1 : index\n  %heads = index.constant 48 : index\n'
                f"  %tiles = index.constant {tiles} : index\n  %c32 = index.constant 32 : index\n{extra}"
                f"  kernel.launch.config workgroups({grid}) workgroup_size(%c32, %c1, %c1) : index\n}} launch(%spans_arg: index) {{\n}}\n"
                f"command.def public target(@device) @llm_advance() launch(%x: buffer) {{\n{body}}}\n")
    gdn = kl.tier_state(occ(32, 166, 9, 0, 16, 16, 9), 4)            # pkg-e480's g_gdn_recurrence_pf report rows: 36 per WGP, 720
    def rnd(text, o=gdn):
        rows = [{"kernel": "g_r", "occ": o}]
        try:
            kl.rounds(rows, [text])
        except TypeError:                                             # r2's rounds(rows, texts, sizes)
            kl.rounds(rows, [text], {})
        line = getattr(kl, "rounds_info", lambda r: None)(rows[0])
        return rows[0].get("rounds") or {}, line, "rounds" in notes(rows[0])
    g, line, n = rnd(prog())
    ok(not n and g.get("resident") == 720 and g.get("sites") == [("@llm.spans", 384, 1)] and line and "one round while spans <= 1" in line
       and "384 workgroups at size 1" in line, f"rounds: 48 x 8 x spans one-wave workgroups, 720 resident: an information line, one "
       f"round while spans <= 1 (E539), never a note ({g}; {line})")
    g, line, n = rnd(prog(tiles=16))
    ok(g.get("sites") == [("@llm.spans", 768, None)] and line is None, f"rounds: 16 tiles (768 at size 1) exceed 720: no line ({g})")
    tok = "  %t = config.get @llm.tokens : index\n  kernel.launch @g_r[%t](%t) : [index](index)\n"
    g, line, n = rnd(prog(params="%t: index", grid="%t, %c1, %c1", body=tok))
    ok(g.get("sites") == [("@llm.tokens", 1, 720)] and line is None, f"rounds: one workgroup at size 1 has no order: no line, bound "
       f"720 kept (above 2, review R12) ({g}; {line})")
    g, line, n = rnd(prog(params="%t: index", grid="%t, %c1, %c1", body=tok.replace("[%t](%t)", "[%t2](%t2)").replace(
        "kernel.launch", "%c2 = index.constant 2 : index\n  %t2 = index.mul %t, %c2 : index\n  kernel.launch")))
    ok(g.get("sites") == [("@llm.tokens", 2, 360)] and line and "tokens <= 360" in line and "2 workgroups" in line,
       f"rounds: 2 x tokens: 2 workgroups at size 1, one round while tokens <= 360 ({g}; {line})")
    sel = ("template.decl @sel1_g_r(%g0: index, %a0: index)\n"
           "template.def<@sel1_g_r> target(@rdna3) @sel1_g_r_rdna3(%g0: index, %a0: index) {\n"
           "  kernel.launch @g_r[%g0](%a0) : [index](index)\n  template.return\n}\n")
    g, line, n = rnd(sel + prog(body="  %two = index.constant 1 : index\n  template.apply<@sel1_g_r>(%two, %two) : (index, index)\n"))
    ok(g.get("sites") == [(None, 384, 1)] and line and "constant grid" in line, f"rounds: a constant grid through a dual-target "
       f"selector ({g}; {line})")
    g, line, n = rnd(prog(extra="  %sh = index.shl %tiles, %c1 : index\n", grid="%heads, %sh, %spans"))
    ok(g.get("unknown") == 1 and line is None, f"rounds: an op the evaluator does not model in the config: unknown ({g})")
    g, line, n = rnd(prog(body="  %s0 = config.get @llm.spans : index\n  %spans = index.shl %s0, %s0 : index\n"
                               "  kernel.launch @g_r[%spans](%spans) : [index](index)\n"))
    ok(g.get("unknown") == 1, f"rounds: an op the evaluator does not model in the launcher: unknown, never a size (R13) ({g})")
    g, line, n = rnd(prog(body="  %spans = config.get @llm.spans : index\n  %spans = index.constant 2 : index\n"
                               "  kernel.launch @g_r[%spans](%spans) : [index](index)\n"))
    ok(g.get("unknown") == 1, f"rounds: a launcher name defined twice: unknown, never either definition (R6) ({g})")
    g, line, n = rnd(prog(body="  %c4 = index.constant 4 : index\n  kernel.launch @g_r[%c4, %c4](%c4) : [index, index](index)\n"))
    ok(g.get("unknown") == 1, f"rounds: a launch with more grid operands than the config's parameters: unknown (R7) ({g})")
    g, line, n = rnd(prog(grid="%heads, %c0, %c1", extra="  %c0 = index.constant 0 : index\n"))
    ok(g.get("unknown") == 1, f"rounds: an empty constant grid: unknown (R3) ({g})")
    g, line, n = rnd(prog(extra="  %c99 = index.constant 99 : index\n  %dn = index.sub %c99, %spans : index\n", grid="%heads, %dn, %c1"))
    ok(g.get("unknown") == 1, f"rounds: a grid that shrinks as the size grows: unknown (R2) ({g})")
    g, line, n = rnd(prog(params="%spans: index, %t: index", grid="%spans, %t, %c1",
                          body="  %spans = config.get @llm.spans : index\n  %t = config.get @llm.tokens : index\n"
                               "  kernel.launch @g_r[%spans, %t](%spans) : [index, index](index)\n"))
    ok(g.get("unknown") == 1 and line is None, f"rounds: a grid of two size keys: unknown, no single bound ({g})")
    g, line, n = rnd(prog(extra="  %sp = index.assume %spans [range(%spans, 1, 8)] : index\n", grid="%heads, %tiles, %sp"))
    ok(g.get("sites") == [("@llm.spans", 384, 1)], f"rounds: index.assume passes its operand through (R8) ({g})")
    two = ("  %spans = config.get @llm.spans : index\n  kernel.launch @g_r[%spans](%spans) : [index](index)\n"
           "  %x = index.shl %spans, %spans : index\n  kernel.launch @g_r[%x](%x) : [index](index)\n")
    g, line, n = rnd(prog(body=two))
    ok(len(g.get("sites", [])) == 1 and g.get("unknown") == 1 and line is None, f"rounds: one site resolved, another unknown: no "
       f"line ({g})")
    g, line, n = rnd(prog(), kl.tier_state(occ(32, 166, 9, 0, 16, 16, 8), 4))
    ok(not g and line is None, f"rounds: an unverified tier ({{'why'}}, no target) gives no figure (R1 / N14) ({g})")
    rows = [{"kernel": "g_a", "rounds": {"sites": [], "unknown": 1, "size_unknown": 0}},     # size_unknown: r2's shape, for the red run
            {"kernel": "g_b", "rounds": {"sites": [("@llm.tokens", 5, 1)], "unknown": 0, "size_unknown": 0}}]
    s = getattr(kl, "rounds_summary", lambda r: "")(rows)
    ok("not resolved (1): g_a" in s and "g_b" not in s, f"rounds: one summary line names the kernels not resolved (review F5) ({s})")
    ok(getattr(kl, "device_domains", lambda t: None)("gfx1151") == 20 and kl.device_domains("nonesuch") is None, "rounds: 20 WGPs")
    ok(not hasattr(kl, "served_sizes"), "rounds: no admitted-size reader (review r2: Loom admits tokens / spans up to the prefill "
       "capacity; rows and draft_depth are not caps)")
    # ---- (c) guardwait (E520). r2 (review F4): a victim is younger than every load the block's wait must retire for the block
    def guard(v_after=False, read_v=False, scale_only=False, write_v=False, scratch=False):
        b = ["s_mov_b32 s0, 0"]
        vl = [f"global_load_b128 v[{40 + 4 * k}:{43 + 4 * k}], v2, s[8:9] offset:{16 * k}" for k in range(3)]
        if scratch:
            vl = ["scratch_load_b128 v[40:43], off, s33 offset:16"]
        b += ([] if v_after or scale_only else vl) + ["s_cmp_eq_u32 s1, 0"]
        br = len(b); b.append("")
        b += ["s_waitcnt vmcnt(0)", "v_cvt_f32_f16_e32 v11, v10"]
        b += ["v_add_f32_e32 v12, v40, v44", "v_add_f32_e32 v12, v48, v12"] if read_v else []
        b += ["v_mov_b32_e32 v40, 0", "v_mov_b32_e32 v44, 0", "v_mov_b32_e32 v48, 0"] if write_v else []
        b[br] = fwd(br, len(b))
        b += ([] if not v_after or scale_only else vl + ["s_waitcnt vmcnt(0)"]) + ["v_add_f32_e32 v13, v40, v44", wm,
                                                                                  "global_load_b32 v10, v3, s[8:9]"]
        b.append(back(len(b), 1))
        return b
    gw = lambda b: (lambda r: (r.get("guard_waits"), "guardwait" in notes(r)))(kl.lint_listing("gw", *listing(b)))
    w, n = gw(guard())
    ok(n and w == [(6, 5, 3)], f"guardwait: the guard waits for the older carried scale; V's three newer loads, read after the "
       f"join, are drained (E520) ({w})")
    gs = guard()
    gs.insert(gs.index("v_cvt_f32_f16_e32 v11, v10") + 1, "global_store_b32 v7, v11, s[38:39]")
    gs[5] = fwd(5, gs.index("v_add_f32_e32 v13, v40, v44"))
    gs[-1] = back(len(gs) - 1, 1)
    w, n = gw(gs)
    ok(n and w == [(6, 5, 3)], f"guardwait: a global store in the block does not excuse a wait that needs the carried scale (only "
       f"the loads carried over the back-edge, seen on the second pass, show the need) ({w})")
    w, n = gw(guard(v_after=True))
    ok(not n, f"guardwait: V's loads issued after the join (STAGE_LIVE alone): no note ({w})")
    w, n = gw(guard(scale_only=True))
    ok(not n, f"guardwait: the wait retires only the scale the block reads: no note ({w})")
    w, n = gw(guard(read_v=True))
    ok(not n, f"guardwait: the block reads V itself: no note ({w})")
    w, n = gw(guard(write_v=True))
    ok(not n, f"guardwait: the block overwrites V's registers (a write counts as a read, review F4 / b023): no note ({w})")
    w, n = gw(guard(scratch=True))
    ok(n and w and w[0][2] == 1, f"guardwait: a spill reload drained the same way counts (scratch loads share vmcnt) ({w})")
    inb = ["s_mov_b32 s0, 0"] + [f"global_load_b128 v[{40 + 4 * k}:{43 + 4 * k}], v2, s[8:9] offset:{16 * k}" for k in range(3)]
    inb += ["s_cmp_eq_u32 s1, 0", ""]
    inb += ["global_load_b32 v60, v2, s[8:9]", "s_waitcnt vmcnt(0)", "v_mov_b32_e32 v61, 0"]
    inb[5] = fwd(5, len(inb))
    inb += ["ds_store_b32 v1, v60", "v_add_f32_e32 v13, v40, v44", wm]
    inb.append(back(len(inb), 1))
    w, n = gw(inb)
    ok(not n, f"guardwait: the wait is for a load issued inside the block, the newest; in-order vmcnt cannot spare the older V "
       f"loads (llama.cpp sample, review F4): no note ({w})")
    st = ["s_mov_b32 s0, 0"] + [f"global_load_b128 v[{40 + 4 * k}:{43 + 4 * k}], v2, s[8:9] offset:{16 * k}" for k in range(3)]
    st += ["s_cmp_eq_u32 s1, 0", "", "s_waitcnt vmcnt(0)", "global_store_b128 v7, v[20:23], s[38:39]"]
    st[5] = fwd(5, len(st))
    st += ["v_add_f32_e32 v13, v40, v44", wm]
    st.append(back(len(st), 1))
    w, n = gw(st)
    ok(not n, f"guardwait: a wait that retires no load the block needs, before a global store (may order the store after a pending "
       f"load; live g_gdn_recurrence_pf / _rec): no note ({w})")
    fen = [x.replace("global_store_b128 v7, v[20:23], s[38:39]", "s_barrier") for x in st]
    w, n = gw(fen)
    ok(not n, f"guardwait: the same wait before a barrier (release: the wait may order it): no note ({w})")
    lds = [x.replace("s_waitcnt vmcnt(0)", "s_waitcnt vmcnt(2)").replace("global_store_b128 v7, v[20:23], s[38:39]", "ds_store_b128 v7, v[20:23]")
           for x in st]
    w, n = gw(lds)
    ok(n and w == [(6, 5, 1)], f"guardwait: the same over-wait before an LDS store (no global memory to order): the oldest V load is "
       f"drained for nothing the block needs (pkg-cmb2's partial kernels, E521's residue) ({w})")
    def blk(block, pre=()):                       # V's three loads (and PRE) before a guard holding BLOCK; V read after the join
        b = ["s_mov_b32 s0, 0"] + [f"global_load_b128 v[{40 + 4 * k}:{43 + 4 * k}], v2, s[8:9] offset:{16 * k}" for k in range(3)]
        b += list(pre) + ["s_cmp_eq_u32 s1, 0", ""]
        br = len(b) - 1
        b += block
        b[br] = fwd(br, len(b))
        b += ["v_add_f32_e32 v13, v40, v44", wm]
        b.append(back(len(b), 1))
        return gw(b)
    w, n = blk(["s_barrier", "s_waitcnt vmcnt(0)", "buffer_gl0_inv"])
    ok(not n, f"guardwait: llama.cpp soft_max's shape (s_barrier, vmcnt(0), buffer_gl0_inv): the invalidate after the wait is an "
       f"acquire the wait may serve: no note ({w})")
    w, n = blk(["s_barrier", "s_waitcnt vmcnt(0)", "v_mov_b32_e32 v61, 0"])
    ok(n and w and w[0][2] == 3, f"guardwait: a barrier before the wait only orders nothing the wait serves: note ({w})")
    w, n = blk(["s_waitcnt vmcnt(0)", "buffer_gl1_inv"])
    ok(not n, f"guardwait: a cache invalidate after the wait (N2): no note ({w})")
    w, n = blk(["s_waitcnt vmcnt(2)", "scratch_store_b32 off, v20, s33", "ds_store_b128 v7, v[20:23]"])
    ok(n and w and w[0][2] == 1, f"guardwait: E521's residue with an unrelated spill store in the block still fires (N1: a spill "
       f"orders nothing) ({w})")
    w, n = blk(["global_store_b32 v7, v20, s[38:39]", "s_waitcnt vmcnt(2)", "ds_store_b128 v7, v[20:23]"])
    ok(n and w and w[0][2] == 1, f"guardwait: E521's residue with a global store before the wait still fires (N3: a wait cannot "
       f"order an earlier store) ({w})")
    w, n = blk(["s_waitcnt vmcnt(0)", "v_add_f32_e32 v61, v5, v5"], pre=["global_atomic_add_u32 v5, v1, v2, s[8:9] glc"])
    ok(not n, f"guardwait: the wait is for a returning atomic issued after V (a vmcnt entry, loop_census.VMCNT_ENTRY, review F-G3): "
       f"the block needs the newest entry: no note ({w})")
    aj = ["s_mov_b32 s0, 0"] + [f"global_load_b128 v[{40 + 4 * k}:{43 + 4 * k}], v2, s[8:9] offset:{16 * k}" for k in range(3)]
    aj += ["s_cmp_eq_u32 s1, 0", "", "v_cvt_f32_f16_e32 v11, v10"]
    aj[5] = fwd(5, len(aj))
    aj += ["s_waitcnt vmcnt(0)", "v_add_f32_e32 v13, v40, v44", wm]
    aj.append(back(len(aj), 1))
    w, n = gw(aj)
    ok(not n, f"guardwait: a wait after the join is not judged against the block it follows ({w})")
    ex = guard()
    ex[5] = fwd(5, len(ex))
    w, n = gw(ex + ["s_endpgm"])
    ok(not n, f"guardwait: a branch out of the loop (its exit) is not a guard ({w})")
    nest = ["s_mov_b32 s0, 0"] + [f"global_load_b128 v[{40 + 4 * k}:{43 + 4 * k}], v2, s[8:9] offset:{16 * k}" for k in range(3)]
    it = len(nest)
    nest += ["s_cmp_lt_i32 s16, s22", "", "s_waitcnt vmcnt(0)", "v_add_f32_e32 v11, v10, v10", "s_add_u32 s16, s16, 1"]
    nest.append(back(len(nest), it))
    nest[it + 1] = fwd(it + 1, len(nest))
    nest += ["v_add_f32_e32 v13, v40, v44", wm, "global_load_b32 v10, v3, s[8:9]"]
    nest.append(back(len(nest), 1))
    w, n = gw(nest)
    ok(not n, f"guardwait: an inner loop's exit test seen from the outer loop is not a guard (g_spec_accept) ({w})")
    # an else block: s_cbranch to the else, the then block issues V and jumps over the else with s_branch; the else's wait is not
    # after V on any executed path (review G3: only s_cbranch opens a guard)
    el = ["s_mov_b32 s0, 0", "s_cmp_eq_u32 s1, 0", ""]
    el += [f"global_load_b128 v[{40 + 4 * k}:{43 + 4 * k}], v2, s[8:9] offset:{16 * k}" for k in range(3)]
    sb = len(el); el.append("")
    el[2] = fwd(2, len(el))
    el += ["s_waitcnt vmcnt(0)", "v_cvt_f32_f16_e32 v11, v10"]
    el[sb] = fwd(sb, len(el), "s_branch")
    el += ["v_add_f32_e32 v13, v40, v44", wm, "global_load_b32 v10, v3, s[8:9]"]
    el.append(back(len(el), 1))
    w, n = gw(el)
    ok(not n, f"guardwait: an else block (behind an unconditional s_branch) is not judged against the then block's loads ({w})")
    two = ["s_mov_b32 s0, 0"] + [f"global_load_b128 v[{40 + 4 * k}:{43 + 4 * k}], v2, s[8:9] offset:{16 * k}" for k in range(3)]
    two += ["s_cmp_eq_u32 s1, 0", "", "s_cmp_eq_u32 s2, 0", "", "s_waitcnt vmcnt(0)", "v_cvt_f32_f16_e32 v11, v10"]
    two[7] = fwd(7, len(two))
    two[5] = fwd(5, len(two) + 1)
    two += ["v_add_f32_e32 v12, v11, v11", "v_add_f32_e32 v13, v40, v44", wm, "global_load_b32 v10, v3, s[8:9]"]
    two.append(back(len(two), 1))
    w, n = gw(two)
    ok(n and w == [(8, 7, 3)], f"guardwait: nested guards: one hit per wait, named by the innermost guard (review G5) ({w})")
    # ---- (d) vopd (C5 / E551, corrected by q210's llvm-mc build, E580: shared VSRC1 and two literals are refused)
    mv = lambda k: f"v_mov_b32_e32 v{100 + k}, v{1 + k}"
    r = kl.lint_listing("vopd", *listing(loop_of([wm] + [mv(k) for k in range(40)])))
    ok("vopd" in notes(r) and r.get("vopd_pairs") == 20 and r.get("backedge") and r["backedge"][:2] == (40, 20),
       f"vopd: 40 single copies at the back-edge (alternating parity) form 20 pairs: note ({r.get('vopd_pairs')}, {r.get('backedge')})")
    r = kl.lint_listing("vopd15", *listing(loop_of([wm] + [mv(k) for k in range(30)])))
    ok("vopd" not in notes(r) and r.get("vopd_pairs") == 15, f"vopd: 15 pairs, under VOPD_MIN: no note ({r.get('vopd_pairs')})")
    def pairs_of(body):
        r = kl.lint_listing("v", *listing(loop_of([wm] + body)))
        return r.get("vopd_pairs"), r.get("vopd_eligible")
    same = [f"v_mov_b32_e32 v{100 + 2 * k}, v{1 + 2 * k}" for k in range(40)]
    ok(pairs_of(same)[0] == 0, f"vopd: destinations and sources of one parity: no pair ({pairs_of(same)})")
    sd = [f"v_mov_b32_e32 v{100 + 2 * k}, v{1 + k}" for k in range(40)]
    ok(pairs_of(sd)[0] == 0, f"vopd: destinations of one parity (R5): no pair ({pairs_of(sd)})")
    sp = [f"v_mov_b32_e32 v{100 + k}, v{1 + 2 * k}" for k in range(40)]
    ok(pairs_of(sp)[0] == 0, f"vopd: SRC0 of one parity (gfx11.5 interlock, R5a): no pair ({pairs_of(sp)})")
    dual = [f"v_dual_mov_b32 v{100 + 2 * k}, v{1 + 2 * k} :: v_dual_mov_b32 v{101 + 2 * k}, v{2 + 2 * k}" for k in range(20)]
    ok(pairs_of(dual) == (0, 0), f"vopd: moves already paired are not eligible ({pairs_of(dual)})")
    dep = [mv(0)] + [f"v_mov_b32_e32 v{100 + k}, v{99 + k}" for k in range(1, 40)]
    ok(pairs_of(dep)[0] == 0, f"vopd: a dependent chain: no pair ({pairs_of(dep)})")
    mid = sum(([mv(2 * k), f"v_mov_b32_e32 v{1 + 2 * k + 1}, v90", mv(2 * k + 1), "s_waitcnt vmcnt(0)"] for k in range(20)), [])
    ok(pairs_of(mid)[0] == 0, f"vopd: an instruction between writes the second half's source: no pair ({pairs_of(mid)})")
    yy = [f"v_add_nc_u32_e32 v{100 + k}, v{1 + k}, v{51 + k}" if k % 2 == 0 else f"v_lshlrev_b32_e32 v{100 + k}, v{1 + k}, v{51 + k}"
          for k in range(40)]
    ok(pairs_of(yy)[0] == 0 and pairs_of(yy)[1] == 40, f"vopd: two Y-only opcodes cannot pair (R5b: one half must be X) ({pairs_of(yy)})")
    wt = sum(([mv(2 * k), "s_waitcnt vmcnt(0)", mv(2 * k + 1), "s_barrier"] for k in range(20)), [])
    ok(pairs_of(wt)[0] == 0, f"vopd: a wait between the halves: no pair ({pairs_of(wt)})")
    far = sum(([mv(2 * k)] + ["s_nop 0"] * 8 + [mv(2 * k + 1)] + ["s_nop 0"] * 8 for k in range(20)), [])
    ok(pairs_of(far)[0] == 0, f"vopd: halves 9 instructions apart, past the window: no pair ({pairs_of(far)})")
    e64 = [f"v_add_f32_e64 v{100 + k}, v{1 + k}, v{51 + k}" for k in range(40)]
    ok(pairs_of(e64)[1] == 0, f"vopd: VOP3 forms are not eligible (C5's port) ({pairs_of(e64)})")
    neg = [f"v_add_f32 v{100 + k}, -v{1 + k}, v{51 + k}" for k in range(40)]
    ok(pairs_of(neg)[1] == 0, f"vopd: a source negation is not eligible ({pairs_of(neg)})")
    vs = [f"v_add_f32_e32 v{100 + k}, v{1 + k}, v238" for k in range(40)]
    ok(pairs_of(vs)[0] == 0, f"vopd: one VSRC1 register in both halves: refused, swaps or not (Loom vopd_component.c: same register "
       f"is same parity) ({pairs_of(vs)})")
    vb = [f"v_sub_f32_e32 v{100 + k}, v{1 + k}, v{60 + 2 * k}" for k in range(40)]
    ok(pairs_of(vb)[0] == 0, f"vopd: VSRC1 of one parity, different banks (llvm-mc accepts; Loom's interlock rule refuses) "
       f"({pairs_of(vb)})")
    cases = {"shared SRC0 (fmac, llama.cpp mul_mat_q)": (["v_fmac_f32_e32 v112, v58, v21", "v_fmac_f32_e32 v113, v58, v22"], 0),
             "SRC0 same parity, a commutable swap makes it legal": (["v_add_f32_e32 v10, v1, v2", "v_add_f32_e32 v11, v3, v4"], 1),
             "the same with sub (not commutable)": (["v_sub_f32_e32 v10, v1, v2", "v_sub_f32_e32 v11, v3, v4"], 0),
             "VSRC1 same parity, mul swaps cannot fix it": (["v_mul_f32_e32 v10, v1, v4", "v_mul_f32_e32 v11, v2, v6"], 0),
             "fmamk's VSRC1 is its 4th operand, in the SRC2 port (v6 / v4 even)": (["v_fmamk_f32 v1, v2, 0x3f800000, v6",
                                                                                   "v_fmac_f32_e32 v4, v3, v10"], 0),
             "fmamk's VSRC1 v6 against fmac's destination v4 in SRC2, VSRC1 ports apart (N11b)": (["v_fmamk_f32 v1, v2, 0x3f800000, v6",
                                                                                               "v_fmac_f32_e32 v4, v3, v11"], 0),
             "fmamk with VSRC1 v7 against fmac's v4: legal (N11)": (["v_fmamk_f32 v1, v2, 0x3f800000, v7", "v_fmac_f32_e32 v4, v3, v10"], 1),
             "a literal SRC0 is not a VGPR binary (Loom reads VGPR sources only)": (["v_mul_f32_e32 v1, 0x3db8aa3b, v2",
                                                                                    "v_mul_f32_e32 v4, v5, v7"], 0),
             "an SGPR SRC0 likewise": (["v_add_f32_e32 v1, s2, v3", "v_add_f32_e32 v4, v5, v7"], 0),
             "a move from an SGPR is not eligible": (["v_mov_b32_e32 v1, s2", "v_mov_b32_e32 v4, v5"], 0),
             "a move of a non-inline literal is not eligible": (["v_mov_b32_e32 v1, 0x400", "v_mov_b32_e32 v4, v5"], 0),
             "a move of inline 64 pairs (N12)": (["v_mov_b32_e32 v1, 64", "v_mov_b32_e32 v4, v5"], 1),
             "a move of inline 1.0 stays single: Loom pairs integer inline constants only (N13, verify r3 F1)": (["v_mov_b32_e32 v1, 1.0", "v_mov_b32_e32 v4, v5"], 0),
             "a move of inline 64 pairs (verify r3 F1)": (["v_mov_b32_e32 v1, 64", "v_mov_b32_e32 v4, v5"], 1),
             "a move of 65 (printed 0x41) stays single (verify r3 F1)": (["v_mov_b32_e32 v1, 0x41", "v_mov_b32_e32 v4, v5"], 0),
             "a move of inline -16 pairs (verify r3 F1)": (["v_mov_b32_e32 v1, -16", "v_mov_b32_e32 v4, v5"], 1),
             "a move of inline -16 pairs": (["v_mov_b32_e32 v1, -16", "v_mov_b32_e32 v4, v5"], 1)}
    for name, (b2, want) in cases.items():
        got = pairs_of(b2)[0]
        ok(got == want, f"vopd: {name}: {want} pair(s) ({got})")
    lit = [f"v_fmamk_f32 v{100 + k}, v{1 + k}, 0x{0x3fb8aa3b + k:x}, v{51 + k}" for k in range(40)]
    ok(pairs_of(lit)[0] == 0 and pairs_of(lit)[1] == 40, f"vopd: two different literals in one pair: refused (R5) ({pairs_of(lit)})")
    lit1 = [f"v_fmamk_f32 v{100 + k}, v{1 + k}, 0x3fb8aa3b, v{51 + k}" for k in range(40)]
    ok(pairs_of(lit1)[0] == 20, f"vopd: the same literal in both halves is one literal: pairs ({pairs_of(lit1)})")
    r = kl.lint_listing("bfar", *listing(loop_of([wm] + [mv(k) for k in range(40)] + ["s_nop 0"] * 9)))
    ok(r.get("backedge") is None and "backedge" in r, f"vopd: a copy run ending 10 instructions before the back-edge is not its block "
       f"({r.get('backedge')})")
    r = kl.lint_listing("bgap", *listing(loop_of([wm] + [mv(k) for k in range(20)] + ["s_nop 0"] * 4 + [mv(20 + k) for k in range(10)])))
    ok(r.get("backedge") and r["backedge"][:2] == (10, 5), f"vopd: the back-edge block is the last run (moves <= 3 apart), not the one "
       f"before a gap of 5 ({r.get('backedge')})")
    r = kl.lint_listing("nomm", *listing(["s_mov_b32 s0, 0"] + [mv(k) for k in range(40)] + ["s_endpgm"]))
    ok("vopd" not in notes(r) and r.get("vopd_pairs") is None, f"vopd: no MMA loop, no figure ({r.get('vopd_pairs')})")
    chain = ["v_wmma_f32_16x16x16_f16 v[0:7], v[8:15], v[16:23], v[0:7]", "v_wmma_f32_16x16x16_f16 v[24:31], v[8:15], v[16:23], v[24:31]",
             "v_wmma_f32_16x16x16_f16 v[0:7], v[8:15], v[16:23], v[0:7]", "v_wmma_f32_16x16x16_f16 v[24:31], v[8:15], v[16:23], v[24:31]"]
    r = kl.lint_listing("dep1", *listing(loop_of(chain)))
    ok(r.get("wmma_dep1") == 4, f"vopd: two interleaved accumulators: 4 of 4 WMMAs at D->C gap 1 (informational, R6a) ({r.get('wmma_dep1')})")
    chain16 = [f"v_wmma_f32_16x16x16_f16 v[{8 * k}:{8 * k + 7}], v[200:207], v[208:215], v[{8 * k}:{8 * k + 7}]" for k in range(16)]
    r = kl.lint_listing("dep16", *listing(loop_of(chain16)))
    ok(r.get("wmma_dep1") == 0, f"vopd: 16 independent accumulators count 0 ({r.get('wmma_dep1')})")
    # live pins (review F1): pkg-e480's flash2_f16o hot loop (C5's 56 corrected to 54 by q210's llvm-mc build, E580) and the m256 GEMM
    def fx(name):
        lst = [x.split("\t", 1) for x in open(os.path.join(HERE, "fixtures", name)).read().split("\n") if x]
        return [i for _, i in lst], [int(a, 16) for a, _ in lst]
    r = kl.lint_listing("flash2_f16o", *fx("flash2_f16o_e480_loop.listing"))
    got = (r.get("vopd_pairs"), r.get("vopd_eligible"), r.get("backedge") and r["backedge"][:2], r.get("wmma_dep1"))
    ok(got == (35, 129, (65, 31), 16), f"vopd: live flash2_f16o by Loom 0.2.0's planner rules: 35 pairs of 129 eligible, back-edge 65 "
       f"moves -> 31 pairs, 16 WMMAs at D->C gap <= 1 (C5 / E551 counted 56 of 186 under looser rules; review r2) ({got})")
    lb = fx("flash2_f16o_e480_loop.listing")[0]
    two = lambda i, j: kl.lint_listing("v2", *listing(loop_of([wm, lb[i], lb[j]]))).get("vopd_pairs")
    got = [two(i, j) for i, j in ((776, 777), (779, 780), (514, 515), (597, 601), (667, 670), (882, 884))]
    ok(got == [0, 0, 0, 0, 0, 1], f"vopd: live pairs refused: C5's two shared-v238 pairs, the reviewer's three VSRC1-parity pairs (one "
       f"by parity, two by a literal SRC0); a back-edge move pair is formed ({got})")
    r = kl.lint_listing("gemm_m256", *fx("gemm_f16_store_m256_e480_loop.listing"))
    got = (r.get("vopd_pairs"), r.get("vopd_eligible"), r.get("wmma_dep1"))
    ok(got == (0, 1, 0), f"vopd: live m256 GEMM: 0 pairs of 1 eligible, 0 at gap <= 1 ({got})")
    ok("ldswait" not in notes(r) and not hasattr(kl, "lds_waits"), "ldswait is deleted (the ldsser lever, refuted E293 / E309 / E335 / "
       "E340; re-add only if E534's SQ_WAIT_INST_LDS contradicts E341)")


def loom_cands():
    """C19: kernel_lint surfaces Loom upstream CANDIDATES, never verdicts (HANDOFF failure list T1-T5, N1-N7). Recurrences: E533 (a
    report's next tier a whole workgroup cannot seat, found by hand after E527 / E531 built on it), E592 (allocation-emitted back-edge
    copies the VOPD planner never sees, found by a source read), E177 / E227 (a lint shape that moved with only the compiler changed,
    attributed by hand)."""
    F = lambda n: os.path.join(HERE, "fixtures", n)
    rep = lambda n: json.load(open(F(n)))
    def fx(name):
        lst = [x.split("\t", 1) for x in open(F(name)).read().split("\n") if x]
        return [i for _, i in lst], [int(a, 16) for a, _ in lst]
    kinds = lambda cs: [c["kind"] for c in cs]
    def one(cs, kind):
        x = [c for c in cs if c["kind"] == kind]
        return x[0] if len(x) == 1 else None
    # ---- T1 / T1b / T1c: the report's next tier vs whole-workgroup arithmetic (E533)
    cs, skip = kl.loom_candidates("g_attention_flash2_f16o", [], [], rep("flash2_f16o_residency.json"))
    c = one(cs, "tier")
    ok(c and (c["report"], c["whole"]) == ((6, 7), (6, 9)) and "compile-report-queries.md:122" in c["line"]
       and "occupancy.c:377" in c["line"] and "VGPR <= 168" in c["line"] and "LDS <= 43008" in c["line"] and not skip,
       f"tier: pkg-cmb2 flash2_f16o, report 6 -> 7, whole workgroups 6 -> 9 at VGPR <= 168 and LDS <= 43008 (T1) ({c and c['line']}, {skip})")
    c = one(kl.loom_candidates("g_gemm_f16_store_m256", [], [], rep("gemm_f16_store_m256_e480_residency.json"))[0], "tier")
    ok(c and (c["report"], c["whole"]) == ((4, 5), (4, 6)) and "bound" not in c["line"],
       f"tier: live m256 GEMM report 4 -> 5, whole workgroups 4 -> 6 (T1b) ({c and c['line']})")
    c = one(kl.loom_candidates("g_mv_q4_k_k17408_t16_q8_r32", [], [], rep("mv_q4_k_k17408_t16_r32_e480_residency.json"))[0], "tier")
    ok(c and (c["report"], c["whole"]) == ((7, 8), (5, 9)) and "bound: placement undocumented" in c["line"],
       f"tier: a 544-lane matvec (17 waves) report 7 -> 8, whole workgroups 5 -> 9, marked bound (T1c) ({c and c['line']})")
    cs, skip = kl.loom_candidates("g_gemm_f16_swiglu_m256", [], [], rep("gemm_f16_swiglu_m256_e480_residency.json"))
    ok(cs == [] and skip == [], f"tier: the swiglu m256 GEMM's report 4 -> 6 agrees: no candidate (N1) ({kinds(cs)}, {skip})")
    best = rep("gemm_f16_swiglu_m256_e480_residency.json")
    rows = best["residency_constraints"]["rows"]
    for x in rows:
        if x["name"] == "amdgpu.lds":
            x.update(units=0, rounded_units=0, independent_tier=16)
        if x["name"] == "amdgpu.vgpr":
            x.update(units=96, rounded_units=96, independent_tier=16)
        if x["name"] == "amdgpu.workgroup_slots":
            x.update(independent_tier=16)
    best["target_resources"]["residency"].update(current_tier=16, next_better_tier=16)
    cs, skip = kl.loom_candidates("best", [], [], best)
    ok(cs == [] and skip == [], f"tier: at the wave cap, report next = current: no candidate (N2) ({kinds(cs)}, {skip})")
    unm = rep("flash2_f16o_residency.json")
    unm["residency_constraints"]["rows"].append({"name": "amdgpu.barrier_slots", "kind": "fixed_limit", "allocation_scope": None,
                                                 "independent_tier": 16})
    cs, skip = kl.loom_candidates("unm", [], [], unm)
    ok(cs == [] and len(skip) == 1 and "tier unverified" in skip[0],
       f"tier: an unmodelled residency row: no candidate, counted not compared (N3) ({kinds(cs)}, {skip})")
    cs, skip = kl.loom_candidates("norep", [], [], None)
    ok(cs == [] and skip == ["no compile report"], f"no report: nothing, said (N4) ({cs}, {skip})")
    # ---- T4 / N5: VOPD for allocation-emitted back-edge copies (E592)
    f2 = fx("flash2_f16o_e480_loop.listing")
    cs, _ = kl.loom_candidates("g_attention_flash2_f16o", *f2, rep("flash2_f16o_residency.json"))
    c = one(cs, "vopd-edge")
    ok(c and c["groups"] == [(65, 31)] and "packet_plan.c:72-77" in c["line"] and "encoding.c:1016" in c["line"]
       and "not itemised" in c["line"] and kinds(cs) == ["tier", "vopd-edge"],
       f"vopd-edge: live flash2_f16o back-edge group 65 moves -> 31 pairs (T4) ({c and c['line']}, {kinds(cs)})")
    withc = rep("flash2_f16o_residency.json")
    withc["move_causes"] = {"kind_count": 1, "packet_count": 20, "unit_count": 407,
                            "causes": [{"cause": "branch_edge", "packet_count": 20, "unit_count": 407}]}
    c = one(kl.loom_candidates("g_attention_flash2_f16o", *f2, withc)[0], "vopd-edge")
    ok(c and "branch_edge 407 moves in 20 packets" in c["line"], f"vopd-edge: a details report's branch_edge cause is cited ({c and c['line']})")
    def listing(body):
        return body, [4 * i for i in range(len(body))]
    def loop_of(body, tail=()):
        b = ["s_mov_b32 s0, 0", "v_wmma_f32_16x16x16_f16 v[200:207], v[192:199], v[184:191], v[200:207]"] + body + list(tail)
        return b + [f"s_branch {(1 - (len(b) + 1)) & 0xFFFF}"]
    mv = lambda k: f"v_mov_b32_e32 v{100 + k}, v{1 + k}"
    edge = lambda body, tail=(): one(kl.loom_candidates("x", *listing(loop_of(body, tail)), None)[0], "vopd-edge")
    c = edge([mv(k) for k in range(16)])
    ok(c and c["groups"] == [(16, 8)], f"vopd-edge: 16 independent copies at the back-edge, 8 pairs = EDGE_PAIRS_MIN: candidate ({c})")
    two = loop_of([mv(k) for k in range(16)], ["s_nop 0"])
    two.append(f"s_branch {(1 - (len(two) + 1)) & 0xFFFF}")          # a second backward branch 2 after the first (pkg-e480 g_post_add_norm)
    c = one(kl.loom_candidates("x", *listing(two), None)[0], "vopd-edge")
    ok(c and c["groups"] == [(16, 8)], f"vopd-edge: one copy group within 8 of two backward branches is one group, not two (fire "
       f"11-25: g_post_add_norm's loops end at 756 and 758 after one 61-move group) ({c})")
    c = edge([mv(k) for k in range(14)])
    ok(c is None, f"vopd-edge: 14 copies, 7 pairs, under EDGE_PAIRS_MIN: none (N5a) ({c})")
    c = edge([mv(k) for k in range(16)], ["s_nop 0"] * 9)
    ok(c is None, f"vopd-edge: a copy run ending 10 instructions before the branch is not the edge group (N5b) ({c})")
    chain = [mv(0)] + [f"v_mov_b32_e32 v{100 + k}, v{99 + k}" for k in range(1, 20)]
    c = edge(chain)
    ok(c is None, f"vopd-edge: a sequentialized copy chain (each move reads the previous one's destination): no pair (N5c) ({c})")
    rot = ["v_mov_b32_e32 v99, v100"] + [f"v_mov_b32_e32 v{100 + k}, v{101 + k}" for k in range(20)]
    c = edge(rot)
    ok(c is None, f"vopd-edge: a rotation sequentialized through a temporary (each move overwrites the previous one's source, so the "
       f"first half reads the other's destination; parities would pair): no pair, order constraint (N5c) ({c})")
    dual = [f"v_dual_mov_b32 v{100 + 2 * k}, v{1 + 2 * k} :: v_dual_mov_b32 v{101 + 2 * k}, v{2 + 2 * k}" for k in range(10)]
    c = edge(dual)
    ok(c is None, f"vopd-edge: copies already paired: none (N5d) ({c})")
    c = one(kl.loom_candidates("flat", *listing(["s_mov_b32 s0, 0"] + [mv(k) for k in range(20)] + ["s_endpgm"]), None)[0], "vopd-edge")
    ok(c is None, f"vopd-edge: no loop, no back-edge group (N5e) ({c})")
    # ---- T5 / N6: the SDK differential (E177 / E227 by hand): the m256 GEMM's loop by 0.2.0 (live) and by 0.3.0
    g2 = kl.lint_listing("g_gemm_f16_store_m256", *fx("gemm_f16_store_m256_e480_loop.listing"))
    g3 = kl.lint_listing("g_gemm_f16_store_m256", *fx("gemm_f16_store_m256_v030_loop.listing"))
    prov = lambda v, path, sha="a" * 64: {"compiler": path, "version": f"hrx-loom {v}", "programs": {"llm.loom": sha},
                                          "roots": None, "target": "amdgpu:gfx1151"}
    pa, pb = prov("0.2.0", "/v020/bin/loom-compile"), prov("0.3.0", "/v030/bin/loom-compile")
    lines = kl.sdk_diff({"g_gemm_f16_store_m256": g3}, {"g_gemm_f16_store_m256": g2}, pb, pa)
    hit = [x for x in lines if x.startswith("loomcand sdk g_gemm_f16_store_m256:")]
    ok(len(hit) == 1 and "+addr64" in hit[0] and "hrx-loom 0.2.0 -> hrx-loom 0.3.0" in hit[0] and "loop209->249" in hit[0],
       f"sdk: addr64 appears in the m256 GEMM with only the compiler changed (T5) ({lines})")
    same = kl.sdk_diff({"k": g2}, {"k": dict(g2)}, pb, pa)
    ok(not [x for x in same if x.startswith("loomcand sdk k:")], f"sdk: equal shapes: no candidate (N6) ({same})")
    for what, a_, b_ in (("programs differ", prov("0.3.0", "/v030/bin/loom-compile", "b" * 64), pa),
                         ("provenance missing", pb, None), ("one compiler", pa, dict(pa)), ("roots differ", dict(pb, roots="^g_gemm"), pa),
                         ("target differs", dict(pb, target="amdgpu:gfx1201"), pa)):
        got = kl.sdk_diff({"g_gemm_f16_store_m256": g3}, {"g_gemm_f16_store_m256": g2}, a_, b_)
        ok(len(got) == 1 and got[0].startswith("loomcand sdk: not attributable"), f"sdk: {what}: not attributable, no per-kernel line (N6) ({got})")
    got = kl.sdk_diff({"a": g3, "new": g3}, {"a": g3, "old": g2}, pb, pa)
    ok(sorted(x.split(":")[0] for x in got if x.startswith("loomcand sdk ")) == ["loomcand sdk new", "loomcand sdk old"],
       f"sdk: a kernel compiled by one side only is a line ({got})")
    tagged = dict(g3, flags=g3["flags"] + ["regress:loop209->249"], notes=g3.get("notes", []) + ["loom:residency_cliff", "outlier"])
    got = kl.sdk_diff({"g": tagged}, {"g": g3}, pb, pa)
    ok(not [x for x in got if x.startswith("loomcand sdk g:")], f"sdk: --baseline's regress flags, Loom's suggest notes and peer notes "
       f"(outlier) are not shapes of the code ({got})")
    w6, w8 = dict(g3, waves=6), dict(g3, waves=8)
    got = kl.sdk_diff({"g": w8}, {"g": w6}, pb, pa)
    ok(any("better waves6->8" in x for x in got) and not any(" worse " in x for x in got),
       f"sdk: a kernel that gained waves reads 'better waves6->8' in the base -> candidate direction ({got})")
    d = tempfile.mkdtemp(prefix="kl_prov.")
    open(os.path.join(d, "loom-provenance"), "w").write("{}")
    try:
        kl.compile_programs([os.path.join(d, "missing.loom")], d, 1, lc="/bin/false")
    except OSError:
        pass
    ok(not os.path.exists(os.path.join(d, "loom-provenance")), "provenance: a compile that dies half-way leaves none (no differential "
       "on mixed objects)")
    # ---- N7: candidates never reach flags or notes (cand_lint calls lint_listing / lint)
    ok("loomcand" not in g2 and "loomcand" not in g3, "isolation: lint_listing rows carry no candidates (N7)")


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
        lr = kl.lint_file(os.path.join(t, "two.hsaco"), rga="")
        ok(len(lr) == 2 and all(kl.live_text(x) == "unknown" and "multi-kernel" in x["live"]["error"] for x in lr),
           f"livereg: a multi-kernel object is 'unknown (multi-kernel)' per row, never blank ({[x.get('live') for x in lr]})")
        ok(len(rows) == 2 and all(x["select"] == r["select"] and x["loop"] == r["loop"] and "select" in x["flags"] for x in rows),
           f"a two-kernel object gives two rows equal to the kernel alone ({[(x['kernel'], x['select'], x['loop']) for x in rows]} vs {r['select']}, {r['loop']})")
        r = compile_lint(lc, t, "iq4_new", qv + new, "g_deq16_iq4_nl")
        ok(not r["flags"], f"IQ4 dequant with the byte mask: no flag ({r['select']} of {r['common']}, flags {r['flags']})")
        # --suggest (common/loom_suggest.py): lintloom-1125's review found every suggest failure killed the lint run (a missing tool
        # FileNotFoundError, exit 0 with non-JSON JSONDecodeError, a JSON list AttributeError). Each must print 'unknown (...)', note
        # loom:unknown and leave the table complete; a working tool's findings become loom:<id> notes, never flags.
        import shutil
        import stat
        sg = os.path.join(t, "sg"); os.makedirs(sg)
        shutil.copy(os.path.join(t, "iq4_new.hsaco"), os.path.join(sg, "g_deq16_iq4_nl.hsaco"))
        shutil.copy(os.path.join(HERE, "fixtures", "loom_suggest", "g_gemm_f16_acc_m256.report.json"), os.path.join(sg, "g_deq16_iq4_nl.json"))
        tools = {"missing": os.path.join(t, "no-such-tool"), "garbage": "echo hello", "list": "echo '[1]'", "crash": "echo boom >&2; exit 2", "nonutf8": r'printf "\377\376 not utf8"',
                 "good": "cat " + os.path.join(HERE, "fixtures", "loom_suggest", "g_gemm_f16_acc_m256.suggest.json")}
        klp = os.path.join(HERE, "..", "common", "kernel_lint.py")
        for name, body in tools.items():
            tp = body
            if name != "missing":
                tp = os.path.join(t, "tool-" + name)
                open(tp, "w").write("#!/bin/sh\n" + body + "\n")
                os.chmod(tp, os.stat(tp).st_mode | stat.S_IXUSR)
            p = subprocess.run([sys.executable, klp, sg, "--suggest", "--suggest-tool", tp], capture_output=True, text=True, timeout=300)
            row = [x for x in p.stdout.splitlines() if x.startswith("g_deq16_iq4_nl ")]
            if name == "good":
                ok(p.returncode == 0 and row and "loom:lds_bank_service" in row[0] and "loom g_deq16_iq4_nl: residency cliff: Loom next tier 5: phantom (E533)" in p.stdout
                   and "kernel_lint: 1 kernels, 0 flagged" in p.stdout and ": 0 of 1 runs unknown" in p.stdout,
                   f"--suggest: findings are notes, the cliff judged in whole workgroups, no flag (rc {p.returncode}, {row}, {p.stderr[-200:]})")
            else:
                ok(p.returncode == 0 and row and "loom:unknown" in row[0] and "loom g_deq16_iq4_nl: unknown (" in p.stdout
                   and "kernel_lint: 1 kernels" in p.stdout and ": every run failed (unknown (" in p.stdout,
                   f"--suggest with a {name} tool: 'unknown', table complete (rc {p.returncode}, {row}, {p.stderr[-300:]})")
        # C19 N7: --loom-candidates adds lines only: every table row and the --strict exit code are the same without it
        base_run = subprocess.run([sys.executable, klp, sg, "--strict"], capture_output=True, text=True, timeout=300)
        cand_run = subprocess.run([sys.executable, klp, sg, "--strict", "--loom-candidates"], capture_output=True, text=True, timeout=300)
        tab = lambda o: [x for x in o.splitlines() if x.startswith(("kernel ", "g_deq16_iq4_nl ", "kernel_lint:"))]
        ok(base_run.returncode == cand_run.returncode and tab(base_run.stdout) == tab(cand_run.stdout) and tab(base_run.stdout)
           and "loomcand tier g_deq16_iq4_nl: report tier 4 -> 5" in cand_run.stdout and "loomcand" not in base_run.stdout,
           f"isolation: --loom-candidates changes no row and no exit code, and prints the tier line (N7) (rc {base_run.returncode} / "
           f"{cand_run.returncode}; {' | '.join(x for x in cand_run.stdout.splitlines() if 'loomcand' in x)[:300]} {cand_run.stderr[-300:]})")
        # C19 provenance: a compile records the compiler, its version and each program's sha256; two compiles by one compiler are
        # not attributable to Loom (N6 through main)
        import hashlib
        pv1, pv2 = os.path.join(t, "pv1"), os.path.join(t, "pv2")
        prog_iq4 = os.path.join(t, "iq4_new.loom")
        for d_ in (pv1, pv2):
            subprocess.run([sys.executable, klp, "--program", prog_iq4, "--out", d_, "--loom-compile", lc], capture_output=True, text=True,
                           timeout=600)
        pr = json.load(open(os.path.join(pv1, "loom-provenance"))) if os.path.exists(os.path.join(pv1, "loom-provenance")) else {}
        ok(pr.get("programs") == {"iq4_new.loom": hashlib.sha256(open(prog_iq4, "rb").read()).hexdigest()} and
           pr.get("compiler") == os.path.realpath(lc) and pr.get("version", "").startswith("hrx-loom "),
           f"provenance: compiler, version and program sha256 recorded beside the objects ({pr})")
        dr = subprocess.run([sys.executable, klp, pv2, "--baseline", pv1, "--loom-candidates"], capture_output=True, text=True, timeout=300)
        ok(dr.returncode == 0 and "loomcand sdk: not attributable (one compiler" in dr.stdout and "loomcand sdk g_" not in dr.stdout,
           f"sdk through main: the same compiler twice is not attributable ({[x for x in dr.stdout.splitlines() if 'loomcand' in x]} {dr.stderr[-300:]})")
        r = compile_lint(lc, t, "q6_old", q6_old + kernel("g_q6_k_icodes", False), "t_g_q6_k_icodes")
        ok("repack" in r["flags"], f"pre-E233 q6_k codes flag repack (pk {r['pk']}, bfe {r['bfe']})")
        r = compile_lint(lc, t, "q6_new", q6_new + kernel("g_q6_k_icodes", False), "t_g_q6_k_icodes")
        ok(not r["flags"], f"q6_k codes on words: no flag (pk {r['pk']}, bfe {r['bfe']}, flags {r['flags']})")
        rp = subprocess.run([lc, os.path.join(t, "q6_old.loom"), "--target=amdgpu:gfx1151", "--root=t_g_q6_k_icodes",
                             f"--output={t}/q6r.hsaco", "--compile-report=summary", f"--compile-report-output={t}/q6r.json"],
                            capture_output=True, text=True)
        r = kl.lint(f"{t}/q6r.hsaco") if not rp.returncode else {}
        ok(isinstance(r.get("waves"), int) and r.get("occ") and "why" not in r["occ"], f"occupancy: a real compile's summary report is "
           f"verified through common/devices/<target_key>.json ({r.get('occ')})")
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
    occupancy_seam()
    livereg()
    lesson_notes()
    loom_cands()
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
    import tempfile as tf
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
