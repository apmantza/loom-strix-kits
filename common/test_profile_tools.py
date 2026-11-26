#!/usr/bin/env python3
"""Tests for the skill's scripts, through their command lines on synthetic runs. usage: python3 -I test_scripts.py"""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.realpath(__file__))   # symlinked copies (HRX-like skill) resolve to common/
MODEL = os.path.join(HERE, "..", "kits", "qwen38-27b", "lanes.json")


def ev(key, sub, ms, wg=(8, 4, 64)):
    return {"type": "dispatch_event", "key": key, "submission_id": sub, "duration_ns": int(ms * 1e6),
            "workgroup_count": list(wg), "valid": True}


def run_dir(root, name, events, prompt=1000, req=None):
    d = os.path.join(root, name)
    os.makedirs(d)
    with open(os.path.join(d, "events.jsonl"), "w") as f:
        f.write(json.dumps({"type": "summary"}) + "\n")
        for e in events:
            f.write(json.dumps(e) + "\n")
    body = req if req is not None else json.dumps({"requests": [{"prompt_tokens": prompt}]})
    open(os.path.join(d, "run.req.json"), "w").write(body + "\n")
    return d


def summary(*args):
    return subprocess.run([sys.executable, "-I", os.path.join(HERE, "events_summary.py"), MODEL, *args],
                          capture_output=True, text=True)


class EventsSummary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_lanes_steps_and_clusters(self):
        # two verify steps (submissions 1, 2) of 16 vsplit dispatches: step 1 aligned (1.6 ms), step 2 straddling (3.0 ms)
        es = [ev("g_attention_vsplit", 1, 1.6) for _ in range(16)] + [ev("g_attention_vsplit", 2, 3.0) for _ in range(16)]
        es += [ev("g_attention_flash2", 7, 50.0), ev("g_gemm_f16_swiglu_m256", 7, 1000.0, (272, 16, 1)), ev("g_row_scale_f16_live_lv", 7, 2.0)]
        d = run_dir(self.root, "a", es)
        r = summary(d, "--split", "g_attention_vsplit")
        self.assertEqual(r.returncode, 0, r.stderr)
        line = next(x for x in r.stdout.splitlines() if x.startswith("| decode: verify attention"))
        self.assertIn("36.80 ms/step", line)          # (16 x 1.6 + 16 x 3.0) / 2 steps
        self.assertIn("32 disp", line)
        gemm = next(x for x in r.stdout.splitlines() if x.startswith("| prefill: GEMMs"))
        self.assertIn("1.00 s", gemm)
        self.assertIn("2 verify steps", r.stdout)
        self.assertIn("below 16 x 1.600 ms, above 16 x 3.000 ms", r.stdout)   # E143's bimodal split, recovered
        other = next(x for x in r.stdout.splitlines() if x.startswith("| prefill: other"))
        self.assertIn("1 disp", other)                 # an unlisted kernel lands in the catch-all lane

    def test_prefill_epochs_excluded_from_decode_lanes(self):
        # the drafter and matvecs also run in prefill epochs (a submission with GEMMs/flash2): per verify step they must
        # count only decode submissions (first real run: drafter 31.8 ms/step vs ~8.7 measured per step, E50)
        es = [ev("g_attention_vsplit", 1, 1.0) for _ in range(16)] + [ev("g_dfl_attention", 1, 2.0)]
        es += [ev("g_attention_flash2", 9, 50.0), ev("g_gemm_f16_store_m256", 9, 10.0), ev("g_dfl_attention", 9, 500.0), ev("g_mv_q5_k_k5120_t8_q8", 9, 300.0)]
        r = summary(run_dir(self.root, "p", es))
        self.assertEqual(r.returncode, 0, r.stderr)
        dr = next(x for x in r.stdout.splitlines() if x.startswith("| decode: drafter"))
        self.assertIn("2.00 ms/step", dr)
        mv = next(x for x in r.stdout.splitlines() if x.startswith("| decode: target weight matvecs"))
        self.assertIn("0.00 ms/step", mv)

    def test_heads_split_target_vs_drafter(self):
        # two head matvecs per verify step: g_head_mv_t8_q8 is @llm_verify on the target head, its _p40 sibling is
        # @llm_dflash_propose on the drafter's q4_0 proposal-head copy (11-21; counted as target weights before)
        es = [ev("g_attention_vsplit", 1, 1.0), ev("g_head_mv_t8_q8", 1, 5.0), ev("g_head_mv_t8_q8_p40", 1, 3.0)]
        r = summary(run_dir(self.root, "h", es))
        self.assertEqual(r.returncode, 0, r.stderr)
        mv = next(x for x in r.stdout.splitlines() if x.startswith("| decode: target weight matvecs"))
        dr = next(x for x in r.stdout.splitlines() if x.startswith("| decode: drafter"))
        self.assertIn("5.00 ms/step", mv)
        self.assertIn("3.00 ms/step", dr)

    def test_same_kernel_split_by_epoch_phase(self):
        # the GDN recurrence runs in both phases: in a decode epoch it is per-step work, in a prefill epoch (a submission with
        # a GEMM) prefill work; each phase has its own lane (fixed-cost lanes, 11-21)
        es = [ev("g_attention_vsplit", 1, 1.0), ev("g_gdn_recurrence_rec", 1, 4.0)]
        es += [ev("g_attention_flash2", 9, 50.0), ev("g_gemm_f16_store_m256", 9, 10.0), ev("g_gdn_recurrence_pf", 9, 2000.0)]
        r = summary(run_dir(self.root, "g", es))
        self.assertEqual(r.returncode, 0, r.stderr)
        dg = next(x for x in r.stdout.splitlines() if x.startswith("| decode: GDN"))
        pg = next(x for x in r.stdout.splitlines() if x.startswith("| prefill: GDN"))
        self.assertIn("4.00 ms/step", dg)
        self.assertIn("2.00 s", pg)

    def test_warm_gemm_submission_is_decode(self):
        # the per-step DFlash warm pass runs GEMMs in its own submission (no flash2): it is decode work (11-21: a ^g_gemm_
        # prefill marker filed 97 such submissions as prefill and hid them from the decode step)
        # per-step warm (8 tokens): decode; a prefill chunk's warm (4096 tokens, no flash2 in its submission): prefill
        es = [ev("g_attention_vsplit", 1, 1.0), ev("g_token_info", 2, 0.01, (8, 1, 1)), ev("g_dfl_kv_store", 2, 0.5), ev("g_gemm_q4_k_store", 2, 3.0)]
        es += [ev("g_attention_flash2", 9, 100.0), ev("g_gemm_f16_store_m256", 9, 10.0)]
        es += [ev("g_token_info", 10, 0.01, (4096, 1, 1)), ev("g_dfl_kv_store", 10, 7.0), ev("g_gemm_q4_k_store_m256", 10, 30.0)]
        r = summary(run_dir(self.root, "w", es))
        self.assertEqual(r.returncode, 0, r.stderr)
        dr = next(x for x in r.stdout.splitlines() if x.startswith("| decode: drafter"))
        self.assertIn("3.51 ms/step", dr)    # the per-step warm submission, all of it: kv_store + its GEMM + its g_token_info (E163)
        pg = next(x for x in r.stdout.splitlines() if x.startswith("| prefill: GEMMs"))
        self.assertIn("0.01 s", pg)          # the flash2 epoch's GEMM only: the chunk warm's 30 ms GEMM is the drafter's (E163)
        pw = next(x for x in r.stdout.splitlines() if x.startswith("| prefill: drafter context"))
        self.assertIn("0.04 s", pw)          # the chunk warm submission: kv_store 7 + GEMM 30 + token_info

    def test_shared_kernel_names_split_by_submission(self):
        # E163 recurrence: by kernel name, the drafter's propose pass (its own submission, opening with g_dfl_prepare) ran
        # g_mv_q4_k_* that went to the target matvecs, and the target's Q8_0 matvecs went to the drafter: 3.6 ms/step swapped,
        # the drafter read 109 % of its floor
        es = [ev("g_attention_vsplit", 1, 1.0), ev("g_mv_q4_k_k5120_t8_q8", 1, 20.0), ev("g_mv_q8_0_k5120_t4_r1", 1, 2.0)]
        es += [ev("g_dfl_prepare", 2, 0.5), ev("g_mv_q4_k_k5120_t8_q8", 2, 4.0), ev("g_head_mv_t8_q8_p40", 2, 3.0)]
        r = summary(run_dir(self.root, "s", es))
        self.assertEqual(r.returncode, 0, r.stderr)
        mv = next(x for x in r.stdout.splitlines() if x.startswith("| decode: target weight matvecs"))
        self.assertIn("22.00 ms/step", mv)   # the target submission's q4_k + q8_0 matvecs
        dr = next(x for x in r.stdout.splitlines() if x.startswith("| decode: drafter"))
        self.assertIn("7.50 ms/step", dr)    # the propose submission: prepare + its q4_k matvec + the proposal head

    def test_runs_side_by_side(self):
        a = run_dir(self.root, "1-base", [ev("g_attention_vsplit", 1, 3.0) for _ in range(16)])
        b = run_dir(self.root, "2-cand", [ev("g_attention_vsplit", 1, 1.5) for _ in range(16)])
        r = summary(a, b)
        self.assertEqual(r.returncode, 0, r.stderr)
        line = next(x for x in r.stdout.splitlines() if x.startswith("| decode: verify attention"))
        self.assertIn("48.00 ms/step", line)
        self.assertIn("24.00 ms/step", line)

    def test_cycles_beside_seconds(self):   # E482: the GEMMs keep their cycles across starts while the clock follows temperature
        def tel(d, ghz):
            open(os.path.join(d, "tel.txt"), "w").write(f"  telemetry x: 10 samples; while busy: power 85.00 (p10 84, p90 85) W, "
                                                       f"clock {ghz:.2f} (p10 2.1, p90 2.3) GHz, temp 80.00 C\n")
        a = run_dir(self.root, "1-cool", [ev("g_attention_flash2", 7, 50.0), ev("g_gemm_f16_swiglu_m256", 7, 1000.0, (272, 16, 1))])
        b = run_dir(self.root, "2-hot", [ev("g_attention_flash2", 7, 50.0), ev("g_gemm_f16_swiglu_m256", 7, 1100.0, (272, 16, 1))])
        tel(a, 2.20); tel(b, 2.00)
        r = summary(a, b)
        self.assertEqual(r.returncode, 0, r.stderr)
        line = next(x for x in r.stdout.splitlines() if x.startswith("| prefill: GEMMs") and "G |" in x)
        self.assertIn("2.20 G | 2.20 G", line)        # 1.0 s x 2.2 GHz, 1.1 s x 2.0 GHz
        self.assertIn("| 9.5 % | 0.0 % |", line)       # seconds moved, cycles did not
        os.remove(os.path.join(b, "tel.txt"))
        r = summary(a, b)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("cycles: no busy clock (tel.tsv / tel.txt) for 2-hot", r.stdout)

    def test_refuses_unreadable_prompt(self):
        # the served size decides every floor: a request record without prompt_tokens is a refusal, not a default
        d = run_dir(self.root, "x", [ev("g_attention_vsplit", 1, 1.0)], req=json.dumps({"requests": [{}]}))
        r = summary(d)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("prompt_tokens", r.stdout + r.stderr)

    def test_floor_scales_with_context(self):
        # the KV floor is linear in the served context: 2x the prompt -> 2x the floor (labels lie, E128: cite served)
        a = run_dir(self.root, "s", [ev("g_attention_vsplit", 1, 1.0)], prompt=40000)
        b = run_dir(self.root, "l", [ev("g_attention_vsplit", 1, 1.0)], prompt=80000)
        fa = next(x for x in summary(a).stdout.splitlines() if x.startswith("| decode: verify attention")).split("|")[2]
        fb = next(x for x in summary(b).stdout.splitlines() if x.startswith("| decode: verify attention")).split("|")[2]
        self.assertAlmostEqual(2 * float(fa.split()[0]), float(fb.split()[0]), places=1)


def abba_run(root, name, flash2, gemm=1000.0, extra=()):
    """One served run: a verify step (decode lanes measurable) and a prefill epoch with flash2 + GEMM (abba-e325's shape)."""
    es = [ev("g_attention_vsplit", 1, 1.0), ev("g_mv_q5_k_k5120_t8_q8", 1, 2.0), ev("g_dfl_attention", 1, 0.5),
          ev("g_gdn_recurrence_rec", 1, 0.3), ev("g_rope", 1, 0.1)]
    es += [ev("g_attention_flash2", 9, flash2, (256, 4, 1)), ev("g_gemm_f16_store_m256", 9, gemm, (40, 16, 1)),
           ev("g_token_info", 9, 0.01, (4096, 1, 1)), ev("g_deq16_q5_k", 9, 50.0), ev("g_gdn_recurrence_pf", 9, 80.0), ev("g_row_scale_f16_live_lv", 9, 90.0)]
    es += [ev("g_token_info", 10, 0.01, (4096, 1, 1)), ev("g_dfl_kv_store", 10, 10.0)]
    es += [ev(k, 9, ms, wg) for k, ms, wg in extra]
    return run_dir(root, name, es)


class LaneGate(unittest.TestCase):
    """--gate (E351/E356/E374): a served profile may STOP a candidate before the 2-hour agent bench, never pass it to go-live."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def abba(self, ref, cand, cgemm=(1000.0, 1000.0), cextra=()):
        return [abba_run(self.root, "1-ref", ref[0]), abba_run(self.root, "2-cand", cand[0], cgemm[0], cextra),
                abba_run(self.root, "3-cand", cand[1], cgemm[1], cextra), abba_run(self.root, "4-ref", ref[1])]

    def gate(self, runs, *extra):
        return summary(*runs, "--top", "0", "--gate", "prefill: attention", "--ref", "1,4", "--cand", "2,3", *extra)

    def test_real_drop_passes(self):
        # a 7 % flash2 drop with a 1 % reference spread is the one shape that may go on to the agent bench
        r = self.gate(self.abba((1000.0, 1010.0), (935.0, 940.0)), "--guard-others", "2")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("LANE GATE: PASS-TO-BENCH", r.stdout)

    def test_drop_inside_reference_spread_stops(self):
        # E374: a 3 % drop inside a 6 % A/A spread is noise; benching it costs two hours for a coin flip
        r = self.gate(self.abba((1000.0, 1060.0), (995.0, 1000.0)))
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("LANE GATE: STOP (targeted lane did not move beyond noise: skip the bench", r.stdout)

    def test_drop_inside_candidate_spread_stops(self):
        # the candidate pair's own spread counts too: a tight reference does not make a scattered candidate a win
        r = self.gate(self.abba((1000.0, 1001.0), (900.0, 1010.0)))
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)

    def test_no_slower_tie_passes(self):
        # HRX-like E466: q179's NaN fix tied (+0.20 %, noise 0.38 %) and the drop-only gate stopped it; a fix's question is 'not slower'
        runs = self.abba((1000.0, 1004.0), (1003.0, 1005.0))
        r = self.gate(runs, "--no-slower", "0.5", "--guard-others", "2")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("not slower beyond noise", r.stdout)
        self.assertEqual(self.gate(runs).returncode, 1)   # the speed gate still stops it

    def test_no_slower_rise_stops(self):
        # a fix that costs 5 % on its lane (noise 0.4 %) stops before the bench
        r = self.gate(self.abba((1000.0, 1004.0), (1050.0, 1052.0)), "--no-slower", "0.5")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("targeted lane got slower beyond noise", r.stdout)

    def test_no_slower_keeps_other_guards(self):
        r = self.gate(self.abba((1000.0, 1004.0), (1003.0, 1005.0), cgemm=(1100.0, 1100.0)), "--no-slower", "0.5", "--guard-others", "2")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("another lane got slower", r.stdout)

    def test_min_drop_pct(self):
        r = self.gate(self.abba((1000.0, 1010.0), (935.0, 940.0)), "--min-drop-pct", "10")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("did not move beyond noise", r.stdout)

    def test_other_lane_slower_stops(self):
        # E351: a lane gain paid for by another lane (here the GEMMs +5 %) is no gain: guard-others stops it
        runs = self.abba((1000.0, 1010.0), (935.0, 940.0), cgemm=(1050.0, 1050.0))
        r = self.gate(runs, "--guard-others", "2")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("another lane got slower: prefill: GEMMs (f16 WMMA) +5.0 %", r.stdout)
        self.assertEqual(self.gate(runs).returncode, 0)   # without the guard the target lane alone decides

    def test_missing_expected_kernel_stops(self):
        # a switch that never reached serving (stale package, env not passed) shows no new kernel: stop, whatever the lanes say
        r = self.gate(self.abba((1000.0, 1010.0), (935.0, 940.0)), "--expect-kernel", "^g_attention_flash2_smx$")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("no candidate-only kernel matches", r.stdout)
        os.makedirs(os.path.join(self.root, "smx"))
        root = os.path.join(self.root, "smx")
        smx = lambda ms: [("g_attention_flash2_smx", ms, (256, 4, 1))]
        runs = [abba_run(root, "1-ref", 1000.0), abba_run(root, "2-cand", 5.0, extra=smx(930.0)),
                abba_run(root, "3-cand", 5.0, extra=smx(935.0)), abba_run(root, "4-ref", 1010.0)]
        r = self.gate(runs, "--expect-kernel", "^g_attention_flash2_smx$")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        cand_only = r.stdout.split("candidate-only (kernel, workgroups): 1")[1].split("reference-only")[0]
        self.assertIn("g_attention_flash2_smx", cand_only)
        self.assertIn("[256, 4, 1]", cand_only)

    def test_lane_missing_from_candidate_stops(self):
        # a candidate kernel renamed out of the lane's regex lands in a catch-all: the gated lane reads 0 s, a '100 % drop'
        os.makedirs(os.path.join(self.root, "gone"))
        root = os.path.join(self.root, "gone")
        moved = [("g_attn_new", 930.0, (256, 4, 1))]
        runs = [abba_run(root, "1-ref", 1000.0), abba_run(root, "2-cand", 0.0, extra=moved),
                abba_run(root, "3-cand", 0.0, extra=moved), abba_run(root, "4-ref", 1010.0)]
        for r in runs[1:3]:   # drop the flash2 dispatch itself from the candidate runs
            p = os.path.join(r, "events.jsonl")
            keep = [x for x in open(p).readlines() if "g_attention_flash2" not in x]
            open(p, "w").writelines(keep)
        r = self.gate(runs)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("cannot be measured", r.stdout)

    def floor_file(self, table):
        p = os.path.join(self.root, "floor.json")
        open(p, "w").write(json.dumps(table) if not isinstance(table, str) else table)
        return p

    def test_noise_floor_stops_drop_inside_measured_floor(self):
        # HRX-like E485: flash2's start-to-start spread was 4.2 % over 8 starts while one ABBA's two runs per arm read ~1 %:
        # a 3 % drop passes on the ABBA's own spread and is noise against the measured floor
        runs = self.abba((1000.0, 1010.0), (975.0, 978.0))
        self.assertEqual(self.gate(runs).returncode, 0)   # without the floor: passes
        r = self.gate(runs, "--noise-floor", self.floor_file({"prefill: attention": 4.2}))
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("did not move beyond noise", r.stdout)
        self.assertIn("| 4.20 % (floor) |", r.stdout)
        r = self.gate(runs, "--noise-floor", self.floor_file({"prefill: attention": 0.5}))   # a floor below the spread changes nothing
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn("(floor)", r.stdout)

    def test_noise_floor_applies_to_guarded_lanes(self):
        runs = self.abba((1000.0, 1010.0), (935.0, 940.0), cgemm=(1030.0, 1030.0))   # GEMMs +3 %
        self.assertEqual(self.gate(runs, "--guard-others", "2").returncode, 1)
        r = self.gate(runs, "--guard-others", "2", "--noise-floor", self.floor_file({"GEMMs": 4.0}))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_noise_floor_refuses_what_it_cannot_place(self):
        # a key that names no lane (a typo) or several, or a value that is not a number >= 0, must not pass silently: exit 2
        runs = self.abba((1000.0, 1010.0), (935.0, 940.0))
        for table in ({"prefill: atention": 4.2}, {"GDN": 1.0}, {"prefill: attention": "4.2"}, {"prefill: attention": -1},
                      {"prefill: attention": True}, ["prefill: attention", 4.2], "{not json"):
            r = self.gate(runs, "--noise-floor", self.floor_file(table))
            self.assertEqual(r.returncode, 2, f"{table!r}: {r.stdout + r.stderr}")
            self.assertNotIn("LANE GATE:", r.stdout)

    def test_lane_must_match_once(self):
        runs = self.abba((1000.0, 1010.0), (935.0, 940.0))
        for lane in ("GDN", "no such lane"):   # GDN names a decode and a prefill lane
            r = summary(*runs, "--top", "0", "--gate", lane, "--ref", "1,4", "--cand", "2,3")
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        r = summary(*runs, "--top", "0", "--gate", "prefill: attention", "--ref", "1", "--cand", "2,3")
        self.assertEqual(r.returncode, 2)   # one reference run has no spread: no noise estimate, no verdict


FIX = os.path.join(HERE, "..", "kit", "fixtures")


def model_with(root, mutate):
    """a copy of the kit's lanes.json with mutate(d) applied (a lane's base model changed), its path"""
    d = json.load(open(MODEL))
    mutate(d)
    p = os.path.join(root, "lanes.json")
    json.dump(d, open(p, "w"))
    return p


def chunked(sub, durs, key="g_attention_flash2_f16o", wg=(256, 4, 1)):
    """one prefill chunk: one flash2 dispatch per attention layer (16), same kernel and grid in one submission"""
    return [ev(key, sub, ms, wg) for ms in durs]


class LaneBase(unittest.TestCase):
    """HRX-like E625: the flash2 lane is a flat per-layer base plus a machine-state excess (some dispatches 1.2-1.8x slower at
    attention layer >= 6 and context >= ~48k: 0.2 s on some nights, 3.4-7 s on others). q205's gate judged the total (-0.57 %,
    noise 1.5 %: STOP) while the candidate was 6 % faster on its base: the excess, larger in the candidate's two starts, hid it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def q205(self):
        fx = json.load(open(os.path.join(FIX, "events_q205_flash2.json")))
        runs = []
        for name, rows in fx["runs"].items():
            es = [{"type": "dispatch_event", "event_id": i, "submission_id": s, "key": k, "workgroup_count": wg,
                   "duration_ns": ns, "valid": True} for i, s, k, wg, ns in rows]
            runs.append(run_dir(self.root, name, es, prompt=81432))
        return runs

    def test_q205_base_decides_when_excess_differs(self):
        # recurrence: E619/E625, q205 STOPped on 'did not move' while its base dropped 6 % (cmb2 excess 6.8 / 7.0 s vs 4.2 / 3.4 s)
        runs = self.q205()
        r = summary(*runs, "--top", "0", "--gate", "prefill: attention", "--ref", "1,4", "--cand", "2,3")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        per_run = next(x for x in r.stdout.splitlines() if x.startswith("| prefill: attention") and "base" in x and "excess" in x)
        self.assertIn("61.21 / 57.01 / 4.20 s", per_run)   # 1-e480: total / base / excess (s), 42 groups x 16 x the 4th-fastest
        self.assertIn("60.34 / 53.53 / 6.81", per_run)     # 2-cmb2
        verdict = next(x for x in r.stdout.splitlines() if x.startswith("LANE GATE:"))
        self.assertIn("PASS-TO-BENCH", verdict)
        self.assertIn("on its base (-6.01 %", verdict)
        self.assertIn("quiet groups -4.78 %", verdict)    # the excess-free groups (below ~48k context) agree: E625's -4.8 %
        self.assertIn("excess 3.81 -> 6.89 s", verdict)    # the verdict names the excess and that it decided
        self.assertIn("total -0.57 %", verdict)            # the total stays in the verdict
        # a floor between the quiet groups' drop and the base's (5 %: base -6.01 %, quiet -4.78 %): the quiet groups do not agree
        fl = os.path.join(self.root, "floor.json")
        open(fl, "w").write(json.dumps({"prefill: attention": 5.0}))
        r = summary(*runs, "--top", "0", "--gate", "prefill: attention", "--ref", "1,4", "--cand", "2,3", "--noise-floor", fl)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("base -6.01 % not taken: quiet groups -4.78 % (bar -5.00 %)", r.stdout)

    def gate(self, runs, *extra, model=None):
        return subprocess.run([sys.executable, "-I", os.path.join(HERE, "events_summary.py"), model or MODEL, *runs, "--top", "0",
                               "--gate", "prefill: attention", "--ref", "1,4", "--cand", "2,3", *extra], capture_output=True, text=True)

    def arm(self, name, base, excess, layers=16, gemm=None):
        """one run of two chunks; each chunk 16 layers at `base` ms, `excess` ms added to layers 6 and 7 of the second chunk"""
        es = chunked(9, [base] * layers)
        es += chunked(10, [base + (excess / 2 if i in (6, 7) else 0.0) for i in range(layers)])
        if gemm is not None:   # and a verify step: every lane measurable, as --guard-others needs (abba_run's decode shape)
            es += [ev("g_gemm_f16_store_m256", 9, gemm, (40, 16, 1)), ev("g_attention_vsplit", 1, 1.0), ev("g_mv_q5_k_k5120_t8_q8", 1, 2.0),
                   ev("g_dfl_attention", 1, 0.5), ev("g_gdn_recurrence_rec", 1, 0.3), ev("g_rope", 1, 0.1)]
        return run_dir(self.root, name, es)

    def test_excess_alike_keeps_the_total(self):
        # an excess alike in both arms (each arm one quiet and one noisy start) is noise on both sides: the total decides, and a
        # base drop inside the total's noise still stops (a gate that always read the base would pass a coin flip here)
        runs = [self.arm("1-ref", 100.0, 0.0), self.arm("2-cand", 95.0, 0.0), self.arm("3-cand", 95.0, 380.0),
                self.arm("4-ref", 100.0, 400.0)]
        r = self.gate(runs)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        verdict = next(x for x in r.stdout.splitlines() if x.startswith("LANE GATE:"))
        self.assertIn("did not move beyond noise", verdict)
        self.assertNotIn("on its base (", verdict)
        self.assertIn("total and base (excess alike)", r.stdout)

    def test_excess_in_one_arm_decides_on_base(self):
        # q205's shape: base -5 %, total flat (the candidate's starts carry the excess), the excess-free chunk -5 %: passes on base
        runs = [self.arm("1-ref", 100.0, 0.0), self.arm("2-cand", 95.0, 160.0), self.arm("3-cand", 95.0, 170.0),
                self.arm("4-ref", 100.0, 5.0)]
        r = self.gate(runs)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("on its base (-5.00 %", r.stdout)
        self.assertIn("quiet groups -5.00 %", r.stdout)
        # the lane's measured floor (--noise-floor) bounds the base's noise too: two runs per arm understate it (E485)
        fl = os.path.join(self.root, "floor.json")
        open(fl, "w").write(json.dumps({"prefill: attention": 6.0}))
        r = self.gate(runs, "--noise-floor", fl)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("did not move beyond noise: skip the bench", r.stdout)
        # a base rise hidden by a smaller excess in the candidate still reads as slower under --no-slower
        runs2 = [self.arm("1-r", 100.0, 400.0), self.arm("2-c", 104.0, 0.0), self.arm("3-c", 104.0, 5.0), self.arm("4-r", 100.0, 410.0)]
        r = self.gate(runs2, "--no-slower", "0.5")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("targeted lane got slower beyond noise on its base: +4.00 %", r.stdout)

    def test_total_rise_stops_whatever_the_base(self):
        # review r1 F1: the excess the candidate's starts carry may be the candidate's own (a regression shaped like the machine
        # state): a total rise beyond noise stops in every mode, even when the base dropped and the excess differs
        runs = [self.arm("1-ref", 100.0, 0.0), self.arm("2-cand", 95.0, 400.0), self.arm("3-cand", 95.0, 410.0),
                self.arm("4-ref", 100.0, 5.0)]
        for extra in ((), ("--no-slower", "0.5")):
            r = self.gate(runs, *extra)
            self.assertEqual(r.returncode, 1, f"{extra}: {r.stdout + r.stderr}")
            self.assertNotIn("PASS-TO-BENCH", r.stdout)

    def test_guarded_lane_stops_on_total_rise(self):
        # review r1 F1, guard path: flash2's excess only in the candidate's starts (+12 % total, base unchanged) may be the
        # candidate's doing: the guard stops on the total, never excuses it by the base
        runs = [self.arm("1-ref", 100.0, 0.0, gemm=1000.0), self.arm("2-cand", 100.0, 400.0, gemm=930.0),
                self.arm("3-cand", 100.0, 390.0, gemm=932.0), self.arm("4-ref", 100.0, 5.0, gemm=1002.0)]
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "events_summary.py"), MODEL, *runs, "--top", "0", "--gate",
                            "prefill: GEMMs", "--ref", "1,4", "--cand", "2,3", "--guard-others", "2"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("another lane got slower: prefill: attention (flash2) +12.3 %", r.stdout)

    def test_guarded_lane_stops_on_base_rise(self):
        # the guard also reads the base: a candidate that slows every flash2 layer 6 % while its starts carry less excess (total
        # flat) is slower; the base shows it
        runs = [self.arm("1-ref", 100.0, 200.0, gemm=1000.0), self.arm("2-cand", 106.0, 10.0, gemm=930.0),
                self.arm("3-cand", 106.0, 15.0, gemm=932.0), self.arm("4-ref", 100.0, 210.0, gemm=1002.0)]
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "events_summary.py"), MODEL, *runs, "--top", "0", "--gate",
                            "prefill: GEMMs", "--ref", "1,4", "--cand", "2,3", "--guard-others", "2"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("another lane got slower on its base: prefill: attention (flash2) +6.0 %", r.stdout)

    def test_excess_ranges_compared_at_the_larger(self):
        # review r1 F3: the excess 'differs' only beyond the LARGER arm's range: ref excess 0 / 20 (range 20), cand 100 / 300
        # (range 200), means 10 vs 200 (190 apart): alike, so the base cannot pass a total that did not move
        runs = [self.arm("1-ref", 100.0, 0.0), self.arm("2-cand", 95.0, 100.0), self.arm("3-cand", 95.0, 300.0),
                self.arm("4-ref", 100.0, 20.0)]
        r = self.gate(runs)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("total and base (excess alike)", r.stdout)

    def test_candidate_base_spread_is_noise(self):
        # review r1 F3: the candidate's own base spread counts in the base's noise: bases 100 / 100 vs 100 / 112 (+6 %, spread
        # 11.3 %) is no measured rise under --no-slower (a gate on the reference spread alone would stop it)
        runs = [self.arm("1-ref", 100.0, 0.0), self.arm("2-cand", 100.0, 0.0), self.arm("3-cand", 112.0, 0.0),
                self.arm("4-ref", 100.0, 0.0)]
        r = self.gate(runs, "--no-slower", "0.5")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def q194(self, transform=None):
        """q194's A/A flash2 dispatches (starts 1-4; 2 and 3 the identical copy), transform applied to runs 2 and 3"""
        fx = json.load(open(os.path.join(FIX, "events_q194_flash2.json")))
        runs = []
        for n, (name, rows) in enumerate(fx["runs"].items()):
            es = [{"type": "dispatch_event", "event_id": i, "submission_id": s, "key": k, "workgroup_count": wg,
                   "duration_ns": ns, "valid": True} for i, s, k, wg, ns in rows]
            if transform and n in (1, 2):
                es = transform(es)
            sub = es[0]["submission_id"]
            es += [ev("g_gemm_f16_store_m256", sub, 1000.0 * (0.95 if transform and n in (1, 2) else 1.0), (40, 16, 1)),
                   ev("g_attention_vsplit", 10 ** 6, 1.0), ev("g_mv_q5_k_k5120_t8_q8", 10 ** 6, 2.0), ev("g_dfl_attention", 10 ** 6, 0.5),
                   ev("g_gdn_recurrence_rec", 10 ** 6, 0.3), ev("g_rope", 10 ** 6, 0.1)]
            runs.append(run_dir(self.root, name, es, prompt=81430))
        return runs

    @staticmethod
    def groups(es):
        g = {}
        for e in es:
            if e["key"].startswith("g_attention_flash2"):
                g.setdefault((e["submission_id"], e["key"], tuple(e["workgroup_count"])), []).append(e)
        return g

    def late(self, es):
        """review r1 synth.py late: layers >= 6 of the chunks from the 13th on (context >= ~48k) 1.4x slower: a regression the
        candidate causes, shaped like E625's machine-state excess"""
        g = self.groups(es)
        subs = sorted({k[0] for k in g})
        for (s, _, _), v in g.items():
            if subs.index(s) >= 12:
                for li, e in enumerate(sorted(v, key=lambda e: e["event_id"])):
                    if li >= 6:
                        e["duration_ns"] = int(e["duration_ns"] * 1.4)
        return es

    def dispersed(self, es):
        """review r1 synth.py disp: per group half the layers 10 % faster, half 10 % slower: no work saved, more spread"""
        for n, v in enumerate(self.groups(es).values()):
            f = [0.9, 1.1] * 8
            for e, x in zip(v, f[n % 2:] + f[:n % 2]):
                e["duration_ns"] = int(e["duration_ns"] * x)
        return es

    def test_late_regression_stops_no_slower(self):
        # review r1 F1 (synth late): total +17 %, base flat: --no-slower passed it on the base (the old total-only gate stopped)
        r = self.gate(self.q194(self.late), "--no-slower", "0.5")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("targeted lane got slower beyond noise: +17.25 %", r.stdout)

    def test_late_regression_stops_guard(self):
        # review r1 F1 (synth guard): GEMMs -5 % with that flash2 regression: the guard must stop on flash2's total
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "events_summary.py"), MODEL, *self.q194(self.late), "--top", "0",
                            "--gate", "prefill: GEMMs", "--ref", "1,4", "--cand", "2,3", "--guard-others", "2"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("another lane got slower: prefill: attention (flash2) +17.2 %", r.stdout)

    def test_dispersion_is_not_a_speedup(self):
        # review r1 F2 (synth disp): more spread at the same mean reads as a base drop (-8 %) with no work saved; with the kit's
        # flash2 floor the total did not move (-1 %, floor 2.54 %): the base drop needs the excess-free groups to agree, and
        # dispersion leaves none, so it stops
        fl = os.path.join(self.root, "floor.json")
        open(fl, "w").write(json.dumps({"prefill: attention": 2.54}))
        r = self.gate(self.q194(self.dispersed), "--noise-floor", fl)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        verdict = next(x for x in r.stdout.splitlines() if x.startswith("LANE GATE:"))
        self.assertIn("did not move beyond noise", verdict)
        self.assertIn("no quiet groups", verdict)

    def test_q194_aa_does_not_pass_on_base(self):
        # the A/A copy (q194 starts 1-4) under the kit floor: neither a speed-up nor a slowdown
        fl = os.path.join(self.root, "floor.json")
        open(fl, "w").write(json.dumps({"prefill: attention": 2.54}))
        runs = self.q194()
        self.assertEqual(self.gate(runs, "--noise-floor", fl).returncode, 1)
        self.assertEqual(self.gate(runs, "--noise-floor", fl, "--no-slower", "0.5").returncode, 0)

    def custom(self, name, chunks):
        """one run of the given prefill chunks, each a list of per-layer dispatch times (ms), one submission per chunk"""
        es = []
        for n, durs in enumerate(chunks):
            es += chunked(9 + n, durs)
        return run_dir(self.root, name, es)

    @staticmethod
    def layers(base, hi=None, excess=0.0, lo=None):
        """16 layers at base; layers 4-15 at hi; layers 0-2 at lo; excess / 2 added to layers 6 and 7"""
        out = [base] * 16
        if hi is not None:
            out[4:] = [hi] * 12
        if lo is not None:
            out[:3] = [lo] * 3
        out[6] += excess / 2
        out[7] += excess / 2
        return out

    def test_base_noise_takes_the_lane_floor(self):
        # review r2 survivor (bnoise without the floor): a base rise of 4 % inside a 6 % lane floor is no measured rise under
        # --no-slower (the floor is the lane's start-to-start spread, E485); total down (the reference starts carry the excess)
        fl = os.path.join(self.root, "floor.json")
        open(fl, "w").write(json.dumps({"prefill: attention": 6.0}))
        runs = [self.arm("1-r", 100.0, 400.0), self.arm("2-c", 104.0, 0.0), self.arm("3-c", 104.0, 5.0), self.arm("4-r", 100.0, 410.0)]
        r = self.gate(runs, "--no-slower", "0.5", "--noise-floor", fl)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("base +4.00 %, bar +6.00 %", r.stdout)

    def test_base_rise_bar_takes_the_mode_allowance(self):
        # review r2 survivor (base rise bar = its noise only): --no-slower 5 allows a 5 % rise; a base +4 % with no spread passes
        runs = [self.arm("1-r", 100.0, 400.0), self.arm("2-c", 104.0, 0.0), self.arm("3-c", 104.0, 5.0), self.arm("4-r", 100.0, 410.0)]
        r = self.gate(runs, "--no-slower", "5")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("base +4.00 %, bar +5.00 %", r.stdout)

    def test_quiet_noise_takes_the_candidate_spread(self):
        # review r2 survivor (qnoise without spread(cq)): the candidate's quiet groups 1520 / 1656.8 ms (spread 8.6 %, mean -0.7 %)
        # do not show a drop; the base (-5 %, no spread) alone must not pass
        runs = [self.custom("1-ref", [self.layers(100.0), self.layers(100.0)]),
                self.custom("2-cand", [self.layers(95.0), self.layers(95.0, excess=160.0)]),
                self.custom("3-cand", [self.layers(95.0, hi=95.0 * 1.12), self.layers(95.0, excess=170.0)]),
                self.custom("4-ref", [self.layers(100.0), self.layers(100.0, excess=5.0)])]
        r = self.gate(runs)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("base -5.00 % not taken: quiet groups -0.72 % (bar -8.61 %)", r.stdout)

    def test_runs_with_different_group_counts_have_no_quiet_groups(self):
        # review r2 survivor (group-count check -> True): a run with a chunk more than the others cannot line its groups up: no
        # quiet groups, no IndexError
        runs = [self.custom("1-ref", [self.layers(100.0)] * 3), self.custom("2-cand", [self.layers(95.0)] * 2),
                self.custom("3-cand", [self.layers(95.0)] * 2), self.custom("4-ref", [self.layers(100.0)] * 2)]
        r = self.gate(runs)
        self.assertNotIn("Traceback", r.stderr)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertNotIn("quiet groups (", r.stdout)

    def test_quiet_drop_bar_takes_min_drop(self):
        # review r2 survivor (quiet bar ignores --min-drop-pct): base -10 % beyond a 5 % minimum, quiet groups -0.55 %: the quiet
        # groups did not drop by the minimum, so the base does not pass
        c1 = self.layers(90.0, hi=90.0 * 1.14)   # rank 4 at 90, the other 12 at 102.6 (inside 1.15 x 90: quiet)
        runs = [self.custom("1-ref", [self.layers(100.0), self.layers(100.0)]),
                self.custom("2-cand", [c1, self.layers(90.0, excess=169.0)]),
                self.custom("3-cand", [c1, self.layers(90.0, excess=179.0)]),
                self.custom("4-ref", [self.layers(100.0), self.layers(100.0, excess=5.0)])]
        r = self.gate(runs, "--min-drop-pct", "5")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("base -10.00 % not taken: quiet groups -0.55 % (bar -5.00 %)", r.stdout)

    def test_quiet_threshold_is_relative_to_rank_r(self):
        # review r2 survivor (x[-1] < q * x[0]): a group whose 3 fastest layers run at 0.8 x and slowest at 1.1 x its 4th-fastest
        # is quiet (1.1 <= 1.15); read against the fastest it would not be, and q205's shape would lose its quiet groups
        runs = [self.custom("1-ref", [self.layers(100.0, hi=100.0, lo=80.0), self.layers(100.0)]),
                self.custom("2-cand", [self.layers(95.0, lo=76.0)[:15] + [104.5], self.layers(95.0, excess=160.0)]),
                self.custom("3-cand", [self.layers(95.0, lo=76.0)[:15] + [104.5], self.layers(95.0, excess=170.0)]),
                self.custom("4-ref", [self.layers(100.0, lo=80.0)[:15] + [110.0], self.layers(100.0, excess=5.0)])]
        r = self.gate(runs)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("quiet groups (1 of 2)", r.stdout)

    def test_zero_base_cannot_be_measured(self):
        # a base of 0 (every group's R-th fastest dispatch read 0 ns) gives no ratio: the lane cannot be measured, never a pass
        # through a NaN comparison (--no-slower's 'rise > bar' is False for NaN)
        runs = [self.arm("1-ref", 0.0, 10.0), self.arm("2-cand", 0.0, 400.0), self.arm("3-cand", 0.0, 410.0), self.arm("4-ref", 0.0, 12.0)]
        r = self.gate(runs, "--no-slower", "0.5")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("cannot be measured", r.stdout)

    def test_lane_without_model_keeps_total(self):
        # a lane with no declared work model is judged on its total and says so (GEMMs: one kernel and grid serve layers of
        # different K, so equal grids are not equal work)
        runs = [self.arm("1-ref", 100.0, 0.0), self.arm("2-cand", 95.0, 400.0), self.arm("3-cand", 95.0, 410.0),
                self.arm("4-ref", 100.0, 5.0)]
        m = model_with(self.root, lambda d: next(ln for ln in d["lanes"] if ln["lane"].startswith("prefill: attention")).pop("base"))
        r = self.gate(runs, model=m)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("total (no base model)", r.stdout)

    def test_model_that_does_not_fit_keeps_total(self):
        # a group that is not one dispatch per layer (15 here: a layer moved to another kernel) cannot give a base: the total
        # decides and the table says why, never a traceback (the prototype raised IndexError on a run without flash2 dispatches)
        runs = [self.arm("1-ref", 100.0, 0.0), self.arm("2-cand", 95.0, 400.0, layers=15), self.arm("3-cand", 95.0, 410.0),
                self.arm("4-ref", 100.0, 5.0)]
        r = self.gate(runs)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        self.assertIn("does not fit", r.stdout)
        self.assertIn("total (base unavailable: model does not fit", r.stdout)

    def test_bad_model_refused(self):
        # a base model whose rank is outside its group (or not an integer) is a configuration error: exit 2, no verdict
        for bad in ({"group_size": 16, "rank": 17}, {"group_size": 16, "rank": 0}, {"group_size": "16", "rank": 4}, {"rank": 4},
                    {"group_size": 16, "rank": 4, "slow_ratio": 1.0}, {"group_size": 16, "rank": 4, "slow_ratio": "1.15"}):
            def mut(d, bad=bad):
                next(ln for ln in d["lanes"] if ln["lane"].startswith("prefill: attention"))["base"] = bad
            r = self.gate([self.arm(f"{i}-x{len(os.listdir(self.root))}", 100.0, 0.0) for i in range(1, 5)], model=model_with(self.root, mut))
            self.assertEqual(r.returncode, 2, f"{bad}: {r.stdout + r.stderr}")
            self.assertNotIn("LANE GATE:", r.stdout)


SPEC = {
    # review r1 round-2 state-space table (the hand-off's): verdict per gate mode for the lane's total (t) and base (b) moving
    # up / flat / down beyond their own noise, the excess alike (a) or differing (d), and the excess-free ('quiet') groups
    # dropping (q+) or not (q-). Speed: PASS-base = passes on the base. Guard: STOP = the guarded lane stops the candidate.
    "speed": """
        b=up      b=flat    b=down,a  b=down,d,q+  b=down,d,q-  b=none
t=up    STOP      STOP      STOP      STOP         STOP         STOP
t=flat  STOP      STOP      STOP      PASS-base    STOP         STOP
t=down  STOP      PASS      PASS      PASS         PASS         PASS
""",
    "no_slower": """
        b=up      b=flat    b=down,a  b=down,d,q+  b=down,d,q-  b=none
t=up    STOP      STOP      STOP      STOP         STOP         STOP
t=flat  STOP      PASS      PASS      PASS         PASS         PASS
t=down  STOP      PASS      PASS      PASS         PASS         PASS
""",
    "guard": """
        b=up      b=flat    b=down,a  b=down,d,q+  b=down,d,q-  b=none
t=up    STOP      STOP      STOP      STOP         STOP         STOP
t=flat  STOP      PASS      PASS      PASS         PASS         PASS
t=down  STOP      PASS      PASS      PASS         PASS         PASS
""",
}


class GateTable(unittest.TestCase):
    """events_summary.verdict implements SPEC cell by cell, for both excess states where the column does not name one; an
    unmeasured total or base stops in every mode (fail closed)"""

    def test_every_cell(self):
        sys.path.insert(0, HERE)
        import events_summary as es_mod
        cols = {"b=up": [("up", e, q) for e in ("alike", "differs") for q in ("down", "not")],
                "b=flat": [("flat", e, q) for e in ("alike", "differs") for q in ("down", "not")],
                "b=down,a": [("down", "alike", q) for q in ("down", "not")],
                "b=down,d,q+": [("down", "differs", "down")], "b=down,d,q-": [("down", "differs", "not")],
                "b=none": [(None, None, None)]}
        for mode, text in SPEC.items():
            lines = [ln.split() for ln in text.strip().splitlines()]
            head = lines[0]
            for row in lines[1:]:
                t = row[0][2:]
                for col, want in zip(head, row[1:]):
                    for b, e, q in cols[col]:
                        got, measure, _ = es_mod.verdict(mode, t, b, e, q)
                        exp = "PASS" if want.startswith("PASS") else "STOP"
                        self.assertEqual(got, exp, f"{mode} t={t} {col} ({b}, {e}, {q}): {got}")
                        if want == "PASS-base":
                            self.assertEqual(measure, "base", f"{mode} t={t} {col}")
            for b in ("up", "flat", "down", None, "unmeasured"):
                self.assertEqual(es_mod.verdict(mode, "unmeasured", b, "differs", "down")[0], "STOP", f"{mode} t unmeasured")
            for t in ("up", "flat", "down"):
                self.assertEqual(es_mod.verdict(mode, t, "unmeasured", "differs", "down")[0], "STOP", f"{mode} b unmeasured")


class BusyClock(unittest.TestCase):
    """HRX-like E624: the cycles table read tel.txt's 2-decimal busy clock (2.18 / 2.19 GHz: 0.46 % steps) against a 0.27 % lane
    floor; q205's tel.tsv medians were 2.176 / 2.185 / 2.186 / 2.180."""

    def test_clock_difference_below_two_decimals_shows(self):
        # two runs whose busy clocks differ by 0.2 % (2.180 vs 2.184 GHz) must read different clocks and cycles
        with tempfile.TemporaryDirectory() as root:
            runs = []
            for name, ghz in (("1-a", 2.180), ("2-b", 2.184)):
                d = run_dir(root, name, [ev("g_attention_flash2", 7, 50.0), ev("g_gemm_f16_swiglu_m256", 7, 1000.0, (272, 16, 1))])
                with open(os.path.join(d, "tel.tsv"), "w") as f:
                    for i in range(20):
                        f.write(f"{1000 + i * 0.1:.2f}\t100\t85.0\t{ghz:.3f}\t80\t1600\t1000\t1104\t80\t0.5\t900\t0\n")
                with open(os.path.join(d, "tel.txt"), "w") as f:   # the summary as events_ab.sh writes it, by the producer itself
                    f.write(subprocess.run([sys.executable, "-I", os.path.join(HERE, "gpu_telemetry.py"), "summary",
                                            os.path.join(d, "tel.tsv"), name], capture_output=True, text=True).stdout)
                runs.append(d)
            r = summary(*runs, "--top", "0")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("1-a @ 2.180 GHz | 2-b @ 2.184 GHz", r.stdout)
            line = next(x for x in r.stdout.splitlines() if x.startswith("| prefill: GEMMs") and "G |" in x)
            self.assertIn("2.18 G | 2.18 G", line)   # 1.0 s x 2.180 vs 1.0 s x 2.184: same seconds, cycles 0.2 % apart
            self.assertIn("| 0.0 % | 0.2 % |", line)

    def test_old_tel_txt_still_parses(self):
        # an 11-25 run directory (q205's 1-e480) holds a 2-decimal tel.txt; a run without its tel.tsv still reads that clock
        with tempfile.TemporaryDirectory() as root:
            d = run_dir(root, "1-e480", [ev("g_attention_flash2", 7, 50.0), ev("g_gemm_f16_swiglu_m256", 7, 1000.0, (272, 16, 1))])
            open(os.path.join(d, "tel.txt"), "w").write(open(os.path.join(FIX, "tel_q205_1-e480.txt")).read())
            r = summary(d, "--top", "0")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("1-e480 @ 2.180 GHz", r.stdout)
            self.assertIn("clock from a 2-decimal tel.txt (no tel.tsv) for 1-e480", r.stdout)   # and says it is coarse
            # the same 11-25 directory with its tel.tsv beside (every events_ab run wrote both): the recorded samples decide
            with open(os.path.join(d, "tel.tsv"), "w") as f:
                for i, ghz in enumerate(["2.172", "2.176", "2.176", "2.190", "2.170", "2.280", "1.100"]):
                    f.write(f"{1000 + i * 0.1:.2f}\t{0 if ghz == '1.100' else 100}\t85.0\t{ghz}\t80\t1600\t1000\t1104\t80\t0.5\t900\t0\n")
            r = summary(d, "--top", "0")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("1-e480 @ 2.176 GHz", r.stdout)   # the busy samples' median (the idle 1.100 is not busy)
            self.assertNotIn("2-decimal tel.txt", r.stdout)


class LaneBreakdown(unittest.TestCase):
    def test_matvec_gbs_uses_row_groups_on_y(self):
        # matvec launches are workgroups(token_groups, row_groups, 1): rows = y x %rows_per (11-21: reading x gave 0-1 GB/s)
        with tempfile.TemporaryDirectory() as root:
            # 1088 row groups x 16 rows x K 5120 x 5.5 bits / 8 = 61.3 MB in 0.3 ms -> ~204 GB/s
            d = run_dir(root, "m", [ev("g_attention_vsplit", 1, 1.0), ev("g_mv_q5_k_k5120_t8_q8", 1, 0.3, (1, 1088, 1))])
            prog = os.path.join(root, "p.loom")
            open(prog, "w").write('kernel.def export("g_mv_q5_k_k5120_t8_q8") @g_mv_q5_k_k5120_t8_q8(%n: index) {\n'
                                  '  %rows_per = index.constant 16 : index\n}\n')
            r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "lane_breakdown.py"), MODEL, d, prog,
                                "decode: target weight matvecs"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            line = next(x for x in r.stdout.splitlines() if "g_mv_q5_k_k5120_t8_q8" in x)
            self.assertEqual(line.split()[-1], "204")


class WaitPid(unittest.TestCase):
    W = os.path.join(HERE, "wait_pid.sh")

    def test_empty_pid_fails_loudly(self):   # 11-21: an empty PID made the waiter return at once
        r = subprocess.run(["bash", self.W, "", "bash q31.sh"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2, r.stderr)

    def test_wrong_process_refused(self):
        p = subprocess.Popen(["sleep", "30"])
        try:
            r = subprocess.run(["bash", self.W, str(p.pid), "bash q31.sh", "1"], capture_output=True, text=True, timeout=10)
            self.assertEqual(r.returncode, 3, r.stderr)
        finally:
            p.kill()

    def test_waits_for_the_right_process(self):
        p = subprocess.Popen(["sleep", "2"])
        t = time.time()
        r = subprocess.run(["bash", self.W, str(p.pid), "sleep 2", "1"], capture_output=True, text=True, timeout=20)
        p.wait()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertGreater(time.time() - t, 1.0)


if __name__ == "__main__":
    unittest.main()
