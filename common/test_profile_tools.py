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
        self.assertIn("cycles: no busy clock (tel.txt) for 2-hot", r.stdout)

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
