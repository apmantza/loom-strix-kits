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
