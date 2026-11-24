#!/usr/bin/env python3
"""Tests of draft_depth.py through its command line. Each case names the recurrence it prevents.
usage: python3 -I test_draft_depth.py"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(HERE, "..")
FLAT = {"lanes": [{"name": "w", "ms": 80, "scaling": "flat"}]}


def run(*args):
    return subprocess.run([sys.executable, "-I", os.path.join(HERE, "draft_depth.py"), *args], capture_output=True, text=True)


def row(out, d):   # the table row of depth d
    return next(l for l in out.splitlines() if l.startswith(f"| {d} |")).split("|")[1:-1]


class DraftDepth(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def doc(self, d, name="d.json"):
        p = os.path.join(self.tmp.name, name)
        with open(p, "w") as f:
            json.dump(d, f)
        return p

    def test_qwen_positive_check(self):
        # positive check: DFlash2 at depth 7 with the agent bench's acceptance reproduces the served 3.25 tokens per step (E381)
        r = run(os.path.join(ROOT, "kits", "qwen38-27b", "drafter.json"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("expect depth 7: 3.25 tokens/step, model 3.250", r.stdout)
        self.assertEqual(row(r.stdout, 7)[5].strip(), "115.09")   # the limits table's step (amdahl.py: 115.10)
        self.assertTrue(r.stdout.rstrip().endswith("draft_depth: BEST d=7"))

    def test_replay_rows_give_agent_ab_tokens_per_step(self):
        # the served point must use agent_ab.sh's formula out / (out - accepted), draft_acc being a rate (replay.py)
        rows = [{"out": 100, "draft_acc": 0.5, "draft_proposed": 100}, {"out": 50, "draft_acc": 0.25, "draft_proposed": 40}]
        rp = self.doc({"summary": {}, "rows": rows}, "h1.jsonl")
        r = run(self.doc({"drafter": {"kind": "block", "ms": 5, "max_depth": 4}, "verify": FLAT}), "--replay", rp, "--replay-depth", "4")
        self.assertIn("served depth 4: 1.667 tokens/step", r.stdout)   # 150 / (150 - 60)
        self.assertEqual(run(self.doc({"drafter": {"kind": "block", "ms": 5}, "verify": FLAT}), "--replay", rp).returncode, 2)
        # llama.cpp rows carry counts, not a rate (the sampling-drafter pricing route): 100 / (100 - 40)
        lp = self.doc({"summary": {}, "rows": [{"out": 100, "draft_n": 70, "draft_n_accepted": 40}]}, "l.jsonl")
        r = run(self.doc({"drafter": {"kind": "block", "ms": 5, "max_depth": 4}, "verify": FLAT}), "--replay", lp, "--replay-depth", "4")
        self.assertIn("served depth 4: 1.667 tokens/step", r.stdout)

    def test_histogram_gives_conditional_acceptance(self):
        # a_2 is conditional on reaching position 2, not the share of all steps (2 of 4 reach it, 1 passes)
        r = run(self.doc({"acceptance": {"accept_hist": {"0": 2, "1": 1, "2": 1}},
                          "drafter": {"kind": "block", "ms": 1, "max_depth": 2}, "verify": FLAT}))
        self.assertEqual(row(r.stdout, 1)[1].strip(), "0.500")
        self.assertEqual(row(r.stdout, 2)[1].strip(), "0.500")
        self.assertEqual(row(r.stdout, 2)[2].strip(), "1.750")

    def test_autoregressive_drafter_pays_per_depth(self):
        # an MTP / small-model drafter runs a step per token: priced as one block pass it would always pick the deepest depth
        acc = {"position_probs": [0.8, 0.6, 0.4, 0.3, 0.2]}
        blk = run(self.doc({"acceptance": acc, "drafter": {"kind": "block", "ms": 10, "max_depth": 5}, "verify": FLAT}))
        ar = run(self.doc({"acceptance": acc, "drafter": {"kind": "autoregressive", "step_ms": 10, "max_depth": 5}, "verify": FLAT}))
        self.assertTrue(blk.stdout.rstrip().endswith("BEST d=5"), blk.stdout)
        self.assertEqual(row(ar.stdout, 3)[4].strip(), "30.00")
        self.assertTrue(ar.stdout.rstrip().endswith("BEST d=2"), ar.stdout)

    def test_moe_verify_grows_with_width_and_measured_widths_do_not_extrapolate(self):
        # MoE: extra verify rows activate extra experts (make_lanes --verify-tokens: 8 tokens ~2.8x one, E217); a flat
        # verify cost would overrate deep drafts. Measured tables are never extrapolated.
        moe = {"lanes": [{"name": "moe", "ms": 30, "width": 8, "scaling": "moe_union", "expert_share": 0.8, "top_k": 8, "experts": 128}]}
        r = run(self.doc({"acceptance": {"position_probs": [0.5] * 7}, "drafter": {"kind": "block", "ms": 1, "max_depth": 7}, "verify": moe}))
        self.assertLess(float(row(r.stdout, 0)[3]), float(row(r.stdout, 7)[3]) / 2)
        r = run(self.doc({"acceptance": {"position_probs": [0.5] * 7}, "drafter": {"kind": "block", "ms": 1, "max_depth": 7},
                          "verify": {"ms_by_width": {"1": 80, "4": 90}}}))
        self.assertEqual(row(r.stdout, 1)[3].strip(), "83.33")   # width 2: linear between the measured 1 and 4
        self.assertEqual(row(r.stdout, 5)[3].strip(), "unknown")
        self.assertTrue(r.stdout.rstrip().endswith("BEST d=3"))

    def test_one_depth_fit_is_a_model_two_depths_fix_it(self):
        # E126 (same 20 turns, card sampling): d7 4.013, d3 3.023 tokens/step. A constant a from d7 predicts d3 at
        # 2.90 (-4 %): the per-position acceptance falls with depth; two depths fit both
        one = run(self.doc({"acceptance": {"served": [{"depth": 7, "tokens_per_step": 4.013}]},
                            "drafter": {"kind": "block", "ms": 1, "max_depth": 7}, "verify": FLAT}))
        self.assertEqual(row(one.stdout, 3)[2].strip(), "2.897")
        two = run(self.doc({"acceptance": {"served": [{"depth": 7, "tokens_per_step": 4.013}, {"depth": 3, "tokens_per_step": 3.023}]},
                            "drafter": {"kind": "block", "ms": 1, "max_depth": 7}, "verify": FLAT}))
        for d, want in ((7, 4.013), (3, 3.023)):
            self.assertAlmostEqual(float(row(two.stdout, d)[2]), want, delta=want * 0.01)

    def test_invalid_and_expect_miss(self):
        # 9 tokens per step at depth 7 is impossible (at most d + 1): a units slip, not data
        bad = run(self.doc({"acceptance": {"served": [{"depth": 7, "tokens_per_step": 9}]}, "drafter": {"kind": "block", "ms": 1}, "verify": FLAT}))
        self.assertEqual(bad.returncode, 2)
        self.assertTrue(bad.stdout.rstrip().endswith("draft_depth: INVALID"))
        miss = run(self.doc({"acceptance": {"position_probs": [0.5] * 7}, "drafter": {"kind": "block", "ms": 1}, "verify": FLAT,
                             "expect": {"depth": 7, "tokens_per_step": 3.25}}))
        self.assertEqual(miss.returncode, 1)
        self.assertTrue(miss.stdout.rstrip().endswith("draft_depth: FAIL"))


if __name__ == "__main__":
    unittest.main()
