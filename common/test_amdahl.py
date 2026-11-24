#!/usr/bin/env python3
"""Tests of amdahl.py and ceiling.py's device profiles through their command lines. Each case names the recurrence it prevents.
usage: python3 -I test_amdahl.py"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(HERE, "..")
DEV = os.path.join(HERE, "devices", "gfx1151.json")


def run(tool, *args):
    return subprocess.run([sys.executable, "-I", os.path.join(HERE, tool), *args], capture_output=True, text=True)


class Amdahl(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def doc(self, d):
        p = os.path.join(self.tmp.name, "c.json")
        with open(p, "w") as f:
            json.dump(d, f)
        return p

    def phase(self, cands, **kw):
        return dict({"phase": "decode", "unit": "ms/step", "wall": 100.0, "candidates": cands}, **kw)

    def test_overlap_group_counts_the_largest_once(self):
        # two rewrites of one kernel summed promise time that exists once (the reason overlap groups exist)
        r = run("amdahl.py", self.doc(self.phase([
            {"name": "a", "target": 40, "reduction": 10, "overlap_group": "attn"},
            {"name": "b", "target": 40, "reduction": 6, "overlap_group": "attn"},
            {"name": "c", "target": 20, "reduction": 5, "overlap_group": "mv"}])))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("100.00 -> 85.00 ms/step", r.stdout)
        self.assertTrue(r.stdout.rstrip().endswith("amdahl: PASS"))

    def test_screen_drops_noise_level_levers_keeps_enablers(self):
        # a 0.5 % lever under a 1 % A/A floor cannot be told from noise (E374) and must not enter the combined gain
        r = run("amdahl.py", self.doc(self.phase([
            {"name": "big", "target": 50, "reduction_pct": 10},
            {"name": "tiny", "target": 50, "reduction": 0.5},
            {"name": "port", "target": 10, "enabler": True}])))
        self.assertIn("| tiny | tiny | 50.00 | 0.50 | 0.5 | +0.5 % | below screen |", r.stdout)
        self.assertIn("| port | port | 10.00 | 0.00 | 0.0 | +0.0 % | ENABLER |", r.stdout)
        self.assertIn("100.00 -> 95.00 ms/step", r.stdout)
        self.assertIn("combined, every lever: 100.00 -> 94.50", r.stdout)

    def test_no_lever_clears_the_screen_fails(self):
        r = run("amdahl.py", self.doc(self.phase([{"name": "tiny", "target": 50, "reduction": 0.5}])))
        self.assertEqual(r.returncode, 1)
        self.assertTrue(r.stdout.rstrip().endswith("amdahl: FAIL"))

    def test_invalid_inputs_exit_2(self):
        # a saving larger than the part it acts on, or two forms of the saving, is a typo that would inflate the total
        for c in ({"name": "x", "target": 10, "reduction": 11}, {"name": "x", "target": 10, "to": 5, "reduction": 5},
                  {"name": "x", "target": 120, "reduction": 1}):
            r = run("amdahl.py", self.doc(self.phase([c])))
            self.assertEqual(r.returncode, 2, c)
            self.assertTrue(r.stdout.rstrip().endswith("amdahl: INVALID"))

    def test_expect_reports_a_discrepancy(self):
        # the limits table's parts must add up to its stated totals (the brief's positive check): a miss is reported
        r = run("amdahl.py", self.doc(self.phase([{"name": "a", "target": 50, "to": 40}],
                                                 expect={"wall": 100, "after": 80, "tolerance_pct": 2})))
        self.assertEqual(r.returncode, 1)
        self.assertIn("DISCREPANCY", r.stdout)

    def test_qwen_limits_table_reproduces(self):
        # positive check: LEDGER's limits table (E369 construction, E350's HEADPACK) at the 85 W floors
        r = run("amdahl.py", os.path.join(ROOT, "kits", "qwen38-27b", "limits.json"), "--device", DEV)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("115.10 -> 90.09 ms/step", r.stdout)
        self.assertIn("196.75 -> 129.38 s/prompt", r.stdout)
        self.assertIn("| compute | x1.631 |", r.stdout)

    def cell(self, **kw):
        return dict({"cell": "d", "kind": "decode", "tokens_per_s": 30.0, "tokens_per_step": 3.0,
                     "min_bytes_per_step": 12e9, "busy_share": 0.8}, **kw)

    def test_headroom_axes_multiply_to_the_bandwidth_gap(self):
        # orthogonal axes: traffic x kernel bandwidth x dispatch must equal capped bw / minimum-bytes rate, or the
        # stack double-counts. 10 steps/s x 12 GB = 120 GB/s of 240: x2.0 in all
        r = run("amdahl.py", self.doc({"cells": [self.cell(measured_bytes_per_step=15e9)]}), "--device", DEV)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("| traffic | x1.250 |", r.stdout)
        self.assertIn("| kernel_bandwidth | x1.280 |", r.stdout)
        self.assertIn("| dispatch | x1.250 |", r.stdout)
        self.assertIn("product of the known axes: x2.000 (some axes unknown)", r.stdout)

    def test_stack_best_per_axis_and_ceiling(self):
        # the overlap-group rule on multipliers; a claim above the physics ceiling (x1.6 on bandwidth = x1.6 here) fails
        c = self.cell(stack=[{"name": "p", "axis": "bandwidth", "multiplier": 1.2},
                             {"name": "q", "axis": "bandwidth", "multiplier": 1.1},
                             {"name": "f", "axis": "dispatch", "multiplier": 1.2}])
        r = run("amdahl.py", self.doc({"cells": [c]}), "--device", DEV)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("(1 on the same axis not counted)", r.stdout)
        self.assertIn("stacked: x1.440", r.stdout)
        c["stack"][0]["multiplier"] = 1.7
        r = run("amdahl.py", self.doc({"cells": [c]}), "--device", DEV)
        self.assertEqual(r.returncode, 1)
        self.assertIn("EXCEEDS the axis ceiling", r.stdout)

    def test_cells_need_a_device(self):
        # the capped limits belong to the device file, not to the model: no silent nominal default
        r = run("amdahl.py", self.doc({"cells": [self.cell()]}))
        self.assertEqual(r.returncode, 2)


class CeilingDevice(unittest.TestCase):
    LANES = os.path.join(ROOT, "kits", "qwen38-27b", "lanes.json")

    def test_cap_profile_derives_the_peak_from_the_clock(self):
        # E369: 80 SIMDs x 8,192 FLOP / 32 cycles x 2.14 GHz = 43.8 TFLOP/s; floors at 45 overstated the reachable limit (E358)
        r = run("ceiling.py", self.LANES, "--device", DEV, "--context", "81430", "--prompt", "81430")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("240 GB/s practical, 43.83 TFLOP/s", r.stdout)
        self.assertIn("decode: target weight matvecs                 floor     69.08", r.stdout)

    def test_clock_override_and_nominal_profile(self):
        r = run("ceiling.py", self.LANES, "--device", DEV, "--clock-ghz", "2.0")
        self.assertIn("40.96 TFLOP/s", r.stdout)
        # the nominal profile must give what lanes.json alone gives (no drift between the two sources)
        a = run("ceiling.py", self.LANES).stdout.splitlines()[1:]
        b = run("ceiling.py", self.LANES, "--device", DEV, "--profile", "nominal").stdout.splitlines()[1:]
        self.assertEqual(a, b)

    def test_profile_without_device_refused(self):
        self.assertNotEqual(run("ceiling.py", self.LANES, "--clock-ghz", "2.0").returncode, 0)


if __name__ == "__main__":
    unittest.main()
