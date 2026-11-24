"""ktest_delta: a candidate's kernel tests judged by their delta against the base package.
Recurrence: E445 (q166 stopped on a stale-KV defect live had too; the fix was a hand-tagged KNOWN-FAIL list that goes stale)."""
import os
import subprocess
import sys
import tempfile
import unittest

H = os.path.dirname(os.path.abspath(__file__))
BASE = """PASS gemm_a
FAIL stale_kv_flash_nan element at index 632832: actual f32 nan is not close
KNOWN-FAIL (E444) stale_kv_split_decode_nan element at index 0: actual f32 nan
PASS stale_kv_negcontrol
226/228 passed
"""


def run(cand, base=BASE):
    d = tempfile.mkdtemp()
    a, b = os.path.join(d, "c.log"), os.path.join(d, "b.log")
    open(a, "w").write(cand); open(b, "w").write(base)
    p = subprocess.run([sys.executable, os.path.join(H, "ktest_delta.py"), a, b], capture_output=True, text=True)
    return p.returncode, p.stdout


class T(unittest.TestCase):
    def test_inherited_failures_pass(self):   # the E445 shape: live fails the same stale-KV cases
        rc, out = run("PASS gemm_a\nFAIL stale_kv_flash_nan x\nFAIL stale_kv_split_decode_nan y\n")
        self.assertEqual(rc, 0, out)
        self.assertIn("INHERITED stale_kv_flash_nan", out)
        self.assertIn("INHERITED stale_kv_split_decode_nan", out)   # a KNOWN-FAIL on the base counts as failing there

    def test_new_failure_stops(self):
        rc, out = run("FAIL gemm_a element 3\nFAIL stale_kv_flash_nan x\n")
        self.assertEqual(rc, 1, out)
        self.assertIn("NEW FAIL gemm_a", out)

    def test_case_the_base_did_not_run_stops(self):   # fail closed: no base result is not an inherited failure
        rc, out = run("FAIL vsplit_new_case z\n")
        self.assertEqual(rc, 1, out)
        self.assertIn("did not run this case", out)

    def test_run_level_failure_stops(self):
        for line in ("FAIL compile: loom-test produced no results", "FAIL select: ktest_select.py failed, no cases run",
                     "FAIL KNOWN-DEFECT NOW PASSES stale_kv_flash_nan: remove the tag (E444)"):
            rc, out = run(line + "\n")
            self.assertEqual(rc, 1, line + " -> " + out)

    def test_run_level_failure_on_both_still_stops(self):   # no cases ran on either side is not an inherited pass
        rc, out = run("FAIL compile: loom-test produced no results\n",
                      base=BASE + "FAIL compile: loom-test produced no results\n")
        self.assertEqual(rc, 1, out)

    def test_empty_logs_refused(self):
        self.assertEqual(run("")[0], 2)
        self.assertEqual(run("PASS gemm_a\n", base="")[0], 2)

    def test_all_pass(self):
        rc, out = run("PASS gemm_a\nPASS stale_kv_negcontrol\n")
        self.assertEqual(rc, 0, out)
        self.assertIn("0 new, 0 inherited", out)


if __name__ == "__main__":
    unittest.main()
