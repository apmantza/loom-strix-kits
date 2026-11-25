#!/usr/bin/env python3
"""kit/ktest_select.py's selected generation pieces (common/ktest.sh KTEST_CHANGED): the changed set read from binary_gate's rows and
the case filter by launched kernel. Recurrences it prevents:
  - selecting cases by name: flash2's cases are named flash_*, stale_kv_*, nob3_*, flataddr_* (HRX-like E532 / q207: 29 of 323 cases
    launch flash2, a name match found fewer); a case is kept when it launches a changed kernel, whatever it is called;
  - a selected run that tests nothing: an unreadable gate file, a kernel the gate could not judge (ERROR) or a verdict this code
    does not know gives 'changed set unknown' (ktest.sh then runs the full suite), never an empty selection;
  - a kt_K copy (a test-local clone of a kernel a case root cannot run) not counted as testing K;
  - helper kernel definitions in the cases file dropped with the cases around them (the kept cases would not link).
usage: python3 kit/test_ktest_select.py   (no GPU)"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ktest_select as ks  # noqa: E402


def case(name, *kernels):
    return f"check.case public @{name} {{\n" + "".join(f"  kernel.launch @{k}[%t](%t) : [index](index)\n" for k in kernels) + "  check.return\n}\n"


HELPER = "kernel.def export(\"test_copy_rows\") @test_copy_rows(%n: index) {\n} launch(%n: index) {\n}\n"
CASES = (HELPER + case("embed", "g_embed") + case("head_argmax", "g_quant_q8", "g_head_mv_q8")
         + case("flash_prefill_rows", "g_attention_fallback_list", "g_attention_flash2", "g_attention_flash2_partial")
         + case("nob3_mix", "g_attention_fallback_list", "g_attention_flash2_f16o", "g_attention_flash2_partial_f16o")
         + case("stale_kv_flash_nan", "g_attention_flash2", "test_copy_rows")
         + case("flataddr_eq", "g_attention_flash2_partial")
         + case("f16o11_eq_g_gemm_f16_swiglu_m256", "g_gemm_f16_swiglu_m256", "g_gemm_f16_swiglu_m256_f16o")
         + case("silu_copy", "kt_g_silu_mul")
         + case("vsplit_4109_8", "g_attention_split", "g_attention_vsplit", "g_attention_combine")
         + case("mv_q4_k", "g_mv_q4_k_k5120"))
CHANGED = ["g_attention_flash2", "g_attention_flash2_f16o", "g_attention_flash2_partial", "g_attention_flash2_partial_f16o",
           "g_gemm_f16_swiglu_m256_f16o"]


def rows(*pairs):
    return [{"kernel": k, "verdict": v, "allowed": False, "base": None, "cand": None, "fail": []} for k, v in pairs]


class Select(unittest.TestCase):
    def test_by_launched_kernel_not_name(self):   # E532 / q207: the names say 'flash', 'nob3', 'flataddr', 'f16o11', never 'flash2'
        text, names, smoke = ks.by_kernels(CASES, CHANGED)
        self.assertEqual(set(names) - set(smoke), {"flash_prefill_rows", "nob3_mix", "stale_kv_flash_nan", "flataddr_eq",
                                                   "f16o11_eq_g_gemm_f16_swiglu_m256"})
        self.assertNotIn("vsplit_4109_8", names)
        self.assertNotIn("mv_q4_k", names)

    def test_every_case_of_a_changed_kernel(self):   # completeness against an independent count of the cases launching each kernel
        _, names, _ = ks.by_kernels(CASES, CHANGED)
        for k in CHANGED:
            want = {b.split("@", 1)[1].split(" ", 1)[0] for b in CASES.split("check.case public ")[1:] if f"@{k}[" in b}
            self.assertTrue(want <= set(names), (k, want - set(names)))

    def test_smoke_and_helpers_kept(self):
        text, names, smoke = ks.by_kernels(CASES, [])
        self.assertEqual(smoke, ["embed", "head_argmax"])        # the smoke cases this model has (no attention_chain / gdn_chain here)
        self.assertEqual(names, smoke)
        self.assertIn('export("test_copy_rows")', text)

    def test_kt_copy_counts_as_its_kernel(self):
        _, names, _ = ks.by_kernels(CASES, ["g_silu_mul"])
        self.assertIn("silu_copy", names)
        self.assertEqual(ks.launches(case("x", "kt_g_silu_mul", "g_embed")), {"g_silu_mul", "g_embed"})


class Changed(unittest.TestCase):
    def write(self, obj):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        f.write(obj if isinstance(obj, str) else json.dumps(obj))
        f.close()
        return f.name

    def test_changed_and_new(self):
        p = self.write(rows(("a", "same"), ("b", "CHANGED"), ("c", "text-only"), ("d", "NEW"), ("e", "MISSING")))
        self.assertEqual(ks.changed_kernels(p), ["b", "d"])

    def test_nothing_changed_is_empty_not_unknown(self):   # the smoke set alone then runs
        self.assertEqual(ks.changed_kernels(self.write(rows(("a", "same"), ("b", "text-only")))), [])

    def test_unknown_fails_closed(self):
        for bad in (rows(("a", "CHANGED"), ("b", "ERROR")), rows(("a", "CHANGED"), ("b", "WEIRD")), [], {"a": 1},
                    [{"kernel": "a"}], "not json"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                ks.changed_kernels(self.write(bad))
        with self.assertRaises(ValueError):
            ks.changed_kernels("/nonexistent/bgate.json")

    def test_cli(self):
        ok = subprocess.run([sys.executable, os.path.join(HERE, "ktest_select.py"), "changed",
                             self.write(rows(("b", "CHANGED"), ("a", "NEW")))], capture_output=True, text=True)
        self.assertEqual((ok.returncode, ok.stdout.strip()), (0, "a,b"))
        bad = subprocess.run([sys.executable, os.path.join(HERE, "ktest_select.py"), "changed",
                              self.write(rows(("b", "ERROR")))], capture_output=True, text=True)
        self.assertEqual(bad.returncode, 1)
        self.assertIn("changed set unknown", bad.stderr)


if __name__ == "__main__":
    unittest.main()
