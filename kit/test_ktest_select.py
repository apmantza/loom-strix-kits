#!/usr/bin/env python3
"""kit/ktest_select.py's selected generation pieces (common/ktest.sh KTEST_CHANGED): the changed set read from binary_gate's rows and
the case filter by launched kernel. Recurrences it prevents:
  - a changed kernel's dedicated tests dropped because they launch a ktn_ / ktm_plane_ / ktm_wrap_ copy (review r1 F1: E444's four
    stale_kv_split_listed cases, FLATADDR's two negative controls); a launched name nothing places keeps its case;
  - a selection accepted while one selected kernel has no case (review F4: each, not any); the declaration check of a gated
    section (R4); a grid-only (text-only) change not selected (F3); a gate file made for another package (F5);
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
         + case("mv_q4_k", "g_mv_q4_k_k5120")
         # test copies other than kt_ (review r1 F1): E444's stale-KV twins launch ktn_ copies of the listed split kernels;
         # FLATADDR's negative controls launch ktm_plane_ / ktm_wrap_ mutants beside kto_ references; kt_fa_poke is a helper
         + case("stale_kv_split_listed_nan", "ktn_g16_attention_split_listed", "ktn_g16_attention_combine_listed", "test_copy_rows")
         + case("flataddr_eq_plane_negcontrol", "g_attention_fallback_list", "kt_fa_poke", "ktm_plane_g_attention_flash2",
                "kto_g_attention_flash2")
         + case("flataddr_eq_wrap_negcontrol", "g_attention_fallback_list", "ktm_wrap_g_attention_flash2_partial")
         + case("attention_chain_listed", "g16_attention_split_listed", "g16_attention_combine_listed"))
PKG = {"g_embed", "g_quant_q8", "g_head_mv_q8", "g_attention_fallback_list", "g_attention_split", "g_attention_vsplit",
       "g_attention_combine", "g_mv_q4_k_k5120", "g_silu_mul", "g_gemm_f16_swiglu_m256", "g16_attention_split_listed",
       "g16_attention_combine_listed"} | set(CHANGED_ := ["g_attention_flash2", "g_attention_flash2_f16o", "g_attention_flash2_partial",
                                                         "g_attention_flash2_partial_f16o", "g_gemm_f16_swiglu_m256_f16o"])
CHANGED = CHANGED_
# the cases that test each kernel, written out by hand from the fixture above (independent of ktest_select's name mapping)
TESTS = {
    "g_attention_flash2": {"flash_prefill_rows", "stale_kv_flash_nan", "flataddr_eq_plane_negcontrol"},
    "g_attention_flash2_f16o": {"nob3_mix"},
    "g_attention_flash2_partial": {"flash_prefill_rows", "flataddr_eq", "flataddr_eq_wrap_negcontrol"},
    "g_attention_flash2_partial_f16o": {"nob3_mix"},
    "g_gemm_f16_swiglu_m256_f16o": {"f16o11_eq_g_gemm_f16_swiglu_m256"},
    "g16_attention_split_listed": {"stale_kv_split_listed_nan", "attention_chain_listed"},
    "g16_attention_combine_listed": {"stale_kv_split_listed_nan", "attention_chain_listed"},
    "g_silu_mul": {"silu_copy"},
}


def rows(*pairs):
    return [{"kernel": k, "verdict": v, "allowed": False, "base": None, "cand": None, "fail": []} for k, v in pairs]


class Select(unittest.TestCase):
    def test_by_launched_kernel_not_name(self):   # E532 / q207: the names say 'flash', 'nob3', 'flataddr', 'f16o11', never 'flash2'
        text, names, smoke, info = ks.by_kernels(CASES, CHANGED, defined=PKG)
        self.assertEqual(set(names) - set(smoke), set().union(*(TESTS[k] for k in CHANGED)))
        self.assertNotIn("vsplit_4109_8", names)
        self.assertNotIn("mv_q4_k", names)

    def test_every_case_of_a_changed_kernel(self):   # against the hand-written TESTS, each kernel alone
        for k, want in TESTS.items():
            _, names, _, info = ks.by_kernels(CASES, [k], defined=PKG)
            self.assertTrue(want <= set(names), (k, want - set(names)))
            self.assertIn(k, info["covered"])

    def test_copies_of_every_prefix(self):   # review r1 F1: ktn_ (E444's four stale_kv_split_listed cases), ktm_plane_ / ktm_wrap_
        for pfx in ks.COPY_PREFIXES:
            self.assertEqual(ks.kernel_of(pfx + "g_attention_flash2", PKG), ("copy", "g_attention_flash2"), pfx)
        self.assertEqual(ks.kernel_of("ktm_plane_g_attention_flash2_partial", PKG), ("copy", "g_attention_flash2_partial"))
        self.assertEqual(ks.kernel_of("ktz_newtag_g_attention_flash2", PKG), ("copy", "g_attention_flash2"))   # a prefix nobody listed
        self.assertEqual(ks.kernel_of("kt_fa_poke", PKG), ("helper", None))
        # verify r2 V8: when one package kernel's name is a '_'-suffix of another's, a copy maps to the longest
        pair = PKG | {"attention_flash2_partial"}
        self.assertEqual(ks.kernel_of("ktn_g_attention_flash2_partial", pair), ("copy", "g_attention_flash2_partial"))
        self.assertEqual(ks.kernel_of("ktn_x_attention_flash2_partial", pair), ("copy", "attention_flash2_partial"))
        self.assertEqual(ks.kernel_of("g_attention_flash2", PKG), ("pkg", "g_attention_flash2"))

    def test_unknown_launch_kept(self):   # fail closed: a launched name nothing places keeps its case
        extra = case("weird", "some_new_kernel_copy")
        _, names, _, info = ks.by_kernels(CASES + extra, ["g_embed"], defined=PKG)
        self.assertIn("weird", names)
        self.assertEqual(info["unknown"], {"weird": ["some_new_kernel_copy"]})

    def test_smoke_and_helpers_kept(self):
        text, names, smoke, _ = ks.by_kernels(CASES, [], defined=PKG)
        self.assertEqual(smoke, ["embed", "head_argmax"])        # the smoke cases this model has (no attention_chain / gdn_chain here)
        self.assertEqual(names, smoke)
        self.assertIn('export("test_copy_rows")', text)

    def test_helper_after_an_unselected_case_kept(self):   # E613 / q213: test_copy_rows sat after flash_prefill's case and left
        # with it, so a selected weq_ case launched a helper the build no longer defined (SKIP: kernel(s) not in this build)
        cases = case("flash_prefill", "g_attention_flash2") + HELPER + case("weq_mv_q4_k", "g_mv_q4_k_k5120", "test_copy_rows")
        text, names, _, _ = ks.by_kernels(cases, ["g_mv_q4_k_k5120"], defined=PKG)
        self.assertEqual(names, ["weq_mv_q4_k"])
        self.assertIn('export("test_copy_rows")', text)
        self.assertNotIn("@flash_prefill ", text)

    def test_select_keeps_a_helper_after_a_skipped_case(self):   # E613, the run-time filter: a case skipped for a kernel this
        # build lacks took the helper glued after it, and the next case was then skipped too
        lib = "kernel.def export(\"g_embed\") @g_embed(%t: index) {\n}\n"
        cases = case("gone", "g_not_built") + HELPER + case("uses_helper", "g_embed", "test_copy_rows")
        text, skipped = ks.select(lib, cases, arch="gfx1151", rdna4=False)
        self.assertIn('export("test_copy_rows")', text)
        self.assertIn("@uses_helper ", text)
        self.assertEqual([x.split(":")[0] for x in skipped], ["SKIP gone"])

    def test_kt_copy_counts_as_its_kernel(self):
        _, names, _, _ = ks.by_kernels(CASES, ["g_silu_mul"], defined=PKG)
        self.assertIn("silu_copy", names)
        self.assertEqual(ks.launches(case("x", "kt_g_silu_mul", "g_embed")), {"g_silu_mul", "g_embed"})


class Finish(unittest.TestCase):
    def quiet(self, *a, **k):
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            return ks.finish_selection(*a, **k)

    def test_each_selected_kernel_needs_a_case(self):   # review r1 F4 (R3): each, not any
        with self.assertRaises(SystemExit) as e:
            self.quiet(CASES, ["g_attention_flash2", "g_no_such_kernel"], PKG)
        self.assertEqual(e.exception.code, 3)
        text, lines = self.quiet(CASES, ["g_attention_flash2"], PKG)
        self.assertIn("@flataddr_eq_plane_negcontrol", text)

    def test_no_smoke_refused(self):
        with self.assertRaises(SystemExit) as e:
            self.quiet(CASES.replace("@embed {", "@embedx {").replace("@head_argmax {", "@headx {"), ["g_attention_flash2"], PKG)
        self.assertEqual(e.exception.code, 3)

    def test_child_never_refuses(self):   # a KTEST_HEADS_SFX child: the parent filters its cases again
        text, _ = self.quiet(CASES, ["g_no_such_kernel"], PKG, child=True)
        self.assertIn("@embed", text)

    def test_declaration_check(self):   # R4: a gated section whose cases test an undeclared kernel stops the generation
        new = [case("vsplit", "g_attention_split", "g_attention_vsplit", "g_attention_combine", "test_copy_rows")]
        ks.check_declared(new, ["g_attention_split", "g_attention_vsplit", "g_attention_combine"], PKG)
        with self.assertRaises(SystemExit) as e:
            ks.check_declared(new, ["g_attention_split", "g_attention_vsplit"], PKG)
        self.assertIn("g_attention_combine", str(e.exception.code))
        with self.assertRaises(SystemExit):   # a copy counts as its kernel
            ks.check_declared([case("c", "ktn_g_attention_combine")], ["g_attention_split"], PKG)


class Changed(unittest.TestCase):
    def write(self, obj):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        f.write(obj if isinstance(obj, str) else json.dumps(obj))
        f.close()
        return f.name

    def test_changed_new_and_text_only(self):   # review r1 F3: a grid-only change is text-only and is selected
        p = self.write(rows(("a", "same"), ("b", "CHANGED"), ("c", "text-only"), ("d", "NEW"), ("e", "MISSING")))
        self.assertEqual(ks.changed_kernels(p), ["b", "c", "d"])

    def test_identity(self):   # review r1 F5: the rows must have been made for the package under test (and the named base)
        d = tempfile.mkdtemp()
        p = os.path.join(d, "bgate.json")
        json.dump(rows(("b", "CHANGED")), open(p, "w"))
        with self.assertRaises(ValueError):
            ks.changed_kernels(p, cand="/w/pkg-x")                        # no verdict.json: identity unknown
        x, y = os.path.join(d, "pkg-x"), os.path.join(d, "pkg-y")
        for q in (x, y):
            with open(q + ".cmd", "w") as fh:
                fh.write("env A=1 kit/build.py c " + q + "\n")
        with open(os.path.join(d, "verdict.json"), "w") as fh:
            json.dump({"base": "/w/pkg-e480", "cand": x, "cand_cmd_sha": ks.cmd_sha(x)}, fh)
        self.assertEqual(ks.changed_kernels(p, cand=x, base="/w/pkg-e480"), ["b"])
        with open(os.path.join(d, "verdict.json"), "w") as fh:          # a gate dir written before the hash existed (r2)
            json.dump({"base": "/w/pkg-e480", "cand": x}, fh)
        with self.assertRaises(ValueError):
            ks.changed_kernels(p, cand=x)
        with open(os.path.join(d, "verdict.json"), "w") as fh:
            json.dump({"base": "/w/pkg-e480", "cand": x, "cand_cmd_sha": ks.cmd_sha(x)}, fh)
        with self.assertRaises(ValueError):
            ks.changed_kernels(p, cand=y)
        with self.assertRaises(ValueError):
            ks.changed_kernels(p, cand=x, base="/w/pkg-other")
        with open(x + ".cmd", "a") as fh:                              # verify r2: PKG rebuilt with other switches
            fh.write("env B=1 kit/build.py c " + x + "\n")
        with self.assertRaises(ValueError):
            ks.changed_kernels(p, cand=x)
        os.remove(x + ".cmd")
        with self.assertRaises(ValueError):                             # no .cmd: identity unknown
            ks.changed_kernels(p, cand=x)

    def test_nothing_changed_is_empty_not_unknown(self):   # the smoke set alone then runs
        self.assertEqual(ks.changed_kernels(self.write(rows(("a", "same"), ("b", "MISSING")))), [])

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
