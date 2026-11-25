"""common/build.sh's post-build lint exit and verdict file, and common/ktest.sh's selected-mode fallbacks, on a stub kit tree (the real
scripts, stubbed kit/build.py, cand_lint.py, make_kernel_tests.py and run_kernel_tests.sh; no compiler, no GPU). Recurrences:
  - build.sh exiting 0 on a failed candidate lint (review r1 R1), or leaving no verdict a resume can check (F7: a rerun that skips
    the build because PKG/model.json exists forgot the rejection);
  - ktest.sh not rerunning the full suite when the generator refuses a selection (exit 3, review R2), or selecting with a gate file
    made for another package (F5).
usage: python3 -m unittest common/test_build_ktest.py"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

H = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.join(H, "..", "kit")

LINT_STUB = """import os, sys
m = os.environ.get("STUB_LINT")
print("CAND_LINT: " + ("PASS" if m in ("pass", "crash") else "FAIL"))
sys.exit(0 if m == "pass" else 1)
"""
BUILD_STUB = """import os, sys
os.makedirs(sys.argv[2], exist_ok=True); open(os.path.join(sys.argv[2], "model.json"), "w").write("{}")
print("serve with --memory_bytes=1")
"""
GEN_STUB = """import os, sys
sel = "KTEST_KERNELS" in os.environ
if sel and os.environ.get("STUB_REFUSE"):
    print("selected generation: stub refuses"); sys.exit(3)
os.makedirs(sys.argv[2], exist_ok=True)
open(os.path.join(sys.argv[2], "cases.loom-test"), "w").write("check.case public @a {\\n}\\n")
print("stub " + ("selected " + os.environ["KTEST_KERNELS"] if sel else "full") + " generation")
"""
RUN_STUB = """#!/bin/bash
echo "1/1 passed"
"""


def wr(path, text):
    with open(path, "w") as f:
        f.write(text)


class Stub(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.mkdtemp(prefix="test_build_ktest.")
        t = self.t
        os.makedirs(os.path.join(t, "common"))
        for f in ("build.sh", "ktest.sh", "env.sh"):
            shutil.copy(os.path.join(H, f), os.path.join(t, "common", f))
        wr(os.path.join(t, "common", "cand_lint.py"), LINT_STUB)
        os.makedirs(os.path.join(t, "kit"))
        wr(os.path.join(t, "kit", "build.py"), BUILD_STUB)
        wr(os.path.join(t, "kit", "make_kernel_tests.py"), GEN_STUB)
        wr(os.path.join(t, "kit", "run_kernel_tests.sh"), RUN_STUB)
        for f in ("ktest_select.py", "dual_target.py"):
            shutil.copy(os.path.join(KIT, f), os.path.join(t, "kit", f))
        os.makedirs(os.path.join(t, "kits", "k"))
        wr(os.path.join(t, "kits", "k", "kit.env"),
            'KIT_CONFIG=config.json\nKIT_TOKENIZER=tok\nKIT_RECIPE="A=1"\nKIT_BUILD_ARGS=""\n')
        wr(os.path.join(t, "kits", "k", "config.json"), "{}")
        self.work = os.path.join(t, "work")
        os.makedirs(os.path.join(self.work, "base"))
        wr(os.path.join(self.work, "base", "model.json"), "{}")
        wr(os.path.join(t, "env"), f"WORK={self.work}\nSTATE={self.work}/state\nLOOM_KB_ROOT={t}\nPYTHON={sys.executable}\n")
        self.env = dict(os.environ, LSK_ENV=os.path.join(t, "env"), HOME=t)
        for k in ("VENV", "KIT_SRC", "EXTRA", "LINT_BASE", "LINT_ALLOW", "LINT_ACCEPT", "KTEST_CHANGED", "KTEST_BASE", "WORK", "STATE"):
            self.env.pop(k, None)

    def tearDown(self):
        shutil.rmtree(self.t)

    def run_sh(self, script, *args, **env):
        p = subprocess.run(["bash", os.path.join(self.t, "common", script), *args], env=dict(self.env, **env),
                           capture_output=True, text=True)
        return p.returncode, p.stdout + p.stderr

    def verdict(self, pkg):
        with open(os.path.join(self.work, pkg, "cand_lint.verdict")) as f:
            return f.read()


class Build(Stub):
    def test_lint_failure_exits_3_with_a_verdict(self):   # R1 / F7
        rc, out = self.run_sh("build.sh", "k", "cand", LINT_BASE="base", STUB_LINT="fail")
        self.assertEqual(rc, 3, out)
        self.assertTrue(os.path.exists(os.path.join(self.work, "cand", "model.json")))   # complete, but rejected
        self.assertTrue(self.verdict("cand").startswith("CAND_LINT: FAIL"), self.verdict("cand"))

    def test_lint_exit_decides_not_its_text(self):   # a lint that printed PASS and then failed is not a pass
        rc, out = self.run_sh("build.sh", "k", "cand", LINT_BASE="base", STUB_LINT="crash")
        self.assertEqual(rc, 3, out)
        self.assertTrue(self.verdict("cand").startswith("CAND_LINT: FAIL"))

    def test_lint_pass(self):
        rc, out = self.run_sh("build.sh", "k", "cand", LINT_BASE="base", STUB_LINT="pass")
        self.assertEqual(rc, 0, out)
        self.assertTrue(self.verdict("cand").startswith("CAND_LINT: PASS"))

    def test_no_lint_base(self):
        rc, out = self.run_sh("build.sh", "k", "cand")
        self.assertEqual(rc, 0, out)
        self.assertTrue(self.verdict("cand").startswith("CAND_LINT: NOT RUN"))


class Ktest(Stub):
    def setUp(self):
        super().setUp()
        c = os.path.join(self.work, "cand")
        os.makedirs(c)
        os.makedirs(c + ".lint")
        wr(os.path.join(c, "model.json"), "{}")
        wr(os.path.join(c, "llm_config.json"), "{}")
        wr(c + ".cmd", "env A=1 kit/build.py kits/k/config.json cand --tokenizer tok\n")
        wr(os.path.join(c + ".lint", "bgate.json"), '[{"kernel": "g_x", "verdict": "CHANGED"}]')
        self.ident(c)

    def ident(self, cand, base=None):
        import json
        wr(os.path.join(self.work, "cand.lint", "verdict.json"),
           json.dumps({"base": base or os.path.join(self.work, "base"), "cand": cand}))

    def gen(self, t="t"):
        with open(os.path.join(self.work, t + ".gen.log")) as f:
            return f.read()

    def test_selected(self):
        rc, out = self.run_sh("ktest.sh", "k", "cand", "t", KTEST_CHANGED="cand.lint/bgate.json")
        self.assertEqual(rc, 0, out)
        self.assertIn("ktest: selected", out)
        self.assertIn("stub selected g_x", self.gen())

    def test_refused_selection_runs_the_full_suite(self):   # R2
        rc, out = self.run_sh("ktest.sh", "k", "cand", "t", KTEST_CHANGED="cand.lint/bgate.json", STUB_REFUSE="1")
        self.assertEqual(rc, 0, out)
        self.assertIn("full suite", out)
        self.assertIn("stub full generation", self.gen())

    def test_gate_file_of_another_package(self):   # F5
        self.ident(os.path.join(self.work, "other"))
        rc, out = self.run_sh("ktest.sh", "k", "cand", "t", KTEST_CHANGED="cand.lint/bgate.json")
        self.assertIn("changed set unknown: full suite", out)
        self.assertIn("stub full generation", self.gen())

    def test_gate_file_of_another_base(self):   # F5: KTEST_BASE names the base the gate must have used
        rc, out = self.run_sh("ktest.sh", "k", "cand", "t", KTEST_CHANGED="cand.lint/bgate.json", KTEST_BASE="cand")
        self.assertIn("changed set unknown: full suite", out)


if __name__ == "__main__":
    unittest.main()
