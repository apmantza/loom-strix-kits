"""kit/run_kernel_tests.sh on a stub venv (the real script and ktest_select.py; a stub loom-test; no compiler, no GPU). Recurrence:
  - a subset run reading as a full one (C11 / E532): the run log states the scope the generator wrote, and says 'unknown' when no
    scope record sits beside the cases (HRX-like scripts/ktest_verdict.sh refuses anything but 'full' unless told).
usage: python3 -m unittest kit/test_run_kernel_tests.py"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ktest_select as ks  # noqa: E402

LIB = 'kernel.def export("g_a") @g_a(%n: index) {\n}\nkernel.def export("g_b") @g_b(%n: index) {\n}\n'


def case(name, kernel, *files):
    return (f"check.case public @{name} {{\n" + "".join(f'  %{i} = check.file.read.npy path("{f}") : tensor<4xf32>\n'
                                                         for i, f in enumerate(files))
            + f"  kernel.launch @{kernel}[%t](%t) : [index](index)\n  check.return\n}}\n")


CASES = case("a_one", "g_a", "a_x.npy", "shared_w.npy") + case("b_one", "g_b", "b_x.npy", "shared_w.npy") + case("b_two", "g_b", "b2_x.npy")
FILES = ["a_x.npy", "b_x.npy", "b2_x.npy", "shared_w.npy"]

# loom-test stand-in: reads every input (so they are page-cache resident, as the real runner leaves them), then per STUB_MODE
# writes results ('pass', 'fail:CASE'), writes no results ('nojson': a compile error), or hangs ('hang': killed by the test)
LOOM_TEST = r"""import json, os, re, sys, time
d = os.getcwd()
for f in sorted(os.listdir(d)):
    if f.endswith(".npy"):
        open(os.path.join(d, f), "rb").read()
names = re.findall(r"check\.case public @(\w+)", open(sys.argv[1]).read())
mode = os.environ.get("STUB_MODE", "pass")
if mode == "nojson":
    print("PARSE error", file=sys.stderr); sys.exit(1)
if mode == "hang":
    open(os.path.join(d, "hanging"), "w").write("1")
    time.sleep(60); sys.exit(1)
bad = mode.split(":", 1)[1].split(",") if mode.startswith("fail:") else []
print(json.dumps({"samples": [{"case": n, "passed": n not in bad, "issues": [],
                               "expectations": {"failures": [{"detail": "element 0"}] if n in bad else [], "failure_count": int(n in bad)}}
                              for n in names]}))
sys.exit(1 if bad else 0)
"""


class Runner(unittest.TestCase):
    def setUp(self):
        # disk-backed (the kit dir), never /tmp: tmpfs pages are not page cache that DONTNEED can drop
        self.t = tempfile.mkdtemp(prefix=".test_run_kernel_tests.", dir=HERE)
        t = self.t
        self.venv = os.path.join(t, "venv")
        os.makedirs(os.path.join(self.venv, "bin"))
        os.symlink(sys.executable, os.path.join(self.venv, "bin", "python"))
        lt = os.path.join(self.venv, "bin", "loom-test")
        with open(lt, "w") as f:
            f.write(f"#!{sys.executable}\n" + LOOM_TEST)
        os.chmod(lt, 0o755)
        self.pkg = os.path.join(t, "pkg")
        os.makedirs(os.path.join(self.pkg, "program"))
        with open(os.path.join(self.pkg, "program", "llm.loom"), "w") as f:
            f.write(LIB)
        self.dir = os.path.join(t, "tests")
        os.makedirs(self.dir)
        with open(os.path.join(self.dir, "cases.loom-test"), "w") as f:
            f.write(CASES)
        for n in FILES:
            with open(os.path.join(self.dir, n), "wb") as f:
                f.write(os.urandom(256 * 1024))
        self.env = dict(os.environ, VENV=self.venv, DEVICE_ARCH="gfx1151", HOME=t)
        self.env.pop("KTEST_KEEP", None)

    def tearDown(self):
        shutil.rmtree(self.t)

    def run_tests(self, **env):
        p = subprocess.run(["bash", os.path.join(HERE, "run_kernel_tests.sh"), self.pkg, self.dir], env=dict(self.env, **env),
                           capture_output=True, text=True, timeout=120)
        return p.returncode, p.stdout + p.stderr

    def npy(self):
        return sorted(f for f in os.listdir(self.dir) if f.endswith(".npy"))

    def scope(self, rec):
        with open(os.path.join(self.dir, ks.SCOPE_FILE), "w") as f:
            json.dump(rec, f)

    def test_pass_removes_inputs(self):
        rc, out = self.run_tests(STUB_MODE="pass")
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.npy(), [], out)

    def test_scope_lines(self):   # C11: the log names what ran; no record is 'unknown', never 'full'
        rc, out = self.run_tests(STUB_MODE="pass")
        self.assertIn("ktest scope: unknown", out)
        for rec, want in (({"scope": "full"}, "ktest scope: full"),
                          ({"scope": "subset", "kernels": ["g_b"], "cases": 2, "of": 3}, "ktest scope: subset (2 of 3 cases; kernels g_b)"),
                          ({"scope": "everything"}, "ktest scope: unknown"), ({"scope": "fullish"}, "ktest scope: unknown"),
                          ({"cases": 3, "of": 3}, "ktest scope: unknown")):
            self.scope(rec)
            rc, out = self.run_tests(STUB_MODE="pass", KTEST_KEEP="1")
            self.assertEqual(rc, 0, out)
            self.assertIn(want, out)



class Scope(unittest.TestCase):
    def test_write_scope(self):
        t = tempfile.mkdtemp(prefix=".test_scope.", dir=HERE)
        try:
            ks.write_scope(t, None, 5, 5)
            self.assertEqual(json.load(open(os.path.join(t, ks.SCOPE_FILE))), {"scope": "full", "cases": 5, "of": 5})
            ks.write_scope(t, {"g_b", "g_a"}, 2, 5)
            self.assertEqual(json.load(open(os.path.join(t, ks.SCOPE_FILE))),
                             {"scope": "subset", "kernels": ["g_a", "g_b"], "cases": 2, "of": 5})
        finally:
            shutil.rmtree(t)


if __name__ == "__main__":
    unittest.main()
