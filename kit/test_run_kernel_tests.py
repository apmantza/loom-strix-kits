"""kit/run_kernel_tests.sh on a stub venv (the real script and ktest_select.py; a stub loom-test; no compiler, no GPU). Recurrences:
  - generated inputs outliving their run (11-26, HRX-like E625 follow-up: 89 GB of .npy sat in five test dirs and held page cache
    the flash2 runs needed): a failed run kept every input (ktest-stgmut7, ktest-q176), a compile failure or an interrupted run
    (timeout, scope stop) kept them too; now only the failing cases' inputs stay, and only on a failure;
  - a subset run reading as a full one (C11 / E532): the run log states the scope the generator wrote, and says 'unknown' when no
    scope record sits beside the cases (scripts/ktest_verdict.sh refuses anything but 'full' unless told);
  - kept inputs left in the page cache (KTEST_KEEP=1, or a failure's kept inputs): they are dropped after the run.
usage: python3 -m unittest kit/test_run_kernel_tests.py   (needs a disk-backed kit dir for the residency test; tmpfs skips it)"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
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


def fs_type(path):
    return subprocess.run(["stat", "-f", "-c", "%T", path], capture_output=True, text=True).stdout.strip()


def resident(path):
    """pages of PATH in the page cache, (resident, total): mmap + mincore through ctypes (no fincore / vmtouch on this box)"""
    import ctypes
    import mmap
    libc = ctypes.CDLL(None, use_errno=True)
    size = os.path.getsize(path)
    if not size:
        return 0, 0
    page = mmap.PAGESIZE
    n = (size + page - 1) // page
    fd = os.open(path, os.O_RDONLY)
    try:
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), "mmap")
        vec = (ctypes.c_ubyte * n)()
        libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        if libc.mincore(ctypes.c_void_p(addr), size, vec) != 0:
            raise OSError(ctypes.get_errno(), "mincore")
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.munmap(ctypes.c_void_p(addr), size)
        return sum(b & 1 for b in vec), n
    finally:
        os.close(fd)


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

    def test_failure_keeps_only_the_failing_cases_inputs(self):   # ktest-stgmut7 / ktest-q176 kept every input on a failure
        rc, out = self.run_tests(STUB_MODE="fail:b_one")
        self.assertEqual(rc, 1, out)
        self.assertEqual(self.npy(), ["b_x.npy", "shared_w.npy"], out)
        self.assertIn("kept 2 input files of the failing cases", out)

    def test_compile_failure_removes_inputs(self):   # no results: nothing names a failing case, the inputs are regenerable
        rc, out = self.run_tests(STUB_MODE="nojson")
        self.assertEqual(rc, 1, out)
        self.assertEqual(self.npy(), [], out)

    def test_keep_keeps_everything(self):
        rc, out = self.run_tests(STUB_MODE="fail:b_one", KTEST_KEEP="1")
        self.assertEqual(rc, 1, out)
        self.assertEqual(self.npy(), sorted(FILES), out)

    def test_interrupt_removes_inputs(self):   # a timeout or a scope stop is a transition: the same cleanup as a failure
        p = subprocess.Popen(["bash", os.path.join(HERE, "run_kernel_tests.sh"), self.pkg, self.dir], env=dict(self.env, STUB_MODE="hang"),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        for _ in range(300):
            if os.path.exists(os.path.join(self.dir, "hanging")):
                break
            time.sleep(0.1)
        else:
            p.kill()
            self.fail("the stub loom-test never started: " + p.communicate()[0])
        os.killpg(p.pid, signal.SIGTERM)
        out = p.communicate(timeout=60)[0]
        self.assertEqual(p.returncode, 143, out)
        self.assertIn("inputs removed (exit 143", out)
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

    def test_kept_inputs_dropped_from_cache(self):
        if fs_type(self.dir) in ("tmpfs", "ramfs"):
            self.skipTest(f"{self.dir} is {fs_type(self.dir)}: no page cache to drop")
        rc, out = self.run_tests(STUB_MODE="pass", KTEST_KEEP="1")   # the stub read every input: resident before the cleanup
        self.assertEqual(rc, 0, out)
        res = {n: resident(os.path.join(self.dir, n)) for n in FILES}
        self.assertTrue(all(r == 0 for r, _ in res.values()), res)
        rc, out = self.run_tests(STUB_MODE="fail:b_one")   # a failure's kept inputs too
        res = {n: resident(os.path.join(self.dir, n)) for n in self.npy()}
        self.assertEqual(sorted(res), ["b_x.npy", "shared_w.npy"], out)
        self.assertTrue(all(r == 0 for r, _ in res.values()), res)


class SaveNpy(unittest.TestCase):   # make_kernel_tests.save: every generated input leaves the page cache once written
    def setUp(self):
        self.t = tempfile.mkdtemp(prefix=".test_save_npy.", dir=HERE)

    def tearDown(self):
        shutil.rmtree(self.t)

    def test_written_file_not_resident(self):
        if fs_type(self.t) in ("tmpfs", "ramfs"):
            self.skipTest(f"{self.t} is {fs_type(self.t)}")
        import numpy as np
        a = np.arange(4 << 20, dtype=np.float32)   # 16 MiB
        p = os.path.join(self.t, "x.npy")
        ks.save_npy(p, a)
        r, n = resident(p)
        self.assertEqual(r, 0, f"{r} of {n} pages resident after save_npy")
        self.assertTrue((np.load(p) == a).all())
        # the probe itself: a plain np.save leaves the file resident (else this test could not see a missing drop)
        q = os.path.join(self.t, "y.npy")
        np.save(q, a)
        self.assertGreater(resident(q)[0], 0)


class Prune(unittest.TestCase):
    def test_inputs_of(self):
        self.assertEqual(ks.inputs_of(CASES, {"b_one"}), {"b_x.npy", "shared_w.npy"})
        self.assertEqual(ks.inputs_of(CASES, {"a_one", "b_two"}), {"a_x.npy", "shared_w.npy", "b2_x.npy"})
        self.assertEqual(ks.inputs_of(CASES, set()), set())

    def test_prune_without_cases_refuses(self):   # cannot tell which inputs the failing cases read: the runner then removes all
        t = tempfile.mkdtemp(prefix=".test_prune.", dir=HERE)
        try:
            open(os.path.join(t, "a_x.npy"), "w").close()
            with self.assertRaises(ValueError):
                ks.prune_inputs(t, ["a_one"])
            self.assertEqual(os.listdir(t), ["a_x.npy"])
        finally:
            shutil.rmtree(t)

    def test_write_scope(self):
        t = tempfile.mkdtemp(prefix=".test_scope.", dir=HERE)
        try:
            ks.write_scope(t, None, 5, 5)
            with open(os.path.join(t, ks.SCOPE_FILE)) as f:
                self.assertEqual(json.load(f), {"scope": "full", "cases": 5, "of": 5})
            ks.write_scope(t, {"g_b", "g_a"}, 2, 5)
            with open(os.path.join(t, ks.SCOPE_FILE)) as f:
                self.assertEqual(json.load(f), {"scope": "subset", "kernels": ["g_a", "g_b"], "cases": 2, "of": 5})
        finally:
            shutil.rmtree(t)


if __name__ == "__main__":
    unittest.main()
