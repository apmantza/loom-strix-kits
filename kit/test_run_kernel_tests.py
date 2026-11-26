"""kit/run_kernel_tests.sh on a stub venv (the real script and ktest_select.py; a stub loom-test; no compiler, no GPU). Recurrences:
  - generated inputs outliving their run (11-26, HRX-like E625 follow-up: 89 GB of .npy sat in five test dirs and held page cache
    the flash2 runs needed): a failed run kept every input (ktest-stgmut7, ktest-q176), a compile failure or an interrupted run
    (timeout, scope stop) kept them too; now only the failing cases' inputs stay, and only on a failure;
  - a subset run reading as a full one (C11 / E532): the run log states the scope the generator wrote, and says 'unknown' when no
    scope record sits beside the cases (scripts/ktest_verdict.sh refuses anything but 'full' unless told);
  - kept inputs left in the page cache (KTEST_KEEP=1, or a failure's kept inputs): they are dropped after the run.
  - review r1 (11-26): a run with a section switched off (SPLIT_VZERO_TESTS=0, static-1124/vzero/tests.sh) labelled full; an untested
    producer call site; the known-defect FAIL branches' inputs untested; test scratch left in kit/ (now kit/.test_tmp/, gitignored).
usage: python3 -m unittest kit/test_run_kernel_tests.py   (needs a disk-backed kit dir for the residency test; tmpfs skips it)"""
import ast
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
# disk-backed scratch for these tests (tmpfs pages are not page cache that DONTNEED drops): kit/.test_tmp/, gitignored
TMP = os.path.join(HERE, ".test_tmp")


def mkdtemp(prefix):
    os.makedirs(TMP, exist_ok=True)
    return tempfile.mkdtemp(prefix=prefix, dir=TMP)
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
bad = mode.split(":", 1)[1].split(",") if mode.startswith(("fail:", "issue:")) else []
iss = mode.startswith("issue:")
print(json.dumps({"samples": [{"case": n, "passed": n not in bad, "issues": [{"message": "fault"}] if iss and n in bad else [],
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
        self.t = mkdtemp(".test_run_kernel_tests.")
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

    def test_known_defect_failures_keep_their_inputs(self):   # review r1 #5: both FAIL branches of a tagged (known-defect) case
        with open(os.path.join(self.dir, "known_defects.json"), "w") as f:
            json.dump({"b_one": "E444", "b_two": "E444"}, f)
        rc, out = self.run_tests(STUB_MODE="pass")   # tagged cases that pass: 'KNOWN-DEFECT NOW PASSES', counted bad
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL KNOWN-DEFECT NOW PASSES b_one", out)
        self.assertEqual(self.npy(), ["b2_x.npy", "b_x.npy", "shared_w.npy"], out)
        for n in FILES:   # regenerate
            with open(os.path.join(self.dir, n), "wb") as f:
                f.write(os.urandom(4096))
        with open(os.path.join(self.dir, "known_defects.json"), "w") as f:
            json.dump({"b_one": "E444"}, f)
        rc, out = self.run_tests(STUB_MODE="issue:b_one")   # a tagged case failing on an issue (a fault), not the known defect
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL b_one (known defect E444, but failed on an issue)", out)
        self.assertEqual(self.npy(), ["b_x.npy", "shared_w.npy"], out)

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
                          ({"scope": "subset", "why": ["KTEST_KERNELS=g_b"], "cases": 2, "generated": 3},
                           "ktest scope: subset (KTEST_KERNELS=g_b; 2 cases kept of 3 generated)"),
                          ({"scope": "subset", "why": ["FLASH_TESTS=0"], "cases": 3, "generated": 3},
                           "ktest scope: subset (FLASH_TESTS=0; 3 cases kept of 3 generated)"),
                          ({"scope": "subset", "cases": 3, "generated": 3}, "ktest scope: unknown"),   # a subset that names no reason
                          ({"scope": "subset", "why": [], "cases": 3, "generated": 3}, "ktest scope: unknown"),
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
        self.t = mkdtemp(".test_save_npy.")

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
        t = mkdtemp(".test_prune.")
        try:
            open(os.path.join(t, "a_x.npy"), "w").close()
            with self.assertRaises(ValueError):
                ks.prune_inputs(t, ["a_one"])
            self.assertEqual(os.listdir(t), ["a_x.npy"])
        finally:
            shutil.rmtree(t)

    def test_scope_record(self):   # review r1 #1: full only with every section switch at its default and no selection
        full = ks.scope_record({}, 5, 5)
        self.assertEqual(full, {"scope": "full", "cases": 5, "generated": 5})
        self.assertEqual(ks.scope_record({"FLASH_TESTS": "1", "GEMM_TESTS": "1", "L17_VSPLIT_TESTS": ""}, 5, 5)["scope"], "full")
        for env, why in (({"KTEST_KERNELS": "g_b,g_a"}, ["KTEST_KERNELS=g_a,g_b"]), ({"KTEST_KERNELS": ""}, ["KTEST_KERNELS="]),
                         ({"SPLIT_VZERO_TESTS": "0"}, ["SPLIT_VZERO_TESTS=0"]), ({"FLASH_TESTS": "yes"}, ["FLASH_TESTS=yes"]),
                         ({"L17_VSPLIT_TESTS": "1"}, ["L17_VSPLIT_TESTS=1"]), ({"KTEST_GEMM_ERR": "1"}, ["KTEST_GEMM_ERR=1"]),
                         ({"NEW_THING_TESTS": "1"}, ["NEW_THING_TESTS=1 (unknown section switch)"]),
                         ({"MOE_TESTS": "0", "WIDTH_EQ_TESTS": "0"}, ["MOE_TESTS=0", "WIDTH_EQ_TESTS=0"])):
            r = ks.scope_record(env, 2, 5)
            self.assertEqual((r["scope"], r.get("why")), ("subset", why), env)
        for sw in ks.SECTION_SWITCHES:   # each switch alone, away from its default
            self.assertEqual(ks.scope_record({sw: "0" if ks.SECTION_SWITCHES[sw] else "1"}, 1, 1)["scope"], "subset", sw)

    def test_switch_list_matches_the_generator(self):   # review r1 #1: the list is the generator's own env reads, not a hand list
        names = set()
        for f in ("make_kernel_tests.py", "ktest_more.py", "ktest_spec.py", "ktest_variants.py"):
            tree = ast.parse(open(os.path.join(HERE, f)).read())
            for n in ast.walk(tree):   # os.environ.get("X", ...), os.environ["X"], "X" in os.environ: names in code, not in prose
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get" and _is_environ(n.func.value):
                    k = n.args[0] if n.args else None
                elif isinstance(n, ast.Subscript) and _is_environ(n.value):
                    k = n.slice
                elif isinstance(n, ast.Compare) and any(_is_environ(c) for c in n.comparators):
                    k = n.left
                else:
                    continue
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    names.add(k.value)
                else:
                    self.fail(f"{f}:{n.lineno}: an environment read by a computed name (cannot be classified)")
        scoped = {k for k in names if k.endswith("_TESTS") or k.startswith("KTEST_")}
        self.assertLessEqual(scoped, set(ks.SECTION_SWITCHES) | set(ks.NOT_SECTION_SWITCHES), "classify every generator *_TESTS / KTEST_* read")
        self.assertLessEqual(set(ks.SECTION_SWITCHES), names, "a section switch the generator no longer reads")

    def test_generator_writes_the_scope_from_its_environment(self):   # review r1 #2: the producer call site
        tree = ast.parse(open(os.path.join(HERE, "make_kernel_tests.py")).read())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "write_scope"]
        self.assertEqual(len(calls), 1, "one write_scope call")
        c = calls[0]
        self.assertEqual([ast.unparse(a) for a in c.args[:1]], ["out"])
        self.assertEqual(len(c.args), 3, "write_scope(out, kept, generated): the scope comes from the environment, never an argument")
        self.assertEqual(c.keywords, [], "no env override at the call site")
        t = mkdtemp(".test_scope.")
        try:
            ks.write_scope(t, 2, 5, env={"KTEST_KERNELS": "g_b"})
            with open(os.path.join(t, ks.SCOPE_FILE)) as f:
                self.assertEqual(json.load(f), {"scope": "subset", "why": ["KTEST_KERNELS=g_b"], "cases": 2, "generated": 5})
            os.environ["FLASH_TESTS"] = "0"
            try:
                ks.write_scope(t, 5, 5)   # the default env is the process's
            finally:
                del os.environ["FLASH_TESTS"]
            self.assertIn("subset (FLASH_TESTS=0", ks.scope_line(t))
        finally:
            shutil.rmtree(t)


def _is_environ(n):
    return isinstance(n, ast.Attribute) and n.attr == "environ" and isinstance(n.value, ast.Name) and n.value.id == "os"


if __name__ == "__main__":
    unittest.main()
