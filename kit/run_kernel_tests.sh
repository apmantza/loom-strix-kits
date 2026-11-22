#!/bin/bash
# usage: VENV=~/loom/.venv run_kernel_tests.sh PACKAGE TESTDIR   (TESTDIR from make_kernel_tests.py)
pkg=$(cd "$1" && pwd); dir=$2; venv=${VENV:-$HOME/loom/.venv}
here=$(cd "$(dirname "$0")" && pwd)   # before the cd below: a relative $0 would point into TESTDIR (HRX-like retro 11-19)
bin=$venv/bin; [ -d "$bin" ] || bin=$venv/Scripts   # Windows venvs use Scripts/
cd "$dir" || exit 1
# The device's architecture (DEVICE_ARCH, else rocminfo / amd-smi): on gfx12 the cases launch the K_rdna4 twin of every
# kernel that has one and skip cases of gfx11-only kernels. Keep only cases whose kernels the generator emitted (mixed
# tensor formats or a changed generator can drop a variant a case launches: that was a SYMBOL error for the whole file).
arch=${DEVICE_ARCH:-$( (rocminfo 2>/dev/null | grep -o -m1 'gfx[1-9][0-9a-f]*' | head -1) || true)}
[ -n "$arch" ] || arch=$( (amd-smi static --asic 2>/dev/null | grep -o -m1 'gfx[1-9][0-9a-f]*' | head -1) || true)
echo "device arch: ${arch:-unknown}"
"$bin/python" "$here/ktest_select.py" "$pkg/program/llm.loom" cases.loom-test cases_used.loom-test "$arch" || { echo "FAIL select: ktest_select.py failed, no cases run"; exit 1; }
cat "$pkg/program/llm.loom" cases_used.loom-test > combined.loom
"$bin/loom-test" combined.loom --device=${DEVICE:-amdgpu} > out.json 2> err.txt
grep -v '^{"format' err.txt | head -20
"$bin/python" - <<'PY'
import json, sys
try:
    d = json.load(open("out.json"))
except ValueError:  # loom-test did not get to run the cases: a compile/link error (PARSE, TYPE, SYMBOL...)
    err = [l for l in open("err.txt", errors="replace") if not l.startswith('{"format')]
    out = open("out.json", errors="replace").read()
    print("FAIL compile: loom-test produced no results. Diagnostics (err.txt / out.json):")
    print("".join(err[-30:]) or out[-3000:])
    print("(a case launching a kernel the generator no longer emits shows up as a SYMBOL error: regenerate the tests "
          "with make_kernel_tests.py after changing the generator)")
    sys.exit(1)
bad = 0
for s in d["samples"]:
    fails = [f.get("detail", f.get("message", "")) if isinstance(f, dict) else str(f) for f in s["expectations"].get("failures", [])]
    issues = [i.get("message", "") for i in s.get("issues", [])]
    neg = "_negcontrol" in s["case"]   # a negative control must FAIL (it feeds the kernel data in the wrong layout)
    ok = (not s["passed"]) if neg else s["passed"]
    bad += not ok
    print(("PASS " if ok else "FAIL ") + s["case"] + (" (negative control failed, as it must)" if neg and ok else " (negative control passed: the test cannot see a layout error)" if neg else ""), *(x[:300] for x in issues + fails[:2]))
print(f"{len(d['samples']) - bad}/{len(d['samples'])} passed")
sys.exit(1 if bad else 0)
PY
rc=$?
# A passing run's inputs are regenerable (make_kernel_tests.py, ~4 min) and ~15 GB per package: 25 test dirs held 541 GB on
# 11-22 (HRX-like retro) and filled the disk. Keep them when anything failed (the diagnosis needs them); KTEST_KEEP=1 keeps them always.
if [ $rc -eq 0 ] && [ "${KTEST_KEEP:-0}" != 1 ]; then
  find . -maxdepth 1 \( -name '*.npy' -o -name combined.loom \) \( -type f -o -type l \) -delete && echo "inputs removed (all passed; KTEST_KEEP=1 keeps them)"
fi
exit $rc
