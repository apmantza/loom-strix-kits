#!/bin/bash
# usage: VENV=~/loom/.venv run_kernel_tests.sh PACKAGE TESTDIR   (TESTDIR from make_kernel_tests.py)
pkg=$(cd "$1" && pwd); dir=$2; venv=${VENV:-$HOME/loom/.venv}
here=$(cd "$(dirname "$0")" && pwd)   # before the cd below: a relative $0 would point into TESTDIR (HRX-like retro 11-19)
bin=$venv/bin; [ -d "$bin" ] || bin=$venv/Scripts   # Windows venvs use Scripts/
cd "$dir" || exit 1
# Inputs never outlive their run unless asked (11-26, HRX-like E625: 89 GB of .npy left in five test dirs held page cache and
# fragmented host memory): on every exit path (pass, fail, compile error, select refusal, INT / TERM / HUP from a timeout or a scope
# stop) the .npy inputs go, except KTEST_KEEP=1 (all kept) and a run with failing cases (only the inputs those cases read stay, for the
# diagnosis; ktest_select.py prune). They are regenerable (make_kernel_tests.py). A SIGKILL (a MemoryMax OOM kill) runs no trap.
rm -f failed_cases.txt
cleanup() {
  local rc=$1
  trap - EXIT INT TERM HUP
  if [ "${KTEST_KEEP:-0}" = 1 ]; then
    "$bin/python" "$here/ktest_select.py" drop . || echo "inputs kept (KTEST_KEEP=1): $PWD (page-cache drop failed)"
  elif [ "$rc" -eq 0 ]; then
    find . -maxdepth 1 \( -name '*.npy' -o -name combined.loom \) \( -type f -o -type l \) -delete && echo "inputs removed (all passed; KTEST_KEEP=1 keeps them)"
  elif [ -s failed_cases.txt ] && "$bin/python" "$here/ktest_select.py" prune . failed_cases.txt; then
    :
  else
    find . -maxdepth 1 -name '*.npy' \( -type f -o -type l \) -delete
    echo "inputs removed (exit $rc with no failing case recorded: a compile error, a refusal or an interrupt; regenerate with make_kernel_tests.py, or KTEST_KEEP=1)"
  fi
  exit "$rc"
}
trap 'cleanup $?' EXIT
trap 'exit 129' HUP; trap 'exit 130' INT; trap 'exit 143' TERM
# The device's architecture (DEVICE_ARCH, else rocminfo / amd-smi): on gfx12 the cases launch the K_rdna4 twin of every
# kernel that has one and skip cases of gfx11-only kernels. Keep only cases whose kernels the generator emitted (mixed
# tensor formats or a changed generator can drop a variant a case launches: that was a SYMBOL error for the whole file).
arch=${DEVICE_ARCH:-$( (rocminfo 2>/dev/null | grep -o -m1 'gfx[1-9][0-9a-f]*' | head -1) || true)}
[ -n "$arch" ] || arch=$( (amd-smi static --asic 2>/dev/null | grep -o -m1 'gfx[1-9][0-9a-f]*' | head -1) || true)
echo "device arch: ${arch:-unknown}"
"$bin/python" "$here/ktest_select.py" "$pkg/program/llm.loom" cases.loom-test cases_used.loom-test "$arch" || { echo "FAIL select: ktest_select.py failed, no cases run"; exit 1; }
# One file (the cases' reference kernels sit past the package's ~160k lines: hrx-loom's 16-bit encoding ids, HRX-like E609 / E616).
"$bin/python" "$here/ktest_select.py" combine "$pkg/program/llm.loom" cases_used.loom-test combined.loom \
  || { echo "FAIL encodings: ktest_select.py combine refused combined.loom, no cases run"; exit 1; }
"$bin/python" "$here/ktest_select.py" scope .   # full / subset / unknown (C11): HRX-like ktest_verdict.sh passes only full by default
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
import os
# Known defects (make_kernel_tests.py writes known_defects.json: {case: ledger id}, e.g. HRX-like E444's split-kernel stale-V cases
# on a package built without the fix's switch). Such a case may fail on an expectation: 'KNOWN-FAIL (id) case', not counted bad.
# If it PASSES, the tag has outlived the defect: 'KNOWN-DEFECT NOW PASSES', counted bad. A failure only on an issue (a fault, a
# timeout) is not the known defect: FAIL. Negative controls keep their must-fail rule whatever the file says.
kd = json.load(open("known_defects.json")) if os.path.exists("known_defects.json") else {}
bad = nneg = nknown = 0
failed = []   # the cases counted bad: their inputs survive the cleanup (failed_cases.txt)
for s in d["samples"]:
    fails = [f.get("detail", f.get("message", "")) if isinstance(f, dict) else str(f) for f in s["expectations"].get("failures", [])]
    issues = [i.get("message", "") for i in s.get("issues", [])]
    if s["case"] in kd and "_negcontrol" not in s["case"]:
        if s["passed"]:
            bad += 1
            failed.append(s["case"])
            print(f"FAIL KNOWN-DEFECT NOW PASSES {s['case']}: remove the tag ({kd[s['case']]})")
        elif (fails or s["expectations"].get("failure_count", 0) > 0) and not issues:
            nknown += 1
            print(f"KNOWN-FAIL ({kd[s['case']]}) {s['case']}", *(x[:300] for x in fails[:2]))
        else:
            bad += 1
            failed.append(s["case"])
            print(f"FAIL {s['case']} (known defect {kd[s['case']]}, but failed on an issue)", *(x[:300] for x in issues + fails[:2]))
        continue
    # A negative control (any case named *_negcontrol: a planted difference its checks must see) must FAIL, and on an
    # expectation: one that fails only on an issue (a fault, a timeout) proves nothing about its checks (HRX-like E398).
    neg = "_negcontrol" in s["case"]
    nneg += neg
    ok = (not s["passed"] and (bool(fails) or s["expectations"].get("failure_count", 0) > 0) and not issues) if neg else s["passed"]
    bad += not ok
    failed += [] if ok else [s["case"]]
    why = ("" if not neg else " (negative control failed on an expectation, as it must)" if ok else
           " (negative control failed only on an issue: its checks are unproven)" if not s["passed"] else
           " (negative control passed: its checks cannot see the planted difference)")
    print(("PASS " if ok else "FAIL ") + s["case"] + why, *(x[:300] for x in issues + fails[:2]))
open("failed_cases.txt", "w").write("".join(c + "\n" for c in failed))
print(f"negative controls: {nneg}")
print(f"known defects (KNOWN-FAIL): {nknown}")
print(f"{len(d['samples']) - bad}/{len(d['samples'])} passed")
sys.exit(1 if bad else 0)
PY
rc=$?
exit $rc
