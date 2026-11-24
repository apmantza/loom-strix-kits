#!/bin/bash
# Kernel tests of a package: generate the cases for exactly the switches the package was built with (read from PKG.cmd,
# which build.sh writes; a test generated for other switches tests other kernels), then run them on the GPU.
# Correctness only; safe beside a running server when memory allows. usage: ktest.sh KIT PKG TESTDIR
set -u -o pipefail
. "$(dirname "$0")/env.sh"; lsk_load "${1:?usage: ktest.sh KIT PKG TESTDIR}"
P=${2:?package dir}; T=${3:?test dir}
# KTEST_BASE=PKG (optional): judge the delta (common/ktest_delta.py). A red candidate runs the base package's tests from the same
# kit tree; a case red on both is INHERITED, not a stop (E445). Unset: every FAIL stops, as before.
run_one() {   # PKG_DIR TESTDIR -> TESTDIR.run.log; the switches come from PKG_DIR.cmd
  local P=$1 T=$2 p SW; p=$(lsk_pkg "$P")
  [ -f "$P.cmd" ] || die "no $P.cmd (build with common/build.sh) - the test switches must be the package's"
  SW=$(sed -n 's/^env \(.*\) kit\/build\.py .*/\1/p' "$P.cmd"); [ -n "$SW" ] || die "cannot read the switches from $P.cmd"
  rm -rf "$T"
  # shellcheck disable=SC2086
  env $SW PATH="${VENV:+$VENV/bin:}$PATH" "$PYTHON" "$KIT_SRC/make_kernel_tests.py" "$p/llm_config.json" "$T" > "$T.gen.log" 2>&1 \
    || die "test generation failed ($WORK/$T.gen.log)"
  VENV=${VENV:-} bash "$KIT_SRC/run_kernel_tests.sh" "$p" "$T" > "$T.run.log" 2>&1
}
run_one "$P" "$T"; rc=$?
grep -E "passed$" "$T.run.log" | tail -1; grep -E "^FAIL" "$T.run.log" | head -5
if [ "$rc" -ne 0 ] && [ -n "${KTEST_BASE:-}" ]; then
  echo "ktest: candidate red; running the base package $KTEST_BASE's tests from the same kit tree ($T-base)"
  run_one "$KTEST_BASE" "$T-base"
  python3 "$(dirname "$0")/ktest_delta.py" "$T.run.log" "$T-base.run.log" | tee "$T.delta.txt"; rc=${PIPESTATUS[0]}
fi
echo "ktest exit $rc ($WORK/$T.run.log)"; exit $rc
