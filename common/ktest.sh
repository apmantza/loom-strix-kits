#!/bin/bash
# Kernel tests of a package: generate the cases for exactly the switches the package was built with (read from PKG.cmd,
# which build.sh writes; a test generated for other switches tests other kernels), then run them on the GPU.
# Correctness only; safe beside a running server when memory allows. usage: ktest.sh KIT PKG TESTDIR
set -u -o pipefail
. "$(dirname "$0")/env.sh"; lsk_load "${1:?usage: ktest.sh KIT PKG TESTDIR}"
P=${2:?package dir}; T=${3:?test dir}
# KTEST_BASE=PKG (optional): judge the delta (common/ktest_delta.py). A red candidate runs the base package's tests from the same
# kit tree; a case red on both is INHERITED, not a stop (E445). Unset: every FAIL stops, as before.
# KTEST_CHANGED=BGATE.json (optional; binary_gate rows against a base, e.g. PKG.lint/bgate.json from build.sh LINT_BASE): generate
# and run only the cases testing a kernel the gate reports CHANGED, NEW or text-only (directly or through a test copy), plus the
# smoke set (kit/ktest_select.py SMOKE); the full generation costs ~35 GB / ~100 s / 18 GB of inputs (HRX-like E532). Fails closed
# to the full suite: a changed set the file cannot give (unreadable, a kernel the gate could not judge, a verdict.json beside it that
# names another candidate than PKG or another base than KTEST_BASE, or none), or a selection without a smoke case or with a changed
# kernel no case tests (make_kernel_tests.py exit 3). With KTEST_BASE the base's run uses the same selection. Go-live candidates
# run without KTEST_CHANGED (the full suite): the run log's 'ktest scope:' line (full / subset / unknown, from the generator's
# ktest_scope.json) says which, and HRX-like scripts/ktest_verdict.sh passes only 'full' unless the step admits more (KTEST_SCOPE_OK).
KS=""; FULL=1
if [ -n "${KTEST_CHANGED:-}" ]; then
  idargs=(--cand "$P"); [ -n "${KTEST_BASE:-}" ] && idargs+=(--base "$KTEST_BASE")
  if KS=$(python3 "$KIT_SRC/ktest_select.py" changed "$KTEST_CHANGED" "${idargs[@]}"); then FULL=0; echo "ktest: selected (changed vs base: ${KS:-none})"
  else echo "ktest: changed set unknown: full suite"; fi
fi
run_one() {   # PKG_DIR TESTDIR -> TESTDIR.run.log; the switches come from PKG_DIR.cmd
  local P=$1 T=$2 p SW rc; p=$(lsk_pkg "$P")
  [ -f "$P.cmd" ] || die "no $P.cmd (build with common/build.sh) - the test switches must be the package's"
  SW=$(sed -n 's/^env \(.*\) kit\/build\.py .*/\1/p' "$P.cmd"); [ -n "$SW" ] || die "cannot read the switches from $P.cmd"
  rm -rf "$T"
  if [ "$FULL" = 0 ]; then
    # shellcheck disable=SC2086
    env $SW KTEST_KERNELS="$KS" PATH="${VENV:+$VENV/bin:}$PATH" "$PYTHON" "$KIT_SRC/make_kernel_tests.py" "$p/llm_config.json" "$T" > "$T.gen.log" 2>&1
    rc=$?; grep -E "^selected generation" "$T.gen.log"
    [ $rc -eq 3 ] && { echo "ktest: the selection refused (see $T.gen.log): full suite"; FULL=1; rm -rf "$T"; }
    [ $rc -ne 0 ] && [ $rc -ne 3 ] && die "test generation failed ($WORK/$T.gen.log)"
  fi
  if [ "$FULL" = 1 ]; then
    # shellcheck disable=SC2086
    env $SW PATH="${VENV:+$VENV/bin:}$PATH" "$PYTHON" "$KIT_SRC/make_kernel_tests.py" "$p/llm_config.json" "$T" > "$T.gen.log" 2>&1 \
      || die "test generation failed ($WORK/$T.gen.log)"
  fi
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
