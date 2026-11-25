#!/bin/bash
# Build a package with a kit's recipe (kits/<kit>/kit.env) plus optional extra switches; the exact command goes to PKG.cmd.
# usage: [EXTRA="SWITCH=1 ..."] [LINT_BASE=PKG [LINT_ALLOW="REGEX ..."] [LINT_ACCEPT="REGEX ..."]] build.sh KIT PKG
# Refuses while one of our measurement servers runs: a CPU build beside a measurement skews it (Strix Halo: CPU and GPU
# share one DRAM bus; a build beside a profile moved GEMM time 123 -> 131 s).
# LINT_BASE=PKG: a candidate lints itself after the build (common/cand_lint.py check: binary_gate against PKG, then kernel_lint's
# prefetch and spill checks on the kernels the gate reports changed; report in PKG.lint.txt, code objects and rows in PKG.lint/).
# HRX-like E520: STAGE_LIVE x FLATADDR drained flash2's V prefetch and took three hand-built pair packages to find; `cand_lint.py
# pairs` builds the pairs. LINT_ALLOW: the switch's kernel patterns (binary_gate --allow); without it binary_gate's untouched-kernel
# verdict is printed, not judged. LINT_ACCEPT: kernels whose loss a ledger row accepted (cand_lint --accept; cite the row).
# Exit codes: 0 built (and, with LINT_BASE, lint passed); 3 = a COMPLETE but REJECTED package: the build succeeded and the lint
# failed, so PKG/model.json exists. The verdict is PKG/cand_lint.verdict, one line starting "CAND_LINT: PASS", "CAND_LINT: FAIL" or
# "CAND_LINT: NOT RUN" (no LINT_BASE). A resume idiom that skips the build when PKG/model.json exists must also require
# `grep -q '^CAND_LINT: PASS' PKG/cand_lint.verdict`, or a rerun forgets the rejection (review r1 F7).
set -u -o pipefail
. "$(dirname "$0")/env.sh"; lsk_load "${1:?usage: [EXTRA=...] [LINT_BASE=PKG] build.sh KIT PKG}"
P=${2:?package dir}
m=$(lsk_measuring) && die "BUILD REFUSED: measurement server $m is running"
for x in loom-benchmark loom-test; do pgrep -x "$x" >/dev/null && die "BUILD REFUSED: $x is running"; done
[ -n "${LOOM_KB_ROOT:-}" ] || die "set LOOM_KB_ROOT (skills/loom-kb of a loom-skills checkout) in the machine env"
[ -z "${LINT_BASE:-}" ] || lsk_pkg "$LINT_BASE" >/dev/null || exit 1
CFGP=$KIT_CONFIG; [[ $CFGP = /* ]] || CFGP=$KD/$KIT_CONFIG
dirty=$(git -C "$LSK" status --porcelain --untracked-files=no -- kit | wc -l)
src=""; [ "$KIT_SRC" = "$LSK/kit" ] || src=" KIT_SRC=$KIT_SRC $(git -C "$KIT_SRC" rev-parse --short HEAD 2>/dev/null || echo '(not a git tree)')"
echo "env $KIT_RECIPE ${EXTRA:-} kit/build.py kits/$KIT/$KIT_CONFIG $P --tokenizer $KIT_TOKENIZER $KIT_BUILD_ARGS  # loom-strix-kits $(git -C "$LSK" rev-parse --short HEAD 2>/dev/null)$([ "$dirty" -gt 0 ] && echo " + $dirty uncommitted kit files")$src" > "$P.cmd"
rm -rf "$P" "$P.lint" "$P.lint.txt"
# shellcheck disable=SC2086
env $KIT_RECIPE ${EXTRA:-} PATH="${VENV:+$VENV/bin:}$PATH" "$PYTHON" "$KIT_SRC/build.py" "$CFGP" "$P" --tokenizer "$KIT_TOKENIZER" $KIT_BUILD_ARGS > "$P.log" 2>&1
rc=$?; grep -E "serve with|workspace" "$P.log" | tail -1; echo "build exit $rc ($WORK/$P.log, command in $P.cmd)"
[ $rc -eq 0 ] || exit $rc
V=$P/cand_lint.verdict
[ -n "${LINT_BASE:-}" ] || { echo "CAND_LINT: NOT RUN (no LINT_BASE)" > "$V"; echo "post-build lint not run (no LINT_BASE)"; exit 0; }
allow=(); for x in ${LINT_ALLOW:-}; do allow+=(--allow "$x"); done
for x in ${LINT_ACCEPT:-}; do allow+=(--accept "$x"); done
python3 "$LSK/common/cand_lint.py" check "$LINT_BASE" "$P" --out "$P.lint" --jobs "${LINT_JOBS:-2}" "${allow[@]}" > "$P.lint.txt" 2>&1
lr=$?; grep -E "^  (FAIL|NEW|ok|ACCEPTED|text) |^CAND_LINT" "$P.lint.txt"
if [ $lr -eq 0 ] && grep -q '^CAND_LINT: PASS' "$P.lint.txt"; then echo "CAND_LINT: PASS against $LINT_BASE ($P.lint.txt)" > "$V"; exit 0; fi
echo "CAND_LINT: FAIL against $LINT_BASE (exit $lr; $P.lint.txt)" > "$V"
echo "CANDIDATE LINT FAILED (exit $lr; $WORK/$P.lint.txt): the package is complete but rejected (exit 3)"; exit 3
