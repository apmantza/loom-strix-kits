#!/bin/bash
# Build a package with a kit's recipe (kits/<kit>/kit.env) plus optional extra switches; the exact command goes to PKG.cmd.
# usage: [EXTRA="SWITCH=1 ..."] [LINT_BASE=PKG [LINT_ALLOW="REGEX ..."]] build.sh KIT PKG
# Refuses while one of our measurement servers runs: a CPU build beside a measurement skews it (Strix Halo: CPU and GPU
# share one DRAM bus; a build beside a profile moved GEMM time 123 -> 131 s).
# LINT_BASE=PKG: a candidate lints itself after the build (common/cand_lint.py check: binary_gate against PKG, then kernel_lint's
# prefetch check on the kernels the gate reports changed; report in PKG.lint.txt, code objects and rows in PKG.lint/). A failed lint
# exits 3 (the build itself succeeded). HRX-like E520: STAGE_LIVE x FLATADDR drained flash2's V prefetch and took three hand-built
# pair packages to find; `cand_lint.py pairs` builds the pairs. LINT_ALLOW: the switch's kernel patterns (binary_gate --allow);
# without it binary_gate's untouched-kernel verdict is printed, not judged.
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
[ -n "${LINT_BASE:-}" ] || { echo "post-build lint not run (no LINT_BASE)"; exit 0; }
allow=(); for x in ${LINT_ALLOW:-}; do allow+=(--allow "$x"); done
python3 "$LSK/common/cand_lint.py" check "$LINT_BASE" "$P" --out "$P.lint" --jobs "${LINT_JOBS:-2}" "${allow[@]}" > "$P.lint.txt" 2>&1
lr=$?; grep -E "^  (FAIL|NEW|ok) |^CAND_LINT" "$P.lint.txt"
[ $lr -eq 0 ] || { echo "CANDIDATE LINT FAILED (exit $lr; $WORK/$P.lint.txt)"; exit 3; }
