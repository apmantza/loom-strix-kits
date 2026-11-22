#!/bin/bash
# Build a package with a kit's recipe (kits/<kit>/kit.env) plus optional extra switches; the exact command goes to PKG.cmd.
# usage: [EXTRA="SWITCH=1 ..."] build.sh KIT PKG
# Refuses while one of our measurement servers runs: a CPU build beside a measurement skews it (Strix Halo: CPU and GPU
# share one DRAM bus; a build beside a profile moved GEMM time 123 -> 131 s).
set -u -o pipefail
. "$(dirname "$0")/env.sh"; lsk_load "${1:?usage: [EXTRA=...] build.sh KIT PKG}"
P=${2:?package dir}
m=$(lsk_measuring) && die "BUILD REFUSED: measurement server $m is running"
for x in loom-benchmark loom-test; do pgrep -x "$x" >/dev/null && die "BUILD REFUSED: $x is running"; done
[ -n "${LOOM_KB_ROOT:-}" ] || die "set LOOM_KB_ROOT (skills/loom-kb of a loom-skills checkout) in the machine env"
CFGP=$KIT_CONFIG; [[ $CFGP = /* ]] || CFGP=$KD/$KIT_CONFIG
dirty=$(git -C "$LSK" status --porcelain --untracked-files=no -- kit | wc -l)
echo "env $KIT_RECIPE ${EXTRA:-} kit/build.py kits/$KIT/$KIT_CONFIG $P --tokenizer $KIT_TOKENIZER $KIT_BUILD_ARGS  # loom-strix-kits $(git -C "$LSK" rev-parse --short HEAD 2>/dev/null)$([ "$dirty" -gt 0 ] && echo " + $dirty uncommitted kit files")" > "$P.cmd"
rm -rf "$P"
# shellcheck disable=SC2086
env $KIT_RECIPE ${EXTRA:-} PATH="${VENV:+$VENV/bin:}$PATH" "$PYTHON" "$KIT_SRC/build.py" "$CFGP" "$P" --tokenizer "$KIT_TOKENIZER" $KIT_BUILD_ARGS > "$P.log" 2>&1
rc=$?; grep -E "serve with|workspace" "$P.log" | tail -1; echo "build exit $rc ($WORK/$P.log, command in $P.cmd)"; exit $rc
