#!/bin/bash
# Hardware counters on a HIP process (ROCm 10.2 rocprofv3 + common/counters/<gfx>.yaml derived metrics), one pass per preset.
# Loom drives /dev/kfd itself, so profile served code objects through a HIP harness (devices/<gfx>.md), never loom-serve.
# Presets (each one hardware pass on gfx1151, checked with rocprofv3-avail pmc-check):
#   waits    what waves wait on: WaitVmemPct WaitDelayAluPct WaitAnyPct + raw SQ_WAIT_* and wave cycles
#   bytes    MemReadBytes (size-weighted L2 -> memory read requests) and GPU busy cycles
#   l2       GL2C hits, misses, DRAM and memory-controller read requests
#   latency  VmemLoadLatency, VmemLoadsInFlight, LdsLatency
#   lds      LDSBankConflict (% of cycles), SQC_LDS_BANK_CONFLICT, LDS activity and instructions, ALUStalledByLDS: is a modelled
#            bank conflict (Loom's compile report, common/kernel_lint.py) a measured cost? (E245)
# usage: counters.sh OUT_DIR PRESET[,PRESET...] [--kernel REGEX] -- COMMAND ...
#        counters.sh OUT_DIR calibrate        (bw_probe under bytes, waits, latency; MemReadBytes against the bytes it reads)
# Trusted counts need power_dpm_force_performance_level=profile_standard (root; restore auto); run on a quiet machine.
set -u -o pipefail
H=$(cd "$(dirname "$0")" && pwd)
f=${LSK_ENV:-$HOME/.config/loom-strix-kits/env}; [ -f "$f" ] && { set -a; . "$f"; set +a; }
R=${ROCM_SDK:?set ROCM_SDK (machine env) to a ROCm 10.2 _rocm_sdk_core}; GFX=${GFX:-gfx1151}
PROF=$(ls "$R/../../../../bin/rocprofv3" "$R/bin/rocprofv3" 2>/dev/null | head -1); [ -x "$PROF" ] || { echo "counters: no rocprofv3 near $R"; exit 1; }
Y=$H/counters/$GFX.yaml; [ -f "$Y" ] || { echo "counters: no derived counters for $GFX ($Y)"; exit 1; }
OUT=${1:?OUT_DIR}; P=${2:?PRESET}; shift 2
declare -A PMC=(
  [waits]="WaitVmemPct WaitDelayAluPct WaitAnyPct SQ_WAIT_CNT_VMVS SQ_WAIT_DELAY_ALU SQ_WAIT_ANY SQ_WAIT_CNT_LGKM SQ_WAIT_INST_LDS SQ_WAVE_CYCLES SQ_BUSY_CYCLES"
  [bytes]="MemReadBytes GRBM_GUI_ACTIVE"
  [l2]="GL2C_HIT GL2C_MISS GL2C_EA_RDREQ_DRAM GL2C_MC_RDREQ GRBM_GUI_ACTIVE"
  [latency]="VmemLoadLatency VmemLoadsInFlight LdsLatency"
  [lds]="LDSBankConflict SQC_LDS_BANK_CONFLICT SQC_LDS_IDX_ACTIVE SQ_INSTS_LDS ALUStalledByLDS SQ_INST_CYCLES_LDS SQ_WAVE_CYCLES"
)
lvl=$(cat /sys/class/drm/card*/device/power_dpm_force_performance_level 2>/dev/null | head -1)
export HSACO_RUN_ANY_LEVEL=1   # counts, not timings: hsaco_run may run at the counter level (it refuses timing at any level but auto)
[ "$lvl" = profile_standard ] || echo "counters: WARNING power level '$lvl' (not profile_standard): some counter clocks may be gated"
if [ "$P" = calibrate ]; then
  B=$(bash "$H/bw_probe.sh" 0.001 1 1 > /dev/null 2>&1; echo "${STATE:-$H}/bw_probe.$GFX")    # build only, a tiny run
  "$0" "$OUT" bytes,waits,latency --kernel 'k_read|k_write|k_copy' -- "$B" 1 3 8 || exit 1
  exec python3 "$H/counters_summary.py" "$OUT/bytes" --expect-bytes 'k_read[^4]=1073741824' --expect-bytes 'k_read4=1073741824' --expect-bytes 'k_copy=1073741824'
fi
KER=""; if [ "${1:-}" = --kernel ]; then KER=$2; shift 2; fi
[ "${1:-}" = -- ] && shift; [ $# -gt 0 ] || { echo "counters: no command after --"; exit 1; }
rc=0
for p in ${P//,/ }; do
  [ -n "${PMC[$p]:-}" ] || { echo "counters: unknown preset $p (${!PMC[*]})"; exit 1; }
  mkdir -p "$OUT/$p"
  LD_LIBRARY_PATH="$R/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$PROF" -E "$Y" --pmc ${PMC[$p]} --kernel-trace \
    ${KER:+--kernel-include-regex "$KER"} --output-format csv -d "$OUT/$p" -o "$p" -- "$@" > "$OUT/$p/run.log" 2>&1 \
    || { echo "counters: preset $p failed (exit $?): $OUT/$p/run.log"; tail -5 "$OUT/$p/run.log"; rc=1; continue; }
  echo "== $p"; python3 "$H/counters_summary.py" "$OUT/$p" ${KER:+--kernel "$KER"} || rc=1
done
exit $rc
