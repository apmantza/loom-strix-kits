#!/bin/bash
# Hardware counters on a HIP process (ROCm 10.2 rocprofv3 + common/counters/<gfx>.yaml derived metrics), one pass per preset.
# Loom drives /dev/kfd itself, so profile served code objects through a HIP harness (devices/<gfx>.md), never loom-serve.
# Presets (each one hardware pass on gfx1151, checked with rocprofv3-avail pmc-check; definitions follow AMD's, see the yaml):
#   waits    what waves wait on: WaitVmemPct WaitLgkmPct WaitDelayAluPct WaitAnyPct + raw SQ_WAIT_* and wave cycles
#   sync     AMD's wait-state table (rocprof-compute 7.9): WaitCntAnyPct WaitBarrierPct WaitIfetchPct WaitAnyPct
#   issue    is the loop issue-bound? WaitInstAnyPct (AMD 2.1 Wave Issue Wait), WaitInstValuPct, IPC (AMD 2.1), VALU and VOPD counts
#   bytes    MemReadBytes (size-weighted L2 -> fabric read requests = AMD GL2-Fabric Read BW x time) and GPU busy cycles
#   l2       L2CacheHit (AMD 13.1), GL2C hits, misses, DRAM and memory-controller read requests
#   cache    GL0HitPct GL1HitPct (AMD 9.3, 11.3), MemWriteBytes, FabricReadBytes / FabricWriteBytes (GCEA, AMD 3.9: fabric requests incl. MALL hits, not DRAM, E497)
#   latency  VmemLoadLatency, VmemLoadsInFlight, LdsLatency (+ the TEX / FLAT load counts VmemLoadLatency divides by)
#   lds      LDSBankConflict (% of LDS-active cycles stalled by conflicts, = AMD 3.4 LDS Bank Conflict Rate), SQC_LDS_BANK_CONFLICT,
#            LDS activity and instructions, ALUStalledByLDS: is a modelled bank conflict (Loom's compile report,
#            common/kernel_lint.py) a measured cost? (E245)
# usage: counters.sh OUT_DIR PRESET[,PRESET...] [--kernel REGEX] -- COMMAND ...
#        counters.sh OUT_DIR calibrate        (bw_probe under bytes, waits, latency; MemReadBytes against the bytes it reads)
# Each preset ends with counter_saturation.py's verdict: a counter at k x 2^32 (32-bit instances saturated on a long launch) fails the run
# unless COUNTERS_ALLOW_SATURATED=1; shorten the launch instead (fewer tokens or less context).
# Trusted counts need power_dpm_force_performance_level=profile_standard (root; restore auto); run on a quiet machine.
set -u -o pipefail
H=$(cd "$(dirname "$0")" && pwd)
f=${LSK_ENV:-$HOME/.config/loom-strix-kits/env}; [ -f "$f" ] && { set -a; . "$f"; set +a; }
R=${ROCM_SDK:?set ROCM_SDK (machine env) to a ROCm 10.2 _rocm_sdk_core}; GFX=${GFX:-gfx1151}
PROF=$(ls "$R/../../../../bin/rocprofv3" "$R/bin/rocprofv3" 2>/dev/null | head -1); [ -x "$PROF" ] || { echo "counters: no rocprofv3 near $R"; exit 1; }
Y=$H/counters/$GFX.yaml; [ -f "$Y" ] || { echo "counters: no derived counters for $GFX ($Y)"; exit 1; }
OUT=${1:?OUT_DIR}; P=${2:?PRESET}; shift 2
declare -A PMC=(
  [waits]="WaitVmemPct WaitLgkmPct WaitDelayAluPct WaitAnyPct SQ_WAIT_CNT_VMVS SQ_WAIT_DELAY_ALU SQ_WAIT_ANY SQ_WAIT_CNT_LGKM SQ_WAIT_INST_LDS SQ_WAVE_CYCLES SQ_BUSY_CYCLES"
  [sync]="WaitCntAnyPct WaitBarrierPct WaitIfetchPct WaitAnyPct SQ_WAVES"
  [issue]="WaitInstAnyPct WaitInstValuPct IPC SQ_INSTS_VALU SQ_INSTS_DUAL_VALU_WAVE32"
  [bytes]="MemReadBytes GRBM_GUI_ACTIVE"
  [l2]="L2CacheHit GL2C_HIT GL2C_MISS GL2C_EA_RDREQ_DRAM GL2C_MC_RDREQ GRBM_GUI_ACTIVE"
  [cache]="GL0HitPct GL1HitPct MemWriteBytes FabricReadBytes FabricWriteBytes GRBM_GUI_ACTIVE"
  [latency]="VmemLoadLatency VmemLoadsInFlight LdsLatency SQ_INSTS_TEX_LOAD SQ_INSTS_FLAT_LOAD"
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
  python3 "$H/counter_saturation.py" "$OUT/$p" ${KER:+--kernel "$KER"}; s=$?   # 32-bit counters stop at 2^32 per instance on long launches
  [ $s = 1 ] && [ "${COUNTERS_ALLOW_SATURATED:-0}" = 1 ] && { echo "counters: saturated counters allowed (COUNTERS_ALLOW_SATURATED=1): ratios on them are wrong"; s=0; }
  [ $s = 0 ] || rc=1
done
exit $rc
