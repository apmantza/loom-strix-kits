#!/bin/bash
# Build (if needed) and run common/bw_probe.hip: read, write and copy DRAM bandwidth on the GPU, best and median GB/s.
# Measure on a quiet machine (no other server or build: common/wait_quiet.sh first); compare the read figure with
# lanes.json's hardware.bw_practical_gbs and devices/<gfx>.md.
# usage: bw_probe.sh [GiB=4] [reps=20] [blocks_per_CU=8]
#   machine env: ROCM_SDK = a ROCm root with hipcc and gfx device libraries (default /opt/rocm); GFX (default gfx1151)
set -u -o pipefail
H=$(cd "$(dirname "$0")" && pwd)
f=${LSK_ENV:-$HOME/.config/loom-strix-kits/env}; [ -f "$f" ] && { set -a; . "$f"; set +a; }
R=${ROCM_SDK:-/opt/rocm}; OUT=${STATE:-$H}/bw_probe.${GFX:-gfx1151}
if [ ! -x "$OUT" ] || [ "$H/bw_probe.hip" -nt "$OUT" ]; then
  "$R/bin/hipcc" -O3 --offload-arch="${GFX:-gfx1151}" --rocm-path="$R" --rocm-device-lib-path="$R/lib/llvm/amdgcn/bitcode" \
    "$H/bw_probe.hip" -o "$OUT" || { echo "bw_probe: build failed (set ROCM_SDK to a ROCm root with gfx device libraries)"; exit 1; }
fi
LD_LIBRARY_PATH="$R/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" exec "$OUT" "$@"
