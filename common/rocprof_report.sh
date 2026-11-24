#!/bin/bash
# rocprof-compute analyze on a profiled workload with our analysis configs (common/counters/rocprof-compute-<arch>: % of peak against
# the device file's clock and bandwidth, unsupported SPI rows marked), then the 32-bit saturation check on its raw counters.
# usage: rocprof_report.sh RUN_DIR [BLOCK...]     (default blocks 2 3 6 7: SOL, memory chart, SPI, WGP; kernel 0)
# Needs ROCPROF_COMPUTE_LIBEXEC (rocprof-compute's libexec/rocprofiler-compute; default derived from ROCM_SDK) and ROCPROF_ANALYZE_PY
# (a python with analyze's pinned packages: pandas, tabulate, textual; default: python3). Offline: no GPU.
# Exit: analyze's failure (2), saturation FOUND (1, unless COUNTERS_ALLOW_SATURATED=1), else 0. The table is printed either way.
set -u -o pipefail
H=$(cd "$(dirname "$0")" && pwd)
f=${LSK_ENV:-$HOME/.config/loom-strix-kits/env}; [ -f "$f" ] && { set -a; . "$f"; set +a; }
RUN=${1:?usage: rocprof_report.sh RUN_DIR [BLOCK...]}; shift
[ $# -gt 0 ] || set -- 2 3 6 7
L=${ROCPROF_COMPUTE_LIBEXEC:-$(dirname "${ROCM_SDK:?set ROCM_SDK or ROCPROF_COMPUTE_LIBEXEC}")/_rocm_profiler/libexec/rocprofiler-compute}
CFG=$H/counters/rocprof-compute-gfx115x
[ -f "$L/rocprof-compute" ] || { echo "rocprof_report: no rocprof-compute under $L"; exit 2; }
[ -f "$CFG/0200_system_speed_of_light.yaml" ] || { echo "rocprof_report: no configs in $CFG (counters/make_rocprof_configs.py)"; exit 2; }
echo "rocprof_report: $RUN, configs $CFG ($(sed -n 3p "$CFG/0200_system_speed_of_light.yaml" | cut -c3-))"
env PYTHONPATH="$L" "${ROCPROF_ANALYZE_PY:-python3}" "$L/rocprof-compute" analyze -p "$RUN" --config-dir "$CFG" -k 0 -b "$@" --view table \
  || { echo "rocprof_report: analyze failed"; exit 2; }
python3 "$H/counter_saturation.py" "$RUN"; s=$?
[ $s = 1 ] && [ "${COUNTERS_ALLOW_SATURATED:-0}" = 1 ] && { echo "rocprof_report: saturated counters allowed (COUNTERS_ALLOW_SATURATED=1): ratios on them are wrong"; s=0; }
exit $s
