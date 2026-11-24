#!/bin/bash
# Wait until no loom-serve/llama-server process is left and MemAvailable stops rising (a stopped server releases its
# memory over seconds; a server started meanwhile can fail its memory check). Gives up after ${1:-180} s.
# Then admission (common/mem_admit.py admit): no other model resident on the GPU, and MemAvailable - CmaFree (and GTT free) at
# least the run's LSK_MEM_NEED_GIB plus a reserve of max(8 GiB, 8 % of RAM); waited for like the memory release, refused at the end.
# A package build, kernel-test generation or compile on the CPU also skews a measurement taken after this (a profile ran beside a build, GEMMs 123 -> 131 s): wait for those first, up to ${WAIT_BUILD:-3600} s.
BUILDERS='kit[-a-z0-9]*/build\.py|make_kernel_tests\.py|build_draft\.py|loom-compile '
bend=$(( $(date +%s) + ${WAIT_BUILD:-3600} )); said=
# Match on the running program, not on any command line that mentions one: a waiter whose bash -c text quotes
# `loom-compile ` (a heredoc) matched itself and held two replay halves for an hour each.
builder() { ps -eo comm=,args= | RE="$BUILDERS" awk '($1 ~ /^python/ || $1 == "loom-compile") && $0 ~ ENVIRON["RE"] {print; exit}'; }
while b=$(builder) && [ -n "$b" ]; do
  [ -n "$said" ] || { echo "wait_quiet: waiting for a build to finish: ${b:0:120}"; said=1; }
  [ $(date +%s) -lt $bend ] || { echo "wait_quiet: gave up, a build is still running: ${b:0:120}"; exit 1; }
  sleep 5
done
end=$(( $(date +%s) + ${1:-180} )); prev=0; adm=
while [ $(date +%s) -lt $end ]; do
  if ! pgrep -x loom-serve >/dev/null && ! pgrep -x llama-server >/dev/null; then
    cur=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    [ "$cur" -le "$prev" ] && adm=$(python3 -I "$(dirname "$0")/mem_admit.py" admit 2>&1) && {
      echo "memory free: MemAvailable ${cur} GiB; $adm"
      # the machine's other load at the start of a measurement (on Strix Halo CPU work competes with the GPU for the
      # one DRAM bus): recorded with every result
      echo "load: $(cut -d' ' -f1-3 /proc/loadavg) on $(nproc) threads; top: $(ps -eo pcpu,comm --sort=-pcpu --no-headers | head -4 | awk '{printf "%s %s%%, ", $2, $1}')"
      exit 0; }
    prev=$cur
  fi
  sleep 3
done
echo "wait_quiet: gave up after ${1:-180} s (MemAvailable $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GiB)${adm:+; $adm}"; exit 1
