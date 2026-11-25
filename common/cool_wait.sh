#!/bin/bash
# Thermal start gate (HRX-like E368, E485): wait until the GPU edge temperature is <= T_START (C, default 55), or for at most
# COOL_MAX s (default 900), then print where the run starts. At the 85 W cap the clock follows temperature (rules section 5):
# a run started hot runs at a lower clock than one started cool, so every timed run starts from the same thermal state.
# The caller stops its server first (the wait is for an idle GPU). One writer of the rule: agent_ab (each half) and
# events_ab (each run) call it.
# usage: cool_wait.sh TAG [OUTFILE]     prints 'TAG start: edge N C after S s of cooling' (and a HOT note at the timeout),
#   tees it to OUTFILE; exit 0 started (cool or HOT), 2 no temperature reading (fail closed: an unmeasured start is not a gate)
#   env: T_START, COOL_MAX, COOL_STEP (poll s, default 10); COOL_TEMP_CMD (default: gpu_telemetry.py temp; tests stub it)
H=$(cd "$(dirname "$0")" && pwd)
tag=${1:?usage: cool_wait.sh TAG [OUTFILE]}; out=${2:-/dev/null}
TS=${T_START:-55}; MAX=${COOL_MAX:-900}; STEP=${COOL_STEP:-10}
read -ra TEMP <<< "${COOL_TEMP_CMD:-python3 $H/gpu_telemetry.py temp}"
t0=$(date +%s)
while :; do
  tc=$("${TEMP[@]}" 2>/dev/null) && [[ "$tc" =~ ^[0-9]+$ ]] || { echo "cool_wait STOP: no GPU temperature for the start gate ($tag)"; exit 2; }
  [ "$tc" -le "$TS" ] && break
  [ $(( $(date +%s) - t0 )) -ge "$MAX" ] && { echo "cool_wait NOTE: $tag starts HOT ($tc C > $TS C after $MAX s)"; break; }
  sleep "$STEP"
done
echo "$tag start: edge $tc C after $(( $(date +%s) - t0 )) s of cooling" | tee "$out"
