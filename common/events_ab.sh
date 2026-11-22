#!/bin/bash
# Served per-dispatch profiles of packages in the order given (A B B A for an ABBA) at one prompt size, then events_summary.py
# (each lane against its hardware floor, kits/<kit>/lanes.json). Per run: stop other servers (and a managed one, LSK_STOP_HOOK),
# wait for a quiet machine, serve with the kit's flags plus device profiling into a bundle, send profile_req.py's long request,
# check the served context against the label, stop, export the dispatch events with iree-profile. The managed server comes back
# on every exit (LSK_RESTORE_HOOK).
# usage: events_ab.sh KIT OUTDIR LENGTH PKG [PKG ...]
#   machine env: IREE_PROFILE = an iree-profile binary built from the installed hrx-system tag (tools/iree-profile; not in the
#   wheel); SPLIT = a kernel whose dispatch-time clusters to print (default g_attention_vsplit)
set -u -o pipefail
H=$(cd "$(dirname "$0")" && pwd)
. "$H/env.sh"; lsk_load "${1:?usage: events_ab.sh KIT OUTDIR LENGTH PKG [PKG ...]}"
OUT=$(realpath -m "${2:?out dir}"); L=${3:?prompt length}; shift 3
[ $# -ge 1 ] || die "usage: events_ab.sh KIT OUTDIR LENGTH PKG [PKG ...]"
[ -n "${IREE_PROFILE:-}" ] && [ -x "$IREE_PROFILE" ] || die "set IREE_PROFILE to an iree-profile binary (machine env)"
for p in "$@"; do lsk_pkg "$p" > /dev/null; done
trap 'lsk_restore; echo "restored $(date +%T)"' EXIT
mkdir -p "$OUT"; n=0; runs=()
for p in "$@"; do
  n=$((n + 1)); d=$OUT/$n-$(basename "$p"); rm -rf "$d"; mkdir -p "$d"; runs+=("$d")
  lsk_stop || exit 1
  bash "$H/wait_quiet.sh" | tail -1 || die "events_ab STOP: machine not quiet before run $n"
  lsk_serve_bg prof "$p" --device_profiling_mode=dispatch-events,executable-metadata --device_profiling_flush_interval_ms=5 \
    --device_profiling_output="$d/run.ireeprof"
  "$PYTHON" "$H/profile_req.py" "http://127.0.0.1:$PORT" "$KIT_MODEL_ID" "$L" > "$d/run.req.json" 2> "$d/run.req.err" || die "events_ab STOP: request $p"
  echo "### $n $p $(python3 -I "$H/context_label_check.py" "$d/run.req.json" "$L")"
  lsk_stop || exit 1
  "$IREE_PROFILE" dispatch --dispatch_events --format=jsonl "$d/run.ireeprof" > "$d/events.jsonl" 2> "$d/events.err" || die "events_ab STOP: iree-profile export $p"
done
python3 -I "$H/events_summary.py" "$KD/$KIT_LANES" "${runs[@]}" --split "${SPLIT:-g_attention_vsplit}" | tee "$OUT/summary.md"
echo "EVENTS_AB_DONE $(date +%T)"
