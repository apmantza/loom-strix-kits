#!/bin/bash
# Served per-dispatch profiles of packages in the order given (A B B A for an ABBA) at one prompt size, then events_summary.py
# (each lane against its hardware floor, kits/<kit>/lanes.json). Per run: stop other servers (and a managed one, LSK_STOP_HOOK),
# wait for a quiet machine, serve with the kit's flags plus device profiling into a bundle, send profile_req.py's long request,
# check the served context against the label (a > 5 % mismatch stops the script: a profile labelled 81k that served 66k is not
# the cell it claims, HRX-like E421), stop, export the dispatch events with iree-profile. The managed server comes back
# on every exit (LSK_RESTORE_HOOK).
# Environment (strix-llama.cpp #241: a Mesa update turned a measured speedup into a loss): env_block.py records the stack before
# each run (<run>/env.txt); summary.md ends with an `environment:` section (the first run's block) and each run whose block
# differs from it, and then the script exits 2 (EVENTS: UNTRUSTED): profiles from two environments are not an A/B.
# usage: events_ab.sh KIT OUTDIR LENGTH PKG [PKG ...]
#   env EXPECT_TOKENS: the prompt tokens LENGTH is expected to serve (default LENGTH); a run more than 5 % off stops the script.
#     LENGTH is a size label, not a token count (HRX-like E421: a label of 100000 served 81,432; a step stopped on LENGTH as the expectation)
#   machine env: IREE_PROFILE = an iree-profile binary built from the installed hrx-system tag (tools/iree-profile; not in the
#   wheel); SPLIT = a kernel whose dispatch-time clusters to print (default g_attention_vsplit)
set -u -o pipefail
H=$(cd "$(dirname "$0")" && pwd)
. "$H/env.sh"; lsk_load "${1:?usage: events_ab.sh KIT OUTDIR LENGTH PKG [PKG ...]}"
OUT=$(realpath -m "${2:?out dir}"); L=${3:?prompt length}; shift 3
[ $# -ge 1 ] || die "usage: events_ab.sh KIT OUTDIR LENGTH PKG [PKG ...]"
[ -n "${IREE_PROFILE:-}" ] && [ -x "$IREE_PROFILE" ] || die "set IREE_PROFILE to an iree-profile binary (machine env)"
eb=(); for p in "$@"; do eb+=(--pkg "$(lsk_pkg "$p")") || exit 1; done   # every package, in the environment block
trap 'lsk_restore; echo "restored $(date +%T)"' EXIT
mkdir -p "$OUT"; n=0; runs=()
for p in "$@"; do
  n=$((n + 1)); d=$OUT/$n-$(basename "$p"); rm -rf "$d"; mkdir -p "$d"; runs+=("$d")
  lsk_stop || exit 1
  python3 "$H/env_block.py" --kit "$LSK" "${eb[@]}" > "$d/env.txt" 2> "$d/env.err"
  bash "$H/wait_quiet.sh" | tail -1 || die "events_ab STOP: machine not quiet before run $n"
  lsk_serve_bg prof "$p" --device_profiling_mode=dispatch-events,executable-metadata --device_profiling_flush_interval_ms=5 \
    --device_profiling_output="$d/run.ireeprof"
  "$PYTHON" "$H/profile_req.py" "http://127.0.0.1:$PORT" "$KIT_MODEL_ID" "$L" > "$d/run.req.json" 2> "$d/run.req.err" || die "events_ab STOP: request $p"
  lab=$(python3 -I "$H/context_label_check.py" "$d/run.req.json" "${EXPECT_TOKENS:-$L}"); lc=$?   # not inside the echo: its status was discarded there
  echo "### $n $p $lab"
  [ $lc -eq 0 ] || die "events_ab STOP: run $n served another context than ${EXPECT_TOKENS:-$L} tokens (LENGTH $L is the request's size label: set EXPECT_TOKENS to what it serves)"
  lsk_stop || exit 1
  "$IREE_PROFILE" dispatch --dispatch_events --format=jsonl "$d/run.ireeprof" > "$d/events.jsonl" 2> "$d/events.err" || die "events_ab STOP: iree-profile export $p"
done
python3 -I "$H/events_summary.py" "$KD/$KIT_LANES" "${runs[@]}" --split "${SPLIT:-g_attention_vsplit}" | tee "$OUT/summary.md"
envok=0   # no pipe around the block: envok must survive it
{ echo "environment:"; sed 's/^/  /' "${runs[0]}/env.txt"
  for d in "${runs[@]}"; do
    python3 "$H/env_block.py" --diff "${runs[0]}/env.txt" "$d/env.txt" > "$d/env.diff" 2>&1 && continue
    envok=1; echo "environment changed before $(basename "$d") (or not comparable):"; sed 's/^/  /' "$d/env.diff"
  done
  [ $envok = 0 ] || echo "EVENTS: UNTRUSTED (the environment differs between runs)"; } > "$OUT/env.md"
tee -a "$OUT/summary.md" < "$OUT/env.md"
echo "EVENTS_AB_DONE $(date +%T)"
[ $envok = 0 ] || exit 2
