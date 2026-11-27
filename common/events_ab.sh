#!/bin/bash
# Served per-dispatch profiles of packages in the order given (A B B A for an ABBA) at one prompt size, then events_summary.py
# (each lane against its hardware floor, kits/<kit>/lanes.json). Per run: stop other servers (and a managed one, LSK_STOP_HOOK),
# start from the same thermal state (cool_wait.sh), wait for a quiet machine, record free large memory blocks (mem_admit.py frag),
# serve with the kit's flags plus device profiling into a bundle, record telemetry beside the request (gpu_telemetry.py: clock,
# power, temperature, busy %, and the CPU beside the server; events_summary's cycles table reads the busy clock from tel.tsv),
# send profile_req.py's long request, check the served context against the label (a > 5 % mismatch stops the script: a profile
# labelled 81k that served 66k is not the cell it claims, HRX-like E421), stop and wait for the server to exit (it writes the
# bundle), export the dispatch events with iree-profile. The managed server comes back on every exit (LSK_RESTORE_HOOK).
# Environment (strix-llama.cpp #241: a Mesa update turned a measured speedup into a loss): env_block.py records the stack before
# each run (<run>/env.txt); summary.md ends with an `environment:` section (the first run's block) and each run whose block
# differs from it. UNTRUSTED (the script ends with 'EVENTS: UNTRUSTED ...' and exits 2 after the summary): the environment differs
# between runs (profiles from two environments are not an A/B), or a run's CPU line is not 'clean' (other CPU work beside the
# server, CONTAMINATED: HRX-like E415; or unknown), or its gpu line is not 'clean' (other GPU clients' engine time above 1 % of the
# run's wall or another KFD process, GPU-CONTAMINATED: HRX-like E631; or unknown: no client record). AB_CPU_MAX: other CPU in cores above which a sample counts (default 3, as agent_ab).
# Output: OUT/summary.md (lanes, then the environment section), OUT/env.md, OUT/starts.txt (each run's thermal start and memfrag),
#   OUT/<n>-<package dir name without a 'pkg-' prefix>/: start.txt env.txt env.err env.diff mem.txt run.ireeprof run.req.json
#   run.req.err tel.tsv tel.txt events.jsonl events.err
# usage: events_ab.sh KIT OUTDIR LENGTH PKG [PKG ...]      (OUTDIR and PKG relative to WORK, or absolute)
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
TEL=
trap '[ -n "$TEL" ] && kill $TEL 2>/dev/null; lsk_restore && echo "restored $(date +%T)"' EXIT
mkdir -p "$OUT"; n=0; runs=(); unclean=(); gpuunclean=()
for p in "$@"; do
  n=$((n + 1)); b=$(basename "$p"); d=$OUT/$n-${b#pkg-}; tag=$(basename "$d"); rm -rf "$d"; mkdir -p "$d"; runs+=("$d")
  lsk_stop || exit 1
  bash "$H/cool_wait.sh" "$tag" "$d/start.txt" || die "events_ab STOP: thermal start gate before run $n"   # E485: every run from the same thermal state
  python3 "$H/env_block.py" --kit "$LSK" "${eb[@]}" > "$d/env.txt" 2> "$d/env.err"
  bash "$H/wait_quiet.sh" | tail -1 || die "events_ab STOP: machine not quiet before run $n"
  python3 "$H/mem_admit.py" frag > "$d/mem.txt" 2>&1   # free large blocks before the server allocates (E471); measured, not judged
  lsk_serve_bg prof "$p" --device_profiling_mode=dispatch-events,executable-metadata --device_profiling_flush_interval_ms=5 \
    --device_profiling_output="$d/run.ireeprof"
  spid=$(cat "$STATE/prof.pid")
  python3 "$H/gpu_telemetry.py" record "$d/tel.tsv" --pid "$spid" & TEL=$!   # the server's own CPU is not contamination
  "$PYTHON" "$H/profile_req.py" "http://127.0.0.1:$PORT" "$KIT_MODEL_ID" "$L" > "$d/run.req.json" 2> "$d/run.req.err" || die "events_ab STOP: request $p"
  kill $TEL 2>/dev/null; TEL=
  python3 "$H/gpu_telemetry.py" summary "$d/tel.tsv" "$tag" --cpu-max "${AB_CPU_MAX:-3}" | tee "$d/tel.txt"
  grep -qF -- "  cpu $tag: clean: " "$d/tel.txt" || unclean+=("$tag")   # fail closed: only a 'clean' CPU line is trusted
  grep -qF -- "  gpu $tag: clean: " "$d/tel.txt" || gpuunclean+=("$tag")   # other GPU clients (E631): only 'clean' is trusted
  lab=$(python3 -I "$H/context_label_check.py" "$d/run.req.json" "${EXPECT_TOKENS:-$L}"); lc=$?   # not inside the echo: its status was discarded there
  echo "### $n $p $lab"
  [ $lc -eq 0 ] || die "events_ab STOP: run $n served another context than ${EXPECT_TOKENS:-$L} tokens (LENGTH $L is the request's size label: set EXPECT_TOKENS to what it serves)"
  lsk_stop || exit 1
  # the port can close before the server has finished writing its bundle: wait for the process itself, as the HRX-like fork's stop did
  # (a child of this shell: a zombie has exited)
  for i in $(seq 1 60); do s=$(ps -o stat= -p "$spid" 2>/dev/null); [ -z "$s" ] || [ "${s:0:1}" = Z ] && break; sleep 0.5; done
  [ -z "$s" ] || [ "${s:0:1}" = Z ] || die "events_ab STOP: server pid $spid still running 30 s after the stop (its bundle may be incomplete)"
  "$IREE_PROFILE" dispatch --dispatch_events --format=jsonl "$d/run.ireeprof" > "$d/events.jsonl" 2> "$d/events.err" || die "events_ab STOP: iree-profile export $p"
done
python3 -I "$H/events_summary.py" "$KD/$KIT_LANES" "${runs[@]}" --split "${SPLIT:-g_attention_vsplit}" | tee "$OUT/summary.md"
envok=0   # no pipe around the block: envok must survive it
{ echo "environment:"; sed 's/^/  /' "${runs[0]}/env.txt"
  for d in "${runs[@]}"; do
    python3 "$H/env_block.py" --diff "${runs[0]}/env.txt" "$d/env.txt" > "$d/env.diff" 2>&1 && continue
    envok=1; echo "environment changed before $(basename "$d") (or not comparable):"; sed 's/^/  /' "$d/env.diff"
  done
  [ $envok = 0 ] || echo "EVENTS: UNTRUSTED (the environment differs between runs)"
  for d in "${runs[@]}"; do echo "memory before $(basename "$d"): $(cat "$d/mem.txt" 2>/dev/null || echo 'memfrag unknown (no mem.txt)')"; done; } > "$OUT/env.md"
tee -a "$OUT/summary.md" < "$OUT/env.md"
for d in "${runs[@]}"; do echo "$(cat "$d/start.txt" 2>/dev/null || echo "$(basename "$d") start: unknown"); $(cut -c1-60 "$d/mem.txt" 2>/dev/null)"; done | tee "$OUT/starts.txt"
[ ${#unclean[@]} = 0 ] || echo "EVENTS: UNTRUSTED: CPU telemetry not 'clean' (other CPU work beside the server, or unknown) in run(s) ${unclean[*]} (HRX-like E415)" | tee -a "$OUT/summary.md"
[ ${#gpuunclean[@]} = 0 ] || echo "EVENTS: UNTRUSTED: GPU clients not 'clean' (other GPU clients beside the server, GPU-CONTAMINATED, or unknown) in run(s) ${gpuunclean[*]} (HRX-like E631)" | tee -a "$OUT/summary.md"
echo "EVENTS_AB_DONE $(date +%T)"
[ $envok = 0 ] && [ ${#unclean[@]} = 0 ] && [ ${#gpuunclean[@]} = 0 ] || exit 2
