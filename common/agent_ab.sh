#!/bin/bash
# Agent benchmark: package B (candidate) vs reference A on a frozen agent session (agent_bench/freeze.py), ABBA halves, each
# package served with the kit's flags (serve_as.sh, with its identity check that the port serves the requested package).
#   half 1: A then B on turns 0,4,8,...   half 2: B then A on turns 2,6,10,...   fixed mode (byte-identical prompts),
#   card sampling, seed 1 (reproducible). Judge agent-facing changes at agent context lengths (50-130k), not on short prompts.
# Thermal start gate (E368): the first half after an idle gap decoded ~7 % faster than a half run straight after other GPU work,
# whatever the package, and half 1 holds most of the decode time with the reference always first: a bias against every
# candidate. Before each half the bench server is stopped and the chip cools to T_START (edge, C; default 55) or for at most
# COOL_MAX s (default 900); where each half started is logged (OUT/<tag>.start.txt), a HOT start is printed.
# Telemetry (E353): common/gpu_telemetry.py records clock / power / temperature / busy % and CPU beside every half
# (OUT/<tag>.tel.tsv, summary in .tel.txt); a half with other CPU work beside the server is CONTAMINATED (E415).
# Environment (strix-llama.cpp #241: a Mesa update turned a measured speedup into a loss): common/env_block.py records kernel,
# GPU boot parameters, firmware, Mesa, ROCm, Loom, kit commit, both packages, power settings, CPU governor before each half
# (OUT/<tag>.env.txt; the first half's is the bench's, OUT/env.txt); a half whose block differs from it makes the bench UNTRUSTED.
# Output: OUT/<half>-<pkg>.jsonl (+ .out), OUT/summary.txt: per package decode tok/s, tokens per verify step, prefill tok/s,
# time to first token, per-turn output-length agreement (equal lengths = content-controlled decode comparison), the paired
# per-turn time ratios B/A with bootstrap 95 % CIs (all / h1 / h2; E371), the start temperatures and telemetry per half, and last:
#   BENCH: FASTER | TIE | SLOWER | UNTRUSTED | INVALID      exit 0 | 0 | 1 | 2 | 2  (a failed run: exit 2, no BENCH line)
#   Rule (E374): a metric (decode time, prefill time) differs when its paired 95 % CI excludes 1.0 AND the ratio is beyond the
#   A/A floor (AB_FLOOR_DECODE 0.01, AB_FLOOR_PREFILL 0.07: measured on one machine; measure your own with A vs a copy of A, over
#   several server starts). The prefill floor was 0.002 (one start per arm, HRX-like q158); A/As across server starts read the identical
#   copy's prefill 1.061 [1.031, 1.089] and 1.004 (HRX-like E473 / E475: up to +-18 % on long late turns between starts), so a bench
#   prefill verdict within 7 % (above the largest A/A, 6.1 %) is per-start noise; judge prefill kernels on served device time (events_ab) until a multi-start floor exists.
#   SLOWER if any metric is slower; FASTER if one is faster and none slower; TIE otherwise. UNTRUSTED: a half CONTAMINATED by
#   CPU work, or recorded without CPU columns, or the environment changed between halves (or could not be compared).
#   INVALID: fewer than 3 turns with equal output lengths, or a half missing.
# Machine specifics come from the environment (machine env, kit.env, or the caller), never from this file:
#   AB_SERVE   command prefix: AB_SERVE PKG serves PKG, AB_SERVE --restore hands the port back (default: bash common/serve_as.sh KIT)
#   AB_STOP    command that stops the bench server before each cooling wait (default: lsk_stop, which also runs LSK_STOP_HOOK)
#   AB_REPLAY  the replay script (default common/agent_bench/replay.py)
#   AB_CPU_MAX other CPU in cores above which a sample counts toward CONTAMINATED (default 3: this box idles at other-CPU median 1.4, p90 2.4
#              cores (q166, 11-24); E415's real contamination was ~7 cores)
# usage: agent_ab.sh KIT PKG_A PKG_B OUT_DIR SESSION.json
set -u -o pipefail
. "$(dirname "$0")/env.sh"; lsk_load "${1:?usage: agent_ab.sh KIT PKG_A PKG_B OUT_DIR SESSION}"
A=$(lsk_pkg "${2:?pkg A}") || exit 2; B=$(lsk_pkg "${3:?pkg B}") || exit 2; OUT=$(realpath -m "${4:?out dir}")
SESSION=$(realpath -m "${5:?session file}"); [ -f "$SESSION" ] || { echo "agent_ab: no session $SESSION"; exit 2; }
H=$LSK/common; T=$PYTHON
read -ra SERVE <<< "${AB_SERVE:-bash $H/serve_as.sh $KIT}"
read -ra STOP <<< "${AB_STOP:-lsk_stop}"
REPLAY=${AB_REPLAY:-$H/agent_bench/replay.py}
mkdir -p "$OUT"; rm -f "$OUT/summary.txt" "$OUT/env.txt"
trap '"${SERVE[@]}" --restore' EXIT
run() {   # half, package, first turn
  local tag=$1-$(basename "$2") rc spid
  # no quiet-machine wait here: AB_SERVE stops the managed and the previous bench server first, then waits (fail closed)
  "${STOP[@]}" >/dev/null 2>&1
  bash "$H/cool_wait.sh" "$tag" "$OUT/$tag.start.txt" || { echo "agent_ab STOP: thermal start gate ($tag)"; exit 2; }
  # the environment of this half against the bench's (the first half's); the summary judges the diff (exit 0 = unchanged)
  python3 "$H/env_block.py" --kit "$LSK" --pkg "$A" --pkg "$B" > "$OUT/$tag.env.txt" 2> "$OUT/$tag.env.err"
  [ -f "$OUT/env.txt" ] || cp "$OUT/$tag.env.txt" "$OUT/env.txt"
  python3 "$H/env_block.py" --diff "$OUT/env.txt" "$OUT/$tag.env.txt" > "$OUT/$tag.env.diff" 2>&1; echo "exit $?" >> "$OUT/$tag.env.diff"
  # free large physical blocks before this half's server allocates (E471: one package's decode matvecs read 79.2 vs 95.5 ms / step on
  # fresh vs fragmented memory); measured beside the verdict, not judged
  echo "  memory $tag: $(python3 "$H/mem_admit.py" frag 2>&1)" > "$OUT/$tag.mem.txt"
  "${SERVE[@]}" "$2" || { echo "agent_ab STOP: serve $2"; exit 2; }
  spid=$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)   # its CPU is not contamination
  # the replay appends to --log: a half log left by an earlier, stopped bench in this OUT would mix two runs (HRX-like E475: q190's
  # h1 log held a pre-reboot run and the fresh one; its late-turn figure compared one against the other)
  if [ -e "$OUT/$tag.jsonl" ]; then mv "$OUT/$tag.jsonl" "$OUT/$tag.jsonl.prev-$(date +%Y%m%d-%H%M%S)"; echo "agent_ab NOTE: $tag.jsonl existed: moved aside"; fi
  python3 "$H/gpu_telemetry.py" record "$OUT/$tag.tel.tsv" --pid "${spid:-none}" & TEL=$!
  "$T" "$REPLAY" "$SESSION" http://127.0.0.1:$PORT "$KIT_MODEL_ID" --kind loom --mode fixed \
    --turns "$3:" --every 4 --sampling card --seed 1 --label "$(basename "$2")" --log "$OUT/$tag.jsonl" > "$OUT/$tag.out" 2>&1; rc=$?
  kill $TEL 2>/dev/null; python3 "$H/gpu_telemetry.py" summary "$OUT/$tag.tel.tsv" "$tag" --cpu-max "${AB_CPU_MAX:-3}" | tee "$OUT/$tag.tel.txt"
  [ $rc = 0 ] || { echo "agent_ab STOP: replay $2 ($OUT/$tag.out)"; tail -5 "$OUT/$tag.out"; exit 2; }
  grep -q '"errors": 0' "$OUT/$tag.out" || { echo "agent_ab STOP: replay errors on $2 ($OUT/$tag.out)"; exit 2; }
  echo "$tag done $(date +%T)"
}
run h1 "$A" 0; run h1 "$B" 0; run h2 "$B" 2; run h2 "$A" 2
# the summary is a separate file, read when the bench ends (HRX-like E481): edit it, not a heredoc here
"$T" "$H/agent_ab_summary.py" "$OUT" "$(basename "$A")" "$(basename "$B")" "$H" "${AB_FLOOR_DECODE:-0.01}" "${AB_FLOOR_PREFILL:-0.07}" | tee "$OUT/summary.txt"
echo "AGENT_AB_DONE $(date +%T)"
# exit from the verdict line itself, failing closed: anything but FASTER / TIE / SLOWER (a crash, no line) is exit 2
case "$(tail -1 "$OUT/summary.txt" 2>/dev/null)" in "BENCH: FASTER"*|"BENCH: TIE"*) exit 0;; "BENCH: SLOWER"*) exit 1;; *) exit 2;; esac
