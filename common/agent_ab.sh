#!/bin/bash
# Agent benchmark: package B vs reference A on a frozen agent session (agent_bench/freeze.py), ABBA halves, each package served
# with the kit's flags (serve_as.sh, with its identity check that the port serves the requested package).
#   half 1: A then B on turns 0,4,8,...   half 2: B then A on turns 2,6,10,...   fixed mode (byte-identical prompts),
#   card sampling, seed 1 (reproducible). Judge agent-facing changes at agent context lengths (50-130k), not on short prompts.
# Output: OUT/<half>-<pkg>.jsonl (+ .out), OUT/summary.txt: per package decode tok/s, tokens per verify step, prefill tok/s,
# time to first token, and per-turn output-length agreement (equal lengths = content-controlled decode comparison).
# usage: agent_ab.sh KIT PKG_A PKG_B OUT_DIR SESSION.json
set -u -o pipefail
. "$(dirname "$0")/env.sh"; lsk_load "${1:?usage: agent_ab.sh KIT PKG_A PKG_B OUT_DIR SESSION}"
A=$(lsk_pkg "${2:?pkg A}"); B=$(lsk_pkg "${3:?pkg B}"); OUT=$(realpath -m "${4:?out dir}"); SESSION=$(realpath "${5:?session file}") || die "no session"
H=$LSK/common; T=$PYTHON
mkdir -p "$OUT"
trap 'bash $H/serve_as.sh $KIT --restore' EXIT
run() {   # half, package, first turn
  local tag=$1-$(basename "$2")
  bash $H/serve_as.sh "$KIT" "$2" || { echo "agent_ab STOP: serve $2"; exit 1; }
  "$T" $H/agent_bench/replay.py "$SESSION" http://127.0.0.1:$PORT "$KIT_MODEL_ID" --kind loom --mode fixed \
    --turns "$3:" --every 4 --sampling card --seed 1 --label "$(basename "$2")" --log "$OUT/$tag.jsonl" > "$OUT/$tag.out" 2>&1 \
    || { echo "agent_ab STOP: replay $2 ($OUT/$tag.out)"; tail -5 "$OUT/$tag.out"; exit 1; }
  grep -q '"errors": 0' "$OUT/$tag.out" || { echo "agent_ab STOP: replay errors on $2 ($OUT/$tag.out)"; exit 1; }
  echo "$tag done $(date +%T)"
}
run h1 "$A" 0; run h1 "$B" 0; run h2 "$B" 2; run h2 "$A" 2
"$T" - "$OUT" "$(basename "$A")" "$(basename "$B")" <<'PY' | tee "$OUT/summary.txt"
import glob, json, sys
out, a, b = sys.argv[1:4]
def rows(pkg):
    r = {}
    for f in sorted(glob.glob(f"{out}/h*-{pkg}.jsonl")):
        for l in open(f):
            d = json.loads(l)
            if d.get("summary", {}).get("label") != pkg:
                continue
            for x in d["rows"]:
                r[x["turn"]] = x
    return r
R = {p: rows(p) for p in (a, b)}
common = sorted(set(R[a]) & set(R[b]))
same = [t for t in common if R[a][t]["out"] == R[b][t]["out"]]
print(f"fixed mode, every 4th turn, ABBA halves; {len(common)} turns on both, {len(same)} with equal output length")
for p in (a, b):
    rs = [R[p][t] for t in common]
    out_t = sum(x["out"] for x in rs); gen = sum(x["gen_s"] for x in rs)
    acc = sum((x.get("draft_acc") or 0) * (x.get("draft_proposed") or 0) for x in rs)   # draft_acc is a rate (replay.py)
    new = sum(x["new"] for x in rs); ttft = sum(x["ttft"] for x in rs)
    eq = [R[p][t] for t in same]
    eqd = sum(x["out"] for x in eq) / max(sum(x["gen_s"] for x in eq), 1e-9)
    print(f"{p:14s} decode {out_t/gen:5.2f} tok/s (equal-length turns {eqd:5.2f})  tokens/step {out_t/max(out_t-acc,1):4.2f}  "
          f"prefill {new/ttft:6.1f} tok/s  mean TTFT {ttft/len(rs):5.1f} s  ({len(rs)} turns)")
PY
echo "AGENT_AB_DONE $(date +%T)"
