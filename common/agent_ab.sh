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
  local tag=$1-$(basename "$2") t0 tc TS=${T_START:-55} rc spid
  # no quiet-machine wait here: AB_SERVE stops the managed and the previous bench server first, then waits (fail closed)
  "${STOP[@]}" >/dev/null 2>&1; t0=$(date +%s)
  while :; do tc=$(python3 "$H/gpu_telemetry.py" temp) || { echo "agent_ab STOP: no GPU temperature for the start gate"; exit 2; }
    [ "$tc" -le "$TS" ] && break
    [ $(( $(date +%s) - t0 )) -ge ${COOL_MAX:-900} ] && { echo "agent_ab NOTE: $tag starts HOT ($tc C > $TS C after ${COOL_MAX:-900} s)"; break; }; sleep 10; done
  echo "$tag start: edge $tc C after $(( $(date +%s) - t0 )) s of cooling" | tee "$OUT/$tag.start.txt"
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
"$T" - "$OUT" "$(basename "$A")" "$(basename "$B")" "$H" "${AB_FLOOR_DECODE:-0.01}" "${AB_FLOOR_PREFILL:-0.07}" <<'PY' | tee "$OUT/summary.txt"
import glob, json, os, random, sys
out, a, b = sys.argv[1:4]
floor = {"gen_s": float(sys.argv[5]), "ttft": float(sys.argv[6])}
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
    if not rs:
        print(f"{p:14s} no turns"); continue
    out_t = sum(x["out"] for x in rs); gen = sum(x["gen_s"] for x in rs)
    acc = sum((x.get("draft_acc") or 0) * (x.get("draft_proposed") or 0) for x in rs)   # draft_acc is a rate (replay.py)
    new = sum(x["new"] for x in rs); ttft = sum(x["ttft"] for x in rs)
    eq = [R[p][t] for t in same]
    eqd = sum(x["out"] for x in eq) / max(sum(x["gen_s"] for x in eq), 1e-9)
    print(f"{p:14s} decode {out_t/gen:5.2f} tok/s (equal-length turns {eqd:5.2f})  tokens/step {out_t/max(out_t-acc,1):4.2f}  "
          f"prefill {new/ttft:6.1f} tok/s  mean TTFT {ttft/len(rs):5.1f} s  ({len(rs)} turns)")
# E371: paired per-turn ratios (candidate / reference time) over the equal-length turns, bootstrap 95 % CI over turns, overall and
# per half (the halves carry the run order: h1 = reference first, h2 = candidate first; a start-temperature bias shows as a
# half-dependent ratio). Seed and percentiles fixed: the figures reproduce the benches judged with them (E374).
def paired(ts, key):
    num = [R[b][t][key] for t in ts]; den = [R[a][t][key] for t in ts]
    if not ts or sum(den) <= 0: return None
    r = sum(num) / sum(den); rng = random.Random(1); bs = []
    for _ in range(2000):
        idx = [rng.randrange(len(ts)) for _ in ts]
        d = sum(den[i] for i in idx); bs.append(sum(num[i] for i in idx) / d if d > 0 else r)
    bs.sort(); return r, bs[50], bs[1949]
halves = {}
for f in sorted(glob.glob(f"{out}/h*-{a}.jsonl")):
    h = f.rsplit("/", 1)[1].split("-", 1)[0]
    for l in open(f):
        d = json.loads(l)
        if d.get("summary", {}).get("label") == a:
            halves[h] = sorted(set(x["turn"] for x in d["rows"]) & set(same))
overall = {}
for key, name in (("gen_s", "decode time"), ("ttft", "prefill time (TTFT)")):
    parts = [("all", same)] + sorted(halves.items())
    txt = []
    for lab, ts in parts:
        v = paired(ts, key)
        if v: txt.append(f"{lab} {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]")
        if lab == "all": overall[key] = v
    print(f"paired {name} {b}/{a} (95 % CI over turns): " + "; ".join(txt))
# informational, not the verdict: median of per-turn decode rates, the ABBA neighbour rule, loop flags (common/checks.py)
sys.path.insert(0, sys.argv[4])
import checks
rate = lambda x: (x["out"] - 1) / x["gen_s"] if x["gen_s"] > 0 and x["out"] > 1 else None
pt = [t for t in common if rate(R[a][t]) and rate(R[b][t])]
if len(pt) >= 3:
    p_, lo, hi = checks.paired_ratio_ci([rate(R[a][t]) for t in pt], [rate(R[b][t]) for t in pt])
    print(f"decode ratio {b}/{a} (median of per-turn rates, {len(pt)} paired turns): {p_:.3f}, 95 % CI [{lo:.3f}, {hi:.3f}]"
          + ("  (CI includes 1: not a measured difference)" if lo <= 1 <= hi else ""))
def half(pkg, h):
    rs = [x for f in glob.glob(f"{out}/{h}-{pkg}.jsonl") for l in open(f) for d in [json.loads(l)] if d.get("summary", {}).get("label") == pkg for x in d["rows"]]
    o_, g_ = sum(x["out"] for x in rs), sum(x["gen_s"] for x in rs)
    return {"decode": o_ / max(g_, 1e-9), "prefill": sum(x["new"] for x in rs) / max(sum(x["ttft"] for x in rs), 1e-9),
            "ttft": sum(x["ttft"] for x in rs) / max(len(rs), 1)}
ok, why = checks.neighbour_verdict({h: {"ref": half(a, h), "cand": half(b, h)} for h in ("h1", "h2")}, "decode", ["prefill", "ttft"],
                                   {"decode": True, "prefill": True, "ttft": False})
print(f"neighbour rule, informational (B beats A in both ABBA halves, prefill / TTFT not > 3 % worse): {'ACCEPT' if ok else 'REJECT'}; " + "; ".join(why))
loops = {p: [t for t in common if (R[p][t].get("loop") or 0) >= 0.5 and R[p][t]["out"] >= 64] for p in (a, b)}
print("loop flags (8-gram repeat share >= 0.5): " + ", ".join(f"{p} {len(v)} turns {v[:6]}" for p, v in loops.items()))
# start temperatures and telemetry beside the verdict, in run order (a faster kernel under a power or thermal limit can lower the clock)
tags = [f"h1-{a}", f"h1-{b}", f"h2-{b}", f"h2-{a}"]
untrusted = []
# one run per half log (E475): a log with two runs is judged by readers that each pick a different one (paired: the last; halves: both)
for tag in tags:
    p = f"{out}/{tag}.jsonl"
    n = sum(1 for l in open(p) if '"rows"' in l) if os.path.exists(p) else 0
    if n != 1: untrusted.append(f"{tag}.jsonl holds {n} runs (one per half)")
for tag in tags:
    for ext in ("start.txt", "tel.txt", "mem.txt"):
        p = f"{out}/{tag}.{ext}"
        txt = open(p).read().rstrip("\n") if os.path.exists(p) else f"  {tag}: no {ext}"
        print(txt)
    tel = open(f"{out}/{tag}.tel.txt").read() if os.path.exists(f"{out}/{tag}.tel.txt") else ""
    if "CONTAMINATED" in tel: untrusted.append(f"{tag} CONTAMINATED by CPU work")
    elif ": clean:" not in tel: untrusted.append(f"{tag} CPU unknown")
# the environment (common/env_block.py, strix-llama.cpp #241): printed once; a half whose block differs from the first half's,
# or could not be compared (no block, an empty one), makes the bench UNTRUSTED with the differing keys
print("environment:")
print("".join("  " + l for l in open(f"{out}/env.txt")).rstrip("\n") if os.path.exists(f"{out}/env.txt") else "  none recorded")
for tag in tags:
    p = f"{out}/{tag}.env.diff"
    d = open(p).read().splitlines() if os.path.exists(p) else ["no env.diff", "exit none"]
    if d[-1:] == ["exit 0"]: continue
    print(f"environment changed before {tag}:" if d[-1] == "exit 1" else f"environment not comparable before {tag}:")
    print("\n".join("  " + l for l in d[:-1]) or "  (no output)")
    keys = [l.split(": ", 1)[0] for l in d[:-1]] if d[-1] == "exit 1" else (d[:-1] or [d[-1]])
    # kit_commit alone is reported, not distrusted (E438): the packages' model.json hashes are the bench's identity, and a commit to the kit
    # during a bench (q166, 11-24) changes no package being benched
    if d[-1] == 'exit 1' and set(keys) <= {'kit_commit'}: continue
    untrusted.append(f"environment {'changed' if d[-1] == 'exit 1' else 'not comparable'} before {tag}: {', '.join(keys)}")
# the verdict (E374): CI excludes 1.0 AND the ratio is beyond the A/A floor
def judge(key):
    v = overall.get(key)
    r, lo, hi = v
    if lo > 1 and r > 1 + floor[key]: return "slower"
    if hi < 1 and r < 1 - floor[key]: return "faster"
    return "tie"
if len(same) < 3 or set(halves) != {"h1", "h2"} or None in overall.values():
    print(f"BENCH: INVALID ({len(same)} equal-length turns, halves {sorted(halves)}: need >= 3 and h1 + h2)")
elif untrusted:
    print("BENCH: UNTRUSTED (" + "; ".join(untrusted) + ")")
else:
    j = {"decode": judge("gen_s"), "prefill": judge("ttft")}
    v = "SLOWER" if "slower" in j.values() else "FASTER" if "faster" in j.values() else "TIE"
    print(f"BENCH: {v} (decode {j['decode']} {overall['gen_s'][0]:.3f} [{overall['gen_s'][1]:.3f}, {overall['gen_s'][2]:.3f}], "
          f"prefill {j['prefill']} {overall['ttft'][0]:.3f} [{overall['ttft'][1]:.3f}, {overall['ttft'][2]:.3f}]; floors "
          f"{floor['gen_s']:.1%} / {floor['ttft']:.1%})")
PY
echo "AGENT_AB_DONE $(date +%T)"
# exit from the verdict line itself, failing closed: anything but FASTER / TIE / SLOWER (a crash, no line) is exit 2
case "$(tail -1 "$OUT/summary.txt" 2>/dev/null)" in "BENCH: FASTER"*|"BENCH: TIE"*) exit 0;; "BENCH: SLOWER"*) exit 1;; *) exit 2;; esac
