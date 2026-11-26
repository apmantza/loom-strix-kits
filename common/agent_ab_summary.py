#!/usr/bin/env python3
"""agent_ab's summary and verdict, run by agent_ab.sh when the last half ends; also the way to re-summarise a finished bench with
today's rules. It lives in its own file so a bench judges with the code on disk when it ends: as a heredoc inside agent_ab.sh it
was frozen when the bench started (bash keeps a running script's old copy), and q186 judged prefill against a floor replaced
mid-run (HRX-like E481: printed SLOWER on the old 0.2 % floor, TIE on the 7 % one).
usage: agent_ab_summary.py OUT A_NAME B_NAME KIT_COMMON FLOOR_DECODE FLOOR_PREFILL     (agent_ab.sh: floors 0.01 / 0.07)"""
import glob, json, os, random, re, sys
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
    tel = open(f"{out}/{tag}.tel.txt").read().splitlines() if os.path.exists(f"{out}/{tag}.tel.txt") else []
    # each flag from its own line (the gpu line also says 'clean:' / 'CONTAMINATED'): only 'clean' is trusted
    cpu = next((l for l in tel if l.startswith(f"  cpu {tag}: ")), "")
    if cpu.startswith(f"  cpu {tag}: CONTAMINATED"): untrusted.append(f"{tag} CONTAMINATED by CPU work")
    elif not cpu.startswith(f"  cpu {tag}: clean: "): untrusted.append(f"{tag} CPU unknown")
    # other GPU clients (HRX-like E631). gpu_telemetry prints the gpu line before the cpu line, so a tel.txt with a cpu line and no
    # gpu line was written before 11-26: reported, not distrusted, so old benches still re-analyse; a gpu line that is not clean
    # (GPU-CONTAMINATED, or unknown: no client record) makes the bench UNTRUSTED
    gpu = next((l for l in tel if l.startswith(f"  gpu {tag}: ")), None)
    if gpu is None: print(f"  gpu {tag}: not recorded (tel.txt before 11-26)")
    elif gpu.startswith(f"  gpu {tag}: GPU-CONTAMINATED"): untrusted.append(f"{tag} GPU-CONTAMINATED by other GPU clients")
    elif not gpu.startswith(f"  gpu {tag}: clean: "): untrusted.append(f"{tag} GPU clients unknown")
# clock and power per arm and half, side by side (R5, HRX-like E450 / E475): at a power cap a candidate that holds a lower clock at the
# same power spends more energy per cycle, so its served loss may be a power cost rather than a slower kernel. Informational, never a
# verdict: A/As of one package read 2.19 vs 2.30 GHz between server starts (E475), so a delta inside ~5 % says nothing alone.
def telm(tag):
    p = f"{out}/{tag}.tel.txt"
    m = re.search(r"while busy: power ([0-9.]+) .*?clock ([0-9.]+) ", open(p).read()) if os.path.exists(p) else None
    return (float(m.group(1)), float(m.group(2))) if m else None
for h in ("h1", "h2"):
    ta, tb = telm(f"{h}-{a}"), telm(f"{h}-{b}")
    if ta and tb:
        print(f"clock / power {h}: {a} {ta[1]:.2f} GHz {ta[0]:.1f} W, {b} {tb[1]:.2f} GHz {tb[0]:.1f} W "
              f"(clock {b}/{a} {tb[1] / ta[1]:.3f}; informational, A/A spread up to ~5 %, E475)")
    else:
        print(f"clock / power {h}: no telemetry median for {'both arms' if not (ta or tb) else (a if not ta else b)}")
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
