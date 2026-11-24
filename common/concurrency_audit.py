#!/usr/bin/env python3
"""Audit a concurrent serving cell: N agent replays run at once against one server (common/agent_bench/replay.py rows).

Reports, per stream: decode tok/s split by what the OTHER streams were doing at the time (prefilling / decoding / idle), the
share of decode time in each phase, TTFT p50 / p95, and re-prefill events (a turn whose prefill covers ~its whole history:
eviction, or a continuation miss when the previous reply hit its cap and the client's history changed). Per cell: aggregate
decode tok/s over the cell's wall time; with --solo (the same streams run one at a time), one stream alone against the N-stream
total, exactness (each stream's replies == its solo replies, by reply hash; a planted difference must be caught first) and
work-normalised throughput (throughput(): tokens per wall second per arm, prefill / decode seconds per token, the aggregate
while every stream decodes, and the cell's wall for the solo arm's token counts, priced per turn at the solo costs, with a
bootstrap CI; E449: an own-mode cell does other work than its solo arm, so the raw wall ratio is not a throughput figure);
with --ref (another package's cell, same streams) the paired per-turn decode / prefill ratio against --floor (the kit's A/A noise
floor: an argument, measured per kit).

Phase split (E49's method): each decode turn [first token, end] is classed by what the other streams did during it (any
overlap with another's prefill [send, first token) -> 'other prefill', else with another's decode -> 'other decode', else
'idle'), and the class's tok/s is its tokens over its decode time; 'sliced' is the fraction of decode time that actually
overlapped another prefill. A turn-level class is an interval average: per-token gaps need streamed stamps, which
common/decode_while_prefill.py records (both use segments / split_interval / gap_stats below). Rows from replay.py with t_send / t_first /
t_end share one epoch timeline; older rows lack them and the timeline is rebuilt from cumulative turn walls (streams start
together, turns back to back), which the report says.

Model-agnostic: nothing here knows the model. For a MoE kit, more rows activate more experts per step, so the per-stream cost
of concurrency is not flat across N: run the cell at each N you care about.

usage: concurrency_audit.py CELL.jsonl [--streams L1,L2] [--solo SOLO.jsonl] [--ref REF.jsonl --floor F] [--json OUT]
       exit 0 OK, 1 FAIL, 2 usage / invalid input / UNKNOWN (exactness not checkable) / CONTROL FAILED
"""
import argparse
import copy
import json
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checks  # noqa: E402  paired_ratio_ci

PHASES = ("prefill", "decode", "idle")


def load(path, labels=None):
    """{label: {"summary": ..., "rows": [...]}} from a replay.py --log file (one record per stream run)"""
    out = {}
    for n, line in enumerate(open(path), 1):
        if not line.strip():
            continue
        d = json.loads(line)
        lab = d["summary"]["label"]
        if labels and lab not in labels:
            continue
        if lab in out:
            raise ValueError(f"{path}:{n}: label {lab!r} twice (one record per stream; pass --streams)")
        out[lab] = d
    if labels:
        missing = [x for x in labels if x not in out]
        if missing:
            raise ValueError(f"{path}: no record for {missing}")
    return out


def timeline(rows):
    """[(row, send, first, end)] on the epoch clock when every row carries it, else rebuilt from walls (start 0); plus a flag"""
    ok = [r for r in rows if not r.get("error")]
    if ok and all(all(k in r for k in ("t_send", "t_first", "t_end")) for r in ok):
        return [(r, r["t_send"], r["t_first"], r["t_end"]) for r in ok], False
    t, res = 0.0, []
    for r in rows:
        wall = r.get("wall", (r.get("ttft") or 0) + (r.get("gen_s") or 0))
        if not r.get("error"):
            res.append((r, t, t + r["ttft"], t + wall))
        t += wall
    return res, True


def segments(intervals):
    """intervals: [(lo, hi, phase)] of the other streams -> sorted disjoint [(lo, hi, phase)]: prefill beats decode beats idle"""
    cuts = sorted({x for lo, hi, _ in intervals for x in (lo, hi)})
    rank = {p: i for i, p in enumerate(PHASES)}
    seg = []
    for lo, hi in zip(cuts, cuts[1:]):
        mid = (lo + hi) / 2
        ph = min((p for a, b, p in intervals if a <= mid < b), key=rank.get, default="idle")
        if seg and seg[-1][2] == ph and seg[-1][1] == lo:
            seg[-1] = (seg[-1][0], hi, ph)
        else:
            seg.append((lo, hi, ph))
    return seg


def phase_at(t, seg, default="idle"):
    return next((p for a, b, p in seg if a <= t < b), default)


def split_interval(lo, hi, seg, default="idle"):
    """seconds of [lo, hi) in each phase of seg (time outside every segment counts as default)"""
    out, covered = {}, 0.0
    for a, b, p in seg:
        d = min(b, hi) - max(a, lo)
        if d > 0:
            out[p] = out.get(p, 0.0) + d
            covered += d
    if hi - lo - covered > 1e-9:
        out[default] = out.get(default, 0.0) + (hi - lo - covered)
    return out


def gap_stats(stamps, seg, default="idle"):
    """per phase of seg: chunk gaps between consecutive token stamps, each gap labelled by the phase at its end (the L19 /
    E59 method); {phase: {chunks, tok_s, median_gap_ms, p90_gap_ms, max_gap_ms}}"""
    by = {}
    for k in range(1, len(stamps)):
        by.setdefault(phase_at(stamps[k], seg, default), []).append(stamps[k] - stamps[k - 1])
    res = {}
    for p, g in by.items():
        span = sum(g)
        res[p] = {"chunks": len(g), "tok_s": round(len(g) / span, 2) if span > 0 else None,
                  "median_gap_ms": round(1000 * statistics.median(g), 1), "p90_gap_ms": round(1000 * sorted(g)[int(0.9 * (len(g) - 1))], 1),
                  "max_gap_ms": round(1000 * max(g), 1)}
    return res


def pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, math.ceil(q / 100 * len(s)) - 1))]


def reprefills(rows, fixed=False):
    """turns after a contiguous predecessor whose new tokens are >= 90 % of the prompt: the retained history was not used.
    fixed: a fixed-mode replay (replay.py --mode fixed) re-sends the RECORDED reply, never the engine's, so every turn misses
    its retained tip and falls back to whatever prefix the cache captured (E105): a whole re-prefill is not attributable"""
    ev, prev = [], None
    for r in rows:
        if r.get("error"):
            prev = None
            continue
        if prev is not None and prev["turn"] == r["turn"] - 1 and r["prompt"] > 0 and r["new"] >= 0.9 * r["prompt"]:
            miss = prev.get("cap") is not None and prev["out"] >= prev["cap"]
            ev.append({"turn": r["turn"], "prompt": r["prompt"], "new": r["new"], "ttft": r["ttft"],
                       "kind": "fixed-mode miss (eviction not attributable)" if fixed else
                       "continuation miss (previous reply hit its cap)" if miss else "eviction"})
        prev = r
    return ev


def stream_stats(rows):
    ok = [r for r in rows if not r.get("error")]
    dec_t = sum(r["gen_s"] for r in ok)
    dec_n = sum(max(0, r["out"] - 1) for r in ok)
    pre_t = sum(r["ttft"] for r in ok)
    pre_n = sum(r["new"] for r in ok)
    return {"turns": len(ok), "errors": len(rows) - len(ok), "decode_tok_s": round(dec_n / dec_t, 2) if dec_t else None,
            "prefill_tok_s": round(pre_n / pre_t, 1) if pre_t else None, "decode_tokens": dec_n,
            "ttft_p50": pct([r["ttft"] for r in ok], 50), "ttft_p95": pct([r["ttft"] for r in ok], 95)}


def cell(records):
    """the concurrent cell: per stream stats, phase split, re-prefills; the cell's wall and aggregate decode rate"""
    tl, rebuilt = {}, False
    for lab, d in records.items():
        tl[lab], rb = timeline(d["rows"])
        rebuilt |= rb
    res = {"streams": {}, "rebuilt_timeline": rebuilt}
    for lab, d in records.items():
        others = [(s, f, "prefill") for o in tl if o != lab for _, s, f, _ in tl[o]] + \
                 [(f, e, "decode") for o in tl if o != lab for _, _, f, e in tl[o]]
        seg = segments(others)
        # E49's method: a decode turn is classed by the most disruptive phase the others were in at any time during it
        # (prefill beats decode beats idle); its tokens / its decode time go to that class. The sliced share is the
        # fraction of decode time that actually overlapped another stream's prefill (the interval cut, no rate assumption).
        sec, tok, overlap, total = {p: 0.0 for p in PHASES}, {p: 0.0 for p in PHASES}, 0.0, 0.0
        for r, s, f, e in tl[lab]:
            if e <= f or r["out"] <= 1:
                continue
            parts = split_interval(f, e, seg)
            cls = min(parts, key=PHASES.index)
            sec[cls] += e - f
            tok[cls] += r["out"] - 1
            overlap += parts.get("prefill", 0.0)
            total += e - f
        st = stream_stats(d["rows"])
        st["by_phase"] = {p: {"tok_s": round(tok[p] / sec[p], 2) if sec[p] else None, "time_share": round(sec[p] / total, 3) if total else None}
                          for p in PHASES}
        st["sliced_prefill_overlap"] = round(overlap / total, 3) if total else None
        st["reprefills"] = reprefills(d["rows"], fixed=d["summary"].get("mode") == "fixed")
        st["span"] = (min(s for _, s, _, _ in tl[lab]), max(e for _, _, _, e in tl[lab])) if tl[lab] else None
        res["streams"][lab] = st
    spans = [st["span"] for st in res["streams"].values() if st["span"]]
    res["overlap"] = bool(spans) and max(a for a, _ in spans) < min(b for _, b in spans)
    wall = (max(b for _, b in spans) - min(a for a, _ in spans)) if spans else 0
    res["wall_s"] = round(wall, 1)
    res["aggregate_decode_tok_s"] = round(sum(st["decode_tokens"] for st in res["streams"].values()) / wall, 2) if wall else None
    # distinct prompts: a stream whose FIRST turn reuses most of its prompt shares a cached prefix with another (no salt)
    res["shared_prefix"] = [lab for lab, d in records.items()
                            if d["rows"] and not d["rows"][0].get("error") and d["rows"][0]["cached"] > 0.5 * d["rows"][0]["prompt"]]
    return res


def exactness(conc, solo):
    """{label: {"compared", "mismatch", "first"}} by reply hash per turn; None when a row lacks its hash"""
    res = {}
    for lab, d in conc.items():
        s = {r["turn"]: r for r in solo[lab]["rows"] if not r.get("error")}
        c = [r for r in d["rows"] if not r.get("error")]
        if any("out_sha" not in r for r in c) or any("out_sha" not in r for r in s.values()):
            return None
        both = [r for r in c if r["turn"] in s]
        bad = [r["turn"] for r in both if r["out_sha"] != s[r["turn"]]["out_sha"]]
        res[lab] = {"compared": len(both), "mismatch": len(bad), "first": bad[0] if bad else None,
                    "missing": len(c) - len(both)}
    return res


def planted_control(conc, solo):
    """the comparator must report a planted one-hash difference before its pass is trusted"""
    lab = next(iter(conc))
    sp = copy.deepcopy(solo)
    rows = [r for r in sp[lab]["rows"] if not r.get("error")]
    if not rows:
        return False
    r = rows[len(rows) // 2]
    r["out_sha"] = ("0" if r["out_sha"][:1] != "0" else "1") + r["out_sha"][1:]
    e = exactness(conc, sp)
    return bool(e) and e[lab]["mismatch"] >= 1 if any(x["turn"] == r["turn"] for x in conc[lab]["rows"]) else False


def ref_compare(cand, ref, floor):
    """paired per-turn ratios cand / ref of decode and prefill tok/s over (stream, turn) pairs present in both"""
    out = {}
    for metric in ("decode_tps", "prefill_tps"):
        a, b = [], []
        for lab, d in cand.items():
            rr = {r["turn"]: r for r in ref.get(lab, {}).get("rows", []) if not r.get("error")}
            for r in d["rows"]:
                x = rr.get(r["turn"])
                if x and not r.get("error") and r.get(metric) and x.get(metric):
                    a.append(x[metric])
                    b.append(r[metric])
        if len(a) < 3:
            out[metric] = None
            continue
        p, lo, hi = checks.paired_ratio_ci(a, b)
        out[metric] = {"pairs": len(a), "ratio": round(p, 4), "ci": [round(lo, 4), round(hi, 4)],
                       "regression": hi < 1.0 and p < 1.0 - floor}
    return out


def apportion(intervals):
    """[(lo, hi, key)] -> {key: seconds}: every instant of the arm's wall split evenly among the turns in flight then (time with
    none in flight is left out); a turn's share of a wall the streams shared, so per-turn units can be resampled"""
    cuts = sorted({x for lo, hi, _ in intervals for x in (lo, hi)})
    share = {k: 0.0 for _, _, k in intervals}
    for lo, hi in zip(cuts, cuts[1:]):
        live = [k for a, b, k in intervals if a < hi and b > lo]
        for k in live:
            share[k] += (hi - lo) / len(live)
    return share


def all_decoding(tl):
    """(seconds, tokens) while EVERY stream of the cell decodes: each turn's tokens spread evenly over its decode interval"""
    iv = [(f, e, (r["out"] - 1) / (e - f), lab) for lab, rows in tl.items() for r, _, f, e in rows if e > f and r["out"] > 1]
    cuts = sorted({x for lo, hi, _, _ in iv for x in (lo, hi)})
    sec = tok = 0.0
    for lo, hi in zip(cuts, cuts[1:]):
        live = [(rate, lab) for a, b, rate, lab in iv if a < hi and b > lo]
        if {lab for _, lab in live} == set(tl):
            sec += hi - lo
            tok += sum(rate for rate, _ in live) * (hi - lo)
    return sec, tok


def throughput(conc, solo, iters=2000, seed=1, min_new=64, min_dec=8):
    """Work-normalised throughput of the concurrent cell against the same streams run one after another (E449: the arms of an
    own-mode cell did different work, so the raw wall ratio was mostly less output). Paired units = (stream, turn) present in
    both. A unit's work is in solo-seconds: its tokens priced at what that turn cost alone (prefill s per new token, decode s
    per output token after the first; a turn with < min_new new / < min_dec decode tokens alone uses its fixed TTFT / the solo
    arm's pooled decode cost). A unit's wall is its share of its arm's wall (apportion; idle gaps spread pro rata). Speedup =
    (work / wall of the cell) / (work / wall of the solo arm), with a bootstrap 95 % interval over the units."""
    ctl = {lab: timeline(d["rows"])[0] for lab, d in conc.items()}
    stl = {lab: timeline(d["rows"])[0] for lab, d in solo.items()}
    c_span = [x for rows in ctl.values() for x in rows]
    c_wall = max(e for _, _, _, e in c_span) - min(s for _, s, _, _ in c_span)
    s_wall = sum(max(e for _, _, _, e in rows) - min(s for _, s, _, _ in rows) for rows in stl.values() if rows)
    csh = apportion([(s, e, (lab, r["turn"])) for lab, rows in ctl.items() for r, s, _, e in rows])
    ssh = {}
    for lab, rows in stl.items():   # one stream at a time: each turn owns its own wall
        ssh.update(apportion([(s, e, (lab, r["turn"])) for r, s, _, e in rows]))
    cf, sf = c_wall / sum(csh.values()), s_wall / sum(ssh.values())
    s_ok = [r for d in solo.values() for r in d["rows"] if not r.get("error")]
    pooled_dec = sum(r["gen_s"] for r in s_ok) / max(1, sum(max(0, r["out"] - 1) for r in s_ok))
    units, unpaired = [], 0
    for lab, rows in ctl.items():
        alone = {r["turn"]: r for r, _, _, _ in stl.get(lab, [])}
        for r, _, _, _ in rows:
            x = alone.get(r["turn"])
            if x is None:
                unpaired += 1
                continue

            def work(y, x=x):
                pre = x["ttft"] * y["new"] / x["new"] if x["new"] >= min_new else x["ttft"]
                dx, dy = max(0, x["out"] - 1), max(0, y["out"] - 1)
                return pre + (x["gen_s"] * dy / dx if dx >= min_dec else pooled_dec * dy)
            units.append({"label": lab, "turn": r["turn"], "c_work": work(r), "s_work": work(x),
                          "c_wall": csh[(lab, r["turn"])] * cf, "s_wall": ssh[(lab, x["turn"])] * sf,
                          "c": r, "s": x})
    if not units:
        return None

    def stat(ix):
        cw, cl = sum(units[i]["c_work"] for i in ix), sum(units[i]["c_wall"] for i in ix)
        sw, sl = sum(units[i]["s_work"] for i in ix), sum(units[i]["s_wall"] for i in ix)
        return (cw / cl) / (sw / sl)
    idx = list(range(len(units)))
    rng = random.Random(seed)
    boots = sorted(stat([rng.choice(idx) for _ in idx]) for _ in range(iters))

    def arm(side, wall):
        rows = [u[side] for u in units]
        pre_n, out_n = sum(r["new"] for r in rows), sum(r["out"] for r in rows)
        dec_n = sum(max(0, r["out"] - 1) for r in rows)
        ttft = [r["ttft"] for r in rows]
        return {"wall_s": round(wall, 1), "prefill_tokens": pre_n, "output_tokens": out_n,
                "prefill_tok_per_wall_s": round(pre_n / wall, 1), "output_tok_per_wall_s": round(out_n / wall, 2),
                "prefill_ms_per_tok": round(1000 * sum(r["ttft"] for r in rows) / pre_n, 3) if pre_n else None,
                "decode_ms_per_tok": round(1000 * sum(r["gen_s"] for r in rows) / dec_n, 2) if dec_n else None,
                "ttft_p50": pct(ttft, 50), "ttft_p95": pct(ttft, 95),
                "work_solo_s": round(sum(u[side[0] + "_work"] for u in units), 1)}
    c_l, s_l = sum(u["c_wall"] for u in units), sum(u["s_wall"] for u in units)
    res = {"paired_turns": len(units), "unpaired_turns": unpaired, "concurrent": arm("c", c_l), "solo": arm("s", s_l)}
    both_s, both_t = all_decoding(ctl)
    res["all_decoding"] = {"seconds": round(both_s, 1), "wall_share": round(both_s / c_wall, 3) if c_wall else None,
                           "aggregate_tok_s": round(both_t / both_s, 2) if both_s else None}
    cw, sw = res["concurrent"]["work_solo_s"], res["solo"]["work_solo_s"]
    res["same_work_concurrent_wall_s"] = round(c_l * sw / cw, 1) if cw else None
    res["raw_wall_ratio"] = round(s_l / c_l, 4)
    res["speedup"] = round(stat(idx), 4)
    res["speedup_ci"] = [round(boots[int(0.025 * iters)], 4), round(boots[int(0.975 * iters) - 1], 4)]
    return res


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cell")
    ap.add_argument("--streams", help="comma-separated labels of the concurrent streams (default: every record in CELL)")
    ap.add_argument("--solo", help="the same streams (same labels) run one at a time: share and exactness")
    ap.add_argument("--ref", help="another package's concurrent cell (same labels): paired per-turn ratios")
    ap.add_argument("--floor", type=float, help="the kit's A/A noise floor as a fraction (e.g. 0.01), required with --ref")
    ap.add_argument("--json", help="write the full result here")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    if a.ref and a.floor is None:
        print("concurrency_audit: --ref needs --floor (the kit's measured A/A noise floor; no default)")
        return 2
    labels = [x for x in (a.streams or "").split(",") if x] or None
    try:
        conc = load(a.cell, labels)
        solo = load(a.solo, list(conc)) if a.solo else None
        ref = load(a.ref, list(conc)) if a.ref else None
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as e:
        print(f"concurrency_audit: {e}")
        print("CONCURRENCY: UNKNOWN (invalid input)")
        return 2
    if len(conc) < 2:
        print(f"concurrency_audit: {len(conc)} stream(s) in {a.cell}: a cell needs two or more")
        return 2
    res = cell(conc)
    fail, unknown = [], []
    print(f"cell: {len(conc)} streams, wall {res['wall_s']} s, aggregate decode {fmt(res['aggregate_decode_tok_s'], 2)} tok/s"
          + (" (timeline rebuilt from turn walls: rows lack t_send)" if res["rebuilt_timeline"] else ""))
    print(f"{'stream':<24}{'turns':>6}{'decode':>8}{'| other prefill':>16}{'decode':>8}{'idle':>8}{'| turn share p/d/i':>20}"
          f"{'sliced':>8}{'prefill':>9}{'ttft p50':>9}{'p95':>7}")
    for lab, st in res["streams"].items():
        bp = st["by_phase"]
        print(f"{lab:<24}{st['turns']:>6}{fmt(st['decode_tok_s']):>8}{fmt(bp['prefill']['tok_s']):>16}{fmt(bp['decode']['tok_s']):>8}"
              f"{fmt(bp['idle']['tok_s']):>8}{'/'.join(fmt(bp[p]['time_share'], 2) for p in PHASES):>20}"
              f"{fmt(st['sliced_prefill_overlap'], 2):>8}{fmt(st['prefill_tok_s'], 0):>9}{fmt(st['ttft_p50']):>9}{fmt(st['ttft_p95']):>7}")
        for ev in st["reprefills"]:
            print(f"  re-prefill turn {ev['turn']}: {ev['new']} of {ev['prompt']} tokens new, TTFT {ev['ttft']} s: {ev['kind']}")
            if ev["kind"] == "eviction":
                fail.append(f"{lab} turn {ev['turn']} re-prefilled its history (eviction)")
            elif ev["kind"].startswith("fixed-mode"):
                unknown.append(f"{lab} turn {ev['turn']} re-prefilled its history in fixed mode (eviction not attributable)")
        if st["errors"]:
            fail.append(f"{lab}: {st['errors']} failed turns")
    if not res["overlap"]:
        fail.append("streams never ran at the same time (no overlap): not a concurrent cell")
    if res["shared_prefix"]:
        fail.append(f"{res['shared_prefix']} reuse a cached prefix on their first turn: prompts not distinct (salt each stream)")
    if solo:
        ss = {lab: stream_stats(d["rows"]) for lab, d in solo.items()}
        alone = statistics.mean(s["decode_tok_s"] for s in ss.values() if s["decode_tok_s"])
        solo_wall = sum(sum(r.get("wall", 0) for r in d["rows"]) for d in solo.values())
        res["solo"] = {"streams": ss, "alone_decode_tok_s": round(alone, 2), "solo_wall_sum_s": round(solo_wall, 1)}
        print(f"share: one stream alone {alone:.2f} tok/s, {len(conc)}-stream total {fmt(res['aggregate_decode_tok_s'], 2)} tok/s "
              f"(total / alone {fmt(res['aggregate_decode_tok_s'] / alone if res['aggregate_decode_tok_s'] else None, 2)}); "
              f"cell wall {res['wall_s']} s vs the solo runs back to back {solo_wall:.1f} s")
        for lab, s in ss.items():
            c = res["streams"][lab]
            print(f"  {lab}: decode {fmt(c['decode_tok_s'])} concurrent vs {fmt(s['decode_tok_s'])} alone, TTFT p50 {fmt(c['ttft_p50'])} vs "
                  f"{fmt(s['ttft_p50'])} s, p95 {fmt(c['ttft_p95'])} vs {fmt(s['ttft_p95'])} s")
        ex = exactness(conc, solo)
        if ex is None:
            unknown.append("exactness (rows lack out_sha: replay.py older than the hash)")
        elif not planted_control(conc, solo):
            print("CONCURRENCY: CONTROL FAILED (a planted reply-hash difference was not reported)")
            return 2
        else:
            res["exactness"] = ex
            for lab, e in ex.items():
                print(f"  exactness {lab}: {e['compared'] - e['mismatch']}/{e['compared']} turns identical to solo"
                      + (f", first difference turn {e['first']}" if e["first"] is not None else "")
                      + (f", {e['missing']} turns without a solo row" if e["missing"] else ""))
                if e["mismatch"]:
                    fail.append(f"{lab}: {e['mismatch']} turns differ from solo (first {e['first']})")
                if e["missing"]:
                    unknown.append(f"exactness of {e['missing']} {lab} turns (no solo row)")
        tp = throughput(conc, solo)
        if tp is None:
            unknown.append("work-normalised throughput (no (stream, turn) in both the cell and --solo)")
        else:
            res["throughput"] = tp
            c, s = tp["concurrent"], tp["solo"]
            print(f"work ({tp['paired_turns']} paired turns, {tp['unpaired_turns']} unpaired): prefill tokens {c['prefill_tokens']} concurrent vs "
                  f"{s['prefill_tokens']} solo ({c['prefill_tokens'] / max(1, s['prefill_tokens']):.3f}), output {c['output_tokens']} vs "
                  f"{s['output_tokens']} ({c['output_tokens'] / max(1, s['output_tokens']):.3f})")
            print(f"  {'arm':<14}{'wall s':>8}{'prefill tok/wall-s':>20}{'out tok/wall-s':>16}{'prefill ms/tok':>16}{'decode ms/tok':>15}"
                  f"{'ttft p50':>10}{'p95':>7}{'work solo-s':>13}")
            for name, x in (("concurrent", c), ("one by one", s)):
                print(f"  {name:<14}{x['wall_s']:>8.1f}{x['prefill_tok_per_wall_s']:>20.1f}{x['output_tok_per_wall_s']:>16.2f}"
                      f"{fmt(x['prefill_ms_per_tok'], 2):>16}{fmt(x['decode_ms_per_tok'], 1):>15}{fmt(x['ttft_p50']):>10}{fmt(x['ttft_p95']):>7}"
                      f"{x['work_solo_s']:>13.1f}")
            ad = tp["all_decoding"]
            print(f"  all streams decoding: {fmt(ad['aggregate_tok_s'], 2)} tok/s aggregate over {ad['seconds']} s "
                  f"({fmt(ad['wall_share'], 3)} of the cell wall)")
            print(f"  same work: the cell would need {tp['same_work_concurrent_wall_s']} s for the solo arm's tokens vs {s['wall_s']} s one by one: "
                  f"work-normalised speedup {tp['speedup']} (95 % CI {tp['speedup_ci'][0]}-{tp['speedup_ci'][1]}, bootstrap over "
                  f"{tp['paired_turns']} turns); raw wall ratio {tp['raw_wall_ratio']}")
            if tp["unpaired_turns"]:
                unknown.append(f"throughput: {tp['unpaired_turns']} cell turns without a solo row")
    else:
        unknown.append("exactness and share (no --solo)")
    if ref:
        rc = ref_compare(conc, ref, a.floor)
        res["ref"] = rc
        for m, v in rc.items():
            if v is None:
                unknown.append(f"{m} against --ref (fewer than 3 paired turns)")
                continue
            print(f"vs ref {m}: ratio {v['ratio']} (95 % CI {v['ci'][0]}-{v['ci'][1]}, {v['pairs']} paired turns, floor {a.floor})")
            if v["regression"]:
                fail.append(f"{m} {v['ratio']} vs ref: CI below 1 and beyond the floor")
    res["fail"], res["unknown"] = fail, unknown
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    if fail:
        print("CONCURRENCY: FAIL (" + "; ".join(fail) + ")")
        return 1
    if unknown:
        print("CONCURRENCY: UNKNOWN (" + "; ".join(unknown) + ")")
        return 2
    print("CONCURRENCY: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
