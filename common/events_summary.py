#!/usr/bin/env python3
"""Lanes vs hardware floors from served per-dispatch events (every lane against its limit).

Each RUN is a directory holding events.jsonl (iree-profile dispatch --dispatch_events --format=jsonl of a loom-serve
--device_profiling_output bundle; events_ab.sh writes it) and one *.req.json (profile_longctx_req.py: the served prompt
size). Dispatches go to the first lane whose regex matches the kernel name (MODEL.json "lanes"); verify steps are the
distinct submissions that ran the verify-attention lane. Several RUNs print side by side (an ABBA's four runs).

usage: events_summary.py MODEL.json RUN [RUN ...] [--top 12] [--split KERNEL[:CUT_MS]]
  --split prints a kernel's dispatch-time clusters (E143: vsplit's aligned 1.62 ms vs straddling 3.01 ms) per run: below
  and above CUT_MS (default: midway between the run's 10th and 90th percentile), count and median each.
"""
import argparse
import glob
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ceiling import floor  # noqa: E402


def load(run):
    path = os.path.join(run, "events.jsonl")
    ev = []
    for ln in open(path):
        d = json.loads(ln)
        if d.get("type") == "dispatch_event" and d.get("valid", True):
            ev.append(d)
    if not ev:
        raise SystemExit(f"events_summary: no valid dispatch events in {path}")
    reqs = glob.glob(os.path.join(run, "*.req.json"))
    if len(reqs) != 1:
        raise SystemExit(f"events_summary: want exactly one *.req.json in {run}, found {len(reqs)}")
    try:
        prompt = int(json.loads(open(reqs[0]).read().splitlines()[-1])["requests"][-1]["prompt_tokens"])
    except (ValueError, KeyError, IndexError, TypeError) as e:   # the served size decides every floor: no guessing
        raise SystemExit(f"events_summary: cannot read prompt_tokens from {reqs[0]}: {e}")
    return ev, prompt


def classify(ev, d):
    """(prefill submission ids, lane_of(event)): the one classification events_summary and lane_breakdown share.
    A submission (one queue submission: an epoch) that runs a prefill marker kernel, or whose g_token_info covers more than
    prefill_min_tokens tokens, is a prefill epoch; everything in it is prefill work, the drafter and matvecs included (the
    drafter consumes the prompt: per name, the first real run read the drafter at 3.6x its per-step cost). A lane with a
    "submission_regex" owns every dispatch of a submission of its phase that runs a kernel matching it: kernel names cannot
    split call sites that share a kernel (E163: the drafter's propose pass runs g_mv_q4_k_* like the target, and the target
    runs g_mv_q8_0_* like the drafter; by name, 3.6 ms/step moved between the two lanes). Otherwise a dispatch goes to the
    first lane of its phase whose regex matches; each phase ends with a catch-all lane."""
    lanes = d["lanes"]
    mark = re.compile(d["prefill_marker"])
    big = d.get("prefill_min_tokens", 64)
    pre = {e["submission_id"] for e in ev if mark.search(e["key"])
           or (e["key"] == "g_token_info" and (e.get("workgroup_count") or [0])[0] > big)}
    rx = [(ln, re.compile(ln["regex"])) for ln in lanes]
    owners = [(ln, re.compile(ln["submission_regex"])) for ln in lanes if ln.get("submission_regex")]
    owned = {}
    for e in ev:
        ph = "prefill" if e["submission_id"] in pre else "decode"
        for ln, r in owners:
            if ln["phase"] == ph and r.search(e["key"]):
                owned.setdefault(e["submission_id"], ln["lane"])
    memo = {}

    def lane_of(e):
        if e["submission_id"] in owned:
            return owned[e["submission_id"]]
        ph = "prefill" if e["submission_id"] in pre else "decode"
        key = (e["key"], ph)
        if key not in memo:
            memo[key] = next((ln["lane"] for ln, r in rx if ln["phase"] == ph and r.search(e["key"])), None)
            if memo[key] is None:
                raise SystemExit(f"events_summary: {ph} kernel {e['key']} matches no lane (each phase needs a catch-all)")
        return memo[key]
    return pre, lane_of


def summarize(ev, prompt, d):
    m, hw, lanes = d["model"], d["hardware"], d["lanes"]
    pre, lane_of = classify(ev, d)
    per = {ln["lane"]: [0, 0.0] for ln in lanes}
    for e in ev:
        p = per[lane_of(e)]
        p[0] += 1
        p[1] += e["duration_ns"] / 1e9
    vlane = next(ln["lane"] for ln in lanes if ln["floor"] == "decode_kv")
    steps = len({e["submission_id"] for e in ev if e["submission_id"] not in pre and lane_of(e) == vlane})
    rows = []
    for ln in lanes:
        n, s = per[ln["lane"]]
        f = floor(ln["floor"], m, hw, context=prompt, prompt=prompt)
        if ln["phase"] == "decode":
            meas = s * 1e3 / steps if steps else float("nan")
            unit = "ms/step"
        else:
            meas, unit = s, "s"
        rows.append((ln["lane"], n, meas, unit, f, (f / meas * 100) if f is not None and meas else None))
    return steps, sum(e["duration_ns"] for e in ev) / 1e9, rows


def clusters(ev, kernel, cut):
    xs = sorted(e["duration_ns"] / 1e6 for e in ev if e["key"] == kernel)
    if not xs:
        return f"{kernel}: no dispatches"
    c = cut if cut is not None else (xs[len(xs) // 10] + xs[9 * len(xs) // 10]) / 2   # p10/p90: outliers do not move it
    lo, hi = [x for x in xs if x < c], [x for x in xs if x >= c]
    f = lambda v: f"{len(v)} x {st.median(v):.3f} ms" if v else "0"
    return f"{kernel} (cut {c:.2f} ms): below {f(lo)}, above {f(hi)}, mean {st.mean(xs):.3f} ms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--split", action="append", default=[])
    a = ap.parse_args()
    d = json.load(open(a.model))
    res = []
    for run in a.runs:
        ev, prompt = load(run)
        res.append((os.path.basename(os.path.normpath(run)), ev, prompt) + summarize(ev, prompt, d))
    print("| lane | floor | " + " | ".join(f"{r[0]}: measured (% of floor)" for r in res) + " |")
    print("|---|---|" + "---|" * len(res))
    for i, ln in enumerate(d["lanes"]):
        name, _, _, unit, f, _ = res[0][5][i]
        cells = []
        for r in res:
            _, n, meas, unit, f, pct = r[5][i]
            cells.append(f"{meas:.2f} {unit}" + (f" ({pct:.0f} %)" if pct is not None else "") + f", {n} disp")
        print(f"| {name} | " + (f"{f:.2f} {unit}" if f is not None else "-") + " | " + " | ".join(cells) + " |")
    for r in res:
        print(f"\n{r[0]}: prompt {r[2]} tokens, {r[3]} verify steps, device busy {r[4]:.1f} s")
        for spec in a.split:
            k, _, c = spec.partition(":")
            print("  " + clusters(r[1], k, float(c) if c else None))
    if a.top:
        ev = res[0][1]
        agg = {}
        for e in ev:
            key = (e["key"], tuple(e.get("workgroup_count", [])))
            agg.setdefault(key, []).append(e["duration_ns"] / 1e6)
        tot = sum(sum(v) for v in agg.values())
        print(f"\ntop {a.top} (kernel, workgroups) in {res[0][0]}:")
        for (k, wg), v in sorted(agg.items(), key=lambda x: -sum(x[1]))[:a.top]:
            print(f"  {k:36s} {str(list(wg)):18s} n {len(v):6d}  {sum(v) / 1e3:8.2f} s  median {st.median(v):9.3f} ms  {sum(v) / tot * 100:5.1f} %")


if __name__ == "__main__":
    main()
