#!/usr/bin/env python3
"""One lane of a served run, kernel by kernel and shape by shape (events_summary.py's lanes, same epoch phases).

Per (kernel, workgroup count): dispatches per verify step (decode) or in total (prefill), time, share of the lane.
Decode matvecs (g_mv_*, g_head_mv_*) also get achieved GB/s: rows = workgroups.y (x is token groups) x the kernel's %rows_per (read from the
served program's kernel.def), bytes = rows x K x the format's bits per weight (x2 for the two-weight swiglu kernels).
usage: lane_breakdown.py MODEL.json RUN PROGRAM.loom "LANE NAME" [--top 20]
"""
import argparse
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from events_summary import classify, load  # noqa: E402

BITS = {"q4_k": 4.5, "q5_k": 5.5, "q6_k": 6.5625, "q8_0": 8.5, "q4_0": 4.5, "iq4_xs": 4.25, "iq4_nl": 4.5, "q3_k": 3.4375,
        "q2_k": 2.625, "bf16": 16, "f16": 16}


def rows_per(program, kernel):
    m = re.search(r'kernel\.def export\("' + re.escape(kernel) + r'"\).*?%rows_per = index\.constant (\d+)', program, re.S)
    return int(m.group(1)) if m else None


def matvec_bytes(kernel, wg_y, program, head_bytes):
    if kernel.startswith("g_head_mv"):
        return head_bytes.get(kernel)
    m = re.match(r"g_mv_([a-z0-9_]+?)_k(\d+)_", kernel)
    if not m or m.group(1) not in BITS:
        return None
    rp = rows_per(program, kernel)
    if rp is None:
        return None
    two = 2 if "_swiglu_" in kernel else 1
    return wg_y * rp * int(m.group(2)) * BITS[m.group(1)] / 8 * two


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("run"); ap.add_argument("program"); ap.add_argument("lane")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    d = json.load(open(a.model))
    lanes = {ln["lane"]: ln for ln in d["lanes"]}
    if a.lane not in lanes:
        raise SystemExit(f"lane_breakdown: no lane {a.lane!r}; lanes: {list(lanes)}")
    ln = lanes[a.lane]
    ev, prompt = load(a.run)
    pre, lane_of = classify(ev, d)
    vl = next(x["lane"] for x in d["lanes"] if x["floor"] == "decode_kv")
    steps = len({e["submission_id"] for e in ev if e["submission_id"] not in pre and lane_of(e) == vl})
    mine = [e for e in ev if lane_of(e) == a.lane]
    program = open(a.program).read()
    head_bytes = {}   # @llm_verify's head view and the propose head copy: bytes from their view types in the program
    for k, v in re.findall(r"kernel\.launch @(g_head_mv\w*)\[[^\n]*view<(\d+)xi8>", program):
        head_bytes[k] = int(v)
    agg = {}
    for e in mine:
        agg.setdefault((e["key"], tuple(e.get("workgroup_count", []))), []).append(e["duration_ns"] / 1e6)
    tot = sum(sum(v) for v in agg.values())
    dec = ln["phase"] == "decode"
    print(f"{a.lane}: {len(mine)} dispatches, {tot / (steps if dec else 1e3):.2f} {'ms/step' if dec else 's'} "
          f"({steps} verify steps, prompt {prompt})")
    print(f"  {'kernel':40s} {'workgroups':16s} {'n/step' if dec else 'n':>7s} {'ms/step' if dec else 's':>8s} {'median ms':>10s} {'share':>6s}  GB/s")
    for (k, wg), v in sorted(agg.items(), key=lambda x: -sum(x[1]))[:a.top]:
        b = matvec_bytes(k, wg[1], program, head_bytes) if len(wg) > 1 else None   # launch: workgroups(token_groups, row_groups, 1)
        gbs = f"{b / (st.median(v) / 1e3) / 1e9:6.0f}" if b else "     -"
        n = len(v) / steps if dec else len(v)
        t = sum(v) / steps if dec else sum(v) / 1e3
        print(f"  {k:40s} {str(list(wg)):16s} {n:7.1f} {t:8.2f} {st.median(v):10.3f} {sum(v) / tot * 100:5.1f} %  {gbs}")


if __name__ == "__main__":
    main()
