#!/usr/bin/env python3
"""Does an isolated harness reproduce the served time of a kernel? (HRX-like E313 / E321; AGENTS 7: a harness figure is a hypothesis
until it reproduces the live kernel's served figure.) Reads a served run's per-dispatch events (events_ab.sh's events.jsonl), takes the
dispatches of KERNEL at GRID, splits them into clusters (one grid can serve several shapes: the f16 GEMM's (40, 16) runs both K 6144 and
K 17408), and compares the cluster nearest the harness time with it. Prints served median / harness and the ratio; exit 1 when the ratio
is outside 1 +- TOL (the harness does not reproduce service at this shape: its A/B results there are hypotheses).
Optional --order L,CHUNK,TOKENS selects prefill dispatches by context (flash2: L launches per chunk, chunks of CHUNK tokens; TOKENS = the
context the harness ran at: only dispatches of the chunk ending there).
usage: calibrate.py EVENTS.jsonl KERNEL GRID(x,y,z) HARNESS_MS [--tol 0.05] [--order L,CHUNK,TOKENS]"""
import argparse
import json
import statistics
import sys


def served(path, kernel, grid):
    """durations (ms) of KERNEL's dispatches at GRID, in dispatch order"""
    out = []
    for line in open(path):
        if f'"{kernel}"' not in line:
            continue
        e = json.loads(line)
        if e.get("key") == kernel and tuple(e["workgroup_count"]) == grid:
            out.append(e["duration_ns"] / 1e6)
    return out


def clusters(values, gap=0.15):
    """1-D clusters: split the sorted values where the next one is more than GAP (relative) above the previous"""
    v = sorted(values)
    out, cur = [], [v[0]]
    for x in v[1:]:
        if x > cur[-1] * (1 + gap):
            out.append(cur)
            cur = []
        cur.append(x)
    out.append(cur)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("events")
    ap.add_argument("kernel")
    ap.add_argument("grid")
    ap.add_argument("harness_ms", type=float)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--order", help="L,CHUNK,TOKENS: only the dispatches of the prefill chunk ending at TOKENS (L launches per chunk)")
    a = ap.parse_args(argv)
    grid = tuple(int(x) for x in a.grid.split(","))
    d = served(a.events, a.kernel, grid)
    if not d:
        print(f"calibrate: no {a.kernel} dispatch at grid {grid} in {a.events}")
        return 2
    if a.order:
        L, chunk, tokens = (int(x) for x in a.order.split(","))
        k = tokens // chunk - 1                       # the chunk that ends at TOKENS
        d = d[k * L:(k + 1) * L]
        if not d:
            print(f"calibrate: no dispatches for the chunk ending at {tokens}")
            return 2
    cs = clusters(d)
    best = min(cs, key=lambda c: abs(statistics.median(c) - a.harness_ms))
    med = statistics.median(best)
    r = med / a.harness_ms
    ok = abs(r - 1) <= a.tol
    print(f"{a.kernel} {grid}: {len(d)} served dispatches in {len(cs)} cluster(s); nearest cluster {len(best)} x median {med:.3f} ms; "
          f"harness {a.harness_ms:.3f} ms; served / harness {r:.3f} {'OK' if ok else 'OUTSIDE'} (tol {a.tol:.0%})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
