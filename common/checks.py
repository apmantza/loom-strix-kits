#!/usr/bin/env python3
"""Shared checks for benches and A/B verdicts (ideas from sybil-solutions/inference-optimisation-skill, reviewed 11-22).

loop_share(text)          share of word 8-grams that repeat an earlier 8-gram; is_loop() flags >= 0.5 over >= 64 words
                          (a 117 tok/s "record" there was a numbered list repeating itself: a fast loop is not a fast model)
paired_ratio_ci(a, b)     bootstrap 95 % interval of median(b) / median(a) over paired samples (turns), resampled together
neighbour_verdict(...)    ABBA rule: the candidate must beat the reference in EVERY half on the primary metric (each half's
                          reference run is a neighbouring control), and no secondary metric may be worse by more than 3 %
drm_memory(pid)           GPU memory a process holds, from /proc/PID/fdinfo (drm-memory-* / drm-resident-* per region, one count per
                          drm-client-id): on unified memory the GPU's share is invisible to MemAvailable
usage: checks.py drm PID | checks.py loop FILE"""
import os
import random
import re
import statistics
import sys


def loop_share(text, n=8):
    w = text.split()
    if len(w) < n + 1:
        return 0.0
    seen, rep = set(), 0
    for i in range(len(w) - n + 1):
        g = tuple(w[i:i + n])
        rep += g in seen
        seen.add(g)
    return rep / (len(w) - n + 1)


def is_loop(text, share=0.5, min_words=64):
    return len(text.split()) >= min_words and loop_share(text) >= share


def paired_ratio_ci(a, b, iters=2000, seed=1, level=0.95):
    """(point, low, high) of median(b) / median(a) over paired samples a[i], b[i]"""
    assert len(a) == len(b) and len(a) >= 3, (len(a), len(b))
    rng = random.Random(seed)
    idx = range(len(a))
    point = statistics.median(b) / statistics.median(a)
    rs = []
    for _ in range(iters):
        s = [rng.choice(idx) for _ in idx]
        rs.append(statistics.median([b[i] for i in s]) / statistics.median([a[i] for i in s]))
    rs.sort()
    lo, hi = rs[int((1 - level) / 2 * iters)], rs[int((1 + level) / 2 * iters) - 1]
    return point, lo, hi


def neighbour_verdict(halves, primary, secondary, higher_better, tol=0.03):
    """halves: {half: {"ref": {metric: value}, "cand": {metric: value}}}; higher_better: {metric: bool}.
    Returns (accept, reasons)."""
    reasons, ok = [], True
    for h, d in sorted(halves.items()):
        r, c = d["ref"][primary], d["cand"][primary]
        better = c > r if higher_better[primary] else c < r
        reasons.append(f"{h}: {primary} {c:.3g} vs {r:.3g} {'better' if better else 'NOT better'}")
        ok &= better
    for m in secondary:
        r = sum(d["ref"][m] for d in halves.values()) / len(halves)
        c = sum(d["cand"][m] for d in halves.values()) / len(halves)
        worse = (r - c) / r if higher_better[m] else (c - r) / r
        if worse > tol:
            ok = False
            reasons.append(f"{m}: {worse:.1%} worse (> {tol:.0%})")
    return ok, reasons


def drm_memory(pid, proc="/proc"):
    """{region: KiB} summed over the process's DRM clients (each drm-client-id once); {} if it has none"""
    out, seen = {}, set()
    d = f"{proc}/{pid}/fdinfo"
    for f in os.listdir(d):
        try:
            txt = open(os.path.join(d, f)).read()
        except OSError:
            continue
        cid = re.search(r"^drm-client-id:\s*(\d+)", txt, re.M)
        if not cid or cid.group(1) in seen:
            continue
        seen.add(cid.group(1))
        for kind, region, kib in re.findall(r"^drm-(memory|resident)-(\w+):\s*(\d+) KiB", txt, re.M):
            if kind == "memory" or f"memory-{region}" not in txt:   # drm-memory-* where given, else drm-resident-*
                out[region] = out.get(region, 0) + int(kib)
    return out


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "drm":
        m = drm_memory(int(sys.argv[2]))
        print(" ".join(f"{k} {v / 1048576:.1f} GiB" for k, v in sorted(m.items()) if v) or "no DRM memory")
    elif len(sys.argv) == 3 and sys.argv[1] == "loop":
        t = open(sys.argv[2]).read()
        print(f"loop share {loop_share(t):.3f} {'LOOP' if is_loop(t) else 'ok'}")
    else:
        sys.exit(__doc__)
