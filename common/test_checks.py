#!/usr/bin/env python3
"""Tests of checks.py. Each case names the recurrence it prevents. usage: python3 test_checks.py (exit 1 on a failure)"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checks  # noqa: E402

fails = []


def case(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


# a fast degenerate output (a list repeating itself) must not pass as a speed record
looping = " ".join(f"{i}. the same item again and again" for i in [1, 2, 3] * 30)
prose = " ".join(f"word{i} varies here" for i in range(200))
case("loop: a repeating list is flagged", checks.is_loop(looping))
case("loop: varied prose is not flagged", not checks.is_loop(prose))
case("loop: a short answer is never flagged", not checks.is_loop("yes " * 40))

# an A/B verdict must not call noise a win: a clear 10 % gain excludes 1, noise does not
a = [20.0, 21.0, 19.5, 20.5, 20.2, 19.8, 20.9, 20.1, 19.7, 20.4]
p, lo, hi = checks.paired_ratio_ci(a, [x * 1.10 for x in a])
case("ci: a 10 % gain excludes 1", lo > 1.0 and abs(p - 1.10) < 1e-9)
noisy = [x * f for x, f in zip(a, [1.08, 0.93, 1.05, 0.96, 1.02, 0.97, 1.06, 0.95, 1.01, 0.99])]
p, lo, hi = checks.paired_ratio_ci(a, noisy)
case("ci: paired noise straddles 1", lo < 1.0 < hi)

# ABBA: a candidate that wins one half and loses the other is not accepted; a secondary metric 5 % worse rejects
hb = {"decode": True, "prefill": True, "ttft": False}
win = {"h1": {"ref": {"decode": 20, "prefill": 400, "ttft": 10}, "cand": {"decode": 21, "prefill": 401, "ttft": 10}},
       "h2": {"ref": {"decode": 20, "prefill": 400, "ttft": 10}, "cand": {"decode": 21, "prefill": 399, "ttft": 10.1}}}
case("verdict: both halves better, secondaries within 3 % -> accept", checks.neighbour_verdict(win, "decode", ["prefill", "ttft"], hb)[0])
split = {"h1": win["h1"], "h2": {"ref": {"decode": 20, "prefill": 400, "ttft": 10}, "cand": {"decode": 19.5, "prefill": 400, "ttft": 10}}}
case("verdict: one half worse -> reject", not checks.neighbour_verdict(split, "decode", ["prefill", "ttft"], hb)[0])
slow = {h: {"ref": d["ref"], "cand": dict(d["cand"], ttft=10.6)} for h, d in win.items()}
case("verdict: ttft 6 % worse -> reject", not checks.neighbour_verdict(slow, "decode", ["prefill", "ttft"], hb)[0])

# unified memory: the GPU's share is only in fdinfo; two fds of one DRM client must be counted once
with tempfile.TemporaryDirectory() as d:
    os.makedirs(f"{d}/fdinfo")
    open(f"{d}/fdinfo/5", "w").write("drm-client-id:\t7\ndrm-memory-gtt: \t1048576 KiB\ndrm-memory-vram:\t1024 KiB\n")
    open(f"{d}/fdinfo/6", "w").write("drm-client-id:\t7\ndrm-memory-gtt: \t1048576 KiB\ndrm-memory-vram:\t1024 KiB\n")   # every fd of a client shows the same totals
    open(f"{d}/fdinfo/7", "w").write("drm-client-id:\t9\ndrm-resident-gtt: \t2048 KiB\n")
    os.makedirs(f"{d}/1234"); os.rename(f"{d}/fdinfo", f"{d}/1234/fdinfo")
    m = checks.drm_memory(1234, proc=d)
    case("drm: one client counted once, resident used when memory is absent", m == {"gtt": 1048576 + 2048, "vram": 1024})

print(f"{8 + 1 - len(fails)}/{8 + 1} passed" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
