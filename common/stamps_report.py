#!/usr/bin/env python3
"""Report a stamped kernel's per-stage timing (kit/stamp_gemm.py, HRX-like E294): rows of 8 i32 per wave
[stage, mma, barrier2, prologue, epilogue, steps, 0, 0] in 100 MHz realtime ticks. Prints each stage's share of the waves' summed
time, per-wave medians and spreads in microseconds, and per-step means; refuses rows that do not look like stamps (steps must be
the expected k-steps, every duration non-negative) so a kernel that never wrote its row is not read as a fast one.
usage: python3 common/stamps_report.py STAMPS.bin WAVES STEPS"""
import sys

import numpy as np

TICK_US = 0.01          # 100 MHz realtime clock (MSG_RTN_GET_REALTIME)
NAMES = ("stage (loads -> LDS, barrier 1)", "mma (fragment loads + WMMA issue)", "barrier 2", "prologue", "epilogue")


def main():
    path, waves, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    a = np.fromfile(path, np.int32).reshape(-1, 8)[:waves].astype(np.int64)
    bad = (a[:, 5] != steps) | (a[:, :5] < 0).any(1)
    if bad.any():
        sys.exit(f"STAMPS BAD: {int(bad.sum())} of {waves} wave rows unwritten or inconsistent (first: {a[np.argmax(bad)].tolist()})")
    tot = a[:, :5].sum(1)
    print(f"{waves} waves, {steps} k-steps each; per-wave lifetime median {np.median(tot) * TICK_US:.1f} us "
          f"(p10 {np.percentile(tot, 10) * TICK_US:.1f}, p90 {np.percentile(tot, 90) * TICK_US:.1f})")
    print(f"{'stage':40s} {'share':>7s} {'median us':>10s} {'per step us':>12s}")
    for i, n in enumerate(NAMES):
        print(f"{n:40s} {a[:, i].sum() / tot.sum():7.1%} {np.median(a[:, i]) * TICK_US:10.2f} "
              f"{(np.median(a[:, i]) / steps if i < 3 else float('nan')) * TICK_US:12.3f}")
    print("STAMPS OK")


if __name__ == "__main__":
    main()
