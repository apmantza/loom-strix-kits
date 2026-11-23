#!/usr/bin/env python3
"""Report a stamped kernel's per-phase timing (kit/stamp_loop.py, HRX-like E294 / E302): one row of 16 i32 per wave,
[work_0, wait_0, ..., work_{N-1}, wait_{N-1}, tail, prologue, epilogue, iterations, 0 ...] in ticks of the realtime clock (100 MHz
assumed: check the per-wave lifetime against the kernel time before quoting microseconds; shares do not depend on it). Prints each
phase's share of the waves' summed time, per-wave medians, and per-iteration means; refuses rows that do not look like stamps
(iterations must match, durations non-negative) so a wave that never wrote its row is not read as a fast one. Waves of workgroups
that skip the loop (no live tile) write no row; give their count as SKIPPED to allow them.
usage: python3 common/stamps_report.py STAMPS.bin|.npy WAVES N ITERATIONS(0: any, per row) [SKIPPED=0] [realtime|cycles]"""
import sys

import numpy as np

TICK_US = 0.01


def main():
    path, waves, N, its = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    skipped = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    global TICK_US
    if len(sys.argv) > 6 and sys.argv[6] == "cycles":   # shader cycles (stamp_loop cycles): ~2.9 GHz at auto, approximate
        TICK_US = 1 / 2900.0
    raw = np.load(path) if path.endswith(".npy") else np.fromfile(path, np.int32)
    a = raw.reshape(-1, 16)[:waves].astype(np.int64)
    k = 2 * N + 3                                             # phases + tail + prologue + epilogue
    written = a[:, k] != 0
    bad = written & (((a[:, k] != its) if its else (a[:, k] < 0)) | (a[:, :k] < 0).any(1))   # ITERATIONS 0: per-row counts (causal tiles)
    if bad.any() or (~written).sum() > skipped:
        sys.exit(f"STAMPS BAD: {int((~written).sum())} rows unwritten (allowed {skipped}), {int(bad.sum())} inconsistent "
                 f"(first: {a[np.argmax(bad | ~written)].tolist()})")
    a = a[written]
    names = [x for i in range(N) for x in (f"work {i} (to barrier {i})", f"wait at barrier {i}")] + ["tail (last barrier to yield)", "prologue", "epilogue"]
    tot = a[:, :k].sum(1)
    print(f"{len(a)} waves, {its or f'{int(a[:, k].min())}-{int(a[:, k].max())}'} iterations each; per-wave lifetime median {np.median(tot) * TICK_US:.1f} us "
          f"(p10 {np.percentile(tot, 10) * TICK_US:.1f}, p90 {np.percentile(tot, 90) * TICK_US:.1f})")
    print(f"{'phase':34s} {'share':>7s} {'median us':>10s} {'per iteration us':>17s}")
    for i, n in enumerate(names):
        per = np.median(a[:, i] / a[:, k]) * TICK_US if i < 2 * N + 1 else float("nan")
        print(f"{n:34s} {a[:, i].sum() / tot.sum():7.1%} {np.median(a[:, i]) * TICK_US:10.2f} {per:17.3f}")
    print("STAMPS OK")


if __name__ == "__main__":
    main()
