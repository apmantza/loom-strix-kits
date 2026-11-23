#!/usr/bin/env python3
"""common/calibrate.py on synthetic events. Recurrences it prevents:
  - a harness figure accepted although service runs the kernel slower (E313: an isolated 0.95 was 1.039 served): a served median
    outside 1 +- tol of the harness must exit 1;
  - two shapes on one grid averaged into one figure (the f16 GEMM's (40, 16) serves K 6144 and K 17408): the cluster nearest the
    harness is compared, not the mix;
  - the wrong prefill chunk compared (flash2's time grows with the context): --order picks the chunk ending at the given context.
usage: python3 common/test_calibrate.py"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrate  # noqa: E402

fails = 0


def ok(cond, what):
    global fails
    print(("ok  " if cond else "FAIL") + " " + what)
    fails += not cond


def ev(key, grid, ms):
    return json.dumps({"type": "dispatch_event", "key": key, "workgroup_count": list(grid), "duration_ns": int(ms * 1e6)})


def main():
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "events.jsonl")
        lines = [ev("g_gemm", (40, 16, 1), 7.6) for _ in range(10)] + [ev("g_gemm", (40, 16, 1), 20.0) for _ in range(10)]
        lines += [ev("g_flash", (256, 4, 1), 100.0 + 10 * (i // 4)) for i in range(16)]      # 4 chunks x 4 launches, time grows
        lines += [ev("g_other", (40, 16, 1), 1.0)]
        open(p, "w").write("\n".join(lines) + "\n")
        ok(calibrate.main([p, "g_gemm", "40,16,1", "19.5"]) == 0, "a 20.0 ms cluster against a 19.5 ms harness: within 5 %")
        ok(calibrate.main([p, "g_gemm", "40,16,1", "18.0"]) == 1, "the same cluster against 18.0 ms: outside (exit 1)")
        ok(calibrate.main([p, "g_gemm", "40,16,1", "7.5"]) == 0, "the 7.6 ms cluster is picked for a 7.5 ms harness, not the mix")
        ok(calibrate.main([p, "g_flash", "256,4,1", "120.0", "--order", "4,4096,12288"]) == 0, "--order picks the third chunk (120 ms)")
        ok(calibrate.main([p, "g_flash", "256,4,1", "100.0", "--order", "4,4096,12288"]) == 1, "and not the first (100 ms)")
        ok(calibrate.main([p, "g_gemm", "1,1,1", "1.0"]) == 2, "no dispatch at the grid: exit 2, not a pass")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
