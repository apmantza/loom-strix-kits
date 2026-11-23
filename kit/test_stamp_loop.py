#!/usr/bin/env python3
"""kit/stamp_loop.py, statically, on a real program (HRX-like E294 / E302; replaces test_stamp_gemm.py). For the served f16 GEMM
(loop over %step, 2 barriers) and flash2 prefill attention (loop over %kt, 3 barriers). Recurrences it prevents:
  - the unstamped copy differing from the served kernel (the run would compare two kernels): equal line for line;
  - a stamp on the wrong side of a barrier (work and wait swapped): every body-level barrier has a stamp just before and just after
    it, each fenced, and the iteration's last stamp sits just before the yield;
  - the final stamp and store outside the region that holds the loop (q111: Loom refused the fence, PARSE/003);
  - low.invoke without parenthesised results (q111: PARSE/003);
  - wave slots not following the grid (q111: x-major slots collided, 14400 of 17408 rows lost): slot = ((z GY + y) GX + x) waves + wave.
usage: python3 kit/test_stamp_loop.py PROGRAM.loom"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stamp_loop  # noqa: E402

fails = 0


def ok(cond, what):
    global fails
    print(("ok  " if cond else "FAIL") + " " + what)
    fails += not cond


def check(lines, kernel, iv, nbar):
    block = stamp_loop.extract(lines, kernel)
    S, N = stamp_loop.stamped(block, kernel, iv, 7, 5, 8)
    ok(N == nbar, f"{kernel}: {N} body-level barriers found in the loop over %{iv} (want {nbar})")
    f = next(i for i, l in enumerate(S) if re.search(rf"= scf\.for %{iv} = \[", l))
    y = next(i for i in range(f, len(S)) if S[i].strip().startswith("scf.yield ") and "%ss_e" in S[i])
    ind = len(S[f]) - len(S[f].lstrip()) + 2
    bars = [i for i in range(f, y) if S[i].startswith(" " * ind + "kernel.barrier")]
    # before a barrier: fence, stamp, fence, then the two accumulation lines; after it: fence, stamp, fence
    around = all(S[b - 5].strip() == "scf.schedule.fence" and "%ss_w" in S[b - 4] and S[b - 3].strip() == "scf.schedule.fence"
                 and S[b + 1].strip() == "scf.schedule.fence" and "%ss_b" in S[b + 2] and S[b + 3].strip() == "scf.schedule.fence" for b in bars)
    ok(len(bars) == nbar and around, f"{kernel}: every loop barrier has a fenced stamp just before and just after it")
    ok("%ss_e = low.invoke" in S[y - 4] and S[y - 3].strip() == "scf.schedule.fence", f"{kernel}: the iteration's last stamp is just before the yield")
    ok(all("-> (i32)" in l for l in S if "low.invoke @g_stamp_clock" in l), f"{kernel}: low.invoke results in parentheses")
    r = next(i for i, l in enumerate(S) if l.strip() == "kernel.return")
    st = next(i for i, l in enumerate(S) if "%st_end = low.invoke" in l)
    D = stamp_loop.depths(S, 0)
    ok(st < r and D[st] == D[f], f"{kernel}: the final stamp sits in the region that holds the loop (depth {D[st]} = loop depth {D[f]})")
    tail = "\n".join(S[st:r])
    ok("index.madd %st_zy, %st_gx, %st_wx" in tail and "index.madd %st_wz, %st_gy, %st_wy" in tail and "%st_gx = index.constant 7" in tail,
       f"{kernel}: wave slots follow the grid ((z GY + y) GX + x)")
    ok(S[:0] == [] and stamp_loop.extract(lines, kernel) == block, f"{kernel}: the unstamped copy equals the program's kernel")


def main():
    prog = sys.argv[1]
    lines = open(prog).read().splitlines()
    check(lines, "g_gemm_f16_store_m256", "step", 2)
    check(lines, "g_attention_flash2", "kt", 3)
    # the default clock is the cycle counter (no memory wait); realtime reads perturbed the GEMM +41 % and flash2 3.2x (E304)
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        out = os.path.join(t, "s.loom")
        sys.argv = ["stamp_loop.py", sys.argv[1], "g_gemm_f16_store_m256", "step", "1", "1", "8", out]
        stamp_loop.main()
        txt = open(out).read()
    ok("s_getreg_b32" in txt and "s_sendmsg_rtn" not in txt and "scalar.andi" in txt, "the default clock is the cycle counter, with wrap-aware deltas")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
