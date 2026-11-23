#!/usr/bin/env python3
"""kit/stamp_gemm.py, statically, on a real program (HRX-like E294). Recurrences it prevents:
  - a stamp landing on the wrong side of a barrier (the stage / mma / barrier-2 split would then measure something else): in the
    stamped k-loop the order must be barrier 1, stamp, ..., stamp, barrier 2, stamp, yield, with a schedule fence around every stamp;
  - the unstamped copy differing from the served kernel (then the run compares two different kernels): it must equal the program's
    kernel.def text line for line;
  - the stamps never reaching memory: the 8-value row is built from the five accumulated durations and stored to %stamps.
usage: python3 kit/test_stamp_gemm.py PROGRAM.loom [KERNEL=g_gemm_f16_store_m256]"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stamp_gemm  # noqa: E402

fails = 0


def ok(cond, what):
    global fails
    print(("ok  " if cond else "FAIL") + " " + what)
    fails += not cond


def main():
    prog = sys.argv[1]
    kernel = sys.argv[2] if len(sys.argv) > 2 else "g_gemm_f16_store_m256"
    lines = open(prog).read().splitlines()
    block = stamp_gemm.extract(lines, kernel)
    with tempfile.TemporaryDirectory() as t:
        out = os.path.join(t, "s.loom")
        sys.argv = ["stamp_gemm.py", prog, kernel, out]
        stamp_gemm.main()
        text = open(out).read().splitlines()
    i0 = next(i for i, l in enumerate(text) if f'export("{kernel}")' in l)
    i1 = next(i for i, l in enumerate(text) if f'export("{kernel}_stamped")' in l)
    ok(text[i0:i0 + len(block)] == block, f"the unstamped copy equals the program's {kernel} ({len(block)} lines)")
    S = text[i1:]
    S = S[:next(i for i, l in enumerate(S) if l == "}" and i > 0 and any("} launch(" in x for x in S[:i]))+ 1]
    f = next(i for i, l in enumerate(S) if "= scf.for %step" in l)
    y = next(i for i in range(f, len(S)) if S[i].startswith("    scf.yield "))
    loop = S[f:y + 1]
    kinds = [("bar" if l.startswith("    kernel.barrier") else "stamp" if "low.invoke @g_stamp_clock" in l else None) for l in loop]
    seq = [k for k in kinds if k]
    ok(seq == ["bar", "stamp", "stamp", "bar", "stamp"], f"loop order barrier 1, stamp, stamp, barrier 2, stamp (got {seq})")
    fenced = all(loop[i - 1].strip() == "scf.schedule.fence" and loop[i + 1].strip() == "scf.schedule.fence"
                 for i, k in enumerate(kinds) if k == "stamp")
    ok(fenced, "every loop stamp has a schedule fence on both sides")
    j_b2 = [i for i, k in enumerate(kinds) if k == "bar"][1]
    ok("%ss_mma" in loop[j_b2 - 2] and "%ss_b2" in loop[j_b2 + 2], "the mma stamp is just before barrier 2 and the barrier-2 stamp just after it")
    tail = "\n".join(S[y:])
    ok("vector.from_elements %st_stage, %st_mma, %st_b2, %st_pro, %st_epi, %st_steps" in tail and "vector.store %st_row, %st_view" in tail,
       "the five durations and the step count are stored to %stamps")
    ok(", %stamps: buffer) {" in "\n".join(S[:f]), "the stamped kernel takes the extra %stamps buffer last")
    r = next(i for i, l in enumerate(S) if l.strip() == "kernel.return")
    closes = [i for i in range(y + 2, r) if S[i] == "  }"]
    st = next(i for i, l in enumerate(S) if "%st_end = low.invoke" in l)
    ok(all(st < c for c in closes), f"the final stamp and store sit inside the region that holds the loop (before its closing brace; q111 PARSE/003)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
