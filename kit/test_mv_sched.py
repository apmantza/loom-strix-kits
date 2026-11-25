#!/usr/bin/env python3
"""MV_SCHED=D:U (sched-1125, HRX-like static-1125/sched/REPORT.md): Loom's `pipeline(D) unroll(U) schedule(recurrence)` on the
decode matvec row loop. Guards the recurrences the static study found: the two-weight (swiglu) kernels spilled and lost residency
under the policy (q5_k swiglu t16: 115 scratch ops, t8 VGPR 113 -> 180) and R <= D x U row loops have no steady loop, so the
policy must reach single-weight kernels with >= 8 rows only; unset, the generator's text is unchanged (byte-identical program)."""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(HERE, "..", "kits", "qwen38-27b", "config.json")
PROBE = r"""
import sys; sys.argv = ["gen_llm.py", sys.argv[1]]
sys.path.insert(0, sys.argv[0] and %r)
import gen_llm as g
for name, kw in (("t8", dict(TG=8, iq=True)), ("swiglu", dict(TG=8, iq=True, swiglu=True)), ("r4", dict(R=4, TG=4, iq=True))):
    t = g._mv("q5_k", 5120, **kw)
    t = t if isinstance(t, str) else "".join(t)
    print(name, "schedule(recurrence)" in t, "pipeline(%%mvs_d) unroll(%%mvs_u)" in t)
""" % HERE


def run(sched):
    env = dict(os.environ, LLM_CONFIG=CFG)
    env.pop("MV_SCHED", None)
    if sched:
        env["MV_SCHED"] = sched
    r = subprocess.run([sys.executable, "-c", PROBE, CFG], capture_output=True, text=True, env=env, cwd=HERE)
    return r.returncode, dict((l.split()[0], l.split()[1:]) for l in r.stdout.splitlines() if l.split() and l.split()[0] in ("t8", "swiglu", "r4")), r.stderr


def main():
    fails = 0
    rc, on, err = run("2:2")
    ok = rc == 0 and on.get("t8") == ["True", "True"]
    print(("ok  " if ok else "FAIL") + f"  MV_SCHED=2:2 puts the policy on the single-weight 16-row t8 kernel {on.get('t8')} {err[-300:] if rc else ''}"); fails += not ok
    ok = rc == 0 and on.get("swiglu") == ["False", "False"]
    print(("ok  " if ok else "FAIL") + f"  ... and not on the swiglu pair (spilled under it) {on.get('swiglu')}"); fails += not ok
    ok = rc == 0 and on.get("r4") == ["False", "False"]
    print(("ok  " if ok else "FAIL") + f"  ... nor on a 4-row kernel (no steady loop) {on.get('r4')}"); fails += not ok
    rc, off, err = run("")
    ok = rc == 0 and all(v == ["False", "False"] for v in off.values()) and len(off) == 3
    print(("ok  " if ok else "FAIL") + f"  unset: no policy anywhere {off}"); fails += not ok
    print(f"{4 - fails}/4 passed"); return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
