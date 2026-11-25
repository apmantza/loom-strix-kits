#!/usr/bin/env python3
"""MV_SCHED=D:U (sched-1125, HRX-like static-1125/sched/REPORT.md): Loom's `pipeline(D) unroll(U) schedule(recurrence)` on the
decode matvec row loop. Each row names the recurrence it guards:
- scope: the two-weight (swiglu) kernels spilled and lost residency under the policy (q5_k swiglu t16: 115 scratch ops, t8 VGPR
  113 -> 180), so only single-weight kernels take it;
- R bound: a row loop of R <= D x U rows has no steady loop; a hard-coded `R >= 8` gave the 12 R=8 kernels the policy at 2:4
  (VGPRs up, 0 loads in flight, vmcnt(0) kept; review F1), so the bound is R > D x U, checked at the boundary for 2:2 and 2:4;
- the f32-activation (dotf) path: the served GDN alpha/beta matvec (q8_0 K5120 t4 r48, no _q8) takes the policy (review M8);
- unset: the whole program is byte-identical, i.e. the 2:2 / 2:4 program is the unset program plus exactly the policy text
  (an independent inverse strips it; review M7);
- refusals: VERIFY_FUSE (gen_pstep anchors the bare row loop) and malformed D:U specs exit 1 with a message, also under
  `python -O`, which strips asserts (review F6);
- build log: the generator prints one MV_SCHED line with D:U and the number of kernels it changed (review F7).
usage: python3 test_mv_sched.py (exit 1 on a failure)."""
import os, re, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(HERE, "..", "kits", "qwen38-27b", "config.json")
POL = " pipeline(%mvs_d) unroll(%mvs_u) schedule(recurrence)"
PROBE = r"""
import sys; sys.argv = ["gen_llm.py", sys.argv[1]]
import gen_llm as g
def has(t):
    t = t if isinstance(t, str) else "".join(t)
    return "schedule(recurrence)" in t and "pipeline(%mvs_d) unroll(%mvs_u)" in t
for name, kw in (("t8", dict(TG=8, iq=True)), ("swiglu", dict(TG=8, iq=True, swiglu=True)), ("r4", dict(R=4, TG=4, iq=True))):
    print(name, has(g._mv("q5_k", 5120, **kw)))
for R in (4, 5, 8, 9, 16):   # the R bound: 2:2 -> boundary 4, 2:4 -> boundary 8
    print(f"R{R}", has(g._mv("q5_k", 17408, R=R, TG=4, iq=True)))
t = g._mv("q8_0", 5120, R=48, TG=4, suffix="_r48")   # as served (GDN_AB_R=1): f32 activations, vector.dotf, not int8 dot4
print("f32", has(t) if 'export("g_mv_q8_0_k5120_t4_r48")' in t and "vector.dotf" in t else "not-the-f32-kernel")
"""


def env_for(**kw):
    env = dict(os.environ, LLM_CONFIG=CFG)
    for k in ("MV_SCHED", "VERIFY_FUSE"):
        env.pop(k, None)
    env.update({k: v for k, v in kw.items() if v is not None})
    return env


def probe(opt=False, **kw):
    r = subprocess.run([sys.executable] + (["-O"] if opt else []) + ["-c", PROBE, CFG], capture_output=True, text=True,
                       env=env_for(**kw), cwd=HERE)
    return r.returncode, dict(l.split(None, 1) for l in r.stdout.splitlines() if len(l.split()) == 2), r.stderr


def program(out, **kw):
    """The whole generator run as build.py runs it (gen_llm.py as __main__ on the kit's config), into out/model.loom."""
    r = subprocess.run([sys.executable, os.path.join(HERE, "gen_llm.py"), CFG], capture_output=True, text=True,
                       env=dict(env_for(**kw), LLM_OUT=out), cwd=HERE)
    text = open(os.path.join(out, "model.loom")).read() if r.returncode == 0 else ""
    return r.returncode, text, r.stderr


def strip(text, d, u):
    """Independent inverse of the policy: drop the policy text and its two constants; nothing else may differ."""
    defs = f"  %mvs_d = index.constant {d} : index\n  %mvs_u = index.constant {u} : index\n"
    return text.replace(defs, "").replace(POL + " {", " {")


fails = 0


def case(what, ok, detail=""):
    global fails
    print(("ok  " if ok else "FAIL") + "  " + what + (f"  [{detail}]" if not ok and detail != "" else ""))
    fails += not ok


def main():
    rc, on, err = probe(MV_SCHED="2:2")
    case("2:2 puts the policy on the single-weight 16-row t8 kernel", rc == 0 and on.get("t8") == "True", f"{on.get('t8')} {err[-300:]}")
    case("... and not on the swiglu pair (spilled under it)", rc == 0 and on.get("swiglu") == "False", on.get("swiglu"))
    case("... nor on a 4-row kernel (no steady loop)", rc == 0 and on.get("r4") == "False", on.get("r4"))
    case("2:2 boundary: R = D x U = 4 skipped, R = 5 and the next served R = 8 scheduled",
         rc == 0 and [on.get(f"R{r}") for r in (4, 5, 8)] == ["False", "True", "True"], [on.get(f"R{r}") for r in (4, 5, 8)])
    case("2:2 schedules the f32-activation (dotf) path: the served GDN alpha/beta q8_0 K5120 t4 r48",
         rc == 0 and on.get("f32") == "True", on.get("f32"))
    rc4, on4, err4 = probe(MV_SCHED="2:4")
    case("2:4 boundary: R = D x U = 8 skipped, R = 9 and the next served R = 16 scheduled",
         rc4 == 0 and [on4.get(f"R{r}") for r in (8, 9, 16)] == ["False", "True", "True"],
         f"{[on4.get(f'R{r}') for r in (8, 9, 16)]} {err4[-300:]}")
    rc0, off, err0 = probe()
    case("unset: no policy on any probed kernel", rc0 == 0 and len(off) == 9 and all(v == "False" for v in off.values()),
         f"{off} {err0[-300:]}")
    for opt in (False, True):
        tag = " (python -O)" if opt else ""
        for spec in ("0:2", "-1:2", "2", "2:2:2", "a:b", "2:0", " 2:2"):
            rc, out, err = probe(opt, MV_SCHED=spec)
            case(f"MV_SCHED={spec!r} refused{tag} with a message naming the spec",
                 rc == 1 and not out and f"MV_SCHED={spec!r}" in err and "Traceback" not in err, f"rc={rc} {err.strip()[-200:]}")
        for vf in ("swq", "1"):
            rc, out, err = probe(opt, MV_SCHED="2:2", VERIFY_FUSE=vf)
            case(f"MV_SCHED=2:2 with VERIFY_FUSE={vf} refused{tag}",
                 rc == 1 and not out and "VERIFY_FUSE" in err and "Traceback" not in err, f"rc={rc} {out} {err.strip()[-200:]}")
        rc, out, err = probe(opt, MV_SCHED="2:2", VERIFY_FUSE="0")
        case(f"MV_SCHED=2:2 with VERIFY_FUSE=0 runs{tag}", rc == 0 and out.get("t8") == "True", f"rc={rc} {err.strip()[-200:]}")
    with tempfile.TemporaryDirectory() as tmp:
        progs = {}
        for spec in (None, "2:2", "2:4"):
            out = os.path.join(tmp, spec.replace(":", "_") if spec else "unset")
            os.makedirs(out)
            progs[spec] = program(out, MV_SCHED=spec)
        rc, base, err = progs[None]
        case("unset: the program builds and prints no MV_SCHED line", rc == 0 and bool(base) and "MV_SCHED" not in err, err[-300:])
        for spec in ("2:2", "2:4"):
            d, u = spec.split(":")
            rc, text, err = progs[spec]
            n = sum(1 for blk in re.split(r"(?m)^(?=kernel\.def )", text) if blk.startswith("kernel.def") and POL + " {" in blk)
            defs = text.count(f"  %mvs_d = index.constant {d} : index\n  %mvs_u = index.constant {u} : index\n")
            case(f"{spec}: the program is the unset program plus exactly the policy text (byte for byte), one D/U pair per kernel",
                 rc == 0 and n > 0 and defs == n and strip(text, d, u) == base, f"rc={rc} n={n} defs={defs} {err[-200:]}")
            lines = [l for l in err.splitlines() if l.startswith("MV_SCHED")]
            case(f"{spec}: one MV_SCHED line naming D:U and the {n} kernels it changed",
                 len(lines) == 1 and f"MV_SCHED={spec}" in lines[0] and f" {n} kernels" in lines[0], lines)
    print("all passed" if not fails else f"{fails} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
