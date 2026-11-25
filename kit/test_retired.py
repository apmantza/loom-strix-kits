#!/usr/bin/env python3
"""Retired switches are refused by every generator entry point (kit/retired.py; switch census 11-25, rules section 4: a folded
switch's off path and a refuted switch's code are deleted). Recurrence it guards: a stale recipe or package .cmd (common/ktest.sh
re-runs a package's switches) that still sets a retired name builds something other than it asks for, silently: FLASH2_PF11_SPLIT=0
now builds the split staging, GEMM256_LDS_PF=1 builds the plain order, FLASH2_SXH=1 without FLASH2_SX11 builds no exchange at all.
A value that asks for exactly what the kit now builds passes, so the pre-fold recipe (pkg-e480.cmd's switches) still generates
the same kernels as the folded recipe.
usage: python3 test_retired.py   (exit 1 on a failure)"""
import json, os, re, shlex, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
# the live package's switches before the fold (pkg-e480.cmd): every folded name at its folded value, with its parent
OLD = shlex.split("PREFILL_F16_SCRATCH=1 ATTN_L17_VS=1 FLASH2_PF11=1 FLASH2_PF11_SPLIT=1 FLASH2_PF11_NOIF=1 FLASH2_PF11_CARRY=1 "
                  "FLASH2_KVH11=1 KV8=1 KV8_LAYOUT=2 GDN_FOLD=1 FLASH2_VS_KVH=1 FLASH2_KV8_KCARRY=1 FLASH2_KV8_VLATE=2 "
                  "FLASH2_SPLITK_ANY=1 DFL_SPLIT_W=1 PF16_PAD=64 GDN_AB_R=1 DRAFT_MM_FROM=17 F16O11=1 EPI_FUSE=1 FLASH2_HEADPACK=1 "
                  "GEMM256_FLATADDR=1 F16O_O=1 FLASH2_SX11=1 KV8_SPLIT_VZERO=1 FLASH2_SXH=1 FLASH2_SMX=1")
OLD = dict(kv.split("=", 1) for kv in OLD)
GEOM = {"kv_heads": 4, "heads": 24, "head_dim": 256, "layers": 64}


def recipe(kit="qwen38-27b"):
    for ln in open(os.path.join(HERE, "..", "kits", kit, "kit.env")):
        if ln.startswith("KIT_RECIPE="):
            words = [w for t in shlex.split(ln.split("=", 1)[1], comments=True) for w in t.split()]   # the quoted value's words
            sw = dict(kv.split("=", 1) for kv in words)
            assert sw and all(re.fullmatch(r"[A-Z][A-Z0-9_]*", k) and re.fullmatch(r"[^\s=]+", v) for k, v in sw.items()), sw
            return sw
    raise SystemExit(f"no KIT_RECIPE in kits/{kit}/kit.env")


def run(script, sw, tmp, cfg=None):
    """Run a generator with exactly the switches SW (the caller's own switches removed); (returncode, stderr, output text)."""
    env = {k: v for k, v in os.environ.items() if not re.fullmatch(r"[A-Z0-9_]+", k) or k in ("PATH", "HOME", "TMPDIR", "VENV")}
    env.update(sw)
    out = os.path.join(tmp, "out.loom")
    if script == "gen_flash2s.py":
        env.update(LLM_CONFIG=cfg, FLASH2_TARGET="llm", FLASH2_OUT=out)
        argv = [sys.executable, os.path.join(HERE, script)]
    elif script == "gen_gemm.py":
        env.update(GEMM_FORMATS="q4_k:store", LOOM_RDNA4="0")
        argv = [sys.executable, os.path.join(HERE, script)]
    else:   # gen_llm.py: a config that does not exist; the refusal must come before the config is read
        argv = [sys.executable, os.path.join(HERE, script), os.path.join(tmp, "no-such-config.json")]
    r = subprocess.run(argv, capture_output=True, text=True, env=env)
    text = r.stdout
    if script == "gen_flash2s.py" and os.path.exists(out):
        text = open(out).read(); os.remove(out)
    return r.returncode, r.stderr, text


def main():
    fails, n = 0, 0

    def check(ok, what):
        nonlocal fails, n
        n += 1; fails += not ok
        print(("ok  " if ok else "FAIL") + "  " + what)

    new = recipe()
    with tempfile.TemporaryDirectory() as tmp:
        cfg = os.path.join(tmp, "cfg.json"); json.dump(GEOM, open(cfg, "w"))
        refused = lambda rc, err, name: rc != 0 and "retired switch" in err and name in err
        for script in ("gen_llm.py", "gen_flash2s.py", "gen_gemm.py"):
            for sw, name, why in (({"FLASH2_PF11": "1", "FLASH2_PF11_SPLIT": "0"}, "FLASH2_PF11_SPLIT", "a folded switch's off value"),
                                  ({"GEMM256_LDS_PF": "1"}, "GEMM256_LDS_PF", "a refuted switch on"),
                                  ({"FLASH2_SXH": "1"}, "FLASH2_SXH", "a folded child without its parent FLASH2_SX11"),
                                  ({"KV8": "1", "KV8_SPLIT_VZERO": "0"}, "KV8_SPLIT_VZERO", "the KV8 guard switched off"),
                                  ({"IQ4_Q8": ""}, "IQ4_Q8", "a folded default-on switch set empty (its old off spelling)")):
                rc, err, _ = run(script, sw, tmp, cfg)
                check(refused(rc, err, name), f"{script}: {why} ({' '.join(f'{k}={v}' for k, v in sw.items())}) is refused by name "
                      f"(exit {rc}: {err.strip().splitlines()[-1][:120] if err.strip() else ''})")
        rc, err, _ = run("gen_llm.py", {"KV8_SPLIT_VZERO": "1"}, tmp)
        check("retired switch" not in err, "gen_llm.py: KV8_SPLIT_VZERO=1 without KV8 (the f16 clamp, still a switch) is not refused")
        for vs in (False, True):
            extra = {"FLASH2_ROLE": "partial", "FLASH2_SPLITK": "1", "FLASH2_SPLITK_FROM": "2048"} if vs else {}
            r0, e0, t0 = run("gen_flash2s.py", {**new, **extra}, tmp, cfg)
            r1, e1, t1 = run("gen_flash2s.py", {**OLD, **extra}, tmp, cfg)
            check(r0 == 0 and r1 == 0 and t0 == t1 and "%sxh_view" in t0,
                  f"gen_flash2s.py {'vsplit' if vs else 'flash2'}: the pre-fold recipe passes and generates the folded recipe's text "
                  f"(exit {r0} / {r1}{': ' + (e0 or e1).strip()[-160:] if r0 or r1 else ''})")
    print(f"{n - fails}/{n} passed"); return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
