#!/usr/bin/env python3
"""K-quant code assembly compiles to word operations, not per-byte extract and repack (HRX-like E233). Recurrence it prevents:
a helper written as vector<32xi8> shifts by a runtime amount made the compiler extract every byte (v_bfe_u32) and repack it
(v_perm_b32): the 8-token q6_k decode matvec loop was 616 instructions against q5_k's 261 for the same 64 dot4, q3_k's 637.
Compiles each helper (@g_q6_k_icodes, @g_q6_k_values from gen_llm.py; @g_q3_k_icodes, @g_q3_k_values from quant_values.loom) in a
one-group-per-lane kernel for gfx1151 with the machine's Loom (VENV) and counts the byte-repack instructions with
common/loop_census.disasm.
usage: python3 kit/test_word_codes.py   (needs loom-compile: VENV from the environment or the machine env file)"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from loop_census import disasm  # noqa: E402
from kit_testlib import helper, kernel, loom_compile  # noqa: E402

LIMIT = 8   # v_bfe_u32 + v_perm_b32 left in the whole kernel; the byte form had 54 + 48 per group in the matvec loop


def main():
    lc = loom_compile()
    qv = open(os.path.join(HERE, "quant_values.loom")).read()
    prog = ("target.decl @device\namdgpu.target<gfx11-generic> @rdna3\n\n" + qv
            + helper(open(os.path.join(HERE, "gen_llm.py")).read(), "g_q6_k_icodes")
            + helper(open(os.path.join(HERE, "gen_llm.py")).read(), "g_q6_k_values")
            + kernel("g_q6_k_icodes", False) + kernel("g_q6_k_values", True) + kernel("g_q3_k_icodes", False) + kernel("g_q3_k_values", True))
    bad = 0
    with tempfile.TemporaryDirectory() as t:
        open(os.path.join(t, "p.loom"), "w").write(prog)
        for h in ("g_q6_k_icodes", "g_q6_k_values", "g_q3_k_icodes", "g_q3_k_values"):
            out = os.path.join(t, h + ".hsaco")
            r = subprocess.run([lc, os.path.join(t, "p.loom"), "--target=amdgpu:gfx1151", f"--root=t_{h}", f"--output={out}"],
                               capture_output=True, text=True)
            if r.returncode:
                print(f"FAIL {h}: compile failed: {r.stderr.strip()[:300]}"); bad += 1; continue
            ins, _ = disasm(out)
            rep = sum(1 for i in ins if i.startswith(("v_bfe_u32", "v_perm_b32")))
            ok = rep <= LIMIT
            bad += not ok
            print(("ok  " if ok else "FAIL") + f" {h}: {len(ins)} instructions, {rep} v_bfe_u32 + v_perm_b32 (limit {LIMIT})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
