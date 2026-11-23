#!/usr/bin/env python3
"""The IQ4 codebook lookup compiles to a byte permute, not a select chain (HRX-like E229/E230). Recurrence it prevents: masking
the nibble indices as i32 words (andi 0x0f0f0f0f, then bitcast to bytes) hides from the compiler that each byte index is < 16,
so vector.table.lookup became 480 v_cmp_eq_i32 + 480 v_cndmask per 64-wide K tile, 62 % of the m256 GEMM's hot loop.
Compiles gen_gemm's g_deq16_iq4_nl / _xs (the dequant16 snippet every IQ4 GEMM stages with) for gfx1151 with the machine's
Loom (VENV) and counts compares and selects with common/loop_census.disasm.
usage: python3 kit/test_iq4_lookup.py   (needs loom-compile: VENV from the environment or the machine env file)"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import gen_gemm  # noqa: E402
from loop_census import disasm  # noqa: E402
from kit_testlib import loom_compile  # noqa: E402

LIMIT = 40   # selects + compares left in a whole deq16 kernel (bounds and address checks); the select chain was 960 / 485


def main():
    lc = loom_compile()
    head = "target.decl @device\namdgpu.target<gfx11-generic> @rdna3\namdgpu.target<gfx12-generic> @rdna4\n\n"
    src = head + open(os.path.join(HERE, "quant_values.loom")).read() + gen_gemm.deq16("iq4_nl") + gen_gemm.deq16("iq4_xs")
    bad = 0
    with tempfile.TemporaryDirectory() as t:
        open(os.path.join(t, "p.loom"), "w").write(src)
        for k in ("g_deq16_iq4_nl", "g_deq16_iq4_xs"):
            out = os.path.join(t, k + ".hsaco")
            r = subprocess.run([lc, os.path.join(t, "p.loom"), "--target=amdgpu:gfx1151", f"--root={k}", f"--output={out}"],
                               capture_output=True, text=True)
            if r.returncode:
                print(f"FAIL {k}: compile failed: {r.stderr.strip()[:300]}"); bad += 1; continue
            ins, _ = disasm(out)
            sel = sum(1 for i in ins if i.startswith(("v_cndmask", "v_cmp_")))
            perm = sum(1 for i in ins if i.startswith("v_perm_b32"))
            ok = sel <= LIMIT and perm > 0
            bad += not ok
            print(("ok  " if ok else "FAIL") + f" {k}: {len(ins)} instructions, {sel} compares + selects (limit {LIMIT}), {perm} v_perm_b32")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
