#!/usr/bin/env python3
"""The 256-row GEMMs keep their next-tile prefetch in flight under the served recipe (HRX-like E242). Recurrence it prevents: with
GEMM_SKIP_PADDING's kernel-wide scf.if, an activation prefetch loaded inside its own scf.if (else: the carried value) made Loom
0.2.0 wait on it at the join (vmcnt(0)) and shift it with 32 moves before the 64 WMMAs: no latency hidden on any tile of the
q6_k and iq4_xs GEMMs (and a 2.4 KB spill in iq4_nl's). Generates the kernels as the build does (gen_gemm.py with the padding
skip and N-fastest order), compiles them for gfx1151 with the machine's Loom and reads them with common/kernel_lint.py.
usage: python3 kit/test_gemm_prefetch.py"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import kernel_lint as kl  # noqa: E402

CASES = ("q6_k:store", "q6_k:swiglu", "iq4_xs:store", "iq4_nl:store")


def main():
    lc = kl.loom_compile()
    env = dict(os.environ, GEMM_FORMATS=",".join(CASES), GEMM_SKIP_PADDING="1", GEMM_N_FASTEST="1", LOOM_RDNA4="0")
    text = subprocess.run([sys.executable, os.path.join(HERE, "gen_gemm.py")], capture_output=True, text=True, env=env, check=True).stdout
    head = "target.decl @device\namdgpu.target<gfx11-generic> @rdna3\n\n" + open(os.path.join(HERE, "quant_values.loom")).read()
    items = {m.group(1): it for it in re.split(r"(?m)^(?=kernel\.def )", text)
             for m in [re.match(r'kernel\.def (?:target\(@\w+\) )?export\("(\w+)"\)', it)] if m}
    bad = 0
    with tempfile.TemporaryDirectory() as t:
        for case in CASES:
            fmt, mode = case.split(":")
            k = f"g_gemm_{fmt}_{mode}_m256"
            open(os.path.join(t, k + ".loom"), "w").write(head + items[k])
            out = os.path.join(t, k + ".hsaco")
            r = subprocess.run([lc, os.path.join(t, k + ".loom"), "--target=amdgpu:gfx1151", f"--root={k}", f"--output={out}"],
                               capture_output=True, text=True)
            if r.returncode:
                print(f"FAIL {k}: compile: {r.stderr.strip()[:300]}"); bad += 1; continue
            x = kl.lint(out)
            ok = x["pf"] == "kept" and x["scratch"] == 0
            bad += not ok
            print(("ok  " if ok else "FAIL") + f" {k}: prefetch {x['pf']}, scratch {x['scratch']}, loop {x['loop']}, longest move run {x['run']}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
