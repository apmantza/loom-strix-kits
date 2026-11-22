#!/usr/bin/env python3
"""kit/gen_loop.py controller, static (HRX-like E271). Recurrences it prevents:
  - the controller weights read more than once per token: at Inkling 2 Mini's shapes they are ~226 MB per token (4 controllers x 12 core
    layers x 2 passes x 2.4 MB, f32), ~1 ms at 228 GB/s, so a second pass over them costs a whole extra millisecond per token (the
    decode cost a GDN second state pass showed, E185). D's rows are read once (36 128-bit loads per row per lane in the row loop), G and
    F once (H / 256 outputs x rank / 16 chunks x 2 matrices x 4 loads);
  - the kernel spilling or scalarised (E230 shapes): it compiles for gfx1151 with no lint flag and no scratch.
The arithmetic gets a GPU value test against llama.cpp's control() later.
usage: python3 kit/test_loop.py   (needs loom-compile: VENV)"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import gen_loop  # noqa: E402
import kernel_lint as kl  # noqa: E402

fails = 0


def ok(cond, what):
    global fails
    print(("ok  " if cond else "FAIL") + " " + what)
    fails += not cond


def main():
    H, rank = 2304, 64
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "c.loom")
        open(p, "w").write("target.decl @device\namdgpu.target<gfx11-generic> @rdna3\n" + gen_loop.controller(H, rank))
        o = os.path.join(t, "c.hsaco")
        r = subprocess.run([kl.loom_compile(), p, "--target=amdgpu:gfx1151", "--root=g_loop_control", f"--output={o}",
                            "--compile-report=summary", f"--compile-report-output={o[:-6]}.json"], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"FAIL compile: {r.stderr.strip()[:300]}")
        ins = kl.disasm(o)[0]
        x = kl.lint(o)
        b128 = sum(1 for i in ins if i.startswith("global_load_b128"))
        want = 2 * H // (32 * 4) + 2 * (H // 256) * (rank // 16) * 4
        ok(b128 == want, f"controller weights read once: {b128} 128-bit loads (want {want}: D row loop {2 * H // 128}, G and F {want - 2 * H // 128})")
        ok(not x["flags"] and not x["scratch"], f"no lint flag ({x['flags']}), no scratch, {x['vgpr']} VGPRs, {x['waves']} waves")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
