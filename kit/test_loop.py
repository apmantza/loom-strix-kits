#!/usr/bin/env python3
"""kit/gen_loop.py controller, static (HRX-like E271). Recurrences it prevents:
  - the controller weights read more than once per token: at Inkling 2 Mini's shapes they are ~226 MB per token (4 controllers x 12 core
    layers x 2 passes x 2.4 MB, f32), ~1 ms at 228 GB/s, so a second pass over them costs a whole extra millisecond per token (the
    decode cost a GDN second state pass showed, E185). D's rows are read once (36 128-bit loads per row per lane in the row loop), G and
    F once (H / 256 outputs x rank / 16 chunks x 2 matrices x 4 loads);
  - the kernel spilling or scalarised (E230 shapes): it compiles for gfx1151 with no lint flag and no scratch.
depth_mix (the block attention residuals, loop messages and output query; ~100 calls per decode token):
  - a state read more than once ((S + NQ) x H / 256 loads per lane);
  - one workgroup reduction (two barriers) per sum: the first form had S + S x NQ of them (40 barriers at S 5, NQ 3) serialised on
    the decode path; the batched form has two.
Values (kit/loomsim.py, float32, against numpy transcriptions of the fork's control() and mix() in inkling2.cpp): the controller's
z = D [rms(cur); rms(msg)] and cur - tanh(F z) cur + tanh(G z) msg (a gain / forget swap, a norm over the wrong half or a missing
tanh fails), the mix's normalised states, softmax over learned-query scores and the mix of the normalised (not raw) states.
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
import loomsim  # noqa: E402
import numpy as np  # noqa: E402

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
        rng = np.random.default_rng(20261122)
        T = 2
        cur = rng.standard_normal((T, H)).astype(np.float32)
        msg = (rng.standard_normal((T, H)) * 2).astype(np.float32)
        D = (rng.standard_normal((rank, 2 * H)) * 0.02).astype(np.float32)
        G = (rng.standard_normal((H, rank)) * 0.3).astype(np.float32)
        F = (rng.standard_normal((H, rank)) * 0.3).astype(np.float32)
        out = np.zeros((T, H), np.float32)
        loomsim.run(gen_loop.controller(H, rank), {"cur": cur, "msg": msg, "d": D, "g": G, "f": F, "out": out}, {"tokens_arg": T}, (T, 1, 1), 256)
        rms = lambda x: x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-6)
        z = np.concatenate([rms(cur), rms(msg)], -1) @ D.T.astype(np.float64)
        ref = cur - np.tanh(z @ F.T) * cur + np.tanh(z @ G.T) * msg
        err = float(np.max(np.abs(out - ref)) / np.sqrt((ref ** 2).mean()))    # max abs error over the output's RMS (scale-free)
        ok(err < 1e-5, f"controller values equal the fork's control() (max err / rms {err:.1e})")
        S, NQ = 4, 3
        st = (rng.standard_normal((S, T, H)) * rng.uniform(0.5, 3, (S, 1, 1))).astype(np.float32)
        qu = (rng.standard_normal((NQ, H)) * 0.05).astype(np.float32)
        mo = np.zeros((T, NQ, H), np.float32)
        loomsim.run(gen_loop.depth_mix(H, S, NQ), {"states": st, "queries": qu, "out": mo}, {"tokens_arg": T}, (T, 1, 1), 256)
        n = rms(st)                                                          # [S, T, H]
        sc = np.einsum("sth,qh->tqs", n, qu.astype(np.float64)) / np.sqrt(H)
        p = np.exp(sc - sc.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        mref = np.einsum("tqs,sth->tqh", p, n)
        err = float(np.max(np.abs(mo - mref)) / np.sqrt((mref ** 2).mean()))
        ok(err < 1e-5, f"depth_mix values equal the fork's mix() (max err / rms {err:.1e})")
        for S, NQ in ((5, 3), (5, 4), (2, 1)):
            p = os.path.join(t, f"m{S}{NQ}.loom")
            open(p, "w").write("target.decl @device\namdgpu.target<gfx11-generic> @rdna3\n" + gen_loop.depth_mix(H, S, NQ))
            o = os.path.join(t, f"m{S}{NQ}.hsaco")
            r = subprocess.run([kl.loom_compile(), p, "--target=amdgpu:gfx1151", f"--root=g_depth_mix_s{S}_q{NQ}", f"--output={o}",
                                "--compile-report=summary", f"--compile-report-output={o[:-6]}.json"], capture_output=True, text=True)
            if r.returncode:
                sys.exit(f"FAIL compile mix {S} {NQ}: {r.stderr.strip()[:300]}")
            ins = kl.disasm(o)[0]
            x = kl.lint(o)
            loads = sum(1 for i in ins if i.startswith("global_load"))
            bars = sum(1 for i in ins if i.startswith("s_barrier"))
            ok(loads == (S + NQ) * (H // 256) and bars == 2 and not x["flags"] and not x["scratch"],
               f"depth_mix S {S} NQ {NQ}: {loads} loads (want {(S + NQ) * (H // 256)}: states once), {bars} barriers (want 2), "
               f"flags {x['flags']}, {x['vgpr']} VGPRs, {x['waves']} waves")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
