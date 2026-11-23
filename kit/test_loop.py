#!/usr/bin/env python3
"""kit/gen_loop.py controller, static (HRX-like E271). Recurrences it prevents:
  - the controller weights read more than once per token: at Inkling 2 Mini's shapes they are ~226 MB per token (4 controllers x 12 core
    layers x 2 passes x 2.4 MB, f32), ~1 ms at 228 GB/s, so a second pass over them costs a whole extra millisecond per token (the
    decode cost a GDN second state pass showed, E185). D's rows are read once (36 128-bit loads per row per lane in the row loop), G and
    F once (H / 256 outputs x rank / 16 chunks x 2 matrices x 4 loads);
  - the kernel spilling or scalarised (E230 shapes): it compiles for gfx1151 with no lint flag and no scratch.
controller(wty="bf16") (E287: Inkling's controller weights are exact BF16):
  - the BF16 form not bit-identical to the f32 one on BF16-exact weights (a widening that rounds, a chunk read from the wrong half);
  - it not halving the weight bytes (its 128-bit weight loads must be half the f32 form's).
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


def split_values(rng, H=2304, rank=64):
    """controller_split (E297) on kit/loomsim.py against the fork's control() and against controller(): z kernel then out kernel, for
    spans of 1, 3 and 8 tokens, f32 and BF16 weights. Recurrences: a lane's chunk taken from the wrong half (cur vs msg), the norm
    applied to the wrong sum, a row of z written to the wrong token or row, a token in the span reading another's z."""
    rms = lambda x: x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-6)
    bf = lambda a: (a.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)
    hw = lambda a: (a.view(np.uint32) >> 16).astype(np.uint16)
    for TS, wty in ((1, "f32"), (3, "bf16"), (8, "bf16")):
        cur = rng.standard_normal((TS, H)).astype(np.float32)
        msg = (rng.standard_normal((TS, H)) * 2).astype(np.float32)
        D = bf((rng.standard_normal((rank, 2 * H)) * 0.02).astype(np.float32))
        G = bf((rng.standard_normal((H, rank)) * 0.3).astype(np.float32))
        Fm = bf((rng.standard_normal((H, rank)) * 0.3).astype(np.float32))
        w = (lambda a: a) if wty == "f32" else hw
        text = gen_loop.controller_split(H, rank, TS=TS, wty=wty)
        parts = text.split("\nkernel.def ")
        zt, ot = "kernel.def " + parts[1], "kernel.def " + parts[2]
        z = np.zeros((TS, rank), np.float32)
        loomsim.run(zt, {"cur": cur, "msg": msg, "d": w(D), "z": z}, {}, (rank // 4, 1, 1), 2 * H // 16)
        out = np.zeros((TS, H), np.float32)
        loomsim.run(ot, {"cur": cur, "msg": msg, "z": z, "g": w(G), "f": w(Fm), "out": out}, {}, (H // 64, 1, 1), 64)
        zref = np.concatenate([rms(cur), rms(msg)], -1) @ D.T.astype(np.float64)
        ref = cur - np.tanh(zref @ Fm.T) * cur + np.tanh(zref @ G.T) * msg
        ez = float(np.max(np.abs(z - zref)) / np.sqrt((zref ** 2).mean()))
        err = float(np.max(np.abs(out - ref)) / np.sqrt((ref ** 2).mean()))
        old = np.zeros((TS, H), np.float32)
        loomsim.run(gen_loop.controller(H, rank), {"cur": cur, "msg": msg, "d": D, "g": G, "f": Fm, "out": old}, {"tokens_arg": TS}, (TS, 1, 1), 256)
        eo = float(np.max(np.abs(out - old)) / np.sqrt((old ** 2).mean()))
        ok(ez < 1e-5 and err < 1e-5 and eo < 1e-5,
           f"controller_split span {TS} {wty}: z err/rms {ez:.1e}, out vs control() {err:.1e}, vs controller() {eo:.1e}")


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
        bf = lambda a: (a.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)       # BF16-exact f32 values
        Db, Gb, Fb = bf(D.copy()), bf(G.copy()), bf(F.copy())
        o32 = np.zeros((T, H), np.float32)
        o16 = np.zeros((T, H), np.float32)
        loomsim.run(gen_loop.controller(H, rank), {"cur": cur, "msg": msg, "d": Db, "g": Gb, "f": Fb, "out": o32}, {"tokens_arg": T}, (T, 1, 1), 256)
        hw = lambda a: (a.view(np.uint32) >> 16).astype(np.uint16)                          # the BF16 words
        loomsim.run(gen_loop.controller(H, rank, wty="bf16"), {"cur": cur, "msg": msg, "d": hw(Db), "g": hw(Gb), "f": hw(Fb), "out": o16},
                    {"tokens_arg": T}, (T, 1, 1), 256)
        ok(np.array_equal(o32.view(np.uint32), o16.view(np.uint32)), "BF16 controller equals the f32 one bit for bit on BF16-exact weights")
        p = os.path.join(t, "cb.loom")
        open(p, "w").write("target.decl @device\namdgpu.target<gfx11-generic> @rdna3\n" + gen_loop.controller(H, rank, name="g_loop_control", wty="bf16"))
        o = os.path.join(t, "cb.hsaco")
        r = subprocess.run([kl.loom_compile(), p, "--target=amdgpu:gfx1151", "--root=g_loop_control", f"--output={o}",
                            "--compile-report=summary", f"--compile-report-output={o[:-6]}.json"], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"FAIL compile bf16 controller: {r.stderr.strip()[:300]}")
        x16 = kl.lint(o)
        b16 = sum(1 for i in kl.disasm(o)[0] if i.startswith("global_load_b128"))
        ok(b16 * 2 == want and not x16["flags"] and not x16["scratch"],
           f"BF16 controller: {b16} 128-bit weight loads (want {want // 2}, half of f32), flags {x16['flags']}, {x16['vgpr']} VGPRs")
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
    split_values(rng)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
