#!/usr/bin/env python3
"""kit/gen_inkattn.py prep against the fork's math, statically (HRX-like E274). Recurrences it prevents:
  - a clip applied before its per-channel scale or with the wrong bound (q / k: qk_clip 22.63, r: rel_clip 8; norm + 1e-6, factor
    clamped to [0, 1]), v not RMS-normalised (or with the wrong length), r's padded lanes (16 of 32) leaking into its norm;
  - tau wrong or applied on local layers (global only: 1 + alpha log(max(1, (pos + 1) / floor)), on q and r, after the clip);
  - heads or waves mixed up (each of 8 waves owns 3 q heads, 1 k, 1 v, 3 r heads).
decode_swa (the sliding-window decode attention):
  - the bias at the wrong distance or the window edge off (distance < 1024: n_kv 1,500 checks the cut), GQA heads mixed up, the
    zero-logit sink left out of the softmax (heads with large sink logits, where it dominates the denominator), the gate missing.
How: kit/loomsim.py runs the emitted kernels (all lanes of a workgroup at once, in float32) and the results are compared with numpy
transcriptions of inkling2.cpp's clip() / rms_norm / tau and of its attention (scores + r . rel_bias[d] - sink, softmax with a zero sink,
sigmoid gate). Summation order differs from ggml's: prep to 2e-6, attention to 2e-4 relative. The kernels compile with no lint flag.
usage: python3 kit/test_inkattn.py"""
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import gen_inkattn  # noqa: E402
import loomsim  # noqa: E402

f32 = np.float32
fails = 0


def ok(cond, what):
    global fails
    print(("ok  " if cond else "FAIL") + " " + what)
    fails += not cond


def reference(q, k, v, r, qc, rc, pos, glob, qk_clip=22.627416610717773, rel_clip=8.0, eps=9.999999974752427e-07):
    def clip(x, scale, thr):
        x = x * scale if scale is not None else x
        n = np.sqrt((x.astype(np.float64) ** 2).sum(-1, keepdims=True)) + 1e-6
        return x * np.clip(thr / n, 0, 1)
    qo, ko = clip(q, qc[None], qk_clip), clip(k, None, qk_clip)
    vo = v / np.sqrt((v.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps)
    ro = clip(r, rc[None], rel_clip)
    if glob:
        tau = 1 + 0.10000000149011612 * np.log(np.maximum(1, (pos + 1) / 8192.0))
        qo, ro = qo * tau[:, None, None], ro * tau[:, None, None]
    return qo, ko, vo, ro


def attention_reference(n_kv, q, r, sinks, gate, rb, kc, vc, W=1024):
    """one token's sliding-window attention, as the fork: s = q . k / sqrt(128) + r . rel_bias[d] - sink over distances d < W, softmax
    with a zero-logit sink (max over the scores and 0), the output times sigmoid(gate); GQA: kv head = h / 3"""
    out = np.zeros((24, 128), np.float64)
    for h in range(24):
        kvh = h // 3
        d = np.arange(min(n_kv, W))
        j = n_kv - 1 - d
        s = kc[j, kvh].astype(np.float64) @ q[h] / np.sqrt(128) + rb[d].astype(np.float64) @ r[h] - sinks[h]
        m = max(0.0, s.max())
        p = np.exp(s - m)
        out[h] = (p @ vc[j, kvh]) / (np.exp(-m) + p.sum()) / (1 + np.exp(-gate[h]))
    return out


def main():
    rng = np.random.default_rng(20261122)
    for T, glob in ((1, False), (3, True), (2, False)):
        q = (rng.standard_normal((T, 24, 128)) * 3).astype(f32)
        k = (rng.standard_normal((T, 8, 128)) * 3).astype(f32)
        v = rng.standard_normal((T, 8, 128)).astype(f32)
        r = (rng.standard_normal((T, 24, 16)) * 4).astype(f32)
        qc = rng.uniform(0.2, 2, (24, 128)).astype(f32)
        rc = rng.uniform(0.2, 2, (24, 16)).astype(f32)
        pos = np.array([7, 9000, 700000][:T], np.int32)
        q[0, 0] *= f32(0.01)              # one head below the clip bound (factor 1) to exercise the clamp
        want = reference(q, k, v, r, qc, rc, pos, glob)
        bufs = {"q": q.copy(), "k": k.copy(), "v": v.copy(), "r": r.copy(), "qclip": qc, "rclip": rc, "positions": pos}
        loomsim.run(gen_inkattn.prep(T, glob), bufs, {}, (T, 1, 1), 256)
        errs = [float(np.max(np.abs(bufs[n] - w) / (np.abs(w) + 1e-6))) for n, w in zip(("q", "k", "v", "r"), want)]
        ok(max(errs) < 2e-6, f"T {T} {'global' if glob else 'local'}: q k v r equal the fork's clips / norms / tau (max rel err {max(errs):.1e})")
    for n_kv in (700, 1500):
        q = rng.standard_normal((24, 128)).astype(f32)
        r = rng.standard_normal((24, 16)).astype(f32)
        sinks = rng.standard_normal(24).astype(f32)
        sinks[::3] += f32(12)                 # scores far below zero: the zero-logit sink dominates the denominator
        gate = rng.standard_normal((24, 128)).astype(f32)
        rb = (rng.standard_normal((1024, 16)) * 0.3).astype(f32)
        kc = rng.standard_normal((n_kv, 8, 128)).astype(f32)
        vc = rng.standard_normal((n_kv, 8, 128)).astype(f32)
        out = np.zeros((24, 128), f32)
        bufs = {"q": q, "r": r, "sinks": sinks, "gate": gate, "rel_bias": rb, "kc": kc, "vc": vc, "out": out}
        loomsim.run(gen_inkattn.decode_swa(), bufs, {"n_kv_arg": n_kv}, (24, 1, 1), 256)
        ref = attention_reference(n_kv, q, r, sinks, gate, rb, kc, vc)
        err = float(np.max(np.abs(out - ref) / (np.abs(ref) + 1e-3)))
        ok(err < 2e-4, f"decode_swa n_kv {n_kv}: equals the fork's attention (bias, window, zero sink, gate; max rel err {err:.1e})")
    import kernel_lint as kl
    with tempfile.TemporaryDirectory() as t:
        for name, text in [(f"g_ink_attn_prep_t{T}" + ("_global" if glob else ""), gen_inkattn.prep(T, glob)) for T, glob in ((1, False), (8, True))] + \
                [("g_ink_attn_dec_swa1024", gen_inkattn.decode_swa())]:
            p = os.path.join(t, "p.loom")
            open(p, "w").write("target.decl @device\namdgpu.target<gfx11-generic> @rdna3\n" + text)
            o = os.path.join(t, "p.hsaco")
            r_ = subprocess.run([kl.loom_compile(), p, "--target=amdgpu:gfx1151", f"--root={name}", f"--output={o}"], capture_output=True, text=True)
            if r_.returncode:
                sys.exit(f"FAIL compile {name}: {r_.stderr.strip()[:300]}")
            ok(not kl.lint(o)["flags"], f"{name} compiles for gfx1151, no lint flag")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
