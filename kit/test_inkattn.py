#!/usr/bin/env python3
"""kit/gen_inkattn.py prep against the fork's math, statically (HRX-like E274). Recurrences it prevents:
  - a clip applied before its per-channel scale or with the wrong bound (q / k: qk_clip 22.63, r: rel_clip 8; norm + 1e-6, factor
    clamped to [0, 1]), v not RMS-normalised (or with the wrong length), r's padded lanes (16 of 32) leaking into its norm;
  - tau wrong or applied on local layers (global only: 1 + alpha log(max(1, (pos + 1) / floor)), on q and r, after the clip);
  - heads or waves mixed up (each of 8 waves owns 3 q heads, 1 k, 1 v, 3 r heads).
decode_swa (the sliding-window decode attention) and decode_global (the global layers' split-K form: chunk + combine kernels):
  - the bias at the wrong distance or the window edge off (distance < 1024: n_kv 1,500 checks the cut), GQA heads mixed up, the
    zero-logit sink left out of the softmax (heads with large sink logits, where it dominates the denominator), the gate missing;
    on global layers the bias past distance 1024 not zero (n_kv 1,300), or the chunks combined without the sink;
  - with kv8=True (int8 codes + one f32 scale per position and KV head, the Qwen kit's KV8 format; E287): a scale applied to the
    wrong key or head, the V scale missing, codes widened as unsigned: compared with the same attention on the dequantised cache.
How: kit/loomsim.py runs the emitted kernels (all lanes of a workgroup at once, in float32) and the results are compared with numpy
transcriptions of inkling2.cpp's clip() / rms_norm / tau and of its attention (scores + r . rel_bias[d] - sink, softmax with a zero sink,
sigmoid gate). Summation order differs from ggml's: prep to 2e-6 relative, attention to 1e-5 of the output's RMS. The kernels compile with no lint flag.
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
    """one token's attention, as the fork: s = q . k / sqrt(128) + r . rel_bias[d] - sink (the bias 0 past distance 1024: the zero pad
    row), over distances d < W (W None: all keys, the global layers), softmax with a zero-logit sink (max over the scores and 0), the
    output times sigmoid(gate); GQA: kv head = h / 3"""
    out = np.zeros((24, 128), np.float64)
    for h in range(24):
        kvh = h // 3
        d = np.arange(n_kv if W is None else min(n_kv, W))
        j = n_kv - 1 - d
        bias = np.where(d < 1024, rb[np.minimum(d, 1023)].astype(np.float64) @ r[h], 0.0)
        s = kc[j, kvh].astype(np.float64) @ q[h] / np.sqrt(128) + bias - sinks[h]
        m = max(0.0, s.max())
        p = np.exp(s - m)
        out[h] = (p @ vc[j, kvh]) / (np.exp(-m) + p.sum()) / (1 + np.exp(-gate[h]))
    return out


def span_values(rng):
    """decode_swa_span (E298) on kit/loomsim.py: every token t of the span equals the one-token reference at its own position
    (n_kv - T + 1 + t keys visible), for spans of 1, 3 and 8, a context shorter than the span's windows (keys before 0 empty), one
    just past the window and one far past it. Recurrences: a token seeing a later token's key (causality), a window off by one at
    either end, a bias row taken at the wrong distance, one token's probabilities applied to another's values."""
    for T, n_kv in ((1, 700), (3, 9), (8, 1030), (8, 1500)):
        q = rng.standard_normal((T, 24, 128)).astype(f32)
        r = rng.standard_normal((T, 24, 16)).astype(f32)
        sinks = rng.standard_normal(24).astype(f32)
        sinks[::3] += f32(12)
        gate = rng.standard_normal((T, 24, 128)).astype(f32)
        rb = (rng.standard_normal((1024, 16)) * 0.3).astype(f32)
        kc = rng.standard_normal((n_kv, 8, 128)).astype(f32)
        vc = rng.standard_normal((n_kv, 8, 128)).astype(f32)
        out = np.zeros((T, 24, 128), f32)
        loomsim.run(gen_inkattn.decode_swa_span(T), {"q": q, "r": r, "sinks": sinks, "gate": gate, "rel_bias": rb, "kc": kc, "vc": vc,
                    "out": out}, {"n_kv_arg": n_kv}, (24, 1, 1), 256)
        errs = []
        for t in range(T):
            nt = n_kv - T + 1 + t
            ref = attention_reference(nt, q[t], r[t], sinks, gate[t], rb, kc[:nt], vc[:nt])
            errs.append(float(np.max(np.abs(out[t] - ref)) / np.sqrt((ref ** 2).mean())))
        ok(max(errs) < 1e-5, f"decode_swa_span T {T} n_kv {n_kv}: every token equals the one-token reference at its position "
                             f"(max err / rms {max(errs):.1e})")
    # KV8: int8 codes + one f32 scale per (position, KV head); the reference sees the dequantised cache (codes x scale)
    for T, n_kv in ((3, 9), (8, 1500)):
        q = rng.standard_normal((T, 24, 128)).astype(f32)
        r = rng.standard_normal((T, 24, 16)).astype(f32)
        sinks = rng.standard_normal(24).astype(f32)
        gate = rng.standard_normal((T, 24, 128)).astype(f32)
        rb = (rng.standard_normal((1024, 16)) * 0.3).astype(f32)
        kf = (rng.standard_normal((n_kv, 8, 128)) * rng.uniform(0.5, 3, (n_kv, 8, 1))).astype(f32)
        vf = (rng.standard_normal((n_kv, 8, 128)) * rng.uniform(0.5, 3, (n_kv, 8, 1))).astype(f32)
        ks = (np.abs(kf).max(-1) / 127).astype(f32)
        vs = (np.abs(vf).max(-1) / 127).astype(f32)
        kc8 = np.round(kf / ks[..., None]).astype(np.int8)
        vc8 = np.round(vf / vs[..., None]).astype(np.int8)
        out = np.zeros((T, 24, 128), f32)
        loomsim.run(gen_inkattn.decode_swa_span(T, kv8=True), {"q": q, "r": r, "sinks": sinks, "gate": gate, "rel_bias": rb, "kc": kc8,
                    "vc": vc8, "ksc": ks, "vsc": vs, "out": out}, {"n_kv_arg": n_kv}, (24, 1, 1), 256)
        kd = (kc8.astype(f32) * ks[..., None]).astype(f32)
        vd = (vc8.astype(f32) * vs[..., None]).astype(f32)
        errs = []
        for t in range(T):
            nt = n_kv - T + 1 + t
            ref = attention_reference(nt, q[t], r[t], sinks, gate[t], rb, kd[:nt], vd[:nt])
            errs.append(float(np.max(np.abs(out[t] - ref)) / np.sqrt((ref ** 2).mean())))
        ok(max(errs) < 1e-5, f"decode_swa_span kv8 T {T} n_kv {n_kv}: equals the reference on the dequantised cache (max err / rms {max(errs):.1e})")


def global_span_values(rng):
    """decode_global_span (E298) on kit/loomsim.py: chunks + combine for a span, every token equal to the one-token global reference
    at its own position (W None: all earlier keys, bias 0 past distance 1024), a span inside one chunk and one crossing chunks, f32 and
    KV8. Recurrences: a token seeing a later key (the span's own later tokens are in the cache), a chunk with no visible key adding
    weight, partials of one token combined with another's."""
    for T, n_kv, kv in ((3, 300, False), (8, 1300, False), (8, 1300, True)):
        q = rng.standard_normal((T, 24, 128)).astype(f32)
        r = rng.standard_normal((T, 24, 16)).astype(f32)
        sinks = rng.standard_normal(24).astype(f32)
        sinks[::3] += f32(12)
        gate = rng.standard_normal((T, 24, 128)).astype(f32)
        rb = (rng.standard_normal((1024, 16)) * 0.3).astype(f32)
        kf = (rng.standard_normal((n_kv, 8, 128)) * rng.uniform(0.5, 3, (n_kv, 8, 1))).astype(f32)
        vf = (rng.standard_normal((n_kv, 8, 128)) * rng.uniform(0.5, 3, (n_kv, 8, 1))).astype(f32)
        bufs = {"q": q, "r": r, "sinks": sinks, "rel_bias": rb}
        if kv:
            ks = (np.abs(kf).max(-1) / 127).astype(f32)
            vs = (np.abs(vf).max(-1) / 127).astype(f32)
            kc8 = np.round(kf / ks[..., None]).astype(np.int8)
            vc8 = np.round(vf / vs[..., None]).astype(np.int8)
            bufs.update(kc=kc8, vc=vc8, ksc=ks, vsc=vs)
            kd, vd = (kc8.astype(f32) * ks[..., None]).astype(f32), (vc8.astype(f32) * vs[..., None]).astype(f32)
        else:
            bufs.update(kc=kf, vc=vf)
            kd, vd = kf, vf
        part = np.zeros((T, 24, 4096, 130), f32)
        bufs["part"] = part
        out = np.zeros((T, 24, 128), f32)
        ptext, ctext = gen_inkattn.decode_global_span(T, kv8=kv)
        loomsim.run(ptext, bufs, {"n_kv_arg": n_kv}, (24, (n_kv + 255) // 256, 1), 256)
        loomsim.run(ctext, {"part": part, "gate": gate, "out": out}, {"n_kv_arg": n_kv}, (24, T, 1), 128)
        errs = []
        for t in range(T):
            nt = n_kv - T + 1 + t
            ref = attention_reference(nt, q[t], r[t], sinks, gate[t], rb, kd[:nt], vd[:nt], W=None)
            errs.append(float(np.max(np.abs(out[t] - ref)) / np.sqrt((ref ** 2).mean())))
        ok(max(errs) < 1e-5, f"decode_global_span T {T} n_kv {n_kv}{' kv8' if kv else ''}: every token equals the global reference at its "
                             f"position (max err / rms {max(errs):.1e})")


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
        err = float(np.max(np.abs(out - ref)) / np.sqrt((ref ** 2).mean()))   # max abs error over the output's RMS (scale-free)
        ok(err < 1e-5, f"decode_swa n_kv {n_kv}: equals the fork's attention (bias, window, zero sink, gate; max err / rms {err:.1e})")
    for n_kv in (300, 1300):
        q = rng.standard_normal((24, 128)).astype(f32)
        r = rng.standard_normal((24, 16)).astype(f32)
        sinks = rng.standard_normal(24).astype(f32)
        sinks[::3] += f32(12)
        gate = rng.standard_normal((24, 128)).astype(f32)
        rb = (rng.standard_normal((1024, 16)) * 0.3).astype(f32)
        kc = rng.standard_normal((n_kv, 8, 128)).astype(f32)
        vc = rng.standard_normal((n_kv, 8, 128)).astype(f32)
        part = np.zeros((24, 4096, 130), f32)
        out = np.zeros((24, 128), f32)
        ptext, ctext = gen_inkattn.decode_global()
        loomsim.run(ptext, {"q": q, "r": r, "sinks": sinks, "rel_bias": rb, "kc": kc, "vc": vc, "part": part}, {"n_kv_arg": n_kv},
                    (24, (n_kv + 255) // 256, 1), 256)
        loomsim.run(ctext, {"part": part, "gate": gate, "out": out}, {"n_kv_arg": n_kv}, (24, 1, 1), 128)
        ref = attention_reference(n_kv, q, r, sinks, gate, rb, kc, vc, W=None)
        err = float(np.max(np.abs(out - ref)) / np.sqrt((ref ** 2).mean()))
        ok(err < 1e-5, f"decode_global n_kv {n_kv}: chunks + combine equal the fork's attention (max err / rms {err:.1e})")
    def kv8(x):
        """the Qwen kit's KV8: scale = f16(max|x| / 127) per (position, head), code = round(x / scale)"""
        sc = (np.abs(x).max(-1) / 127).astype(np.float16).astype(f32)
        codes = np.where(sc[..., None] > 0, np.round(x / np.where(sc[..., None] > 0, sc[..., None], 1)), 0).clip(-127, 127).astype(np.int8)
        return codes, sc
    for kind, n_kv in (("swa", 1500), ("global", 1300)):
        q = rng.standard_normal((24, 128)).astype(f32)
        r = rng.standard_normal((24, 16)).astype(f32)
        sinks = rng.standard_normal(24).astype(f32)
        sinks[::3] += f32(12)
        gate = rng.standard_normal((24, 128)).astype(f32)
        rb = (rng.standard_normal((1024, 16)) * 0.3).astype(f32)
        kcf = (rng.standard_normal((n_kv, 8, 128)) * rng.uniform(0.3, 3, (n_kv, 8, 1))).astype(f32)
        vcf = (rng.standard_normal((n_kv, 8, 128)) * rng.uniform(0.3, 3, (n_kv, 8, 1))).astype(f32)
        kc, ksc = kv8(kcf)
        vc, vsc = kv8(vcf)
        out = np.zeros((24, 128), f32)
        if kind == "swa":
            loomsim.run(gen_inkattn.decode_swa(kv8=True), {"q": q, "r": r, "sinks": sinks, "gate": gate, "rel_bias": rb, "kc": kc, "vc": vc,
                        "ksc": ksc, "vsc": vsc, "out": out}, {"n_kv_arg": n_kv}, (24, 1, 1), 256)
        else:
            part = np.zeros((24, 4096, 130), f32)
            ptext, ctext = gen_inkattn.decode_global(kv8=True)
            loomsim.run(ptext, {"q": q, "r": r, "sinks": sinks, "rel_bias": rb, "kc": kc, "vc": vc, "ksc": ksc, "vsc": vsc, "part": part},
                        {"n_kv_arg": n_kv}, (24, (n_kv + 255) // 256, 1), 256)
            loomsim.run(ctext, {"part": part, "gate": gate, "out": out}, {"n_kv_arg": n_kv}, (24, 1, 1), 128)
        deq = lambda c, sc: (c.astype(f32) * sc[..., None]).astype(f32)
        kd, vd = deq(kc, ksc), deq(vc, vsc)
        ref = attention_reference(n_kv, q, r, sinks, gate, rb, kd, vd, W=1024 if kind == "swa" else None)
        err = float(np.max(np.abs(out - ref)) / np.sqrt((ref ** 2).mean()))
        # bound: the f32 kernel's own error on the same dequantised cache (float32 attention over 1024+ keys with sink-dominated heads
        # reaches ~1e-5 by itself: E304 measured 4-8e-6 for both kernels on one cache). Codes x f16 scales are exact in f32, so both
        # kernels evaluate the same values in different orders (KV8 scales after the dot): neither is the other's rounding reference
        o32 = np.zeros((24, 128), f32)
        if kind == "swa":
            loomsim.run(gen_inkattn.decode_swa(), {"q": q, "r": r, "sinks": sinks, "gate": gate, "rel_bias": rb, "kc": kd, "vc": vd, "out": o32},
                        {"n_kv_arg": n_kv}, (24, 1, 1), 256)
        else:
            p32 = np.zeros((24, 4096, 130), f32)
            pt32, ct32 = gen_inkattn.decode_global()
            loomsim.run(pt32, {"q": q, "r": r, "sinks": sinks, "rel_bias": rb, "kc": kd, "vc": vd, "part": p32}, {"n_kv_arg": n_kv},
                        (24, (n_kv + 255) // 256, 1), 256)
            loomsim.run(ct32, {"part": p32, "gate": gate, "out": o32}, {"n_kv_arg": n_kv}, (24, 1, 1), 128)
        e32 = float(np.max(np.abs(o32 - ref)) / np.sqrt((ref ** 2).mean()))
        ok(err < max(1e-5, 2 * e32), f"decode_{kind} kv8 n_kv {n_kv}: equals the attention on the dequantised cache within the f32 kernel's own "
                                       f"error (max err / rms {err:.1e}; f32 kernel on the same cache {e32:.1e})")
    import kernel_lint as kl
    with tempfile.TemporaryDirectory() as t:
        for name, text in [(f"g_ink_attn_prep_t{T}" + ("_global" if glob else ""), gen_inkattn.prep(T, glob)) for T, glob in ((1, False), (8, True))] + \
                [("g_ink_attn_dec_swa1024", gen_inkattn.decode_swa()), ("g_ink_attn_dec_glob_part", "".join(gen_inkattn.decode_global())),
                 ("g_ink_attn_dec_glob_comb", "".join(gen_inkattn.decode_global())), ("g_ink_attn_dec_swa1024_kv8", gen_inkattn.decode_swa(kv8=True)),
                 ("g_ink_attn_dec_glob_part_kv8", "".join(gen_inkattn.decode_global(kv8=True)))]:
            p = os.path.join(t, "p.loom")
            open(p, "w").write("target.decl @device\namdgpu.target<gfx11-generic> @rdna3\n" + text)
            o = os.path.join(t, "p.hsaco")
            r_ = subprocess.run([kl.loom_compile(), p, "--target=amdgpu:gfx1151", f"--root={name}", f"--output={o}"], capture_output=True, text=True)
            if r_.returncode:
                sys.exit(f"FAIL compile {name}: {r_.stderr.strip()[:300]}")
            ok(not kl.lint(o)["flags"], f"{name} compiles for gfx1151, no lint flag")
    span_values(rng)
    global_span_values(rng)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
