#!/usr/bin/env python3
"""kit/gen_sconv.py against ggml's short convolution, statically (HRX-like E273). Recurrences it prevents:
  - a tap applied to the wrong lag or window position (ggml_ssm_conv: out_t = sum_k w[k] x[t + k] over the window [history; span]; the
    dynamic taps: lag 0 = token t itself), the residual or the 1 / sqrt((dyn ? 2 : 1) n_conv + 1) scale wrong, or the new history not
    the window's last n_conv - 1 inputs;
  - a summation order other than ggml's (each sum term by term, the dynamic lags added one at a time after the static taps), so outputs
    match llama.cpp's bit for bit.
How: kit/loomsim.py runs the emitted kernel (all lanes at once, float32; the kernel uses only * and +, so results are bit-exact for its
operation order) and its outputs and new history are compared bit for bit with a float32 transcription of the fork's convolve()
(src/models/inkling2.cpp). The kernels also compile for gfx1151 with no lint flag.
usage: python3 kit/test_sconv.py"""
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import gen_sconv  # noqa: E402
import loomsim  # noqa: E402

fails = 0
f32 = np.float32


def ok(cond, what):
    global fails
    print(("ok  " if cond else "FAIL") + " " + what)
    fails += not cond


def reference(cur, w, hist, taps, n_conv=4):
    """the fork's convolve() in float32, ggml's order"""
    T, C = cur.shape
    heads = taps.shape[1] if taps is not None else 0
    x = np.concatenate([hist, cur.T], axis=1).astype(f32)            # [C][P + T]
    out = np.zeros((T, C), f32)
    for t in range(T):
        for c in range(C):
            s = f32(0)
            for k in range(n_conv):
                s = f32(s + f32(x[c, t + k] * w[c, k]))
            if heads:
                h = c // (C // heads)
                for lag in range(n_conv):
                    s = f32(s + f32(x[c, t + n_conv - 1 - lag] * taps[t, h, lag]))
            out[t, c] = s
    n = (2 if heads else 1) * n_conv + 1
    scale = f32(f32(1) / np.sqrt(f32(n)))
    y = ((cur + out).astype(f32) * scale).astype(f32)
    return y, x[:, T:T + n_conv - 1] if T >= n_conv - 1 else x[:, -(n_conv - 1):]


def main():
    rng = np.random.default_rng(20261122)
    for C, T, heads in ((64, 1, 0), (64, 5, 0), (64, 1, 4), (64, 3, 4), (96, 8, 8)):
        cur = rng.standard_normal((T, C)).astype(f32)
        w = rng.standard_normal((C, 4)).astype(f32)
        hist = rng.standard_normal((C, 3)).astype(f32)
        taps = rng.standard_normal((T, heads, 4)).astype(f32) if heads else None
        y_ref, h_ref = reference(cur, w, hist, taps)
        arrays = {"cur": cur.copy(), "weight": w, "history": hist.copy(), "out": np.zeros((T, C), f32)}
        if heads:
            arrays["taps"] = taps
        loomsim.run(gen_sconv.sconv(C, T, heads), arrays, {}, ((C + 255) // 256, 1, 1), 256)
        ok(np.array_equal(arrays["out"].view(np.uint32), y_ref.view(np.uint32)) and
           np.array_equal(arrays["history"].view(np.uint32), h_ref.astype(f32).view(np.uint32)),
           f"C {C} T {T} heads {heads}: outputs and new history equal ggml's bit for bit")
    import kernel_lint as kl
    with tempfile.TemporaryDirectory() as t:
        for C, T, heads in ((2304, 1, 0), (1024, 8, 8)):
            p = os.path.join(t, "s.loom")
            open(p, "w").write("target.decl @device\namdgpu.target<gfx11-generic> @rdna3\n" + gen_sconv.sconv(C, T, heads))
            o = os.path.join(t, "s.hsaco")
            name = f"g_sconv_c{C}_t{T}" + (f"_dyn{heads}" if heads else "")
            r = subprocess.run([kl.loom_compile(), p, "--target=amdgpu:gfx1151", f"--root={name}", f"--output={o}"], capture_output=True, text=True)
            if r.returncode:
                sys.exit(f"FAIL compile {name}: {r.stderr.strip()[:300]}")
            ok(not kl.lint(o)["flags"], f"{name} compiles for gfx1151, no lint flag")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
