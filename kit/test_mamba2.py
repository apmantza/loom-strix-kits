#!/usr/bin/env python3
"""kit/gen_mamba2.py, static (HRX-like E263). Recurrences it prevents:
  - a second pass over the recurrent state: a decode SSM step is bandwidth-bound on its state (Bolt: 4 MiB per layer each way), so
    every extra read or write of it costs a whole state pass (E185: Qwen's g_gdn_fold, a second state pass, 1.3 ms of 3.6 ms per
    step). The probe must load and store each lane's state slice exactly once (N / RL / 4 128-bit loads and stores);
  - the step scalarised, spilled or copied (E230 / E179 shapes) or its residency dropped (E236): it compiles for gfx1151 with no
    kernel-lint flag at 16 waves per SIMD.
Values (kit/loomsim.py, float32): y and the new state equal a numpy transcription of llama.cpp's ssm_scan step (softplus(dt + bias),
dA = exp(sp A), s' = s dA + B (x sp), y = sum s' C + D x) within 1e-5 of the RMS: a wrong group (B / C shared by H / G heads), a row
split summed over the wrong lanes or D missing fails it.
usage: python3 kit/test_mamba2.py   (needs loom-compile: VENV)"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import gen_mamba2  # noqa: E402
import kernel_lint as kl  # noqa: E402
import loomsim  # noqa: E402
import numpy as np  # noqa: E402

fails = 0


def ok(cond, what):
    global fails
    print(("ok  " if cond else "FAIL") + " " + what)
    fails += not cond


def main():
    N, RL = 128, 8
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "p.loom")
        open(p, "w").write("target.decl @device\namdgpu.target<gfx11-generic> @rdna3\n\n" + gen_mamba2.probe(N=N, RL=RL))
        o = os.path.join(t, "p.hsaco")
        r = subprocess.run([kl.loom_compile(), p, "--target=amdgpu:gfx1151", "--root=g_mamba2_step_probe", f"--output={o}",
                            "--compile-report=summary", f"--compile-report-output={o[:-6]}.json"],   # residency for the lint
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"FAIL compile: {r.stderr.strip()[:300]}")
        ins = kl.disasm(o)[0]
        x = kl.lint(o)
        b128_loads = sum(1 for i in ins if i.startswith("global_load_b128"))
        b128_stores = sum(1 for i in ins if i.startswith("global_store_b128"))
        per = N // RL // 4                       # 128-bit packets of one lane's state slice (and of its B and C slices)
        ok(b128_stores == per and b128_loads == 3 * per,
           f"state moved once: {b128_stores} 128-bit stores (want {per}), {b128_loads} 128-bit loads (want {3 * per}: state, B, C)")
        ok(not x["flags"] and x["waves"] == 16, f"no lint flag ({x['flags']}), {x['waves']} waves per SIMD at {x['vgpr']} VGPRs")
    rng = np.random.default_rng(20261122)
    H, P, G = 128, 64, 8
    st = rng.standard_normal((H, P, N)).astype(np.float32)
    bufs = {"state": st.copy(), "x": rng.standard_normal((H, P)).astype(np.float32), "b": rng.standard_normal((G, N)).astype(np.float32),
            "c": rng.standard_normal((G, N)).astype(np.float32), "dt": rng.standard_normal(H).astype(np.float32),
            "dt_bias": rng.standard_normal(H).astype(np.float32), "a": -np.exp(rng.standard_normal(H)).astype(np.float32),
            "d": rng.standard_normal(H).astype(np.float32), "y": np.zeros((H, P), np.float32)}
    ref = {k: v.astype(np.float64) for k, v in bufs.items()}
    sp = np.logaddexp(0, ref["dt"] + ref["dt_bias"])
    da = np.exp(sp * ref["a"])
    g = np.arange(H) // (H // G)
    s2 = ref["state"] * da[:, None, None] + ref["b"][g][:, None, :] * (ref["x"] * sp[:, None])[:, :, None]
    y = (s2 * ref["c"][g][:, None, :]).sum(-1) + ref["d"][:, None] * ref["x"]
    loomsim.run(gen_mamba2.probe(N=N, RL=RL), bufs, {}, (H, P // (32 // RL), 1), 32)
    ey = float(np.max(np.abs(bufs["y"] - y)) / np.sqrt((y ** 2).mean()))
    es = float(np.max(np.abs(bufs["state"] - s2)) / np.sqrt((s2 ** 2).mean()))
    ok(ey < 1e-5 and es < 1e-5, f"Mamba-2 step values equal llama.cpp's scan (y {ey:.1e}, state {es:.1e} of rms)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
