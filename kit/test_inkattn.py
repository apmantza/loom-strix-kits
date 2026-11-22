#!/usr/bin/env python3
"""kit/gen_inkattn.py prep against the fork's math, statically (HRX-like E274). Recurrences it prevents:
  - a clip applied before its per-channel scale or with the wrong bound (q / k: qk_clip 22.63, r: rel_clip 8; norm + 1e-6, factor
    clamped to [0, 1]), v not RMS-normalised (or with the wrong length), r's padded lanes (16 of 32) leaking into its norm;
  - tau wrong or applied on local layers (global only: 1 + alpha log(max(1, (pos + 1) / floor)), on q and r, after the clip);
  - heads or waves mixed up (each of 8 waves owns 3 q heads, 1 k, 1 v, 3 r heads).
How: the emitted kernel body is executed by a lane-vectorised interpreter (8 waves x 32 lanes per token; subgroup sums over the lanes) in
float32 and compared with a numpy transcription of inkling2.cpp's clip() / rms_norm / tau. Summation order differs from ggml's, so the
comparison is to 2e-6 relative, not bit for bit. The kernels also compile for gfx1151 with no lint flag.
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

f32 = np.float32
fails = 0


def ok(cond, what):
    global fails
    print(("ok  " if cond else "FAIL") + " " + what)
    fails += not cond


def interpret(text, bufs, T):
    """run the launch body for every token and wave; values are arrays over the 32 lanes of a wave"""
    body = text[text.index("} launch(") :]
    body = body[body.index("{") + 1: body.rindex("kernel.return")]
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    views = {}
    for token in range(T):
        for wave in range(8):
            lanes = np.arange(32)
            env = {"token_id": np.full(32, token), "wave_id": np.full(32, wave), "wl_id": lanes}
            skip = [False]
            depth = []
            for ln in lines:
                if ln.startswith("scf.if "):
                    depth.append(env[ln.split()[1][1:]].astype(bool))
                    continue
                if ln == "}":
                    depth.pop()
                    continue
                m = re.match(r"%(\w+) = buffer\.view %(\w+)\[%zero\] : buffer -> view<([^>]+)>$", ln)
                if m:
                    views[m.group(1)] = m.group(2)
                    continue
                if re.match(r"%\w+ = index\.constant 0 : offset$", ln):
                    continue
                m = re.match(r"%(\w+) = index\.constant (-?\d+) : index$", ln)
                if m:
                    env[m.group(1)] = np.full(32, int(m.group(2)))
                    continue
                m = re.match(r"%(\w+) = scalar\.constant (\S+) : (f32|i32)$", ln)
                if m:
                    env[m.group(1)] = np.full(32, f32(float(m.group(2)))) if m.group(3) == "f32" else np.full(32, int(m.group(2)))
                    continue
                m = re.match(r"%(\w+) = index\.(add|mul|min) %(\w+), %(\w+) : index$", ln)
                if m:
                    a, b = env[m.group(3)], env[m.group(4)]
                    env[m.group(1)] = {"add": a + b, "mul": a * b, "min": np.minimum(a, b)}[m.group(2)]
                    continue
                m = re.match(r"%(\w+) = index\.assume %(\w+) ", ln)
                if m:
                    env[m.group(1)] = env[m.group(2)]
                    continue
                m = re.match(r"%(\w+) = index\.cmp ult, %(\w+), %(\w+) : index$", ln)
                if m:
                    env[m.group(1)] = env[m.group(2)] < env[m.group(3)]
                    continue
                m = re.match(r"%(\w+) = kernel\.subgroup\.id : index$", ln) or re.match(r"%(\w+) = kernel\.(workgroup\.id<x>|subgroup\.lane\.id) : index$", ln)
                if m:
                    continue
                m = re.match(r"%(\w+) = view\.load %(\w+)\[([^\]]+)\] :", ln)
                if m:
                    arr = bufs[views[m.group(2)]]
                    idx = tuple(env[t.strip()[1:]] for t in m.group(3).split(","))
                    env[m.group(1)] = arr[idx]
                    continue
                m = re.match(r"view\.store %(\w+), %(\w+)\[([^\]]+)\] :", ln)
                if m:
                    arr = bufs[views[m.group(2)]]
                    idx = tuple(env[t.strip()[1:]] for t in m.group(3).split(","))
                    live = np.logical_and.reduce(depth) if depth else np.ones(32, bool)
                    for ln_i in np.nonzero(live)[0]:
                        arr[tuple(i[ln_i] for i in idx)] = env[m.group(1)][ln_i]
                    continue
                m = re.match(r"%(\w+) = scalar\.(mulf|addf|divf|maxnumf|minnumf) %(\w+), %(\w+) : f32$", ln)
                if m:
                    a, b = env[m.group(3)].astype(f32), env[m.group(4)].astype(f32)
                    op = m.group(2)
                    env[m.group(1)] = (a * b if op == "mulf" else a + b if op == "addf" else a / b if op == "divf"
                                       else np.maximum(a, b) if op == "maxnumf" else np.minimum(a, b)).astype(f32)
                    continue
                m = re.match(r"%(\w+) = scalar\.(sqrtf|rsqrtf|logf<afn>) %(\w+) : f32$", ln)
                if m:
                    a = env[m.group(3)].astype(f32)
                    env[m.group(1)] = (np.sqrt(a) if m.group(2) == "sqrtf" else f32(1) / np.sqrt(a) if m.group(2) == "rsqrtf" else np.log(a)).astype(f32)
                    continue
                m = re.match(r"%(\w+) = scalar\.sitofp %(\w+) : i32 to f32$", ln)
                if m:
                    env[m.group(1)] = env[m.group(2)].astype(f32)
                    continue
                m = re.match(r"%(\w+) = scf\.select %(\w+), %(\w+), %(\w+) : f32$", ln)
                if m:
                    env[m.group(1)] = np.where(env[m.group(2)], env[m.group(3)], env[m.group(4)]).astype(f32)
                    continue
                m = re.match(r"%(\w+) = kernel\.subgroup\.reduce<addf> %(\w+) : f32$", ln)
                if m:
                    env[m.group(1)] = np.full(32, env[m.group(2)].astype(np.float64).sum(), f32)
                    continue
                raise SystemExit(f"FAIL interpreter: unknown line {ln!r}")


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
        interpret(gen_inkattn.prep(T, glob), bufs, T)
        errs = [float(np.max(np.abs(bufs[n] - w) / (np.abs(w) + 1e-6))) for n, w in zip(("q", "k", "v", "r"), want)]
        ok(max(errs) < 2e-6, f"T {T} {'global' if glob else 'local'}: q k v r equal the fork's clips / norms / tau (max rel err {max(errs):.1e})")
    import kernel_lint as kl
    with tempfile.TemporaryDirectory() as t:
        for T, glob in ((1, False), (8, True)):
            name = f"g_ink_attn_prep_t{T}" + ("_global" if glob else "")
            p = os.path.join(t, "p.loom")
            open(p, "w").write("target.decl @device\namdgpu.target<gfx11-generic> @rdna3\n" + gen_inkattn.prep(T, glob))
            o = os.path.join(t, "p.hsaco")
            r_ = subprocess.run([kl.loom_compile(), p, "--target=amdgpu:gfx1151", f"--root={name}", f"--output={o}"], capture_output=True, text=True)
            if r_.returncode:
                sys.exit(f"FAIL compile {name}: {r_.stderr.strip()[:300]}")
            ok(not kl.lint(o)["flags"], f"{name} compiles for gfx1151, no lint flag")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
