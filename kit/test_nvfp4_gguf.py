#!/usr/bin/env python3
"""gen_nvfp4.gguf_icodes (ggml NVFP4, Inkling 2's GGUF experts; HRX-like E270) against ggml's own decode. Recurrences it prevents:
  - the nibble order of a 16-value sub-block wrong (ggml: byte j = element j in the low nibble, j + 8 in the high one; the native
    safetensors pair adjacent elements instead), the halves of a block swapped, or a scale applied to the wrong half;
  - the UE4M3 scale decoded wrong: ggml_ue4m3_to_fp32 = e4m3 magnitude x 0.5 (the codes are doubled), 0x7F and 0 give 0, the sign
    bit ignored; every one of the 256 scale bytes is exercised.
Static (always): the test kernel compiles for gfx1151 and the lint finds no flag. With --gpu (a GPU window: it runs a kernel): loom-run
on amdgpu decodes synthetic blocks (all scale bytes, every nibble value in every position) and, when the Inkling GGUF is given with
--gguf, the first real NVFP4 blocks of blk.0.ffn_gate_up_exps.weight, and the codes and scales must equal the numpy reference
below exactly (a transcription of ggml-quants.c dequantize_row_nvfp4 and ggml-impl.h ggml_ue4m3_to_fp32).
usage: python3 kit/test_nvfp4_gguf.py [--gpu] [--gguf PATH]   (needs loom-compile / loom-run: VENV)"""
import argparse
import math
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import gen_nvfp4  # noqa: E402

KVALUES = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]    # ggml kvalues_mxfp4 (e2m1 doubled)
fails = 0


def ok(cond, what):
    global fails
    print(("ok  " if cond else "FAIL") + " " + what)
    fails += not cond


def ue4m3(x):
    """ggml_ue4m3_to_fp32"""
    if x in (0, 0x7F):
        return 0.0
    e, m = (x >> 3) & 0xF, x & 7
    raw = math.ldexp(m, -9) if e == 0 else math.ldexp(1.0 + m / 8.0, e - 7)
    return raw * 0.5


def reference(block):
    """codes [64] (int8) and scales [4] of one 36-byte block, in the helper's order: group h = values 32 h .. 32 h + 31"""
    d, qs = block[:4], block[4:]
    codes = [0] * 64
    for s in range(4):
        for j in range(8):
            q = qs[s * 8 + j]
            codes[s * 16 + j] = KVALUES[q & 0xF]
            codes[s * 16 + j + 8] = KVALUES[q >> 4]
    return codes, [ue4m3(b) for b in d]


def kernel(nblocks):
    groups = 2 * nblocks
    return gen_nvfp4.gguf_icodes() + f"""
kernel.def export("t_nvfp4_gguf") @t_nvfp4_gguf() {{
  %c1 = index.constant 1 : index
  %c64 = index.constant 64 : index
  %wgs = index.constant {(groups + 63) // 64} : index
  kernel.launch.config workgroups(%wgs, %c1, %c1) workgroup_size(%c64, %c1, %c1) : index
}} launch(%w: buffer, %codes_out: buffer, %scales_out: buffer) {{
  %c0 = index.constant 0 : index
  %c2 = index.constant 2 : index
  %c64 = index.constant 64 : index
  %zero = index.constant 0 : offset
  %groups = index.constant {groups} : index
  %wg = kernel.workgroup.id<x> : index
  %lid = kernel.workitem.id<x> : index
  %base = index.mul %wg, %c64 : index
  %gid = index.add %base, %lid : index
  %live = index.cmp ult, %gid, %groups : index
  scf.if %live {{
    %g = index.assume %gid [lt(%gid, %groups)] : index
    %blk = index.div %g, %c2 : index
    %half = index.rem %g, %c2 : index
    %bb = index.constant 36 : offset
    %origin = index.scale %blk, %bb : index, offset -> offset
    %codes, %s_lo, %s_hi = func.call @g_nvfp4_icodes(%w, %origin, %half) : (buffer, offset, index) -> (vector<32xi8>, f32, f32)
    %cv = buffer.view %codes_out[%zero] : buffer -> view<{groups}x32xi8>
    vector.store %codes, %cv[%g, %c0] : vector<32xi8>, view<{groups}x32xi8>
    %sv = buffer.view %scales_out[%zero] : buffer -> view<{groups}x2xf32>
    %pair = vector.from_elements %s_lo, %s_hi : vector<2xf32>
    vector.store %pair, %sv[%g, %c0] : vector<2xf32>, view<{groups}x2xf32>
  }}
  kernel.return
}}
"""


def synthetic():
    """36-byte blocks: block b's scale bytes are 4b .. 4b + 3 (all 256 bytes over 64 blocks), its nibble bytes a pattern that puts
    every nibble value in every low and high position over the set"""
    out = []
    for b in range(64):
        d = bytes([(4 * b + s) & 0xFF for s in range(4)])
        qs = bytes([((b + j) & 0xF) | ((((b * 5) + 3 * j) & 0xF) << 4) for j in range(32)])
        out.append(d + qs)
    return out


def gguf_blocks(path, name="blk.0.ffn_gate_up_exps.weight", n=64):
    """the first n 36-byte blocks of an NVFP4 tensor of a GGUF (header parsed with the standard library)"""
    def rd(f, fmt):
        return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))

    def rstr(f):
        n, = rd(f, "Q")
        return f.read(n).decode(errors="replace")

    def skip(f, t):
        size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if t in size:
            f.read(size[t])
        elif t == 8:
            rstr(f)
        elif t == 9:
            at, k = rd(f, "IQ")
            for _ in range(k):
                skip(f, at)
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        _, nt, nkv = rd(f, "IQQ")
        align = 32
        for _ in range(nkv):
            k = rstr(f)
            t, = rd(f, "I")
            if k == "general.alignment":
                align, = rd(f, "I")
            else:
                skip(f, t)
        found = None
        for _ in range(nt):
            tn = rstr(f)
            nd, = rd(f, "I")
            rd(f, "Q" * nd)
            ty, off = rd(f, "IQ")
            if tn == name:
                found = (ty, off)
        data = (f.tell() + align - 1) // align * align
        assert found and found[0] == 40, f"{name}: not an NVFP4 tensor ({found})"
        f.seek(data + found[1])
        raw = f.read(36 * n)
    return [raw[36 * i:36 * i + 36] for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--gguf")
    a = ap.parse_args()
    blocks = synthetic() + (gguf_blocks(a.gguf) if a.gguf else [])
    src = "target.decl @device\namdgpu.target<gfx11-generic> @rdna3\n\n" + kernel(len(blocks))
    import kernel_lint as kl
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "t.loom")
        open(p, "w").write(src)
        o = os.path.join(t, "t.hsaco")
        r = subprocess.run([kl.loom_compile(), p, "--target=amdgpu:gfx1151", "--root=t_nvfp4_gguf", f"--output={o}"], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"FAIL compile: {r.stderr.strip()[:400]}")
        x = kl.lint(o)
        ok(not x["flags"], f"test kernel compiles for gfx1151, no lint flag ({x['flags']}, {x['instr']} instructions)")
        if not a.gpu:
            print("(static only; --gpu runs the decode against ggml's reference)")
            sys.exit(1 if fails else 0)
        import numpy as np
        codes, scales = [], []
        for b in blocks:
            c, s = reference(b)
            codes += c
            scales += s
        G = 2 * len(blocks)
        w = np.frombuffer(b"".join(blocks), dtype=np.uint8)
        np.save(os.path.join(t, "w.npy"), w)
        np.save(os.path.join(t, "codes.npy"), np.array(codes, dtype=np.int8).reshape(G, 32))
        np.save(os.path.join(t, "scales.npy"), np.array(scales, dtype=np.float32).reshape(G, 2))
        run = os.path.join(os.path.dirname(kl.loom_compile()), "loom-run")
        cmd = [run, p, "--device=amdgpu", "--root=t_nvfp4_gguf",
               f"--kernel-input-buffer=@{t}/w.npy", f"--kernel-input-buffer={G}x32xi8=0", f"--kernel-input-buffer={G}x2xf32=0",
               f"--expected-kernel-buffer=@{t}/w.npy", f"--expected-kernel-buffer=@{t}/codes.npy", f"--expected-kernel-buffer=@{t}/scales.npy"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        ok(r.returncode == 0, f"loom-run on amdgpu: {G} groups ({len(blocks) - 64} real blocks) equal ggml's decode (rc {r.returncode}: {tail})")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
