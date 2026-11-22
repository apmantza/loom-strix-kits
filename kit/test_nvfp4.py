#!/usr/bin/env python3
"""kit/gen_nvfp4.py, static (HRX-like E250). Recurrences it prevents:
  - an e4m3 scale decoded wrong on gfx11 (no fp8 conversion: the helper builds the f32 from integer fields): the constants the
    generator emits are read back from its Loom text and the decode they define is compared, for every byte with a finite
    e4m3fn value, against the format's definition (2^(e-7) * (1 + m/8); subnormal 2^-6 * m/8);
  - the e2m1 lookup scalarised or the codes repacked byte by byte (E230 / E233 shapes): the probe matvec compiles for gfx1151
    and common/kernel_lint.py finds no flag, with the lookup on v_perm_b32.
usage: python3 kit/test_nvfp4.py   (the compile part needs loom-compile: VENV)"""
import os
import re
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import gen_nvfp4  # noqa: E402

fails = 0


def ok(cond, what):
    global fails
    print(("ok  " if cond else "FAIL") + " " + what)
    fails += not cond


def spec(byte):
    """e4m3fn by definition (sign dropped: NVFP4 scales are non-negative); None for NaN (0x7f / 0xff)"""
    e, m = (byte >> 3) & 15, byte & 7
    if e == 15 and m == 7:
        return None
    return 2.0 ** -6 * m / 8 if e == 0 else 2.0 ** (e - 7) * (1 + m / 8)


def from_text(byte, text):
    """the decode the generator's Loom text defines, evaluated with the constants read from that text"""
    c = {k: float(v) if "." in v else int(v) for k, v in re.findall(r"%t_(c\w+|sub_scale) = scalar\.constant (\S+) :", text)}
    mant, exp = byte & c["c7"], (byte >> c["c3"]) & c["c15"]
    bits = ((exp + c["c120"]) << c["c23"]) | (mant << c["c20"])
    normal = struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]
    return mant * c["sub_scale"] if exp == c["c0"] else normal


def main():
    text = gen_nvfp4.e4m3_scalar("b", "t")
    bad = [b for b in range(256) if spec(b) is not None and from_text(b, text) != spec(b)]
    ok(not bad, f"e4m3 decode equals the definition for all {sum(spec(b) is not None for b in range(256))} finite bytes ({bad[:5]})")
    sys.path.insert(0, os.path.join(HERE, "..", "common"))
    import kernel_lint as kl
    lc = kl.loom_compile()
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "p.loom")
        open(p, "w").write("target.decl @device\namdgpu.target<gfx11-generic> @rdna3\n\n" + gen_nvfp4.probe())
        o = os.path.join(t, "probe.hsaco")
        r = subprocess.run([lc, p, "--target=amdgpu:gfx1151", "--root=g_mv_nvfp4_probe", f"--output={o}"], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"FAIL compile: {r.stderr.strip()[:300]}")
        x = kl.lint(o)
        perm = sum(1 for i in kl.disasm(o)[0] if i.startswith("v_perm_b32"))
        ok(not x["flags"] and perm > 0, f"NVFP4 probe matvec: no lint flag ({x['flags']}), lookup on v_perm ({perm}), {x['instr']} instructions")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
