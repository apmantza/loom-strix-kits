#!/usr/bin/env python3
"""kit/gen_iq2xxs.py, static (HRX-like E262). Recurrences it prevents:
  - a wrong or truncated codebook: the embedded grid must be ggml's iq2xxs_grid (256 entries of magnitudes 8 / 25 / 43, pinned by
    its sha256; any edit of the hex changes it);
  - the word-level sign arithmetic decoding wrong (a carry between bytes, a wrong parity bit, a spread constant off by a bit): the
    constants and op order the generator emits are read back from its Loom text and evaluated for every one of the 128 sign
    indices on every grid entry, against the format's definition (ksigns = i | parity(i) << 7, value negated where bit j is set);
  - the decode scalarised, repacked byte by byte or spilled (E230 / E233 shapes): both grid modes compile for gfx1151 with 8 dot4
    and no kernel-lint flag.
usage: python3 kit/test_iq2xxs.py   (the compile part needs loom-compile: VENV)"""
import hashlib
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import gen_iq2xxs  # noqa: E402

GRID_SHA = "05826b5d3e472a3a"      # sha256 prefix of ggml-common.h iq2xxs_grid as 2,048 little-endian bytes (llama.cpp 2b66bfbf7)
fails = 0


def ok(cond, what):
    global fails
    print(("ok  " if cond else "FAIL") + " " + what)
    fails += not cond


def reference(grid8, sidx):
    """the 8 signed values of one grid entry under sign index sidx, by the format's definition"""
    ks = sidx | ((bin(sidx).count("1") & 1) << 7)
    return [-v if ks >> j & 1 else v for j, v in enumerate(grid8)]


OPS = {"shrui": lambda a, b: a >> b, "shli": lambda a, b: a << b, "andi": lambda a, b: a & b, "ori": lambda a, b: a | b,
       "xori": lambda a, b: a ^ b, "muli": lambda a, b: a * b, "addi": lambda a, b: a + b}


def evaluate(text, env):
    """run the i32 scalar ops of the generator's Loom text in order (constants, shifts, logic, mul, add, as written), from the
    given inputs; values wrap at 32 bits. Ops on other types are skipped."""
    v = dict(env)
    for name, op, args in re.findall(r"^\s*%(\w+) = scalar\.(\w+) ([^:]+): i32$", text, re.M):
        if op == "constant":
            v[name] = int(args) & 0xFFFFFFFF
        elif op in OPS:
            a, b = [x.strip().lstrip("%") for x in args.split(",")]
            if a in v and b in v:
                v[name] = OPS[op](v[a], v[b]) & 0xFFFFFFFF
    return v


def main():
    raw = bytes.fromhex(gen_iq2xxs.GRID_HEX)
    ok(len(raw) == 2048 and set(raw) == {8, 25, 43} and hashlib.sha256(raw).hexdigest()[:16] == GRID_SHA,
       f"grid: 256 entries of 8 / 25 / 43, ggml's table (sha256 {hashlib.sha256(raw).hexdigest()[:16]})")
    text = gen_iq2xxs.icodes("rodata")
    bad = 0
    words = [(int.from_bytes(raw[8 * e:8 * e + 4], "little"), int.from_bytes(raw[8 * e + 4:8 * e + 8], "little")) for e in range(256)]
    for slot in range(4):                                  # each of the four 8-value chunks, its sign index at bits 7 slot ..
        for sidx in range(128):
            others = sum(((sidx * 37 + 11 * l) % 128) << (7 * l) for l in range(4) if l != slot)   # the other slots' indices
            w1 = others | (sidx << (7 * slot)) | (9 << 28)
            env = {"w0": 0, "w1": w1}
            for l in range(4):
                env[f"ga{l}"], env[f"gb{l}"] = words[(sidx + 61 * l) % 256]
            out = evaluate(text, env)
            ga, gb = env[f"ga{slot}"], env[f"gb{slot}"]
            g8 = list(ga.to_bytes(4, "little") + gb.to_bytes(4, "little"))
            got = [b if b < 128 else b - 256 for b in out[f"wa{slot}"].to_bytes(4, "little") + out[f"wb{slot}"].to_bytes(4, "little")]
            bad += got != reference(g8, sidx)
    ok(bad == 0, f"signs on words, evaluated from the generator's ops, equal the definition in all 4 slots x 128 sign indices ({bad} wrong)")
    bad = 0
    for e in range(0, 256, 17):                            # every grid entry index and slot: the byte the generator extracts
        w0 = (e | ((255 - e) << 8) | (((e * 7) % 256) << 16) | (((e * 13) % 256) << 24))
        out = evaluate(text, {"w0": w0, "w1": 0})
        bad += [out[f"ix_m{l}"] for l in range(4)] != [w0 >> 8 * l & 255 for l in range(4)]
    ok(bad == 0, "grid indices: byte l of word 0 for every slot")
    out = evaluate(text, {"w0": 0, "w1": 0xB0000000})
    ok(out.get("ls") == 11, f"group scale index ls = bits 28-31 of word 1 ({out.get('ls')})")
    import kernel_lint as kl
    lc = kl.loom_compile()
    with tempfile.TemporaryDirectory() as t:
        for grid in ("rodata", "lds"):
            p = os.path.join(t, f"{grid}.loom")
            open(p, "w").write("target.decl @device\namdgpu.target<gfx11-generic> @rdna3\n\n" + gen_iq2xxs.probe(grid=grid))
            o = os.path.join(t, f"{grid}.hsaco")
            r = subprocess.run([lc, p, "--target=amdgpu:gfx1151", f"--root=g_mv_iq2xxs_probe_{grid}", f"--output={o}"],
                               capture_output=True, text=True)
            if r.returncode:
                sys.exit(f"FAIL compile {grid}: {r.stderr.strip()[:300]}")
            x = kl.lint(o)
            dot = sum(1 for i in kl.disasm(o)[0] if i.startswith("v_dot4"))
            ok(not x["flags"] and dot == 8, f"IQ2_XXS probe ({grid} grid): {x['instr']} instructions, {dot} dot4, no lint flag ({x['flags']})")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
