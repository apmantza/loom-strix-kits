#!/usr/bin/env python3
"""Per-phase timing inside a served kernel's hot loop (HRX-like E294 / E302; replaces kit/stamp_gemm.py, which handled only the
two-barrier GEMM): copy KERNEL out of a package program and add fenced clock reads (s_sendmsg_rtn_b64 message 131, the 100 MHz
realtime clock, in a low.func; Loom's own rdna3 test low_message_clock.loom-test) around every barrier at the body level of the
loop over IV:
    ... work_0 ... | barrier 0 | ... work_1 ... | barrier 1 | ... | barrier N-1 | ... tail ... (yield)
Each wave accumulates, over all iterations, work_i (previous stamp -> just before barrier i), wait_i (across barrier i) and the
tail (after the last barrier -> the yield; the next iteration's work_0 starts there), plus the prologue (kernel start -> loop
start) and the epilogue (loop end -> the end of the region holding the loop, e.g. a peeled last tile). One row of 16 i32 per wave
goes to an extra last argument %stamps:
    [work_0, wait_0, ..., work_{N-1}, wait_{N-1}, tail, prologue, epilogue, iterations, 0 ...]   (N <= 6 barriers)
slot = ((wg.z * GY + wg.y) * GX + wg.x) * waves + subgroup (GX, GY, waves from the caller: the run's grid), clamped to ROWS - 1.
Every clock read waits for outstanding LDS loads (lgkmcnt(0)), so the stamped kernel is slower than the served one (+41 % on the
f16 GEMM, E302): compare shares, and report the perturbation (stamped vs unstamped time) with them. Fixes carried over from
stamp_gemm: low.invoke results in parentheses (PARSE/003 otherwise); the tail inside the region that holds the loop (served
kernels wrap the tile in an unindented `scf.if %tile_live {`); slots from the real grid (x-major slots collided, q111).
usage: python3 kit/stamp_loop.py PROGRAM.loom KERNEL IV GX GY WAVES OUT.loom [cycles (default)|realtime]   (OUT holds KERNEL unchanged and KERNEL_stamped)"""
import re
import sys

ROWS = 1 << 20
CLOCK = """low.func.def target<amdgpu.gfx11.generic.core>(@rdna3) @g_stamp_clock() -> (reg<amdgpu.sgpr>) asm {
  %wide = s_sendmsg_rtn_b64 {message = 131}
  %low = slice %wide[0] : reg<amdgpu.sgpr x2> -> reg<amdgpu.sgpr>
  return %low
}
"""
# cycles: s_getreg_b32 of HW_REG_SHADER_CYCLES (id 29, offset 0, 20 bits: hwreg = 29 | (20 - 1) << 11), a register read with no
# memory wait. The realtime read waits for every outstanding LDS load (lgkmcnt(0)): +41 % on the f16 GEMM, 3.2x on flash2 (E304).
# The counter is 20 bits wide: every delta is taken modulo 2^20 (~0.36 ms at full clock; phases are microseconds).
CYCLES = """low.func.def target<amdgpu.gfx11.generic.core>(@rdna3) @g_stamp_clock() -> (reg<amdgpu.sgpr>) asm {
  %c = s_getreg_b32 {hwreg = 38941}
  return %c
}
"""
MASK = None   # set by main for the cycles clock: the 20-bit wrap


def extract(lines, kernel):
    """the kernel.def block of KERNEL (rdna3 target) as a list of lines"""
    i = next(j for j, l in enumerate(lines) if re.match(rf'kernel\.def (?:target\(@rdna3\) )?export\("{kernel}"\)', l))
    c = next(k for k in range(i, len(lines)) if lines[k].startswith("} launch("))
    j = next(k for k in range(c + 1, len(lines)) if lines[k] == "}")
    return lines[i:j + 1]


def stamp(n, t):
    return [f"{t}scf.schedule.fence", f"{t}%{n} = low.invoke @g_stamp_clock() : () -> (i32)", f"{t}scf.schedule.fence"]


def depths(L, start):
    """brace depth of each line from `start` (the depth before the line's own opening brace)"""
    d, out = 0, []
    for l in L[start:]:
        code = re.sub(r'"[^"]*"', "", l)
        out.append(d)
        d += code.count("{") - code.count("}")
    return out


def stamped(block, kernel, iv, gx, gy, waves):
    L = list(block)
    L[0] = L[0].replace(f'export("{kernel}") @{kernel}(', f'export("{kernel}_stamped") @{kernel}_stamped(')
    c = next(k for k, l in enumerate(L) if l.startswith("} launch("))
    assert L[c].endswith(") {"), L[c]
    L[c] = L[c][:-3] + ", %stamps: buffer) {"
    f = [k for k, l in enumerate(L) if re.search(rf"= scf\.for %{iv} = \[", l)]
    assert len(f) == 1, f"{kernel}: {len(f)} loops over %{iv} with results"
    f = f[0]
    m = re.match(r"(?P<ind> *)(?P<res>.*?) = scf\.for (?P<hdr>%\w+ = \[%(?P<lb>\w+) to %(?P<ub>\w+) step %(?P<st>\w+)\])\((?P<it>.*)\) -> \((?P<ty>.*)\) \{$", L[f])
    assert m, L[f][:200]
    ind, inner = m["ind"], m["ind"] + "  "
    D = depths(L, f)                                   # D[k - f]: depth before line k, relative to the loop line
    net = lambda k: re.sub(r'"[^"]*"', "", L[k]).count("{") - re.sub(r'"[^"]*"', "", L[k]).count("}")
    end = next(k for k in range(f + 1, len(L)) if D[k - f] + net(k) == 0)   # the line that closes the loop
    y = end - 1
    assert L[y].strip().startswith("scf.yield "), L[y]
    bars = [k for k in range(f + 1, y) if D[k - f] == 1 and L[k].strip().startswith("kernel.barrier")]
    N = len(bars)
    assert 1 <= N <= 6, f"{kernel}: {N} body-level barriers in the loop"
    names = [f"w{i}" for i in range(N)] + [f"b{i}" for i in range(N)] + ["tl"]
    L[f] = (f"{ind}{m['res']}, " + ", ".join(f"%sr_{n}" for n in names) + ", %sr_t = scf.for "
            f"{m['hdr']}({m['it']}, " + ", ".join(f"%sa_{n} = %sz : i32" for n in names) + ", %sa_t = %st_loop0 : i32) -> ("
            f"{m['ty']}, " + ", ".join(["i32"] * (len(names) + 1)) + ") {")
    ym = re.match(r"(\s*scf\.yield .*?) : (.*)$", L[y])
    L[y] = f"{ym[1]}, " + ", ".join(f"%sn_{n}" for n in names) + f", %ss_e : {ym[2]}, " + ", ".join(["i32"] * (len(names) + 1))
    ins = {}                                           # line index -> lines inserted before it
    prev = "%sa_t"
    for i, k in enumerate(bars):
        ins.setdefault(k, []).extend(stamp(f"ss_w{i}", inner) + [f"{inner}%sd_w{i} = scalar.subi %ss_w{i}, {prev} : i32",
                                                               f"{inner}%sn_w{i} = scalar.addi %sa_w{i}, %sd_w{i} : i32"])
        ins.setdefault(k + 1, []).extend(stamp(f"ss_b{i}", inner) + [f"{inner}%sd_b{i} = scalar.subi %ss_b{i}, %ss_w{i} : i32",
                                                                   f"{inner}%sn_b{i} = scalar.addi %sa_b{i}, %sd_b{i} : i32"])
        prev = f"%ss_b{i}"
    ins.setdefault(y, []).extend(stamp("ss_e", inner) + [f"{inner}%sd_tl = scalar.subi %ss_e, {prev} : i32",
                                                       f"{inner}%sn_tl = scalar.addi %sa_tl, %sd_tl : i32"])
    # the epilogue ends at the end of the region holding the loop: the first later line that closes it, or kernel.return
    R = depths(L, 0)
    base = R[f]
    tail_at = next(k for k in range(end + 1, len(L)) if L[k].strip() == "kernel.return" or
                   (R[k] == base and L[k].strip().startswith("}")))
    vals = [f"%sr_{n}" for n in [x for i in range(N) for x in (f"w{i}", f"b{i}")]] + ["%sr_tl", "%st_pro", "%st_epi", "%st_its"]
    vals += ["%st_zero"] * (16 - len(vals))
    tail = stamp("st_end", ind) + [
        f"{ind}%st_pro = scalar.subi %st_loop0, %st_k0 : i32",
        f"{ind}%st_epi = scalar.subi %st_end, %sr_t : i32",
        f"{ind}%st_itn = index.sub %{m['ub']}, %{m['lb']} : index",
        f"{ind}%st_its = index.cast %st_itn : index to i32",
        f"{ind}%st_zero = scalar.constant 0 : i32",
        f"{ind}%st_row = vector.from_elements {', '.join(vals)} : vector<16xi32>",
        f"{ind}%st_wx = kernel.workgroup.id<x> : index",
        f"{ind}%st_wy = kernel.workgroup.id<y> : index",
        f"{ind}%st_wz = kernel.workgroup.id<z> : index",
        f"{ind}%st_gx = index.constant {gx} : index",
        f"{ind}%st_gy = index.constant {gy} : index",
        f"{ind}%st_zy = index.madd %st_wz, %st_gy, %st_wy : index",
        f"{ind}%st_wg = index.madd %st_zy, %st_gx, %st_wx : index",
        f"{ind}%st_nw = index.constant {waves} : index",
        f"{ind}%st_sg = kernel.subgroup.id : index",
        f"{ind}%st_slot_raw = index.madd %st_wg, %st_nw, %st_sg : index",
        f"{ind}%st_cap = index.constant {ROWS - 1} : index",
        f"{ind}%st_slot = index.min %st_slot_raw, %st_cap : index",
        f"{ind}%st_off = index.constant 0 : offset",
        f"{ind}%st_view = buffer.view %stamps[%st_off] : buffer -> view<{ROWS}x16xi32>",
        f"{ind}%st_c0 = index.constant 0 : index",
        f"{ind}vector.store %st_row, %st_view[%st_slot, %st_c0] : vector<16xi32>, view<{ROWS}x16xi32>"]
    ins.setdefault(tail_at, []).extend(tail)
    ins.setdefault(f, []).extend(stamp("st_loop0", ind) + [f"{ind}%sz = scalar.constant 0 : i32"])
    ins.setdefault(c + 1, []).extend(stamp("st_k0", "  "))
    out = []
    for k, l in enumerate(L):
        out += ins.get(k, [])
        out.append(l)
    if MASK:   # deltas of a wrapping counter: (b - a) & mask
        res = []
        for l in out:
            m = re.match(r"(\s*)%(s[dt]_\w+|st_pro|st_epi) = scalar\.subi (%\w+), (%\w+) : i32$", l)
            if m:
                res += [f"{m[1]}%{m[2]}_w = scalar.subi {m[3]}, {m[4]} : i32", f"{m[1]}%{m[2]}_m = scalar.constant {MASK} : i32",
                        f"{m[1]}%{m[2]} = scalar.andi %{m[2]}_w, %{m[2]}_m : i32"]
            else:
                res.append(l)
        out = res
    return out, N


def helpers(src, block):
    """the func.def items the kernel calls, transitively, in program order (a standalone file must define them: flash2 calls
    @g_kv8_scale_offset; q115 11-23: SYMBOL/002 without it)"""
    items = re.split(r"(?m)^(?=[a-z])", src)
    defs = {m.group(1): it for it in items for m in [re.match(r"func\.def\b[^@]*@(\w+)\(", it)] if m}
    want, todo = set(), set(re.findall(r"func\.call @(\w+)", "\n".join(block)))
    while todo:
        n = todo.pop()
        if n in want or n not in defs:
            continue
        want.add(n)
        todo |= set(re.findall(r"func\.call @(\w+)", defs[n]))
    return [it for it in items if any(re.match(rf"func\.def\b[^@]*@{n}\(", it) for n in want)]


def main():
    global MASK
    prog, kernel, iv, gx, gy, waves, dst = sys.argv[1:8]
    clock = sys.argv[8] if len(sys.argv) > 8 else "cycles"   # default: the cycle counter (realtime waits for every LDS load: +41 % / 3.2x, E304)
    assert clock in ("realtime", "cycles"), clock
    MASK = (1 << 20) - 1 if clock == "cycles" else None
    src = open(prog).read()
    lines = src.splitlines()
    head = [l for l in lines[:4] if l.startswith(("target.decl", "amdgpu.target"))]
    block = extract(lines, kernel)
    st, N = stamped(block, kernel, iv, int(gx), int(gy), int(waves))
    open(dst, "w").write("\n".join(head) + "\n\n" + (CYCLES if clock == "cycles" else CLOCK) + "\n" + "".join(helpers(src, block)) + "\n" + "\n".join(block) + "\n\n"
                         + "\n".join(st) + "\n")
    print(f"{dst}: {kernel} and {kernel}_stamped ({N} barriers in the loop over %{iv}; stamps buffer {ROWS} x 16 i32 = {ROWS * 64} bytes)")


if __name__ == "__main__":
    main()
