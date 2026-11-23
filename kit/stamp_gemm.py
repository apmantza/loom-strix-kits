#!/usr/bin/env python3
"""Per-stage timing inside a WMMA GEMM (HRX-like E294): copy a served kernel out of a package program and add clock reads at its
stage boundaries, so a run says where each wave's time goes instead of only the kernel's total.

The kernel must have one k-loop (`= scf.for %step`) holding exactly two workgroup barriers: per step, stage (store the prefetched
tile to LDS, wait for every wave) -> barrier 1 -> mma (issue the next tile's global loads, LDS fragment loads, WMMAs) -> barrier 2.
Each wave accumulates, over all steps, the time from the previous stamp to after barrier 1 (stage: global-load waits, LDS stores,
waiting for the slowest wave), from there to before barrier 2 (mma issue), and across barrier 2; plus the prologue (kernel start to
loop start) and the epilogue (loop end to return). One row of 8 i32 per wave goes to an extra buffer argument `%stamps`:
  [stage, mma, barrier2, prologue, epilogue, steps, 0, 0]   in ticks of the 100 MHz realtime clock (10 ns)
slot = (workgroup.id.y * row_tiles + workgroup.id.x) * waves + wave, clamped to STAMP_ROWS - 1.

The clock is s_sendmsg_rtn_b64 message(131) (MSG_RTN_GET_REALTIME; Loom's own rdna3 test low_message_clock.loom-test), read in a
low.func and fenced (scf.schedule.fence) so the scheduler does not move work across it. The read waits for lgkmcnt(0): at the
stamps after a barrier that is free; before barrier 2 the LDS fragments are consumed already. Perturbation is measured, not assumed:
the stamped kernel's total time is compared with the unstamped one in the same run.
usage: python3 kit/stamp_gemm.py PROGRAM.loom KERNEL OUT.loom   (writes OUT.loom with KERNEL and KERNEL_stamped)"""
import re
import sys

STAMP_ROWS = 65536
CLOCK = """low.func.def target<amdgpu.gfx11.generic.core>(@rdna3) @g_stamp_clock() -> (reg<amdgpu.sgpr>) asm {
  %wide = s_sendmsg_rtn_b64 {message = 131}
  %low = slice %wide[0] : reg<amdgpu.sgpr x2> -> reg<amdgpu.sgpr>
  return %low
}
"""


def extract(lines, kernel):
    """the kernel.def block of KERNEL (rdna3 target) as a list of lines"""
    i = next(j for j, l in enumerate(lines) if re.match(rf'kernel\.def target\(@rdna3\) export\("{kernel}"\)', l))
    c = next(k for k in range(i, len(lines)) if lines[k].startswith("} launch("))
    j = next(k for k in range(c + 1, len(lines)) if lines[k] == "}")
    return lines[i:j + 1]


def stamp(n, t):
    """fenced clock read into %n"""
    return [f"{t}scf.schedule.fence", f"{t}%{n} = low.invoke @g_stamp_clock() : () -> (i32)", f"{t}scf.schedule.fence"]


def stamped(block, kernel, waves=8):
    L = list(block)
    L[0] = L[0].replace(f'export("{kernel}") @{kernel}(', f'export("{kernel}_stamped") @{kernel}_stamped(')
    c = next(k for k, l in enumerate(L) if l.startswith("} launch("))
    assert L[c].endswith(") {"), L[c]
    L[c] = L[c][:-3] + ", %stamps: buffer) {"
    f = [k for k, l in enumerate(L) if "= scf.for %step" in l]
    assert len(f) == 1, f"{kernel}: {len(f)} k-loops"
    f = f[0]
    bars = [k for k, l in enumerate(L) if l.startswith("    kernel.barrier") and k > f]
    assert len(bars) == 2, f"{kernel}: {len(bars)} barriers in the loop"
    y = bars[1] + 1
    assert L[y].startswith("    scf.yield ") and L[y + 1] == "  }", (L[y], L[y + 1])
    ret = [k for k, l in enumerate(L) if l.strip() == "kernel.return"]
    assert len(ret) == 1
    m = re.match(r"(?P<res>  .*?) = scf\.for (?P<hdr>%step = \[.*?\])\((?P<it>.*)\) -> \((?P<ty>.*)\) \{$", L[f])
    assert m, L[f][:200]
    L[f] = (f"{m['res']}, %st_stage, %st_mma, %st_b2, %st_t = scf.for {m['hdr']}({m['it']}, %sa_stage = %sz : i32, %sa_mma = %sz : i32, "
            f"%sa_b2 = %sz : i32, %sa_t = %st_loop0 : i32) -> ({m['ty']}, i32, i32, i32, i32) {{")
    ym = re.match(r"(    scf\.yield .*?) : (.*)$", L[y])
    L[y] = f"{ym[1]}, %sn_stage, %sn_mma, %sn_b2, %ss_b2 : {ym[2]}, i32, i32, i32, i32"
    after_b2 = stamp("ss_b2", "    ") + ["    %sd_b2 = scalar.subi %ss_b2, %ss_mma : i32", "    %sn_b2 = scalar.addi %sa_b2, %sd_b2 : i32"]
    before_b2 = stamp("ss_mma", "    ") + ["    %sd_mma = scalar.subi %ss_mma, %ss_b1 : i32", "    %sn_mma = scalar.addi %sa_mma, %sd_mma : i32"]
    after_b1 = stamp("ss_b1", "    ") + ["    %sd_stage = scalar.subi %ss_b1, %sa_t : i32", "    %sn_stage = scalar.addi %sa_stage, %sd_stage : i32"]
    tail = stamp("st_end", "  ") + [
        "  %st_pro = scalar.subi %st_loop0, %st_k0 : i32",
        "  %st_epi = scalar.subi %st_end, %st_t : i32",
        "  %st_steps = index.cast %k_steps : index to i32",
        "  %st_zero = scalar.constant 0 : i32",
        "  %st_row = vector.from_elements %st_stage, %st_mma, %st_b2, %st_pro, %st_epi, %st_steps, %st_zero, %st_zero : vector<8xi32>",
        "  %st_rt = index.div %rows, %c256 : index",
        "  %st_wg_y = kernel.workgroup.id<y> : index",
        "  %st_wg_x = kernel.workgroup.id<x> : index",
        "  %st_wg = index.madd %st_wg_y, %st_rt, %st_wg_x : index",
        f"  %st_waves = index.constant {waves} : index",
        "  %st_slot_raw = index.madd %st_wg, %st_waves, %bounded_subgroup : index",
        f"  %st_cap = index.constant {STAMP_ROWS - 1} : index",
        "  %st_slot = index.min %st_slot_raw, %st_cap : index",
        "  %st_off = index.constant 0 : offset",
        f"  %st_view = buffer.view %stamps[%st_off] : buffer -> view<{STAMP_ROWS}x8xi32>",
        "  %st_c0 = index.constant 0 : index",
        f"  vector.store %st_row, %st_view[%st_slot, %st_c0] : vector<8xi32>, view<{STAMP_ROWS}x8xi32>"]
    # the tail goes at the end of the region that holds the k-loop: served kernels wrap the tile in `scf.if %tile_live {` with an
    # unindented body, closed by '  }' just before kernel.return; after that brace the loop's results are out of scope and Loom
    # refuses the fence (q111 11-23: PARSE/003 'unexpected token scf.schedule.fence, expected }')
    at = ret[0] - 1 if L[ret[0] - 1] == "  }" else ret[0]
    out = L[:at] + tail + L[at:]
    out = out[:y] + after_b2 + out[y:]                     # insert from the bottom up so earlier indices stay valid
    out = out[:bars[1]] + before_b2 + [out[bars[1]]] + out[bars[1] + 1:]
    out = out[:bars[0] + 1] + after_b1 + out[bars[0] + 1:]
    out = out[:f] + stamp("st_loop0", "  ") + ["  %sz = scalar.constant 0 : i32"] + out[f:]
    out = out[:c + 1] + stamp("st_k0", "  ") + out[c + 1:]
    return out


def main():
    prog, kernel, dst = sys.argv[1:4]
    lines = open(prog).read().splitlines()
    head = [l for l in lines[:4] if l.startswith(("target.decl", "amdgpu.target"))]
    block = extract(lines, kernel)
    text = "\n".join(head) + "\n\n" + CLOCK + "\n" + "\n".join(block) + "\n\n" + "\n".join(stamped(block, kernel)) + "\n"
    open(dst, "w").write(text)
    print(f"{dst}: {kernel} and {kernel}_stamped (stamps buffer {STAMP_ROWS} x 8 i32 = {STAMP_ROWS * 32} bytes)")


if __name__ == "__main__":
    main()
