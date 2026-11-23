"""GEMM256_FLATADDR=1 (HRX-like E345): the 256-row f16 GEMMs' in-loop global loads through a uniform per-step base and a loop-invariant
per-lane element index.

Today each k step addresses the next tile's activations as x_view[x_row, k_next + stage_col] (a 2-D view with a runtime row length) and
its weights as w + w_row_origin + 2 (k_next + stage_col): both lower to 64-bit per-lane address arithmetic every iteration, plus copies of
the address into one register pair per load and a rotation of the prefetched registers (live g_gemm_f16_store_m256: 39 VALU, 8 v_mov,
8 v_swap in a 248-instruction loop). Here the per-lane part (row x pitch + stage column) is computed once before the loop, and each step
views the buffer from the uniform byte offset 2 k (k_next or the activation step's column), so every load takes the SGPR-base form
(global_load v_off, s[base:base+1]) like a constant base. Same bytes from the same addresses: outputs are identical.

The flat views' extents are padded by one load's width so Loom's bounds prover sees origin + width <= extent; every access reads exactly
the addresses the 2-D form read (rows are padded: kx = k + 64), so nothing past the old bounds is touched.
Fails closed: every anchor must match exactly once per kernel."""
import re


def flataddr(text):
    """rewrite one g_gemm_f16_*_m256 kernel's text (from gemm_big256, f16)"""
    def one(old, new, what):
        nonlocal text
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"GEMM256_FLATADDR: {what}: {n} matches (want 1); the generator's text changed")
        text = text.replace(old, new)

    one("  %x_row1 = index.assume %x_row1_value [lt(%x_row1_value, %rows)] : index\n",
        "  %x_row1 = index.assume %x_row1_value [lt(%x_row1_value, %rows)] : index\n"
        "  %x_flat_n = index.mul %rows, %kx : index\n"
        "  %x_c32 = index.constant 32 : index\n"
        "  %x_flat_np = index.add %x_flat_n, %x_c32 : index\n"
        "  %x_lane0_v = index.madd %x_row, %kx, %x_stage_col : index\n"
        "  %x_lane0 = index.assume %x_lane0_v [range(%x_lane0_v, 0, 1073741823), lt(%x_lane0_v, %x_flat_n)] : index\n"
        "  %x_lane1_v = index.madd %x_row1, %kx, %x_stage_col : index\n"
        "  %x_lane1 = index.assume %x_lane1_v [range(%x_lane1_v, 0, 1073741823), lt(%x_lane1_v, %x_flat_n)] : index\n"
        "  %x_two = index.constant 2 : offset\n", "activation row index hoist")
    one("      %xv = vector.load %x_view[%x_row, %x_coln] : view<[%rows]x[%kx]xf16> -> vector<32xf16>\n"
        "      %xv1 = vector.load %x_view[%x_row1, %x_coln] : view<[%rows]x[%kx]xf16> -> vector<32xf16>\n",
        "      %x_kob = index.scale %x_ko, %x_two : index, offset -> offset\n"
        "      %x_sv = buffer.view %x[%x_kob] : buffer -> view<[%x_flat_np]xf16>\n"
        "      %xv = vector.load %x_sv[%x_lane0] : view<[%x_flat_np]xf16> -> vector<32xf16>\n"
        "      %xv1 = vector.load %x_sv[%x_lane1] : view<[%x_flat_np]xf16> -> vector<32xf16>\n", "activation loads")
    one("  %w_row_origin = index.scale %w_out_row, %row_bytes : index, offset -> offset\n",
        "  %w_row_origin = index.scale %w_out_row, %row_bytes : index, offset -> offset\n"
        "  %w_c2 = index.constant 2 : index\n  %w_c8 = index.constant 8 : index\n"
        "  %w_pitch32 = index.div %kx, %w_c2 : index\n"
        "  %w_flat_n = index.mul %n, %w_pitch32 : index\n  %w_flat_np = index.add %w_flat_n, %w_c8 : index\n"
        "  %w_sc32 = index.div %stage_col, %w_c2 : index\n"
        "  %w_lanea_v = index.madd %w_out_row, %w_pitch32, %w_sc32 : index\n"
        "  %w_lanea = index.assume %w_lanea_v [range(%w_lanea_v, 0, 1073741823), lt(%w_lanea_v, %w_flat_n)] : index\n"
        "  %w_laneb_v = index.add %w_lanea, %w_c8 : index\n"
        "  %w_laneb = index.assume %w_laneb_v [range(%w_laneb_v, 0, 1073741823), lt(%w_laneb_v, %w_flat_n)] : index\n",
        "weight row index hoist")
    old_w = re.search(r"    %wcolan = index\.add %k_next, %stage_col : index\n.*?%dqbn_raw = vector\.load %dqbn_view\[0\] : view<8xi32> -> vector<8xi32>\n",
                      text, re.S)
    if not old_w or old_w.group(0).count("vector.load") != 2:
        raise SystemExit("GEMM256_FLATADDR: in-loop weight loads not in the expected form")
    one(old_w.group(0),
        "    %w_kb = index.scale %k_next, %two : index, offset -> offset\n"
        "    %w_sv = buffer.view %w[%w_kb] : buffer -> view<[%w_flat_np]xi32>\n"
        "      %dqan_raw = vector.load %w_sv[%w_lanea] : view<[%w_flat_np]xi32> -> vector<8xi32>\n"
        "      %dqbn_raw = vector.load %w_sv[%w_laneb] : view<[%w_flat_np]xi32> -> vector<8xi32>\n", "weight loads")
    for gone in ("%dqan_origin", "%dqbn_origin"):
        if gone in text:
            raise SystemExit(f"GEMM256_FLATADDR: {gone} still referenced after the rewrite")
    return text
