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
Fails closed: every anchor must match exactly once per kernel.

apply() picks the kernels on a whole program: every gfx11 kernel named g_gemm_f16_{acc,store,swiglu}_m256 or its F16O11 copy
(_m256_f16o); gfx12 twins (target(@rdna4)) and SKIP are skipped by name; any other g_gemm_f16_*m256* kernel stops the build. The
pattern before HRX-like E512 ended at `_m256"`, so the served swiglu GEMM (g_gemm_f16_swiglu_m256_f16o since F16O11) was never
rewritten while the build log listed the unserved g_gemm_f16_swiglu_m256 as done; E514 rewrote it, and E744 measured that a served
loss (1.0093x in cycles), so SKIP returns it to the form live pkg-e480 serves (the unserved g_gemm_f16_swiglu_m256 stays rewritten,
as in pkg-e480)."""
import re


def flataddr(text):
    """rewrite one g_gemm_f16_*_m256 kernel's text (from gemm_big256, f16)"""
    import os
    if os.environ.get("GEMM256_KT", "64") != "64":
        # the anchors are the K-tile-64 forms (32-value activation loads); GEMM256_KT=32 stages 16-value loads. KT=32 was measured
        # exact and 1.21-1.43x slower (HRX-like E163), so the rewrite is not ported to it: refuse the combination by name (E407).
        raise SystemExit("GEMM256_FLATADDR supports GEMM256_KT=64 only (KT=32: exact but slower, HRX-like E163); unset one of them")
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
    # the buffer the loads read: %w (store) or %wsel (swiglu: a per-lane choice between gate %w and up %w2, kit 718c339 hardcoded %w and the
    # f16o11 swiglu m256 kernel tests caught it, HRX-like E372); both loads must read the same one
    bufs = set(re.findall(r"buffer\.view (%\w+)\[%dq[ab]n_origin\]", old_w.group(0)))
    if len(bufs) != 1:
        raise SystemExit(f"GEMM256_FLATADDR: the weight loads read {sorted(bufs)} (want one buffer)")
    wbuf = bufs.pop()
    one(old_w.group(0),
        "    %w_kb = index.scale %k_next, %two : index, offset -> offset\n"
        f"    %w_sv = buffer.view {wbuf}[%w_kb] : buffer -> view<[%w_flat_np]xi32>\n"
        "      %dqan_raw = vector.load %w_sv[%w_lanea] : view<[%w_flat_np]xi32> -> vector<8xi32>\n"
        "      %dqbn_raw = vector.load %w_sv[%w_laneb] : view<[%w_flat_np]xi32> -> vector<8xi32>\n", "weight loads")
    for gone in ("%dqan_origin", "%dqbn_origin"):
        if gone in text:
            raise SystemExit(f"GEMM256_FLATADDR: {gone} still referenced after the rewrite")
    return text


KERNEL = re.compile(r'^kernel\.def (?:target\(@(\w+)\) )?export\("(g_gemm_f16_\w*m256\w*)"\).*?\n\} launch\(.*?\n\}\n', re.S | re.M)
REWRITE = re.compile(r"g_gemm_f16_(acc|store|swiglu)_m256(_f16o|_qgs)?")   # _qgs: QG_SPLIT (qg_split.py), same loads as store
# known skip by name: the served swiglu GEMM keeps its 2-D addressing, as live pkg-e480 serves it. HRX-like E744 (q258, 2 x ABBA,
# 4 starts per arm): the rewrite (E514) made it 1.0093x slower in served cycles (+0.48 s per 81k); mechanism hypothesis E738
SKIP = frozenset({"g_gemm_f16_swiglu_m256_f16o"})


def apply(text):
    """rewrite every served 256-row f16 GEMM in a program; returns (text, names rewritten). Known-good forms only (above)."""
    done, out, last = [], [], 0
    for m in KERNEL.finditer(text):
        target, name = m.group(1), m.group(2)
        if target == "rdna4" and name.endswith("_rdna4"):
            continue   # gfx12 twin: not served on gfx1151 (the rewrite's anchors are the gfx11 K-tile-64 forms)
        if name in SKIP:
            continue   # left as generated (E744)
        if target not in (None, "rdna3") or not REWRITE.fullmatch(name):
            raise SystemExit(f"GEMM256_FLATADDR: {name} (target {target}) is neither rewritten nor a known skip: extend apply() or skip it by name")
        out += [text[last:m.start()], flataddr(m.group(0))]; last = m.end(); done.append(name)
    if not done:
        raise SystemExit("GEMM256_FLATADDR: no g_gemm_f16_*_m256 kernel in the program")
    return "".join(out) + text[last:], done
