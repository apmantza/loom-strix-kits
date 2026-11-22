"""IQ2_XXS weights (Nemotron 3.5 Bolt's routed and shared experts, kits/nemotron-bolt/NOTES.md): 256-value blocks of 66 bytes, an f16
d then 32 u16. Per 32-value group (8 bytes): word 0 = four grid indices (one byte each) into the 256-entry grid of eight magnitudes
in {8, 25, 43}; word 1 = four 7-bit sign indices (bits 7l .. 7l+6) and the group scale ls (bits 28-31), value = d (0.5 + ls) / 4
x grid x sign (llama.cpp dequantize_row_iq2_xxs). The sign table ksigns_iq2xs is i | parity(i) << 7, so it is computed, not read.

@g_iq2xxs_icodes(%w, %origin, %group) -> (codes, scale): the 32 signed int8 codes (grid magnitudes with their signs) and the
group scale d (0.5 + ls) / 4, for the int8 dot4 matvec path (ACT_Q8). The signs are applied on 32-bit words (HRX-like E233: no byte
vectors with runtime shifts): 4 sign bits spread to the bytes' low bits by one multiply (x 0x00204081: the 16 partial products
land on distinct bits, so nothing carries), times 255 for the byte masks, then (v ^ mask) + flag negates the flagged bytes (no
byte carries: magnitudes are at most 43). GRID picks where the 2 KB grid is read: 'rodata' (global constant memory, as
quant_values.loom's IQ3_S grid) or 'lds' (copied to workgroup memory by the kernel first; the helper then takes the LDS buffer).

probe(): a single-token matvec over one expert's rows (one workgroup per row, one lane per 32-value group), for static compiles
and the kernel lint before the MoE wiring (HRX-like E262). Grid: ggml-common.h iq2xxs_grid (llama.cpp, MIT)."""

GRID_HEX = "08080808080808082b080808080808081919080808080808082b0808080808082b2b0808080808081908190808080808081919080808080808082b08080808082b082b0808080808082b2b08080808082b2b2b0808080808190808190808080808190819080808080808191908080808082b19190808080819082b190808080808192b19080808080808082b080808082b08082b080808082b2b082b080808082b082b2b08080808190808081908080808190808190808080808190819080808191919081908080808080819190808080819082b19080808082b192b19080808080808082b0808082b0808082b0808082b082b082b0808082b08082b2b08080819080808081908080819080808190808080819080819080819082b080819080808192b080819080808080819081908082b08081908190808082b08190819080808082b19081908081908082b081908080819082b081908080808192b0819080808192b2b0819080808080808191908082b08080819190808082b08081919080808082b08191908082b19081919190808192b2b19191908080808082b191908081908192b19190808192b08082b190808080819082b190808080808192b1908080819082b2b19080808192b2b2b19080808080808082b080819190808082b0808082b0808082b080808191908082b0808082b2b08082b080819080819082b080808190819082b080808081919082b08082b081919082b0808082b082b082b080808190808192b080808080819192b08082b0808082b2b0808081919082b2b080819080808080819080819080808081908080819080808190819082b0808081908080808190808190808082b19080819080819082b080819080808192b080819081919192b080819080808080819081908082b08081908190808082b081908190808081919190819082b2b1919190819080808082b1908190808192b082b081908191908192b0819080808080808191908082b08080819190808082b080819190819192b0808191908192b0819081919080808082b08191908082b1908191919082b082b1919191908080808082b1919082b1919082b19190819080808082b190808190808082b190808081908082b190808080819082b19081908082b082b190808080808192b190819190808192b190808082b2b192b1908190819192b2b19080808080808082b082b08080808082b082b2b080808082b080819081908082b0819082b1908082b080808082b08082b082b08082b08082b08192b2b0819082b08082b081919082b08080808082b082b082b0808082b082b081908080808192b080819080808192b080808190808192b080808081908192b082b19191908192b080808080819192b081908081919192b0808192b1919192b080808192b2b192b08082b0808082b2b0808082b08082b2b080819192b082b2b08081908192b2b2b08190808080808081908190808080808190808190808080819082b19080808081919082b080808081908192b08080808190808081908080819082b0819080808192b1919190808081908082b19080808191908082b080808190819082b080808190808192b08080819080808081908081908082b081908081919082b19190808190808082b190808191919082b19080819190808082b080819080819082b080819082b08192b0808192b1919192b080819082b2b192b0808190808080808190819082b08080819081908082b08081908190808082b08190819192b192b081908192b0819081919081908192b0819190819080808082b19081919080808082b081908190808082b081908081908082b081908080819082b081919190819082b081908080808192b0819082b1919192b081919082b19192b08192b08082b192b0819191908192b2b08190808192b2b2b08190808080808081919082b0808080819191908190808081919192b19080808191908082b08080819190808082b08081919082b082b0808191908190808190819192b0808191908191908192b2b190819191908192b2b0819190808192b081919192b08192b081919192b2b080819191919190808082b191919081919192b19191908080808082b191919081908082b1919192b1908082b191908192b19082b191908080819192b1919082b08082b2b19190819080808082b190808190808082b190808081908082b19082b2b1908082b190808080819082b191919191919082b19082b19082b082b1908082b192b082b190808080808192b191919080808192b190808190819192b192b08190819192b190819082b19192b192b080819082b2b19080808080808082b2b0808080808082b2b2b08080808082b190808190808082b2b08082b0808082b081908081908082b082b19081908082b080808191908082b190819082b08082b190808080819082b081908080819082b080819080819082b191919080819082b080808190819082b08082b190819082b080808081919082b2b1908191919082b0819192b1919082b192b08082b19082b080808192b19082b08082b192b19082b2b080808082b082b08190808192b082b190819082b2b082b081908080808192b080819080808192b08192b080808192b080808190808192b19082b2b0808192b2b1919081908192b0808082b1908192b191908192b08192b080808080819192b2b082b080819192b081908190819192b190819191919192b1908082b082b192b08082b08192b192b2b08080808082b2b0808191908082b2b1919082b08082b2b192b080819082b2b080808082b082b2b082b190808192b2b08081919082b2b2b08190808192b2b2b"


def grid_rodata():
    return f'global.rodata.def @g_iq2xxs_grid_data = align(64) bytes("{GRID_HEX}")\n'


def sign_word(src, m4, out):
    """Loom text: i32 %{out} = the 4 int8 magnitudes in word %{src} with the signs of the 4 low bits of %{m4}"""
    return f"""  %{out}_spread = scalar.muli %{m4}, %k_spread : i32
  %{out}_flag = scalar.andi %{out}_spread, %k_lsb : i32
  %{out}_mask = scalar.muli %{out}_flag, %k_255 : i32
  %{out}_x = scalar.xori %{src}, %{out}_mask : i32
  %{out} = scalar.addi %{out}_x, %{out}_flag : i32
"""


def icodes(grid="rodata"):
    if grid == "rodata":
        sig, gsrc = "(%w: buffer, %origin: offset, %group: index)", "  %gdata = global.load @g_iq2xxs_grid_data : buffer\n"
    else:
        sig, gsrc = "(%w: buffer, %origin: offset, %group: index, %gdata: buffer)", ""
    body = []
    words = []
    for l in range(4):
        body.append(f"""  %sh{l} = scalar.constant {8 * l} : i32
  %ix_raw{l} = scalar.shrui %w0, %sh{l} : i32
  %ix_m{l} = scalar.andi %ix_raw{l}, %k_255 : i32
  %ix_b{l} = scalar.assume %ix_m{l} [range(%ix_m{l}, 0, 255)] : i32
  %ix{l} = index.cast %ix_b{l} : i32 to index
  %gw{l} = vector.load %gview[%ix{l}, %c0] : view<256x2xi32> -> vector<2xi32>
  %ga{l} = vector.extract %gw{l}[0] : vector<2xi32> -> i32
  %gb{l} = vector.extract %gw{l}[1] : vector<2xi32> -> i32
  %ssh{l} = scalar.constant {7 * l} : i32
  %s_raw{l} = scalar.shrui %w1, %ssh{l} : i32
  %s{l} = scalar.andi %s_raw{l}, %k_127 : i32
  %p4_{l} = scalar.shrui %s{l}, %k_4 : i32
  %pa{l} = scalar.xori %s{l}, %p4_{l} : i32
  %p2_{l} = scalar.shrui %pa{l}, %k_2 : i32
  %pb{l} = scalar.xori %pa{l}, %p2_{l} : i32
  %p1_{l} = scalar.shrui %pb{l}, %k_1 : i32
  %pc{l} = scalar.xori %pb{l}, %p1_{l} : i32
  %par{l} = scalar.andi %pc{l}, %k_1 : i32
  %pt{l} = scalar.shli %par{l}, %k_7 : i32
  %m8_{l} = scalar.ori %s{l}, %pt{l} : i32
  %lo{l} = scalar.andi %m8_{l}, %k_15 : i32
  %hi{l} = scalar.shrui %m8_{l}, %k_4 : i32
""" + sign_word(f"ga{l}", f"lo{l}", f"wa{l}") + sign_word(f"gb{l}", f"hi{l}", f"wb{l}"))
        words += [f"%wa{l}", f"%wb{l}"]
    return f"""
func.def inline @g_iq2xxs_icodes{sig} -> (vector<32xi8>, f32) {{
  %c0 = index.constant 0 : index
  %two = index.constant 2 : offset
  %d_view = buffer.view %w[%origin] : buffer -> view<1xf16>
  %d_half = view.load %d_view[0] : view<1xf16> -> f16
  %d = scalar.extf %d_half : f16 to f32
  %g8 = index.constant 8 : offset
  %c8i = index.constant 8 : index
  %sub_raw = index.rem %group, %c8i : index
  %q_off = index.scale %sub_raw, %g8 : index, offset -> offset
  %q_base = index.add %origin, %two : offset
  %q_origin = index.add %q_base, %q_off : offset
  %q_view = buffer.view %w[%q_origin] : buffer -> view<4xi16>
  %q16 = vector.load %q_view[0] : view<4xi16> -> vector<4xi16>
  %q32 = vector.bitcast %q16 : vector<4xi16> to vector<2xi32>
  %w0 = vector.extract %q32[0] : vector<2xi32> -> i32
  %w1 = vector.extract %q32[1] : vector<2xi32> -> i32
{gsrc}  %gzero = index.constant 0 : offset
  %gview = buffer.view %gdata[%gzero] : buffer -> view<256x2xi32>
  %k_1 = scalar.constant 1 : i32
  %k_2 = scalar.constant 2 : i32
  %k_4 = scalar.constant 4 : i32
  %k_7 = scalar.constant 7 : i32
  %k_15 = scalar.constant 15 : i32
  %k_28 = scalar.constant 28 : i32
  %k_127 = scalar.constant 127 : i32
  %k_255 = scalar.constant 255 : i32
  %k_spread = scalar.constant 2113665 : i32
  %k_lsb = scalar.constant 16843009 : i32
{"".join(body)}  %ls = scalar.shrui %w1, %k_28 : i32
  %lsf = scalar.sitofp %ls : i32 to f32
  %half = scalar.constant 0.5 : f32
  %quarter = scalar.constant 0.25 : f32
  %lsh = scalar.addf %lsf, %half : f32
  %dq = scalar.mulf %d, %quarter : f32
  %scale = scalar.mulf %dq, %lsh : f32
  %cw = vector.from_elements {", ".join(words)} : vector<8xi32>
  %codes = vector.bitcast %cw : vector<8xi32> to vector<32xi8>
  func.return %codes, %scale : vector<32xi8>, f32
}}
"""


def probe(K=2560, grid="rodata"):
    """g_mv_iq2xxs_probe_<grid>(%rows): y[row] = sum over K of w[row] . x for one token: one workgroup of 128 lanes per row, lane g
    takes the 32-value group g (K / 32 <= 128 lanes), the ACT_Q8 codes and scale of the token (%xq int8 [K], %xd f32 [K / 32])"""
    assert K % 256 == 0 and K // 32 <= 128
    G = K // 32
    dots, acc = [], "%z1"
    for k in range(8):
        dots.append(f"""  %wk{k} = vector.slice %codes[{4 * k}] : vector<32xi8> -> vector<4xi8>
  %xk{k} = vector.slice %xc[{4 * k}] : vector<32xi8> -> vector<4xi8>
  %d{k} = vector.dot4i<s8s8> %wk{k}, %xk{k}, {acc} : vector<4xi8>, vector<4xi8>, vector<1xi32>""")
        acc = f"%d{k}"
    name = f"g_mv_iq2xxs_probe_{grid}"
    if grid == "lds":
        stage = """  %gl_bytes = index.constant 2048 : offset
  %gl = buffer.alloca<workgroup> align(16) %gl_bytes : buffer
  %gsrc = global.load @g_iq2xxs_grid_data : buffer
  %gsrc_view = buffer.view %gsrc[%zero] : buffer -> view<128x4xi32>
  %gl_view = buffer.view %gl[%zero] : buffer -> view<128x4xi32>
  %gpart = vector.load %gsrc_view[%lane, %c0] : view<128x4xi32> -> vector<4xi32>
  vector.store %gpart, %gl_view[%lane, %c0] : vector<4xi32>, view<128x4xi32>
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
"""
        call = "func.call @g_iq2xxs_icodes(%w, %origin, %g, %gl) : (buffer, offset, index, buffer) -> (vector<32xi8>, f32)"
    else:
        stage = ""
        call = "func.call @g_iq2xxs_icodes(%w, %origin, %g) : (buffer, offset, index) -> (vector<32xi8>, f32)"
    return grid_rodata() + icodes(grid) + f"""
kernel.def export("{name}") @{name}(%rows: index) {{
  %c1 = index.constant 1 : index
  %c128 = index.constant 128 : index
  kernel.launch.config workgroups(%rows, %c1, %c1) workgroup_size(%c128, %c1, %c1) : index
}} launch(%rows_arg: index, %w: buffer, %xq: buffer, %xd: buffer, %y: buffer) {{
  %rows = index.assume %rows_arg [range(%rows_arg, 1, 65536)] : index
  %row_id = kernel.workgroup.id<x> : index
  %row = index.assume %row_id [lt(%row_id, %rows)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 127)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c8 = index.constant 8 : index
  %groups = index.constant {G} : index
  %z1 = vector.splati 0 : vector<1xi32>
  %zero_f = scalar.constant 0.0 : f32
{stage}  %active = index.cmp ult, %lane, %groups : index
  %glast = index.constant {G - 1} : index
  %g_raw = index.min %lane, %glast : index
  %g = index.assume %g_raw [range(%g_raw, 0, {G - 1})] : index
  %row_bytes = index.constant {K // 256 * 66} : offset
  %block_bytes = index.constant 66 : offset
  %blk = index.div %g, %c8 : index
  %ro = index.scale %row, %row_bytes : index, offset -> offset
  %bo = index.scale %blk, %block_bytes : index, offset -> offset
  %origin = index.add %ro, %bo : offset
  %codes, %scale = {call}
  %xq_view = buffer.view %xq[%zero] : buffer -> view<{G}x32xi8>
  %xd_view = buffer.view %xd[%zero] : buffer -> view<{G}xf32>
  %xc = vector.load %xq_view[%g, %c0] : view<{G}x32xi8> -> vector<32xi8>
  %xs = view.load %xd_view[%g] : view<{G}xf32> -> f32
{chr(10).join(dots)}
  %isum = vector.extract {acc}[0] : vector<1xi32> -> i32
  %fsum = scalar.sitofp %isum : i32 to f32
  %ps = scalar.mulf %fsum, %scale : f32
  %px = scalar.mulf %ps, %xs : f32
  %pa = scf.select %active, %px, %zero_f : f32
  %sum = kernel.workgroup.reduce<addf> %pa : f32
  %first = index.cmp eq, %lane, %c0 : index
  scf.if %first {{
    %y_view = buffer.view %y[%zero] : buffer -> view<[%rows]xf32>
    view.store %sum, %y_view[%row] : f32, view<[%rows]xf32>
  }}
  kernel.return
}}
"""
