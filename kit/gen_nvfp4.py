"""NVFP4 experts (Inkling 2 layers 0-19, kits/inkling-2-mini/NOTES.md): e2m1 nibbles + one e4m3 scale per 16 values + a per-expert fp32
factor. Split planes as the native checkpoint stores them: the nibble plane %w (16 bytes per 32 values, the low nibble the even
element: checked on Inkling's values) and the scale plane %ws (2 e4m3 bytes per 32 values).

@g_nvfp4_icodes(%w, %ws, %origin, %scale_origin, %group) -> (codes, s_lo, s_hi): the signed codes 2 * e2m1 (0 1 2 3 4 6 8 12 and
negatives, the MXFP4 table of gen_mxfp4s: a v_perm lookup on bytes) of the 32-value group at byte %origin of %w, and its two
half scales e4m3 / 2 (the codes carry the factor 2) for values 0-15 and 16-31: the two-scale call shape of gen_llm.iq_dot (as q6_k).
The per-expert factor is applied by the caller once per row. gfx11 has no fp8 conversion: the e4m3 byte is decoded on integers
(normal: (exp - 7 + 127) << 23 | mant << 20; subnormal: mant * 2^-9; checked over all 256 bytes in test_nvfp4.py).

probe(): a single-token matvec over one expert's rows (one workgroup per row, one lane per 32-value group, the int8 dot of
ACT_Q8 activations), for static compiles and the kernel lint before the MoE wiring (HRX-like E250)."""


def e4m3_scalar(byte_var, out):
    """Loom text: f32 %{out} = the e4m3fn value of the i32 %{byte_var} (0..255; sign ignored: NVFP4 scales are non-negative)"""
    return f"""  %{out}_c7 = scalar.constant 7 : i32
  %{out}_c15 = scalar.constant 15 : i32
  %{out}_c3 = scalar.constant 3 : i32
  %{out}_c120 = scalar.constant 120 : i32
  %{out}_c20 = scalar.constant 20 : i32
  %{out}_c23 = scalar.constant 23 : i32
  %{out}_c0 = scalar.constant 0 : i32
  %{out}_mant = scalar.andi %{byte_var}, %{out}_c7 : i32
  %{out}_eraw = scalar.shrui %{byte_var}, %{out}_c3 : i32
  %{out}_exp = scalar.andi %{out}_eraw, %{out}_c15 : i32
  %{out}_ebias = scalar.addi %{out}_exp, %{out}_c120 : i32
  %{out}_ebits = scalar.shli %{out}_ebias, %{out}_c23 : i32
  %{out}_mbits = scalar.shli %{out}_mant, %{out}_c20 : i32
  %{out}_nbits = scalar.ori %{out}_ebits, %{out}_mbits : i32
  %{out}_normal = scalar.bitcast %{out}_nbits : i32 to f32
  %{out}_mf = scalar.sitofp %{out}_mant : i32 to f32
  %{out}_sub_scale = scalar.constant 0.001953125 : f32
  %{out}_sub = scalar.mulf %{out}_mf, %{out}_sub_scale : f32
  %{out}_is_sub = scalar.cmpi eq, %{out}_exp, %{out}_c0 : i32
  %{out} = scf.select %{out}_is_sub, %{out}_sub, %{out}_normal : f32
"""


def nvfp4_icodes():
    return f"""
func.def inline @g_nvfp4_icodes(%w: buffer, %ws: buffer, %origin: offset, %scale_origin: offset, %group: index) -> (vector<32xi8>, f32, f32) {{
  %q_view = buffer.view %w[%origin] : buffer -> view<16xi8>
  %s_view = buffer.view %ws[%scale_origin] : buffer -> view<2xi8>
  %sb = vector.load %s_view[0] : view<2xi8> -> vector<2xi8>
  %sb0 = vector.extract %sb[0] : vector<2xi8> -> i8
  %sb1 = vector.extract %sb[1] : vector<2xi8> -> i8
  %b0 = scalar.extui %sb0 : i8 to i32
  %b1 = scalar.extui %sb1 : i8 to i32
{e4m3_scalar("b0", "e0")}{e4m3_scalar("b1", "e1")}  %half = scalar.constant 0.5 : f32
  %s_lo = scalar.mulf %e0, %half : f32
  %s_hi = scalar.mulf %e1, %half : f32
  %payload = vector.load %q_view[0] : view<16xi8> -> vector<16xi8>
  %mask = vector.splati 15 : vector<16xi8>
  %four = vector.splati 4 : vector<16xi8>
  %lower = vector.andi %payload, %mask : vector<16xi8>
  %shifted = vector.shrui %payload, %four : vector<16xi8>
  %upper = vector.andi %shifted, %mask : vector<16xi8>
  %k0 = scalar.constant 0 : i8
  %k1 = scalar.constant 1 : i8
  %k2 = scalar.constant 2 : i8
  %k3 = scalar.constant 3 : i8
  %k4 = scalar.constant 4 : i8
  %k5 = scalar.constant 6 : i8
  %k6 = scalar.constant 8 : i8
  %k7 = scalar.constant 12 : i8
  %k9 = scalar.constant -1 : i8
  %k10 = scalar.constant -2 : i8
  %k11 = scalar.constant -3 : i8
  %k12 = scalar.constant -4 : i8
  %k13 = scalar.constant -6 : i8
  %k14 = scalar.constant -8 : i8
  %k15 = scalar.constant -12 : i8
  %codebook = vector.from_elements %k0, %k1, %k2, %k3, %k4, %k5, %k6, %k7, %k0, %k9, %k10, %k11, %k12, %k13, %k14, %k15 : vector<16xi8>
  %lo_codes = vector.table.lookup %codebook[%lower] : vector<16xi8>, vector<16xi8> -> vector<16xi8>
  %hi_codes = vector.table.lookup %codebook[%upper] : vector<16xi8>, vector<16xi8> -> vector<16xi8>
  %codes = vector.interleave<0> %lo_codes, %hi_codes : vector<16xi8>, vector<16xi8> -> vector<32xi8>
  func.return %codes, %s_lo, %s_hi : vector<32xi8>, f32, f32
}}
"""


def probe(K=2304):
    """g_mv_nvfp4_probe(%rows): y[row] = factor * sum over K of w[row] . x for one token: one workgroup per row, lane g takes the
    32-value group g (K / 32 lanes), the ACT_Q8 codes and scale of the token (%xq int8 [K], %xd f32 [K / 32])"""
    G = K // 32
    dots = []
    for half, k0 in (("l", 0), ("h", 4)):
        acc = "%z1"
        for k in range(k0, k0 + 4):
            dots.append(f"""    %wk{half}{k} = vector.slice %codes[{4 * k}] : vector<32xi8> -> vector<4xi8>
    %xk{half}{k} = vector.slice %xc[{4 * k}] : vector<32xi8> -> vector<4xi8>
    %d{half}{k} = vector.dot4i<s8s8> %wk{half}{k}, %xk{half}{k}, {acc} : vector<4xi8>, vector<4xi8>, vector<1xi32>""")
            acc = f"%d{half}{k}"
        dots.append(f"    %i{half} = vector.extract {acc}[0] : vector<1xi32> -> i32\n    %f{half} = scalar.sitofp %i{half} : i32 to f32")
    body = "\n".join(dots)
    return nvfp4_icodes() + f"""
kernel.def export("g_mv_nvfp4_probe") @g_mv_nvfp4_probe(%rows: index) {{
  %c1 = index.constant 1 : index
  %c128 = index.constant 128 : index
  kernel.launch.config workgroups(%rows, %c1, %c1) workgroup_size(%c128, %c1, %c1) : index
}} launch(%rows_arg: index, %w: buffer, %ws: buffer, %xq: buffer, %xd: buffer, %factor: buffer, %y: buffer) {{
  %rows = index.assume %rows_arg [range(%rows_arg, 1, 65536)] : index
  %row_id = kernel.workgroup.id<x> : index
  %row = index.assume %row_id [lt(%row_id, %rows)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 127)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %groups = index.constant {G} : index
  %z1 = vector.splati 0 : vector<1xi32>
  %zero_f = scalar.constant 0.0 : f32
  %active = index.cmp ult, %lane, %groups : index
  %glast = index.constant {G - 1} : index
  %g_raw = index.min %lane, %glast : index
  %g = index.assume %g_raw [range(%g_raw, 0, {G - 1})] : index
  %row_bytes = index.constant {K // 2} : offset
  %srow_bytes = index.constant {K // 16} : offset
  %sixteen = index.constant 16 : offset
  %two = index.constant 2 : offset
  %ro = index.scale %row, %row_bytes : index, offset -> offset
  %go = index.scale %g, %sixteen : index, offset -> offset
  %origin = index.add %ro, %go : offset
  %sro = index.scale %row, %srow_bytes : index, offset -> offset
  %sgo = index.scale %g, %two : index, offset -> offset
  %sorigin = index.add %sro, %sgo : offset
  %codes, %s_lo, %s_hi = func.call @g_nvfp4_icodes(%w, %ws, %origin, %sorigin, %g) : (buffer, buffer, offset, offset, index) -> (vector<32xi8>, f32, f32)
  %xq_view = buffer.view %xq[%zero] : buffer -> view<{G}x32xi8>
  %xd_view = buffer.view %xd[%zero] : buffer -> view<{G}xf32>
  %xc = vector.load %xq_view[%g, %c0] : view<{G}x32xi8> -> vector<32xi8>
  %xs = view.load %xd_view[%g] : view<{G}xf32> -> f32
{body}
  %pl = scalar.mulf %fl, %s_lo : f32
  %p = scalar.fmaf %fh, %s_hi, %pl : f32
  %px = scalar.mulf %p, %xs : f32
  %pa = scf.select %active, %px, %zero_f : f32
  %sum = kernel.workgroup.reduce<addf> %pa : f32
  %first = index.cmp eq, %lane, %c0 : index
  scf.if %first {{
    %f_view = buffer.view %factor[%zero] : buffer -> view<1xf32>
    %fac = view.load %f_view[0] : view<1xf32> -> f32
    %out = scalar.mulf %sum, %fac : f32
    %y_view = buffer.view %y[%zero] : buffer -> view<[%rows]xf32>
    view.store %out, %y_view[%row] : f32, view<[%rows]xf32>
  }}
  kernel.return
}}
"""
