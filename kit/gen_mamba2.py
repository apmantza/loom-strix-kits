"""Mamba-2 (SSD) recurrence, decode (Nemotron 3.5 Bolt's 40 Mamba layers, kits/nemotron-bolt/NOTES.md): per head h a scalar decay,
per state row p of the head a 128-wide state (d_state N), B and C shared by the heads of a group. One token:
    sp = softplus(dt[h] + dt_bias[h]);  dA = exp(sp * A[h])                     (A = -exp(A_log), per head)
    s'[h, p, :] = s[h, p, :] * dA + B[g, :] * (x[h, p] * sp)                     (g = h / (H / G))
    y[h, p] = sum_n s'[h, p, n] * C[g, n] + D[h] * x[h, p]
in llama.cpp's operation order (ggml_compute_forward_ssm_scan_f32: two products, then the sum; D added after the scan,
mamba-base.cpp), as GDN's decay is computed here (gen_gdn.py: scalar.softplusf, scalar.expf). The gate y * silu(z), the grouped
RMSNorm and the conv1d stay separate kernels.

probe(): one decode step over a layer's whole state: a 32-lane workgroup owns 4 state rows of one head; lane r of a row's 8 lanes
holds 16 of its 128 state values, the row's sum is a 3-step xor-shuffle sum (gen_gdn's row-lanes form). Per layer the state is read
and written once (Bolt: 128 x 64 x 128 f32 = 4 MiB each way). For static compiles and the kernel lint (HRX-like E263)."""


def probe(H=128, P=64, N=128, G=8, RL=8):
    assert P % (32 // RL) == 0 and N % RL == 0 and H % G == 0
    RPW, NL = 32 // RL, N // RL                 # state rows per workgroup, state values per lane
    VN = f"vector<{NL}xf32>"
    xs = [1 << i for i in range(RL.bit_length() - 1)]
    red, cur = [], "%part"
    for d in xs:
        red.append(f"  %part_x{d}, %part_u{d} = kernel.subgroup.shuffle<xor> {cur}, %x{d}, %w32 : f32, i32, i32\n"
                   f"  %part_s{d} = scalar.addf {cur}, %part_x{d} : f32")
        cur = f"%part_s{d}"
    shuf = "\n".join(f"  %x{d} = scalar.constant {d} : i32" for d in xs)
    return f"""
kernel.def export("g_mamba2_step_probe") @g_mamba2_step_probe() {{
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  %heads = index.constant {H} : index
  %blocks = index.constant {P // RPW} : index
  kernel.launch.config workgroups(%heads, %blocks, %c1) workgroup_size(%c32, %c1, %c1) : index
}} launch(%state: buffer, %x: buffer, %b: buffer, %c: buffer, %dt: buffer, %dt_bias: buffer, %a: buffer, %d: buffer, %y: buffer) {{
  %h_id = kernel.workgroup.id<x> : index
  %h = index.assume %h_id [range(%h_id, 0, {H - 1})] : index
  %blk_id = kernel.workgroup.id<y> : index
  %blk = index.assume %blk_id [range(%blk_id, 0, {P // RPW - 1})] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %zero_f = scalar.constant 0.0 : f32
  %w32 = scalar.constant 32 : i32
{shuf}
  %rl = index.constant {RL} : index
  %rpw = index.constant {RPW} : index
  %nl = index.constant {NL} : index
  %hpg = index.constant {H // G} : index
  %prow = index.div %lane, %rl : index
  %r = index.rem %lane, %rl : index
  %pbase = index.mul %blk, %rpw : index
  %p = index.add %pbase, %prow : index
  %n0 = index.mul %r, %nl : index
  %g = index.div %h, %hpg : index
  %head_view = buffer.view %dt[%zero] : buffer -> view<{H}xf32>
  %bias_view = buffer.view %dt_bias[%zero] : buffer -> view<{H}xf32>
  %a_view = buffer.view %a[%zero] : buffer -> view<{H}xf32>
  %d_view = buffer.view %d[%zero] : buffer -> view<{H}xf32>
  %dtv = view.load %head_view[%h] : view<{H}xf32> -> f32
  %dtb = view.load %bias_view[%h] : view<{H}xf32> -> f32
  %av = view.load %a_view[%h] : view<{H}xf32> -> f32
  %dv = view.load %d_view[%h] : view<{H}xf32> -> f32
  %z = scalar.addf %dtv, %dtb : f32
  %sp = scalar.softplusf<afn> %z : f32
  %gate = scalar.mulf %sp, %av : f32
  %da = scalar.expf<afn> %gate : f32
  %x_view = buffer.view %x[%zero] : buffer -> view<{H}x{P}xf32>
  %xv = view.load %x_view[%h, %p] : view<{H}x{P}xf32> -> f32
  %xdt = scalar.mulf %xv, %sp : f32
  %s_view = buffer.view %state[%zero] : buffer -> view<{H}x{P}x{N}xf32>
  %b_view = buffer.view %b[%zero] : buffer -> view<{G}x{N}xf32>
  %c_view = buffer.view %c[%zero] : buffer -> view<{G}x{N}xf32>
  %s = vector.load %s_view[%h, %p, %n0] : view<{H}x{P}x{N}xf32> -> {VN}
  %bv = vector.load %b_view[%g, %n0] : view<{G}x{N}xf32> -> {VN}
  %cv = vector.load %c_view[%g, %n0] : view<{G}x{N}xf32> -> {VN}
  %dav = vector.splat %da : {VN}
  %xdtv = vector.splat %xdt : {VN}
  %sd = vector.mulf %s, %dav : {VN}
  %bx = vector.mulf %bv, %xdtv : {VN}
  %sn = vector.addf %sd, %bx : {VN}
  vector.store %sn, %s_view[%h, %p, %n0] : {VN}, view<{H}x{P}x{N}xf32>
  %part = vector.dotf<reassoc|contract> %sn, %cv, %zero_f : {VN}, {VN}, f32
{chr(10).join(red)}
  %dx = scalar.mulf %dv, %xv : f32
  %yv = scalar.addf {cur}, %dx : f32
  %c0 = index.constant 0 : index
  %lead = index.cmp eq, %r, %c0 : index
  scf.if %lead {{
    %y_view = buffer.view %y[%zero] : buffer -> view<{H}x{P}xf32>
    view.store %yv, %y_view[%h, %p] : f32, view<{H}x{P}xf32>
  }}
  kernel.return
}}
"""
