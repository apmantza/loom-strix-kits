"""Inkling 2 attention inputs (kits/inkling-2-mini/NOTES.md; HRX-like E274). As Unsloth's llama.cpp fork builds them (inkling2.cpp graph and
clip()), per token after the projections and the K / V short convolutions:
    clip(x, scale, thr) = (x * scale) * clamp(thr / (|x * scale| + 1e-6), 0, 1)        (per head; scale per channel, or none)
    q = clip(q, q_clip[head], qk_clip)        k = clip(k, -, qk_clip)        v = rms_norm(v, eps)        r = clip(r, rel_clip[head], rel_clip)
    global layers: q *= tau, r *= tau, tau = 1 + alpha * log(max(1, (pos + 1) / floor))
The attention then scores q . k / sqrt(head_dim) + r . rel_bias[min(d, extent)] - sink (rel_bias padded with a zero row at d = extent,
so keys at distance >= extent get -sink), softmax with a zero-logit sink, and a sigmoid output gate: the bias is 16 multiply-adds per key
from a 64 KB table, computed in the attention kernel, so nothing per distance is materialised here.

prep(T, global_layer): the clips / norms of T tokens (one workgroup per token, 8 waves; each wave owns a share of the heads, a head's
values spread over its 32 lanes, the norm a subgroup sum). Inkling 2 Mini: 24 query heads, 8 KV heads, head 128, relative length 16."""


def prep(T=1, glob=False, H=24, HKV=8, D=128, R=16, qk_clip=22.627416610717773, rel_clip=8.0, eps=9.999999974752427e-07,
         alpha=0.10000000149011612, floor=8192, name=None):
    assert D % 32 == 0 and H % 8 == 0 and HKV % 8 == 0 and R <= 32
    name = name or f"g_ink_attn_prep_t{T}" + ("_global" if glob else "")
    DL = D // 32
    L = []

    def head(kind, view, n_heads, per_wave, scale_view=None, thr=None, rms=False, tau=False):
        """lines processing `per_wave` heads of `kind` in this wave: load, (scale), norm by subgroup sum, clip or rms, (tau), store"""
        for j in range(per_wave):
            h = f"%{kind}h{j}"
            L.append(f"  %{kind}hb{j} = index.constant {j} : index\n  %{kind}hm{j} = index.mul %wave, %{kind}pw : index\n"
                     f"  {h} = index.add %{kind}hm{j}, %{kind}hb{j} : index")
            width = R if kind == "r" else D
            per = 1 if kind == "r" else DL
            for i in range(per):
                col = f"%{kind}c{j}_{i}"
                if kind == "r":
                    L.append(f"  %{kind}cl{j} = index.min %wl, %rlast : index\n  {col} = index.assume %{kind}cl{j} [range(%{kind}cl{j}, 0, {R - 1})] : index")
                else:
                    L.append(f"  %{kind}ci{j}_{i} = index.constant {i * 32} : index\n  {col} = index.add %wl, %{kind}ci{j}_{i} : index")
                L.append(f"  %{kind}x{j}_{i} = view.load {view}[%token, {h}, {col}] : view<{T}x{n_heads}x{width}xf32> -> f32")
                if scale_view:
                    L.append(f"  %{kind}s{j}_{i} = view.load {scale_view}[{h}, {col}] : view<{n_heads}x{width}xf32> -> f32\n"
                             f"  %{kind}y{j}_{i} = scalar.mulf %{kind}x{j}_{i}, %{kind}s{j}_{i} : f32")
                else:
                    L.append(f"  %{kind}y{j}_{i} = scalar.addf %{kind}x{j}_{i}, %zero_f : f32")
                if kind == "r":
                    L.append(f"  %{kind}yz{j}_{i} = scf.select %rlive, %{kind}y{j}_{i}, %zero_f : f32\n"
                             f"  %{kind}q{j}_{i} = scalar.mulf %{kind}yz{j}_{i}, %{kind}yz{j}_{i} : f32")
                else:
                    L.append(f"  %{kind}q{j}_{i} = scalar.mulf %{kind}y{j}_{i}, %{kind}y{j}_{i} : f32")
            acc = f"%{kind}q{j}_0"
            for i in range(1, per):
                L.append(f"  %{kind}a{j}_{i} = scalar.addf {acc}, %{kind}q{j}_{i} : f32")
                acc = f"%{kind}a{j}_{i}"
            L.append(f"  %{kind}ss{j} = kernel.subgroup.reduce<addf> {acc} : f32")
            if rms:
                L.append(f"""  %{kind}mean{j} = scalar.divf %{kind}ss{j}, %dfl : f32
  %{kind}st{j} = scalar.addf %{kind}mean{j}, %eps : f32
  %{kind}f{j} = scalar.rsqrtf %{kind}st{j} : f32""")
            else:
                L.append(f"""  %{kind}n0{j} = scalar.sqrtf %{kind}ss{j} : f32
  %{kind}n{j} = scalar.addf %{kind}n0{j}, %clip_eps : f32
  %{kind}t{j} = scalar.constant {thr!r} : f32
  %{kind}d{j} = scalar.divf %{kind}t{j}, %{kind}n{j} : f32
  %{kind}d0{j} = scalar.maxnumf %{kind}d{j}, %zero_f : f32
  %{kind}f{j} = scalar.minnumf %{kind}d0{j}, %one : f32""")
            for i in range(per):
                L.append(f"  %{kind}o{j}_{i} = scalar.mulf %{kind}y{j}_{i}, %{kind}f{j} : f32")
                out = f"%{kind}o{j}_{i}"
                if tau:
                    L.append(f"  %{kind}ot{j}_{i} = scalar.mulf {out}, %tau : f32")
                    out = f"%{kind}ot{j}_{i}"
                width = R if kind == "r" else D
                st = f"  view.store {out}, {view}[%token, {h}, %{kind}c{j}_{i}] : f32, view<{T}x{n_heads}x{width}xf32>"
                L.append(f"  scf.if %rlive {{\n  {st}\n  }}" if kind == "r" else st)
    head("q", "%q_view", H, H // 8, "%qc_view", qk_clip, tau=glob)
    head("k", "%k_view", HKV, HKV // 8, None, qk_clip)
    head("v", "%v_view", HKV, HKV // 8, rms=True)
    head("r", "%r_view", H, H // 8, "%rc_view", rel_clip, tau=glob)
    tau = ""
    if glob:   # tau = 1 + alpha log(max(1, (pos + 1) / floor)), as the fork: scale_bias(pos, 1, 1), scale 1/floor, clamp, log, scale_bias
        tau = f"""  %pos_view = buffer.view %positions[%zero] : buffer -> view<{T}xi32>
  %pos_i = view.load %pos_view[%token] : view<{T}xi32> -> i32
  %pos_f = scalar.sitofp %pos_i : i32 to f32
  %pos1 = scalar.addf %pos_f, %one : f32
  %inv_floor = scalar.constant {1.0 / floor!r} : f32
  %ratio = scalar.mulf %pos1, %inv_floor : f32
  %ratio1 = scalar.maxnumf %ratio, %one : f32
  %lg = scalar.logf<afn> %ratio1 : f32
  %alpha = scalar.constant {alpha!r} : f32
  %al = scalar.mulf %lg, %alpha : f32
  %tau = scalar.addf %al, %one : f32
"""
    pargs = ", %positions: buffer" if glob else ""
    return f"""
kernel.def export("{name}") @{name}() {{
  %c1 = index.constant 1 : index
  %tok = index.constant {T} : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tok, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%q: buffer, %k: buffer, %v: buffer, %r: buffer, %qclip: buffer, %rclip: buffer{pargs}) {{
  %zero = index.constant 0 : offset
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [range(%token_id, 0, {T - 1})] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, 7)] : index
  %wl_id = kernel.subgroup.lane.id : index
  %wl = index.assume %wl_id [range(%wl_id, 0, 31)] : index
  %zero_f = scalar.constant 0.0 : f32
  %one = scalar.constant 1.0 : f32
  %clip_eps = scalar.constant 1e-06 : f32
  %eps = scalar.constant {eps!r} : f32
  %dfl = scalar.constant {float(D)!r} : f32
  %rlimit = index.constant {R} : index
  %rlive = index.cmp ult, %wl, %rlimit : index
  %rlast = index.constant {R - 1} : index
  %qpw = index.constant {H // 8} : index
  %kpw = index.constant {HKV // 8} : index
  %vpw = index.constant {HKV // 8} : index
  %rpw = index.constant {H // 8} : index
  %q_view = buffer.view %q[%zero] : buffer -> view<{T}x{H}x{D}xf32>
  %k_view = buffer.view %k[%zero] : buffer -> view<{T}x{HKV}x{D}xf32>
  %v_view = buffer.view %v[%zero] : buffer -> view<{T}x{HKV}x{D}xf32>
  %r_view = buffer.view %r[%zero] : buffer -> view<{T}x{H}x{R}xf32>
  %qc_view = buffer.view %qclip[%zero] : buffer -> view<{H}x{D}xf32>
  %rc_view = buffer.view %rclip[%zero] : buffer -> view<{H}x{R}xf32>
{tau}{chr(10).join(L)}
  kernel.return
}}
"""


def decode_swa(W=1024, H=24, HKV=8, D=128, R=16, name=None):
    """g_ink_attn_dec_swa: one query token's sliding-window attention (Inkling 2's 40 local slots), f32 K / V cache [n_kv][HKV][D] (the
    token's own K / V already appended: n_kv = its position + 1). One workgroup per query head (GQA: kv head = h / (H / HKV)):
      1. each lane scores its keys j (distance d = n_kv - 1 - j < W): s = (q . k_j) / sqrt(D) + r . rel_bias[d] - sink, into LDS;
      2. m = max(0, max s) (the zero-logit sink takes part), p_j = exp(s_j - m), den = exp(-m) + sum p_j (ggml soft_max with zero sinks);
      3. lanes 0-127 / 128-255 sum p_j v_j over the even / odd keys for one output dim each, added through LDS;
      4. out = (sum / den) * sigmoid(gate), the fork's attn_gate (wqkv_gate) applied per element.
    Inputs: q [H][D], r [H][R], sink [H], gate [H][D] (the gate projection's logits), rel_bias [W][R] (ggml {R, extent}), k / v cache;
    output [H][D]. The global layers (distances past W: bias -sink, contexts far past the LDS) need a split-K form, as vsplit."""
    T = 256
    assert D == 128 and W % T == 0
    name = name or f"g_ink_attn_dec_swa{W}"
    KPL = W // T
    sc = []
    for c in range(KPL):
        sc.append(f"""  %jc{c} = index.constant {c * T} : index
  %dd{c} = index.add %lane, %jc{c} : index
  %ok{c} = index.cmp ult, %dd{c}, %nwin : index
  %dist{c} = index.min %dd{c}, %wlast : index
  %d{c} = index.assume %dist{c} [range(%dist{c}, 0, {W - 1})] : index
  %dl{c} = index.min %d{c}, %last : index
  %jr{c} = index.sub %last, %dl{c} : index
  %j{c} = index.assume %jr{c} [range(%jr{c}, 0, 1048575), lt(%jr{c}, %n_kv)] : index
  %kd{c} = scf.for %ch = [%c0 to %cD step %c16] (%acc = %zero_f : f32) -> (f32) {{
    %kv{c} = vector.load %k_view[%j{c}, %kvh, %ch] : view<[%n_kv]x{HKV}x{D}xf32> -> vector<16xf32>
    %qv{c} = vector.load %q_view[%h, %ch] : view<{H}x{D}xf32> -> vector<16xf32>
    %pp{c} = vector.dotf<reassoc|contract> %kv{c}, %qv{c}, %acc : vector<16xf32>, vector<16xf32>, f32
    scf.yield %pp{c} : f32
  }}
  %rb{c} = vector.load %rb_view[%d{c}, %c0] : view<{W}x{R}xf32> -> vector<{R}xf32>
  %bias{c} = vector.dotf<reassoc|contract> %rb{c}, %rv, %zero_f : vector<{R}xf32>, vector<{R}xf32>, f32
  %qk{c} = scalar.mulf %kd{c}, %kscale : f32
  %sb{c} = scalar.addf %qk{c}, %bias{c} : f32
  %s{c}_raw = scalar.subf %sb{c}, %sink : f32
  %s{c} = scf.select %ok{c}, %s{c}_raw, %neg_inf : f32
  %ci{c} = index.add %lane, %jc{c} : index
  view.store %s{c}, %s_lds[%ci{c}] : f32, view<{W}xf32>""")
    mx = "%zero_f"
    for c in range(KPL):
        sc.append(f"  %lm{c} = scalar.maxnumf {mx}, %s{c} : f32")
        mx = f"%lm{c}"
    return f"""
kernel.def export("{name}") @{name}(%n_kv: index) {{
  %c1 = index.constant 1 : index
  %heads = index.constant {H} : index
  %c{T} = index.constant {T} : index
  kernel.launch.config workgroups(%heads, %c1, %c1) workgroup_size(%c{T}, %c1, %c1) : index
}} launch(%n_kv_arg: index, %q: buffer, %r: buffer, %sinks: buffer, %gate: buffer, %rel_bias: buffer, %kc: buffer, %vc: buffer, %out: buffer) {{
  %n_kv = index.assume %n_kv_arg [range(%n_kv_arg, 1, 1048576)] : index
  %h_id = kernel.workgroup.id<x> : index
  %h = index.assume %h_id [range(%h_id, 0, {H - 1})] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, {T - 1})] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c16 = index.constant 16 : index
  %c{T} = index.constant {T} : index
  %cD = index.constant {D} : index
  %zero_f = scalar.constant 0.0 : f32
  %one = scalar.constant 1.0 : f32
  %neg_inf = scalar.constant -3.0e38 : f32
  %kscale = scalar.constant {1.0 / D ** 0.5!r} : f32
  %gq = index.constant {H // HKV} : index
  %kvh_raw = index.div %h, %gq : index
  %kvh = index.assume %kvh_raw [range(%kvh_raw, 0, {HKV - 1})] : index
  %wconst = index.constant {W} : index
  %wlast = index.constant {W - 1} : index
  %nwin = index.min %n_kv, %wconst : index
  %last = index.sub %n_kv, %c1 : index
  %q_view = buffer.view %q[%zero] : buffer -> view<{H}x{D}xf32>
  %r_view = buffer.view %r[%zero] : buffer -> view<{H}x{R}xf32>
  %sk_view = buffer.view %sinks[%zero] : buffer -> view<{H}xf32>
  %g_view = buffer.view %gate[%zero] : buffer -> view<{H}x{D}xf32>
  %rb_view = buffer.view %rel_bias[%zero] : buffer -> view<{W}x{R}xf32>
  %k_view = buffer.view %kc[%zero] : buffer -> view<[%n_kv]x{HKV}x{D}xf32>
  %v_view = buffer.view %vc[%zero] : buffer -> view<[%n_kv]x{HKV}x{D}xf32>
  %o_view = buffer.view %out[%zero] : buffer -> view<{H}x{D}xf32>
  %rv = vector.load %r_view[%h, %c0] : view<{H}x{R}xf32> -> vector<{R}xf32>
  %sink = view.load %sk_view[%h] : view<{H}xf32> -> f32
  %s_bytes = index.constant {W * 4} : offset
  %s_mem = buffer.alloca<workgroup> align(16) %s_bytes : buffer
  %s_lds = buffer.view %s_mem[%zero] : buffer -> view<{W}xf32>
{chr(10).join(sc)}
  %m = kernel.workgroup.reduce<maxnumf> {mx} : f32
  %psum = scf.for %pc = [%c0 to %nwin step %c{T}] (%acc = %zero_f : f32) -> (f32) {{
    %pi_raw = index.add %pc, %lane : index
    %pok = index.cmp ult, %pi_raw, %nwin : index
    %pi_c = index.min %pi_raw, %wlast : index
    %pi = index.assume %pi_c [range(%pi_c, 0, {W - 1})] : index
    %sv = view.load %s_lds[%pi] : view<{W}xf32> -> f32
    %sh = scalar.subf %sv, %m : f32
    %ev = scalar.expf<afn> %sh : f32
    %e = scf.select %pok, %ev, %zero_f : f32
    scf.if %pok {{
      view.store %e, %s_lds[%pi] : f32, view<{W}xf32>
    }}
    %nacc = scalar.addf %acc, %e : f32
    scf.yield %nacc : f32
  }}
  %tot = kernel.workgroup.reduce<addf> %psum : f32
  %nm = scalar.subf %zero_f, %m : f32
  %esink = scalar.expf<afn> %nm : f32
  %den = scalar.addf %tot, %esink : f32
  %rden = scalar.divf %one, %den : f32
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %c128 = index.constant 128 : index
  %dim_raw = index.rem %lane, %c128 : index
  %dim = index.assume %dim_raw [range(%dim_raw, 0, {D - 1})] : index
  %par = index.div %lane, %c128 : index
  %c2 = index.constant 2 : index
  %osum = scf.for %kk = [%par to %nwin step %c2] (%acc = %zero_f : f32) -> (f32) {{
    %kd_c = index.min %kk, %wlast : index
    %kdd = index.assume %kd_c [range(%kd_c, 0, {W - 1})] : index
    %pv = view.load %s_lds[%kdd] : view<{W}xf32> -> f32
    %kdl = index.min %kdd, %last : index
    %jvr = index.sub %last, %kdl : index
    %jv = index.assume %jvr [range(%jvr, 0, 1048575), lt(%jvr, %n_kv)] : index
    %vv = view.load %v_view[%jv, %kvh, %dim] : view<[%n_kv]x{HKV}x{D}xf32> -> f32
    %nacc = scalar.fmaf %pv, %vv, %acc : f32
    scf.yield %nacc : f32
  }}
  %h_bytes = index.constant {D * 4} : offset
  %h_mem = buffer.alloca<workgroup> align(16) %h_bytes : buffer
  %h_lds = buffer.view %h_mem[%zero] : buffer -> view<{D}xf32>
  %odd = index.cmp eq, %par, %c1 : index
  scf.if %odd {{
    view.store %osum, %h_lds[%dim] : f32, view<{D}xf32>
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %even = index.cmp eq, %par, %c0 : index
  scf.if %even {{
    %other = view.load %h_lds[%dim] : view<{D}xf32> -> f32
    %both = scalar.addf %osum, %other : f32
    %attn = scalar.mulf %both, %rden : f32
    %gl = view.load %g_view[%h, %dim] : view<{H}x{D}xf32> -> f32
    %gs = scalar.logisticf<afn> %gl : f32
    %gated = scalar.mulf %attn, %gs : f32
    view.store %gated, %o_view[%h, %dim] : f32, view<{H}x{D}xf32>
  }}
  kernel.return
}}
"""


def decode_global(CH=256, H=24, HKV=8, D=128, R=16, E=1024, name=None):
    """The global layers' decode attention, split over the keys (any context length): two kernels, as vsplit.
    g_ink_attn_dec_glob_part (grid heads x chunks): chunk c's keys j = c CH + lane, s_j = q . k_j / sqrt(D) + bias(d) - sink with
    bias(d) = r . rel_bias[d] for d = n_kv - 1 - j < E and 0 past it (the fork pads rel_bias with a zero row at d = E); writes the
    chunk's max m_c, sum l_c = sum exp(s - m_c) and o_c = sum exp(s - m_c) v (unnormalised) to %part [heads][chunks][D + 2].
    g_ink_attn_dec_glob_comb (grid heads): M = max(0, max m_c) (the zero-logit sink), L = exp(-M) + sum l_c exp(m_c - M),
    out = (sum o_c exp(m_c - M) / L) * sigmoid(gate)."""
    T = 256
    assert CH == T and D == 128
    part = f"""
kernel.def export("g_ink_attn_dec_glob_part") @g_ink_attn_dec_glob_part(%n_kv: index) {{
  %c1 = index.constant 1 : index
  %heads = index.constant {H} : index
  %cch = index.constant {CH} : index
  %cm = index.constant {CH - 1} : index
  %nk = index.add %n_kv, %cm : index
  %chunks = index.div %nk, %cch : index
  %c{T} = index.constant {T} : index
  kernel.launch.config workgroups(%heads, %chunks, %c1) workgroup_size(%c{T}, %c1, %c1) : index
}} launch(%n_kv_arg: index, %q: buffer, %r: buffer, %sinks: buffer, %rel_bias: buffer, %kc: buffer, %vc: buffer, %part: buffer) {{
  %n_kv = index.assume %n_kv_arg [range(%n_kv_arg, 1, 1048576)] : index
  %h_id = kernel.workgroup.id<x> : index
  %h = index.assume %h_id [range(%h_id, 0, {H - 1})] : index
  %c_id = kernel.workgroup.id<y> : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, {T - 1})] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c16 = index.constant 16 : index
  %c128 = index.constant 128 : index
  %cD = index.constant {D} : index
  %cCH = index.constant {CH} : index
  %zero_f = scalar.constant 0.0 : f32
  %neg_inf = scalar.constant -3.0e38 : f32
  %kscale = scalar.constant {1.0 / D ** 0.5!r} : f32
  %gq = index.constant {H // HKV} : index
  %kvh_raw = index.div %h, %gq : index
  %kvh = index.assume %kvh_raw [range(%kvh_raw, 0, {HKV - 1})] : index
  %ext = index.constant {E} : index
  %elast = index.constant {E - 1} : index
  %last = index.sub %n_kv, %c1 : index
  %cmax = index.constant 4095 : index
  %cid_c = index.min %c_id, %cmax : index
  %cid = index.assume %cid_c [range(%cid_c, 0, 4095)] : index
  %base = index.mul %cid, %cCH : index
  %q_view = buffer.view %q[%zero] : buffer -> view<{H}x{D}xf32>
  %r_view = buffer.view %r[%zero] : buffer -> view<{H}x{R}xf32>
  %sk_view = buffer.view %sinks[%zero] : buffer -> view<{H}xf32>
  %rb_view = buffer.view %rel_bias[%zero] : buffer -> view<{E}x{R}xf32>
  %k_view = buffer.view %kc[%zero] : buffer -> view<[%n_kv]x{HKV}x{D}xf32>
  %v_view = buffer.view %vc[%zero] : buffer -> view<[%n_kv]x{HKV}x{D}xf32>
  %rv = vector.load %r_view[%h, %c0] : view<{H}x{R}xf32> -> vector<{R}xf32>
  %sink = view.load %sk_view[%h] : view<{H}xf32> -> f32
  %jraw = index.add %base, %lane : index
  %okj = index.cmp ult, %jraw, %n_kv : index
  %jc = index.min %jraw, %last : index
  %j = index.assume %jc [range(%jc, 0, 1048575), lt(%jc, %n_kv)] : index
  %kd = scf.for %ch = [%c0 to %cD step %c16] (%acc = %zero_f : f32) -> (f32) {{
    %kv = vector.load %k_view[%j, %kvh, %ch] : view<[%n_kv]x{HKV}x{D}xf32> -> vector<16xf32>
    %qv = vector.load %q_view[%h, %ch] : view<{H}x{D}xf32> -> vector<16xf32>
    %pp = vector.dotf<reassoc|contract> %kv, %qv, %acc : vector<16xf32>, vector<16xf32>, f32
    scf.yield %pp : f32
  }}
  %dist = index.sub %last, %j : index
  %near = index.cmp ult, %dist, %ext : index
  %dcl = index.min %dist, %elast : index
  %dd = index.assume %dcl [range(%dcl, 0, {E - 1})] : index
  %rb = vector.load %rb_view[%dd, %c0] : view<{E}x{R}xf32> -> vector<{R}xf32>
  %bnear = vector.dotf<reassoc|contract> %rb, %rv, %zero_f : vector<{R}xf32>, vector<{R}xf32>, f32
  %bias = scf.select %near, %bnear, %zero_f : f32
  %qk = scalar.mulf %kd, %kscale : f32
  %sb = scalar.addf %qk, %bias : f32
  %s_raw = scalar.subf %sb, %sink : f32
  %s = scf.select %okj, %s_raw, %neg_inf : f32
  %m = kernel.workgroup.reduce<maxnumf> %s : f32
  %sh = scalar.subf %s, %m : f32
  %ev = scalar.expf<afn> %sh : f32
  %e = scf.select %okj, %ev, %zero_f : f32
  %l = kernel.workgroup.reduce<addf> %e : f32
  %p_bytes = index.constant {CH * 4} : offset
  %p_mem = buffer.alloca<workgroup> align(16) %p_bytes : buffer
  %p_lds = buffer.view %p_mem[%zero] : buffer -> view<{CH}xf32>
  view.store %e, %p_lds[%lane] : f32, view<{CH}xf32>
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %dim_raw = index.rem %lane, %c128 : index
  %dim = index.assume %dim_raw [range(%dim_raw, 0, {D - 1})] : index
  %par = index.div %lane, %c128 : index
  %osum = scf.for %kk = [%par to %cCH step %c2] (%acc = %zero_f : f32) -> (f32) {{
    %kq = index.assume %kk [range(%kk, 0, {CH - 1})] : index
    %pv = view.load %p_lds[%kq] : view<{CH}xf32> -> f32
    %jv0 = index.add %base, %kq : index
    %jv1 = index.min %jv0, %last : index
    %jv = index.assume %jv1 [range(%jv1, 0, 1048575), lt(%jv1, %n_kv)] : index
    %vv = view.load %v_view[%jv, %kvh, %dim] : view<[%n_kv]x{HKV}x{D}xf32> -> f32
    %nacc = scalar.fmaf %pv, %vv, %acc : f32
    scf.yield %nacc : f32
  }}
  %h_bytes = index.constant {D * 4} : offset
  %h_mem = buffer.alloca<workgroup> align(16) %h_bytes : buffer
  %h_lds = buffer.view %h_mem[%zero] : buffer -> view<{D}xf32>
  %odd = index.cmp eq, %par, %c1 : index
  scf.if %odd {{
    view.store %osum, %h_lds[%dim] : f32, view<{D}xf32>
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %pt_view = buffer.view %part[%zero] : buffer -> view<{H}x4096x{D + 2}xf32>
  %even = index.cmp eq, %par, %c0 : index
  scf.if %even {{
    %other = view.load %h_lds[%dim] : view<{D}xf32> -> f32
    %both = scalar.addf %osum, %other : f32
    view.store %both, %pt_view[%h, %cid, %dim] : f32, view<{H}x4096x{D + 2}xf32>
  }}
  %first = index.cmp eq, %lane, %c0 : index
  scf.if %first {{
    %iM = index.constant {D} : index
    %iL = index.constant {D + 1} : index
    view.store %m, %pt_view[%h, %cid, %iM] : f32, view<{H}x4096x{D + 2}xf32>
    view.store %l, %pt_view[%h, %cid, %iL] : f32, view<{H}x4096x{D + 2}xf32>
  }}
  kernel.return
}}
"""
    comb = f"""
kernel.def export("g_ink_attn_dec_glob_comb") @g_ink_attn_dec_glob_comb(%n_kv: index) {{
  %c1 = index.constant 1 : index
  %heads = index.constant {H} : index
  %cD = index.constant {D} : index
  kernel.launch.config workgroups(%heads, %c1, %c1) workgroup_size(%cD, %c1, %c1) : index
}} launch(%n_kv_arg: index, %part: buffer, %gate: buffer, %out: buffer) {{
  %n_kv = index.assume %n_kv_arg [range(%n_kv_arg, 1, 1048576)] : index
  %h_id = kernel.workgroup.id<x> : index
  %h = index.assume %h_id [range(%h_id, 0, {H - 1})] : index
  %dim_id = kernel.workitem.id<x> : index
  %dim = index.assume %dim_id [range(%dim_id, 0, {D - 1})] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %cCH = index.constant {CH} : index
  %cm = index.constant {CH - 1} : index
  %nk = index.add %n_kv, %cm : index
  %chunks = index.div %nk, %cCH : index
  %iM = index.constant {D} : index
  %iL = index.constant {D + 1} : index
  %zero_f = scalar.constant 0.0 : f32
  %one = scalar.constant 1.0 : f32
  %pt_view = buffer.view %part[%zero] : buffer -> view<{H}x4096x{D + 2}xf32>
  %g_view = buffer.view %gate[%zero] : buffer -> view<{H}x{D}xf32>
  %o_view = buffer.view %out[%zero] : buffer -> view<{H}x{D}xf32>
  %mx = scf.for %c = [%c0 to %chunks step %c1] (%acc = %zero_f : f32) -> (f32) {{
    %cc = index.assume %c [range(%c, 0, 4095)] : index
    %mc = view.load %pt_view[%h, %cc, %iM] : view<{H}x4096x{D + 2}xf32> -> f32
    %nm = scalar.maxnumf %acc, %mc : f32
    scf.yield %nm : f32
  }}
  %nmx = scalar.subf %zero_f, %mx : f32
  %esink = scalar.expf<afn> %nmx : f32
  %osum, %lsum = scf.for %c = [%c0 to %chunks step %c1] (%ao = %zero_f : f32, %al = %esink : f32) -> (f32, f32) {{
    %cc = index.assume %c [range(%c, 0, 4095)] : index
    %mc = view.load %pt_view[%h, %cc, %iM] : view<{H}x4096x{D + 2}xf32> -> f32
    %lc = view.load %pt_view[%h, %cc, %iL] : view<{H}x4096x{D + 2}xf32> -> f32
    %oc = view.load %pt_view[%h, %cc, %dim] : view<{H}x4096x{D + 2}xf32> -> f32
    %dm = scalar.subf %mc, %mx : f32
    %w = scalar.expf<afn> %dm : f32
    %no = scalar.fmaf %oc, %w, %ao : f32
    %nl = scalar.fmaf %lc, %w, %al : f32
    scf.yield %no, %nl : f32, f32
  }}
  %attn = scalar.divf %osum, %lsum : f32
  %gl = view.load %g_view[%h, %dim] : view<{H}x{D}xf32> -> f32
  %gs = scalar.logisticf<afn> %gl : f32
  %gated = scalar.mulf %attn, %gs : f32
  view.store %gated, %o_view[%h, %dim] : f32, view<{H}x{D}xf32>
  kernel.return
}}
"""
    return part, comb
