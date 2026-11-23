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


def _kload(out, j, HKV, D, kv8):
    """Loom lines: %out = 16 K values (as f32) at [j, kvh, ch]: f32 directly, or int8 codes widened (the scale is applied once per key,
    after the dot: q . (c s) = s (q . c))"""
    if not kv8:
        return f"    %{out} = vector.load %k_view[{j}, %kvh, %ch] : view<[%n_kv]x{HKV}x{D}xf32> -> vector<16xf32>"
    return (f"    %{out}_c = vector.load %k_view[{j}, %kvh, %ch] : view<[%n_kv]x{HKV}x{D}xi8> -> vector<16xi8>\n"
            f"    %{out} = vector.sitofp %{out}_c : vector<16xi8> to vector<16xf32>")


def _kscale(kd, j, HKV, kv8):
    """KV8: %{kd}s = the dot times the key's scale"""
    if not kv8:
        return ""
    return (f"\n  %{kd}_sc = view.load %ks_view[{j}, %kvh] : view<[%n_kv]x{HKV}xf32> -> f32"
            f"\n  %{kd}s = scalar.mulf %{kd}, %{kd}_sc : f32")


def _vload(HKV, D, kv8):
    """Loom lines: %vv = V[jv, kvh, dim] as f32, and with KV8 %pvs = the key's weight times its V scale"""
    if not kv8:
        return f"    %vv = view.load %v_view[%jv, %kvh, %dim] : view<[%n_kv]x{HKV}x{D}xf32> -> f32"
    return (f"    %vc8 = view.load %v_view[%jv, %kvh, %dim] : view<[%n_kv]x{HKV}x{D}xi8> -> i8\n"
            f"    %vci = scalar.extsi %vc8 : i8 to i32\n"
            f"    %vv = scalar.sitofp %vci : i32 to f32\n"
            f"    %vsc = view.load %vs_view[%jv, %kvh] : view<[%n_kv]x{HKV}xf32> -> f32\n"
            f"    %pvs = scalar.mulf %pv, %vsc : f32")


def _scale_views(HKV, kv8):
    if not kv8:
        return ""
    return (f"\n  %ks_view = buffer.view %ksc[%zero] : buffer -> view<[%n_kv]x{HKV}xf32>"
            f"\n  %vs_view = buffer.view %vsc[%zero] : buffer -> view<[%n_kv]x{HKV}xf32>")


def decode_swa(W=1024, H=24, HKV=8, D=128, R=16, name=None, kv8=False):
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
    name = name or f"g_ink_attn_dec_swa{W}" + ("_kv8" if kv8 else "")
    KT = "i8" if kv8 else "f32"                         # KV8 (as the Qwen kit's): int8 codes + one f32 scale per (position, KV head)
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
{_kload(f"kv{c}", f"%j{c}", HKV, D, kv8)}
    %qv{c} = vector.load %q_view[%h, %ch] : view<{H}x{D}xf32> -> vector<16xf32>
    %pp{c} = vector.dotf<reassoc|contract> %kv{c}, %qv{c}, %acc : vector<16xf32>, vector<16xf32>, f32
    scf.yield %pp{c} : f32
  }}{_kscale(f"kd{c}", f"%j{c}", HKV, kv8)}
  %rb{c} = vector.load %rb_view[%d{c}, %c0] : view<{W}x{R}xf32> -> vector<{R}xf32>
  %bias{c} = vector.dotf<reassoc|contract> %rb{c}, %rv, %zero_f : vector<{R}xf32>, vector<{R}xf32>, f32
  %qk{c} = scalar.mulf %kd{c}{"s" if kv8 else ""}, %kscale : f32
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
}} launch(%n_kv_arg: index, %q: buffer, %r: buffer, %sinks: buffer, %gate: buffer, %rel_bias: buffer, %kc: buffer, %vc: buffer{", %ksc: buffer, %vsc: buffer" if kv8 else ""}, %out: buffer) {{
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
  %k_view = buffer.view %kc[%zero] : buffer -> view<[%n_kv]x{HKV}x{D}x{"i8" if kv8 else "f32"}>
  %v_view = buffer.view %vc[%zero] : buffer -> view<[%n_kv]x{HKV}x{D}x{"i8" if kv8 else "f32"}>{_scale_views(HKV, kv8)}
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
{_vload(HKV, D, kv8)}
    %nacc = scalar.fmaf %pv{"s" if kv8 else ""}, %vv, %acc : f32
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


def decode_global(CH=256, H=24, HKV=8, D=128, R=16, E=1024, name=None, kv8=False):
    """The global layers' decode attention, split over the keys (any context length): two kernels, as vsplit.
    g_ink_attn_dec_glob_part (grid heads x chunks): chunk c's keys j = c CH + lane, s_j = q . k_j / sqrt(D) + bias(d) - sink with
    bias(d) = r . rel_bias[d] for d = n_kv - 1 - j < E and 0 past it (the fork pads rel_bias with a zero row at d = E); writes the
    chunk's max m_c, sum l_c = sum exp(s - m_c) and o_c = sum exp(s - m_c) v (unnormalised) to %part [heads][chunks][D + 2].
    g_ink_attn_dec_glob_comb (grid heads): M = max(0, max m_c) (the zero-logit sink), L = exp(-M) + sum l_c exp(m_c - M),
    out = (sum o_c exp(m_c - M) / L) * sigmoid(gate)."""
    T = 256
    assert CH == T and D == 128
    pname = "g_ink_attn_dec_glob_part" + ("_kv8" if kv8 else "")
    part = f"""
kernel.def export("{pname}") @{pname}(%n_kv: index) {{
  %c1 = index.constant 1 : index
  %heads = index.constant {H} : index
  %cch = index.constant {CH} : index
  %cm = index.constant {CH - 1} : index
  %nk = index.add %n_kv, %cm : index
  %chunks = index.div %nk, %cch : index
  %c{T} = index.constant {T} : index
  kernel.launch.config workgroups(%heads, %chunks, %c1) workgroup_size(%c{T}, %c1, %c1) : index
}} launch(%n_kv_arg: index, %q: buffer, %r: buffer, %sinks: buffer, %rel_bias: buffer, %kc: buffer, %vc: buffer{", %ksc: buffer, %vsc: buffer" if kv8 else ""}, %part: buffer) {{
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
  %k_view = buffer.view %kc[%zero] : buffer -> view<[%n_kv]x{HKV}x{D}x{"i8" if kv8 else "f32"}>
  %v_view = buffer.view %vc[%zero] : buffer -> view<[%n_kv]x{HKV}x{D}x{"i8" if kv8 else "f32"}>{_scale_views(HKV, kv8)}
  %rv = vector.load %r_view[%h, %c0] : view<{H}x{R}xf32> -> vector<{R}xf32>
  %sink = view.load %sk_view[%h] : view<{H}xf32> -> f32
  %jraw = index.add %base, %lane : index
  %okj = index.cmp ult, %jraw, %n_kv : index
  %jc = index.min %jraw, %last : index
  %j = index.assume %jc [range(%jc, 0, 1048575), lt(%jc, %n_kv)] : index
  %kd = scf.for %ch = [%c0 to %cD step %c16] (%acc = %zero_f : f32) -> (f32) {{
{_kload("kv", "%j", HKV, D, kv8)}
    %qv = vector.load %q_view[%h, %ch] : view<{H}x{D}xf32> -> vector<16xf32>
    %pp = vector.dotf<reassoc|contract> %kv, %qv, %acc : vector<16xf32>, vector<16xf32>, f32
    scf.yield %pp : f32
  }}{_kscale("kd", "%j", HKV, kv8)}
  %dist = index.sub %last, %j : index
  %near = index.cmp ult, %dist, %ext : index
  %dcl = index.min %dist, %elast : index
  %dd = index.assume %dcl [range(%dcl, 0, {E - 1})] : index
  %rb = vector.load %rb_view[%dd, %c0] : view<{E}x{R}xf32> -> vector<{R}xf32>
  %bnear = vector.dotf<reassoc|contract> %rb, %rv, %zero_f : vector<{R}xf32>, vector<{R}xf32>, f32
  %bias = scf.select %near, %bnear, %zero_f : f32
  %qk = scalar.mulf %kd{"s" if kv8 else ""}, %kscale : f32
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
{_vload(HKV, D, kv8)}
    %nacc = scalar.fmaf %pv{"s" if kv8 else ""}, %vv, %acc : f32
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


def decode_swa_span(T=8, W=1024, H=24, HKV=8, D=128, R=16, name=None, kv8=False):
    """g_ink_attn_span_swa: the sliding-window attention of a verify span (HRX-like E298): T query tokens at positions n_kv - T .. n_kv - 1
    (their K / V already appended), each over its own window of W keys, with each key and value read once for the span instead of
    once per token (decode_swa per token reads the window T times). One workgroup per query head, 256 lanes:
      1. key slots i = 0 .. W + T - 2 cover keys j = n_kv - (W + T - 1) + i (slots before key 0 are empty); lane i loads K_j once and
         scores it against every token t: s[t][i] = (q_t . k_j) / sqrt(D) + r_t . rel_bias[d] - sink, d = (n_kv - T + t) - j, the
         slot kept only if 0 <= d < W; scores in LDS [T][slots];
      2. per token: m_t = max(0, max_i s[t][i]), p = exp(s - m_t), den_t = exp(-m_t) + sum p (the zero-logit sink);
      3. lanes 0-127 / 128-255 take even / odd slots for one output dim, load V_j once and add p[t][i] v_j for every t;
      4. out[t] = (sum / den_t) sigmoid(gate[t]).
    Inputs q / gate [T][H][D], r [T][H][R], sinks [H], rel_bias [W][R], f32 K / V cache [n_kv][HKV][D]; output [T][H][D].
    kv8: int8 K / V codes with one f32 scale per (position, KV head) (%ksc, %vsc), as decode_swa(kv8): the K scale multiplies each
    token's dot once per key, the V scale the widened value once per key."""
    L = 256
    assert D == 128 and R == 16 and T >= 1
    name = name or f"g_ink_attn_span_swa{W}_t{T}" + ("_kv8" if kv8 else "")
    KT = "i8" if kv8 else "f32"
    NS = W + T - 1                                   # key slots a span needs
    SP = -(-NS // L) * L                             # padded to whole lane rounds
    KPL = SP // L
    CH = D // 16
    sc = []
    for c in range(KPL):
        dots = [f"""  %i{c} = index.add %lane, %ic{c} : index
  %u{c}_raw = index.sub %ns1, %i{c} : index
  %slot{c} = index.cmp ult, %i{c}, %nsc : index
  %u{c} = index.min %u{c}_raw, %ns1 : index
  %haskey{c} = index.cmp ule, %u{c}, %last : index
  %uk{c} = index.min %u{c}, %last : index
  %jr{c} = index.sub %last, %uk{c} : index
  %j{c} = index.assume %jr{c} [range(%jr{c}, 0, 1048575), lt(%jr{c}, %n_kv)] : index"""]
        for ch in range(CH):
            dots.append(f"""  %kc{c}_{ch} = index.constant {16 * ch} : index
""" + (f"""  %kv{c}_{ch}_c = vector.load %k_view[%j{c}, %kvh, %kc{c}_{ch}] : view<[%n_kv]x{HKV}x{D}xi8> -> vector<16xi8>
  %kv{c}_{ch} = vector.sitofp %kv{c}_{ch}_c : vector<16xi8> to vector<16xf32>""" if kv8 else
  f"  %kv{c}_{ch} = vector.load %k_view[%j{c}, %kvh, %kc{c}_{ch}] : view<[%n_kv]x{HKV}x{D}xf32> -> vector<16xf32>"))
            for t in range(T):
                prev = "%zero_f" if ch == 0 else f"%qk{c}_{t}_{ch - 1}"
                dots.append(f"  %qk{c}_{t}_{ch} = vector.dotf<reassoc|contract> %kv{c}_{ch}, %qv{t}_{ch}, {prev} : vector<16xf32>, vector<16xf32>, f32")
        if kv8:
            dots.append(f"  %ks{c} = view.load %ks_view[%j{c}, %kvh] : view<[%n_kv]x{HKV}xf32> -> f32")
            for t in range(T):
                dots.append(f"  %qs{c}_{t} = scalar.mulf %qk{c}_{t}_{CH - 1}, %ks{c} : f32")
        for t in range(T):
            dots.append(f"""  %tb{c}_{t} = index.constant {T - 1 - t} : index
  %ge{c}_{t} = index.cmp ule, %tb{c}_{t}, %u{c} : index
  %dr{c}_{t} = index.sub %u{c}, %tb{c}_{t} : index
  %dw{c}_{t} = index.min %dr{c}_{t}, %wlast : index
  %d{c}_{t} = index.assume %dw{c}_{t} [range(%dw{c}_{t}, 0, {W - 1})] : index
  %inw{c}_{t} = index.cmp ult, %dr{c}_{t}, %wconst : index
  %rb{c}_{t} = vector.load %rb_view[%d{c}_{t}, %c0] : view<{W}x{R}xf32> -> vector<{R}xf32>
  %bias{c}_{t} = vector.dotf<reassoc|contract> %rb{c}_{t}, %rv{t}, %zero_f : vector<{R}xf32>, vector<{R}xf32>, f32
  %sq{c}_{t} = scalar.mulf {f"%qs{c}_{t}" if kv8 else f"%qk{c}_{t}_{CH - 1}"}, %kscale : f32
  %sb{c}_{t} = scalar.addf %sq{c}_{t}, %bias{c}_{t} : f32
  %sr{c}_{t} = scalar.subf %sb{c}_{t}, %sink : f32
  %s1{c}_{t} = scf.select %ge{c}_{t}, %sr{c}_{t}, %neg_inf : f32
  %s2{c}_{t} = scf.select %inw{c}_{t}, %s1{c}_{t}, %neg_inf : f32
  %s3{c}_{t} = scf.select %haskey{c}, %s2{c}_{t}, %neg_inf : f32
  %s{c}_{t} = scf.select %slot{c}, %s3{c}_{t}, %neg_inf : f32
  %iw{c}_{t} = index.min %i{c}, %splast : index
  view.store %s{c}_{t}, %s_lds[%t{t}, %iw{c}_{t}] : f32, view<{T}x{SP}xf32>""")
        sc.append("\n".join(dots))
    qloads, mx, den = [], [], []
    for t in range(T):
        qloads.append(f"  %t{t} = index.constant {t} : index\n"
                      f"  %rv{t} = vector.load %r_view[%t{t}, %h, %c0] : view<{T}x{H}x{R}xf32> -> vector<{R}xf32>")
        for ch in range(CH):
            qloads.append(f"  %qc{t}_{ch} = index.constant {16 * ch} : index\n"
                          f"  %qv{t}_{ch} = vector.load %q_view[%t{t}, %h, %qc{t}_{ch}] : view<{T}x{H}x{D}xf32> -> vector<16xf32>")
        acc = "%zero_f"
        for c in range(KPL):
            mx.append(f"  %lm{c}_{t} = scalar.maxnumf {acc}, %s{c}_{t} : f32")
            acc = f"%lm{c}_{t}"
        mx.append(f"  %m{t} = kernel.workgroup.reduce<maxnumf> {acc} : f32")
    exps = []
    for t in range(T):
        exps.append(f"""  %ps{t} = scf.for %pc = [%c0 to %spc step %c{L}] (%acc = %zero_f : f32) -> (f32) {{
    %pi_raw = index.add %pc, %lane : index
    %pi = index.assume %pi_raw [range(%pi_raw, 0, {SP - 1})] : index
    %sv = view.load %s_lds[%t{t}, %pi] : view<{T}x{SP}xf32> -> f32
    %sh = scalar.subf %sv, %m{t} : f32
    %e = scalar.expf<afn> %sh : f32
    view.store %e, %s_lds[%t{t}, %pi] : f32, view<{T}x{SP}xf32>
    %nacc = scalar.addf %acc, %e : f32
    scf.yield %nacc : f32
  }}
  %tot{t} = kernel.workgroup.reduce<addf> %ps{t} : f32
  %nm{t} = scalar.subf %zero_f, %m{t} : f32
  %es{t} = scalar.expf<afn> %nm{t} : f32
  %den{t} = scalar.addf %tot{t}, %es{t} : f32
  %rden{t} = scalar.divf %one, %den{t} : f32""")
    accs = ", ".join(f"%o{t}" for t in range(T))
    inits = ", ".join(f"%a{t} = %zero_f : f32" for t in range(T))
    pv = [f"    %pv{t} = view.load %s_lds[%t{t}, %kdd] : view<{T}x{SP}xf32> -> f32\n"
          f"    %n{t} = scalar.fmaf %pv{t}, %vv, %a{t} : f32" for t in range(T)]
    fin = []
    for t in range(T):
        fin.append(f"""    %oth{t} = view.load %h_lds[%t{t}, %dim] : view<{T}x{D}xf32> -> f32
    %both{t} = scalar.addf %o{t}, %oth{t} : f32
    %att{t} = scalar.mulf %both{t}, %rden{t} : f32
    %gl{t} = view.load %g_view[%t{t}, %h, %dim] : view<{T}x{H}x{D}xf32> -> f32
    %gs{t} = scalar.logisticf<afn> %gl{t} : f32
    %gt{t} = scalar.mulf %att{t}, %gs{t} : f32
    view.store %gt{t}, %o_view[%t{t}, %h, %dim] : f32, view<{T}x{H}x{D}xf32>""")
    stores = "\n".join(f"    view.store %o{t}, %h_lds[%t{t}, %dim] : f32, view<{T}x{D}xf32>" for t in range(T))
    return f"""
kernel.def export("{name}") @{name}(%n_kv: index) {{
  %c1 = index.constant 1 : index
  %heads = index.constant {H} : index
  %cl = index.constant {L} : index
  kernel.launch.config workgroups(%heads, %c1, %c1) workgroup_size(%cl, %c1, %c1) : index
}} launch(%n_kv_arg: index, %q: buffer, %r: buffer, %sinks: buffer, %gate: buffer, %rel_bias: buffer, %kc: buffer, %vc: buffer{", %ksc: buffer, %vsc: buffer" if kv8 else ""}, %out: buffer) {{
  %n_kv = index.assume %n_kv_arg [range(%n_kv_arg, {T}, 1048576)] : index
  %h_id = kernel.workgroup.id<x> : index
  %h = index.assume %h_id [range(%h_id, 0, {H - 1})] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, {L - 1})] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c{L} = index.constant {L} : index
  %zero_f = scalar.constant 0.0 : f32
  %one = scalar.constant 1.0 : f32
  %neg_inf = scalar.constant -3.0e38 : f32
  %kscale = scalar.constant {1.0 / D ** 0.5!r} : f32
  %gq = index.constant {H // HKV} : index
  %kvh_raw = index.div %h, %gq : index
  %kvh = index.assume %kvh_raw [range(%kvh_raw, 0, {HKV - 1})] : index
  %wconst = index.constant {W} : index
  %wlast = index.constant {W - 1} : index
  %nsc = index.constant {NS} : index
  %ns1 = index.constant {NS - 1} : index
  %spc = index.constant {SP} : index
  %splast = index.constant {SP - 1} : index
  %last = index.sub %n_kv, %c1 : index
{chr(10).join(f"  %ic{c} = index.constant {c * L} : index" for c in range(KPL))}
  %q_view = buffer.view %q[%zero] : buffer -> view<{T}x{H}x{D}xf32>
  %r_view = buffer.view %r[%zero] : buffer -> view<{T}x{H}x{R}xf32>
  %sk_view = buffer.view %sinks[%zero] : buffer -> view<{H}xf32>
  %g_view = buffer.view %gate[%zero] : buffer -> view<{T}x{H}x{D}xf32>
  %rb_view = buffer.view %rel_bias[%zero] : buffer -> view<{W}x{R}xf32>
  %k_view = buffer.view %kc[%zero] : buffer -> view<[%n_kv]x{HKV}x{D}x{KT}>
  %v_view = buffer.view %vc[%zero] : buffer -> view<[%n_kv]x{HKV}x{D}x{KT}>{_scale_views(HKV, kv8)}
  %o_view = buffer.view %out[%zero] : buffer -> view<{T}x{H}x{D}xf32>
  %sink = view.load %sk_view[%h] : view<{H}xf32> -> f32
  %s_bytes = index.constant {T * SP * 4} : offset
  %s_mem = buffer.alloca<workgroup> align(16) %s_bytes : buffer
  %s_lds = buffer.view %s_mem[%zero] : buffer -> view<{T}x{SP}xf32>
{chr(10).join(qloads)}
{chr(10).join(sc)}
{chr(10).join(mx)}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
{chr(10).join(exps)}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %c128 = index.constant 128 : index
  %dim_raw = index.rem %lane, %c128 : index
  %dim = index.assume %dim_raw [range(%dim_raw, 0, {D - 1})] : index
  %par = index.div %lane, %c128 : index
  {accs} = scf.for %kk = [%par to %nsc step %c2] ({inits}) -> ({", ".join(["f32"] * T)}) {{
    %kd_c = index.min %kk, %ns1 : index
    %kdd = index.assume %kd_c [range(%kd_c, 0, {NS - 1})] : index
    %uv_raw = index.sub %ns1, %kdd : index
    %uv = index.min %uv_raw, %last : index
    %jvr = index.sub %last, %uv : index
    %jv = index.assume %jvr [range(%jvr, 0, 1048575), lt(%jvr, %n_kv)] : index
{f"""    %vc8 = view.load %v_view[%jv, %kvh, %dim] : view<[%n_kv]x{HKV}x{D}xi8> -> i8
    %vci = scalar.extsi %vc8 : i8 to i32
    %vf = scalar.sitofp %vci : i32 to f32
    %vsc = view.load %vs_view[%jv, %kvh] : view<[%n_kv]x{HKV}xf32> -> f32
    %vv = scalar.mulf %vf, %vsc : f32""" if kv8 else f"    %vv = view.load %v_view[%jv, %kvh, %dim] : view<[%n_kv]x{HKV}x{D}xf32> -> f32"}
{chr(10).join(pv)}
    scf.yield {", ".join(f"%n{t}" for t in range(T))} : {", ".join(["f32"] * T)}
  }}
  %h_bytes = index.constant {T * D * 4} : offset
  %h_mem = buffer.alloca<workgroup> align(16) %h_bytes : buffer
  %h_lds = buffer.view %h_mem[%zero] : buffer -> view<{T}x{D}xf32>
  %odd = index.cmp eq, %par, %c1 : index
  scf.if %odd {{
{stores}
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %even = index.cmp eq, %par, %c0 : index
  scf.if %even {{
{chr(10).join(fin)}
  }}
  kernel.return
}}
"""


def decode_global_span(T=8, CH=256, H=24, HKV=8, D=128, R=16, E=1024, kv8=False):
    """The global layers' attention for a verify span (HRX-like E298): decode_global's split-K pair with each K / V row read once for
    all T tokens (positions n_kv - T .. n_kv - 1, their K / V appended). Token t sees keys j <= pos_t = n_kv - T + t.
    g_ink_attn_span_glob_part_t{T} (grid heads x chunks): lane = key j = c CH + lane; per token s_t = q_t . k_j / sqrt(D) + bias(d_t) - sink,
    d_t = pos_t - j (bias 0 past E, key masked past pos_t); per token the chunk's m, l and o = sum exp(s - m) v to
    %part [T][heads][4096 chunks][D + 2].  g_ink_attn_span_glob_comb_t{T} (grid heads x T): decode_global's combine per (head, token)."""
    L = 256
    assert CH == L and D == 128
    sfx = f"_t{T}" + ("_kv8" if kv8 else "")
    KT = "i8" if kv8 else "f32"
    pname, cname = f"g_ink_attn_span_glob_part{sfx}", f"g_ink_attn_span_glob_comb_t{T}"
    q, kd = [], []
    for t in range(T):
        q.append(f"  %t{t} = index.constant {t} : index\n"
                 f"  %rv{t} = vector.load %r_view[%t{t}, %h, %c0] : view<{T}x{H}x{R}xf32> -> vector<{R}xf32>")
        for ch in range(D // 16):
            q.append(f"  %qc{t}_{ch} = index.constant {16 * ch} : index\n"
                     f"  %qv{t}_{ch} = vector.load %q_view[%t{t}, %h, %qc{t}_{ch}] : view<{T}x{H}x{D}xf32> -> vector<16xf32>")
    for ch in range(D // 16):
        kd.append(f"  %kcc{ch} = index.constant {16 * ch} : index")
        if kv8:
            kd.append(f"  %kv{ch}_c = vector.load %k_view[%j, %kvh, %kcc{ch}] : view<[%n_kv]x{HKV}x{D}xi8> -> vector<16xi8>\n"
                      f"  %kv{ch} = vector.sitofp %kv{ch}_c : vector<16xi8> to vector<16xf32>")
        else:
            kd.append(f"  %kv{ch} = vector.load %k_view[%j, %kvh, %kcc{ch}] : view<[%n_kv]x{HKV}x{D}xf32> -> vector<16xf32>")
        for t in range(T):
            prev = "%zero_f" if ch == 0 else f"%qk{t}_{ch - 1}"
            kd.append(f"  %qk{t}_{ch} = vector.dotf<reassoc|contract> %kv{ch}, %qv{t}_{ch}, {prev} : vector<16xf32>, vector<16xf32>, f32")
    if kv8:
        kd.append("  %ksj = view.load %ks_view[%j, %kvh] : view<[%n_kv]x{HKV}xf32> -> f32".replace("{HKV}", str(HKV)))
    per = []
    for t in range(T):
        dot = f"%qs{t}" if kv8 else f"%qk{t}_{D // 16 - 1}"
        per.append((f"  %qs{t} = scalar.mulf %qk{t}_{D // 16 - 1}, %ksj : f32\n" if kv8 else "") + f"""  %tb{t} = index.constant {T - 1 - t} : index
  %pos{t} = index.sub %last, %tb{t} : index
  %vis{t} = index.cmp ule, %jraw, %pos{t} : index
  %ok{t} = index.cmp ult, %jraw, %n_kv : index
  %djc{t} = index.min %j, %pos{t} : index
  %dist{t} = index.sub %pos{t}, %djc{t} : index
  %near{t} = index.cmp ult, %dist{t}, %ext : index
  %dcl{t} = index.min %dist{t}, %elast : index
  %dd{t} = index.assume %dcl{t} [range(%dcl{t}, 0, {E - 1})] : index
  %rb{t} = vector.load %rb_view[%dd{t}, %c0] : view<{E}x{R}xf32> -> vector<{R}xf32>
  %bn{t} = vector.dotf<reassoc|contract> %rb{t}, %rv{t}, %zero_f : vector<{R}xf32>, vector<{R}xf32>, f32
  %bias{t} = scf.select %near{t}, %bn{t}, %zero_f : f32
  %sq{t} = scalar.mulf {dot}, %kscale : f32
  %sb{t} = scalar.addf %sq{t}, %bias{t} : f32
  %sr{t} = scalar.subf %sb{t}, %sink : f32
  %sv{t} = scf.select %vis{t}, %sr{t}, %neg_inf : f32
  %s{t} = scf.select %ok{t}, %sv{t}, %neg_inf : f32
  %m{t} = kernel.workgroup.reduce<maxnumf> %s{t} : f32
  %sh{t} = scalar.subf %s{t}, %m{t} : f32
  %ev{t} = scalar.expf<afn> %sh{t} : f32
  %e1{t} = scf.select %vis{t}, %ev{t}, %zero_f : f32
  %e{t} = scf.select %ok{t}, %e1{t}, %zero_f : f32
  %l{t} = kernel.workgroup.reduce<addf> %e{t} : f32
  view.store %e{t}, %p_lds[%t{t}, %lane] : f32, view<{T}x{CH}xf32>""")
    accs = ", ".join(f"%o{t}" for t in range(T))
    inits = ", ".join(f"%a{t} = %zero_f : f32" for t in range(T))
    pv = "\n".join(f"    %pv{t} = view.load %p_lds[%t{t}, %kq] : view<{T}x{CH}xf32> -> f32\n    %n{t} = scalar.fmaf %pv{t}, %vv, %a{t} : f32"
                   for t in range(T))
    vload = (f"""    %vc8 = view.load %v_view[%jv, %kvh, %dim] : view<[%n_kv]x{HKV}x{D}xi8> -> i8
    %vci = scalar.extsi %vc8 : i8 to i32
    %vf = scalar.sitofp %vci : i32 to f32
    %vsc = view.load %vs_view[%jv, %kvh] : view<[%n_kv]x{HKV}xf32> -> f32
    %vv = scalar.mulf %vf, %vsc : f32""" if kv8 else
             f"    %vv = view.load %v_view[%jv, %kvh, %dim] : view<[%n_kv]x{HKV}x{D}xf32> -> f32")
    stores = "\n".join(f"    view.store %o{t}, %h_lds[%t{t}, %dim] : f32, view<{T}x{D}xf32>" for t in range(T))
    fin = "\n".join(f"""    %oth{t} = view.load %h_lds[%t{t}, %dim] : view<{T}x{D}xf32> -> f32
    %both{t} = scalar.addf %o{t}, %oth{t} : f32
    view.store %both{t}, %pt_view[%t{t}, %h, %cid, %dim] : f32, view<{T}x{H}x4096x{D + 2}xf32>""" for t in range(T))
    ml = "\n".join(f"""    view.store %m{t}, %pt_view[%t{t}, %h, %cid, %iM] : f32, view<{T}x{H}x4096x{D + 2}xf32>
    view.store %l{t}, %pt_view[%t{t}, %h, %cid, %iL] : f32, view<{T}x{H}x4096x{D + 2}xf32>""" for t in range(T))
    part = f"""
kernel.def export("{pname}") @{pname}(%n_kv: index) {{
  %c1 = index.constant 1 : index
  %heads = index.constant {H} : index
  %cch = index.constant {CH} : index
  %cm = index.constant {CH - 1} : index
  %nk = index.add %n_kv, %cm : index
  %chunks = index.div %nk, %cch : index
  %cl = index.constant {L} : index
  kernel.launch.config workgroups(%heads, %chunks, %c1) workgroup_size(%cl, %c1, %c1) : index
}} launch(%n_kv_arg: index, %q: buffer, %r: buffer, %sinks: buffer, %rel_bias: buffer, %kc: buffer, %vc: buffer{", %ksc: buffer, %vsc: buffer" if kv8 else ""}, %part: buffer) {{
  %n_kv = index.assume %n_kv_arg [range(%n_kv_arg, {T}, 1048576)] : index
  %h_id = kernel.workgroup.id<x> : index
  %h = index.assume %h_id [range(%h_id, 0, {H - 1})] : index
  %c_id = kernel.workgroup.id<y> : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, {L - 1})] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c128 = index.constant 128 : index
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
  %q_view = buffer.view %q[%zero] : buffer -> view<{T}x{H}x{D}xf32>
  %r_view = buffer.view %r[%zero] : buffer -> view<{T}x{H}x{R}xf32>
  %sk_view = buffer.view %sinks[%zero] : buffer -> view<{H}xf32>
  %rb_view = buffer.view %rel_bias[%zero] : buffer -> view<{E}x{R}xf32>
  %k_view = buffer.view %kc[%zero] : buffer -> view<[%n_kv]x{HKV}x{D}x{KT}>
  %v_view = buffer.view %vc[%zero] : buffer -> view<[%n_kv]x{HKV}x{D}x{KT}>{_scale_views(HKV, kv8)}
  %sink = view.load %sk_view[%h] : view<{H}xf32> -> f32
  %p_bytes = index.constant {T * CH * 4} : offset
  %p_mem = buffer.alloca<workgroup> align(16) %p_bytes : buffer
  %p_lds = buffer.view %p_mem[%zero] : buffer -> view<{T}x{CH}xf32>
{chr(10).join(q)}
  %jraw = index.add %base, %lane : index
  %jc = index.min %jraw, %last : index
  %j = index.assume %jc [range(%jc, 0, 1048575), lt(%jc, %n_kv)] : index
{chr(10).join(kd)}
{chr(10).join(per)}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %dim_raw = index.rem %lane, %c128 : index
  %dim = index.assume %dim_raw [range(%dim_raw, 0, {D - 1})] : index
  %par = index.div %lane, %c128 : index
  {accs} = scf.for %kk = [%par to %cCH step %c2] ({inits}) -> ({", ".join(["f32"] * T)}) {{
    %kq = index.assume %kk [range(%kk, 0, {CH - 1})] : index
    %jv0 = index.add %base, %kq : index
    %jv1 = index.min %jv0, %last : index
    %jv = index.assume %jv1 [range(%jv1, 0, 1048575), lt(%jv1, %n_kv)] : index
{vload}
{pv}
    scf.yield {", ".join(f"%n{t}" for t in range(T))} : {", ".join(["f32"] * T)}
  }}
  %h_bytes = index.constant {T * D * 4} : offset
  %h_mem = buffer.alloca<workgroup> align(16) %h_bytes : buffer
  %h_lds = buffer.view %h_mem[%zero] : buffer -> view<{T}x{D}xf32>
  %odd = index.cmp eq, %par, %c1 : index
  scf.if %odd {{
{stores}
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %pt_view = buffer.view %part[%zero] : buffer -> view<{T}x{H}x4096x{D + 2}xf32>
  %even = index.cmp eq, %par, %c0 : index
  scf.if %even {{
{fin}
  }}
  %first = index.cmp eq, %lane, %c0 : index
  scf.if %first {{
    %iM = index.constant {D} : index
    %iL = index.constant {D + 1} : index
{ml}
  }}
  kernel.return
}}
"""
    comb = f"""
kernel.def export("{cname}") @{cname}(%n_kv: index) {{
  %c1 = index.constant 1 : index
  %heads = index.constant {H} : index
  %ct = index.constant {T} : index
  %cD = index.constant {D} : index
  kernel.launch.config workgroups(%heads, %ct, %c1) workgroup_size(%cD, %c1, %c1) : index
}} launch(%n_kv_arg: index, %part: buffer, %gate: buffer, %out: buffer) {{
  %n_kv = index.assume %n_kv_arg [range(%n_kv_arg, {T}, 1048576)] : index
  %h_id = kernel.workgroup.id<x> : index
  %h = index.assume %h_id [range(%h_id, 0, {H - 1})] : index
  %t_id = kernel.workgroup.id<y> : index
  %t = index.assume %t_id [range(%t_id, 0, {T - 1})] : index
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
  %pt_view = buffer.view %part[%zero] : buffer -> view<{T}x{H}x4096x{D + 2}xf32>
  %g_view = buffer.view %gate[%zero] : buffer -> view<{T}x{H}x{D}xf32>
  %o_view = buffer.view %out[%zero] : buffer -> view<{T}x{H}x{D}xf32>
  %mx = scf.for %c = [%c0 to %chunks step %c1] (%acc = %zero_f : f32) -> (f32) {{
    %cc = index.assume %c [range(%c, 0, 4095)] : index
    %mc = view.load %pt_view[%t, %h, %cc, %iM] : view<{T}x{H}x4096x{D + 2}xf32> -> f32
    %nm = scalar.maxnumf %acc, %mc : f32
    scf.yield %nm : f32
  }}
  %nmx = scalar.subf %zero_f, %mx : f32
  %esink = scalar.expf<afn> %nmx : f32
  %osum, %lsum = scf.for %c = [%c0 to %chunks step %c1] (%ao = %zero_f : f32, %al = %esink : f32) -> (f32, f32) {{
    %cc = index.assume %c [range(%c, 0, 4095)] : index
    %mc = view.load %pt_view[%t, %h, %cc, %iM] : view<{T}x{H}x4096x{D + 2}xf32> -> f32
    %lc = view.load %pt_view[%t, %h, %cc, %iL] : view<{T}x{H}x4096x{D + 2}xf32> -> f32
    %oc = view.load %pt_view[%t, %h, %cc, %dim] : view<{T}x{H}x4096x{D + 2}xf32> -> f32
    %dm = scalar.subf %mc, %mx : f32
    %w = scalar.expf<afn> %dm : f32
    %no = scalar.fmaf %oc, %w, %ao : f32
    %nl = scalar.fmaf %lc, %w, %al : f32
    scf.yield %no, %nl : f32, f32
  }}
  %attn = scalar.divf %osum, %lsum : f32
  %gl = view.load %g_view[%t, %h, %dim] : view<{T}x{H}x{D}xf32> -> f32
  %gs = scalar.logisticf<afn> %gl : f32
  %gated = scalar.mulf %attn, %gs : f32
  view.store %gated, %o_view[%t, %h, %dim] : f32, view<{T}x{H}x{D}xf32>
  kernel.return
}}
"""
    return part, comb
