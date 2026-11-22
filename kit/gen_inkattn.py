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
