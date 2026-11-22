"""Causal short convolutions with a residual (Inkling 2: four per layer, on K, V, the attention output and the FFN output; on the global
layers K and V also get input-dependent taps per KV head). As Unsloth's llama.cpp fork builds it (inkling2.cpp convolve(); HRX-like E273),
per channel c with the window x = [the last n_conv - 1 inputs of the sequence (its history); this span's inputs]:
    out_t = sum_k w[c, k] x[t + k]                       (ggml_ssm_conv: k = 0 .. n_conv - 1 in order, x[t + n_conv - 1] = token t)
    dynamic: out_t += sum_lag tap[lag, head(c), t] x[t + n_conv - 1 - lag]      (lag 0 = token t itself, added one lag at a time)
    y_t = (cur_t + out_t) / sqrt((dynamic ? 2 : 1) n_conv + 1)
and the history becomes the window's last n_conv - 1 inputs.

sconv(C, T, heads=0): T tokens of one sequence (T = 1 decode, up to the verify span), one lane per channel: the history read once,
the T outputs, the new history written once. weight [C][n_conv] (ggml {n_conv, C}), history [C][n_conv - 1], cur / out [T][C], taps
[T][heads][n_conv] when heads > 0 (channels per head = C / heads). Speculative rollback (keeping each step's history) is a later
variant, as gen_gdn's origins."""


def sconv(C, T=1, heads=0, n_conv=4, name=None):
    assert T >= 1 and (heads == 0 or C % heads == 0)
    lanes = 256
    name = name or f"g_sconv_c{C}_t{T}" + (f"_dyn{heads}" if heads else "")
    P = n_conv - 1
    scale = 1.0 / ((2 if heads else 1) * n_conv + 1) ** 0.5   # as f32 equal to ggml's 1.0f/sqrtf(n) for n 5 and 9 (test_sconv)
    L = []
    for k in range(P):
        L.append(f"    %x{k} = view.load %h_view[%ch, {k}] : view<{C}x{P}xf32> -> f32")
    for k in range(n_conv):
        L.append(f"    %w{k} = view.load %w_view[%ch, {k}] : view<{C}x{n_conv}xf32> -> f32")
    if heads:
        L.append(f"    %cph = index.constant {C // heads} : index\n    %head = index.div %ch, %cph : index")
    for t in range(T):
        L.append(f"    %x{P + t} = view.load %cur_view[{t}, %ch] : view<{T}x{C}xf32> -> f32")
        acc = None
        for k in range(n_conv):
            L.append(f"    %m{t}_{k} = scalar.mulf %x{t + k}, %w{k} : f32")
            if acc is None:
                acc = f"%m{t}_{k}"
            else:
                L.append(f"    %s{t}_{k} = scalar.addf {acc}, %m{t}_{k} : f32")
                acc = f"%s{t}_{k}"
        if heads:
            for lag in range(n_conv):
                L.append(f"""    %tap{t}_{lag} = view.load %tap_view[{t}, %head, {lag}] : view<{T}x{heads}x{n_conv}xf32> -> f32
    %dt{t}_{lag} = scalar.mulf %x{t + P - lag}, %tap{t}_{lag} : f32
    %ds{t}_{lag} = scalar.addf {acc}, %dt{t}_{lag} : f32""")
                acc = f"%ds{t}_{lag}"
        L.append(f"""    %r{t} = scalar.addf %x{P + t}, {acc} : f32
    %y{t} = scalar.mulf %r{t}, %scale : f32
    view.store %y{t}, %out_view[{t}, %ch] : f32, view<{T}x{C}xf32>""")
    for k in range(P):
        L.append(f"    view.store %x{T + k}, %h_view[%ch, {k}] : f32, view<{C}x{P}xf32>")
    targs = ", %taps: buffer" if heads else ""
    tview = f"\n  %tap_view = buffer.view %taps[%zero] : buffer -> view<{T}x{heads}x{n_conv}xf32>" if heads else ""
    return f"""
kernel.def export("{name}") @{name}() {{
  %c1 = index.constant 1 : index
  %wgs = index.constant {(C + lanes - 1) // lanes} : index
  %c{lanes} = index.constant {lanes} : index
  kernel.launch.config workgroups(%wgs, %c1, %c1) workgroup_size(%c{lanes}, %c1, %c1) : index
}} launch(%cur: buffer, %weight: buffer, %history: buffer{targs}, %out: buffer) {{
  %zero = index.constant 0 : offset
  %wg = kernel.workgroup.id<x> : index
  %lane = kernel.workitem.id<x> : index
  %c{lanes} = index.constant {lanes} : index
  %base = index.mul %wg, %c{lanes} : index
  %ch_raw = index.add %base, %lane : index
  %channels = index.constant {C} : index
  %live = index.cmp ult, %ch_raw, %channels : index
  %scale = scalar.constant {scale!r} : f32
  %cur_view = buffer.view %cur[%zero] : buffer -> view<{T}x{C}xf32>
  %out_view = buffer.view %out[%zero] : buffer -> view<{T}x{C}xf32>
  %w_view = buffer.view %weight[%zero] : buffer -> view<{C}x{n_conv}xf32>
  %h_view = buffer.view %history[%zero] : buffer -> view<{C}x{P}xf32>{tview}
  scf.if %live {{
    %ch = index.assume %ch_raw [range(%ch_raw, 0, {C - 1})] : index
{chr(10).join(L)}
  }}
  kernel.return
}}
"""
