"""Looped layers (Inkling 2: the core layers [core_start, core_end) run num_passes times per token, kits/inkling-2-mini/NOTES.md). As
Unsloth's llama.cpp fork builds it (src/models/inkling2-loop.cpp, inkling2.cpp control(); HRX-like E271):
  - the prelude layers run once; every pass starts again from the prelude's output (and its block states); weights are shared across
    passes, and each pass has its own KV / conv cache slots (24 physical layers + 2 x 12 = 48 slots for Inkling 2 Mini);
  - from the second pass on, each core layer mixes a message (a depth-attention mix of the previous pass's block states and output
    with the layer's 4 message queries) into its q, k, v and FFN inputs through a controller:
        z = D [rms_norm(cur); rms_norm(msg)]          (D: rank x 2H, eps 1e-6, no norm weight)
        out = cur - tanh(F z) * cur + tanh(G z) * msg  (F, G: H x rank)
    ggml's op order: add(sub(cur, mul(forget, cur)), mul(gain, message)).
  - after the last pass the coda layers run once.

depth_mix(H, S, NQ): the depth attention (fork's mix(), the block attention residuals, the loop messages and the output query): S states
(the earlier block outputs and the current value, stacked [S][tokens][H]) are RMS-normalised (eps 1e-6, no weight), each of NQ learned
queries scores them (q . n_s), softmax over s with scale 1 / sqrt(H), and the result is sum_s p_s n_s per query ([tokens][NQ][H]).
One workgroup per token; each state is read once and kept in registers (H / 256 values per lane per state).

controller(H, rank): the controller for one token per workgroup (256 lanes): the normalised [cur; msg] staged in LDS, z (rank values)
in LDS, then each lane's H / 256 outputs. Each weight byte is read once per token (Inkling: D 64 x 4608 + F, G 2304 x 64 f32, 2.4 MB per
controller; 4 controllers x 12 core layers x 2 passes ~ 226 MB per token, ~1 ms at 228 GB/s). A verify span (8 tokens) re-reads them
per token in this form: a multi-token variant is the next step once it is measured."""


def controller(H=2304, rank=64, name="g_loop_control"):
    T = 256
    assert H % T == 0 and 2 * H % 32 == 0 and rank % 8 == 0
    PL = H // T                        # outputs per lane
    RPW = rank // 8                    # z rows per wave (8 waves)
    CH = 16                            # dot chunk (Loom's vector.dotf takes short vectors)

    def dots(i):
        """gain and forget pre-activations of output %j{i}: rank / CH chunked dots of its G and F rows with z"""
        L, ga, fa = [], "%zero_f", "%zero_f"
        for q in range(rank // CH):
            L.append(f"""    %zq{i}_{q} = vector.slice %zv[{q * CH}] : vector<{rank}xf32> -> vector<{CH}xf32>
    %cq{i}_{q} = index.constant {q * CH} : index
    %gr{i}_{q} = vector.load %g_view[%j{i}, %cq{i}_{q}] : view<{H}x{rank}xf32> -> vector<{CH}xf32>
    %fr{i}_{q} = vector.load %f_view[%j{i}, %cq{i}_{q}] : view<{H}x{rank}xf32> -> vector<{CH}xf32>
    %gp{i}_{q} = vector.dotf<reassoc|contract> %gr{i}_{q}, %zq{i}_{q}, {ga} : vector<{CH}xf32>, vector<{CH}xf32>, f32
    %fp{i}_{q} = vector.dotf<reassoc|contract> %fr{i}_{q}, %zq{i}_{q}, {fa} : vector<{CH}xf32>, vector<{CH}xf32>, f32""")
            ga, fa = f"%gp{i}_{q}", f"%fp{i}_{q}"
        L.append(f"""    %gd{i} = scalar.addf {ga}, %zero_f : f32
    %fd{i} = scalar.addf {fa}, %zero_f : f32""")
        return "\n".join(L)

    zchunks = []
    acc = "%zero_f"
    for q in range(2 * H // (32 * CH)):    # a wave's 32 lanes read 32 x CH consecutive values per chunk (coalesced)
        zchunks.append(f"""    %kq{q} = index.constant {q * 32 * CH} : index
    %kk{q} = index.add %kq{q}, %k0 : index
    %xq{q} = vector.load %x_lds[%kk{q}] : view<{2 * H}xf32> -> vector<{CH}xf32>
    %dq{q} = vector.load %d_view[%row, %kk{q}] : view<{rank}x{2 * H}xf32> -> vector<{CH}xf32>
    %pq{q} = vector.dotf<reassoc|contract> %dq{q}, %xq{q}, {acc} : vector<{CH}xf32>, vector<{CH}xf32>, f32""")
        acc = f"%pq{q}"
    assert 2 * H % (32 * CH) == 0
    outs = []
    for i in range(PL):
        outs.append(f"""    %j{i}_raw = index.constant {i * T} : index
    %j{i} = index.add %lane, %j{i}_raw : index
{dots(i)}
    %gain{i} = scalar.tanhf<afn> %gd{i} : f32
    %forget{i} = scalar.tanhf<afn> %fd{i} : f32
    %cv{i} = view.load %cur_view[%token, %j{i}] : view<[%tokens]x{H}xf32> -> f32
    %m{i} = view.load %msg_view[%token, %j{i}] : view<[%tokens]x{H}xf32> -> f32
    %fc{i} = scalar.mulf %forget{i}, %cv{i} : f32
    %keep{i} = scalar.subf %cv{i}, %fc{i} : f32
    %gm{i} = scalar.mulf %gain{i}, %m{i} : f32
    %o{i} = scalar.addf %keep{i}, %gm{i} : f32
    view.store %o{i}, %out_view[%token, %j{i}] : f32, view<[%tokens]x{H}xf32>""")
    loads = []
    for i in range(PL):
        loads.append(f"""  %ci{i}_raw = index.constant {i * T} : index
  %ci{i} = index.add %lane, %ci{i}_raw : index
  %xc{i} = view.load %cur_view[%token, %ci{i}] : view<[%tokens]x{H}xf32> -> f32
  %xm{i} = view.load %msg_view[%token, %ci{i}] : view<[%tokens]x{H}xf32> -> f32
  %sc{i} = scalar.mulf %xc{i}, %xc{i} : f32
  %sm{i} = scalar.mulf %xm{i}, %xm{i} : f32""")
    sum_c, sum_m = "%sc0", "%sm0"
    adds = []
    for i in range(1, PL):
        adds.append(f"  %acc_c{i} = scalar.addf {sum_c}, %sc{i} : f32\n  %acc_m{i} = scalar.addf {sum_m}, %sm{i} : f32")
        sum_c, sum_m = f"%acc_c{i}", f"%acc_m{i}"
    stores = []
    for i in range(PL):
        stores.append(f"""  %nc{i} = scalar.mulf %xc{i}, %inv_c : f32
  %nm{i} = scalar.mulf %xm{i}, %inv_m : f32
  view.store %nc{i}, %x_lds[%ci{i}] : f32, view<{2 * H}xf32>
  %hm{i} = index.add %ci{i}, %hconst : index
  view.store %nm{i}, %x_lds[%hm{i}] : f32, view<{2 * H}xf32>""")
    return f"""
kernel.def export("{name}") @{name}(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c{T} = index.constant {T} : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c{T}, %c1, %c1) : index
}} launch(%tokens_arg: index, %cur: buffer, %msg: buffer, %d: buffer, %g: buffer, %f: buffer, %out: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, {T - 1})] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %zero_f = scalar.constant 0.0 : f32
  %hconst = index.constant {H} : index
  %cur_view = buffer.view %cur[%zero] : buffer -> view<[%tokens]x{H}xf32>
  %msg_view = buffer.view %msg[%zero] : buffer -> view<[%tokens]x{H}xf32>
  %out_view = buffer.view %out[%zero] : buffer -> view<[%tokens]x{H}xf32>
  %d_view = buffer.view %d[%zero] : buffer -> view<{rank}x{2 * H}xf32>
  %g_view = buffer.view %g[%zero] : buffer -> view<{H}x{rank}xf32>
  %f_view = buffer.view %f[%zero] : buffer -> view<{H}x{rank}xf32>
{chr(10).join(loads)}
{chr(10).join(adds)}
  %ssq_c = kernel.workgroup.reduce<addf> {sum_c} : f32
  %ssq_m = kernel.workgroup.reduce<addf> {sum_m} : f32
  %hf = scalar.constant {float(H)!r} : f32
  %eps = scalar.constant 1e-06 : f32
  %mean_c = scalar.divf %ssq_c, %hf : f32
  %mean_m = scalar.divf %ssq_m, %hf : f32
  %st_c = scalar.addf %mean_c, %eps : f32
  %st_m = scalar.addf %mean_m, %eps : f32
  %inv_c = scalar.rsqrtf %st_c : f32
  %inv_m = scalar.rsqrtf %st_m : f32
  %x_bytes = index.constant {2 * H * 4} : offset
  %x_mem = buffer.alloca<workgroup> align(16) %x_bytes : buffer
  %x_lds = buffer.view %x_mem[%zero] : buffer -> view<{2 * H}xf32>
{chr(10).join(stores)}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %z_bytes = index.constant {rank * 4} : offset
  %z_mem = buffer.alloca<workgroup> align(16) %z_bytes : buffer
  %z_lds = buffer.view %z_mem[%zero] : buffer -> view<{rank}xf32>
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, 7)] : index
  %wl_id = kernel.subgroup.lane.id : index
  %wl = index.assume %wl_id [range(%wl_id, 0, 31)] : index
  %rpw = index.constant {RPW} : index
  %row0 = index.mul %wave, %rpw : index
  %chs = index.constant {CH} : index
  %k0 = index.mul %wl, %chs : index
  scf.for %rr = [%c0 to %rpw step %c1] {{
    %row_raw = index.add %row0, %rr : index
    %row = index.assume %row_raw [range(%row_raw, 0, {rank - 1})] : index
{chr(10).join(zchunks)}
    %zsum = kernel.subgroup.reduce<addf> {acc} : f32
    %lead = index.cmp eq, %wl, %c0 : index
    scf.if %lead {{
      view.store %zsum, %z_lds[%row] : f32, view<{rank}xf32>
    }}
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %zv = vector.load %z_lds[%c0] : view<{rank}xf32> -> vector<{rank}xf32>
{chr(10).join(outs)}
  kernel.return
}}
"""


def _wg_sums(vals, tag, waves):
    """Loom lines: the workgroup sums of several per-lane f32 values with one barrier: a subgroup sum per value, each wave's lane 0
    writes its sums to LDS, one barrier, every lane adds the waves' rows. Returns the lines and the names of the sums."""
    n = len(vals)
    L = [f"""  %{tag}_bytes = index.constant {waves * n * 4} : offset
  %{tag}_mem = buffer.alloca<workgroup> align(16) %{tag}_bytes : buffer
  %{tag}_lds = buffer.view %{tag}_mem[%zero] : buffer -> view<{waves}x{n}xf32>"""]
    for k, v in enumerate(vals):
        L.append(f"  %{tag}_w{k} = kernel.subgroup.reduce<addf> {v} : f32")
    L.append(f"  scf.if %wave_lead {{")
    for k in range(n):
        L.append(f"    %{tag}_k{k} = index.constant {k} : index\n    view.store %{tag}_w{k}, %{tag}_lds[%wave, %{tag}_k{k}] : f32, view<{waves}x{n}xf32>")
    L.append("  }\n  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    out = []
    for k in range(n):
        acc = None
        for w in range(waves):
            L.append(f"  %{tag}_r{k}_{w} = view.load %{tag}_lds[{w}, {k}] : view<{waves}x{n}xf32> -> f32")
            if acc is None:
                acc = f"%{tag}_r{k}_{w}"
            else:
                L.append(f"  %{tag}_a{k}_{w} = scalar.addf {acc}, %{tag}_r{k}_{w} : f32")
                acc = f"%{tag}_a{k}_{w}"
        out.append(acc)
    return L, out


def depth_mix(H=2304, S=5, NQ=3, name=None):
    T, waves = 256, 8
    assert H % T == 0 and S >= 1 and NQ >= 1
    PL = H // T
    name = name or f"g_depth_mix_s{S}_q{NQ}"
    L, sq = [], []
    for st in range(S):                      # load each state once; per-lane sums of squares
        acc = None
        for i in range(PL):
            L.append(f"  %x{st}_{i} = view.load %s_view[{st}, %token, %i{i}] : view<{S}x[%tokens]x{H}xf32> -> f32")
            L.append(f"  %sq{st}_{i} = scalar.mulf %x{st}_{i}, %x{st}_{i} : f32")
            if acc is None:
                acc = f"%sq{st}_{i}"
            else:
                L.append(f"  %ss{st}_{i} = scalar.addf {acc}, %sq{st}_{i} : f32")
                acc = f"%ss{st}_{i}"
        sq.append(acc)
    for q in range(NQ):
        for i in range(PL):
            L.append(f"  %q{q}_{i} = view.load %q_view[{q}, %i{i}] : view<{NQ}x{H}xf32> -> f32")
    red, tots = _wg_sums(sq, "ssq", waves)
    L += red
    for st in range(S):
        L.append(f"""  %mean{st} = scalar.divf {tots[st]}, %hf : f32
  %stab{st} = scalar.addf %mean{st}, %eps : f32
  %inv{st} = scalar.rsqrtf %stab{st} : f32""")
        for i in range(PL):
            L.append(f"  %n{st}_{i} = scalar.mulf %x{st}_{i}, %inv{st} : f32")
    parts = []
    for q in range(NQ):
        for st in range(S):                  # per-lane partial score of state st under query q
            acc = None
            for i in range(PL):
                L.append(f"  %pr{q}_{st}_{i} = scalar.mulf %q{q}_{i}, %n{st}_{i} : f32")
                if acc is None:
                    acc = f"%pr{q}_{st}_{i}"
                else:
                    L.append(f"  %pa{q}_{st}_{i} = scalar.addf {acc}, %pr{q}_{st}_{i} : f32")
                    acc = f"%pa{q}_{st}_{i}"
            parts.append(acc)
    red, dots = _wg_sums(parts, "dot", waves)
    L += red
    for q in range(NQ):
        for st in range(S):
            L.append(f"  %sc{q}_{st} = scalar.mulf {dots[q * S + st]}, %scale : f32")
        mx = f"%sc{q}_0"
        for st in range(1, S):
            L.append(f"  %mx{q}_{st} = scalar.maxnumf {mx}, %sc{q}_{st} : f32")
            mx = f"%mx{q}_{st}"
        den = None
        for st in range(S):
            L.append(f"  %sh{q}_{st} = scalar.subf %sc{q}_{st}, {mx} : f32\n  %e{q}_{st} = scalar.expf<afn> %sh{q}_{st} : f32")
            if den is None:
                den = f"%e{q}_0"
            else:
                L.append(f"  %den{q}_{st} = scalar.addf {den}, %e{q}_{st} : f32")
                den = f"%den{q}_{st}"
        L.append(f"  %rden{q} = scalar.divf %one, {den} : f32")
        for st in range(S):
            L.append(f"  %p{q}_{st} = scalar.mulf %e{q}_{st}, %rden{q} : f32")
        for i in range(PL):
            acc = None
            for st in range(S):
                L.append(f"  %w{q}_{i}_{st} = scalar.mulf %n{st}_{i}, %p{q}_{st} : f32")
                if acc is None:
                    acc = f"%w{q}_{i}_{st}"
                else:
                    L.append(f"  %o{q}_{i}_{st} = scalar.addf {acc}, %w{q}_{i}_{st} : f32")
                    acc = f"%o{q}_{i}_{st}"
            L.append(f"  view.store {acc}, %o_view[%token, {q}, %i{i}] : f32, view<[%tokens]x{NQ}x{H}xf32>")
    idx = "\n".join(f"  %ic{i} = index.constant {i * T} : index\n  %i{i} = index.add %lane, %ic{i} : index" for i in range(PL))
    return f"""
kernel.def export("{name}") @{name}(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c{T} = index.constant {T} : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c{T}, %c1, %c1) : index
}} launch(%tokens_arg: index, %states: buffer, %queries: buffer, %out: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, {T - 1})] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, {waves - 1})] : index
  %wl_id = kernel.subgroup.lane.id : index
  %c0 = index.constant 0 : index
  %wave_lead = index.cmp eq, %wl_id, %c0 : index
  %zero = index.constant 0 : offset
  %hf = scalar.constant {float(H)!r} : f32
  %eps = scalar.constant 1e-06 : f32
  %one = scalar.constant 1.0 : f32
  %scale = scalar.constant {1.0 / H ** 0.5!r} : f32
  %s_view = buffer.view %states[%zero] : buffer -> view<{S}x[%tokens]x{H}xf32>
  %q_view = buffer.view %queries[%zero] : buffer -> view<{NQ}x{H}xf32>
  %o_view = buffer.view %out[%zero] : buffer -> view<[%tokens]x{NQ}x{H}xf32>
{idx}
{chr(10).join(L)}
  kernel.return
}}
"""
