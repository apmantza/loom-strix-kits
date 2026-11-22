"""F16O11=1 (HRX-like E180): the prefill FFN's f16 swiglu GEMM writes the f16 activation copy and its row scales itself, on gfx11
and on its gfx12 twin alike; the g_row_scale_f16* pass over the f32 result (2.69 s per 81k prefill, E172) runs only on the paths
that still produce f32 (t < pf_min: the quantized swiglu GEMMs, the mixed-format store + silu_mul fallback).

Exact while every live FFN row has max|f| < 2^14, where the row scale is 1 (g_row_scale_f16: e = clamp(floor(log2 max) - 13, 0,
100)); E178's probe found no row >= 2^11 over 81k prompt tokens. A larger row rounds differently; > 65504 becomes inf. Precision
change: go-live is the user's decision.

The rewrite, on gen_llm's final text (after pf16_pad and kv8, before dual_target):
  kernels  `<K>_f16o` for K in g_gemm_f16_swiglu{,_m128,_m256} and `<K>_f16o_rdna4` for their twins: the y view is f16 at the
           padded row stride n + PF16_PAD (the down GEMM's X rows; pf16_pad.py), and the kernel writes scales_out[row] = (1, 1)
           for a row whose input scale is nonzero, (0, 0) otherwise (where it decides the tile skip), as g_row_scale_f16_live_lv
           does from the row's info word (the producer zeroes exactly those rows' scales).
  @llm_advance  each FFN row-scale site (`g_row_scale_f16*[%t](%t, [%info, ]%nFF, %f, %s_f, %f16)` in `scf.if %mm`) whose
           `scf.if %pf_big` then-branch writes %f only through the three f16 swiglu launches: those launches write %f16 and %s_f
           instead, and the row scale runs under `scf.if %f16o_small` (t < pf_min).
Fails closed: a site whose pf_big branch has another writer of %f, a count other than one site per layer, or a kernel anchor
that moved stops the build. Off (F16O11 unset): the text is returned unchanged."""
import os
import re

ON = os.environ.get("F16O11", "0") not in ("", "0", "false", "no", "off")
KERNELS = ("g_gemm_f16_swiglu", "g_gemm_f16_swiglu_m128", "g_gemm_f16_swiglu_m256")
RS = re.compile(r"^(?P<ind> *)kernel\.launch @g_row_scale_f16(?:_live)?(?:_lv)?\[%t\]\(%t, (?:%info, )?%nFF, %f, %s_f, %f16\) : .*$", re.M)


def _kernel(text, name):
    m = re.search(r'(?ms)^kernel\.def (?:target\(@\w+\) )?export\("' + name + r'"\) @' + name + r"\(.*?\n\} launch\(.*?\n\}\n", text)
    assert m, f"f16o11: kernel {name} not found"
    return m


def f16out(k, old, new, pad):
    """`new` from the f32-output swiglu kernel text `k` (export `old`): f16 output at row stride n + pad, plus scales_out."""
    t = k.replace(f'export("{old}") @{old}(', f'export("{new}") @{new}(', 1)
    for a, b in ((", %y: buffer) {", ", %y: buffer, %scales_out: buffer) {"),
                 ("  %y_layout = encoding.layout.strided [%n, %c1] : encoding<layout>\n",
                  f"  %f16o_pad = index.constant {pad} : index\n  %f16o_ny = index.add %n, %f16o_pad : index\n"
                  "  %y_layout = encoding.layout.strided [%f16o_ny, %c1] : encoding<layout>\n"),
                 ("  %tile_first = index.madd %row_origin, %n, %out_origin : index\n",
                  "  %tile_first = index.madd %row_origin, %f16o_ny, %out_origin : index\n"),
                 ("  %tile_offset = index.scale %tile_first, %four : index, offset -> offset\n",
                  "  %tile_offset = index.scale %tile_first, %two : index, offset -> offset\n")):
        assert t.count(a) == 1, f"f16o11: {old}: anchor {a.strip()[:60]!r} seen {t.count(a)} times"
        t = t.replace(a, b, 1)
    assert "  %two = index.constant 2 : offset\n" in t, f"f16o11: {old}: no %two offset constant"
    t, n = re.subn(r"view<(\d+)x(\d+)xf32, %y_layout>", r"view<\1x\2xf16, %y_layout>", t)
    assert n >= 3 and "xf32, %y_layout>" not in t, f"f16o11: {old}: {n} y views converted"
    live = "  %tile_live = scalar.cmpf one, %tile_any, %no_scale : f32\n"
    assert t.count(live) == 1 and "%skip_in = " in t and "%tile_scale = " in t, f"f16o11: {old}: tile-skip anchors"
    return t.replace(live, live + """  %so_view = buffer.view %scales_out[%origin] : buffer -> view<[%rows]x2xf32>
  %so_n0 = index.cmp eq, %out_group, %c0 : index
  %so_act = scalar.andi %so_n0, %skip_in : i1
  scf.if %so_act {
    %so_one = scalar.constant 1.0 : f32
    %so_live = scalar.cmpf one, %tile_scale, %no_scale : f32
    %so_val = scf.select %so_live, %so_one, %no_scale : f32
    view.store %so_val, %so_view[%tile_row, %c0] : f32, view<[%rows]x2xf32>
    view.store %so_val, %so_view[%tile_row, %c1] : f32, view<[%rows]x2xf32>
  }
""", 1)


def _block_end(lines, i):
    """index of the line closing the `{` block opened on lines[i] (brace depth over the following lines)"""
    depth = lines[i].count("{") - lines[i].count("}")
    j = i
    while depth > 0:
        j += 1
        depth += lines[j].count("{") - lines[j].count("}")
    return j


LEAF = re.compile(r"^(?P<ind> *)kernel\.launch @(?P<k>g_gemm_f16_swiglu(?:_m128|_m256)?)\[(?P<g>[^\]]*)\]\((?P<a>.*), %f\) : \[(?P<gt>[^\]]*)\]\((?P<at>.*), buffer\)$")


def _advance(seg, layers):
    lines = seg.split("\n")
    sites = [i for i, l in enumerate(lines) if RS.match(l)]
    assert len(sites) == layers, f"f16o11: {len(sites)} FFN row-scale sites in @llm_advance, expected {layers}"
    prev = 0
    for r in sites:
        assert lines[r - 1].strip() == "scf.if %mm {" and lines[r + 1].strip() == "} else {" and "g_quant_q8" in lines[r + 2], \
            f"f16o11: row scale at line {r}: not in `scf.if %mm {{ .. }} else {{ g_quant_q8 }}`"
        # the FFN's block: the last outermost `scf.if %pf_big {` that closes before the site (each layer's attention projections
        # have their own, earlier; uniform-format layers nest it in `scf.if %mm`, the mixed-format fallback has a second one
        # inside its else-branch)
        heads, outer_end = [], -1
        for i in range(prev, r):
            if lines[i].strip() == "scf.if %pf_big {":
                e_i = _block_end(lines, i)
                if i > outer_end and e_i < r:
                    heads.append(i)
                    outer_end = e_i
        assert heads, f"f16o11: row scale at line {r}: no scf.if %pf_big block before it"
        h = heads[-1]
        # its then-branch ends at the `} else {` at brace depth 1 (the generator's indentation is not reliable: the else of
        # a nested `scf.if %big` can sit at the head's indentation)
        depth, els = 1, None
        for i in range(h + 1, _block_end(lines, h) + 1):
            if lines[i].strip() == "} else {" and depth == 1:
                els = i
                break
            depth += lines[i].count("{") - lines[i].count("}")
        assert els, f"f16o11: pf_big block at line {h}: no else-branch"
        writers = [i for i in range(h + 1, els) if re.search(r"kernel\.launch @\w+\[[^\]]*\]\((?:[^)]*, )?%f[,)]", lines[i])]
        leaves = [i for i in writers if LEAF.match(lines[i])]
        assert len(leaves) == 3 and writers == leaves, f"f16o11: pf_big branch at line {h}: writers of %f {[lines[i].strip()[:60] for i in writers]}"
        for i in leaves:
            m = LEAF.match(lines[i])
            lines[i] = (f"{m['ind']}kernel.launch @{m['k']}_f16o[{m['g']}]({m['a']}, %f16, %s_f) : "
                        f"[{m['gt']}]({m['at']}, buffer, buffer)")
        ri = lines[r][:len(lines[r]) - len(lines[r].lstrip())]
        lines[r] = f"{ri}scf.if %f16o_small {{\n  {lines[r]}\n{ri}}}"
        prev = r + 1
    seg = "\n".join(lines)
    a = "  %pf_big = index.cmp uge, %t, %pf_min : index\n"
    assert seg.count(a) == 1, "f16o11: %pf_big definition"
    return seg.replace(a, a + "  %f16o_small = index.cmp ult, %t, %pf_min : index   // F16O11 (E180): the row scale runs off the f16o path\n", 1)


def apply(text, layers):
    if not ON:
        return text
    import pf16_pad
    pad = pf16_pad.PAD
    first = text.index("command.def ")
    extra = []
    for k in KERNELS:
        m = _kernel(text, k)
        extra.append(f16out(m.group(0), k, k + "_f16o", pad))
        tw = re.search(r'(?ms)^kernel\.def (?:target\(@\w+\) )?export\("' + k + r'_rdna4"\) @' + k + r"_rdna4\(.*?\n\} launch\(.*?\n\}\n", text)
        if tw:
            extra.append(f16out(tw.group(0), k + "_rdna4", k + "_f16o_rdna4", pad))
    text = text[:first] + "\n".join(extra) + "\n" + text[first:]   # kernels before the programs (dual_target: a twin before its launches)
    segs = re.split(r"(?m)^(?=command\.def )", text)
    done = 0
    for i, s in enumerate(segs):
        if s.startswith("command.def ") and "@llm_advance()" in s.split("\n", 1)[0]:
            try:
                segs[i] = _advance(s, layers)
            except AssertionError:
                if os.environ.get("F16O11_DEBUG"):   # the program text the pass refused, for the diagnosis
                    open(os.environ["F16O11_DEBUG"], "w").write(s)
                raise
            done += 1
    assert done == 1, f"f16o11: {done} @llm_advance programs"
    return "".join(segs)
