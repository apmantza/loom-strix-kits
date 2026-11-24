"""F16O_O=1 (HRX-like E375): the producers of the attention / GDN output %o write its f16 copy %o16 themselves in prefill-shape
epochs (t >= 64, `%gemm`), as F16O11 (f16o11.py) does for the FFN; the o-site g_row_scale_f16* pass over %o (128 launches per
4096-row epoch in @llm_advance, ~1.0 s per 81k prefill) runs only on the epochs that still produce f32 (17 <= t < 64: the
non-_lv g_gdn_norm_gate and the decode-shaped attention chain).

Exact while every live output row has max|o| < 2^14, where g_row_scale_f16's row scale is 1 (e = clamp(floor(log2 max) - 13, 0,
100)); E375's probe found every served attention / GDN output row < 2^11. A larger row rounds differently (its f16 copy is not
rescaled); > 65504 becomes inf. Precision change: go-live is the user's decision.

The rewrite, on gen_llm's final text (after f16o11, before dual_target):
  kernels  `<K>_f16o` for K in g_gdn_norm_gate_lv, g_attention_flash2, g_attention_flash2_partial, g16_attention_combine_listed,
           and `<K>_f16o_rdna4` for the twins the text has: the output view is f16 at the padded row stride width + PF16_PAD
           (the out-projection GEMM's X rows, pf16_pad.py); the values are rounded as the row scale rounds them at scale 1
           (vector.fptrunc / scalar.fptrunc, or the f32 fragment store into an f16 view, as f16o11's GEMMs). %o itself is not
           written: its only readers are the row scale (kept for t < 64) and g_quant_q8 (t < 17).
  scales   g_gdn_norm_gate_lv_f16o writes %s_o: (1, 1) for a live row (info word 2 != 0), (0, 0) for a dead one, from workgroup
           (row, 0) lane 0, as g_row_scale_f16_live_lv does for rows < 2^14. The attention producers write no scales: their
           workgroups never visit the dead (padding) rows, so they could not zero them. An attention layer's %s_o is the one the
           GDN layer before it wrote in the same epoch (info is per epoch, every o-site runs over [%t] rows): the pass fails the
           build unless the first o-site of @llm_advance is a GDN site and %s_o has no writer but the o-sites (rival kept out: a
           separate scale-only kernel per attention site, one more launch for state the GDN layer already leaves).
  @llm_advance  each o-site (`g_row_scale_f16*[%t](%t, [%info, ]%nQ|%nVD, %o, %s_o, %o16)` in `scf.if %mm {..} else {g_quant_q8}`)
           whose `scf.if %gemm` writer block just before it writes %o only through those kernels: they write %o16 (and %s_o)
           instead, and the row scale runs under `scf.if %f16oo_small` (t < 64).
Fails closed: a site whose %gemm branch has another writer of %o (or any writer of %o between sites), a count other than one site
per layer, a writer of %s_o outside the sites, or a kernel anchor that moved stops the build. Off (F16O_O unset): the text is
returned unchanged."""
import os
import re

ON = os.environ.get("F16O_O", "0") not in ("", "0", "false", "no", "off")
GDN = "g_gdn_norm_gate_lv"
ATTN = ("g_attention_flash2", "g_attention_flash2_partial", "g16_attention_combine_listed")
SITE = re.compile(r"^(?P<ind> *)kernel\.launch @g_row_scale_f16(?:_live)?(?:_lv)?\[%t\]\(%t, (?:%info, )?(?P<k>%nQ|%nVD), %o, %s_o, %o16\) : .*$")
WRITES_O = re.compile(r"(?:kernel\.launch|template\.apply)\W[^(]*\((?:[^)]*, )?%o[,)]")
LAUNCH_O = re.compile(r"^(?P<ind> *)kernel\.launch @(?P<k>\w+)\[(?P<g>[^\]]*)\]\((?P<a>.*), %o\) : \[(?P<gt>[^\]]*)\]\((?P<at>.*), buffer\)$")


def _kernel(text, name):
    return re.search(r'(?ms)^kernel\.def (?:target\(@\w+\) )?export\("' + name + r'"\) @' + name + r"\(.*?\n\} launch\(.*?\n\}\n", text)


def _sub1(t, a, b, who):
    assert t.count(a) == 1, f"f16o_o: {who}: anchor {a.strip()[:70]!r} seen {t.count(a)} times"
    return t.replace(a, b, 1)


def _rename(k, old, new):
    return _sub1(k, f'export("{old}") @{old}(', f'export("{new}") @{new}(', old)


def gdn_f16o(k, old, new, pad):
    """g_gdn_norm_gate_lv -> f16 rows at width + pad, scales_out (1, 1) live / (0, 0) dead."""
    t = _rename(k, old, new)
    t = _sub1(t, ", %z: buffer, %o: buffer) {", ", %z: buffer, %o: buffer, %scales_out: buffer) {", old)
    m = re.findall(r"^  %o_view = buffer\.view %o\[%zero\] : buffer -> view<\[%tokens\]x(\d+)xf32>\n", t, re.M)
    assert len(m) == 1, f"f16o_o: {old}: o view {m}"
    w = int(m[0]); f32v, f16v = f"view<[%tokens]x{w}xf32>", f"view<[%tokens]x{w + pad}xf16>"
    t = _sub1(t, f"  %o_view = buffer.view %o[%zero] : buffer -> {f32v}\n", f"""  %o_view = buffer.view %o[%zero] : buffer -> {f16v}
  %so_c0 = index.constant 0 : index
  %so_c1 = index.constant 1 : index
  %so_item = kernel.workitem.id<x> : index
  %so_l0 = index.cmp eq, %so_item, %so_c0 : index
  %so_g0 = index.cmp eq, %group_id, %so_c0 : index
  %so_w = scalar.andi %so_l0, %so_g0 : i1
  scf.if %so_w {{
    %so_one = scalar.constant 1.0 : f32
    %so_view = buffer.view %scales_out[%zero] : buffer -> view<[%tokens]x2xf32>
    view.store %so_one, %so_view[%token, %so_c0] : f32, view<[%tokens]x2xf32>
    view.store %so_one, %so_view[%token, %so_c1] : f32, view<[%tokens]x2xf32>
  }}
""", old)
    st = re.compile(r"^(?P<i> *)view\.store (?P<v>%\w+), %o_view\[(?P<r>%\w+), (?P<c>%\w+)\] : f32, " + re.escape(f32v) + "$", re.M)
    t, n = st.subn(lambda s: f"{s['i']}{s['v']}_h = scalar.fptrunc {s['v']} : f32 to f16\n"
                             f"{s['i']}view.store {s['v']}_h, %o_view[{s['r']}, {s['c']}] : f16, {f16v}", t)
    assert n >= 1 and t.count("%o_view[") == n, f"f16o_o: {old}: {n} of {t.count('%o_view[')} o stores converted"
    # the dead branch: the row-live block is the kernel's body; its else zeroes the row's scales (g_row_scale_f16_live_lv's
    # dead-row convention)
    assert t.count("  scf.if %lv_live {\n") == 1 and t.endswith("\n  }\n  kernel.return\n}\n"), f"f16o_o: {old}: row-live block"
    return t[:-len("  }\n  kernel.return\n}\n")] + """  } else {
    %sd_zero = index.constant 0 : offset
    %sd_c0 = index.constant 0 : index
    %sd_c1 = index.constant 1 : index
    %sd_item = kernel.workitem.id<x> : index
    %sd_group = kernel.workgroup.id<y> : index
    %sd_l0 = index.cmp eq, %sd_item, %sd_c0 : index
    %sd_g0 = index.cmp eq, %sd_group, %sd_c0 : index
    %sd_w = scalar.andi %sd_l0, %sd_g0 : i1
    scf.if %sd_w {
      %sd_zf = scalar.constant 0.0 : f32
      %sd_view = buffer.view %scales_out[%sd_zero] : buffer -> view<[%tokens]x2xf32>
      view.store %sd_zf, %sd_view[%token, %sd_c0] : f32, view<[%tokens]x2xf32>
      view.store %sd_zf, %sd_view[%token, %sd_c1] : f32, view<[%tokens]x2xf32>
    }
  }
  kernel.return
}
"""


def flash_f16o(k, old, new, pad):
    """g_attention_flash2: the transposed result-fragment store into an f16 view at row stride width + pad."""
    t = _rename(k, old, new)
    w = re.findall(r"^  %cqd = index\.constant (\d+) : index\n  %out_layout = encoding\.layout\.strided \[%c1, %cqd\] : encoding<layout>\n", t, re.M)
    assert len(w) == 1, f"f16o_o: {old}: out layout {w}"
    v = re.findall(r"^  %out_view = buffer\.view %out\[%out_offset\] : buffer -> view<(\d+x\d+)xf32, %out_layout>\n", t, re.M)
    assert len(v) == 1, f"f16o_o: {old}: out view {v}"
    for a in ("  %out_first_a = index.mul %first_token, %cqd : index\n", "  %out_first = index.add %out_first_a, %head_half : index\n"):
        assert t.count(a) == 1, f"f16o_o: {old}: anchor {a.strip()!r}"
    shp = v[0]
    t = _sub1(t, f"  %out_view = buffer.view %out[%out_offset] : buffer -> view<{shp}xf32, %out_layout>\n", f"""  %o16_pad = index.constant {pad} : index
  %o16_ld = index.add %cqd, %o16_pad : index
  %o16_layout = encoding.layout.strided [%c1, %o16_ld] : encoding<layout>
  %o16_first_a = index.mul %first_token, %o16_ld : index
  %o16_first = index.add %o16_first_a, %head_half : index
  %o16_two = index.constant 2 : offset
  %o16_offset = index.scale %o16_first, %o16_two : index, offset -> offset
  %out_view = buffer.view %out[%o16_offset] : buffer -> view<{shp}xf16, %o16_layout>
""", old)
    st = re.compile(r"^(?P<h> *vector\.fragment\.store<result> %\w+, %out_view\[[^\]]*\] shape \[[^\]]*\] : vector<8xf32>, )view<"
                    + shp + r"xf32, %out_layout>$", re.M)
    t, n = st.subn(lambda s: s["h"] + f"view<{shp}xf16, %o16_layout>", t)
    assert n >= 1 and t.count("%out_view[") == n, f"f16o_o: {old}: {n} of {t.count('%out_view[')} out stores converted"
    return t


def partial_f16o(k, old, new, pad):
    """g_attention_flash2_partial: per-element stores of the live columns, rounded to f16 at row stride width + pad."""
    t = _rename(k, old, new)
    m = re.findall(r"^  %out_flat = buffer\.view %out\[%origin\] : buffer -> view<\[%tokens\]x(\d+)xf32>\n", t, re.M)
    assert len(m) == 1, f"f16o_o: {old}: out view {m}"
    w = int(m[0]); f32v, f16v = f"view<[%tokens]x{w}xf32>", f"view<[%tokens]x{w + pad}xf16>"
    t = _sub1(t, f"  %out_flat = buffer.view %out[%origin] : buffer -> {f32v}\n", f"  %out_flat = buffer.view %out[%origin] : buffer -> {f16v}\n", old)
    st = re.compile(r"^(?P<i> *)view\.store (?P<v>%\w+), %out_flat\[(?P<r>%\w+), (?P<c>%\w+)\] : f32, " + re.escape(f32v) + "$", re.M)
    t, n = st.subn(lambda s: f"{s['i']}{s['v']}_h = scalar.fptrunc {s['v']} : f32 to f16\n"
                             f"{s['i']}view.store {s['v']}_h, %out_flat[{s['r']}, {s['c']}] : f16, {f16v}", t)
    assert n >= 1 and t.count("%out_flat[") == n, f"f16o_o: {old}: {n} of {t.count('%out_flat[')} out stores converted"
    return t


def combine_f16o(k, old, new, pad):
    """g16_attention_combine_listed: each lane's 8 values (head, lane) rounded to f16 at row stride width + pad."""
    t = _rename(k, old, new)
    m = re.findall(r"^  %out_view = buffer\.view %out\[%zero\] : buffer -> view<\[%tokens\]x(\d+)x32x8xf32>\n", t, re.M)
    assert len(m) == 1, f"f16o_o: {old}: out view {m}"
    h = int(m[0]); w = h * 256
    assert (w + pad) % 8 == 0, f"f16o_o: {old}: padded width {w + pad} not a multiple of 8"
    f32v, f16v = f"view<[%tokens]x{h}x32x8xf32>", f"view<[%tokens]x{(w + pad) // 8}x8xf16>"
    t = _sub1(t, f"  %out_view = buffer.view %out[%zero] : buffer -> {f32v}\n", f"""  %out_view = buffer.view %out[%zero] : buffer -> {f16v}
  %o16_c32 = index.constant 32 : index
  %o16_colv = index.madd %head, %o16_c32, %lane : index
  %o16_col = index.assume %o16_colv [range(%o16_colv, 0, {h * 32 - 1})] : index
""", old)
    st = re.compile(r"^(?P<i> *)vector\.store (?P<v>%\w+), %out_view\[%token, %head, %lane, %c0\] : vector<8xf32>, " + re.escape(f32v) + "$", re.M)
    t, n = st.subn(lambda s: f"{s['i']}{s['v']}_h = vector.fptrunc {s['v']} : vector<8xf32> to vector<8xf16>\n"
                             f"{s['i']}vector.store {s['v']}_h, %out_view[%token, %o16_col, %c0] : vector<8xf16>, {f16v}", t)
    assert n >= 1 and t.count("%out_view[") == n, f"f16o_o: {old}: {n} of {t.count('%out_view[')} out stores converted"
    return t


MAKERS = {GDN: gdn_f16o, "g_attention_flash2": flash_f16o, "g_attention_flash2_partial": partial_f16o, "g16_attention_combine_listed": combine_f16o}


def _block_end(lines, i):
    depth = lines[i].count("{") - lines[i].count("}")
    j = i
    while depth > 0:
        j += 1
        depth += lines[j].count("{") - lines[j].count("}")
    return j


def _advance(seg, layers):
    lines = seg.split("\n")
    sites = [i for i, l in enumerate(lines) if SITE.match(l)]
    assert len(sites) == layers, f"f16o_o: {len(sites)} o-sites in @llm_advance, expected {layers}"
    s_o = [i for i, l in enumerate(lines) if re.search(r"%s_o\b", l) and "buffer.alloca" not in l]
    for i in s_o:   # readers: the GEMMs (scales operand, never their output); the only writers: the o-sites
        m = re.match(r"\s*kernel\.launch @(\w+)\[[^\]]*\]\((.*)\) : ", lines[i])
        assert m and (i in sites or (m[1].startswith("g_gemm_") and not m[2].endswith("%s_o"))), \
            f"f16o_o: %s_o used outside the o-sites and GEMM scale operands: {lines[i].strip()[:90]}"
    prev, kinds = 0, []
    for r in sites:
        assert lines[r - 1].strip() == "scf.if %mm {" and lines[r + 1].strip() == "} else {" and \
            re.match(r"\s*kernel\.launch @g_quant_q8\[[^\]]*\]\([^)]*, %o, %o16q\)", lines[r + 2]), \
            f"f16o_o: o-site at line {r}: not in `scf.if %mm {{ row scale }} else {{ g_quant_q8 }}`"
        e = r - 2   # the writer block: `scf.if %gemm { .. } else { .. }` ending just before the site's `scf.if %mm {`
        assert lines[e].strip() == "}", f"f16o_o: o-site at line {r}: no writer block before it"
        h = next((i for i in range(e, prev - 1, -1) if lines[i].strip() == "scf.if %gemm {" and _block_end(lines, i) == e), None)
        assert h is not None, f"f16o_o: o-site at line {r}: the block before it is not `scf.if %gemm`"
        for i in range(prev, h):
            assert not WRITES_O.search(lines[i]), f"f16o_o: writer of %o outside a %gemm block: {lines[i].strip()[:90]}"
        depth, els = 1, None
        for i in range(h + 1, e + 1):
            if lines[i].strip() == "} else {" and depth == 1:
                els = i
                break
            depth += lines[i].count("{") - lines[i].count("}")
        assert els, f"f16o_o: %gemm block at line {h}: no else-branch"
        body = [i for i in range(h + 1, els)]
        writers = [i for i in body if WRITES_O.search(lines[i])]
        names = [LAUNCH_O.match(lines[i])["k"] if LAUNCH_O.match(lines[i]) else None for i in writers]
        kind = "gdn" if names == [GDN] else "attn" if names == list(ATTN) else None
        assert kind, f"f16o_o: %gemm branch at line {h}: writers of %o {[lines[i].strip()[:70] for i in writers]}"
        for i in body:
            assert i in writers or (lines[i].lstrip().startswith("kernel.launch @") and not re.search(r"%o16\b|%s_o\b", lines[i])), \
                f"f16o_o: %gemm branch at line {h}: unexpected line {lines[i].strip()[:90]}"
        for i in writers:
            m = LAUNCH_O.match(lines[i])
            out, ty = ("%o16, %s_o", "buffer, buffer") if kind == "gdn" else ("%o16", "buffer")
            lines[i] = f"{m['ind']}kernel.launch @{m['k']}_f16o[{m['g']}]({m['a']}, {out}) : [{m['gt']}]({m['at']}, {ty})"
        ri = lines[r][:len(lines[r]) - len(lines[r].lstrip())]
        lines[r] = f"{ri}scf.if %f16oo_small {{\n  {lines[r]}\n{ri}}}"
        kinds.append(kind)
        prev = r + 3
    assert kinds[0] == "gdn", "f16o_o: the first o-site is not a GDN site: the attention sites' %s_o would not be written in this epoch"
    seg = "\n".join(lines)
    a = "  %gemm = index.cmp uge, %t, %c64 : index\n"
    assert seg.count(a) == 1, "f16o_o: %gemm definition"
    return seg.replace(a, a + "  %f16oo_small = index.cmp ult, %t, %c64 : index   // F16O_O (E375): the o-site row scale runs off the f16o path\n", 1), kinds


def apply(text, layers):
    if not ON:
        return text
    import sys
    import pf16_pad
    pad = pf16_pad.PAD
    first = text.index("command.def ")
    extra = []
    for k, make in MAKERS.items():
        m = _kernel(text, k)
        assert m, f"f16o_o: kernel {k} not found"
        extra.append(make(m.group(0), k, k + "_f16o", pad))
        tw = _kernel(text, k + "_rdna4")
        if tw:
            extra.append(make(tw.group(0), k + "_rdna4", k + "_f16o_rdna4", pad))
    text = text[:first] + "\n".join(extra) + "\n" + text[first:]   # kernels before the programs (dual_target: a twin before its launches)
    segs = re.split(r"(?m)^(?=command\.def )", text)
    done = 0
    for i, s in enumerate(segs):
        if s.startswith("command.def ") and "@llm_advance()" in s.split("\n", 1)[0]:
            segs[i], kinds = _advance(s, layers)
            done += 1
            print(f"F16O_O: {kinds.count('gdn')} GDN + {kinds.count('attn')} attention o-sites on the f16o producers (t >= 64)", file=sys.stderr)
    assert done == 1, f"f16o_o: {done} @llm_advance programs"
    return "".join(segs)
