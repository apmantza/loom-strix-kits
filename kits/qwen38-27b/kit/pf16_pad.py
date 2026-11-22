"""PF16_PAD=<halves> (HRX-like E167): pad the row stride of every f16 GEMM operand by PAD halves, so no row stride is a
4 KB multiple (K 6144 -> 12,288 B camps on the memory channels: 23 TFLOPS vs ~37 padded; K 5120 34.6-35.0 -> 37-38 at 5184;
Strata src/prefill/gemm.cu:1383-1385). Bitwise the same products: only addresses change.

Writers and readers (the closed set, listed from the served program pkg-u60x, 11-21):
  activations (f16 copies %h16 / %o16 / %f16 / drafter %hc16 / %features16):
    writers  g_row_scale_f16[_live[_lv]] (view [rows]x[k4]x4xf16), g_post_add_norm_rs[_live[_lv]] (view [tokens]xHGx32xf16)
    readers  every g_gemm_* kernel (view [rows]x[k]xf16)                      -> stride k + PAD
  weights (the PREFILL_F16_SCRATCH f16 copy):
    writer   g_deq16_<fmt> (%elem = row * k + col)                              -> row * (k + PAD) + col
    readers  g_gemm_f16_* (%blocks = k / 32, row_bytes = blocks * 64)          -> (k + PAD) / 32
  the quantized GEMMs keep their GGUF weight rows; only their X view changes.
Allocation sizes (gen_llm: half_row / half_wide / half_ffn, *16s_bytes, PF16 fbytes / PF16_MAX; gen_dflash's drafter copies)
are padded at their source with pad_bytes(). apply() fails the build when a kernel of these families is left unconverted."""
import os
import re

PAD = int(os.environ.get("PF16_PAD", "0") or 0)
assert PAD % 64 == 0 and PAD >= 0, "PF16_PAD: a multiple of 64 halves (whole 32-value f16 blocks, 16-byte aligned rows)"


def pad_halves(width):
    """the padded row length in halves of an f16 GEMM operand of `width` columns"""
    return width + PAD


def _items(text):
    return re.split(r"(?m)^(?=[a-z])", text)


def _gemm(k):
    n = 0
    old_def = "  %blocks = index.div %k, "
    assert k.count(old_def) == 1, "gemm: expected one %blocks = index.div %k"
    k = k.replace(old_def, f"  %pfpad = index.constant {PAD} : index\n  %kx = index.add %k, %pfpad : index\n" + old_def, 1)
    xt = "view<[%rows]x[%k]xf16>"
    n = k.count(xt)
    assert n >= 2, f"gemm: X view type seen {n} times"
    k = k.replace(xt, "view<[%rows]x[%kx]xf16>")
    return k


def _gemm_f16_w(k):
    old = "  %blocks = index.div %k, %c32 : index\n"
    assert k.count(old) == 1, "f16 gemm: expected one %blocks = index.div %k, %c32"
    return k.replace(old, "  %blocks = index.div %kx, %c32 : index\n", 1)


def _deq16(k):
    old = "    %elem = index.madd %w_out_row, %k, %w_col : index\n"
    assert k.count(old) == 1, "deq16: expected one %elem"
    return k.replace(old, f"    %pfpad = index.constant {PAD} : index\n    %kx = index.add %k, %pfpad : index\n"
                          "    %elem = index.madd %w_out_row, %kx, %w_col : index\n", 1)


def _row_scale(k):
    old = "  %h4_view = buffer.view %x16[%zero] : buffer -> view<[%rows]x[%k4]x4xf16>\n"
    assert k.count(old) == 1, "row_scale_f16: expected one %h4_view"
    k = k.replace(old, f"  %pfpad4 = index.constant {PAD // 4} : index\n  %k4x = index.add %k4, %pfpad4 : index\n"
                       "  %h4_view = buffer.view %x16[%zero] : buffer -> view<[%rows]x[%k4x]x4xf16>\n", 1)
    t = "view<[%rows]x[%k4]x4xf16>"
    assert k.count(t) >= 1, "row_scale_f16: store type"
    return k.replace(t, "view<[%rows]x[%k4x]x4xf16>")


def _post_add(k):
    m = re.findall(r"view<\[%tokens\]x(\d+)x32xf16>", k)
    assert m and len(set(m)) == 1, f"post_add_norm_rs: h16 view types {m}"
    g = int(m[0])
    return k.replace(f"view<[%tokens]x{g}x32xf16>", f"view<[%tokens]x{g + PAD // 32}x32xf16>")


def _audit(text):
    """Every writer of an f16 copy must fit its padded rows into the copy's allocation, in every command program: row_scale's
    width argument (an index constant) or post_add_norm_rs's (the H of its h16 view), + PAD, x 2 bytes <= the allocation's per-row
    constant. Fails closed on a writer whose width or destination size it cannot resolve (E170)."""
    hg = {}
    for it in _items(text):
        m = re.match(r'kernel\.def (?:target\(@\w+\) )?export\("(g_post_add_norm_rs\w*)"\)', it)
        if m:
            g = re.search(r"view<\[%tokens\]x(\d+)x32xf16>", it)
            hg[m.group(1)] = int(g.group(1)) * 32 - PAD if g else None
    checked = 0
    for seg in re.split(r"(?m)^(?=command\.def )", text):
        def const(name, kind):
            m = re.search(r"^  " + re.escape(name) + r" = index\.constant (\d+) : " + kind + "$", seg, re.M)
            return int(m.group(1)) if m else None
        for m in re.finditer(r"kernel\.launch @(g_row_scale_f16\w*|g_post_add_norm_rs\w*)\[[^\]]*\]\(([^)]*)\)", seg):
            k, args = m.group(1), [a.strip() for a in m.group(2).split(",")]
            dst = args[-1]
            if k.startswith("g_row_scale_f16"):
                if k.endswith("_tiles"):
                    continue
                w = const(args[-4], "index") if args[-4].startswith("%") else int(args[-4])
            else:
                w = hg.get(k)
            a = re.search(r"^  " + re.escape(dst) + r" = buffer\.alloca<global> align\(\d+\) (%\w+) : buffer$", seg, re.M)
            assert w and a, f"PF16_PAD audit: {k} -> {dst}: width {w} / allocation unresolved"
            s = re.search(r"^  " + re.escape(a.group(1)) + r" = index\.scale %\w+, (%\w+) : index, offset -> offset$", seg, re.M)
            per_row = const(s.group(1), "offset") if s else None
            assert per_row and per_row >= (w + PAD) * 2, f"PF16_PAD audit: {k} writes {dst} rows of {(w + PAD) * 2} B into {per_row} B rows"
            checked += 1
    assert checked, "PF16_PAD audit: no f16 writer found"
    return checked


RULES = [(r"g_gemm_f16_\w+", lambda k: _gemm_f16_w(_gemm(k))), (r"g_gemm_\w+", _gemm), (r"g_deq16_\w+", _deq16),
         (r"g_row_scale_f16(_live)?(_lv)?", _row_scale), (r"g_post_add_norm_rs(_live)?(_lv)?", _post_add)]


ALLOC = re.compile(r"^  (%\w+16) = buffer\.alloca<global> align\(\d+\) (%\w+) : buffer$", re.M)


def _sizes(text):
    """Pad the per-row byte constant of every f16 activation copy (%h16, %o16, %f16, %hc16, %features16, %hs16, ... in every
    command program), found through its allocation: alloca(size), size = index.scale %t|%tp, CONST, CONST = index.constant.
    Fails closed: an f16 copy whose size cannot be traced to a per-row constant stops the build. (E170: a list of known constant
    names missed the drafter programs' %h16_bytes / %o16_bytes / %f16_bytes; the padded writers overran %h16 into %h16q and the
    drafter accepted 0 drafts at 8k+ while the target stayed exact.)"""
    out, n = [], 0
    for seg in re.split(r"(?m)^(?=command\.def )", text):
        consts = set()
        for buf, size in ALLOC.findall(seg):
            if buf in ("%pf16a", "%pf16b"):   # the weight scratch: sized at its source (PF16_MAX)
                continue
            m = re.search(r"^  " + re.escape(size) + r" = index\.scale %(?:t|tp|ts), (%\w+) : index, offset -> offset$", seg, re.M)
            assert m, f"PF16_PAD: {buf}: size {size} is not index.scale of the token count by a per-row constant"
            assert re.search(r"^  " + re.escape(m.group(1)) + r" = index\.constant \d+ : offset$", seg, re.M), f"PF16_PAD: {buf}: {m.group(1)} not a constant"
            consts.add(m.group(1))
        for c in sorted(consts):
            seg, k = re.subn(r"^  " + re.escape(c) + r" = index\.constant (\d+) : offset$",
                             lambda mm: f"  {c} = index.constant {int(mm.group(1)) + 2 * PAD} : offset", seg, count=1, flags=re.M)
            n += k
        out.append(seg)
    assert n >= 3, f"PF16_PAD: {n} activation size constants padded"
    return "".join(out)


def apply(text):
    """the program with every listed kernel converted; fails when PAD is set and a family member stays unconverted"""
    if not PAD:
        return text
    text = _sizes(text)
    out, done = [], {}
    for it in _items(text):
        m = re.match(r'kernel\.def (?:target\(@\w+\) )?export\("(\w+)"\)', it)
        if m:
            name = m.group(1)
            for pat, f in RULES:
                if re.fullmatch(pat, name):
                    it = f(it)
                    done[name] = pat
                    break
            else:
                assert not re.match(r"g_(gemm|deq16|row_scale_f16|post_add_norm_rs)", name), f"PF16_PAD: no rule for {name}"
        out.append(it)
    assert any(p.startswith("g_gemm_f16") for p in done.values()), "PF16_PAD: no f16 GEMM converted"
    out = "".join(out)
    _audit(out)
    return out
