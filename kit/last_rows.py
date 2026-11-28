"""LAST_ROWS=1 (HRX-like static-1127/siblings SWEEP.md shape A, top-5 #1; after Gufo prefill_chunk.cpp:189-201): the program's last
layer of @llm_advance computes its post-K/V GEMMs (q projection, o projection, gate/up, down) only on the 256-row tiles that hold a
row the head reads, by zeroing the row scales of every other row right before each of those GEMMs.

Why it is enough: every prefill GEMM (f16 and quantized; m256, m128 and plain) skips a whole row tile whose scale column 0 is zero
on every row (%tile_live), and the f16-output swiglu writes its output flags s_f[row] = (s_h[row] != 0) for every row before that
skip, so the down GEMM after it skips the same tiles. The head reads one row per selected span, the span's last token (flags bit
0; g_head_mv's g_span_output_row); an intermediate prefill chunk selects no span (loom-serve service/text/epoch.c), so all its
scales are zeroed and every tile of those four GEMMs exits at the tile check.

What changes in the last layer (fail closed on any other shape):
- the q projection block and the blocks up to the K projection (the q / gate split, or QG_SPLIT's form of it) move after the V
  projection block, right before g_qk_rope_store: K and V read the same %s_h and need every row (they are stored to the arena);
- one g_scale_head_rows launch under %mm right before each of the q, o, gate/up and down GEMM blocks, on the scale buffer that
  block's GEMMs read (%s_h, %s_o, %s_h, %s_f).

Exact by construction for the head rows: each is computed by the same kernel launch on the same operands (its scale is kept, the
tile holding it stays live, GEMM rows are independent along M); the four GEMMs' other rows become zero or stale, and every reader
of them is row-local up to the head (the allowlist below, shared with LAST_Q), so they reach no row the head reads. g_qk_rope_store
writes the arena from %k / %v only, which are computed before the first mask. The launches run in the command body's serial order
(no command.concurrent: Loom 0.2.0 docs guide/command-programs.md:293-313), so the in-place mask runs after K and V have read %s_h.

Unset: the program is unchanged (the pass is not called). Applied twice: the second application returns the text unchanged; a
half-converted layer (some masks, or the q block not after V) stops the build."""
import re
import sys

import last_q

KERNEL = r"""
// LAST_ROWS (HRX-like static-1127 SWEEP shape A): zero the row scales (both columns) of every row the head does not read, in
// place: the rows past the epoch's tokens and every row that is not the last token of a selected span (g_span_output_row's row).
// The GEMMs reading the scales then skip every 256-row tile without such a row (%tile_live). One lane per row.
kernel.def export("g_scale_head_rows") @g_scale_head_rows(%rows: index) {
  %c1 = index.constant 1 : index
  %c255 = index.constant 255 : index
  %c256 = index.constant 256 : index
  %r255 = index.add %rows, %c255 : index
  %groups = index.div %r255, %c256 : index
  kernel.launch.config workgroups(%groups, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%rows_arg: index, %tokens_arg: index, %spans_arg: index, %metadata: buffer, %info: buffer, %scales: buffer) {
  %rows = index.assume %rows_arg [range(%rows_arg, 1, 8192)] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %zero = index.constant 0 : offset
  %c16o = index.constant 16 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c3 = index.constant 3 : index
  %c256 = index.constant 256 : index
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %no_scale = scalar.constant 0.0 : f32
  %true_i1 = scalar.constant true : i1
  %group = kernel.workgroup.id<x> : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %row_value = index.madd %group, %c256, %lane : index
  %in_rows = index.cmp ult, %row_value, %rows : index
  scf.if %in_rows {
    %row = index.assume %row_value [lt(%row_value, %rows)] : index
    %s_view = buffer.view %scales[%zero] : buffer -> view<[%rows]x2xf32>
    %is_token = index.cmp ult, %row, %tokens : index
    %keep = scf.if %is_token -> (i1) {
      %trow = index.assume %row [lt(%row, %tokens)] : index
      %header = buffer.view %metadata[%zero] : buffer -> view<4xi32>
      %count32 = view.load %header[0] : view<4xi32> -> i32
      %packets = buffer.view %metadata[%c16o] : buffer -> view<[%spans]x9xi32>
      %spans32 = index.cast %spans : index to i32
      %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
      %live = view.load %info_view[%trow, %c2] : view<[%tokens]x4xi32> -> i32
      %span32 = view.load %info_view[%trow, %c3] : view<[%tokens]x4xi32> -> i32
      %is_live = scalar.cmpi ne, %live, %c0_i32 : i32
      %span_low = scalar.cmpi sge, %span32, %c0_i32 : i32
      %span_counted = scalar.cmpi slt, %span32, %count32 : i32
      %span_fits = scalar.cmpi slt, %span32, %spans32 : i32
      %s_a = scalar.andi %span_low, %span_counted : i1
      %span_ok = scalar.andi %s_a, %span_fits : i1
      %span_safe = scf.select %span_ok, %span32, %c0_i32 : i32
      %span_bound = scalar.assume %span_safe [range(%span_safe, 0, 8191)] : i32
      %span_value = index.cast %span_bound : i32 to index
      %span = index.assume %span_value [lt(%span_value, %spans)] : index
      %flags = view.load %packets[%span, 6] : view<[%spans]x9xi32> -> i32
      %length32 = view.load %packets[%span, 0] : view<[%spans]x9xi32> -> i32
      %input32 = view.load %packets[%span, 4] : view<[%spans]x9xi32> -> i32
      %sel_bits = scalar.andi %flags, %c1_i32 : i32
      %selected = scalar.cmpi ne, %sel_bits, %c0_i32 : i32
      %end32 = scalar.addi %input32, %length32 : i32
      %last32 = scalar.subi %end32, %c1_i32 : i32
      %row32 = index.cast %trow : index to i32
      %is_last = scalar.cmpi eq, %row32, %last32 : i32
      %o_a = scalar.andi %span_ok, %selected : i1
      %o_b = scalar.andi %o_a, %is_last : i1
      %o_c = scalar.andi %o_b, %is_live : i1
      scf.yield %o_c : i1
    } else {
      %false = scalar.constant false : i1
      scf.yield %false : i1
    }
    %drop = scalar.xori %keep, %true_i1 : i1
    scf.if %drop {
      view.store %no_scale, %s_view[%row, %c0] : f32, view<[%rows]x2xf32>
      view.store %no_scale, %s_view[%row, %c1] : f32, view<[%rows]x2xf32>
    }
  }
  kernel.return
}
"""

NAME = "g_scale_head_rows"
MASK = ("  scf.if %mm {{\n    kernel.launch @g_scale_head_rows[%tp](%tp, %t, %spans, %metadata, %info, {s}) : "
        "[index](index, index, index, buffer, buffer, buffer)\n  }}\n")
STMT = re.compile(r"(?m)^  (?![ }])")           # a top-level statement of the command body starts at two spaces, not a closing brace
LAUNCH = re.compile(r"(?m)^ *kernel\.launch @(\w+)\[")
GEMM = re.compile(r"(?m)^ *kernel\.launch @g_gemm_\w+\[[^\n]*$")
SCALE = re.compile(r"%(?:h16|o16|f16), (%s_[hof])\b")   # a GEMM reads its row scales right after its f16 rows
# what the moved q region may launch: the weight conversion, the projection (GEMM, decode matvec) and the q / gate split
# kernels after the moved q region: LAST_Q's row-local list plus the f16-output flash kernels (F16O_O), whose output row is a
# function of its own query row and the arena only (LAST_Q checks only what follows them)
AFTER = re.compile(last_q.ROW_LOCAL.pattern + r"|g_attention_flash2(_partial)?_f16o")
Q_REGION = re.compile(r"g_deq16_\w+|g_gemm_\w+|g_mv_\w+|g_split_qgate(_lv)?")


def die(msg):
    sys.exit(f"LAST_ROWS: {msg}")


def _stmts(block):
    """the top-level statements of a layer block, as (start, end) offsets"""
    starts = [m.start() for m in STMT.finditer(block)]
    return [(s, starts[i + 1] if i + 1 < len(starts) else len(block)) for i, s in enumerate(starts)]


def _one(block, st, needle, what):
    hits = [i for i, (a, b) in enumerate(st) if block[a:b].startswith("  scf.if ") and needle in block[a:b]]
    if len(hits) != 1:
        die(f"{what}: {len(hits)} launch blocks read {needle} (expected 1)")
    return hits[0]


def _scale(text, what):
    """the one scale buffer every GEMM launch of a block reads (its first scale operand, after its f16 rows)"""
    gemms = GEMM.findall(text)
    found = {m.group(1) if m else None for m in map(SCALE.search, gemms)}
    if not gemms or len(found) != 1 or None in found:
        die(f"{what}: its GEMM launches read scale buffers {sorted(map(str, found)) or 'none'} (expected one of %s_h / %s_o / %s_f)")
    return found.pop()


def _layer(text):
    """(segment start, segment end, layer, block start, block end) of the program's last layer in @llm_advance"""
    m = re.search(r"(?m)^command\.def [^\n]*@llm_advance\(", text)
    if not m:
        die("no @llm_advance program")
    end = text.find("\ncommand.def ", m.end())
    end = len(text) if end < 0 else end + 1
    seg = text[m.start():end]
    marks = list(last_q.MARK.finditer(seg))
    if not marks:
        die("no layer markers in @llm_advance")
    layer, b0 = int(marks[-1].group(1)), marks[-1].start()
    head = seg.find("\n  %head = command.parameter", b0)
    if head < 0:
        die(f"layer {layer}: no head after the last layer")
    return m.start(), end, layer, m.start() + b0, m.start() + head + 1


def apply(text):
    """The LAST_ROWS rewrite of the generated program text (after LAST_Q, if set). Returns (text, layer index)."""
    s0, s1, L, b0, b1 = _layer(text)
    block, tail = text[b0:b1], text[b1:s1]
    st = _stmts(block)
    seg_masks = text[s0:s1].count(f"@{NAME}[")
    iq, ik, iv = (_one(block, st, f"%{w}_{L},", w) for w in ("wq", "wk", "wv"))
    io, ig, idn = (_one(block, st, f"%{w}_{L},", w) for w in ("wo", "w_gate", "w_down"))
    rope = [i for i, (a, b) in enumerate(st) if block[a:b].startswith("  kernel.launch @g_qk_rope_store[")]
    if len(rope) != 1:
        die(f"layer {L}: {len(rope)} g_qk_rope_store launches (expected 1)")
    ir = rope[0]
    if seg_masks:   # applied before: return it unchanged only if it is the whole conversion, in this layer only
        masks = [i for i, (a, b) in enumerate(st) if f"@{NAME}[" in block[a:b]]
        ok = (seg_masks == 4 and len(masks) == 4 and ik < iv < iq < ir < io < ig < idn and masks[0] == iv + 1 and iq == iv + 2
              and [m + 1 for m in masks] == [iq, io, ig, idn]
              and all(block[st[m][0]:st[m][1]] == MASK.format(s=_scale(block[st[m + 1][0]:st[m + 1][1]], "block")) for m in masks)
              and text.count(f'kernel.def export("{NAME}")') == 1)
        if not ok:
            die(f"layer {L}: half converted ({seg_masks} {NAME} launches in @llm_advance, q block {'after' if iq > iv else 'before'} V)")
        return text, L
    if not iq < ik < iv < ir < io < ig < idn:
        die(f"layer {L}: blocks out of the known order q < k < v < rope < o < gate/up < down ({[iq, ik, iv, ir, io, ig, idn]})")
    if ir != iv + 1:
        die(f"layer {L}: statements between the V projection and g_qk_rope_store")
    # the q region: the q block and what follows it up to the K block (the q / gate split, or its QG_SPLIT form)
    qa, qb = st[iq][0], st[ik][0]
    region = block[qa:qb]
    # the q region moves past the %s_h mask, before the post-flash gate's window: the same gate judges it (statement forms; launches
    # of Q_REGION only, whose buffers are allocations, views of them or weights: no command argument; keeps-review O1, 11-28)
    last_q.gate(text[s0:s1], region, L, "LAST_ROWS", Q_REGION, (), "layer {}'s q region (weight conversion, projection, q / gate split)")
    if re.search(r"%(k|v|wk_\d+|wv_\d+)\b", region):
        die(f"layer {L}: the q region reads or writes the K / V projections")
    kv = block[qb:st[ir][0]]
    if re.search(r"%(q|gate|qg)\b", kv):
        die(f"layer {L}: the K / V blocks read the q projection")
    # every launch after the q region's new place is row-local up to the head (LAST_Q's allowlist), and %s_h is not read between
    # the attention and the FFN norm (the masked q scales must not reach another consumer)
    after = block[st[ir][1]:] + tail
    other = sorted({k for k in LAUNCH.findall(after) if not AFTER.fullmatch(k)})
    if other or re.search(r"(?m)^ *template\.apply", after):
        die(f"layer {L}: launches after g_qk_rope_store that are not known row-local: {other or 'template.apply'}")
    norm = block[st[ig - 1][0]:st[ig - 1][1]]
    if not re.search(r"kernel\.launch @g_(post_)?add_norm_rs_live(_noh)?_lv\[[^\n]*%s_h, %h16\)", norm):
        die(f"layer {L}: the statement before gate/up is not the FFN norm writing %s_h")
    if "%s_h" in block[st[ir][1]:st[ig - 1][0]]:
        die(f"layer {L}: %s_h is read between the attention and the FFN norm")
    scales = {i: _scale(block[st[i][0]:st[i][1]], what) for i, what in ((iq, "q"), (io, "o"), (ig, "gate/up"), (idn, "down"))}
    if [scales[i] for i in (iq, io, ig, idn)] != ["%s_h", "%s_o", "%s_h", "%s_f"]:
        die(f"layer {L}: scale buffers {[scales[i] for i in (iq, io, ig, idn)]} (expected %s_h, %s_o, %s_h, %s_f)")
    out = []
    for i, (a, b) in enumerate(st):
        if iq <= i < ik:
            continue   # moved
        if i in (io, ig, idn):
            out.append(MASK.format(s=scales[i]))
        out.append(block[a:b])
        if i == iv:
            out.append(MASK.format(s="%s_h") + region)
    nb = block[:st[0][0]] + "".join(out)
    if len(nb) != len(block) + 4 * len(MASK.format(s="%s_h")):
        die("internal: the rewritten layer is not the old one plus four masks")
    # LAST_Q's post-flash gate on this pass's output (it runs before the masks exist when LAST_Q is set, and not at all when it is
    # unset): statement forms, row-local launches with their command arguments (the masks: ROW_LOCAL + ARGS_OK), names defined once
    flash = [m for m in re.finditer(r"(?m)^ *kernel\.launch @(\w+)\[[^\n]*\n", nb) if last_q.LIST_READERS[1].fullmatch(m.group(1))]
    if len(flash) != 1:
        die(f"layer {L}: {len(flash)} partial flash launches (expected 1)")
    out = text[:b0] + nb + text[b1:]
    last_q.gate(out[s0:s1 + len(nb) - len(block)], nb[flash[0].end():] + tail, L, "LAST_ROWS")
    return out + "\n" + KERNEL, L
