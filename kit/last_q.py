"""LAST_Q=1 (HRX-like E661 lever 1, after Gufo's prefill_chunk.cpp): the last attention layer of @llm_advance computes flash
attention only for the work items that hold a token the head reads.

After the program's last attention layer nothing but row-local kernels and the head run (checked here, fail closed), and the
head reads one row per span: the span's last token, only when the span is selected (flags bit 0; g_head_mv's
g_span_output_row reading). An intermediate prefill chunk is not selected (loom-serve service/text/epoch.c: select only when a
chunk consumes the rest of the input), so its last-layer attention output is never read; the final chunk's is read at one row.
g_flash_tiles_out filters g_attention_fallback_list's flash work list (whole 16-token tiles from slot 1, count at 0; partial
blocks from slot tokens/2 + 1, count at tokens/2) to the tiles / blocks that hold such a token, in the same layout, and the last
attention layer's g_attention_flash2 + g_attention_flash2_partial launches read it instead of %flash_tiles.

Exact: a flash workgroup's result depends on its list entry (the tile's first token) and the keys, never on its slot (the
workgroup id only picks the entry), so a kept tile is computed as before and the head's rows are bit-identical. The tiles left
out keep stale rows in the attention output; every kernel after them is row-local (the allowlist below), so those rows reach
no row the head reads. The listed split-K fallback (runs shorter than 9 tokens: decode tokens) is unchanged. K/V of the layer
are written by g_qk_rope_store before, for every token, as before.

Unset: the program is unchanged (the pass is not called)."""
import collections
import re
import sys

KERNEL = r"""
// LAST_Q (HRX-like E661): the flash work list of the program's last attention layer, which feeds only the head: the whole
// tiles and partial blocks of g_attention_fallback_list's list that hold the last token of a selected span (the row the head
// reads), in the same layout (whole count at 0, entries from 1, token 0 up to slot tokens/16; partial count at tokens/2,
// entries after it). One workgroup; the order of the kept entries does not matter (each is computed on its own).
kernel.def export("g_flash_tiles_out") @g_flash_tiles_out(%tokens: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %spans_arg: index, %metadata: buffer, %info: buffer, %tiles: buffer, %out: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 16, 8192), multiple_of(%tokens_arg, 16)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c16 = index.constant 16 : index
  %c256 = index.constant 256 : index
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %scratch_bytes = index.constant 16 : offset
  %scratch = buffer.alloca<workgroup> align(16) %scratch_bytes : buffer
  %scratch_view = buffer.view %scratch[%zero] : buffer -> view<4xi32>
  %first = index.cmp eq, %lane, %c0 : index
  scf.if %first {
    view.store %c0_i32, %scratch_view[%c0] : i32, view<4xi32>
    view.store %c0_i32, %scratch_view[%c1] : i32, view<4xi32>
  }
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %tiles_view = buffer.view %tiles[%zero] : buffer -> view<[%tokens]xi32>
  %out_view = buffer.view %out[%zero] : buffer -> view<[%tokens]xi32>
  %half = index.div %tokens, %c2 : index
  %half_slot = index.assume %half [lt(%half, %tokens)] : index
  %first_part = index.add %half, %c1 : index
  // whole tiles
  %wcount32 = view.load %tiles_view[%c0] : view<[%tokens]xi32> -> i32
  %wcount_bound = scalar.assume %wcount32 [range(%wcount32, 0, 512)] : i32
  %wcount = index.cast %wcount_bound : i32 to index
  scf.for %e = [%lane to %wcount step %c256] {
    %wslot_value = index.add %e, %c1 : index
    %wslot = index.assume %wslot_value [range(%wslot_value, 1, 512), lt(%wslot_value, %tokens)] : index
    %wentry32 = view.load %tiles_view[%wslot] : view<[%tokens]xi32> -> i32
    %wentry_bound = scalar.assume %wentry32 [range(%wentry32, 0, 8191)] : i32
    %wentry_value = index.cast %wentry_bound : i32 to index
    %wentry = index.assume %wentry_value [lt(%wentry_value, %tokens)] : index
    %wkeep = func.call @g_block_holds_output(%metadata, %spans, %info, %tokens, %wentry) : (buffer, index, buffer, index, index) -> (i1)
    scf.if %wkeep {
      %wn32 = view.atomic.rmw<addi> %c1_i32, %scratch_view[%c0] {ordering = relaxed, scope = workgroup} : i32, view<4xi32> -> i32
      %wn_bound = scalar.assume %wn32 [range(%wn32, 0, 511)] : i32
      %wn = index.cast %wn_bound : i32 to index
      %wdst_value = index.add %wn, %c1 : index
      %wdst = index.assume %wdst_value [range(%wdst_value, 1, 512), lt(%wdst_value, %tokens)] : index
      view.store %wentry32, %out_view[%wdst] : i32, view<[%tokens]xi32>
    }
  }
  // partial blocks
  %pcount32 = view.load %tiles_view[%half_slot] : view<[%tokens]xi32> -> i32
  %pcount_bound = scalar.assume %pcount32 [range(%pcount32, 0, 4095)] : i32
  %pcount = index.cast %pcount_bound : i32 to index
  scf.for %p = [%lane to %pcount step %c256] {
    %pslot_value = index.add %first_part, %p : index
    %pslot = index.assume %pslot_value [lt(%pslot_value, %tokens)] : index
    %pentry32 = view.load %tiles_view[%pslot] : view<[%tokens]xi32> -> i32
    %pentry_bound = scalar.assume %pentry32 [range(%pentry32, 0, 8191)] : i32
    %pentry_value = index.cast %pentry_bound : i32 to index
    %pentry = index.assume %pentry_value [lt(%pentry_value, %tokens)] : index
    %pkeep = func.call @g_block_holds_output(%metadata, %spans, %info, %tokens, %pentry) : (buffer, index, buffer, index, index) -> (i1)
    scf.if %pkeep {
      %pn32 = view.atomic.rmw<addi> %c1_i32, %scratch_view[%c1] {ordering = relaxed, scope = workgroup} : i32, view<4xi32> -> i32
      %pn_bound = scalar.assume %pn32 [range(%pn32, 0, 4094)] : i32
      %pn = index.cast %pn_bound : i32 to index
      %pdst_value = index.add %first_part, %pn : index
      %pdst = index.assume %pdst_value [lt(%pdst_value, %tokens)] : index
      view.store %pentry32, %out_view[%pdst] : i32, view<[%tokens]xi32>
    }
  }
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %kept32 = view.load %scratch_view[%c0] : view<4xi32> -> i32
  scf.if %first {
    %pkept32 = view.load %scratch_view[%c1] : view<4xi32> -> i32
    view.store %kept32, %out_view[%c0] : i32, view<[%tokens]xi32>
    view.store %pkept32, %out_view[%half_slot] : i32, view<[%tokens]xi32>
  }
  // g_attention_flash2 reads the entry of every slot up to tokens/16 before it checks the count: token 0 past the count.
  %kept_bound = scalar.assume %kept32 [range(%kept32, 0, 512)] : i32
  %kept = index.cast %kept_bound : i32 to index
  %slots = index.div %tokens, %c16 : index
  %fill_from = index.add %kept, %lane : index
  scf.for %f = [%fill_from to %slots step %c256] {
    %fslot_value = index.add %f, %c1 : index
    %fslot = index.assume %fslot_value [range(%fslot_value, 1, 512), lt(%fslot_value, %tokens)] : index
    view.store %c0_i32, %out_view[%fslot] : i32, view<[%tokens]xi32>
  }
  kernel.return
}

// Whether the flash work item starting at token ENTRY (a whole tile or a partial block) holds the last token of a selected
// span. Its tokens: ENTRY, then each next token while it continues ENTRY's run (same row, the next position, live) inside
// ENTRY's 16-position block, as g_attention_flash2_partial bounds its block; a whole tile is 16 such tokens.
func.def inline @g_block_holds_output(%metadata: buffer, %spans: index, %info: buffer, %tokens: index, %entry: index) -> (i1) {
  %zero = index.constant 0 : offset
  %c16o = index.constant 16 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c3 = index.constant 3 : index
  %c16 = index.constant 16 : index
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %c15_i32 = scalar.constant 15 : i32
  %c16_i32 = scalar.constant 16 : i32
  %true = scalar.constant true : i1
  %false = scalar.constant false : i1
  %header = buffer.view %metadata[%zero] : buffer -> view<4xi32>
  %count32 = view.load %header[0] : view<4xi32> -> i32
  %packets = buffer.view %metadata[%c16o] : buffer -> view<[%spans]x9xi32>
  %spans32 = index.cast %spans : index to i32
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %bentry = index.assume %entry [lt(%entry, %tokens)] : index
  %row0 = view.load %info_view[%bentry, %c0] : view<[%tokens]x4xi32> -> i32
  %pos0 = view.load %info_view[%bentry, %c1] : view<[%tokens]x4xi32> -> i32
  %skew = scalar.andi %pos0, %c15_i32 : i32
  %block0 = scalar.subi %pos0, %skew : i32
  %block_end = scalar.addi %block0, %c16_i32 : i32
  %alive_end, %holds = scf.for %k = [%c0 to %c16 step %c1](%alive = %true : i1, %acc = %false : i1) -> (i1, i1) {
    %at_value = index.add %entry, %k : index
    %room = index.cmp ult, %at_value, %tokens : index
    %at_clamped = scf.select %room, %at_value, %bentry : index
    %at = index.assume %at_clamped [lt(%at_clamped, %tokens)] : index
    %row = view.load %info_view[%at, %c0] : view<[%tokens]x4xi32> -> i32
    %pos = view.load %info_view[%at, %c1] : view<[%tokens]x4xi32> -> i32
    %live = view.load %info_view[%at, %c2] : view<[%tokens]x4xi32> -> i32
    %span32 = view.load %info_view[%at, %c3] : view<[%tokens]x4xi32> -> i32
    %k32 = index.cast %k : index to i32
    %want = scalar.addi %pos0, %k32 : i32
    %same_row = scalar.cmpi eq, %row, %row0 : i32
    %at_pos = scalar.cmpi eq, %pos, %want : i32
    %is_live = scalar.cmpi ne, %live, %c0_i32 : i32
    %in_block = scalar.cmpi slt, %want, %block_end : i32
    %m_a = scalar.andi %same_row, %at_pos : i1
    %m_b = scalar.andi %m_a, %is_live : i1
    %m_c = scalar.andi %m_b, %in_block : i1
    %m_d = scalar.andi %m_c, %room : i1
    %next_alive = scalar.andi %alive, %m_d : i1
    // the head's row of the token's span (g_span_output_row): selected, and this token is its last
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
    %at32 = index.cast %at : index to i32
    %is_last = scalar.cmpi eq, %at32, %last32 : i32
    %o_a = scalar.andi %span_ok, %selected : i1
    %o_b = scalar.andi %o_a, %is_last : i1
    %out_here = scalar.andi %o_b, %next_alive : i1
    %next_acc = scalar.ori %acc, %out_here : i1
    scf.yield %next_alive, %next_acc : i1, i1
  }
  func.return %holds : i1
}
"""

# Kernels that may run after the last attention layer's flash launches: each is row-local (a token's output row depends only on
# that token's input rows, or on per-span head rows), so stale attention rows of the tiles left out reach no row the head reads.
# Any other kernel (a DFlash tap, a drafter's feature write, a later layer) stops the build: fail closed (rules section 4).
ROW_LOCAL = re.compile(r"g16_attention_split_listed|g16_attention_combine_listed(_f16o)?|g_attention_split|g_attention_vsplit"
                       r"|g_attention_combine|g_attention_flash2(_partial)?|g_row_scale_f16(_live_lv)?|g_quant_q8|g_deq16_\w+"
                       r"|g_gemm_\w+|g_mv_\w+|g_add_norm_rs_live(_noh)?_lv|g_post_add_norm(_rs_live(_noh)?_lv|_lv)?"
                       r"|g_head_mv(_t\d+)?(_q8)?|g_head_select|g_logits_export"
                       r"|g_scale_head_rows")   # LAST_ROWS (kit/last_rows.py): zeroes the scales of rows the head does not read, per row
# The two kernels that read the list in the layout g_flash_tiles_out writes ([0] / [tokens/2] counts, the workgroup id picks the
# entry only): one launch of each in the layer, as the served program has them (F16O_O: their f16-output forms).
LIST_READERS = (re.compile(r"g_attention_flash2(_f16o)?"), re.compile(r"g_attention_flash2_partial(_f16o)?"))
PROLOGUE = ("    kernel.launch @g_attention_fallback_list[%t](%t, %info, %flash_tiles, %fallback_list, %fallback_count) : "
            "[index](index, buffer, buffer, buffer, buffer)\n")
ALLOC = "  %flash_tiles = buffer.alloca<global> align(256) %list_size : buffer\n"
CROP = ("    kernel.launch @g_flash_tiles_out[%t](%t, %spans, %metadata, %info, %flash_tiles, %flash_tiles_out) : "
        "[index](index, index, buffer, buffer, buffer, buffer)\n")
LAUNCH = re.compile(r"(?m)^ *kernel\.launch @(\w+)\[")
# Every other statement after those launches must be one of these forms, each one the served program uses there (keeps-review
# 11-27: the launch-name list let a buffer.copy into %features and a func.call through). None of them writes memory: values
# (index), control (scf.if, its braces), a new buffer or a view of one (a write happens only through an admitted launch), the
# head's weights, config reads and declarations, the return. Anything else (template.apply, buffer.copy, func.call, scf.for, ...)
# stops the build.
AFTER_OK = re.compile(r"|//[^\n]*|\}|\} else \{|scf\.if %\w+ \{|command\.return|config\.decl @[^\n]*"
                      r"|%\w+ = (?:index\.(?:constant|cmp|scale)|buffer\.alloca<global>|buffer\.view|command\.parameter|config\.get) [^\n]*")
LAUNCH_LINE = re.compile(r"kernel\.launch @(\w+)\[[^\n]*")
LAUNCH_SIG = re.compile(r"kernel\.launch @(\w+)\[[^\]]*\]\((.*)\) : (?:\[[^\]]*\])?\((.*)\)")
SCALAR = re.compile(r"index|[if]\d+")
DEF = re.compile(r"(?m)^ *(%\w+) = ([\w.]+)(?:<[^>\n]*>)?(?: (%\w+))?")
# The command arguments of @llm_advance a launch after those flash launches may take, per kernel (keeps-review r1 arg_census.txt,
# E722: the served program's only uses). Any other use of a command argument there stops the build: a row-local kernel writing
# %features, %arena or %outputs would carry the cropped tiles' stale rows out of the program. Every other buffer operand must be
# an allocation of the program, a weight (command.parameter of %weights) or a view of an allocation.
ARGS_OK = ((re.compile(r"g16_attention_split_listed|g_attention_split|g_attention_vsplit"), {"%arena", "%origins"}),
           (re.compile(r"g_head_mv(_t\d+)?(_q8)?"), {"%metadata"}),
           (re.compile(r"g_head_select|g_logits_export"), {"%metadata", "%outputs"}),
           (re.compile(r"g_scale_head_rows"), {"%metadata"}))   # LAST_ROWS's masks: the span packets, read only


def _launch_ok(s, cargs, defs, kernels=ROW_LOCAL, args_ok=ARGS_OK):
    """a launch of KERNELS whose command arguments are its ARGS_OK entry and whose other buffers are local (see ARGS_OK)"""
    m = LAUNCH_SIG.fullmatch(s)
    if not m or not kernels.fullmatch(m.group(1)):
        return False
    ops, tys = [x.strip() for x in m.group(2).split(",")], [x.strip() for x in m.group(3).split(",")]
    if len(ops) != len(tys):
        return False
    allowed = set().union(*[a for r, a in args_ok if r.fullmatch(m.group(1))])
    for o, t in zip(ops, tys):
        if o in cargs:
            if o not in allowed:
                return False
        elif not SCALAR.fullmatch(t):
            op, src = defs.get(o, (None, None))
            if not (op == "buffer.alloca" or (op == "command.parameter" and src == "%weights")
                    or (op == "buffer.view" and defs.get(src, (None,))[0] == "buffer.alloca")):
                return False
    return True
MARK = re.compile(r"(?m)^  // ---- layer (\d+)[^\n]*$")


def die(msg):
    sys.exit(f"LAST_Q: {msg}")


def apply(text):
    """The LAST_Q rewrite of the generated program text (after the partial-kernel launches are inserted). Returns
    (text, layer index). Stops the build on any shape it does not know."""
    m = re.search(r"(?m)^command\.def [^\n]*@llm_advance\(", text)
    if not m:
        die("no @llm_advance program")
    end = text.find("\ncommand.def ", m.end())
    end = len(text) if end < 0 else end + 1
    seg = text[m.start():end]
    marks = list(MARK.finditer(seg))
    if not marks:
        die("no layer markers in @llm_advance")
    blocks = [(int(x.group(1)), x.start(), marks[i + 1].start() if i + 1 < len(marks) else len(seg)) for i, x in enumerate(marks)]
    attn = [b for b in blocks if any(LIST_READERS[0].fullmatch(k) for k in LAUNCH.findall(seg[b[1]:b[2]]))]
    if not attn:
        die("no flash attention layer in @llm_advance")
    layer, b0, b1 = attn[-1]
    if (layer, b0, b1) != blocks[-1]:
        die(f"layer {layer}, the last attention layer, is not the program's last layer (layer {blocks[-1][0]} reads its rows)")
    block = seg[b0:b1]
    readers = [ln for ln in block.splitlines(True) if "%flash_tiles," in ln]
    names = [LAUNCH.match(ln).group(1) if LAUNCH.match(ln) else None for ln in readers]
    if len(names) != 2 or None in names or not all(r.fullmatch(n) for r, n in zip(LIST_READERS, names)):
        die(f"layer {layer}: the flash list readers are {names}, expected g_attention_flash2[_f16o] then its partial kernel")
    gate(seg, block[block.index(readers[-1]) + len(readers[-1]):] + seg[b1:], layer)
    if seg.count(PROLOGUE) != 1 or seg.count(ALLOC) != 1:
        die(f"@llm_advance has {seg.count(PROLOGUE)} work-list launches and {seg.count(ALLOC)} list allocations (expected 1, 1)")
    for ln in readers:
        block = block.replace(ln, ln.replace("%flash_tiles,", "%flash_tiles_out,", 1), 1)
    seg = seg[:b0] + block + seg[b1:]
    seg = seg.replace(ALLOC, ALLOC + ALLOC.replace("%flash_tiles ", "%flash_tiles_out "), 1).replace(PROLOGUE, PROLOGUE + CROP, 1)
    return text[:m.start()] + seg + text[end:] + "\n" + KERNEL, layer


def gate(seg, after, layer, who="LAST_Q", kernels=ROW_LOCAL, args_ok=ARGS_OK, where="after layer {}'s flash launches"):
    """Stops the build unless every statement of AFTER (the text of @llm_advance segment SEG after the last attention layer's flash
    launches) is a known memory-free form or a ROW_LOCAL launch with its ARGS_OK command arguments, and SEG defines each name
    once. LAST_Q judges the program before its rewrite; LAST_ROWS (applied after it) judges its own output, masks included, and
    its moved q region (KERNELS: that region's kernels, ARGS_OK empty: no command argument; WHERE names the region, {} its layer)."""
    def stop(msg):
        sys.exit(f"{who}: {msg}")
    cargs = set(re.findall(r"(%\w+): buffer", seg.split("\n", 1)[0]))
    defs = {}
    for d in DEF.finditer(seg):
        defs.setdefault(d.group(1), (d.group(2), d.group(3)))
    # each name defined once in @llm_advance, so an operand's definition is the one read above (keeps-review r2, E726: a nested scope
    # may shadow a name; the served program defines each of its names once)
    names = collections.Counter(n for d in re.finditer(r"(?m)^ *(%\w+(?:, *%\w+)*) = ", seg) for n in re.split(r", *", d.group(1)))
    twice = sorted(n for n, k in names.items() if k > 1)
    if twice:
        stop(f"@llm_advance defines {len(twice)} names more than once: {twice[:4]}")
    other = []
    for ln in after.splitlines():
        s = ln.strip()
        if LAUNCH_LINE.fullmatch(s):
            ok = _launch_ok(s, cargs, defs, kernels, args_ok)
        else:
            used = set(re.findall(r"%\w+", s)) & cargs if not s.startswith("//") else set()
            ok = AFTER_OK.fullmatch(s) and (not used or (used == {"%weights"} and s.split(" = ", 1)[-1].startswith("command.parameter %weights,")))
        if not ok:
            other.append(s[:100])
    if other:
        stop(f"{where.format(layer)}: {len(other)} statements not known row-local or memory-free: {other[:4]}")
