"""QG_SPLIT=1 (HRX-like static-1127/upsw): the fused query+gate projection stores q and gate from its 256-row f16 GEMM's epilogue, so
g_split_qgate does not run in the epochs that took that GEMM. Ported from upstream loom-skills (gen_llm.qg_split /
gen_gemm.qg_split at f7ad9e6, choice prefill.fuse.qg_split, PR 1323), adapted to our f16-scratch path the way EPI_FUSE was.

The q GEMM writes %qg as [rows][heads][q | gate][D] f32; g_split_qgate(_lv) copies q and gate into %q and %gate
([rows][heads * D] each). g_gemm_f16_store_m256_qgs is the store kernel with one epilogue change: its 128-column output tile
(inside one D-wide half, D a multiple of 128) is stored to %q or %gate at [row, head * D + dim] with row stride n / 2. The same
f32 values land at the same places: exact by construction (the kernel test compares it with store + split bit for bit).

Our projection launches m256 kernels in both the f16-scratch branch (`scf.if %pf_big`) and the quantized branch, so the split is
skipped only under %qgs_ok = the AND of PATH's conditions (%mm, %pf_big, %big, %huge), where the one converted f16 m256 store is the
only %qg writer that can run. A layer is converted only when, on that path, every %qg writer is one f16 m256 store run exactly under
%mm / %pf_big / %big / %huge, and the only %qg readers in @llm_advance are the split launches (exact names). %qg is written only by
the WRITER launches, as their last operand; any other launch naming %qg is a reader and leaves every layer unchanged (counted).
The build stops on a split launch outside a block SPLIT matches, on a launch naming %qg after the last split block, and on any
statement other than a launch or its allocation that names %qg (fail closed, keeps-review 11-27 r1). Off: the text is unchanged."""
import os
import re
import sys

ON = os.environ.get("QG_SPLIT", "0").strip().lower() not in ("", "0", "false", "no", "off")
PATH = {"%mm": "then", "%pf_big": "then", "%big": "then", "%huge": "then"}   # the epoch the split is skipped in
LAUNCH = re.compile(r"^(?P<ind>\s*)kernel\.launch @(?P<k>\w+)\[(?P<g>[^\]]*)\]\((?P<args>.*)\) : (?P<ty>.*)$")
SPLIT = re.compile(r"(?m)^  scf\.if %gemm \{\n\s*kernel\.launch @g_split_qgate_lv\[[^\n]*\n  \} else \{\n\s*kernel\.launch @g_split_qgate\[[^\n]*\n  \}\n")
SPLITS = ("g_split_qgate", "g_split_qgate_lv")
# the launches that write %qg (as their last operand) in the served program (pkg-keeps-unset @llm_advance): the q projection's GEMM
# stores and decode matvecs in every format the kit's quantizations put there
WRITER = re.compile(r"g_gemm_(?:f16|iq4_xs|q4_k|q5_k)_store(?:_m128|_m256)?|g_mv_(?:iq4_xs|q4_k|q5_k)_k\d+(?:_t\d+)?_q8(?:_r\d+)?")
QG_ALLOC = re.compile(r"  %qg = buffer\.alloca<global> [^\n]*")
KDEF = re.compile(r'(?m)^kernel\.def (?:target\(@\w+\) )?export\("(g_gemm_f16_store_m256(?:_rdna4)?)"\)')
from epi_fuse import _advance, _stacks, _reachable, _operands, kernel_text   # noqa: E402  (the same block reader as EPI_FUSE)


def program(text):
    """the @llm_advance rewrite. Returns (text, layers converted, layers refused)."""
    a, b = _advance(text)
    body = text[a:b]
    splits = list(SPLIT.finditer(body))
    lines_all = body.split("\n")
    nsplit = sum(1 for l in lines_all if LAUNCH.match(l) and LAUNCH.match(l).group("k") in SPLITS)
    if nsplit != 2 * len(splits):   # its layer's store would share the next layer's segment while its split reads %qg
        raise SystemExit(f"QG_SPLIT: {nsplit} split launches, {len(splits)} split blocks in the expected form (2 launches each)")
    odd = [l.strip() for l in lines_all if "%qg" in re.findall(r"%\w+", l) and not l.strip().startswith("//")
           and not LAUNCH.match(l) and not QG_ALLOC.fullmatch(l)]
    if odd:
        raise SystemExit(f"QG_SPLIT: %qg used by a statement that is not a launch: {odd[:2]}")
    tail = body[splits[-1].end():] if splits else ""
    late = [LAUNCH.match(l).group("k") for l in tail.split("\n") if LAUNCH.match(l) and "%qg" in _operands(LAUNCH.match(l).group("args"))]
    if late:   # after the last split block nothing may touch %qg: in a skipped epoch it holds the previous epoch's values
        raise SystemExit(f"QG_SPLIT: launches naming %qg after the last split block: {late}")
    for l in lines_all:
        lm = LAUNCH.match(l)
        ops = _operands(lm.group("args")) if lm else []
        if "%qg" not in ops or lm.group("k") in SPLITS:
            continue
        if not (WRITER.fullmatch(lm.group("k")) and ops[-1] == "%qg" and "%qg" not in ops[:-1]):
            return text, 0, len(splits)   # another %qg reader: leave every layer as it is
    out, pos, n, refused = [], 0, 0, 0
    for m in splits:
        seg = body[pos:m.start()]
        lines = seg.split("\n")
        st = _stacks(lines)
        conv, ok = [], True
        for j, line in enumerate(lines):
            lm = LAUNCH.match(line)
            if not lm or not _reachable(st[j], PATH):
                continue
            ops = _operands(lm.group("args"))
            if "%qg" not in ops:
                continue
            conds = {c: br for c, br in st[j]}
            exact = all(conds.get(c) == "then" for c in PATH) and set(conds) <= set(PATH)
            if lm.group("k") == "g_gemm_f16_store_m256" and exact:   # a writer: %qg is its last operand only (checked above)
                conv.append((j, lm))
            else:
                ok = False
        if ok and len(conv) == 1:
            j, lm = conv[0]
            ops = _operands(lm.group("args"))
            ty = lm.group("ty")   # %qg (the store's last operand, a buffer) becomes %q, %gate: the _qgs launch ABI adds %y2
            assert ty.endswith(", buffer)"), f"QG_SPLIT: launch type not in the expected form: {ty}"
            lines[j] = (f"{lm.group('ind')}kernel.launch @g_gemm_f16_store_m256_qgs[{lm.group('g')}]({', '.join(ops[:-1] + ['%q', '%gate'])})"
                        f" : {ty[:-1]}, buffer)")
            blk = m.group(0).rstrip("\n").split("\n")
            out += ["\n".join(lines), "  scf.if %qgs_run {\n" + "\n".join("  " + l for l in blk) + "\n  }\n"]
            n += 1
        else:
            out += [seg, m.group(0)]
            refused += 1
        pos = m.end()
    body = "".join(out) + body[pos:]
    if n:
        # %qgs_ok: every PATH condition true (PATH's branches are all 'then'), defined after the last of them
        conds = list(PATH)
        at = max(re.search(rf"(?m)^  {c} = [^\n]*\n", body).end() for c in conds)   # a condition not defined: AttributeError, the build stops
        chain, acc = [], conds[0]
        for i, c in enumerate(conds[1:], 1):
            name = "%qgs_ok" if i == len(conds) - 1 else f"%qgs_ok{i}"
            chain.append(f"  {name} = scalar.andi {acc}, {c} : i1\n")
            acc = name
        body = (body[:at] + "".join(chain) + "  %qgs_t = index.cmp eq, %c0, %c0 : index\n"
                "  %qgs_f = index.cmp ne, %c0, %c0 : index\n  %qgs_run = scf.select %qgs_ok, %qgs_f, %qgs_t : i1\n" + body[at:])
    return text[:a] + body + text[b:], n, refused


def kernel(store_text, D):
    """g_gemm_f16_store_m256[_rdna4] -> its _qgs copy: the output tile goes to %y (query) or %y2 (gate) at [row, head * D + dim],
    row stride n / 2 (upstream gen_gemm.qg_split, f7ad9e6)."""
    old = re.search(r'kernel\.def (?:target\(@\w+\) )?export\("(\w+)"\)', store_text).group(1)
    new = old.replace("_store_m256", "_store_m256_qgs")
    assert new != old, old
    t = store_text.replace(f'export("{old}") @{old}(', f'export("{new}") @{new}(', 1)
    assert t.count(", %y: buffer) {") == 1, old
    t = t.replace(", %y: buffer) {", ", %y: buffer, %y2: buffer) {", 1)
    a = ("  %y_layout = encoding.layout.strided [%n, %c1] : encoding<layout>\n"
         "  %tile_first = index.madd %row_origin, %n, %out_origin : index\n"
         "  %tile_offset = index.scale %tile_first, %four : index, offset -> offset\n"
         "  %y_view = buffer.view %y[%tile_offset] : buffer")
    assert t.count(a) == 1, f"QG_SPLIT: {old}: epilogue not in the expected form"
    b = (f"  %qgs_n2 = index.div %n, %c2 : index\n"
         f"  %y_layout = encoding.layout.strided [%qgs_n2, %c1] : encoding<layout>\n"
         f"  %qgs_d = index.constant {D} : index\n"
         f"  %qgs_2d = index.constant {2 * D} : index\n"
         f"  %qgs_h = index.div %out_origin, %qgs_2d : index\n"
         f"  %qgs_r = index.rem %out_origin, %qgs_2d : index\n"
         f"  %qgs_gate = index.cmp uge, %qgs_r, %qgs_d : index\n"
         f"  %qgs_dd = index.rem %out_origin, %qgs_d : index\n"
         f"  %qgs_col = index.madd %qgs_h, %qgs_d, %qgs_dd : index\n"
         f"  %tile_first = index.madd %row_origin, %qgs_n2, %qgs_col : index\n"
         f"  %tile_offset = index.scale %tile_first, %four : index, offset -> offset\n"
         f"  %qgs_buf = scf.select %qgs_gate, %y2, %y : buffer\n"
         f"  %y_view = buffer.view %qgs_buf[%tile_offset] : buffer")
    return t.replace(a, b, 1)


def apply(text, D):
    if not ON:
        return text
    assert D % 128 == 0, "QG_SPLIT: a 128-column output tile must not straddle a head's query / gate halves"
    text, n, refused = program(text)
    print(f"qg_split: layers {n} (refused {refused})", file=sys.stderr)   # build.py discards gen_llm's stdout
    assert n, "QG_SPLIT: no layer converted"
    names = KDEF.findall(text)
    assert "g_gemm_f16_store_m256" in names, "QG_SPLIT: no g_gemm_f16_store_m256 kernel to copy"
    add = [kernel(kernel_text(text, nm), D) for nm in names]
    first = text.index("command.def ")
    first = text.rfind("\n", 0, first) + 1
    return text[:first] + "".join(k + "\n" for k in add) + text[first:]
