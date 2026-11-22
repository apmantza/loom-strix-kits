"""EPI_FUSE=1 (HRX-like E199/E200): two prefill epilogue fusions ported from upstream loom-skills (gen_llm.resid_epi /
noh_dead_h, choices prefill.fuse.resid_epi and prefill.fuse.post_norm_noh_all, PR 1323), adapted to our f16-scratch path.

noh: in @llm_advance, a GEMM-epoch g_post_add_norm_rs_live_lv whose f32 h row nothing reads before the next GEMM-epoch norm
rewrites %h becomes g_post_add_norm_rs_live_noh_lv (no 20 KB/row f32 h store; h16 + scales unchanged).

resid: an output projection (attention o, ssm_out, ffn down) adds its result into the residual stream x in its 256-row
epilogue (g_gemm_<fmt>_acc_m256: x += acc * inverse, the f32 sum the norm did) and the norm reads x only
(g_add_norm_rs_live[_noh]_lv: no y read, no x store). Upstream converts the one m256 launch before each norm; our projections
also have m256 launches in the quantized branch (`scf.if %pf_big` else) and the f16 scratch runs only from PF16_MIN_ROWS, so
the x-only norm runs under %re_ok = %pf_big and %huge, and a block is converted only when every launch that can write or read
%y on that path is one f16-scratch m256 store run exactly under it. Anything else leaves the block unchanged and is counted
(fail closed: a refused block keeps the old, correct code). Off (unset): the text is unchanged."""
import os
import re

ON = os.environ.get("EPI_FUSE", "0").strip().lower() not in ("", "0", "false", "no", "off")
PATH = {"%mm": "then", "%pf_big": "then", "%big": "then", "%huge": "then"}   # the epoch the x-only norm runs in
NORM = re.compile(r"^(?P<ind>\s*)kernel\.launch @g_post_add_norm_rs_live(?P<noh>_noh)?_lv\[%t\]\((?P<args>%t, %info, %y, [^)]*)\) : (?P<ty>.*)$")
ANY_NORM = re.compile(r"^\s*kernel\.launch @g_(?:post_)?add_norm\w*\[")
LAUNCH = re.compile(r"^(?P<ind>\s*)kernel\.launch @(?P<k>\w+)\[(?P<g>[^\]]*)\]\((?P<args>.*)\) : (?P<ty>.*)$")
STORE256 = re.compile(r"g_gemm_(?P<fmt>\w+?)_store_m256")


def _advance(text):
    a = text.index("command.def public @llm_advance(")
    b = text.find("\ncommand.def ", a + 10)
    return a, (len(text) if b < 0 else b + 1)


def _stacks(lines):
    """per line: the enclosing [(cond, branch)] of `scf.if %c {` blocks"""
    out, stack = [], []
    for line in lines:
        s = line.strip()
        mi = re.match(r"scf\.if (%\w+) \{$", s)
        if mi:
            out.append(list(stack))
            stack.append((mi.group(1), "then"))
            continue
        if s == "} else {" and stack:
            stack[-1] = (stack[-1][0], "else")
        elif s == "}" and stack:
            stack.pop()
        out.append(list(stack))
    return out


def _reachable(st, path):
    """can a line under stack `st` run in the epoch `path`?"""
    return all(path.get(c, br) == br for c, br in st)


def _operands(args):
    return [a.strip() for a in args.split(",")]


def noh(text):
    """the noh rewrite (see module doc). Returns (text, launches rewritten)."""
    a, b = _advance(text)
    lines = text[a:b].split("\n")
    st = _stacks(lines)
    gemm_epoch = {"%mm": "then"}
    n = 0
    for i, line in enumerate(lines):
        m = NORM.match(line)
        if not m or m.group("noh") or not _reachable(st[i], gemm_epoch) or ("%mm", "then") not in st[i]:
            continue
        h = _operands(m.group("args"))[-3]
        assert h == "%h", line
        dead = None
        for j in range(i + 1, len(lines)):
            if not _reachable(st[j], gemm_epoch):
                continue
            lm = LAUNCH.match(lines[j])
            if not lm:
                continue
            ops = _operands(lm.group("args"))
            if ANY_NORM.match(lines[j]) and ops[-3:-2] == ["%h"]:
                dead = True   # the next GEMM-epoch norm rewrites %h
                break
            if "%h" in ops:
                dead = False
                break
        if dead:
            lines[i] = line.replace("@g_post_add_norm_rs_live_lv[", "@g_post_add_norm_rs_live_noh_lv[", 1)
            n += 1
    return text[:a] + "\n".join(lines) + text[b:], n


def resid(text):
    """the resid rewrite (see module doc). Returns (text, blocks rewritten, refused blocks, formats needing acc_m256)."""
    a, b = _advance(text)
    lines = text[a:b].split("\n")
    st = _stacks(lines)
    hu = [i for i, l in enumerate(lines) if re.match(r"  %huge = ", l)]
    pb = [i for i, l in enumerate(lines) if re.match(r"  %pf_big = ", l)]
    if not hu or not pb:
        return text, 0, 0, set()
    assert len(hu) == 1 and len(pb) == 1 and pb[0] < hu[0], (hu, pb)
    n, refused, fmts, edits = 0, 0, set(), []
    start = 0
    for i, line in enumerate(lines):
        m = NORM.match(line)
        if not m:
            if ANY_NORM.match(line):
                start = i + 1
            continue
        if st[i] != [("%mm", "then")] or lines[i - 1].strip() != "scf.if %mm {" or lines[i + 1].strip() != "} else {":
            refused += 1
            start = i + 1
            continue
        ok, conv = True, []
        for j in range(start, i):
            lm = LAUNCH.match(lines[j])
            if not lm or not _reachable(st[j], PATH):
                continue
            ops = _operands(lm.group("args"))
            if "%y" not in ops:
                continue
            sm = STORE256.fullmatch(lm.group("k"))
            conds = {c: br for c, br in st[j]}
            exact = all(conds.get(c) == "then" for c in ("%mm", "%pf_big", "%huge")) and set(conds) <= set(PATH)
            if sm and ops[-1] == "%y" and "%y" not in ops[:-1] and exact and sm.group("fmt") == "f16":
                conv.append((j, lm))
            else:
                ok = False
        if not ok or len(conv) != 1:
            refused += 1
            start = i + 1
            continue
        j, lm = conv[0]
        ops = _operands(lm.group("args"))
        fmts.add(STORE256.fullmatch(lm.group("k")).group("fmt"))
        edits.append((j, f"{lm.group('ind')}kernel.launch @{lm.group('k').replace('_store_m256', '_acc_m256')}[{lm.group('g')}]"
                         f"({', '.join(ops[:-1] + ['%x'])}) : {lm.group('ty')}"))
        name = f"g_add_norm_rs_live{m.group('noh') or ''}_lv"
        ind = m.group("ind")
        edits.append((i, f"{ind}scf.if %re_ok {{\n{ind}  kernel.launch @{name}[%t]({m.group('args')}) : {m.group('ty')}\n"
                         f"{ind}}} else {{\n  {line}\n{ind}}}"))
        n += 1
        start = i + 1
    for j, new in edits:
        lines[j] = new
    if n:
        lines[hu[0]] += "\n  %re_ok = scalar.andi %pf_big, %huge : i1"
    return text[:a] + "\n".join(lines) + text[b:], n, refused, fmts


def x_only_norm(lv_text, noh_):
    """g_add_norm_rs_live[_noh]_lv from g_post_add_norm_rs_live[_noh]_lv: x is read (y + x with y = 0) and not stored; same
    launch signature (the y operand is unused)."""
    old = f"g_post_add_norm_rs_live{'_noh' if noh_ else ''}_lv"
    name = old.replace("g_post_add_norm", "g_add_norm")
    t = lv_text.replace(f'export("{old}") @{old}(', f'export("{name}") @{name}(')
    assert f'export("{name}")' in t
    t, n1 = re.subn(r"    %a = vector\.load %y_view\[%token, %bl, %c0\] : [^\n]*\n", "    %a = vector.addf %zero32, %zero32 : vector<32xf32>\n", t)
    t, n2 = re.subn(r"    vector\.store %sum, %x_view\[%token, %bl, %c0\] : [^\n]*\n", "", t)
    assert n1 == 1 and n2 == 1, (n1, n2)
    return t


def noh_kernel(lv_text):
    """g_post_add_norm_rs_live_noh_lv from g_post_add_norm_rs_live_lv: no f32 h store"""
    t = lv_text.replace('export("g_post_add_norm_rs_live_lv") @g_post_add_norm_rs_live_lv(',
                        'export("g_post_add_norm_rs_live_noh_lv") @g_post_add_norm_rs_live_noh_lv(')
    t, k = re.subn(r"    vector\.store %hv, %h_view\[%token, %bl, %c0\] : [^\n]*\n", "", t)
    assert k == 1 and 'export("g_post_add_norm_rs_live_noh_lv")' in t, k
    return t


def kernel_text(text, name):
    m = re.search(r'(?m)^kernel\.def (?:target\(@\w+\) )?export\("' + name + r'"\)', text)
    assert m, name
    e = text.index("\n  kernel.return\n}\n", m.start()) + len("\n  kernel.return\n}\n")
    return text[m.start():e]


def apply(text, gemm_kernels):
    """`gemm_kernels(fmts)`: the g_gemm_<fmt>_acc_m256 kernel texts (gen_llm runs gen_gemm.py with the same switches)."""
    if not ON:
        return text
    text, n_noh = noh(text)
    text, n_re, n_ref, fmts = resid(text)
    print(f"epi_fuse: noh norms {n_noh}, resid blocks {n_re} (refused {n_ref}), acc formats {sorted(fmts)}")
    assert n_re, "EPI_FUSE: no residual block converted"
    lv = kernel_text(text, "g_post_add_norm_rs_live_lv")
    add = []
    if "@g_post_add_norm_rs_live_noh_lv[" in text:
        add.append(noh_kernel(lv))
    for noh_ in (False, True):
        if f"@g_add_norm_rs_live{'_noh' if noh_ else ''}_lv[" in text:
            add.append(x_only_norm(noh_kernel(lv) if noh_ else lv, noh_))
    need = [f for f in sorted(fmts) if f'export("g_gemm_{f}_acc_m256")' not in text]
    first = text.index("command.def ")
    first = text.rfind("\n", 0, first) + 1
    text = text[:first] + "".join(k + "\n" for k in add) + (gemm_kernels(need) + "\n" if need else "") + text[first:]
    assert all(f'export("g_gemm_{f}_acc_m256")' in text for f in fmts), fmts
    return text
