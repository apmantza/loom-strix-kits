#!/usr/bin/env python3
"""Loom pass-boundary IR tools on one kernel root: an instruction-level ablation in prepared Low, and a pass bisect.

Both start from one trace: `loom-compile PROGRAM --root=ROOT --dump-ir-after-all --dump-ir-format=jsonl --dump-ir-output=DIR/`
(docs: workflows/compile-report-queries.md 'Follow an IR trace', workflows/oracles/native-schedule.md). On Loom 0.2.0 the trace
covers the prepared-Low pipeline only (70 events for flash2: specialize-target-callgraph ... source-to-low ... low-dce); the
emitter's scheduling, register allocation and wait insertion (`s_waitcnt`) are not passes and never appear in an IR snapshot.

ablate: take the last snapshot (the emitter's input), apply named edits, and assemble it with `--pipeline=none` into a code object
(plus a compile report) that common/hsaco_run.py launches. The unedited snapshot is assembled the same way as the CONTROL: the
prepared-Low text round trip is not byte-faithful on 0.2.0 (live flash2_f16o: served 4805 instructions, reassembled 4808, a
different QK load interleave; the source-level snapshot 0 does round-trip exactly), so an ablation is judged against the control,
never against the served binary. The tool prints the served-vs-control distance and the control-vs-edited opcode delta.
  EDIT := delete OP [prev=/R/] [next=/R/]
        | move OP [prev=/R/] [next=/R/] to=before:/R/|after:/R/
  OP and every R are /regex/ (Python re.search) over one stripped line of ROOT's body in the snapshot; prev / next must match the
  adjacent line (block labels included). An anchor that matches zero or more than one line is refused (exit 2), as is: deleting an
  op whose result is used, a block's last op (terminator), or a label; moving across a block boundary, above a definition of an
  operand, below a use of a result, or after a terminator. Use source value names (%pl7_1, %one_f): numbered names (%1674) are
  renumbered by every pass. Waits cannot be edited here (they are emitted later); a load whose result is used cannot be deleted.

bisect: evaluate a predicate after every pass and name the first pass from which it holds through the end.
  PRED := present /A/ | order /A/ /B/ | between /A/ /B/ /C/
    present: some line matches A; order: the first A line precedes the first B line; between: after the first A line, some B line
    comes before the first C line that follows A. A predicate whose anchors find no line is U (undefined), never 'holds'.
  --on ir (default): over ROOT's body in the input program and in each snapshot. --on asm: over the disassembly (the longest loop
  with --region loop, the default, or the whole kernel with --region all) of the served code object and of each snapshot that
  assembles with --pipeline=none (prepared-Low ones); refused (exit 2) when the served binary and the reassembled final snapshot
  disagree (the round trip changed the property, so reassembly cannot locate it). An asm property already present at the first
  assemblable snapshot is reported as placed at code emission from IR present there: name the IR-level cause with --on ir.
usage: ir_trace.py ablate PROGRAM ROOT --edit EDIT [--edit EDIT ...] --out X.hsaco [--work DIR] [--loom-compile PATH]
       ir_trace.py bisect PROGRAM ROOT --pred PRED [--on ir|asm] [--region loop|all] [--work DIR] [--loom-compile PATH]
exit: 0 done, 2 refused (anchor, guard, predicate, control), 3 the compiler failed (its message is printed)."""
import argparse
import collections
import json
import os
import re
import subprocess
import sys
import tempfile

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
from loop_census import disasm, loops  # noqa: E402  (one decoder for every tool: E230)

NAME = r"%[$A-Za-z0-9_.]+"
LABEL = re.compile(r"^\^[\w.$]+(\(|:|$)")


class Refused(Exception):
    """an edit, a predicate or a control the tool will not act on (exit 2)"""


class CompileFailed(Exception):
    """loom-compile returned non-zero (exit 3)"""


# ------------------------------------------------------------------------------------------------------------ compiler
def compiler(lc=None):
    if lc:
        return lc
    from kernel_lint import loom_compile
    return loom_compile()


def run_lc(lc, args):
    r = subprocess.run([lc] + args, capture_output=True, text=True)
    if r.returncode:
        raise CompileFailed(f"{os.path.basename(lc)} {' '.join(args)}\n{(r.stderr or r.stdout).strip()[:2000]}")
    return r


def trace(lc, program, root, work):
    """compile ROOT of PROGRAM with every pass boundary dumped; returns ([event dict with 'path'], served code object)"""
    d = os.path.join(work, "trace")
    os.makedirs(work, exist_ok=True)
    served = os.path.join(work, "served.hsaco")
    run_lc(lc, [program, "--target=amdgpu:gfx1151", f"--root={root}", f"--output={served}", "--dump-ir-after-all",
                "--dump-ir-format=jsonl", f"--dump-ir-output={d}/"])
    ev = []
    for line in open(os.path.join(d, "trace.jsonl")):
        e = json.loads(line)
        if e.get("ir_path"):
            e["path"] = os.path.join(d, e["ir_path"])
            ev.append(e)
    if not ev:
        raise Refused(f"the trace of {root} has no IR snapshot")
    return ev, served


def assemble(lc, low, out, report=True):
    """assemble a prepared-Low module with --pipeline=none (no transformation) into OUT (+ OUT.json, a summary compile report)"""
    a = [low, "--pipeline=none", "--format=amdgpu-hsaco", f"--output={out}"]
    if report:
        a += ["--compile-report=summary", f"--compile-report-output={os.path.splitext(out)[0]}.json"]
    run_lc(lc, a)


# ------------------------------------------------------------------------------------------------------------ IR text
def body(lines, root):
    """(first, last) line indices of ROOT's body (exclusive of its def line and closing brace) in a Loom text"""
    pat = re.compile(r'^(low\.)?kernel\.def\b.*\bexport\("' + re.escape(root) + r'"\)')
    starts = [i for i, l in enumerate(lines) if pat.match(l)]
    if len(starts) != 1:
        raise Refused(f"{len(starts)} definitions of kernel {root} in the text (need exactly one)")
    s, depth = starts[0], 0
    for j in range(s, len(lines)):                    # brace depth: a source kernel is `{ config } launch(...) { body }`
        t = re.sub(r'"(?:\\.|[^"\\])*"', '""', lines[j])
        depth += t.count("{") - t.count("}")
        if depth == 0 and j > s:
            return s + 1, j - 1
    raise Refused(f"kernel {root}: no closing brace")


def regex(src, what):
    try:
        return re.compile(src)
    except re.error as e:
        raise Refused(f"{what}: bad regex /{src}/: {e}")


def split_slashes(s):
    """tokens of an edit / predicate: words, key=/re/, key=word:/re/ and bare /re/ (a '/' inside a regex is written '\\/')"""
    toks, i = [], 0
    while i < len(s):
        if s[i].isspace():
            i += 1
            continue
        m = re.compile(r"([A-Za-z_]+=(?:(?:before|after):)?)?/((?:\\.|[^/\\])*)/").match(s, i)
        if m:
            toks.append((m.group(1) or "", m.group(2).replace("\\/", "/")))
            i = m.end()
            continue
        m = re.compile(r"[A-Za-z_-]+").match(s, i)
        if not m:
            raise Refused(f"cannot read {s[i:]!r} (expected a word or /regex/)")
        toks.append((None, m.group(0)))
        i = m.end()
    return toks


def parse_edit(s):
    toks = split_slashes(s)
    if not toks or toks[0][0] is not None or toks[0][1] not in ("delete", "move"):
        raise Refused(f"edit {s!r}: must start with 'delete' or 'move'")
    if len(toks) < 2 or toks[1][0] != "":
        raise Refused(f"edit {s!r}: the second token must be the op's /regex/")
    e = {"action": toks[0][1], "op": regex(toks[1][1], s), "text": s}
    for key, val in toks[2:]:
        if key in ("prev=", "next="):
            e[key[:-1]] = regex(val, s)
        elif key in ("to=before:", "to=after:"):
            e["to"] = (key[3:-1], regex(val, s))
        else:
            raise Refused(f"edit {s!r}: unknown part {key or ''}{val!r} (known: prev=/R/, next=/R/, to=before:/R/, to=after:/R/)")
    if e["action"] == "move" and "to" not in e:
        raise Refused(f"edit {s!r}: move needs to=before:/R/ or to=after:/R/")
    if e["action"] == "delete" and "to" in e:
        raise Refused(f"edit {s!r}: delete takes no to=")
    return e


def defs_uses(line):
    """(names the line defines, names it reads): a label's arguments are definitions"""
    t = line.strip()
    if LABEL.match(t):
        return set(re.findall(r"(" + NAME + r")\s*:", t)), set()
    m = re.match(r"^((?:" + NAME + r")(?:\s*,\s*" + NAME + r")*)\s*=\s*(.*)$", t)
    if m:
        return set(re.findall(NAME, m.group(1))), set(re.findall(NAME, m.group(2)))
    return set(), set(re.findall(NAME, t))


def blocks(lines, lo, hi):
    """block id of every body line: a label line opens a new block"""
    b, cur = {}, 0
    for i in range(lo, hi + 1):
        if LABEL.match(lines[i].strip()):
            cur += 1
        b[i] = cur
    return b


def is_op(t):
    return bool(t) and not LABEL.match(t) and not t.startswith("//")


def find(lines, lo, hi, op, prev=None, nxt=None, what=""):
    """the one body line matching OP (and PREV / NEXT on its neighbours); refuses zero or several"""
    hits = []
    for i in range(lo, hi + 1):
        t = lines[i].strip()
        if not is_op(t) or not op.search(t):
            continue
        if prev is not None and not (i - 1 >= lo and prev.search(lines[i - 1].strip())):
            continue
        if nxt is not None and not (i + 1 <= hi and nxt.search(lines[i + 1].strip())):
            continue
        hits.append(i)
    if len(hits) != 1:
        show = "".join(f"\n    line {i + 1}: {lines[i].strip()[:140]}" for i in hits[:8]) + ("\n    ..." if len(hits) > 8 else "")
        raise Refused(f"{what}: anchor matches {len(hits)} lines (need exactly 1; add prev=/R/ or next=/R/){show}")
    return hits[0]


def uses_of(lines, lo, hi, names, skip):
    return [i for i in range(lo, hi + 1) if i != skip and defs_uses(lines[i])[1] & names]


def apply_edit(lines, root, e):
    """apply one parsed edit to a module's lines (a new list) and return (lines, a one-line description)"""
    lo, hi = body(lines, root)
    i = find(lines, lo, hi, e["op"], e.get("prev"), e.get("next"), f"edit {e['text']!r}")
    t = lines[i].strip()
    blk = blocks(lines, lo, hi)
    last_in_block = i == hi or LABEL.match(lines[i + 1].strip()) or blk.get(i + 1) != blk[i]
    if last_in_block:
        raise Refused(f"edit {e['text']!r}: line {i + 1} is its block's last op (a terminator): {t[:120]}")
    d, _ = defs_uses(t)
    if e["action"] == "delete":
        used = uses_of(lines, lo, hi, d, i)
        if used:
            raise Refused(f"edit {e['text']!r}: line {i + 1} defines {sorted(d)}, read at line(s) {[u + 1 for u in used[:6]]}: "
                          "deleting it would leave a use without a definition")
        return lines[:i] + lines[i + 1:], f"deleted line {i + 1}: {t}"
    where, tre = e["to"]
    j = find(lines, lo, hi, tre, what=f"edit {e['text']!r} target")
    if j == i:
        raise Refused(f"edit {e['text']!r}: the target is the moved op itself")
    if blk[j] != blk[i]:
        raise Refused(f"edit {e['text']!r}: target line {j + 1} is in another block than line {i + 1}")
    pos = j if where == "before" else j + 1           # insertion index in the original numbering
    if where == "after" and (j == hi or blk.get(j + 1) != blk[j]):
        raise Refused(f"edit {e['text']!r}: cannot move after the block's last op (line {j + 1})")
    _, u = defs_uses(t)
    if pos < i:                                       # moving up: no operand may be defined in [pos, i)
        bad = [k for k in range(pos, i) if defs_uses(lines[k])[0] & u]
        if bad:
            raise Refused(f"edit {e['text']!r}: an operand is defined at line {bad[0] + 1}, between the target and the op")
    else:                                             # moving down: no result may be read in (i, pos)
        bad = [k for k in range(i + 1, pos) if defs_uses(lines[k])[1] & d]
        if bad:
            raise Refused(f"edit {e['text']!r}: a result is read at line {bad[0] + 1}, between the op and the target")
    out = lines[:]
    moved = out.pop(i)
    out.insert(pos if pos < i else pos - 1, moved)
    return out, f"moved line {i + 1} {where} line {j + 1}: {t}"


# ------------------------------------------------------------------------------------------------------------ predicates
def parse_pred(s):
    toks = split_slashes(s)
    arity = {"present": 1, "order": 2, "between": 3}
    if not toks or toks[0][0] is not None or toks[0][1] not in arity:
        raise Refused(f"predicate {s!r}: must start with one of {sorted(arity)}")
    res = [t for t in toks[1:]]
    if len(res) != arity[toks[0][1]] or any(k != "" for k, _ in res):
        raise Refused(f"predicate {s!r}: {toks[0][1]} takes exactly {arity[toks[0][1]]} /regex/ arguments")
    return toks[0][1], [regex(v, s) for _, v in res]


def evaluate(pred, lines):
    """'H' holds, 'N' does not, 'U' an anchor found no line; plus a short witness"""
    kind, rs = pred
    first = lambda r, start=0: next((i for i in range(start, len(lines)) if r.search(lines[i])), None)  # noqa: E731
    a = first(rs[0])
    if kind == "present":
        return ("H", f"line {a + 1}") if a is not None else ("N", "no line")
    if a is None:
        return "U", "A: no line"
    if kind == "order":
        b = first(rs[1])
        if b is None:
            return "U", "B: no line"
        return ("H" if a < b else "N"), f"A {a + 1}, B {b + 1}"
    c = first(rs[2], a + 1)
    if c is None:
        return "U", f"A {a + 1}, no C after it"
    b = first(rs[1], a + 1)
    return ("H" if b is not None and b < c else "N"), f"A {a + 1}, B {b + 1 if b is not None else '-'}, C {c + 1}"


def ir_region(path, root):
    lines = open(path).read().split("\n")
    lo, hi = body(lines, root)
    return [l.strip() for l in lines[lo:hi + 1]]


def asm_region(hs, region):
    ins, addr = disasm(hs)
    if region == "all":
        return ins
    L = loops(ins, addr)
    if not L:
        raise Refused(f"{hs}: no loop (use --region all)")
    t, e = max(L, key=lambda x: x[1] - x[0])
    return ins[t:e + 1]


def first_steady(states):
    """index of the first entry from which every state is H, or None when the last is not H"""
    if not states or states[-1] != "H":
        return None
    k = len(states) - 1
    while k > 0 and states[k - 1] == "H":
        k -= 1
    return k


# ------------------------------------------------------------------------------------------------------------ commands
def opcode_delta(a, b):
    ca = collections.Counter(x.split()[0] for x in a)
    cb = collections.Counter(x.split()[0] for x in b)
    return {k: cb[k] - ca[k] for k in sorted(set(ca) | set(cb)) if cb[k] != ca[k]}


def cmd_ablate(a):
    lc = compiler(a.loom_compile)
    edits = [parse_edit(s) for s in a.edit]            # syntax refused before any compile
    work = a.work or os.path.splitext(a.out)[0] + ".work"
    ev, served = trace(lc, a.program, a.root, work)
    final = ev[-1]
    print(f"snapshot: event {final['event_ordinal']} after {final['pass']} ({final['stage']}), {len(ev)} events; {final['path']}")
    text = open(final["path"]).read()
    lines = text.split("\n")
    for e in edits:
        lines, what = apply_edit(lines, a.root, e)
        print(f"  {e['action']}: {what[:160]}")
    base = os.path.splitext(a.out)[0]
    ctrl_low, edit_low = base + ".control.loom", base + ".loom"
    open(ctrl_low, "w").write(text)
    open(edit_low, "w").write("\n".join(lines))
    assemble(lc, ctrl_low, base + ".control.hsaco")
    assemble(lc, edit_low, a.out)
    s, _ = disasm(served)
    c, _ = disasm(base + ".control.hsaco")
    x, _ = disasm(a.out)
    rt = "identical" if s == c else f"DIFFERS ({len(s)} -> {len(c)} instructions, opcode delta {opcode_delta(s, c)})"
    print(f"round trip served -> control (unedited, --pipeline=none): {rt}")
    print(f"control -> edited: {len(c)} -> {len(x)} instructions, opcode delta {opcode_delta(c, x)}")
    print(f"wrote {a.out} (+ .json report), control {base}.control.hsaco, Low {edit_low} / {ctrl_low}; trace {work}/trace")
    print("judge the ablation against the control, not the served binary" + ("" if s == c else " (the round trip changed it)"))
    return 0


def cmd_bisect(a):
    lc = compiler(a.loom_compile)
    pred = parse_pred(a.pred)
    work = a.work or tempfile.mkdtemp(prefix="ir_trace.")
    ev, served = trace(lc, a.program, a.root, work)
    rows = []                                          # (label, pass, state, witness)
    if a.on == "ir":
        st, w = evaluate(pred, ir_region(a.program, a.root))
        rows.append(("input", "(program text)", st, w))
        for e in ev:
            st, w = evaluate(pred, ir_region(e["path"], a.root))
            rows.append((str(e["event_ordinal"]), e["pass"], st, w))
    else:
        s_st, s_w = evaluate(pred, asm_region(served, a.region))
        asm_rows = []
        for e in reversed(ev):                         # from the emitter's input back to the first snapshot that assembles
            hs = os.path.join(work, f"snap{e['event_ordinal']:06d}.hsaco")
            try:
                assemble(lc, e["path"], hs, report=False)
            except CompileFailed:
                break
            st, w = evaluate(pred, asm_region(hs, a.region))
            asm_rows.append((str(e["event_ordinal"]), e["pass"], st, w))
        if not asm_rows:
            raise Refused("no snapshot assembles with --pipeline=none")
        asm_rows.reverse()
        if asm_rows[-1][2] != s_st:
            raise Refused(f"served binary {s_st} ({s_w}) but the reassembled final snapshot {asm_rows[-1][2]} ({asm_rows[-1][3]}): "
                          "the prepared-Low round trip changes this property, so reassembly cannot locate it")
        rows = asm_rows + [("served", "(default pipeline)", s_st, s_w)]
    states = [r[2] for r in rows]
    tsv = os.path.join(work, "bisect.tsv")
    with open(tsv, "w") as f:
        f.write("event\tpass\tstate\twitness\n" + "".join("\t".join(r) + "\n" for r in rows))
    prev = None
    for r in rows:                                     # every transition, the first and the last row
        if r[2] != prev or r is rows[-1] or r is rows[0]:
            print(f"  {r[0]:>7} {r[1]:42s} {r[2]}  {r[3]}")
        prev = r[2]
    k = first_steady(states)
    print(f"predicate {a.pred!r} on {a.on}" + (f" ({a.region})" if a.on == "asm" else "") + f"; table {tsv}")
    if k is None:
        print(f"verdict: does not hold at the end ({states[-1]})")
    elif a.on == "ir" and k == 0:
        print("verdict: holds in the input program and after every pass: no pass introduced it (the program text did)")
    elif a.on == "asm" and k == 0:
        print(f"verdict: holds from the first assemblable snapshot (event {rows[0][0]}, after {rows[0][1]}): placed at code "
              "emission from IR present there; no prepared-Low pass to blame (use --on ir for earlier passes)")
    else:
        print(f"verdict: first holds after event {rows[k][0]} {rows[k][1]} (state before: {states[k - 1]}) and through the end")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ablate")
    p.add_argument("program")
    p.add_argument("root")
    p.add_argument("--edit", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--work")
    p.add_argument("--loom-compile")
    p = sub.add_parser("bisect")
    p.add_argument("program")
    p.add_argument("root")
    p.add_argument("--pred", required=True)
    p.add_argument("--on", choices=("ir", "asm"), default="ir")
    p.add_argument("--region", choices=("loop", "all"), default="loop")
    p.add_argument("--work")
    p.add_argument("--loom-compile")
    a = ap.parse_args(argv)
    try:
        return cmd_ablate(a) if a.cmd == "ablate" else cmd_bisect(a)
    except Refused as e:
        print(f"ir_trace: refused: {e}", file=sys.stderr)
        return 2
    except CompileFailed as e:
        print(f"ir_trace: compile failed: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
