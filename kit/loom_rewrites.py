#!/usr/bin/env python3
"""Text-rewrite sites in the kit's generators, and the ratchet over them (kit/test_loom_rewrites.py runs it).

Recurrence it prevents (HRX-like 11-25, user: 'how can we make sure that any future dev takes into account loom's built-ins?'):
kit/gen_llm.py applies 18 text post-passes over the generated Loom program (flash2_b2.py and gen_gdn.py dozens more), while
Loom 0.2.0 offers source-level loop policies (`pipeline(%depth)`, `unroll(%factor)`, `schedule(recurrence)`, a `--config`
search: hrx-system loom/docs/src/workflows/tune-loop-schedules.md, search-loop-schedules.md) nobody had tried; and HRX-like
E145: a NaN-fragile text workaround plus an upstream request for `scf.schedule.fence`, which 0.2.0 already documented.

What counts as a site (decided by the Python AST, so comments and string contents never match; a string's text is never read):
  call      any attribute named replace / sub / subn / translate / insert, called or not (str.replace, re.sub, a compiled
            pattern's .sub, a list's .insert, an alias `f = text.replace`)
  import    `from re import sub` (or subn; also from regex)
  getattr   getattr(x, NAME) with NAME one of the above, or not a string literal (unresolvable: counted)
  store     an assignment to a slice (`L[a:b] = ...`, `L[i:i] = [...]`), an augmented one, or `del X[...]`
  splice    an expression that joins a slice with other parts: an a + b chain, an f-string, or a .join([...]) holding `x[a:b]`
            (`text[:a] + new + text[b:]`); counted once at the outermost chain
  splitjoin `SEP.join(X.split(...))`
  helper    a reference to a kit function that takes the text to search for from its caller (a parameter is the pattern of one
            of its sites, or is passed to another helper): `def rw(t, a, b): return t.replace(a, b)` makes every `rw(...)` a site
'Program text' cannot be told from other strings, lists or arrays by syntax, so every site in scope counts (fail closed), except
these forms, which are provably not program text:
  os.replace(...)  (a file rename)        sys.path.insert(...)  (the import path)
  del X['key']     (a mapping key: the subscript is a string literal)
  a slice store whose value is built only from numbers, arithmetic, tuples, np./numpy. calls and the array-only methods
  reshape / astype / view / ravel / flatten (`orig[10:] = np.arange(n)`, `meta[4:9] = rec.reshape(-1)`)
Scope: every .py under kit/ except test_*.py (tests mutate fixtures on purpose) and kit/fixtures/. No file is exempt by name or
by content: flash2_smx2.py rewrites program text and holds no Loom token in any literal.
Known residual (not detected): a rewrite spelled with none of the above (character loops, index arithmetic into a new list,
str.format over a whole program, a helper in another repository). The ratchet raises the price of the usual spellings; review
catches the rest.

Identity of a site: (file, enclosing function qualname, anchor), the anchor being a hash of the site's AST without positions, so
moving code keeps it, and editing a rewrite's pattern makes it a new site. Identical sites in one function are a multiset.

The ratchet (check): every site is in the baseline (kit/loom_rewrites.baseline, inherited debt, may only shrink) or its enclosing
function's header comment (the # lines directly above the def or its decorators) carries
    # loom-builtin: <feature> ruled out: <reason> (<path>:<line> at vX.Y.Z)
where <path> is a file of the hrx-system repository at tag vX.Y.Z with at least <line> lines (read with `git show`, so the checkout's
own state does not matter). A dangling citation, a missing tag or no hrx-system checkout fails: an unverifiable citation is not one.
A site at module level has no header and always fails: put it in a function. A baseline row with no site left fails too
(remove it with `shrink`), so a freed row cannot admit a later site.
hrx-system: $HRX_SYSTEM, else HRX_SYSTEM= in the machine env ($LSK_ENV or ~/.config/loom-strix-kits/env).
usage: loom_rewrites.py list [KIT_DIR]              every site (file, function, kind, anchor, excerpt)
       loom_rewrites.py check [KIT_DIR] [--baseline F]   exit 1 on a new site without a valid header, or a stale row
       loom_rewrites.py shrink [KIT_DIR] [--baseline F]  drop stale rows (never adds one)
       loom_rewrites.py init [KIT_DIR] [--baseline F]    write the baseline from today's sites (only when none exists)"""
import argparse
import ast
import collections
import hashlib
import io
import os
import re
import subprocess
import sys
import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "loom_rewrites.baseline")
REWRITE_ATTRS = {"replace", "sub", "subn", "translate", "insert"}
NOT_TEXT_CALLS = {("os", "replace"), ("sys.path", "insert")}
ARRAY_METHODS = {"reshape", "astype", "view", "ravel", "flatten"}
HEADER = re.compile(r"#\s*loom-builtin:\s*(?P<feature>\S.*?)\s+ruled out:\s*(?P<reason>\S.*?)\s*"
                    r"\((?P<path>[^\s():]+):(?P<line>\d+) at (?P<tag>v\d+\.\d+\.\d+)\)\s*$")


def files(kit):
    out = []
    for root, dirs, names in os.walk(kit):
        if {"fixtures", "__pycache__"} & set(os.path.relpath(root, kit).split(os.sep)):
            continue
        out += [os.path.join(root, n) for n in sorted(names) if n.endswith(".py") and not n.startswith("test_")]
    return sorted(out)


def _sliced(n):
    return isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice)


def _numeric(v):
    """a slice store's value provably not text: numbers, arithmetic, tuples, numpy calls, array-only methods"""
    if isinstance(v, ast.Constant):
        return isinstance(v.value, (int, float)) and not isinstance(v.value, bool) or v.value is None
    if isinstance(v, ast.UnaryOp):
        return _numeric(v.operand)
    if isinstance(v, ast.BinOp):
        return _numeric(v.left) and _numeric(v.right)
    if isinstance(v, ast.Tuple):
        return all(_numeric(e) for e in v.elts)
    if isinstance(v, ast.Call):
        f = v.func
        if isinstance(f, ast.Attribute) and f.attr in ARRAY_METHODS:
            return True
        root = f
        while isinstance(root, ast.Attribute):
            root = root.value
        return isinstance(root, ast.Name) and root.id in ("np", "numpy") and f is not root
    if isinstance(v, ast.Name):   # a name is not provable (could hold lines): only through a numeric call or literal
        return False
    return False


def _dotted(n):
    parts = []
    while isinstance(n, ast.Attribute):
        parts.append(n.attr)
        n = n.value
    if isinstance(n, ast.Name):
        parts.append(n.id)
        return ".".join(reversed(parts))
    return None


def _splice_parts(n):
    """the operands of an a + b chain"""
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
        return _splice_parts(n.left) + _splice_parts(n.right)
    return [n]


class _Scan(ast.NodeVisitor):
    def __init__(self, helpers):
        self.helpers = helpers
        self.stack = []          # enclosing FunctionDef nodes
        self.sites = []          # (qualname, kind, node, def node or None)
        self.helper_defs = set()  # function names whose pattern comes from a parameter
        self.names = []          # class / function names for the qualname

    def _qual(self):
        return ".".join(self.names) or "<module>"

    def _add(self, kind, node, pattern=None):
        fn = self.stack[-1] if self.stack else None
        call = getattr(node, "_call", None)
        node = call if call is not None and call.func is node else node   # a called site is identified by its whole call
        self.sites.append((self._qual(), kind, node, fn))
        if fn is not None and pattern is not None:
            params = {a.arg for a in fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs}
            params |= {a.arg for a in (fn.args.vararg, fn.args.kwarg) if a}
            if any(isinstance(p, ast.Name) and p.id in params for p in pattern):
                self.helper_defs.add(fn.name)

    def visit_FunctionDef(self, n):
        self.names.append(n.name)
        self.stack.append(n)
        self.generic_visit(n)
        self.stack.pop()
        self.names.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, n):
        self.names.append(n.name)
        self.generic_visit(n)
        self.names.pop()

    def visit_Attribute(self, n):
        if n.attr in REWRITE_ATTRS and (_dotted(n.value), n.attr) not in NOT_TEXT_CALLS:
            self._add("call", n, self._pattern_of(n))
        elif isinstance(n.ctx, ast.Load) and n.attr in self.helpers:
            self._add("helper", n, self._call_args(n))
        self.generic_visit(n)

    def _parent_call(self, n):
        return getattr(n, "_call", None)

    def _pattern_of(self, attr):
        call = self._parent_call(attr)
        if call is None:
            return []
        if attr.attr in ("sub", "subn") and _dotted(attr.value) in ("re", "regex", "_re"):
            return call.args[:1]
        return call.args[:1] if attr.attr in ("replace", "sub", "subn") else []

    def _call_args(self, n):
        call = self._parent_call(n)
        return list(call.args) + [k.value for k in call.keywords] if call is not None else []

    def visit_Name(self, n):
        if isinstance(n.ctx, ast.Load) and n.id in self.helpers:
            self._add("helper", n, self._call_args(n))

    def visit_ImportFrom(self, n):
        if n.module in ("re", "regex") and any(a.name in ("sub", "subn", "*") for a in n.names):
            self._add("import", n)

    def visit_Call(self, n):
        n.func._call = n
        f = n.func
        if isinstance(f, ast.Name) and f.id == "getattr" and len(n.args) >= 2:
            a = n.args[1]
            if not (isinstance(a, ast.Constant) and isinstance(a.value, str)) or a.value in REWRITE_ATTRS:
                self._add("getattr", n)
        if isinstance(f, ast.Attribute) and f.attr == "join" and n.args:
            a = n.args[0]
            if isinstance(a, ast.Call) and isinstance(a.func, ast.Attribute) and a.func.attr == "split":
                self._add("splitjoin", n)
            elif isinstance(a, (ast.List, ast.Tuple)) and any(_sliced(e) for e in a.elts) and len(a.elts) > 1:
                self._add("splice", n)
        self.generic_visit(n)

    def _store(self, targets, value, n):
        for t in targets:
            if _sliced(t) and not (value is not None and _numeric(value)):
                self._add("store", n)
                return

    def visit_Assign(self, n):
        self._store(n.targets, n.value, n)
        self.generic_visit(n)

    def visit_AugAssign(self, n):
        self._store([n.target], None, n)
        self.generic_visit(n)

    def visit_Delete(self, n):
        for t in n.targets:
            if isinstance(t, ast.Subscript) and not (isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str)):
                self._add("store", n)
                break
        self.generic_visit(n)

    def visit_BinOp(self, n):
        if isinstance(n.op, ast.Add):
            parts = _splice_parts(n)
            if any(_sliced(p) for p in parts):
                self._add("splice", n)
            for p in parts:   # the chain counts once; look inside its operands only
                self.visit(p)
            return
        self.generic_visit(n)

    def visit_JoinedStr(self, n):
        vals = [v.value for v in n.values if isinstance(v, ast.FormattedValue)]
        if any(_sliced(v) for v in vals) and len(n.values) > 1:
            self._add("splice", n)
        self.generic_visit(n)


def _anchor(node):
    return hashlib.sha1(ast.dump(node, annotate_fields=False, include_attributes=False).encode()).hexdigest()[:12]


def _excerpt(node):
    return ast.unparse(node).splitlines()[0][:90]   # unparse writes no tab; a multi-line site shows its first line


def scan(kit):
    """[(relpath, qualname, kind, anchor, excerpt, def node, source)] for every site; helpers found to a fixed point"""
    trees = []
    for p in files(kit):
        src = open(p, encoding="utf-8").read()
        trees.append((p, src, ast.parse(src, p)))
    helpers = set()
    while True:
        scans = []
        for p, src, tree in trees:
            s = _Scan(helpers)
            s.visit(tree)
            scans.append(s)
        found = set().union(*(s.helper_defs for s in scans))
        if found <= helpers:
            break
        helpers |= found
    out = []
    for (p, src, _), s in zip(trees, scans):
        rel = os.path.relpath(p, kit)
        for qual, kind, node, fn in s.sites:
            out.append((rel, qual, kind, _anchor(node), _excerpt(node), fn, src))
    return out


def _comment_lines(src):
    """line numbers holding only a comment (tokenize: a '#' inside a string is not a comment)"""
    lines = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT and tok.line.lstrip().startswith("#"):
            lines.add(tok.start[0])
    return lines


def header(fn, src, _cache={}):
    """the comment lines directly above a def (above its decorators), top to bottom"""
    if _cache.get("src") is not src:   # one entry, held with its text (an id() key was reused by a later text)
        _cache.update(src=src, com=_comment_lines(src))
    com = _cache["com"]
    first = min([fn.lineno] + [d.lineno for d in fn.decorator_list])
    text = src.splitlines()
    out, i = [], first - 1
    while i >= 1 and i in com:
        out.append(text[i - 1].strip())
        i -= 1
    return list(reversed(out))


def hrx_system():
    if os.environ.get("HRX_SYSTEM"):
        return os.environ["HRX_SYSTEM"]
    env = os.environ.get("LSK_ENV") or os.path.expanduser("~/.config/loom-strix-kits/env")
    try:
        for ln in open(env):
            m = re.match(r"\s*HRX_SYSTEM=(\S+)", ln)
            if m:
                return m.group(1).strip("'\"")
    except OSError:
        pass
    return None


def verify_citation(m, hrx, _cache={}):
    """None when <path>:<line> exists at <tag> in the hrx-system repository, else the reason"""
    if not hrx or not os.path.isdir(hrx):
        return f"no hrx-system checkout to verify it (HRX_SYSTEM={hrx!r})"
    key = (hrx, m["tag"], m["path"])
    if key not in _cache:
        r = subprocess.run(["git", "-C", hrx, "show", f"{m['tag']}:{m['path']}"], capture_output=True, text=True)
        _cache[key] = r.stdout.count("\n") + (1 if r.stdout and not r.stdout.endswith("\n") else 0) if r.returncode == 0 else None
    n = _cache[key]
    if n is None:
        return f"{m['path']} does not exist at {m['tag']} in {hrx}"
    if not 1 <= int(m["line"]) <= n:
        return f"{m['path']} has {n} lines at {m['tag']}, the citation names line {m['line']}"
    return None


def admitted(fn, src, hrx):
    """(True, citation) when the def's header carries a verified loom-builtin line, else (False, reason)"""
    if fn is None:
        return False, "module-level site: no function header can admit it (move it into a function)"
    hs = [h for h in header(fn, src) if "loom-builtin" in h]
    if not hs:
        return False, f"no '# loom-builtin: <feature> ruled out: <reason> (<path>:<line> at vX.Y.Z)' line above def {fn.name}"
    why = []
    for h in hs:
        m = HEADER.search(h)
        if not m:
            why.append(f"not in the known form: {h!r}")
            continue
        bad = verify_citation(m, hrx)
        if bad is None:
            return True, f"{m['path']}:{m['line']} at {m['tag']}"
        why.append(bad)
    return False, "; ".join(why)


def row(site):
    return "\t".join(site[:4])


def read_baseline(path):
    rows = collections.Counter()
    for ln in open(path, encoding="utf-8"):
        if ln.strip() and not ln.startswith("#"):
            f = ln.rstrip("\n").split("\t")
            rows["\t".join(f[:4])] += 1
    return rows


def check(kit, baseline, hrx=None):
    """(problems, notes)"""
    sites = scan(kit)
    base = read_baseline(baseline)
    have = collections.Counter(row(s) for s in sites)
    probs, notes = [], []
    left = base.copy()
    new = []
    for s in sites:
        r = row(s)
        if left[r] > 0:
            left[r] -= 1
        else:
            new.append(s)
    hrx = hrx if hrx is not None else hrx_system()
    for s in new:
        ok, why = admitted(s[5], s[6], hrx)
        if ok:
            notes.append(f"admitted by its header ({why}): {s[0]} {s[1]} {s[2]} {s[4]}")
        else:
            probs.append(f"new text rewrite {s[0]} {s[1]} [{s[2]}] {s[4]!r}: {why}")
    stale = [r for r, k in (base - have).items() for _ in range(k)]
    for r in stale:
        probs.append(f"stale baseline row (its site is gone or changed; run loom_rewrites.py shrink): {r}")
    notes.append(f"{len(sites)} sites, {sum(base.values())} baseline rows, {len(new)} outside the baseline")
    return probs, notes


def write_baseline(path, sites):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Text-rewrite sites over generated Loom admitted as inherited debt (kit/loom_rewrites.py; HRX-like 11-25).\n"
                "# This file may only shrink: `python3 kit/loom_rewrites.py shrink` drops rows whose site is gone; nothing adds one.\n"
                "# A new site needs '# loom-builtin: <feature> ruled out: <reason> (<path>:<line> at vX.Y.Z)' above its def.\n"
                "# columns: file, function, kind, anchor (AST hash), excerpt (not compared)\n")
        for s in sorted(sites, key=lambda s: s[:4]):
            f.write(row(s) + "\t" + s[4] + "\n")   # the excerpt holds no tab (ast.unparse escapes them)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["list", "check", "shrink", "init"])
    ap.add_argument("kit", nargs="?", default=HERE)
    ap.add_argument("--baseline", default=BASELINE)
    a = ap.parse_args()
    if a.cmd == "list":
        for s in scan(a.kit):
            print("\t".join(s[:5]))
        return 0
    if a.cmd == "init":
        if os.path.exists(a.baseline):
            print(f"{a.baseline} exists: the baseline may only shrink (use shrink)")
            return 1
        write_baseline(a.baseline, scan(a.kit))
        return 0
    if a.cmd == "shrink":
        sites = scan(a.kit)
        have = collections.Counter(row(s) for s in sites)
        keep, kept = [], collections.Counter()
        lines = open(a.baseline, encoding="utf-8").read().splitlines(True)
        head = [ln for ln in lines if ln.startswith("#") or not ln.strip()]
        for ln in lines:
            if ln.startswith("#") or not ln.strip():
                continue
            r = "\t".join(ln.rstrip("\n").split("\t")[:4])
            if kept[r] < have[r]:
                kept[r] += 1
                keep.append(ln)
        open(a.baseline, "w", encoding="utf-8").write("".join(head + keep))
        print(f"kept {len(keep)} of {len(lines) - len(head)} rows")
        return 0
    probs, notes = check(a.kit, a.baseline)
    for n in notes:
        print("note: " + n)
    for p in probs:
        print("LOOM REWRITE REFUSED " + p)
    print("LOOM REWRITES: " + ("REFUSED" if probs else "OK"))
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
