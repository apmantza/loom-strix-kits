#!/usr/bin/env python3
"""The ratchet over text rewrites of generated Loom (kit/loom_rewrites.py): today's sites are inherited debt in
kit/loom_rewrites.baseline, which may only shrink; a new site needs a verified loom-builtin citation above its def.
Recurrences (HRX-like 11-25): gen_llm.py's 18 text post-passes (flash2_b2.py, gen_gdn.py dozens more) were written while Loom 0.2.0's
loop policies (pipeline / unroll / schedule(recurrence), --config search: loom/docs/src/workflows/tune-loop-schedules.md) went
untried; E145 built a NaN-fragile text workaround and asked upstream for `scf.schedule.fence`, which 0.2.0 already documented.
Cases (each on a copy of the kit's .py files with one change; the real tree is checked first):
  - the tree as committed: no site outside the baseline, no stale row;
  - a new rewrite with no header: refused; with a verified header: admitted; a header citing a missing file, a line past the end,
    a missing tag, or no hrx-system checkout: refused (a dangling citation is not one); a header in the wrong form: refused;
  - the citation as prose does not admit: in a docstring, in a string whose line starts with '#', or separated from the def by a
    blank line; a rewrite written only in a comment or a string is not a site (detectors match code, not prose);
  - every detector kind is a site in a new function (call, import, getattr, store, splice, splitjoin, a new call of a helper);
    the not-text forms are not (os.replace, sys.path.insert, a numpy slice store, a mapping-key del); a slice store of a name is;
  - a module-level site is refused even with a valid header above it;
  - editing an inherited site's pattern makes it a new site, and leaves its row stale;
  - a removed site leaves a stale row: refused; shrink drops it and never admits a new site.
HRX_SYSTEM (an hrx-system checkout with tag v0.2.0) is needed for the admitted cases: env, or HRX_SYSTEM= in the machine env file.
usage: python3 kit/test_loom_rewrites.py   (exit 1 on a failure; LOOM_REWRITES_MOD=path runs the cases against a mutant)"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MOD = os.environ.get("LOOM_REWRITES_MOD", os.path.join(HERE, "loom_rewrites.py"))
spec = importlib.util.spec_from_file_location("loom_rewrites", MOD)
lr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lr)
BASE = os.path.join(HERE, "loom_rewrites.baseline")
GOOD = ("# loom-builtin: unroll(%factor) ruled out: the post-pass reorders across two loops, unroll keeps one "
        "(loom/docs/src/workflows/tune-loop-schedules.md:42 at v0.2.0)")
fails = 0


def ok(cond, what):
    global fails
    print(("ok  " if cond else "FAIL") + " " + what)
    fails += not cond


def copy_kit(dst):
    for p in lr.files(HERE):
        q = os.path.join(dst, os.path.relpath(p, HERE))
        os.makedirs(os.path.dirname(q), exist_ok=True)
        shutil.copy(p, q)


def main():
    hrx = lr.hrx_system()
    probs, notes = lr.check(HERE, BASE, hrx)
    ok(not probs, f"the kit as committed: {notes[-1]} ({probs[:3]})")
    ok(hrx and os.path.isdir(hrx), f"hrx-system checkout for the admitted cases: {hrx}")
    with tempfile.TemporaryDirectory() as t:
        kit = os.path.join(t, "kit")
        copy_kit(kit)
        gen = os.path.join(kit, "gen_llm.py")
        orig = open(gen).read()

        def run(add, base=BASE, h=hrx, replace=None):
            text = orig if replace is None else orig.replace(*replace)
            open(gen, "w").write(text + add)
            return lr.check(kit, base, h)

        fake = "\n\ndef fake_post_pass(text):\n    return text.replace('scf.for', 'scf.for')\n"
        pr, _ = run(fake)
        ok(len(pr) == 1 and "fake_post_pass" in pr[0] and "no '# loom-builtin" in pr[0], f"a new rewrite, no header: refused ({pr})")
        hdr = lambda line: fake.replace("\ndef fake", "\n" + line + "\ndef fake")
        pr, notes = run(hdr(GOOD))
        ok(not pr and any("admitted" in n and "tune-loop-schedules.md:42" in n for n in notes), f"with a verified header: admitted ({pr})")
        pr, _ = run(fake.replace("\ndef fake", "\n# the header above\n" + GOOD + "\n# more of the header\ndef fake"))
        ok(not pr, f"the citation anywhere in the contiguous comment block above the def: admitted ({pr})")
        pr, _ = run("\n\ndef decorate(f):\n    return f\n" + fake.replace("\ndef fake", "\n" + GOOD + "\n@decorate\ndef fake"))
        ok(not pr, f"above the decorators: admitted ({pr})")
        for bad, what in ((GOOD.replace("tune-loop-schedules.md", "no-such-doc.md"), "a missing file"),
                          (GOOD.replace(":42 at", ":99999 at"), "a line past the end"),
                          (GOOD.replace(":42 at", ":0 at"), "line 0"),
                          (GOOD.replace("v0.2.0", "v9.9.9"), "a missing tag"),
                          (GOOD.replace(" at v0.2.0", ""), "no tag (not the known form)"),
                          (GOOD.replace("ruled out:", "considered:"), "no 'ruled out' (not the known form)")):
            pr, _ = run(hdr(bad))
            ok(len(pr) == 1 and "fake_post_pass" in pr[0], f"header citing {what}: refused ({pr[0][-110:] if pr else pr})")
        pr, _ = run(hdr(GOOD), h=os.path.join(t, "no-hrx"))
        ok(len(pr) == 1 and "no hrx-system checkout" in pr[0], f"no hrx-system checkout to verify against: refused ({pr})")
        pr, _ = run(fake.replace("\ndef fake", "\n" + GOOD + "\n\ndef fake"))
        ok(len(pr) == 1, f"citation separated from the def by a blank line: refused ({pr})")
        pr, _ = run(fake.replace("(text):\n", "(text):\n    " + repr(GOOD) + "\n"))
        ok(len(pr) == 1, f"citation in the docstring: refused ({pr})")
        pr, _ = run("\n\nX = '''\n" + GOOD + "\n# end of the string'''\ndef fake_post_pass(text):\n    return text.replace('a', 'b')\n")
        ok(len(pr) == 1, f"citation as string lines starting with '#' directly above the def: refused ({pr})")
        pr, _ = run("\n\ndef prose_only(text):\n    # text.replace('a', 'b'); re.sub('x', 'y', text)\n"
                    "    return 'text.replace(a, b) re.sub(x, y) L[i:i] = [s]'\n")
        ok(not pr, f"a rewrite only in a comment or a string: not a site ({pr})")
        kinds = {
            "call": "    return re.compile('a').sub('b', text)\n",
            "import": "    from re import subn\n    return text\n",
            "getattr": "    return getattr(text, 'rep' + 'lace')('a', 'b')\n",
            "store": "    L = text.splitlines()\n    L[1:1] = moved\n    return L\n",
            "store-del": "    L = text.splitlines()\n    del L[3]\n    return L\n",
            "splice": "    return text[:a] + 'x' + text[b:]\n",
            "splice-fstring": "    return f'{text[:a]}x{text[b:]}'\n",
            "splice-join": "    return ''.join([text[:a], 'x', text[b:]])\n",
            "splitjoin": "    return 'b'.join(text.split('a'))\n",
            "helper": "    return _sub1(text, 'a', 'b')\n",
        }
        for k, body in kinds.items():
            pr, _ = run(f"\n\ndef fake_{k.replace('-', '_')}(text, a=1, b=2, moved=()):\n" + body)
            ok(len(pr) >= 1 and all(f"fake_{k.replace('-', '_')}" in x for x in pr), f"detector kind {k}: a new site ({pr[:1]})")
        nottext = ("    import numpy as np, os, sys\n    os.replace('a', 'b')\n    sys.path.insert(0, 'x')\n"
                   "    arr = np.zeros(8)\n    arr[2:4] = np.arange(2)\n    arr[4:6] = rec.reshape(-1)\n    arr[6:] = (1, 2)\n"
                   "    d = {}\n    del d['k']\n    return arr\n")
        pr, _ = run("\n\ndef not_text(rec):\n" + nottext)
        ok(not pr, f"the not-text forms (os.replace, sys.path.insert, numpy slice stores, mapping-key del): not sites ({pr})")
        pr, _ = run("\n\n" + GOOD + "\nTEXT = 'a'.replace('a', 'b')\n")
        ok(len(pr) == 1 and "module-level" in pr[0], f"a module-level site: refused even with a header above it ({pr})")
        # an inherited site in gen_llm.py, edited: a new site, and its row is left stale
        tree = lr.ast.parse(orig)
        fn, pat = next((f, c.args[0]) for f in tree.body if isinstance(f, lr.ast.FunctionDef) for c in lr.ast.walk(f)
                       if isinstance(c, lr.ast.Call) and lr._dotted(c.func) == "re.sub" and isinstance(c.args[0], lr.ast.Constant))
        lines = orig.splitlines(True)
        at = sum(len(x) for x in lines[:pat.end_lineno - 1]) + pat.end_col_offset - 1   # before the literal's closing quote
        edited = orig[:at] + "X" + orig[at:]
        pr, _ = run("", replace=(orig, edited))
        ok(any("new text rewrite" in x and f" {fn.name} " in x for x in pr) and any("stale baseline row" in x and fn.name in x for x in pr),
           f"editing an inherited re.sub's pattern in {fn.name}: a new site and a stale row ({[x[:80] for x in pr]})")
        # a removed site: stale row refused; shrink drops it; a new site stays refused after shrink
        named = [s[1] for s in lr.scan(HERE) if s[0] == "gen_llm.py" and s[1] != "<module>"]
        top = next(f for f in tree.body if isinstance(f, lr.ast.FunctionDef) and f.name in named)
        lines[top.lineno - 1] = lines[top.lineno - 1].replace(f"def {top.name}(", f"def _renamed_for_test_{top.name}(", 1)
        gone = "".join(lines)
        b2 = os.path.join(t, "baseline")
        shutil.copy(BASE, b2)
        open(gen, "w").write(gone)
        pr, _ = lr.check(kit, b2, hrx)
        ok(any("stale baseline row" in x for x in pr), f"a function's sites renamed away: stale rows refused ({len(pr)} problems)")
        n0 = sum(1 for ln in open(b2) if not ln.startswith("#"))
        r = subprocess.run([sys.executable, MOD, "shrink", kit, "--baseline", b2], capture_output=True, text=True)
        n1 = sum(1 for ln in open(b2) if not ln.startswith("#"))
        pr, _ = lr.check(kit, b2, hrx)
        ok(r.returncode == 0 and n1 < n0 and all("stale" not in x for x in pr) and all("_renamed_for_test_" in x for x in pr),
           f"shrink drops the stale rows ({n0} -> {n1}) and admits nothing: the renamed sites stay refused ({len(pr)})")
        r = subprocess.run([sys.executable, MOD, "init", kit, "--baseline", b2], capture_output=True, text=True)
        ok(r.returncode == 1, "init refuses to overwrite an existing baseline")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
