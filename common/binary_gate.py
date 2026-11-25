#!/usr/bin/env python3
"""Untouched-kernel gate between two packages: every gfx11 kernel the switch was not meant to change must compile to the same
code, and the switch must have changed something it was meant to. Recurrence (HRX-like E372): GEMM256_FLATADDR (kit 718c339)
was meant for the 256-row f16 store GEMM and also rewrote the swiglu GEMM's weight loads (hardcoded %w where the kernel reads
%wsel); end-to-end equality passed and only a later candidate's kernel tests caught it. The gate replaces the per-step
"rewrites present" greps: an --allow pattern that matches no changed kernel means the switch did not apply.

Per kernel (both packages' program/*.loom): a fingerprint of its source closure, i.e. its own text plus every definition it
references (inline helpers, globals, configs, targets), transitively, full-line comments dropped. Loom inlines helpers into
the kernel, so a kernel's own text is NOT a sound proxy for its code: a changed helper leaves every caller's text unchanged.
The closure is: anything the compiler can read for `--root=KERNEL` is reached through a symbol reference. Kernels whose closure
is identical in both packages are not compiled (same compiler, same input: same code). The others are compiled for gfx1151
with --root (about 0.2 s each) and compared on: normalized ISA hash (disassembly without addresses or encodings), VGPRs, SGPRs,
scratch (private segment) and LDS (group segment) from the code object's metadata. --all compiles every kernel (checks the
proxy). _rdna4 variants are skipped (not served on gfx1151; kernel_lint does the same).

Verdicts per kernel: same (closure identical), text-only (closure changed, code identical), CHANGED, NEW, MISSING, ERROR.
Exit 1 (BINARY GATE: FAIL) on: a CHANGED or NEW kernel no --allow pattern matches; a MISSING kernel no --allow-missing pattern matches; scratch > 0 in a compiled
candidate kernel whose base had none (or that is new); an --allow-missing pattern that matches no MISSING kernel; an ERROR (a kernel that does not compile: fail closed); an --allow
pattern that matches no CHANGED / NEW kernel. Exit 2 on usage errors.
usage: binary_gate.py BASE_PKG CAND_PKG [--allow REGEX ...] [--allow-missing REGEX ...] [--all] [--out DIR] [--jobs N] [--json FILE] [--loom-compile PATH]
       (a PKG is a package dir with program/*.loom, or one .loom file)"""
import argparse
import concurrent.futures
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HEAD = re.compile(r"^[A-Za-z]")                 # a top-level definition starts in column 0 with a letter ('}' continues one)
REF = re.compile(r"@([A-Za-z_][\w.$]*)")
KERNEL = re.compile(r'^kernel\.def (?:target\(@\w+\) )?export\("(\w+)"\)')


def items(text):
    """name -> source text of every top-level definition (full-line comments and trailing blanks dropped). The name is the first
    @symbol of the head line after target(@..) and template.def<@..> qualifiers; the kernel's name is its export."""
    out, name, buf = {}, None, []
    def flush():
        if name is not None:
            out[name] = out.get(name, "") + "\n".join(buf).strip() + "\n"
    for line in text.split("\n"):
        if line.lstrip().startswith("//"):
            continue
        line = line.rstrip()
        if HEAD.match(line):
            flush()
            m = KERNEL.match(line)
            if m:
                name = m.group(1)
            else:
                h = re.sub(r"target(?:<[^>]*>)?\(@\w+\)|<@[\w.]+>", "", line)   # low.func.def target<core>(@t) @f: named @f, not @t
                r = REF.search(h)
                name = r.group(1) if r else line.split()[0] + "#" + str(len(out))
            buf = [line]
        elif name is not None:
            buf.append(line)
    flush()
    return out


def closures(text):
    """kernel -> sha256 of its source closure (its text and every definition reached through @references, transitively)"""
    it = items(text)
    def deps(name):
        ds = set()
        for r in REF.findall(it[name]):
            parts = r.split(".")
            for i in range(len(parts), 0, -1):           # @llm.tokens -> llm.tokens, then llm (a config.def of the namespace)
                p = ".".join(parts[:i])
                if p in it and p != name:
                    ds.add(p)
        return ds
    graph = {n: deps(n) for n in it}
    out = {}
    for line in text.split("\n"):
        m = KERNEL.match(line)
        if not m or m.group(1).endswith("_rdna4"):
            continue
        k = m.group(1)
        seen, todo = set(), [k]
        while todo:
            n = todo.pop()
            if n not in seen:
                seen.add(n)
                todo.extend(graph.get(n, ()))
        h = hashlib.sha256()
        for n in sorted(seen):
            h.update(n.encode() + b"\0" + it[n].encode() + b"\0")
        out[k] = h.hexdigest()
    return out


def programs(pkg):
    """the .loom files of a package (program/*.loom) or the one file given"""
    if os.path.isfile(pkg):
        return [pkg]
    fs = sorted(glob.glob(os.path.join(pkg, "program", "*.loom")))
    if not fs:
        raise SystemExit(f"binary_gate: no program/*.loom under {pkg}")
    return fs


def kernels(pkg):
    """kernel -> (program file, closure fingerprint) over a package's programs; a kernel defined in two files is refused"""
    out = {}
    for f in programs(pkg):
        for k, fp in closures(open(f).read()).items():
            if k in out:
                raise SystemExit(f"binary_gate: kernel {k} defined in {out[k][0]} and {f}")
            out[k] = (f, fp)
    return out


def measure(lc, prog, k, out, report=False):
    """compile kernel K of PROG for gfx1151 into OUT: {isa, vgpr, sgpr, scratch, lds}, or {error}; REPORT: Loom's summary
    report beside the code object as OUT/K.json (kernel_lint.lint reads its residency; common/autotune.py)"""
    import hsaco_run
    import loop_census
    hs = os.path.join(out, f"{k}.hsaco")
    rep = ["--compile-report=summary", f"--compile-report-output={os.path.join(out, k + '.json')}"] if report else []
    r = subprocess.run([lc, prog, "--target=amdgpu:gfx1151", f"--root={k}", f"--output={hs}"] + rep, capture_output=True, text=True)
    if r.returncode:
        return {"error": (r.stderr.strip().split("\n") or [""])[0][:200]}
    ins, _ = loop_census.disasm(hs)
    m = hsaco_run.kernel_meta(hs, k)
    return {"isa": hashlib.sha256("\n".join(ins).encode()).hexdigest()[:16], "instructions": len(ins),
            "vgpr": m.get(".vgpr_count"), "sgpr": m.get(".sgpr_count"),
            "scratch": m.get(".private_segment_fixed_size"), "lds": m.get(".group_segment_fixed_size")}


METRICS = ("isa", "vgpr", "sgpr", "scratch", "lds")


def judge(base, cand, measured, allow, allow_missing=()):
    """the verdict rows and the failures. BASE / CAND: kernel -> (file, fingerprint); MEASURED: (side, kernel) -> metrics for
    the kernels compiled; ALLOW: regexes; ALLOW_MISSING: regexes of kernels the candidate may drop (each must match a dropped
    kernel). Pure: the tests drive it with synthetic metrics."""
    rows, fails = [], []
    for k in sorted(set(base) | set(cand)):
        b, c = measured.get(("base", k)), measured.get(("cand", k))
        if k not in cand:
            v = "MISSING"
        elif k not in base:
            v = "ERROR" if c is None or "error" in c else "NEW"
        elif base[k][1] == cand[k][1] and (not (b and c) or ("error" in b and "error" in c)):
            v = "same"                                   # (--all: a root that compiles standalone on neither side, same source)
        elif b is None or c is None or "error" in b or "error" in c:
            v = "ERROR"
        else:
            v = "text-only" if all(b[x] == c[x] for x in METRICS) else "CHANGED"
        allowed = any(re.search(a, k) for a in allow)
        why = []
        if v == "CHANGED" and base[k][1] == cand[k][1]:
            why.append("source closure identical but the code differs (--all): the text proxy is unsound or the compiler nondeterministic")
        if v == "MISSING" and not any(re.search(a, k) for a in allow_missing):
            why.append("missing in the candidate")
        if v in ("CHANGED", "NEW") and not allowed:
            why.append(f"{v.lower()} and no --allow matches")
        if v == "ERROR":
            why.append("did not compile: " + ((c or {}).get("error") or (b or {}).get("error") or "not measured"))
        if c and "error" not in c and (c.get("scratch") or 0) > 0 and not (b and "error" not in b and (b.get("scratch") or 0) > 0):
            why.append(f"new scratch {c['scratch']} B")
        rows.append({"kernel": k, "verdict": v, "allowed": allowed, "base": b, "cand": c, "fail": why})
        fails += [f"{k}: {w}" for w in why]
    for a in allow:
        if not any(r["verdict"] in ("CHANGED", "NEW") and re.search(a, r["kernel"]) for r in rows):
            fails.append(f"--allow {a!r} matched no changed kernel: the switch did not apply")
    for a in allow_missing:
        if not any(r["verdict"] == "MISSING" and re.search(a, r["kernel"]) for r in rows):
            fails.append(f"--allow-missing {a!r} matched no missing kernel")
    return rows, fails


def gate(base_pkg, cand_pkg, allow, out, jobs=2, all_=False, lc=None, allow_missing=()):
    """parse both packages, compile what the closures cannot clear, judge; returns (rows, failures, compiles)"""
    if lc is None:
        from kernel_lint import loom_compile
        lc = loom_compile()
    base, cand = kernels(base_pkg), kernels(cand_pkg)
    work = []
    for k in sorted(set(base) & set(cand)):
        if all_ or base[k][1] != cand[k][1]:
            work += [("base", k, base[k][0]), ("cand", k, cand[k][0])]
    work += [("cand", k, cand[k][0]) for k in sorted(set(cand) - set(base))]
    for side in ("base", "cand"):
        os.makedirs(os.path.join(out, side), exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(jobs) as ex:
        res = list(ex.map(lambda w: measure(lc, w[2], w[1], os.path.join(out, w[0])), work))
    measured = {(w[0], w[1]): r for w, r in zip(work, res)}
    rows, fails = judge(base, cand, measured, allow, allow_missing)
    return rows, fails, len(work)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("base")
    ap.add_argument("cand")
    ap.add_argument("--allow", action="append", default=[], help="regex of kernels the switch may change (repeatable; each must match a change)")
    ap.add_argument("--allow-missing", action="append", default=[], help="regex of kernels the candidate may drop (a model or drafter "
                    "change removes the kernels only it called; repeatable; each must match a missing kernel)")
    ap.add_argument("--all", action="store_true", help="compile every kernel, not only those whose source closure changed")
    ap.add_argument("--out", help="where the code objects go (default: a temporary dir)")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--json", help="write the rows here")
    ap.add_argument("--loom-compile", help="the compiler (default: the machine's VENV)")
    a = ap.parse_args()
    for x in a.allow + a.allow_missing:
        try:
            re.compile(x)
        except re.error as e:
            ap.error(f"--allow {x!r}: {e}")
    tmp = None if a.out else tempfile.TemporaryDirectory(prefix="binary_gate.")
    rows, fails, n = gate(a.base, a.cand, a.allow, a.out or tmp.name, a.jobs, a.all, a.loom_compile, a.allow_missing)
    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    for r in rows:
        if r["verdict"] != "same":
            b, c = r["base"] or {}, r["cand"] or {}
            d = " ".join(f"{x} {b.get(x)}->{c.get(x)}" for x in METRICS[1:] if b.get(x) != c.get(x))
            print(f"  {r['verdict']:9s} {'allowed' if r['allowed'] else '       '} {r['kernel']}  {d}".rstrip())
    print(f"{len(rows)} kernels: " + ", ".join(f"{v} {n_}" for v, n_ in sorted(counts.items())) + f"; {n} compiles")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
    for f in fails:
        print("  FAIL " + f)
    print("BINARY GATE: " + ("FAIL" if fails else "PASS"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
