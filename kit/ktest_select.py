"""Chooses the kernel-unit-test cases a device runs (run_kernel_tests.sh).

usage: ktest_select.py LIB.loom CASES.loom-test OUT.loom-test [ARCH]     (ARCH: gfx1151, gfx1201, ...; default DEVICE_ARCH)

A case launches kernels of the generated library. On a gfx12 device (RDNA4) every launch of a kernel K that has a
`K_rdna4` twin (dual_target.py's convention) goes to the twin; a case launching a gfx11-only kernel there (target(@rdna3)
without a twin: it cannot be compiled for gfx12) is skipped, so one test file covers both architectures. On gfx11
(or LOOM_RDNA4=0) the cases are launched as written. A case whose kernel is not in this build is skipped in both.

       ktest_select.py changed BGATE.json [--cand PKG] [--base PKG]      (common/ktest.sh KTEST_CHANGED)
prints the kernels common/binary_gate.py's rows (--json; common/cand_lint.py writes PKG.lint/bgate.json) report CHANGED, NEW or
text-only, comma-separated (empty: nothing changed), for a selected generation (make_kernel_tests.py KTEST_KERNELS: by_kernels).
Exit 1 with the reason when the changed set is unknown (an unreadable file, a row the gate could not judge: ERROR, an unknown
verdict, or --cand / --base not the packages verdict.json beside the rows names): the caller then runs the full suite.

       ktest_select.py combine LIB.loom CASES.loom-test OUT.loom             (run_kernel_tests.sh; any step that runs cases itself)
writes LIB + CASES as one loom-test file, reordered by dual_target.encodings_first: the cases sit past the package's ~160k lines, so
their encodings lie beyond hrx-loom's 16-bit SSA encoding ids (ENCODING/004; HRX-like E609, E616). Exit 1 when the reorder refuses.

       ktest_select.py scope [TESTDIR]                                       (run_kernel_tests.sh)
prints 'ktest scope: full', 'ktest scope: subset (...)' or 'ktest scope: unknown (...)' from TESTDIR's ktest_scope.json (write_scope)."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dual_target  # noqa: E402

EXPORT = re.compile(r'^kernel\.def\b[^\n]*?\bexport\("([^"]+)"\)', re.M)
SYMBOL = re.compile(r'^kernel\.def\b[^\n]*?\s@([\w.]+)\(', re.M)
RDNA4_ONLY = re.compile(r'^kernel\.def\b[^\n]*?target\(@rdna4\)[^\n]*?\s@([\w.]+)\(', re.M)
RDNA3_ONLY = re.compile(r'^kernel\.def\b[^\n]*?target\(@rdna3\)[^\n]*?\s@([\w.]+)\(', re.M)


def is_gfx12(arch):
    return bool(arch) and arch.startswith("gfx12")


def select(lib, cases, arch=None, rdna4=None):
    """(text of the cases to run, [skip messages])."""
    rdna4 = dual_target.enabled() if rdna4 is None else rdna4
    gfx12 = is_gfx12(arch)
    defined = set(EXPORT.findall(lib)) | set(SYMBOL.findall(lib))
    defined |= set(EXPORT.findall(cases)) | set(SYMBOL.findall(cases))   # test helper kernels
    gfx11_only = set(RDNA3_ONLY.findall(lib)) - {k[:-len(dual_target.SUFFIX)] for k in defined if k.endswith(dual_target.SUFFIX)}
    gfx12_only = {k for k in RDNA4_ONLY.findall(lib) if k.endswith(dual_target.SUFFIX) and k[:-len(dual_target.SUFFIX)] not in defined}   # K_rdna4 with no K (f16o fusions)
    keep, skipped = [], []
    for b in re.split(r'(?m)^(?=check\.case |kernel\.def )', cases):
        if not b.startswith("check.case"):
            keep.append(b)
            continue
        name = re.search(r'@([\w.]+)', b).group(1)
        if gfx12 and rdna4:
            b = re.sub(r'(kernel\.launch @)([\w.]+)(?=\[)',
                       lambda m: m.group(1) + (m.group(2) + dual_target.SUFFIX if m.group(2) + dual_target.SUFFIX in defined else m.group(2)), b)
        used = set(re.findall(r'kernel\.launch @([\w.]+)', b))
        if used - defined:
            skipped.append(f"SKIP {name}: kernel(s) not in this build: {', '.join(sorted(used - defined))}")
        elif not (gfx12 and rdna4) and used & gfx12_only:
            skipped.append(f"SKIP {name}: gfx12-only kernel(s): {', '.join(sorted(used & gfx12_only))}")
        elif gfx12 and used & gfx11_only:
            skipped.append(f"SKIP {name}: gfx11-only kernel(s) without a _rdna4 twin: {', '.join(sorted(used & gfx11_only))}")
        else:
            keep.append(b)
    return "".join(keep), skipped


def _base(k, defined):
    """K for K and its gfx12 twin K_rdna4 (one kernel identity)."""
    return k[:-len(dual_target.SUFFIX)] if k.endswith(dual_target.SUFFIX) and k[:-len(dual_target.SUFFIX)] in defined else k


LAUNCH = re.compile(r'kernel\.launch @([\w.]+)')
CASE = re.compile(r'check\.case public @(\w+)')
# The smoke set of a selected generation: whichever of these the generator emits for the model (embed and head_argmax always;
# attention_chain for an attention model, gdn_chain for a hybrid). They are the generator's end-to-end chains through the model's
# entry, a layer of each kind and its exit, each launching several kernels in the served order: a change outside the kernels
# binary_gate reports (a shared case helper, the generator's own plumbing) still meets a case.
SMOKE = ("embed", "attention_chain", "gdn_chain", "head_argmax")
# binary_gate.judge's verdicts but ERROR. text-only (source closure changed, code identical) is selected too: the closure holds the
# kernel's launch region, so a grid or workgroup-count change is text-only and only a test that launches it sees it (review F3).
GATE_VERDICTS = {"same", "text-only", "CHANGED", "NEW", "MISSING"}
SELECTED_VERDICTS = ("CHANGED", "NEW", "text-only")
# Test-local copies of a package kernel K the generator emits (make_kernel_tests.py, ktest_variants.py, 11-25): kt_K (a case root
# loom-test can run), ktn_K (E444's stale-KV twins), kto_K / kto_ref_K (f16o reference copies), ktm_plane_K / ktm_wrap_K (FLATADDR's
# mutated negative controls). A launched name that ends in _K for a package kernel K is K's copy whatever its prefix; a kt*_ name
# whose rest is no package kernel is a test helper (kt_kv, kt_fa_poke); test_* are helpers. Anything else is unknown, and a case
# that launches an unknown name is kept (fail closed).
COPY_PREFIXES = ("kt_", "ktn_", "kto_", "kto_ref_", "ktm_plane_", "ktm_wrap_")
HELPER = re.compile(r'kt[a-z]*_|test_')


def launches(case):
    """the kernels a case launches; a kt_K copy counts as K (the kernel it tests; coverage's notion: kto_ copies do not count)"""
    return {k[3:] if k.startswith("kt_") else k for k in LAUNCH.findall(case)}


def defined_kernels(lib):
    """the kernel names a program text defines (export names and symbols)"""
    return set(EXPORT.findall(lib)) | set(SYMBOL.findall(lib))


def kernel_of(name, defined):
    """('pkg' | 'copy', K) for a launched NAME that is package kernel K or a test copy of it, ('helper', None), or ('unknown', None).
    DEFINED: the package's kernel names."""
    if name in defined:
        return "pkg", name
    hits = [k for k in defined if name.endswith("_" + k)]
    if hits:
        return "copy", max(hits, key=len)
    if HELPER.match(name):
        return "helper", None
    return "unknown", None


def case_kernels(case, defined):
    """(package kernels the case tests, directly or through a copy; the unknown names it launches)"""
    got, unknown = set(), set()
    for n in LAUNCH.findall(case):
        kind, k = kernel_of(n, defined)
        if k:
            got.add(k)
        elif kind == "unknown":
            unknown.add(n)
    return got, unknown


def by_kernels(cases, kernels, smoke=SMOKE, defined=None):
    """(text, kept case names, kept smoke names, {'covered': selected kernels a kept case tests, 'unknown': {case: names}}): the
    cases of CASES (a cases.loom-test text) that test one of KERNELS (launch it or a copy of it), or are named in SMOKE, or launch a
    name kernel_of cannot place (fail closed); non-case blocks (helper kernel definitions) are kept. DEFINED: the package's kernel
    names (default: KERNELS; unknown names are then not detected). Selection is by the launched kernel, never by the case name: the
    names do not say which kernel they test (11-25: flash2's cases are 'flash_*', 'stale_kv_*', 'nob3_*', 'flataddr_*')."""
    kernels = set(kernels)
    dfn = set(defined) if defined is not None else kernels
    keep, names, sm, covered, unknown = [], [], [], set(), {}
    for b in re.split(r'(?m)^(?=check\.case |kernel\.def )', cases):
        m = CASE.match(b)
        if not m:
            keep.append(b)
            continue
        got, unk = case_kernels(b, dfn)
        unk = unk if defined is not None else set()
        if got & kernels or m.group(1) in smoke or unk:
            keep.append(b)
            names.append(m.group(1))
            covered |= got & kernels
            if m.group(1) in smoke:
                sm.append(m.group(1))
            if unk:
                unknown[m.group(1)] = sorted(unk)
    return "".join(keep), names, sm, {"covered": covered, "unknown": unknown}


def finish_selection(cases, kernels, defined=None, child=False):
    """make_kernel_tests.py's last step of a selected generation: (kept text, report lines), or SystemExit(3) after printing why
    when the selection must not run (common/ktest.sh then runs the full suite): no smoke case, or a selected kernel no kept case
    tests (each, not any: review F4). CHILD (a KTEST_HEADS_SFX run, whose cases the parent filters again): never refuses."""
    total = len(CASE.findall(cases))
    text, names, sm, info = by_kernels(cases, kernels, defined=defined)
    lines = [f"selected generation: {len(names)} of {total} cases test {', '.join(sorted(kernels)) or 'no listed kernel'} or are "
             f"smoke ({', '.join(sm) or 'none'})"]
    lines += [f"selected generation: kept {c}: it launches {', '.join(u)}, which is neither a package kernel, a copy of one nor a "
              "helper" for c, u in sorted(info["unknown"].items())]
    missing = sorted(set(kernels) - info["covered"])
    why = (["no smoke case"] if not sm else []) + ([f"no case tests {', '.join(missing)}"] if missing else [])
    if why and not child:
        print("\n".join(lines))
        print(f"selected generation: {'; '.join(why)}: refusing (run the full suite)")
        raise SystemExit(3)
    return text, lines


def check_declared(new_cases, declared, defined):
    """make_kernel_tests.kt_section's declaration check: SystemExit when NEW_CASES test a package kernel outside DECLARED (copies
    count as their kernel; helpers aside)."""
    extra = set()
    for c in new_cases:
        extra |= case_kernels(c, defined)[0]
    extra -= set(declared)
    if extra:
        raise SystemExit(f"make_kernel_tests: kt_section({sorted(declared)}) also tests {sorted(extra)}: add them to its declaration")


def cmd_sha(pkg):
    """sha256 of PKG.cmd, None without one (the digest common/cand_lint.py cmd_sha stores in verdict.json)"""
    import hashlib
    try:
        with open(os.path.realpath(pkg) + ".cmd", "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def changed_kernels(path, cand=None, base=None):
    """the kernels binary_gate's rows (a --json file) report CHANGED, NEW or text-only; ValueError when the changed set is unknown.
    CAND / BASE (package dirs): the gate must have been run for them: verdict.json beside the rows (common/cand_lint.py check)
    names its base and cand, and each given one must be the same directory (review F5: a copied path selected another package's
    kernels); the cand's .cmd must hash as it did when the gate ran (verify r2: a gate dir outside PKG.lint outlived a rebuild)."""
    import json
    try:
        with open(path) as fh:
            rows = json.load(fh)
    except (OSError, ValueError) as e:
        raise ValueError(f"cannot read {path}: {e}")
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) and "kernel" in r and "verdict" in r for r in rows):
        raise ValueError(f"{path}: not binary_gate rows")
    bad = [f"{r['kernel']} {r['verdict']}" for r in rows if r["verdict"] not in GATE_VERDICTS]
    if bad:
        raise ValueError(f"{path}: rows the gate could not judge: {', '.join(bad[:5])}")
    if cand is not None or base is not None:
        vf = os.path.join(os.path.dirname(os.path.abspath(path)), "verdict.json")
        try:
            with open(vf) as fh:
                v = json.load(fh)
            ids = {"cand": v["cand"], "base": v["base"]}
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise ValueError(f"no gate identity ({vf}: {e})")
        for side, want in (("cand", cand), ("base", base)):
            if want is not None and os.path.realpath(ids[side]) != os.path.realpath(want):
                raise ValueError(f"{path} was made for {side} {ids[side]}, not {want}")
        if cand is not None and (v.get("cand_cmd_sha") is None or v.get("cand_cmd_sha") != cmd_sha(cand)):
            raise ValueError(f"{path} was made for another build of {cand} (its .cmd changed since, or no .cmd)")
    return sorted(r["kernel"] for r in rows if r["verdict"] in SELECTED_VERDICTS)


# Launched kernels hrx-loom 0.2.0's loom-test cannot run as a case root (loom-kb topic sdk, "kernel tests"): reported apart.
UNTESTABLE = {
    r"_listed$": "indirect dispatch (the launch config reads its count from a buffer)",
    r"^g_silu_mul(_lv)?$": "SwiGLU g_silu_mul: the vector bound is proven only for the program's constant width",
}


def untestable(name):
    return next((why for pat, why in UNTESTABLE.items() if re.search(pat, name)), None)


def coverage(lib, selected):
    """(launched, untested): the kernels the library's programs launch (command, template and func bodies; a K / K_rdna4 pair
    counts once) and those of them no selected case launches."""
    defined = set(EXPORT.findall(lib)) | set(SYMBOL.findall(lib))
    launched = set()
    for d in re.split(r'(?m)^(?=\S+\.(?:def|decl) )', lib):
        if d.startswith(("command.def", "template.def", "func.def")):
            launched |= {_base(k, defined) for k in re.findall(r'kernel\.launch @([\w.]+)', d)}
    tested = {_base(k, defined) for k in launches(selected)}   # kt_K: a copy of K
    return sorted(launched), sorted(launched - tested)


# The generation's scope, written beside the cases (make_kernel_tests.py) and printed by run_kernel_tests.sh as 'ktest scope: full |
# subset (...) | unknown'; HRX-like scripts/ktest_verdict.sh refuses anything but full unless the step admits it (KTEST_SCOPE_OK), so a
# go-live check cannot pass on a selected run (C11, E532).
SCOPE_FILE = "ktest_scope.json"


def write_scope(out, kernels, kept, total):
    """record OUT's generation: KERNELS None (every case: full) or the selected kernels (subset), with KEPT of TOTAL cases"""
    import json
    rec = {"scope": "full"} if kernels is None else {"scope": "subset", "kernels": sorted(kernels)}
    rec.update(cases=kept, of=total)
    with open(os.path.join(out, SCOPE_FILE), "w") as fh:
        json.dump(rec, fh)


def scope_line(testdir):
    """the run log's scope line for TESTDIR: full and subset only from a record that says so; anything else is unknown"""
    import json
    try:
        with open(os.path.join(testdir, SCOPE_FILE)) as fh:
            r = json.load(fh)
        if r.get("scope") == "full":
            return "ktest scope: full"
        if r.get("scope") == "subset":
            return f"ktest scope: subset ({r['cases']} of {r['of']} cases; kernels {','.join(r['kernels']) or 'none (smoke only)'})"
        why = f"unrecognized record {json.dumps(r)[:200]}"
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        why = f"no readable {SCOPE_FILE}: {e}"
    return f"ktest scope: unknown ({why})"


def combine(lib, cases):
    """LIB + CASES as one loom-test text, reordered so every encoding-typed line sits within hrx-loom's 16-bit value ids
    (dual_target.encodings_first; ValueError when it cannot)"""
    return dual_target.encodings_first(lib + cases)


if __name__ == "__main__":
    if sys.argv[1:2] == ["combine"]:
        if len(sys.argv) != 5:
            sys.exit(__doc__)
        try:
            text = combine(open(sys.argv[2]).read(), open(sys.argv[3]).read())
        except ValueError as e:
            sys.exit(f"ktest_select: combine refused: {e}")
        open(sys.argv[4], "w", newline="\n").write(text)
        sys.exit(0)
    if sys.argv[1:2] == ["scope"]:
        print(scope_line(sys.argv[2] if len(sys.argv) > 2 else "."))
        sys.exit(0)
    if sys.argv[1:2] == ["changed"]:
        import argparse
        ap = argparse.ArgumentParser(prog="ktest_select.py changed")
        ap.add_argument("cmd")
        ap.add_argument("bgate")
        ap.add_argument("--cand")
        ap.add_argument("--base")
        a = ap.parse_args()
        try:
            print(",".join(changed_kernels(a.bgate, a.cand, a.base)))
        except ValueError as e:
            sys.exit(f"ktest_select: changed set unknown: {e}")
        sys.exit(0)
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    arch = sys.argv[4] if len(sys.argv) > 4 else os.environ.get("DEVICE_ARCH", "")
    text, skipped = select(open(sys.argv[1]).read(), open(sys.argv[2]).read(), arch)
    open(sys.argv[3], "w", newline="\n").write(text)
    print(*skipped, sep="\n") if skipped else None
    launched, untested = coverage(open(sys.argv[1]).read(), text)
    sdk = [k for k in untested if untestable(k)]
    rest = [k for k in untested if not untestable(k)]
    print(f"COVERAGE {len(launched) - len(untested)}/{len(launched)} launched kernels have a case"
          + (f"; untested {', '.join(rest)}" if rest else "") + (f"; not runnable by loom-test (SDK) {', '.join(sdk)}" if sdk else ""))
