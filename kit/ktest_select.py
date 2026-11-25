"""Chooses the kernel-unit-test cases a device runs (run_kernel_tests.sh).

usage: ktest_select.py LIB.loom CASES.loom-test OUT.loom-test [ARCH]     (ARCH: gfx1151, gfx1201, ...; default DEVICE_ARCH)

A case launches kernels of the generated library. On a gfx12 device (RDNA4) every launch of a kernel K that has a
`K_rdna4` twin (dual_target.py's convention) goes to the twin; a case launching a gfx11-only kernel there (target(@rdna3)
without a twin: it cannot be compiled for gfx12) is skipped, so one test file covers both architectures. On gfx11
(or LOOM_RDNA4=0) the cases are launched as written. A case whose kernel is not in this build is skipped in both.

       ktest_select.py changed BGATE.json      (common/ktest.sh KTEST_CHANGED)
prints the kernels common/binary_gate.py's rows (--json; common/cand_lint.py writes PKG.lint/bgate.json) report CHANGED or NEW,
comma-separated (empty: nothing changed), for a selected generation (make_kernel_tests.py KTEST_KERNELS: by_kernels). Exit 1 with
the reason when the changed set is unknown (an unreadable file, a row the gate could not judge: ERROR, an unknown verdict): the
caller then runs the full suite, never zero cases."""
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
    for b in re.split(r'(?m)^(?=check\.case )', cases):
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
GATE_VERDICTS = {"same", "text-only", "CHANGED", "NEW", "MISSING"}     # binary_gate.judge's verdicts but ERROR


def launches(case):
    """the kernels a case launches; a kt_K copy counts as K (the kernel it tests)"""
    return {k[3:] if k.startswith("kt_") else k for k in LAUNCH.findall(case)}


def by_kernels(cases, kernels, smoke=SMOKE):
    """(text, kept case names, kept smoke names): the cases of CASES (a cases.loom-test text) that launch one of KERNELS, or are named
    in SMOKE; non-case blocks (helper kernel definitions) are kept. Selection is by the launched kernel, never by the case name: the
    names do not say which kernel they test (11-25: flash2's cases are 'flash_*', 'stale_kv_*', 'nob3_*', 'flataddr_*')."""
    kernels = set(kernels)
    keep, names, sm = [], [], []
    for b in re.split(r'(?m)^(?=check\.case )', cases):
        m = CASE.match(b)
        if not m:
            keep.append(b)
            continue
        if launches(b) & kernels or m.group(1) in smoke:
            keep.append(b)
            names.append(m.group(1))
            if m.group(1) in smoke:
                sm.append(m.group(1))
    return "".join(keep), names, sm


def changed_kernels(path):
    """the kernels binary_gate's rows (a --json file) report CHANGED or NEW; ValueError when the changed set is unknown"""
    import json
    try:
        rows = json.load(open(path))
    except (OSError, ValueError) as e:
        raise ValueError(f"cannot read {path}: {e}")
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) and "kernel" in r and "verdict" in r for r in rows):
        raise ValueError(f"{path}: not binary_gate rows")
    bad = [f"{r['kernel']} {r['verdict']}" for r in rows if r["verdict"] not in GATE_VERDICTS]
    if bad:
        raise ValueError(f"{path}: rows the gate could not judge: {', '.join(bad[:5])}")
    return sorted(r["kernel"] for r in rows if r["verdict"] in ("CHANGED", "NEW"))


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


if __name__ == "__main__":
    if sys.argv[1:2] == ["changed"] and len(sys.argv) == 3:
        try:
            print(",".join(changed_kernels(sys.argv[2])))
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
