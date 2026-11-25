#!/usr/bin/env python3
"""common/ir_trace.py. Recurrences it prevents:
  - an ablation edit landing on the wrong op: six identical `s_barrier` lines in live flash2's prepared Low; an anchor that matches
    several (or none) must be refused, never resolved to the first (HRX-like E511: B1, B2, B3 of the hot loop and the peeled tile);
  - an anchor resolved outside the root kernel (a module holds other kernels and helpers);
  - an edit that makes the Low invalid in a way the assembler may not report clearly: deleting an op whose result is read, a
    terminator, moving an op above its operand's definition, below its result's use, across blocks or past a terminator;
  - an ablation judged against the served binary: the prepared-Low round trip changes the schedule on 0.2.0 (live flash2_f16o
    4805 -> 4808 instructions), so the control must be the unedited reassembly (checked with --program);
  - a bisect calling a property 'introduced' by a pass when its anchors found nothing (U is never 'holds'), or naming a pass
    while the served binary and the reassembled snapshot disagree (E520's wait, checked with --sf / --fixed).
usage: test_ir_trace.py                                   (text-level cases, no compiler)
       test_ir_trace.py --program LIVE.loom --root ROOT [--nob3 NOB3.loom] [--sf SF.loom --fixed FIXED.loom]
                                                          (also compiles: ablate on live, bisect on E520's pair)"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import ir_trace as it  # noqa: E402

fails = 0


def ok(cond, what):
    global fails
    print(("ok  " if cond else "FAIL") + " " + what)
    fails += not cond


def refused(fn, *a):
    try:
        fn(*a)
    except it.Refused as e:
        return str(e)
    return None


MOD = """amdgpu.target<gfx11-generic> @rdna3

low.kernel.def retain target<amdgpu.rdna3_5.core>(@ctx) export("other") workgroup_size(64, 1, 1) @other() asm {
  %w = live_in<amdgpu.workitem_id.x> : reg<amdgpu.vgpr>
  s_barrier
  %only_other = v_mov_b32 7
  s_endpgm
}

low.kernel.def retain target<amdgpu.rdna3_5.core>(@ctx) export("k") workgroup_size(384, 1, 1) @k() asm {
  %workitem = live_in<amdgpu.workitem_id.x> : reg<amdgpu.vgpr>
  %a = v_mov_b32 1
  %b = v_add_u32 %a, %workitem
  s_barrier
  %ld = global_load_b32_vaddr %b
  %c = v_add_u32 %b, %a
  s_barrier
  %pl7_1 = v_wmma_f32_16x16x16_f16 %c, %c, %c
  s_barrier
  %free = v_mov_b32 3
  low.cond_br %c, ^_bb1, ^_bb2 : reg<amdgpu.scc>
^_bb1:
  %d = v_mov_b32 %ld
  low.br ^_bb2
^_bb2:
  s_barrier
  %one_f = v_mov_b32 1065353216
  s_endpgm
}
"""


def L():
    return MOD.split("\n")


def edit(s, lines=None):
    return it.apply_edit(lines or L(), "k", it.parse_edit(s))


def body_text(lines):
    lo, hi = it.body(lines, "k")
    return [l.strip() for l in lines[lo:hi + 1]]


def text_cases():
    # anchors
    r = refused(edit, "delete /^s_barrier$/")
    ok(r is not None and "matches 4 lines" in r, "ambiguous anchor (4 barriers in k) refused, naming the count")
    ok(r is not None and "line 5:" not in r, "the other kernel's barrier is not counted (root scoping)")
    r = refused(edit, "delete /^%only_other = /")
    ok(r is not None and "matches 0 lines" in r, "anchor found only outside the root refused")
    out, _ = edit("delete /^s_barrier$/ prev=/^%pl7_1 = v_wmma/")
    b = body_text(out)
    ok(b.count("s_barrier") == 3 and b[b.index("%pl7_1 = v_wmma_f32_16x16x16_f16 %c, %c, %c") + 1] == "%free = v_mov_b32 3",
       "prev= selects the barrier after %pl7_1, and only that line goes")
    out, _ = edit("delete /^s_barrier$/ next=/^%one_f = /")
    b = body_text(out)
    ok(b.count("s_barrier") == 3 and b[b.index("^_bb2:") + 1].startswith("%one_f"), "next= selects the barrier before %one_f")
    ok(len(out) == len(L()) - 1, "delete removes exactly one line")
    # delete guards
    r = refused(edit, "delete /^%a = /")
    ok(r is not None and "read at line" in r, "deleting an op whose result is read refused")
    r = refused(edit, "delete /^low.br /")
    ok(r is not None and "last op" in r, "deleting a terminator refused")
    out, _ = edit("delete /^%free = /")
    ok("%free = v_mov_b32 3" not in body_text(out), "deleting an unused result is allowed")
    # move guards
    r = refused(edit, "move /^%c = / to=before:/^%b = /")
    ok(r is not None and "operand is defined" in r, "moving above an operand's definition refused")
    r = refused(edit, "move /^%b = / to=after:/^%c = /")
    ok(r is not None and "result is read" in r, "moving below a use of the result refused")
    r = refused(edit, "move /^%ld = / to=before:/^%d = /")
    ok(r is not None and "another block" in r, "moving across a block boundary refused")
    r = refused(edit, "move /^%free = / to=after:/^low.cond_br /")
    ok(r is not None, "moving after a terminator refused")
    out, _ = edit("move /^%ld = / to=after:/^%c = /")
    b = body_text(out)
    ok(b.index("%ld = global_load_b32_vaddr %b") == b.index("%c = v_add_u32 %b, %a") + 1, "a legal move lands after its target")
    r = refused(edit, "move /^%ld = / to=before:/^s_barrier$/")
    ok(r is not None and "target" in r and "matches 4 lines" in r, "an ambiguous move target refused")
    # syntax
    for bad, why in (("remove /x/", "unknown action"), ("delete x", "op not a regex"), ("delete /x/ nth=/2/", "unknown key"),
                     ("move /x/", "move without to="), ("delete /x/ to=after:/y/", "delete with to="), ("delete /(/", "bad regex")):
        ok(refused(it.parse_edit, bad) is not None, f"edit syntax refused: {why}")
    ok(it.parse_edit(r"delete /a\/b/")["op"].pattern == "a/b", "an escaped slash inside a regex")
    # predicates
    P = lambda s, lines: it.evaluate(it.parse_pred(s), lines)[0]  # noqa: E731
    seq = ["global_load_b128 v[66:69]", "x", "s_waitcnt vmcnt(0)", "v_cvt_f16_f32_e32 v127.l, v213", "s_waitcnt vmcnt(0)"]
    fixed = ["global_load_b128 v[66:69]", "s_waitcnt vmcnt(3)", "v_cvt_f16_f32_e32 v127.l, v213", "s_waitcnt vmcnt(0)",
             "v_cvt_f16_f32_e32 v1.l, v2"]
    bt = r"between /^global_load_b128/ /^s_waitcnt vmcnt\(0\)/ /^v_cvt_f16_f32/"
    ok(P(bt, seq) == "H", "between: full wait before the first use after the load holds (E520 shape)")
    ok(P(bt, fixed) == "N", "between: a later full wait (after the first use) does not count (E521 shape)")
    ok(P(bt, ["s_waitcnt vmcnt(0)", "v_cvt_f16_f32 v1"]) == "U", "between: no A line is U, not N or H")
    ok(P(bt, ["global_load_b128 v1", "s_waitcnt vmcnt(0)"]) == "U", "between: no C after A is U")
    ok(P("order /b/ /a/", ["a", "b"]) == "N" and P("order /a/ /b/", ["a", "b"]) == "H", "order")
    ok(P("order /a/ /z/", ["a"]) == "U", "order: missing B is U")
    ok(P("present /q/", ["a"]) == "N" and P("present /a/", ["a"]) == "H", "present")
    for bad in ("between /a/ /b/", "order /a/", "maybe /a/", "order /a/ x=/b/"):
        ok(refused(it.parse_pred, bad) is not None, f"predicate refused: {bad!r}")
    src = ['kernel.def target(@rdna3) export("k") @k(%tokens: index) {', "  %c1 = index.constant 1 : index",
           "} launch(%t: index, %q: buffer) {", "  %vh = vector.load %q[%c1] : view<[4]xi8> -> vector<4xi8>",
           '  scf.if %p { %x = op {s = "}"} }', "}", 'kernel.def export("k2") @k2() {', "  %vh = other", "}"]
    lo, hi = it.body(src, "k")
    ok((lo, hi) == (1, 4), "a source kernel's body spans its launch region (`} launch(...) {`) up to its own closing brace")
    ok(it.evaluate(it.parse_pred("present /^%vh = vector.load/"), [l.strip() for l in src[lo:hi + 1]])[0] == "H",
       "an anchor after `} launch` is found in a source kernel (E520's V loads)")
    ok(it.first_steady(list("NNHH")) == 2 and it.first_steady(list("HNH")) == 2 and it.first_steady(list("HHN")) is None
       and it.first_steady(list("UHH")) == 1 and it.first_steady(list("HHH")) == 0, "first_steady: the pass from which H lasts")


def run(args):
    r = subprocess.run([sys.executable, os.path.join(H, "ir_trace.py")] + args, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def compile_cases(a, d):
    b3 = ["--edit", "delete /^s_barrier$/ prev=/^%pl7_1 = v_wmma/", "--edit", "delete /^s_barrier$/ next=/^%one_f = /"]
    rc, out = run(["ablate", a.program, a.root, "--out", f"{d}/x.hsaco"] + b3)
    ok(rc == 0 and "control -> edited" in out and "opcode delta {'s_barrier': -2}" in out,
       "ablate: B3 removed at Low; control -> edited is exactly s_barrier -2")
    ok(os.path.exists(f"{d}/x.json") and os.path.exists(f"{d}/x.control.hsaco"), "ablate writes the report and the control")
    m = re.search(r"round trip served -> control \(unedited, --pipeline=none\): (.*)", out)
    ok(m is not None, f"ablate reports the round trip ({m.group(1)[:60] if m else 'missing'})")
    rc2, out2 = run(["ablate", a.program, a.root, "--out", f"{d}/y.hsaco", "--edit", "delete /^s_barrier$/"])
    ok(rc2 == 2 and "anchor matches 6 lines" in out2 and not os.path.exists(f"{d}/y.hsaco"), "ablate: ambiguous anchor exits 2, no object")
    # the round trip moves live's second QK chain (offset 8448) behind the first one's loads: a property the reassembly changes
    qk = r"order /^ds_load_b128 \S+, \S+ offset:8448$/ /^ds_load_b128 \S+, \S+ offset:64$/"
    rc3, out3 = run(["bisect", a.program, a.root, "--on", "asm", "--region", "all", "--pred", qk, "--work", f"{d}/rt"])
    ok(rc3 == 2 and "round trip changes this property" in out3, "bisect asm refuses when served and reassembled final disagree")
    if a.nob3:
        lc = it.compiler()
        ev, _ = it.trace(lc, a.nob3, a.root, f"{d}/nob3")
        it.assemble(lc, ev[-1]["path"], f"{d}/nob3.rt.hsaco", report=False)
        ok(open(f"{d}/nob3.rt.hsaco", "rb").read() == open(f"{d}/x.hsaco", "rb").read(),
           "Low edit == the generator's NOB3 through the same path (byte-identical code object)")
        ok(open(ev[-1]["path"]).read() == open(f"{d}/x.loom").read(), "Low edit text == the generator NOB3 snapshot")
    if a.sf and a.fixed:
        pred = r"between /^global_load_b128/ /^s_waitcnt vmcnt\(0\)/ /^v_cvt_f16_f32/"
        rc, out = run(["bisect", a.sf, a.root, "--on", "asm", "--pred", pred, "--work", f"{d}/sf"])
        ok(rc == 0 and "placed at code emission" in out, "bisect asm on E520's STAGE_LIVE x FLATADDR: emission, no pass")
        rc, out = run(["bisect", a.fixed, a.root, "--on", "asm", "--pred", pred, "--work", f"{d}/fx"])
        ok(rc == 0 and "does not hold at the end" in out, "bisect asm on the fixed package: the predicate does not hold")
        ipred = r"order /^%vh_v_k8s = / /^%kc_sh = \S+ %pfks/"
        rc, out = run(["bisect", a.sf, a.root, "--pred", ipred, "--work", f"{d}/sfi"])
        ok(rc == 0 and "holds in the input program" in out, "bisect ir on E520: the V loads precede the guarded convert in the text")
        rc, out = run(["bisect", a.fixed, a.root, "--pred", ipred, "--work", f"{d}/fxi"])
        ok(rc == 0 and "does not hold at the end (N)" in out, "bisect ir on the fixed package: the convert precedes the V loads")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--program")
    ap.add_argument("--root", default="g_attention_flash2_f16o")
    ap.add_argument("--nob3")
    ap.add_argument("--sf")
    ap.add_argument("--fixed")
    a = ap.parse_args()
    text_cases()
    if a.program:
        d = tempfile.mkdtemp(prefix="test_ir_trace.", dir=os.environ.get("TEST_IR_TRACE_TMP"))
        compile_cases(a, d)
        if not fails:
            shutil.rmtree(d)
        else:
            print(f"kept {d}")
    print(f"{fails} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
