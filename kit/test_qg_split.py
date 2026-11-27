"""Static tests of qg_split.program's fail-closed rule (HRX-like static-1127/upsw). Recurrence each case prevents: the split is skipped while a
%qg writer other than the converted f16 m256 store can run on that path (q / gate keep the previous epoch's values: wrong
attention, silently), or while something else still reads %qg. Mirrors test_epi_fuse.py (E200: the same two-branch projection).
usage: python3 test_qg_split.py (exit 1 on a failure)."""
import re
import sys

import qg_split

HEAD = """command.def public @llm_advance() launch() {
  %mm = index.cmp uge, %t, %c5_mm : index
  %pf_big = index.cmp uge, %t, %pf_min : index
  %big = index.cmp uge, %t, %c128 : index
  %huge = scalar.andi %t_full256, %t_ge256 : i1
"""
SPLIT = """  scf.if %gemm {
    kernel.launch @g_split_qgate_lv[%t](%t, %qg, %q, %gate, %info) : [index](index, buffer, buffer, buffer, buffer)
  } else {
    kernel.launch @g_split_qgate[%t](%t, %qg, %q, %gate) : [index](index, buffer, buffer, buffer)
  }
"""
TAIL = "}\n"


def arity_ok(out):
    """every converted launch's operand type list has one type per operand (11-27 pkg-qgs: the two buffer types were appended after
    the closing paren, PARSE/003 at loom-compile, which no static case caught)"""
    for line in out.split("\n"):
        if "@g_gemm_f16_store_m256_qgs[" not in line:
            continue
        m = re.search(r"\]\((.*)\) : \[[^\]]*\]\((.*)\)$", line)
        if not m or len(m.group(1).split(", ")) != len(m.group(2).split(", ")):
            return False
        # %qg is replaced by %q, %gate: the _qgs kernel's launch ABI is the store's with %y2 added (11-27 pkg-qgs r2: 9 values passed
        # to its 8 ABI arguments, STRUCTURE/013, while the static case expected '%qg, %q, %gate')
        ops = m.group(1).split(", ")
        if "%qg" in ops or ops[-2:] != ["%q", "%gate"] or len(ops) != 8:
            return False
    return True
F16 = "kernel.launch @g_gemm_f16_store_m256[%tp, %nQ2](%tp, %nH, %nQ2, %h16, %s_h, %w16, %qg) : [index, index](index, index, index, buffer, buffer, buffer, buffer)"
Q256 = "kernel.launch @g_gemm_q5_k_store_m256[%tp, %nQ2](%tp, %nH, %nQ2, %h16, %s_h, %wq, %qg) : [index, index](index, index, index, buffer, buffer, buffer, buffer)"
Q128 = "kernel.launch @g_gemm_q5_k_store_m128[%tp, %nQ2](%tp, %nH, %nQ2, %h16, %s_h, %wq, %qg) : [index, index](index, index, index, buffer, buffer, buffer, buffer)"
MV = "kernel.launch @g_mv_q5_k_k5120_t4_q8_r2[%nQ2, %t](%nQ2, %t, %h16q, %wq, %qg) : [index, index](index, index, buffer, buffer, buffer)"
R_LAST = "  kernel.launch @g_probe[%t](%t, %y, %qg) : [index](index, buffer, buffer)\n"   # reads %qg as its last operand
R_NAME = "  kernel.launch @g_split_qgate_probe[%t](%t, %qg, %y) : [index](index, buffer, buffer)\n"
R_COMMENT = "  kernel.launch @g_probe[%t](%t, %qg, %y) : [index](index, buffer, buffer) // g_split_qgate\n"
LIVE = f"""  scf.if %mm {{
    scf.if %pf_big {{
      scf.if %big {{
        scf.if %huge {{
          {F16}
        }} else {{
          {F16.replace('m256', 'm128')}
        }}
      }} else {{
        {F16.replace('_m256', '')}
      }}
    }} else {{
      scf.if %big {{
        scf.if %huge {{
          {Q256}
        }} else {{
          {Q128}
        }}
      }} else {{
        {Q128}
      }}
    }}
  }} else {{
    {MV}
  }}
"""
CASES = {
    # the served shape (pkg-e480 llm.loom, layer 3): one f16 m256 store exactly under mm / pf_big / big / huge
    "live_shape": (LIVE + SPLIT, 1, 0),
    # two attention layers: both convert, %qgs_run defined once
    "two_layers": (LIVE + SPLIT + LIVE + SPLIT, 2, 0),
    # a second %qg writer on the skipped path (a quantized m256 store inside the pf_big branch)
    "second_writer_on_path": (LIVE.replace("    scf.if %pf_big {", "    scf.if %pf_big {\n      " + Q256, 1) + SPLIT, 0, 1),
    # a writer under an unknown condition reachable on the skipped path
    "unknown_condition": (LIVE + "  scf.if %other {\n    " + F16 + "\n  }\n" + SPLIT, 0, 1),
    # the f16 m256 store also runs when huge and not pf_big (no scratch condition around it)
    "f16_m256_without_pf_big": (f"""  scf.if %mm {{
    scf.if %big {{
      scf.if %huge {{
        {F16}
      }} else {{
        {Q128}
      }}
    }} else {{
      {Q128}
    }}
  }} else {{
    {MV}
  }}
""" + SPLIT, 0, 1),
    # no f16 m256 at all
    "no_m256": (LIVE.replace(F16, F16.replace("m256", "m128")) + SPLIT, 0, 1),
    # another reader of %qg between two layers: no layer is converted
    "extra_qg_reader": (LIVE + SPLIT + "  kernel.launch @g_probe[%t](%t, %qg, %y) : [index](index, buffer, buffer)\n" + LIVE + SPLIT, 0, 2),
    # keeps-review r1 (qgs_gap.txt HOLE shapes): a launch naming %qg after the last split block stops the build, whatever its name or
    # operand position (the split it would read after was skipped); readers between layers refuse every layer
    "last_operand_reader_after_last_split": (LIVE + SPLIT + R_LAST, "stop", 0),
    "last_operand_reader_between_layers": (LIVE + SPLIT + R_LAST + LIVE + SPLIT, 0, 2),
    "substring_named_reader_after_split": (LIVE + SPLIT + R_NAME, "stop", 0),
    "comment_substring_reader_after_split": (LIVE + SPLIT + R_COMMENT, "stop", 0),
    # the split exemption is by exact kernel name, not a substring of the line (r2: r1 exempted any line containing g_split_qgate)
    "substring_named_reader_between_layers": (LIVE + SPLIT + R_NAME + LIVE + SPLIT, 0, 2),
    "comment_substring_reader_between_layers": (LIVE + SPLIT + R_COMMENT + LIVE + SPLIT, 0, 2),
    # %qg used by a statement that is not a launch (a view a reader could take): unrecognized, the build stops
    "qg_view_between_layers": (LIVE + SPLIT + "  %qv = buffer.view %qg[%c0] : buffer\n"
                               "  kernel.launch @g_probe[%t](%t, %qv, %y) : [index](index, buffer, buffer)\n" + LIVE + SPLIT, "stop", 0),
    # the store runs under the skipped path's conditions and one more (a recipe switch around it): when %other is false the split
    # is skipped and %q / %gate keep the previous epoch's values (keeps-review 11-27, mutant q5: the subset check removed)
    "extra_condition_inside_path": (LIVE.replace("          " + F16, "          scf.if %other {\n            " + F16 + "\n          }", 1)
                                    + SPLIT, 0, 1),
    # a f16 m256 store that also reads %qg: the program-wide reader rule refuses every layer before the per-store operand rule runs
    # (keeps-review 11-27, mutants q1 / q2 survive: that rule is unreachable; this case records it)
    "f16_m256_reads_qg": (LIVE.replace(F16, F16.replace("%h16, %s_h", "%qg, %s_h"), 1) + SPLIT, 0, 1),
    # layer 2's split block does not match SPLIT (re-indented): layers 2 and 3 fall into one segment, so layer 2 is never converted
    # and the census undercounts silently. The build stops (keeps-review 11-27 F1: the split launches are counted against the blocks)
    # two convertible stores before one split block (a store duplicated in the layer): refused, not converted (keeps-review 11-27,
    # mutant q3: len(conv) == 1 -> >= 1 converted the first and left the second writing %qg, which nothing reads once the split is
    # skipped; with two different stores the later one's values were the ones the split copied)
    "two_stores_one_layer": (LIVE.replace("          " + F16, "          " + F16 + "\n          " + F16, 1) + SPLIT, 0, 1),
    "split_unmatched": (LIVE + SPLIT + LIVE + "".join("  " + l + "\n" for l in SPLIT.splitlines()) + LIVE + SPLIT, "stop", 0),
}
# the conditions the LIVE fixture's converted store is nested under: %qgs_ok must be exactly their AND (keeps-review 11-27 F2, mutant
# q6: %qgs_ok was hand-written as %pf_big AND %huge, true wherever the split's own path is wider than the store's)
STORE_PATH = {"%mm", "%pf_big", "%big", "%huge"}


def and_leaves(out, name):
    """the operands %qgs_ok is the AND of, following only the pass's own %qgs_* temporaries"""
    m = re.search(rf"(?m)^  {name} = scalar\.andi (%\w+), (%\w+) : i1$", out)
    if not m:
        return None
    leaves = set()
    for x in m.groups():
        sub = and_leaves(out, x) if x.startswith("%qgs_") else {x}
        if sub is None:
            return None
        leaves |= sub
    return leaves


def main():
    bad = 0
    for name, (body, want, want_ref) in CASES.items():
        text = HEAD + body + TAIL
        try:
            out, n, refused = qg_split.program(text)
        except SystemExit as e:
            ok = want == "stop" and str(e).startswith("QG_SPLIT: ")
            bad += not ok
            print(("PASS " if ok else "FAIL ") + f"{name}: stopped ({e})")
            continue
        ok = n == want and refused == want_ref
        if want == "stop":
            ok = False
        elif want:
            ok = ok and out.count("@g_gemm_f16_store_m256_qgs[") == want and out.count("%qgs_run = ") == 1 \
                and out.count("  scf.if %qgs_run {") == want and "%h16, %s_h, %w16, %q, %gate)" in out \
                and "@g_gemm_q5_k_store_m256[" in out and out.count("@g_split_qgate_lv[") == want and arity_ok(out)
            leaves = and_leaves(out, "%qgs_ok")
            if leaves != STORE_PATH:
                ok = False
                print(f"     %qgs_ok is the AND of {leaves}, expected {sorted(STORE_PATH)}")
            ok = ok and "  %qgs_run = scf.select %qgs_ok, %qgs_f, %qgs_t : i1\n" in out
        else:
            ok = ok and out == text
        bad += not ok
        print(("PASS " if ok else "FAIL ") + f"{name}: converted {n}, refused {refused}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
