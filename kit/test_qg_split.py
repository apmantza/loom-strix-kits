"""Static tests of qg_split.program's fail-closed rule (HRX-like static-1127/upsw). Recurrence each case prevents: the split is skipped while a
%qg writer other than the converted f16 m256 store can run on that path (q / gate keep the previous epoch's values: wrong
attention, silently), or while something else still reads %qg. Mirrors test_epi_fuse.py (E200: the same two-branch projection).
usage: python3 test_qg_split.py (exit 1 on a failure)."""
import re
import sys

import qg_split

HEAD = """command.def public @llm_advance() launch() {
  %pf_big = index.cmp uge, %t, %pf_min : index
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
MV = "kernel.launch @g_mv_q5_k[%nQ2, %t](%nQ2, %t, %h16q, %wq, %qg) : [index, index](index, index, buffer, buffer, buffer)"
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
    # another reader of %qg in the program: no layer is converted
    "extra_qg_reader": (LIVE + SPLIT + "  kernel.launch @g_probe[%t](%t, %qg, %y) : [index](index, buffer, buffer)\n", 0, 1),
}


def main():
    bad = 0
    for name, (body, want, want_ref) in CASES.items():
        text = HEAD + body + TAIL
        out, n, refused = qg_split.program(text)
        ok = n == want and refused == want_ref
        if want:
            ok = ok and out.count("@g_gemm_f16_store_m256_qgs[") == want and out.count("%qgs_run = ") == 1 \
                and out.count("  scf.if %qgs_run {") == want and "%h16, %s_h, %w16, %q, %gate)" in out \
                and "@g_gemm_q5_k_store_m256[" in out and out.count("@g_split_qgate_lv[") == want and arity_ok(out)
        else:
            ok = ok and out == text
        bad += not ok
        print(("PASS " if ok else "FAIL ") + f"{name}: converted {n}, refused {refused}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
