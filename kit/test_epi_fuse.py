"""Static tests of epi_fuse.resid's fail-closed rule (HRX-like E200). Recurrence each case prevents:
upstream's resid_epi converts the last m256 launch before a norm; on our f16-scratch path the quantized branch also stores
%y (m128 / m256 under `scf.if %pf_big` else), so a converted block whose x-only norm runs where a writer kept storing y
drops that projection from the residual stream. usage: python3 test_epi_fuse.py (exit 1 on a failure)."""
import sys

import epi_fuse

HEAD = """command.def public @llm_advance() launch() {
  %pf_big = index.cmp uge, %t, %pf_min : index
  %huge = scalar.andi %t_full256, %t_ge256 : i1
"""
NORM = """  scf.if %mm {
    kernel.launch @g_post_add_norm_rs_live_lv[%t](%t, %info, %y, %wp, %x, %wn, %h, %s_h, %h16) : [index](index, buffer, buffer, view<5120xf32>, buffer, view<5120xf32>, buffer, buffer, buffer)
  } else {
    kernel.launch @g_post_add_norm[%t](%t, %y, %wp, %x, %wn, %h) : [index](index, buffer, view<5120xf32>, buffer, view<5120xf32>, buffer)
  }
}
"""
F16 = "kernel.launch @g_gemm_f16_store_m256[%tp, %nH](%tp, %k, %nH, %o16, %s_o, %w16, %y) : [index, index](index, index, index, buffer, buffer, buffer, buffer)"
Q256 = "kernel.launch @g_gemm_q5_k_store_m256[%tp, %nH](%tp, %k, %nH, %o16, %s_o, %wq, %y) : [index, index](index, index, index, buffer, buffer, buffer, buffer)"
Q128 = "kernel.launch @g_gemm_q5_k_store_m128[%tp, %nH](%tp, %k, %nH, %o16, %s_o, %wq, %y) : [index, index](index, index, index, buffer, buffer, buffer, buffer)"
MV = "kernel.launch @g_mv_q5_k[%nH, %t](%nH, %t, %f16q, %wq, %y) : [index, index](index, index, buffer, buffer, buffer)"
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
    # the served shape: one f16 m256 store exactly under mm/pf_big/big/huge; the quantized branch cannot run there
    "live_shape": (LIVE, 1),
    # a second y writer on the converted path (a quantized m256 store inside the pf_big branch): not convertible to f16 acc
    "second_writer_on_path": (LIVE.replace("    scf.if %pf_big {", "    scf.if %pf_big {\n      " + Q256, 1), 0),
    # another reader of y between the projection and the norm
    "extra_y_reader": (LIVE + "  kernel.launch @g_probe[%t](%t, %y) : [index](index, buffer)\n", 0),
    # a writer of y under an unknown condition reachable on the converted path
    "unknown_condition": (LIVE + "  scf.if %other {\n    " + F16 + "\n  }\n", 0),
    # the only y writer is an f16 m256 store that also runs when huge and not pf_big (no scratch condition around it): its acc
    # would add into x while the old norm (taken when not pf_big) adds the stale y as well
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
""", 0),
    # no f16 m256 at all
    "no_m256": (LIVE.replace(F16, F16.replace("m256", "m128")), 0),
}


def main():
    bad = 0
    for name, (body, want) in CASES.items():
        text = HEAD + body + NORM
        out, n, refused, fmts = epi_fuse.resid(text)
        ok = n == want and refused == 1 - want and (("acc_m256" in out and "%re_ok" in out) if want else out == text)
        bad += not ok
        print(("PASS " if ok else "FAIL ") + f"{name}: converted {n}, refused {refused}, formats {sorted(fmts)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
