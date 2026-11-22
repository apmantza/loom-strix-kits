"""Shared pieces of the kit's compile-and-count tests (test_iq4_lookup, test_word_codes, test_kernel_lint): the machine's
loom-compile, a helper's text out of a generator file, and a one-group-per-lane kernel around a K-quant code helper."""
import os
import re
import sys


def loom_compile():
    """the machine's loom-compile (one resolver for the kit's tests and common/kernel_lint.py)"""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
    from kernel_lint import loom_compile as lc
    return lc()


def helper(src, name):
    a = src.index(f"func.def inline @{name}(")
    return src[a:src.index("\n}\n", a) + 3]


def kernel(helper, f32):
    """a kernel storing helper(w, block * BYTES, group) for one 32-value group per lane"""
    ty, n = ("vector<32xf32>", 128) if f32 else ("vector<32xi8>", 32)
    call = (f"%codes = func.call @{helper}(%w, %origin, %grp) : (buffer, offset, index) -> ({ty})" if f32 else
            f"%codes, %s0, %s1 = func.call @{helper}(%w, %origin, %grp) : (buffer, offset, index) -> ({ty}, f32, f32)")
    nbytes = 210 if "q6" in helper else 110
    return f"""
kernel.def export("t_{helper}") @t_{helper}(%n: index) {{
  %c1 = index.constant 1 : index
  %c255 = index.constant 255 : index
  %c256 = index.constant 256 : index
  %padded = index.add %n, %c255 : index
  %groups = index.div %padded, %c256 : index
  kernel.launch.config workgroups(%groups, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%n_arg: index, %w: buffer, %out: buffer) {{
  %n = index.assume %n_arg [range(%n_arg, 1, 262144)] : index
  %c0 = index.constant 0 : index
  %c8 = index.constant 8 : index
  %c256 = index.constant 256 : index
  %wg = kernel.workgroup.id<x> : index
  %lid = kernel.workitem.id<x> : index
  %base = index.mul %wg, %c256 : index
  %gid = index.add %base, %lid : index
  %live = index.cmp ult, %gid, %n : index
  scf.if %live {{
    %blk = index.div %gid, %c8 : index
    %grp = index.rem %gid, %c8 : index
    %bb = index.constant {nbytes} : offset
    %origin = index.scale %blk, %bb : index, offset -> offset
    {call}
    %ob = index.constant {n} : offset
    %off = index.scale %gid, %ob : index, offset -> offset
    %ov = buffer.view %out[%off] : buffer -> view<{ty[7:-1]}>
    vector.store %codes, %ov[%c0] : {ty}, view<{ty[7:-1]}>
  }}
  kernel.return
}}
"""
