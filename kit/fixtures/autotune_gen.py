#!/usr/bin/env python3
"""kit/test_autotune.py's generator: a three-kernel program whose code the knobs below change, written to argv[1].
  AT_LIVE   values each lane of at_hold keeps live (8: 16 waves/SIMD; 100: 9 waves, no spill; 300: 256 VGPRs and a spill)
  AT_SCALE  at_scale's multiplier (a constant: different code, same cost)
  AT_OTHER  at_other's stored constant (a kernel outside the family: a knob that reaches it is reported, not judged)
  AT_BAD    1: at_scale calls an undefined helper (a compile error)
  AT_RETIRED is named here and in a comment only (no code reads it): kit/test_autotune.py checks prose does not make a knob known"""
# os.environ.get("AT_RETIRED") in a comment is not a read either
import os
import sys

N = int(os.environ.get("AT_LIVE", "8"))
SCALE = os.environ.get("AT_SCALE", "2.0")
OTHER = os.environ.get("AT_OTHER", "1.0")
BAD = os.environ.get("AT_BAD", "0") == "1"

HEAD = '''kernel.def target(@rdna3) export("{k}") @{k}(%n: index) {{
  %c1 = index.constant 1 : index
  %c64 = index.constant 64 : index
  %g = index.div %n, %c64 : index
  kernel.launch.config workgroups(%g, %c1, %c1) workgroup_size(%c64, %c1, %c1) : index
}} launch(%n_arg: index, {args}) {{
  %n = index.assume %n_arg [range(%n_arg, 64, 65536), multiple_of(%n_arg, 64)] : index
  %c64 = index.constant 64 : index
  %wg = kernel.workgroup.id<x> : index
  %lid = kernel.workitem.id<x> : index
  %b = index.mul %wg, %c64 : index
  %i = index.add %b, %lid : index
  %z = index.constant 0 : offset
  %live = index.cmp ult, %i, %n : index
  scf.if %live {{'''
TAIL = '''    %yv = buffer.view %y[%z] : buffer -> view<[%n]xf32>
    view.store {r}, %yv[%i] : f32, view<[%n]xf32>
  }}
  kernel.return
}}
'''

L = ["target.decl @device", "amdgpu.target<gfx11-generic> @rdna3", ""]
# at_hold: N loads, their sum, then each load times a partial sum: every load stays live until the second pass
L.append(HEAD.format(k="at_hold", args="%w: buffer, %y: buffer"))
L.append(f"    %wv = buffer.view %w[%z] : buffer -> view<[%n]x{N}xf32>")
for j in range(N):
    L.append(f"    %j{j} = index.constant {j} : index")
    L.append(f"    %x{j} = view.load %wv[%i, %j{j}] : view<[%n]x{N}xf32> -> f32")
L.append("    %a0 = scalar.addf %x0, %x0 : f32")
L += [f"    %a{j} = scalar.addf %a{j - 1}, %x{j} : f32" for j in range(1, N)]
L.append(f"    %s0 = scalar.mulf %x0, %a{N - 1} : f32")
for j in range(1, N):
    L.append(f"    %p{j} = scalar.mulf %x{j}, %a{N - 1 - j} : f32")
    L.append(f"    %s{j} = scalar.addf %s{j - 1}, %p{j} : f32")
L.append(TAIL.format(r=f"%s{N - 1}"))
L.append(HEAD.format(k="at_scale", args="%w: buffer, %y: buffer"))
L.append("    %wv = buffer.view %w[%z] : buffer -> view<[%n]xf32>")
L.append("    %x = view.load %wv[%i] : view<[%n]xf32> -> f32")
L.append(f"    %k = scalar.constant {SCALE} : f32")
L.append("    %r = scalar.mulf %x, %k : f32" if not BAD else "    %r = func.call @at_missing(%x) : (f32) -> (f32)")
L.append(TAIL.format(r="%r"))
L.append(HEAD.format(k="at_other", args="%y: buffer"))
L.append(f"    %one = scalar.constant {OTHER} : f32")
L.append(TAIL.format(r="%one"))
os.makedirs(os.path.dirname(os.path.abspath(sys.argv[1])), exist_ok=True)
open(sys.argv[1], "w").write("\n".join(L))
