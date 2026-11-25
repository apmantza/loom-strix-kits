"""FLASH2_SMX=1 (gfx11 g_attention_flash2 and g_attention_flash2_partial, and so their F16O_O `_f16o` copies; HRX-like E394, static
study static-1124/f2sm/STUDY.md levers 1 and 3): two exact trims of the softmax in both copies of the peeled tile loop (hot "" and
peeled "_pt"). Not the split-K kernel (g_attention_vsplit: it stores m for the combine, in the scaled units) and not the gfx12 twins.

  PH    P^T is built from f16: each 8-vector of P is rounded to f16 first (the same per-element RNE fptrunc the interleaved
        vector had), its 4 packed i32 words go through the xor-16 exchange (4 shuffles per 16-key block instead of 8), and the
        hi_half selects and the interleave move f16 bits. Same bits in every lane: rounding is elementwise, the rest is data
        movement.
  FOLD  S stays in raw units: the 2^-4 softmax scale is not multiplied into S (16 v_mul per tile); it is folded into the exp2
        constant instead, exp(x / 16) = exp2(x * (log2e / 16)). expf<afn> lowers to x * 0x3fb8aa3b then v_exp_f32; log2e / 16
        is 0x3db8aa3b exactly, and with a power-of-two scale every intermediate is the old one times 16 (round(s - m) =
        16 * round(s/16 - m/16); round(x * c/16) = round(x/16 * c)), so the exp arguments are bit-identical. The running max m
        is then 16x the old one (only the kernel uses it: no m output). Masked scores stay -3e38 in raw units: exp2 of
        <= -2.7e37 is +0, as exp of -3e38 was, and the post-exp select zeroes them anyway. Residual: f32 subnormal scores
        (|s| < 2^-122), where s/16 may flush; their exp is 1.0 either way.

Fails closed: every anchor must match exactly once per (copy, key block), and the scale must be an exact power of two.
"""
import math
import re
import struct
from decimal import Decimal

LOG2E_AFN = struct.unpack("<f", struct.pack("<I", 0x3fb8aa3b))[0]   # the constant expf<afn> lowers to (STUDY.md section 1)


def _once(text, old, new, what):
    n = text.count(old)
    assert n == 1, f"FLASH2_SMX: {what}: anchor seen {n} times (want 1): {old.strip().splitlines()[0][:100]!r}"
    return text.replace(old, new, 1)


def _ph_old(kb, s):
    ls = []
    for i in range(8):
        ls += [f"    %pi{kb}_{i}{s} = vector.extract %p{kb}{s}[%e{i}] : vector<8xf32> -> f32",
               f"    %po{kb}_{i}{s}, %pu{kb}_{i}{s} = kernel.subgroup.shuffle<xor> %pi{kb}_{i}{s}, %exchange, %width : f32, i32, i32"]
    ls += [f"    %partner{kb}{s} = vector.from_elements " + ", ".join(f"%po{kb}_{i}{s}" for i in range(8)) + " : vector<8xf32>",
           f"    %even{kb}{s} = scf.select %hi_half, %partner{kb}{s}, %p{kb}{s} : vector<8xf32>",
           f"    %odd{kb}{s} = scf.select %hi_half, %p{kb}{s}, %partner{kb}{s} : vector<8xf32>",
           f"    %pt{kb}_f{s} = vector.interleave<0> %even{kb}{s}, %odd{kb}{s} : vector<8xf32>, vector<8xf32> -> vector<16xf32>",
           f"    %pt{kb}_h{s} = vector.fptrunc %pt{kb}_f{s} : vector<16xf32> to vector<16xf16>\n"]
    return "\n".join(ls)


def _ph_new(kb, s):
    ls = [f"    %ph{kb}{s} = vector.fptrunc %p{kb}{s} : vector<8xf32> to vector<8xf16>",
          f"    %pw{kb}{s} = vector.bitcast %ph{kb}{s} : vector<8xf16> to vector<4xi32>"]
    for i in range(4):
        ls += [f"    %pwi{kb}_{i}{s} = vector.extract %pw{kb}{s}[%e{i}] : vector<4xi32> -> i32",
               f"    %pwo{kb}_{i}{s}, %pwu{kb}_{i}{s} = kernel.subgroup.shuffle<xor> %pwi{kb}_{i}{s}, %exchange, %width : i32, i32, i32"]
    ls += [f"    %pwp{kb}{s} = vector.from_elements " + ", ".join(f"%pwo{kb}_{i}{s}" for i in range(4)) + " : vector<4xi32>",
           f"    %partner{kb}{s} = vector.bitcast %pwp{kb}{s} : vector<4xi32> to vector<8xf16>",
           f"    %even{kb}{s} = scf.select %hi_half, %partner{kb}{s}, %ph{kb}{s} : vector<8xf16>",
           f"    %odd{kb}{s} = scf.select %hi_half, %ph{kb}{s}, %partner{kb}{s} : vector<8xf16>",
           f"    %pt{kb}_h{s} = vector.interleave<0> %even{kb}{s}, %odd{kb}{s} : vector<8xf16>, vector<8xf16> -> vector<16xf16>\n"]
    return "\n".join(ls)


def smx(text):
    sm = re.findall(r"^  %scale = scalar\.constant (\S+) : f32$", text, re.M)
    assert len(sm) == 1, f"FLASH2_SMX: {len(sm)} %scale constants (want 1)"
    scale = struct.unpack("<f", struct.pack("<f", float(sm[0])))[0]
    mant, _ = math.frexp(scale)
    assert mant == 0.5, f"FLASH2_SMX: softmax scale {sm[0]} is not a power of two (the fold is exact only then)"
    l2s = LOG2E_AFN * scale   # a power-of-two multiple of an f32: exact while it stays a normal f32 (checked)
    assert struct.unpack("<f", struct.pack("<f", l2s))[0] == l2s and abs(l2s) >= 2.0 ** -126, "FLASH2_SMX: log2e x scale not an exact normal f32"
    anchor = "  %scale_v = vector.splat %scale : vector<8xf32>\n"
    text = _once(text, anchor, anchor + f"  %smx_l2s = scalar.constant {Decimal(l2s)} : f32\n"
                 "  %smx_l2s_v = vector.splat %smx_l2s : vector<8xf32>\n", "scale splat")
    for s in ("", "_pt"):
        for kb in (0, 1):
            # FOLD: S unscaled into the mask select, the scale folded into the exp2 constant
            text = _once(text, f"    %sc{kb}{s} = vector.mulf %st{kb}{s}, %scale_v : vector<8xf32>\n", "", f"scale {kb}{s}")
            text = _once(text, f"    %masked{kb}{s} = vector.select %valid{kb}{s}, %sc{kb}{s}, %neg_v : vector<8xf32>\n",
                         f"    %masked{kb}{s} = vector.select %valid{kb}{s}, %st{kb}{s}, %neg_v : vector<8xf32>\n", f"mask {kb}{s}")
            text = _once(text, f"    %pe{kb}{s} = vector.expf<afn> %sh{kb}{s} : vector<8xf32>\n",
                         f"    %shl{kb}{s} = vector.mulf %sh{kb}{s}, %smx_l2s_v : vector<8xf32>\n"
                         f"    %pe{kb}{s} = vector.exp2f<afn> %shl{kb}{s} : vector<8xf32>\n", f"exp {kb}{s}")
            # PH: P^T from f16, packed exchange
            text = _once(text, _ph_old(kb, s), _ph_new(kb, s), f"P^T build {kb}{s}")
        text = _once(text, f"    %alpha{s} = scalar.expf<afn> %m_shift{s} : f32\n",
                     f"    %msl{s} = scalar.mulf %m_shift{s}, %smx_l2s : f32\n    %alpha{s} = scalar.exp2f<afn> %msl{s} : f32\n", f"alpha{s}")
    left = re.findall(r"%sc[01](?:_pt)?\b|%pt[01]_f(?:_pt)?\b|%pi[01]_\d(?:_pt)?\b|expf<afn>|mulf [^\n]*%scale_v\b", text)
    assert not left, f"FLASH2_SMX: left over after the rewrite: {left[:4]}"
    return text
