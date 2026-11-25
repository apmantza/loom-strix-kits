"""P^T rewrites of the split-K and flash2 attention kernels built on FLASH2_SMX (flash2_smx.py).

vsplit_ph: FLASH2_VSPLIT_PH=1, FLASH2_SMX's P^T half alone on g_attention_vsplit (exact; HRX-like E408, E493).
Fails closed: every anchor must match exactly once.
"""
import re

from flash2_smx import _ph_new


def _once(text, old, new, what):
    n = text.count(old)
    assert n == 1, f"flash2_smx2: {what}: anchor seen {n} times (want 1): {old.strip().splitlines()[0][:100]!r}"
    return text.replace(old, new, 1)


def vsplit_ph(text):
    """FLASH2_VSPLIT_PH=1 (gfx11 g_attention_vsplit, the split-K decode/verify kernel; no peeled copy): FLASH2_SMX's PH alone (P
    rounded to f16 before the xor-16 exchange, 4 packed i32 shuffles per 16-key block instead of 8 f32). Same bits as
    flash2_smx.py states: the per-element RNE fptrunc is the one the interleaved vector had, the rest moves f16 bits. Not FOLD:
    vsplit stores m for g_attention_combine in the scaled units."""
    from flash2_smx import _ph_old
    names = re.findall(r'^kernel\.def (?:target\(@rdna3\) )?export\("(\w+)"\)', text, re.M)
    assert names == ["g_attention_vsplit"], f"FLASH2_VSPLIT_PH: kernels {names} (want g_attention_vsplit alone)"
    for kb in (0, 1):
        text = _once(text, _ph_old(kb, ""), _ph_new(kb, ""), f"vsplit P^T build {kb}")
    left = re.findall(r"%pt[01]_f\b|%pi[01]_\d\b", text)
    assert not left, f"FLASH2_VSPLIT_PH: left over after the rewrite: {left[:4]}"
    return text
