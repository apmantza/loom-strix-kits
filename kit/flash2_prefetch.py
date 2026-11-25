"""Software prefetch of the next K/V tile for the gfx12 flash2s kernels (gen_flash2s.py, FLASH2_PF=1, gfx12 twin only).

The tile loop of gen_flash2s.py stages a tile right before it uses it: every wave of the workgroup runs the paged-KV lookup
(two dependent global loads), the K/V loads, a wait, the LDS stores, a barrier, and only then the matrix work. The 16 waves
of a workgroup are barrier-locked, so the whole workgroup waits on memory together (about 650 instructions and two or three
global round trips per 16-key tile at head_dim 512, 16 of them WMMA).

This pass moves the loads of tile kt + 1 behind the first barrier of tile kt (the tile's own K/V are already in workgroup
memory), keeping the loaded vectors in two loop-carried values that the next iteration stores. The loads are in flight
during the S / softmax / O work of tile kt. The prologue loads tile 0; the peeled last tile (flash2_peel.py) takes its
values from the loop's results; the last loop iteration's prefetch is predicated off (no tile kt + 1).

Applied to the generated kernel text (before the peel), so the gfx11 text is untouched.
"""
import os
import re

HEADER = re.compile(r"^(?P<ind> *)%m_final, %l_final, (?P<outs>[^=]*) = scf\.for %kt = \[(?P<lo>%\w+) to (?P<hi>%\w+) step %c1\]\((?P<args>.*)\) -> \((?P<types>.*)\) \{$")
INNER = ("kvals", "vvals", "kp", "k_offset", "v_offset", "k_cache", "v_cache", "kh", "vh")


KH = ("kh_pg_v", "kh_pg", "kh_ph32", "kh_phb", "kh_ph", "kh_sb", "kh_wi", "kh_sl", "kh_ps")   # flash2_trim.kvhoist's per-tile names


def _suffix(lines, tag):
    names = "|".join(sorted(INNER + KH, key=len, reverse=True))
    return [re.sub("%(" + names + r")\b", lambda m: "%" + m.group(1) + tag, ln) for ln in lines]


def prefetch(text):
    lines = text.split("\n")
    hi = next(i for i, ln in enumerate(lines) if HEADER.match(ln))
    m = HEADER.match(lines[hi])
    assert m.group("lo") == "%c0" and m.group("hi") == "%key_tiles", "this pass writes a [%c0, %key_tiles) loop; the split-K kernel uses prefetch_split"
    ind = m.group("ind")
    assert lines[hi + 1].strip().startswith("%tile_first = index.madd %kt"), lines[hi + 1]
    assert lines[hi + 2].strip() == "scf.if %st_live {", lines[hi + 2]
    ib = next(i for i in range(hi + 3, len(lines)) if lines[i].strip().startswith("kernel.barrier"))
    assert lines[ib - 1] == ind + "  }", repr(lines[ib - 1])
    stage = lines[hi + 2:ib]                      # scf.if %st_live { ... }
    vt = re.search(r"vector<(\d+)xf16>", "\n".join(stage)).group(0)
    kp = next(i for i, ln in enumerate(stage) if ln.strip().startswith("%key_pos = "))
    kv = next(i for i, ln in enumerate(stage) if ln.strip().startswith("%key_valid = "))
    l0 = next(i for i, ln in enumerate(stage) if ln.strip().startswith("%kvals, %vvals = scf.if %key_valid"))
    l1 = next(i for i in range(l0 + 1, len(stage)) if stage[i] == ind + "    }")
    limit = re.search(r"index\.cmp ule, %key_pos, (%\w+)", stage[kv]).group(1)
    zeros = re.search(r"scf\.yield (%kz), (%kz)", "\n".join(stage[l0:l1 + 1])).group(1)
    load = stage[l0:l1 + 1]
    rest = [ln.replace("%kvals", "%pfk").replace("%vvals", "%pfv") for ln in stage[l1 + 1:]]   # the stores
    new_stage = [stage[0]] + rest

    def emit(tag, first, extra, pad):
        o = [f"{pad}%key_pos{tag} = index.add {first}, %st_key : index",
             f"{pad}%key_valid{tag}_0 = index.cmp ule, %key_pos{tag}, {limit} : index"]
        valid = f"%key_valid{tag}_0"
        if extra:
            o.append(f"{pad}%key_valid{tag} = scalar.andi %key_valid{tag}_0, {extra} : i1")
            valid = f"%key_valid{tag}"
        body = _suffix(load, tag)
        body = [re.sub(r"%key_pos\b", f"%key_pos{tag}", ln).replace("scf.if %key_valid ", f"scf.if {valid} ") for ln in body]
        o += body
        return o

    # the prologue: tile 0
    pro = [f"{ind}%pfk_p0, %pfv_p0 = scf.if %st_live -> ({vt}, {vt}) {{"]
    pro += ["  " + ln for ln in emit("_p0", "%key_low", None, ind + "  ")]
    pro += [f"{ind}  scf.yield %kvals_p0, %vvals_p0 : {vt}, {vt}", f"{ind}}} else {{", f"{ind}  scf.yield {zeros}, {zeros} : {vt}, {vt}", f"{ind}}}"]
    # the next tile's loads, behind the first barrier
    nxt = [f"{ind}%pf_nt = index.add %kt, %c1 : index",
           f"{ind}%pf_has = index.cmp ult, %pf_nt, %key_tiles : index",
           f"{ind}%pf_first = index.madd %pf_nt, %ckt, %key_low : index",
           f"{ind}%pfk_n, %pfv_n = scf.if %st_live -> ({vt}, {vt}) {{"]
    nxt += ["  " + ln for ln in emit("_n", "%pf_first", "%pf_has", ind + "  ")]
    nxt += [f"{ind}  scf.yield %kvals_n, %vvals_n : {vt}, {vt}", f"{ind}}} else {{", f"{ind}  scf.yield {zeros}, {zeros} : {vt}, {vt}", f"{ind}}}"]
    # the loop
    head = (f"{ind}%m_final, %l_final, {m.group('outs').strip()}, %pfk_f, %pfv_f = scf.for %kt = [%c0 to %key_tiles step %c1]"
            f"({m.group('args')}, %pfk = %pfk_p0 : {vt}, %pfv = %pfv_p0 : {vt}) -> ({m.group('types')}, {vt}, {vt}) {{")
    yi = next(i for i in range(ib, len(lines)) if lines[i].lstrip().startswith("scf.yield %m_new, %l_new,"))
    y = lines[yi]
    y = re.sub(r"^(.*?) : (f32, f32, .*)$", lambda mm: f"{mm.group(1)}, %pfk_n, %pfv_n : {mm.group(2)}, {vt}, {vt}", y)
    out = lines[:hi] + pro + [head, lines[hi + 1]] + new_stage + lines[ib:ib + 1] + nxt + lines[ib + 1:yi] + [y] + lines[yi + 1:]
    return "\n".join(out)


def prefetch_uncond(text):
    """FLASH2_PF11 on gfx11 (L18, E54): the same move as prefetch(), but the next tile's loads are UNCONDITIONAL. In prefetch() the
    loads sit inside scf.if %st_live / scf.if %key_valid, and the backend joins the branch results right away: the disassembly waits
    vmcnt(0) right behind the loads, before any of the tile's WMMAs, so nothing overlaps (pf16 1.83x the time of base). Here every
    lane loads: a key past the limit (or past the last tile) reads the clamped position min(key, limit), whose page exists, and the
    lanes beyond the staging slots read their clamped segment (st_seg). The zeros the original staging gives masked keys are applied
    at the store in the next iteration (select on the recomputed key_valid), so workgroup memory holds what it held before."""
    lines = text.split("\n")
    hi = next(i for i, ln in enumerate(lines) if HEADER.match(ln))
    m = HEADER.match(lines[hi])
    assert m.group("lo") == "%c0" and m.group("hi") == "%key_tiles", "this pass writes a [%c0, %key_tiles) loop; the split-K kernel uses prefetch_split"
    ind = m.group("ind")
    assert lines[hi + 1].strip().startswith("%tile_first = index.madd %kt"), lines[hi + 1]
    assert lines[hi + 2].strip() == "scf.if %st_live {", lines[hi + 2]
    ib = next(i for i in range(hi + 3, len(lines)) if lines[i].strip().startswith("kernel.barrier"))
    assert lines[ib - 1] == ind + "  }", repr(lines[ib - 1])
    stage = lines[hi + 2:ib]
    vt = re.search(r"vector<(\d+)xf16>", "\n".join(stage)).group(0)
    kv = next(i for i, ln in enumerate(stage) if ln.strip().startswith("%key_valid = "))
    kpl = next(i for i, ln in enumerate(stage) if ln.strip().startswith("%key_pos = "))
    l0 = next(i for i, ln in enumerate(stage) if ln.strip().startswith("%kvals, %vvals = scf.if %key_valid"))
    l1 = next(i for i in range(l0 + 1, len(stage)) if stage[i] == ind + "    }")
    limit = re.search(r"index\.cmp ule, %key_pos, (%\w+)", stage[kv]).group(1)
    zeros = re.search(r"scf\.yield (%kz), (%kz)", "\n".join(stage[l0:l1 + 1])).group(1)
    yl = next(i for i in range(l0, l1) if stage[i].strip().startswith("scf.yield %kh, %vh"))
    body = [ln for ln in stage[l0 + 1:yl] if not ln.strip().startswith("%kp = index.assume")]
    assert any("vector.load %k_cache" in ln for ln in body) and any("vector.load %v_cache" in ln for ln in body), body
    rest = stage[l1 + 1:]                                   # the K store and the V^T store loop
    # the store of tile kt: zeros for masked keys, as the original staging yields them
    store = [stage[0], stage[kpl], stage[kv],
             f"{ind}    %pfk_s = scf.select %key_valid, %pfk, {zeros} : {vt}",
             f"{ind}    %pfv_s = scf.select %key_valid, %pfv, {zeros} : {vt}"]
    store += [ln.replace("%kvals", "%pfk_s").replace("%vvals", "%pfv_s") for ln in rest]

    def emit(tag, first, pad):
        o = [f"{pad}%key_pos{tag} = index.add {first}, %st_key : index",
             f"{pad}%kpc{tag} = index.min %key_pos{tag}, {limit} : index",
             f"{pad}%kp{tag} = index.assume %kpc{tag} [range(%kpc{tag}, 0, 262143)] : index"]
        o += [pad + ln.strip() for ln in _suffix(body, tag)]
        return o

    pro = emit("_p0", "%key_low", ind)
    nxt = [f"{ind}%pf_nt = index.add %kt, %c1 : index", f"{ind}%pf_first = index.madd %pf_nt, %ckt, %key_low : index"]
    nxt += emit("_n", "%pf_first", ind)
    head = (f"{ind}%m_final, %l_final, {m.group('outs').strip()}, %pfk_f, %pfv_f = scf.for %kt = [%c0 to %key_tiles step %c1]"
            f"({m.group('args')}, %pfk = %kh_p0 : {vt}, %pfv = %vh_p0 : {vt}) -> ({m.group('types')}, {vt}, {vt}) {{")
    yi = next(i for i in range(ib, len(lines)) if lines[i].lstrip().startswith("scf.yield %m_new, %l_new,"))
    y = re.sub(r"^(.*?) : (f32, f32, .*)$", lambda mm: f"{mm.group(1)}, %kh_n, %vh_n : {mm.group(2)}, {vt}, {vt}", lines[yi])
    out = lines[:hi] + pro + [head, lines[hi + 1]] + store + lines[ib:ib + 1] + nxt + lines[ib + 1:yi] + [y] + lines[yi + 1:]
    return "\n".join(out)


def prefetch_split(text):
    """FLASH2_PF11_SPLIT=1 (L18, E55): split staging, Strata's attn_prompt order on the gfx11 kernel. K and V of a tile are loaded
    at different times, so at most one of them is held in registers (prefetch_uncond holds both across the loop and spills at
    32-key tiles):
      top of tile kt:  store K(kt) (loaded during tile kt-1), issue V(kt)'s loads           -> barrier 1 (K visible)
      S = K Q^T, S partial store, V^T(kt) store (V(kt)'s loads were in flight during the S MMAs) -> barrier 2 (S and V^T visible)
      issue K(kt+1)'s loads, softmax, O += V^T P^T (K(kt+1) in flight)                       -> barrier 3 (end of tile)
    Barrier count unchanged. Stores are safe: K(kt) is written after barrier 2 of tile kt-1 (every S read of K(kt-1) is done),
    V^T(kt) after barrier 3 of tile kt-1 (every PV read of V^T(kt-1) is done). Loads are unconditional at the clamped key
    position (prefetch_uncond's reason); masked keys get the zeros the original staging gives them, at the store."""
    lines = text.split("\n")
    hi = next(i for i, ln in enumerate(lines) if HEADER.match(ln))
    m = HEADER.match(lines[hi])
    ind = m.group("ind")
    assert lines[hi + 2].strip() == "scf.if %st_live {", lines[hi + 2]
    ib = next(i for i in range(hi + 3, len(lines)) if lines[i].strip().startswith("kernel.barrier"))
    stage = lines[hi + 2:ib]
    vt = re.search(r"vector<(\d+)xf16>", "\n".join(stage)).group(0)
    kv = next(i for i, ln in enumerate(stage) if ln.strip().startswith("%key_valid = "))
    kpl = next(i for i, ln in enumerate(stage) if ln.strip().startswith("%key_pos = "))
    l0 = next(i for i, ln in enumerate(stage) if ln.strip().startswith("%kvals, %vvals = scf.if %key_valid"))
    l1 = next(i for i in range(l0 + 1, len(stage)) if stage[i] == ind + "    }")
    limit = re.search(r"index\.cmp ule, %key_pos, (%\w+)", stage[kv]).group(1)
    zeros = re.search(r"scf\.yield (%kz), (%kz)", "\n".join(stage[l0:l1 + 1])).group(1)
    yl = next(i for i in range(l0, l1) if stage[i].strip().startswith("scf.yield %kh, %vh"))
    body = [ln.strip() for ln in stage[l0 + 1:yl] if not ln.strip().startswith("%kp = index.assume")]
    rest = stage[l1 + 1:]
    kst = [i for i, ln in enumerate(rest) if "vector.store %kvals, %k_view" in ln]
    assert len(kst) == 1, rest
    kstore = rest[kst[0]].strip()
    vstore = [ln for i, ln in enumerate(rest) if i != kst[0] and ln != ind + "  }"]   # the V^T store loop
    assert any("%vt_view" in ln for ln in vstore), vstore
    # the second barrier of the tile: right before the partner's S partial is read
    sp = next(i for i in range(ib + 1, len(lines)) if re.search(r"%sp0\w* = vector\.fragment\.load<result> %s_view", lines[i]))
    b2 = max(i for i in range(ib + 1, sp) if lines[i].strip().startswith("kernel.barrier"))

    def loads(tag, first, which):
        o = [f"{ind}%key_pos{tag} = index.add {first}, %st_key : index",
             f"{ind}%kpc{tag} = index.min %key_pos{tag}, {limit} : index",
             f"{ind}%kp{tag} = index.assume %kpc{tag} [range(%kpc{tag}, 0, 262143)] : index"]
        keep = [ln for ln in body if not (which == "k" and ("%v_cache" in ln.split("=")[0] or ln.startswith("%vh ")))
                and not (which == "v" and ("%k_cache" in ln.split("=")[0] or ln.startswith("%kh ")))]
        o += [ind + ln for ln in _suffix(keep, tag)]
        return o

    top = [f"{ind}%key_pos = index.add %tile_first, %st_key : index".replace("%key_pos =", "%key_pos ="),
           f"{ind}{stage[kv].strip()}"]
    assert stage[kpl].strip() == "%key_pos = index.add %tile_first, %st_key : index", stage[kpl]
    # FLASH2_PF11_NOIF=1: the stores without scf.if %st_live. Loom's wait planner drains outstanding loads where an scf.if
    # rejoins (E55: a minimal kernel shows vmcnt(0) after the join), which waited for V(kt) right after its loads. Lanes past
    # the staging slots have a clamped dim segment (st_seg = min(seg, last)): they store the same values to the same slots.
    noif = os.environ.get("FLASH2_PF11_NOIF", "0") == "1"
    if noif:
        top += [f"{ind}%pfk_s = scf.select %key_valid, %pfk, {zeros} : {vt}", f"{ind}{kstore.replace('%kvals', '%pfk_s')}"]
    else:
        top += [f"{ind}scf.if %st_live {{",
                f"{ind}  %pfk_s = scf.select %key_valid, %pfk, {zeros} : {vt}",
                f"{ind}  {kstore.replace('%kvals', '%pfk_s')}",
                f"{ind}}}"]
    carry = os.environ.get("FLASH2_PF11_CARRY", "0") == "1"   # E55 (a): V(kt) at the offset K(kt) was loaded from (same page)
    if carry:
        vline = next(ln for ln in body if ln.startswith("%v_offset = "))
        vc = next(ln for ln in body if ln.startswith("%v_cache = "))
        vl = next(ln for ln in body if ln.startswith("%vh = "))
        top += [ind + _suffix([vline.replace("%k_offset", "%pfo")], "_v")[0], ind + _suffix([vc], "_v")[0], ind + _suffix([vl], "_v")[0]]
    else:
        top += loads("_v", "%tile_first", "v")
    if noif:
        vblock = [f"{ind}%pfv_s = scf.select %key_valid, %vh_v, {zeros} : {vt}"]
        vblock += [ln[2:].replace("%vvals", "%pfv_s") for ln in vstore]
    else:
        vblock = [f"{ind}scf.if %st_live {{", f"{ind}  %pfv_s = scf.select %key_valid, %vh_v, {zeros} : {vt}"]
        vblock += [ln.replace("%vvals", "%pfv_s") for ln in vstore] + [f"{ind}}}"]
    nxt = [f"{ind}%pf_nt = index.add %kt, %c1 : index", f"{ind}%pf_first = index.madd %pf_nt, %ckt, %key_low : index"]
    nxt += loads("_n", "%pf_first", "k")
    lo, hi_b = m.group("lo"), m.group("hi")
    assert lo == "%c0", "prefetch_split writes the prefill flash2 loop (from tile 0)"
    pro = loads("_p0", "%key_low", "k")
    if carry:
        head = (f"{ind}%m_final, %l_final, {m.group('outs').strip()}, %pfk_f, %pfo_f = scf.for %kt = [{lo} to {hi_b} step %c1]"
                f"({m.group('args')}, %pfk = %kh_p0 : {vt}, %pfo = %k_offset_p0 : offset) -> ({m.group('types')}, {vt}, offset) {{")
    else:
        head = (f"{ind}%m_final, %l_final, {m.group('outs').strip()}, %pfk_f = scf.for %kt = [{lo} to {hi_b} step %c1]"
                f"({m.group('args')}, %pfk = %kh_p0 : {vt}) -> ({m.group('types')}, {vt}) {{")
    yi = next(i for i in range(ib, len(lines)) if lines[i].lstrip().startswith("scf.yield %m_new, %l_new,"))
    y = re.sub(r"^(.*?) : (f32, f32, .*)$", lambda mm: f"{mm.group(1)}, %kh_n{', %k_offset_n' if carry else ''} : {mm.group(2)}, {vt}{', offset' if carry else ''}", lines[yi])
    out = (lines[:hi] + pro + [head, lines[hi + 1]] + top + lines[ib:b2] + vblock + [lines[b2]] + nxt
           + lines[b2 + 1:yi] + [y] + lines[yi + 1:])
    return "\n".join(out)
