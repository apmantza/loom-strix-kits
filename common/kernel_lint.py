#!/usr/bin/env python3
"""Static lint of compiled gfx11 kernels: per kernel, the hot loop's instruction mix and the compiler's residency, flagged
against thresholds that each caught a real defect. A flag is a candidate to read, not a verdict: read the loop before acting.
Flags (the recurrence each one names); shares are of the hot loop's common path (nested loops excluded: rare masked paths):
  select    compares + selects >= 25 % of the common path, with equality compares against >= 8 distinct constants (an index
            tested against each table entry): a vector.table.lookup scalarised into a select chain because the index range was
            hidden (E229/E230: IQ4 masked as i32 words, 960 of 1,544 loop instructions, 15 constants). Selects that are the
            algorithm (index scatter, butterflies, quantizers: llama.cpp, E244) compare against no constants and are not flagged.
  repack    packed 16-bit shifts (v_pk_lshrrev/lshlrev_b16) >= 8 and byte extracts (v_bfe_*) >= 32 in the hot loop: byte
            vectors shifted by a runtime amount, every byte extracted and repacked (E233: q6_k / q3_k codes, 616 vs 290).
  scratch   any scratch load/store in the hot loop: a spill; on gfx11 spills share vmcnt with the loads (E230: 2,464 B).
  copies    a run of >= COPY_RUN register-to-register moves in the hot loop (moves of constants are initialisation, E251) (gaps of up to 3 other instructions): the allocator copying a
            carried value or a branch's results (E179 / ROCm/hrx-system#3542: flash2's 45-move run before its back-edge).
  join      a run of >= JOIN_RUN moves right after an exec-mask restore, anywhere in the kernel: a value-yielding scf.if whose
            branches leave the result in different registers, copied at the join (E242: a prefetch under scf.if; E247: the norm's
            idle-lane zero branch, 61-64 moves per pass). Fix shape: compute on a clamped index and select.
  serial    (blind check E284: 11 of 12 hits confirmed, all material) in any loop, >= SERIAL_LINKS independent links in a run of lone loads: each issued with nothing pending on its counter
            and retired before the next load of that counter (vmcnt for vector loads, lgkmcnt for scalar ones), the
            next load's address not computed from the previous load's data. One load in flight pays the whole latency per load
            (E109: KV8 loads retired 2-26 instructions after issue; E254: g_attention_split's 7 per pass, llama.cpp NVFP4
            mul_mat_q's per-byte scales). Pointer chases and scratch reloads (the scratch flag) are not counted.
  [addr64]  a NOTE until a served A/B prices its fix (E345): in the hot loop's common path, min(global loads taking a 64-bit per-lane
            VGPR address `v[a:b], off`, 64-bit carry adds v_add_co_ci_u32) >= 2: the loop rebuilds 64-bit load addresses every
            iteration where a uniform base + loop-invariant 32-bit lane offset (the SGPR-base form) would not (the f16 GEMMs:
            39 -> 15 VALU, 248 -> 209 instructions). Fire 11-23 on live: every GEMM (38) and flash2 / flash2_partial / vsplit; no matvec.
  (an 'ldsser' note, WMMAs each right after an lgkmcnt(0) on a just-loaded fragment, was removed 11-23: its lever was measured four
  times on the 256-row GEMMs and never gained (E293, E309, E335, E340): their WMMA phase already runs at the hardware rate, E341.)
  (a 'barriers' note, >= 8 straight-line barriers, was removed after a blind check: 1 of 9 hits on llama.cpp were serialised
  reductions, the rest tile staging or a peeled last iteration; E284. Barrier counts stay in the kit's own kernel tests.)
  [reread] a NOTE (blind check E284: 2 of 5 hits confirmed): >= REREAD global loads repeating an earlier load's operands while its address registers are unchanged and no store
            intervened: the same bytes read twice (E278: Strata's Inkling depth mix reads each state twice; llama.cpp's
            flash_attn_ext_vec at head 256 with f16 KV repeats 24 loads). Column-free: r['reread'].
  [tiny]    a NOTE (with --profile): launched >= TINY_CALLS times in the profiled request with a median call under TINY_US: a
            launch-bound small kernel, a fusion candidate (E278: Inkling's short convolutions, ~190 launches per token).
  [outlier] a NOTE, not a flag (E251: 0 of 2 confirmed on llama.cpp, where one template spans formats of different cost):
            non-MMA instructions per MMA (or dot) > OUTLIER x the median of its peers (same family, place, MMA kind and MMA
            count, >= 3 kernels): a kernel doing much more per multiply than its siblings (E233: q6_k's matvec 616 vs q5_k's 261).
  regress:  (with --baseline) worse than the same kernel in another build: a new spill, the prefetch now drained, fewer
            waves per SIMD, or the hot loop / its longest move run >= 10 % longer (E226: Loom 0.3.0's vmcnt(0) + 32-move shift;
            E230: a rewrite's spill; E236: a variant's residency drop).
Reported, not flagged: non-MMA instructions per MMA, moves, VGPRs, waves/SIMD, VGPRs to free for the next residency tier
(Loom's summary report beside the code object, KERNEL.json), pf ('drain' when a vmcnt(0) sits between the loop's last load and
its first WMMA), and with --package the kernel's launch sites in the package. Calibrated on 225 served kernels (E237 / E241):
v_mul_lo_u32 (address math) and a drain alone are not defects.
Decoding goes through loop_census.disasm (ROCm 10.2 llvm-objdump; it refuses undecoded words).
Also reported: LDS waits (lgkmcnt) per MMA (E61), the shortest load-to-wait distance and the loads still in flight at the
back-edge (E109), the extra LDS bank-service rounds of Loom's gfx1151 bank model (whole kernel, not the hot loop: read the report's
bank_service_groups before acting; E249: the f16 GEMMs' 48 extra rounds were the epilogue's once-per-tile store), and with --profile
each kernel's served ms.
A code object holding many kernels (another engine's build: *.co / *.hsaco) is linted kernel by kernel (lint_object).
usage: kernel_lint.py DIR_OR_HSACO ... [--baseline DIR] [--profile RUN] [--flagged] [--strict]
       kernel_lint.py --package PKG | --program FILE.loom ... --out DIR [--loom-compile PATH] [--roots REGEX] [--jobs N] [...]
         (compiles every gfx11 kernel root into DIR first; two compilers: run twice with --loom-compile and pass the first
          run's DIR as --baseline of the second; --strict: exit 1 when anything is flagged)"""
import argparse
import collections
import concurrent.futures
import glob
import json
import os
import re
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop_census import cover, disasm, loops, moves  # noqa: E402

FLAGS = ("select", "repack", "scratch", "copies", "join", "serial")
MM = ("v_wmma", "v_dot4", "v_dot2", "v_dot8")
COPY_RUN = 16
JOIN_RUN = 32
SERIAL_LINKS = 4
TINY_CALLS = 1000
REREAD = 2
TINY_US = 10.0
VMLOAD = re.compile(r"(?:global|buffer|flat)_load_")
SMLOAD = re.compile(r"s_(?:load|buffer_load)_")
REG = re.compile(r"\b([vs])(\d+)\b|\b([vs])\[(\d+):(\d+)\]|\b(vcc|exec|scc|m0)(?:_lo|_hi)?\b")
NO_DEF = re.compile(r"(?:global|buffer|flat|scratch)_store|ds_store|ds_write|s_waitcnt|s_cbranch|s_branch|s_barrier|s_nop|s_delay_alu|"
                    r"s_endpgm|s_cmp|s_bitcmp|v_cmpx|s_clause|s_sleep|s_set|s_trap|s_wait")
OUTLIER = 2.0
GROW = 1.10


def loom_compile():
    """the machine's loom-compile: VENV from the environment or the machine env file; exits when there is none"""
    venv = os.environ.get("VENV")
    if not venv:
        f = os.environ.get("LSK_ENV") or os.path.expanduser("~/.config/loom-strix-kits/env")
        m = re.search(r"(?m)^VENV=(\S+)", open(f).read()) if os.path.exists(f) else None
        venv = m.group(1) if m else None
    p = os.path.join(venv, "bin", "loom-compile") if venv else None
    if not (p and os.access(p, os.X_OK)):
        sys.exit("kernel_lint: no loom-compile (set VENV)")
    return p


def common_path(ins, L, t, e):
    """indices of loop [t, e] outside every loop nested in it"""
    inner = set()
    for a, z in L:
        if t <= a and z <= e and (a, z) != (t, e):
            inner.update(range(a, z + 1))
    return [j for j in range(t, e + 1) if j not in inner]


def hot(ins, addr):
    """the loop whose own common path holds the most WMMA / dot instructions (ties: the longest), else None. Counting the
    common path, not the whole range, picks a nested inner loop over the outer shell around it (E237: the 16-token swiglu
    matvecs keep their dot4s in an inner loop)."""
    L = loops(ins, addr)
    best = None
    for t, e in L:
        mm = sum(1 for j in common_path(ins, L, t, e) if ins[j].startswith(MM))
        key = (mm, e - t)
        if best is None or key > best[0]:
            best = (key, t, e)
    return (best[1], best[2], L) if best else None


def copy_moves(x):
    """register-to-register moves an instruction performs (v_mov / v_dual_mov halves whose source is a VGPR): a move of a
    constant or an SGPR initialises a register, it does not copy one (E251: llama.cpp's accumulator zero-inits read as copies)"""
    halves = [h.strip() for h in x.split("::")]
    n = 0
    for h in halves:
        m = re.match(r"(?:v_dual_mov_b32|v_mov_b(?:16|32)(?:_e32|_e64)?)\s+v\d+(?:\.[lh])?,\s*([^\s,]+)", h)
        if m and re.match(r"v\d+(?:\.[lh])?$|v\[", m.group(1)):            # 16-bit halves too (gfx11.5 true16: v_mov_b16 vN.l, vM.h)
            n += 1
    return n


def regs(text):
    """registers named in an operand text: v12, s[4:5] -> s4 s5, vcc / exec / scc / m0"""
    out = set()
    for m in REG.finditer(text):
        if m.group(1):
            out.add(m.group(1) + m.group(2))
        elif m.group(3):
            out.update(f"{m.group(3)}{k}" for k in range(int(m.group(4)), int(m.group(5)) + 1))
        else:
            out.add(m.group(6))
    return out


def defs_uses(x):
    """(registers written, registers read) of one instruction, approximately (the first operand is the destination; enough for
    the taint of one load's data); VOPD halves joined by '::' together"""
    if "::" in x:
        d, u = set(), set()
        for h in x.split("::"):
            a, b = defs_uses(h.strip())
            d |= a
            u |= b
        return d, u
    op, _, rest = x.partition(" ")
    ops = [o.strip() for o in re.split(r",(?![^\[]*\])", rest)] if rest.strip() else []
    if not ops or NO_DEF.match(op):
        return ({"scc"} if op.startswith(("s_cmp", "s_bitcmp")) else set()), regs(rest)
    return regs(ops[0]), set().union(*[regs(o) for o in ops[1:]])


def serial_links(ins, lo, hi, kind):
    """the most independent links in one run of lone loads of `kind` ('vm' vector, 'sm' scalar) in ins[lo..hi]: a lone load is
    issued with nothing pending on its counter and retired before the next load of that counter is issued; a run is lone loads
    consecutive in load order; a link is independent when the next load's address registers carry none of the previous load's
    data (taint followed through the instructions between them)"""
    is_load = (VMLOAD if kind == "vm" else SMLOAD).match
    cnt = "vmcnt" if kind == "vm" else "lgkmcnt"
    pend, lone, order = [], [], []
    for i in range(lo, hi + 1):
        x = ins[i]
        if is_load(x):
            for p in pend:
                p[2] = True
            pend.append([i, not pend, False])
            order.append(i)
        w = re.search(cnt + r"\((\d+)\)", x) if x.startswith("s_waitcnt") else None
        if w:
            n = int(w.group(1))                          # scalar loads return out of order, but lgkmcnt(N > 0) leaves N pending,
            while len(pend) > n:                         # so it never makes a lone one: counting it as retiring is exact here
                p = pend.pop(0)
                if p[1] and not p[2]:
                    lone.append(p[0])
    pos = {j: k for k, j in enumerate(order)}
    best, cur = 0, []
    for j in lone + [None]:
        if j is not None and cur and pos[j] == pos[cur[-1]] + 1:
            cur.append(j)
            continue
        links = 0
        for a, b in zip(cur, cur[1:]):
            taint = set(defs_uses(ins[a])[0])
            for k in range(a + 1, b):
                d, u = defs_uses(ins[k])
                taint = (taint | d) if u & taint else (taint - d)
            links += not (defs_uses(ins[b])[1] & taint)
        best = max(best, links)
        cur = [j] if j is not None else []
    return best


def longest_move_run(body):
    """the most register-to-register moves in one run (moves at most 3 instructions apart), as loop_census.census groups them"""
    best = cur = 0
    last = None
    for j, x in enumerate(body):
        m = copy_moves(x)
        if not m:
            continue
        cur = cur + m if last is not None and j - last <= 3 else m
        last = j
        best = max(best, cur)
    return best


def join_runs(ins):
    """the longest move run that starts within 3 instructions after an exec-mask restore (the end of a divergent branch):
    a value-yielding scf.if whose branches leave its result in different registers, copied at the join (E242, E247)"""
    best = 0
    restores = [j for j, x in enumerate(ins) if re.match(r"s_(mov|or)_b(32|64) exec(_lo)?,", x)]
    for j in restores:
        n, last = 0, None
        for k in range(j + 1, min(len(ins), j + 400)):
            m = moves(ins[k])                    # a join zero-fills and copies: both count (E247's 64 zero moves + 61 copies)
            if m:
                if last is not None and k - last > 3:
                    break
                if last is None and k - j > 4:
                    break
                n += m
                last = k
            elif last is not None and k - last > 3:
                break
        best = max(best, n)
    return best


def residency(js):
    """(VGPRs, waves/SIMD, VGPRs to free for the next better tier) from a Loom summary report, or Nones"""
    if not os.path.exists(js):
        return None, None, None
    tr = json.load(open(js)).get("target_resources") or {}
    res = tr.get("residency") or {}
    lim = res.get("unique_limiting_resource") or {}
    return (((tr.get("vector") or {}).get("final") or {}).get("register_count"), res.get("current_tier"),
            lim.get("reduction_units_to_next_better_tier"))


def bank(js):
    """(conflicted LDS packets, extra bank-service rounds) from a Loom summary report's gfx1151 bank model, or Nones"""
    if not os.path.exists(js):
        return None, None
    st = ((((json.load(open(js)).get("source_low") or {}).get("memory") or {}).get("bank_service") or {}).get("structural") or {})
    return st.get("conflicted_packet_count"), st.get("extra_round_count")


def kernel_symbols(path):
    """(name, start address, size) of every kernel function in a code object, from its symbol table"""
    tool = __import__("loop_census").objdump()
    t = subprocess.run([tool, "-t", path], capture_output=True, text=True, check=True).stdout
    kds = set(re.findall(r"\s(\S+)\.kd$", t, re.M))
    out = []
    for m in re.finditer(r"(?m)^([0-9a-f]+)\s.*?\bF\s+\.text\s+([0-9a-f]+)\s+(?:\.\w+\s+)?(\S+)$", t):
        if m.group(3) in kds:
            out.append((m.group(3), int(m.group(1), 16), int(m.group(2), 16)))
    return out


def demangle(names):
    """C++ names demangled with the ROCm SDK's llvm-cxxfilt (the names unchanged when it is missing)"""
    tool = os.path.join(os.path.dirname(__import__("loop_census").objdump()), "llvm-cxxfilt")
    if not os.access(tool, os.X_OK):
        return list(names)
    r = subprocess.run([tool], input="\n".join(names), capture_output=True, text=True)
    out = r.stdout.split("\n")
    return out[:len(names)] if r.returncode == 0 and len(out) >= len(names) else list(names)


ADDR64 = 2   # E345: g_gemm_f16_store_m256 had 4 such loads and 4 carry adds per iteration; the SGPR-base form has 0


def addr64(b):
    """global loads in the hot loop's common path that take a 64-bit per-lane VGPR address (`global_load ... v[a:b], off`), when the
    loop also does 64-bit carry adds (v_add_co_ci_u32) to build them: min(loads, carry adds). The SGPR-base form (`v_off, s[a:b]`)
    computes the per-lane part once and needs neither (HRX-like E345: the f16 GEMM's activation and weight addresses, 39 -> 15 VALU)."""
    loads = sum(1 for x in b if re.match(r"global_load\w*\s+v\[?\d+[:\d\]]*,\s*v\[\d+:\d+\],\s*off\b", x))
    carries = sum(1 for x in b if x.startswith("v_add_co_ci_u32"))
    return min(loads, carries)


def lint_object(path):
    """every kernel of a code object holding several (another engine's build): each kernel's instructions sliced from the
    disassembly by its symbol's address range, VGPRs from its metadata note, its family from the demangled function name
    (template arguments dropped) so that `outlier` compares instances of one kernel template"""
    syms = kernel_symbols(path)
    ins, addr = disasm(path)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import hsaco_run
    names = demangle([n for n, _, _ in syms])
    rows = []
    for (n, a0, size), dn in zip(syms, names):
        idx = [j for j, x in enumerate(addr) if a0 <= x < a0 + size]
        if not idx:
            continue
        sl, sa = ins[idx[0]:idx[-1] + 1], addr[idx[0]:idx[-1] + 1]
        try:
            vg = hsaco_run.kernel_meta(path, n).get(".vgpr_count")
        except SystemExit:
            vg = None
        r = lint_listing(n, sl, sa, (vg, None, None))
        r["family"] = re.split(r"[<(]", dn, maxsplit=1)[0].split("::")[-1].strip() or n
        r["object"] = os.path.basename(path)
        r["bank_conf"] = r["bank_extra"] = None
        rows.append(r)
    return rows


def lint_file(path):
    """the rows of one code object: one kernel (a Loom compile) or several (another engine's build)"""
    syms = kernel_symbols(path)
    if len(syms) == 1 and syms[0][0] == os.path.basename(path).rsplit(".", 1)[0]:
        return [lint(path)]
    return lint_object(path)


def lint(path):
    ins, addr = disasm(path)
    r = lint_listing(os.path.basename(path)[:-6], ins, addr, residency(path[:-6] + ".json"))
    r["bank_conf"], r["bank_extra"] = bank(path[:-6] + ".json")
    return r


def lint_listing(name, ins, addr, res=(None, None, None)):
    """the lint of one kernel's decoded listing (instructions, byte addresses), as lint() reports it"""
    r = {"kernel": name, "instr": len(ins), "flags": []}
    r["vgpr"], r["waves"], r["to_next"] = res
    h = hot(ins, addr)
    if h is None or not any(ins[j].startswith(MM) for j in common_path(ins, h[2], h[0], h[1])):
        b = full = ins                                                # no multiply loop: the whole kernel (e.g. a dequant pass)
        r["where"] = "kernel"
    else:
        t, e, L = h
        b = [ins[j] for j in common_path(ins, L, t, e)]              # the hot loop's common path
        full = ins[t:e + 1]
        r["where"] = "loop"
    n = lambda *p: sum(1 for x in b if x.startswith(p))
    mm = n("v_wmma"); dot = n("v_dot4", "v_dot2", "v_dot8")
    r.update(loop=len(full), common=len(b), wmma=mm, dot=dot, moves=sum(moves(x) for x in b), run=longest_move_run(b),
             select=n("v_cmp", "v_cndmask"), pk=n("v_pk_lshrrev_b16", "v_pk_lshlrev_b16"), bfe=n("v_bfe_"),
             scratch=sum(1 for x in full if "scratch_" in x))
    per = mm or dot
    r["other_per_op"] = round((len(b) - per) / per, 1) if per else None   # common path
    lgkm = sum(1 for x in b if x.startswith("s_waitcnt") and "lgkmcnt" in x)
    r["lds_waits_per_op"] = round(lgkm / per, 2) if per else None      # E61: lgkmcnt(0) after each LDS read pair before the MMAs
    cv, pend = cover(full, 0, len(full) - 1)
    r["cover_min"], r["in_flight"] = (min(cv) if cv else None), pend     # E109: loads waited right after issue; carried across
    eq_imm = {m.group(1) for x in b for m in [re.match(r"v_cmp_eq_[iu](?:16|32)_e(?:32|64)\s.*,\s*(-?\d+)$", x)] if m}
    r["eq_imm"] = len(eq_imm)
    if r["select"] >= 0.25 * len(b) and r["eq_imm"] >= 8:       # a lookup chain compares its index with each table entry
        r["flags"].append("select")
    if r["pk"] >= 8 and r["bfe"] >= 32:
        r["flags"].append("repack")
    if r["scratch"]:
        r["flags"].append("scratch")
    if r["where"] == "loop" and r["run"] >= COPY_RUN:
        r["flags"].append("copies")
    r["serial"] = max([serial_links(ins, a, z, k) for a, z in loops(ins, addr) for k in ("vm", "sm")] or [0])
    if r["serial"] >= SERIAL_LINKS:
        r["flags"].append("serial")
    r["reread"] = rereads(ins)
    if r["reread"] >= REREAD:
        r.setdefault("notes", []).append("reread")              # E278: the same bytes loaded twice
    r["addr64"] = addr64(b) if r["where"] == "loop" else 0
    if r["addr64"] >= ADDR64:
        r.setdefault("notes", []).append("addr64")              # E345: per-iteration 64-bit address math feeding the loop's global loads
    r["join"] = join_runs(ins)
    if r["join"] >= JOIN_RUN:
        r["flags"].append("join")
    r["pf"] = "-"
    loads = [j for j, x in enumerate(full) if re.match(r"(global|buffer)_load_", x)]
    w = [j for j, x in enumerate(full) if x.startswith("v_wmma")]
    if loads and w:
        before = [j for j in loads if j < w[0]]
        if before:
            r["pf"] = "drain" if any(re.match(r"s_waitcnt.*vmcnt\(0\)", full[j]) for j in range(before[-1], w[0])) else "kept"
    return r


def family(name):
    """the kernel's family from its name's leading words (g_mv, g_gemm, g_attention, g_head, v_gemm ...)"""
    m = re.match(r"([a-z0-9]+_[a-z]+)", name)
    return m.group(1) if m else name


def row_family(r):
    """a row's family: the demangled function name for other engines' kernels (set by lint_object), else family(name)"""
    return r.get("family") or family(r["kernel"])


def peer_key(r):
    """kernels comparable per multiply: same family, place (loop / kernel), MMA kind and MMA count. The family keeps attention
    from being judged against GEMMs with the same WMMA count (E241 calibration)."""
    return (row_family(r), r["where"], "wmma" if r["wmma"] else "dot", r["wmma"] or r["dot"])


def outliers(rows):
    """flag 'outlier' on rows whose non-MMA instructions per MMA exceed OUTLIER x their peers' median (groups of >= 3);
    returns {kernel: (value, peer median, peers)} for the flagged ones"""
    groups = collections.defaultdict(list)
    for r in rows:
        if r.get("other_per_op") is not None:
            groups[peer_key(r)].append(r)
    hit = {}
    for g in groups.values():
        if len(g) < 3:
            continue
        med = statistics.median(x["other_per_op"] for x in g)
        for r in g:
            if med and r["other_per_op"] > OUTLIER * med:
                r.setdefault("notes", []).append("outlier")   # a prompt to read, not a flag (E251: 0 of 2 confirmed on llama.cpp)
                hit[r["kernel"]] = (r["other_per_op"], med, len(g))
    return hit


def regressions(r, base):
    """what got worse in row r against the same kernel's row in another build"""
    out = []
    if r["scratch"] and not base["scratch"]:
        out.append("spill")
    if r["pf"] == "drain" and base["pf"] == "kept":
        out.append("drain")
    if r["waves"] is not None and base["waves"] is not None and r["waves"] < base["waves"]:
        out.append(f"waves{base['waves']}->{r['waves']}")
    if r["loop"] >= GROW * base["loop"] and r["loop"] - base["loop"] >= 8:
        out.append(f"loop{base['loop']}->{r['loop']}")
    if r["run"] >= GROW * base["run"] and r["run"] - base["run"] >= 4:
        out.append(f"run{base['run']}->{r['run']}")
    return out


def launch_sites(texts):
    """kernel -> launch sites in a package's program texts: direct `kernel.launch @K[` outside template bodies, plus every
    `template.apply<@selN_K>` (the dual-target selectors launch K or its gfx12 twin)"""
    sites = collections.Counter()
    for t in texts:
        in_template = False
        for line in t.split("\n"):
            if line.startswith("template.def"):
                in_template = True
            elif in_template and line.startswith("}"):
                in_template = False
            m = re.search(r"kernel\.launch @([\w]+)[\[(]", line)
            if m and not in_template:
                sites[m.group(1)] += 1
            for m in re.finditer(r"template\.apply<@sel\d+_([\w]+)>", line):
                sites[m.group(1)] += 1
    return sites


def compile_programs(progs, out, jobs, lc=None, roots=None):
    """every gfx11 kernel root (matching ROOTS, a regex) of the program files compiled into OUT (code object + summary report)
    with loom-compile LC (default: the machine's); returns the program texts and the roots that failed to compile standalone
    (named, never hidden)"""
    lc = lc or loom_compile()
    os.makedirs(out, exist_ok=True)
    texts, work = [], []
    for prog in progs:
        t = open(prog).read()
        texts.append(t)
        for k in sorted(set(re.findall(r'(?m)^kernel\.def (?:target\(@\w+\) )?export\("(\w+)"\)', t))):
            if not k.endswith("_rdna4") and (roots is None or re.search(roots, k)):
                work.append((prog, k))
    def one(pk):
        prog, k = pk
        r = subprocess.run([lc, prog, "--target=amdgpu:gfx1151", f"--root={k}", f"--output={out}/{k}.hsaco",
                            "--compile-report=summary", f"--compile-report-output={out}/{k}.json"], capture_output=True, text=True)
        return k, r.returncode, r.stderr
    failed = []
    with concurrent.futures.ThreadPoolExecutor(jobs) as ex:
        for k, rc, err in ex.map(one, work):
            if rc:
                failed.append((k, (err.strip().split("\n") or [""])[0][:160]))
    return texts, failed


def served_stats(run):
    """kernel -> (served ms, calls, median us per call) in a profiled run (common/events_summary.py's loader: iree-profile dispatch
    events of one served request), and the run's prompt size"""
    import events_summary
    ev, prompt = events_summary.load(run)
    per = collections.defaultdict(list)
    for e in ev:
        per[e["key"]].append(e["duration_ns"] / 1e3)
    return {k: (sum(v) / 1e3, len(v), statistics.median(v)) for k, v in per.items()}, prompt


def served_ms(run):
    """kernel -> served milliseconds in a profiled run, and the run's prompt size"""
    st, prompt = served_stats(run)
    return collections.Counter({k: v[0] for k, v in st.items()}), prompt


def tiny(rows, stats):
    """note `tiny` on kernels launched at least TINY_CALLS times in the profiled request with a median call under TINY_US: launch-bound
    small kernels, fusion candidates (Inkling's short convolutions would be ~190 launches per token, E278)"""
    for r in rows:
        ms, calls, med = stats.get(r["kernel"], (0, 0, 0))
        if calls >= TINY_CALLS and med < TINY_US:
            r.setdefault("notes", []).append("tiny")


def rereads(ins):
    """global / buffer loads that repeat an earlier load with the same operands while its address registers were not redefined and
    no store intervened: the same bytes read twice (Strata's Inkling depth mix reads each state twice, E278)"""
    n = 0
    seen = {}                                     # load operand text (destination dropped) -> address registers
    for x in ins:
        op = x.split()[0]
        if re.match(r"(?:global|buffer|flat|scratch)_store|buffer_gl|s_dcache", op):
            seen.clear()
            continue
        if re.match(r"(?:global|buffer|flat)_load_", op):
            parts = x.split(None, 1)[1] if " " in x else ""
            dst, _, rest = parts.partition(",")
            key = (op, rest.strip())
            if key in seen:
                n += 1
            seen[key] = regs(rest)
            d = defs_uses(x)[0]
        else:
            d = defs_uses(x)[0]
        if d:
            for k in [k for k, a in seen.items() if a & d]:
                del seen[k]
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--package", help="a package directory: compile its kernel roots into --out first")
    ap.add_argument("--program", action="append", default=[], help="a program file to compile into --out first (repeatable)")
    ap.add_argument("--out", help="with --package / --program: where the code objects go")
    ap.add_argument("--loom-compile", help="the compiler for --package / --program (default: the machine's VENV); two runs with "
                                           "different compilers plus --baseline compare two Loom versions on the same kernels")
    ap.add_argument("--roots", help="with --package / --program: only kernel roots matching this regex")
    ap.add_argument("--profile", help="a served profile run directory (events.jsonl + *.req.json): each kernel's served ms, and "
                                      "flagged kernels ordered by it")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--baseline", help="a directory of the same kernels from another build: flag what got worse")
    ap.add_argument("--flagged", action="store_true", help="print flagged kernels only")
    ap.add_argument("--strict", action="store_true", help="exit 1 when any kernel is flagged")
    a = ap.parse_args()
    sites, failed = None, []
    progs = sorted(glob.glob(os.path.join(a.package, "program", "*.loom"))) if a.package else []
    progs += a.program
    if progs:
        if not a.out:
            sys.exit("kernel_lint: --package / --program need --out")
        texts, failed = compile_programs(progs, a.out, a.jobs, a.loom_compile, a.roots)
        sites = launch_sites(texts)
        a.paths = [a.out] + a.paths
    stats, prompt = served_stats(a.profile) if a.profile else ({}, None)
    ms = collections.Counter({k: v[0] for k, v in stats.items()})
    files = []
    for p in a.paths:
        files += sorted(glob.glob(os.path.join(p, "*.hsaco")) + glob.glob(os.path.join(p, "*.co"))) if os.path.isdir(p) else [p]
    if not files:
        sys.exit("kernel_lint: no code objects")
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
        rows = [r for rs in ex.map(lint_file, files) for r in rs]
    outl = outliers(rows)
    tiny(rows, stats)
    if a.baseline:
        for r in rows:
            b = os.path.join(a.baseline, r["kernel"] + ".hsaco")
            if os.path.exists(b):
                r["flags"] += ["regress:" + x for x in regressions(r, lint(b))]
    rows.sort(key=lambda r: (-bool(r["flags"]), -ms.get(r["kernel"], 0), -len(r["flags"]), -(sites or {}).get(r["kernel"], 0), r["kernel"]))
    print(f"{'kernel':42s} {'where':>6s} {'loop':>5s} {'comm':>5s} {'mm':>4s} {'other/op':>8s} {'sel':>4s} {'pk':>3s} {'bfe':>4s} {'scr':>4s} {'mov':>4s} "
          f"{'run':>4s} {'vgpr':>4s} {'w/S':>3s} {'next':>4s} {'pf':>5s} {'sites':>5s} {'ldsW':>5s} {'cov':>4s} {'fly':>3s} {'bank':>5s} {'ms':>8s}  flags")
    for r in rows:
        if a.flagged and not r["flags"]:
            continue
        f = lambda v: "-" if v is None else str(v)
        s = "-" if sites is None else str(sites.get(r["kernel"], 0))
        print(f"{r['kernel']:42s} {r['where']:>6s} {r['loop']:5d} {r['common']:5d} {r['wmma'] or r['dot']:4d} {f(r['other_per_op']):>8s} {r['select']:4d} {r['pk']:3d} "
              f"{r['bfe']:4d} {r['scratch']:4d} {r['moves']:4d} {r['run']:4d} {f(r['vgpr']):>4s} {f(r['waves']):>3s} {f(r['to_next']):>4s} {r['pf']:>5s} {s:>5s} "
              f"{f(r['lds_waits_per_op']):>5s} {f(r['cover_min']):>4s} {r['in_flight']:3d} {f(r.get('bank_extra')):>5s} "
              f"{('%.1f' % ms[r['kernel']]) if r['kernel'] in ms else '-':>8s}  "
              f"{' '.join(r['flags'])}{('  [' + ' '.join(r['notes']) + ']') if r.get('notes') else ''}")
    for k, (v, med, n) in sorted(outl.items()):
        print(f"outlier {k}: {v} non-MMA instructions per MMA vs a median of {med} over {n} peers")
    if sites is not None:
        unl = sorted(r["kernel"] for r in rows if not sites.get(r["kernel"]))
        print(f"not launched in the package ({len(unl)}): {' '.join(unl)}")
    for k, why in failed:
        print(f"not compiled standalone: {k}: {why}")
    if ms:
        tot = sum(ms.values()); fl = sum(ms.get(r["kernel"], 0) for r in rows if r["flags"])
        print(f"served profile {a.profile} ({prompt} prompt tokens): {tot / 1e3:.1f} s of dispatches; flagged kernels {fl / 1e3:.2f} s "
              f"({100 * fl / tot:.1f} %)")
    flagged = sum(1 for r in rows if r["flags"])
    print(f"kernel_lint: {len(rows)} kernels, {flagged} flagged" + (f", {len(failed)} roots did not compile standalone" if failed else ""))
    sys.exit(1 if a.strict and flagged else 0)


if __name__ == "__main__":
    main()
