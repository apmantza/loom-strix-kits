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
  [ldsbank] a NOTE until a blind sample confirms it: Loom's gfx1151 bank model puts >= LDSBANK % of the hot loop's modelled LDS
            rounds into bank conflicts. The report's per-packet rows (source_low.memory_rows, one per static LDS instruction, in
            program order) are matched to the binary's ds_ instructions of the same type in order, so only packets inside the hot
            loop count (E249: the f16 GEMMs' 48 extra rounds sit in the once-per-tile epilogue store; rocprof 0.06-0.20 %, E435). A
            type whose row and instruction counts differ is not placed; a conflicted row in one leaves the kernel without a figure.
            Calibration (model vs rocprof SQC_LDS_BANK_CONFLICT / SQC_LDS_IDX_ACTIVE): flash2_f16o 32 of 320 rounds (10.0 %) vs
            8.18 / 8.28 % measured, vsplit 32 of 448 (7.1 %) vs 7.96 % (E424 / E435 / E456: SX11's S-partial exchange, 2-way);
            FLASH2_SXH's 'half' layout 0 of 288 / 0 of 416 vs 0.00 % (E477); f16 GEMM 0 of 304 vs 0.20 %; matvecs 0 vs 0.00 %.
            Fire 11-25 (--package): pkg-e379 g_attention_flash2, g_attention_flash2_f16o (32 of 320), g_attention_vsplit (32 of
            448), nothing else; pkg-e465 (FLASH2_SXH) none; the 14 m256 GEMMs carry 48 whole-kernel extra rounds and 0 of 304 in
            the loop. A kernel without an MMA loop gets no figure (the vision v_attn_72x16: 72 of 144 whole-kernel, unplaced).
            --package compiles the details report (per-packet rows) only for kernels whose summary shows a conflicted packet.
 (with --package / --program: the kernel's own program text, not the binary, not the func.defs it calls): a
            key-valid bit masks only the score while V is read unconditionally: a select of the score against a float fill
            <= MASK_NEG that feeds a softmax exp, with no select of a value outside the score chain against 0.0 and no index.min
            of a compared key (E441 / E444: g_attention_split multiplied the V of keys past the block's last live one by p = +0, so
            a NaN in unwritten KV reached the output; zeroing p is no guard, 0 x NaN = NaN). Fire 11-25: pkg-e379 / pkg-e465
            g_attention_split, g16_attention_split_listed (and the _rdna4 twin), nothing else; KV8_SPLIT_VZERO (pkg-q179) clears
            all three. Quiet: flash2 / flash2_partial / vsplit (K and V zeroed under the bit), g_dfl_ctx_split (key clamped),
            g_dfl_topk (a logit mask, no softmax). The stale_kv kernel tests are the guard; this points at a kernel to read.
  [barrier] a NOTE (with --package / --program / --source: the kernel's Loom text, checked against the binary's s_barrier count;
            user 11-25, after cuda-oxide's no_data_race / uniform_barriers; E504: one extra barrier per flash2 tile cost +1.4 %):
            a kernel.barrier<workgroup> that orders no workgroup-memory hazard (RAW / WAR / WAW between overlapping accesses of one
            workgroup alloca, one writing) that no other barrier on the same path also orders: removable on its own (two such
            barriers removed together are not covered). The model fails closed: views and accesses are allowlists (BAR_VIEW,
            BAR_ACCESS), any other op touching a region reads and writes all of it, and two accesses overlap unless their views sit
            at disjoint constant byte extents (indices assumed in bounds, as Loom requires); review 11-25 (static-1125/barrier-lint).
            --barriers prints every barrier with the pairs it orders. Fire 11-25 on pkg-e480 (259 kernels, 85 MB, 3.4 s): 8
            barriers, all flash2 (main-loop end-of-tile barrier and the tail's last barrier, in flash2 / _f16o / _partial /
            _partial_f16o); author-judged sample 2 of 2 real.
  [divbar]  a NOTE: a barrier or workgroup collective under an scf.if / scf.for whose condition or bound may differ between a
            workgroup's invocations: some waves may never arrive. Uniform is an allowlist: workgroup id / size / count, subgroup
            count / size, a workgroup reduce, constants and kernel arguments, and arithmetic, views and plain loads of those;
            everything else (ids, lanes, shuffles, scans, fragments, mma, atomics, volatile loads, unlisted ops, a loop's values
            under a non-uniform trip count) may differ. Fire 11-25 on pkg-e480: none.
  [rounds]  INFORMATION, never a note (with --package / --program / --source: launch sites and launch configs; C18, user 11-25; review
            r2 option iii): for a kernel every launch site of which fits one round of resident workgroups at size 1 (tier_state()'s
            workgroups per WGP x the device file's WGPs), with >= 2 workgroups, one line gives the grid at size 1, the resident count
            and the one-round bound in the size variable (tokens <= N, spans <= M; a grid of one size key, searched to ROUNDS_MAX).
            E539 asked whether all of g_gdn_recurrence_pf's workgroups are resident at once: 'one round while spans <= 1'. No served
            size is claimed: Loom 0.2.0 admits tokens and spans up to the prefill capacity (serve/service/text/schedule.c
            loom_serve_text_default_shapes, serve/server/model_registration.c; the package's prepare.loom stage shapes); read the
            bound against a served profile. An unmodelled op, a name defined twice, an arity mismatch, two size keys, an empty or a
            shrinking grid: one summary line names the kernel ('not resolved'). Lines 11-25 on pkg-e480: 18 (g_gdn_recurrence_pf: spans <= 1;
            g_gdn_conv_commit: spans <= 4; matvecs, norms, dflash helpers: tokens <= 3-80), 38 not resolved; llama.cpp: no Loom text.
  [guardwait] a NOTE (C18): a vmcnt wait inside a conditional block of a loop (loop_census.guarded's regions behind a forward
            s_cbranch, minus an exit test that leaves or enters a loop) that drains loads issued before the branch, untouched by the
            block and younger than every load the block needs (issued in the block, or read / written by it): vmcnt(#those) would
            have served the block. Recurrence E520: STAGE_LIVE x FLATADDR, the carried K scale's vmcnt(0) inside the K guard retired
            V(kt)'s prefetch (cover 166-203 -> 23-25); C10 catches it only against a base. Fire 11-25 (r2): pkg-cmb-sf all four flash2 kernels;
            pkg-cmb2 (E521's fix) the two partial kernels (its residue); pkg-cmb-sn / -nf and live pkg-e480 none; llama.cpp 1 of 8,374
            (solve_tri_f32_fast: E520's shape, waits for an older load and drains a newer one; r1's 129 were 0 / 6 of that shape;
            soft_max's barrier / wait / invalidate is not judged).
  [vopd]    a NOTE (C18 from C5, E551; review r2: the oracle is Loom 0.2.0's VOPD planner, not llvm-mc): VOPD pairs a greedy pass
            could still form from single VALU instructions as allocated, >= VOPD_MIN in the hot loop, by the planner's rules for
            amdgpu.rdna3_5.core (vopd_eligible / vopd_registers_ok: VGPR sources only except an inline move or fmaak / fmamk's one
            literal; destinations of opposite parity; per source-cache port both registers of opposite parity, one register in both
            halves refused; commutable swaps where Loom's descriptors allow them), independent, halves <= 8 apart. The line adds the
            back-edge copy block (moves, pairs inside it) and, informational only, WMMAs whose accumulator was written <= 1 WMMA
            earlier (D -> C latency undocumented, C5 R6a). Live flash2_f16o: 35 of 129 eligible (C5's 56 / 186 counted literal and
            SGPR sources Loom's planner never pairs), back-edge 65 moves -> 31 pairs, dep1 16; m256 GEMM 0 of 1, dep1 0 (pinned by
            fixtures). A wasted s_delay_alu count is not a rule (C5: zero cycles, 2.1 % ceiling at q202). Fire 11-25 on pkg-e480: 7
            (flash2 x4, vsplit, iq4_xs_store_m256, q6_k_swiglu_m256); llama.cpp 209 of 8,374 (r2: 269).
  (an 'ldswait' note, LDS loads waited right after issue, was deleted in C18's review: the ldsser lever on the same m256 GEMMs was
  refuted four times, E293 / E309 / E335 / E340, and E341 puts their WMMA phase at the hardware rate. Re-add only if E534's
  SQ_WAIT_INST_LDS / SQ_WAVE_CYCLES measurement on the m256 GEMM contradicts E341.)
  (potential races, two overlapping accesses with no barrier between, are computed and listed by --barriers only, never a note:
  lane- or wave-indexed staging stores are not proved disjoint (no per-invocation index model), and an author-judged sample of 10
  of the 60 on pkg-e480 (132 without r1's index lattice) found 0 real; all were `row = tid / 2, col = (tid % 2) * 32 + c` staging or per-wave rows.)
  Not analysed (printed, never clean): unknown region ops (scf.while ...), op lines it cannot parse, a return before the end,
  kernel.exit, kernel.async.* / kernel.tensor.* (data lands after issue), an alloca in a space other than workgroup / private /
  global or a buffer.alloca.init<workgroup>, workgroup scan / vote, a workgroup view through a loop or yield, a callee that
  synchronises, allocates workgroup memory or copies asynchronously, barrier forms other than <workgroup> scope(workgroup), and
  a text whose implied s_barrier count (two per workgroup reduce, times constant unrolls) is not the binary's. Two exports of one
  name in the texts: the last wins (kernel_texts); the s_barrier cross-check refuses it only when the counts differ. Another
  engine's code objects carry no Loom text: the check does not apply to them.
  regress:  (with --baseline) worse than the same kernel in another build: a new spill, the prefetch now drained, fewer
            waves per SIMD, or the hot loop / its longest move run >= 10 % longer (E226: Loom 0.3.0's vmcnt(0) + 32-move shift;
            E230: a rewrite's spill; E236: a variant's residency drop).
Reported, not flagged: non-MMA instructions per MMA, moves, VGPRs, w/S (waves per SIMD) and 'next': the tier reached once the
VGPRs (v) and LDS bytes (l) shown are freed together, 'best' at the wave cap, 'blocked' by the slot limit; both counted in whole
workgroups from the summary report beside the code object, KERNEL.json (tier_state(); E533: Loom 0.2.0's own tier rounds VGPRs per
wave and named flash2's next tier 7, unreachable); 'bound' marks a workgroup whose waves are not a multiple of the SIMDs (Loom's
ceil convention: placement unknown), pf ('drain' when a vmcnt(0) sits between the loop's last load and
its first WMMA), and with --package the kernel's launch sites in the package. Calibrated on 225 served kernels (E237 / E241):
v_mul_lo_u32 (address math) and a drain alone are not defects.
Decoding goes through loop_census.disasm (ROCm 10.2 llvm-objdump; it refuses undecoded words).
Also reported: LDS waits (lgkmcnt) per MMA (E61), the shortest load-to-wait distance and the loads still in flight at the
back-edge (E109), the extra LDS bank-service rounds of Loom's gfx1151 bank model (whole kernel, not the hot loop: read the report's
bank_service_groups before acting; E249: the f16 GEMMs' 48 extra rounds were the epilogue's once-per-tile store), and with --profile
each kernel's served ms.
A code object holding many kernels (another engine's build: *.co / *.hsaco) is linted kernel by kernel (lint_object).
--loom-candidates (C19): Loom upstream CANDIDATES, never flags, notes or an exit code; each is a prompt for a compiler packet that a
human still makes airtight (minimal .loom repro, upstream main, existing issues, the user's approval to post). Per Loom kernel object:
'tier' (the report's next tier vs whole-workgroup arithmetic, E533) and 'vopd-edge' (back-edge copy groups VOPD could pair, E592);
with --baseline, 'sdk' (two compiles of identical programs by two compilers, from each directory's loom-provenance: shapes that
appear or disappear with only the compiler changed; E177 / E227 by hand). Fire 11-25 on pkg-e480: 0.2.0 24 tier / 17 vopd-edge,
0.3.0 20 / 17, 0.2.0 -> 0.3.0 4 sdk lines (f16 m256 GEMMs +addr64; split attention better waves 6 -> 8).
usage: kernel_lint.py DIR_OR_HSACO ... [--baseline DIR] [--profile RUN] [--source PKG_OR_LOOM] [--barriers] [--loom-candidates] [--flagged] [--strict]
       kernel_lint.py --package PKG | --program FILE.loom ... --out DIR [--loom-compile PATH] [--roots REGEX] [--jobs N] [...]
         (compiles every gfx11 kernel root into DIR first; two compilers: run twice with --loom-compile and pass the first
          run's DIR as --baseline of the second; --strict: exit 1 when anything is flagged)"""
import argparse
import collections
import concurrent.futures
import glob
import json
import math
import os
import re
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop_census import VMCNT_ENTRY, cover, disasm, guarded, loops, moves  # noqa: E402

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


OCC_ROWS = {("amdgpu.vgpr", "pooled_resource", "subgroup"), ("amdgpu.lds", "pooled_resource", "workgroup"),
            ("amdgpu.workgroup_slots", "fixed_limit", None), ("amdgpu.sgpr", "unconstrained_resource", "subgroup")}
# (name, kind, scope): the residency rows occupancy() models (every row of pkg-e480's 259 reports); any other row: unverified


def occ_inputs(doc):
    """occupancy()'s inputs from a Loom compile report: workgroup lanes, wave size, wave cap, the report's current tier, its
    residency_constraints rows and target"""
    tr = doc.get("target_resources") or {}
    return {"wg": ((doc.get("workload") or {}).get("workgroup_size") or {}).get("flat"), "wave": tr.get("subgroup_size"),
            "cap": tr.get("max_subgroups_per_simd"), "tier": (tr.get("residency") or {}).get("current_tier"),
            "rows": (doc.get("residency_constraints") or {}).get("rows", []), "target": doc.get("target_key")}


def device_simds(target):
    """SIMDs per occupancy domain from common/devices/<target>.json, or None (occupancy() then says unverified)"""
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "devices", f"{target}.json")
    return json.load(open(f)).get("simds_per_occupancy_domain") if target and os.path.exists(f) else None


def occupancy(occ, simds):
    """the occupancy LEVER (lint_counters (d2), loom_suggest's cliff verdict): None at the best tier (the wave cap); {"why"} when
    unverified or when a fixed limit blocks the next tier ('next tier blocked by ...', with "tier" and "bound" kept); else
    tier_state()'s dict {"tier", "bound", "q", "report_tier", "next", "needs": [(name, units, unit, max, cut)]}"""
    st = tier_state(occ, simds)
    if "why" in st:
        return st
    if st["blocked"]:
        return {"why": st["blocked"], "tier": st["tier"], "bound": st["bound"]}
    return None if st["next"] is None else st


def tier_state(occ, simds):
    """The residency tier in whole workgroups, the next tier a workgroup can reach and what each resource must drop to for it:
    {"tier", "bound", "q", "report_tier", "next", "needs": [(resource, units, unit, max units, cut)], "blocked"}; {"why": reason}
    when it cannot be verified. The one model behind kernel_lint's w/S and next columns, lint_counters (a) and (d2) (through
    occupancy(), the lever view of the same result), and livereg's budget.
    A workgroup of w waves lives on one occupancy domain of SIMDS SIMDs (Loom runs WGP mode: 4); n resident workgroups put
    ceil(n w / SIMDS) waves on the busiest SIMD, Loom's own rule (v0.2.0 occupancy.c:377-383, wave_limit_for_workgroups). The tier is
    that integer; it is exact when w % SIMDS == 0 and otherwise a bound (Loom's balanced placement; how the SPI places a workgroup's
    waves over the SIMDs is documented nowhere we searched: 'bound', never a fraction, HRX-like review r1 F3). Loom 0.2.0's own tier
    takes VGPRs per wave, not per whole workgroup (HRX-like E533): flash2 (384 lanes, 12 waves) reported 7, reachable is 9, so the
    report's next tier and reductions are not used. Each resource's independent tier is recomputed (subgroup scope: pool / rounded
    units waves; workgroup scope: ceil((pool / rounded units) w / SIMDS)) and must equal the report's, and the whole-workgroup tier
    must equal its current tier or the unrounded minimum, or the tier is 'unverified'. next is the tier actually reached once every
    cut is applied (a VGPR granule can skip past n + 1 workgroups: g_dfl_ctx_split's -3 VGPRs lands on 12, not 11; review r1 F2);
    None at the wave cap, or with "blocked" naming the fixed limit that stops it."""
    if not occ:
        return {"why": "no compile report"}
    wg, wave, cap, cur = occ.get("wg"), occ.get("wave"), occ.get("cap"), occ.get("tier")
    if not simds or not all(isinstance(v, int) and v > 0 for v in (wg, wave, cap, cur)):
        return {"why": "tier unverified: workgroup size, wave size, wave cap or current tier missing from the report, or the device "
                       "file has no simds_per_occupancy_domain"}
    w = -(-wg // wave)
    busiest = lambda n: -(-n * w // simds)
    fit = lambda t: max([0] + [n for n in range(1, cap * simds + 1) if busiest(n) <= t])   # workgroups whose busiest SIMD fits t waves
    pooled, fixed = [], []
    for r in occ["rows"]:
        name, kind = r.get("name", "?").split(".")[-1], r.get("kind")
        if (r.get("name"), kind, r.get("allocation_scope")) not in OCC_ROWS:     # fail closed: a row nobody modelled is no figure
            return {"why": f"tier unverified: residency row {r.get('name')} ({kind}, {r.get('allocation_scope')}) not modelled"}
        if kind == "unconstrained_resource":
            continue
        if kind == "fixed_limit":
            fixed.append((name, r.get("independent_tier")))
            continue
        try:
            pool, gran, units, scope = int(r["pool_units"]), int(r["allocation_granularity"]), int(r["units"]), r["allocation_scope"]
        except (KeyError, TypeError, ValueError):
            return {"why": f"tier unverified: {name} ({kind}) lacks pool, granularity, units or scope"}
        if kind != "pooled_resource" or scope not in ("subgroup", "workgroup") or gran <= 0 or units < 0:
            return {"why": f"tier unverified: {name}: kind {kind!r}, scope {scope!r}, granularity {gran}, units {units} not modelled"}
        pooled.append((name, units, r.get("unit") or "", pool, gran, scope, r.get("independent_tier")))
    if any(not isinstance(t, int) for _, t in fixed):
        return {"why": "tier unverified: a fixed limit without a tier"}
    def groups(name, units, pool, gran, scope):           # workgroups this resource admits, and its independent tier as Loom counts it
        ru = -(-units // gran) * gran
        if ru == 0:                                        # a resource the kernel does not use (0 B of LDS) never limits
            return fit(cap), cap
        if scope == "subgroup":
            t = min(cap, pool // ru)
            return fit(t), t
        return min(pool // ru, fit(cap)), min(cap, busiest(pool // ru))
    n_of = {}
    for name, units, unit, pool, gran, scope, rep in pooled:
        n_of[name], got = groups(name, units, pool, gran, scope)
        if got != rep:
            return {"why": f"tier unverified: {name} computes to tier {got}, the report says {rep}"}
    for name, t in fixed:
        n_of[name] = fit(t)
    n = min(list(n_of.values()) + [fit(cap)])
    tier = busiest(n)
    raw = min([rep for *_, rep in pooled] + [t for _, t in fixed] + [cap])
    if cur not in (tier, raw):   # raw: the report's own unrounded minimum (E533: flash2_partial, VGPR tier 5 at 3 waves per SIMD is 3)
        return {"why": f"tier unverified: computes to {tier} in whole workgroups ({raw} unrounded), the report says {cur}"}
    out = {"tier": tier, "bound": w % simds != 0, "q": w // simds if w % simds == 0 else round(w / simds, 2), "report_tier": cur,
           "next": None, "needs": [], "blocked": None, "wgs": n, "target": occ.get("target")}   # wgs: resident workgroups per domain
    m = n + 1
    while busiest(m) <= tier:     # 1-3-wave workgroups: n + 1 can leave the busiest SIMD's ceil unchanged (verify r2 V-F1); Loom
        m += 1                    # inverts from tier + 1 the same way (occupancy.c local_memory_budget)
    if busiest(m) > cap:
        return out                                         # the wave cap: the best tier
    block = [f"{x} (tier {t})" for x, t in fixed if n_of[x] < m]
    if block:
        out["blocked"] = "next tier blocked by a fixed limit: " + ", ".join(block)
        return out
    cut = {}
    for name, units, unit, pool, gran, scope, rep in pooled:
        if n_of[name] < m:
            mx = (pool // busiest(m) if scope == "subgroup" else pool // m) // gran * gran
            out["needs"].append((name, units, unit, mx, units - mx))
            cut[name] = mx
    reached = min([groups(nm, cut.get(nm, u), pl, gr, sc)[0] for nm, u, _, pl, gr, sc, _ in pooled] +
                  [n_of[x] for x, _ in fixed] + [fit(cap)])
    out["next"] = busiest(reached)
    return out


def residency(js):
    """(VGPRs, tier_state() of the summary report beside the code object) or (None, None); the row keys follow from it (lint())"""
    if not os.path.exists(js):
        return None, None
    d = json.load(open(js))
    occ = occ_inputs(d)
    return (((d.get("target_resources") or {}).get("vector") or {}).get("final") or {}).get("register_count"), \
        tier_state(occ, device_simds(occ["target"]))


def suggest_occ(js):
    """occupancy() of the compile report JS, the SIMD count from its target's device file: loom_suggest.survey's occ_of"""
    occ = occ_inputs(json.load(open(js)))
    return occupancy(occ, device_simds(occ["target"]))


def tier_text(o, key="tier"):
    """a tier for print: the integer, plus ' bound' where the workgroup's waves are not a multiple of the SIMDs (placement unknown)"""
    return "-" if not o or o.get(key) is None else f"{o[key]}{' bound' if o.get('bound') else ''}"


def next_text(r):
    """the 'next' column: the tier reached after the cuts and what to free for it together (v VGPRs, l LDS bytes); 'best' at the wave
    cap, 'blocked' by a fixed limit, '-' unverified"""
    o = r.get("occ")
    if not o or "why" in o:
        return "-"
    if o["next"] is None:
        return "blocked" if o["blocked"] else "best"
    return f"{tier_text(o, 'next')}:" + "+".join(f"{n[0]}{c}" for n, _, _, _, c in o["needs"])


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
        r = lint_listing(n, sl, sa, (vg, None))
        r["family"] = re.split(r"[<(]", dn, maxsplit=1)[0].split("::")[-1].strip() or n
        r["object"] = os.path.basename(path)
        r["bank_conf"] = r["bank_extra"] = None
        rows.append(r)
    return rows


def lint_file(path, rga=None):
    """the rows of one code object: one kernel (a Loom compile) or several (another engine's build); RGA: live registers (--livereg)"""
    syms = kernel_symbols(path)
    if len(syms) == 1 and syms[0][0] == os.path.basename(path).rsplit(".", 1)[0]:
        return [lint(path, rga)]
    rows = lint_object(path)
    for r in rows:
        if rga is not None:
            r["live"] = {"error": "rga: a multi-kernel object is not analysed"}
    return rows


LIVE_ROW = re.compile(r"^\s*\d+ \|\s*(\d+) \|[^|]*\| (?:label_\d+:\s*)?(.*)$")   # Line | Rn | register map | instruction


def rga_tool(explicit=None):
    """RGA (Radeon GPU Analyzer, `rga -s bin`): --rga, else RGA from the environment or the machine env file; None when absent"""
    p = explicit or os.environ.get("RGA")
    if not p:
        f = os.environ.get("LSK_ENV") or os.path.expanduser("~/.config/loom-strix-kits/env")
        m = re.search(r"(?m)^RGA=(\S+)", open(f).read()) if os.path.exists(f) else None
        p = m.group(1) if m else None
    return p if p and os.access(p, os.X_OK) else None


def live_rows(text):
    """(live VGPRs, instruction) per instruction of an RGA --livereg file, in program order"""
    return [(int(m.group(1)), m.group(2).strip()) for m in map(LIVE_ROW.match, text.split("\n")) if m]


def live(rows, ins, addr, budget):
    """live-register figures of one kernel from RGA's rows matched to our listing (E527 / E531 / E533): {peak, loop_peak, at (offset
    in the hot loop, mnemonic), floor (the loop's minimum: the carried values no schedule frees), over (loop instructions with more
    live VGPRs than BUDGET, the next reachable tier's), loop}; {error} when the rows are not our instructions one for one (fail closed)"""
    mn = lambda x: re.sub(r"(?:_e32|_e64|_dpp|_sdwa)+$", "", x.split()[0]) if x.split() else ""   # RGA drops encoding suffixes
    if len(rows) != len(ins):
        return {"error": f"rga: {len(rows)} rows for {len(ins)} instructions"}
    j = next((j for j, (x, (_, y)) in enumerate(zip(ins, rows)) if mn(x) != mn(y)), None)
    if j is not None:
        what = "cannot decode" if rows[j][1].startswith("<unknown>") else f"has {rows[j][1].split()[0]!r} for"
        return {"error": f"rga {what} instruction {j}, {ins[j].split()[0]!r}"}
    out = {"peak": max((n for n, _ in rows), default=0)}
    h = hot(ins, addr)
    if h:
        seg = [n for n, _ in rows[h[0]:h[1] + 1]]
        k = max(range(len(seg)), key=seg.__getitem__)
        out.update(loop=(h[0], h[1]), loop_peak=seg[k], at=(k, ins[h[0] + k].split()[0]), floor=min(seg),
                   over=None if budget is None else sum(1 for n in seg if n > budget))
    return out


def rga_live(path, rga):
    """RGA's live VGPR rows for one code object (a temporary dir per call), or {error}"""
    import tempfile
    if not rga:
        return {"error": "rga: not found (--rga, or RGA in the environment / machine env file)"}
    with tempfile.TemporaryDirectory(prefix="kl_rga.") as d:
        p = subprocess.run([rga, "-s", "bin", "--isa", f"{d}/k.isa", "--livereg", f"{d}/k.live", "--co", path],
                           capture_output=True, text=True)
        fs = glob.glob(f"{d}/*.live")
        if p.returncode or len(fs) != 1:
            why = [x for x in (p.stdout + p.stderr).split("\n") if "rror" in x or "failed" in x]
            return {"error": "rga: " + ((why[-1].strip() if why else f"exit {p.returncode}, {len(fs)} .live files")[:160])}
        return live_rows(open(fs[0]).read())


def live_text(r):
    """the 'live' column: peak/floor/over (the kernel's peak live VGPRs, the hot loop's minimum, its instructions over the next tier's
    VGPR budget); '-' without --livereg; 'unknown' when RGA failed (the line says why)"""
    x = r.get("live")
    if x is None:
        return "-"
    if "error" in x:
        return "unknown"
    return f"{x['peak']}/{x.get('floor', '-')}/{'-' if x.get('over') is None else x['over']}"


def live_budget(r):
    """the VGPRs a wave may hold at the next reachable tier (tier_state()'s VGPR cut, or today's count when only another resource must
    move); None at the best tier, blocked or unverified (no over count then)"""
    o = r.get("occ")
    if not o or "why" in o or o["next"] is None:
        return None
    cut = [mx for n, _, _, mx, _ in o["needs"] if n == "vgpr"]
    return cut[0] if cut else r.get("vgpr")


def lint(path, rga=None):
    ins, addr = disasm(path)
    r = lint_listing(os.path.basename(path)[:-6], ins, addr, residency(path[:-6] + ".json"))
    if rga is not None:
        rows = rga_live(path, rga)
        budget = live_budget(r)
        r["live"] = rows if isinstance(rows, dict) else live(rows, ins, addr, budget)
        r["live_budget"] = budget
    r["bank_conf"], r["bank_extra"] = bank(path[:-6] + ".json")
    ldsbank(r, path[:-6] + ".json", ins)
    return r


LDSBANK = 2.0   # % of the hot loop's modelled LDS rounds that are conflict rounds (11-25: flash2 10.0 / vsplit 7.1 fire, SXH and GEMMs 0)


def loop_banks(js, ins, t, e):
    """(extra, required) LDS bank-service rounds of the packets in ins[t..e] by Loom's gfx1151 bank model, or None: the report's
    per-packet rows (source_low.memory_rows, one per static LDS instruction, in program order) matched to the binary's ds_
    instructions of the same packet type in order. A type whose row and instruction counts differ is not placed; None when such a
    type holds a conflicted row (where it sits is unknown) or the report has no rows. Rows the model could not prove count nowhere."""
    if not os.path.exists(js):
        return None
    rows = [x for x in (json.load(open(js)).get("source_low") or {}).get("memory_rows") or [] if x.get("memory_space") == "workgroup"]
    if not rows:
        return None
    by, pos = collections.defaultdict(list), collections.defaultdict(list)
    for x in rows:
        by[x["packet"].split(".", 1)[-1].replace("ds_write", "ds_store").replace("ds_read", "ds_load")].append(x)
    for j, x in enumerate(ins):
        pos[x.split(" ", 1)[0]].append(j)
    ext = req = 0
    for op, rs in by.items():
        if len(rs) != len(pos[op]):
            if any((x.get("bank_service") or {}).get("extra_rounds") for x in rs):
                return None
            continue
        for x, j in zip(rs, pos[op]):
            b = x.get("bank_service") or {}
            if t <= j <= e and b.get("proof") == "exact":
                ext += b.get("extra_rounds") or 0
                req += b.get("required_rounds") or 0
    return ext, req


def ldsbank(r, js, ins):
    """note `ldsbank` when the hot loop's modelled conflict rounds are >= LDSBANK % of its modelled LDS rounds (E424 / E435 / E456)"""
    r["bank_loop"] = loop_banks(js, ins, *r["hot_range"]) if r.get("hot_range") else None
    if r["bank_loop"] and r["bank_loop"][1] and 100 * r["bank_loop"][0] >= LDSBANK * r["bank_loop"][1]:
        r.setdefault("notes", []).append("ldsbank")


# ---- 11-25 lesson notes (C18): guardwait, vopd (binary); rounds (program text + residency: an information line, no size claim). Notes,
# never flags (E251). ldswait: deleted in review (docstring).
VOPD_MIN = 16    # VOPD pairs left unformed as allocated (whole loop) at or above which the note fires (C5 / E551 / E580 / E585: flash2_f16o 35 by Loom's rule)


FENCE = re.compile(r"(?:global|buffer|flat)_store|s_barrier|buffer_gl\d_inv")   # memory ordering a wait may serve (no spill store)


def guard_waits(ins, addr):
    """vmcnt waits inside a conditional block of a loop (after a forward s_cbranch_*, before its target, both in the loop; a block
    that leaves or enters a loop is an exit test, not a guard) that drain loads the block did not need: of the loads the wait
    retires (vmcnt retires in order), 'needed' = issued inside the block, or whose destination the block reads or writes; victims =
    loads issued before the branch, not needed, and younger than every needed one, so a wait of vmcnt(#victims) would have served the
    block. HRX-like E520: STAGE_LIVE x FLATADDR's guard waited vmcnt(0) for the older carried K scale and drained V(kt)'s newer
    prefetch (cover 166-203 -> 23-25). A wait whose newest retired load the block needs (a load issued in the block: in-order vmcnt
    cannot spare the older ones) is not one (review r1 F4).
    A wait that retires no load the block needs, followed in the block by a global / buffer / flat store, an s_barrier or a
    buffer_gl*_inv, may order that memory operation after the pending loads (release: wait, then store or barrier; acquire: wait,
    then invalidate), a reason no register shows: not judged (live g_gdn_recurrence_pf / _rec: vmcnt(0) before a guarded state
    store; llama.cpp soft_max: s_barrier, vmcnt(0), buffer_gl0_inv). A store or fence before the wait, or a spill store anywhere,
    orders nothing the wait could serve: every retired pre-branch load is then a victim (pkg-cmb2's partial kernels, E521's residue,
    fire with or without a spill in the block; review r2 F-G1 / F-G2). The queue takes every vmcnt entry (loop_census.VMCNT_ENTRY:
    spill reloads and returning atomics too). Each loop's body runs twice, so loads carried over the back-edge are pending at its
    top. Returns [(wait index, innermost guard's branch index, victims)], one per wait"""
    L = loops(ins, addr)
    cross = lambda b, j: any((a <= b <= z < j) or (b < a <= j - 1 < z) for a, z in L)   # leaves or enters a loop
    hits, rw = {}, {}
    for t, e in L:
        regions = sorted(((b, hi + 1) for b, _, hi in guarded(ins, addr, t, e)
                          if ins[b].startswith("s_cbranch_") and not cross(b, hi + 1)), reverse=True)   # innermost (latest) first
        n = e - t + 1
        q = []
        for p in range(2 * n):
            i = t + p % n
            x = ins[i]
            if VMCNT_ENTRY.match(x):                          # loads, spill reloads, returning atomics
                q.append((p, i))
            w = re.search(r"vmcnt\((\d+)\)", x) if x.startswith("s_waitcnt") else None
            if not w:
                continue
            gone = []
            while len(q) > int(w.group(1)):
                gone.append(q.pop(0))
            for b, j in regions:
                if not b < i < j or i in hits:
                    continue
                if (b, j) not in rw:
                    rw[(b, j)] = set().union(*[a | c for a, c in map(defs_uses, ins[b + 1:j])])
                need = [g for g in gone if defs_uses(ins[g[1]])[0] & rw[(b, j)]]   # a load in the block writes its own registers
                if not need and any(FENCE.match(y) for y in ins[i + 1:j]):
                    continue                                    # may order what follows it (a store, a fence): no register says why
                victims = [g for g in gone if g not in need and g[0] > max([-1] + [x[0] for x in need])]
                if victims:
                    hits[i] = (i, b, len(victims))
    return sorted(hits.values())


def guardwait(r, ins, addr):
    """note `guardwait` on any guard_waits() hit"""
    r["guard_waits"] = guard_waits(ins, addr)
    if r["guard_waits"]:
        r.setdefault("notes", []).append("guardwait")


# VOPD components as Loom 0.2.0's planner reads them on amdgpu.rdna3_5.core (hrx-system v0.2.0: amdgpu_vopd_component_tables.py
# _component_definitions; planning/vopd_plan.c read_*_component; descriptors/alu.py COMMUTABLE constraints): opcode -> (lanes,
# form, commutable). Loom's dot2 component is the VOP3P v_dot2_f32_f16, not v_dot2acc: not modelled; nor is the planner's
# transcendental-result rejection window (a pair right after v_exp may be refused)
VOPD_OPS = {"fmac_f32": ("XY", "tied", True), "fmaak_f32": ("XY", "fmaak", True), "fmamk_f32": ("XY", "fmamk", False),
            "mul_f32": ("XY", "binary", True), "add_f32": ("XY", "binary", True), "sub_f32": ("XY", "binary", False),
            "subrev_f32": ("XY", "binary", False), "max_f32": ("XY", "binary", True), "min_f32": ("XY", "binary", True),
            "cndmask_b32": ("XY", "binary", False), "mov_b32": ("XY", "mov", False), "add_nc_u32": ("Y", "binary", True),
            "lshlrev_b32": ("Y", "binary", False), "and_b32": ("Y", "binary", True)}
INLINE = re.compile(r"(?:[0-9]|[1-5][0-9]|6[0-4]|-(?:[1-9]|1[0-6]))$")   # integer inline constants 0..64, -1..-16 only: Loom 0.2.0's
# inline move takes loom_amdgpu_encoding_inline_u32_source (vopd_plan.c:840-860; loom-check: 1.0 / 0.5 / 65 stay single, C18 verify r3);
# the disassembler prints every other integer in hex


def vnum(tok):
    m = re.match(r"v(\d+)$", tok or "")
    return int(m.group(1)) if m else None


def vopd_eligible(x):
    """the VOPD component Loom 0.2.0's planner would read from a single VALU instruction, or None: {"lanes", "vdst", "src0",
    "vsrc1", "form", "comm", "lit"}. VOP2 / VOP1 form only (_e32 or unsuffixed: v_dual_* halves and VOP3 _e64 do not match); a
    negated, absolute or clamped operand is not a bare register and fails the source checks. Binary and cndmask forms need src0 and
    vsrc1 in VGPRs, no literal, SGPR or inline constant (vopd_plan.c read_binary_vgpr_component: immediate_count 0, single physical
    VGPRs); fmaak / fmamk take VGPR sources and one literal (fmamk's operands are dst, src0, K, vsrc1); fmac's destination is its
    accumulator; v_mov_b32 is a register move from a VGPR or an inline move of an inline constant (an SGPR or a non-inline literal
    source is not eligible)"""
    mn, _, rest = x.partition(" ")
    m = re.match(r"v_(\w+?)(?:_(e32|e64|dpp|sdwa))?$", mn)
    if not m or m.group(2) not in (None, "e32") or m.group(1) not in VOPD_OPS:
        return None
    lanes, form, comm = VOPD_OPS[m.group(1)]
    ops = [o.strip() for o in re.split(r",(?![^\[]*\])", rest)] if rest.strip() else []
    c = {"lanes": lanes, "form": form, "comm": comm, "lit": None, "vdst": vnum(ops[0]), "src0": None, "vsrc1": None}
    if form == "mov":
        c["src0"] = vnum(ops[1])
        return c if c["src0"] is not None or INLINE.match(ops[1]) else None
    s0, s1, k = (ops[1:4] + [None])[:3] if form != "fmamk" else (ops[1], ops[3], ops[2])
    c["src0"], c["vsrc1"] = vnum(s0), vnum(s1)
    if c["src0"] is None or c["vsrc1"] is None:
        return None
    if form in ("fmaak", "fmamk"):
        c["lit"] = k
    return c


def vopd_caches(c):
    """the registers a component reads through VOPD's three source-cache ports (SRC0, VSRC1, SRC2), Loom's
    _component_cache_registers: fmamk's vsrc1 and fmac's tied destination sit in SRC2, a register move reads SRC0 only"""
    if c["form"] == "fmamk":
        return (c["src0"], None, c["vsrc1"])
    if c["form"] == "tied":
        return (c["src0"], c["vsrc1"], c["vdst"])
    return (c["src0"], c["vsrc1"], None)


def vopd_registers_ok(x, y):
    """Loom 0.2.0's register constraints for one pair on an interlock target (vopd_component.c register_constraint_flags with
    SOURCE_PARITY, set on amdgpu.rdna3_5.core: amdgpu_vopd_component_tables.py:601-602, target_info.py rdna3_5.core): destinations
    of opposite parity, and on each cache port both registers of opposite parity (mask 1: one register in both halves is refused);
    a commutable component may swap SRC0 / VSRC1 (try_register_orientations: X, then Y, then both)"""
    def ok(a, b):
        return (a["vdst"] ^ b["vdst"]) & 1 and all(p is None or q is None or (p ^ q) & 1 for p, q in zip(vopd_caches(a), vopd_caches(b)))
    sw = lambda c: dict(c, src0=c["vsrc1"], vsrc1=c["src0"])
    return any(ok(a, b) for a, b in [(x, y)] + ([(sw(x), y)] if x["comm"] else []) + ([(x, sw(y))] if y["comm"] else [])
               + ([(sw(x), sw(y))] if x["comm"] and y["comm"] else []))


def vopd_pairs(body, lo=0, hi=None, win=8):
    """VOPD pairs a greedy pass could form from eligible singles in body[lo..hi] as allocated (an estimate, not a bound either way:
    the planner's transcendental window is not modelled, and a matching at any distance finds 1-2 more; on flash2_f16o 32 of 35 are
    adjacent, most in the back-edge copies Loom left single), by Loom 0.2.0's planner rules (the
    oracle; review r2: ROCm 10.2 llvm-mc checks banks, not the gfx11.5 interlock): one half X-capable, the other Y-capable, at most
    WIN instructions apart, independent (no def / use overlap, nothing between writing their sources or reading / writing their
    destinations, no wait, barrier or branch between), vopd_registers_ok, and two literals only when equal (vopd_component.c
    resolve_pair_literal). Defs / uses from defs_uses. Returns (pairs, eligible)"""
    hi = len(body) - 1 if hi is None else hi
    el = {i: v for i in range(lo, hi + 1) for v in [vopd_eligible(body[i])] if v}
    du = {}
    f = lambda k: du.setdefault(k, defs_uses(body[k]))
    def can(i, j):
        x, y = el[i], el[j]
        if not (("X" in x["lanes"] and "Y" in y["lanes"]) or ("Y" in x["lanes"] and "X" in y["lanes"])):
            return False
        (di, ui), (dj, uj) = f(i), f(j)
        if di & uj or dj & ui or di & dj:
            return False
        for k in range(i + 1, j):
            if body[k].startswith(("s_waitcnt", "s_barrier", "s_cbranch", "s_branch", "s_setprio")):
                return False
            dk, uk = f(k)
            if dk & (uj | dj) or uk & dj:
                return False
        if x["lit"] is not None and y["lit"] is not None and x["lit"] != y["lit"]:
            return False
        return vopd_registers_ok(x, y)
    used, n = set(), 0
    for i in sorted(el):
        if i in used:
            continue
        for j in range(i + 1, min(i + win, hi) + 1):
            if j in el and j not in used and can(i, j):
                used |= {i, j}
                n += 1
                break
    return n, len(el)


def backedge_block(body):
    """(first, last) of the register-copy run (moves at most 3 apart) whose last move sits within 8 instructions of the loop's end, or
    None: the allocator's back-edge copies (E179 / #3542; C5: flash2's +882..+946, 65 v_mov). Informational (C18's vopd line): it joins
    runs over gaps of up to 3 other instructions, so on vsplit it spans an s_barrier (64 moves, s_add_u32, s_barrier, 65 moves: 129 ->
    64 pairs; C19 review F2). The Loom candidate uses edge_group(), which stops at any non-move."""
    mv = [j for j, x in enumerate(body) if copy_moves(x)]
    if not mv or len(body) - 1 - mv[-1] > 8:
        return None
    a = len(mv) - 1
    while a > 0 and mv[a] - mv[a - 1] <= 3:
        a -= 1
    return mv[a], mv[-1]


EDGE_PASS = ("s_delay_alu", "s_nop", "v_swap")   # inside an edge-copy group besides moves: Loom's exchange moves and issue hints


def edge_group(body):
    """(first, last) of the edge-copy group before a loop body's closing branch (body[-1]), or None: the maximal run of VALU moves,
    v_swap, s_delay_alu and s_nop ending right before it, holding at least one register copy. encode_branch_packet emits a branch's
    group contiguously right before it (v0.2.0 emit/native/amdgpu/encoding.c:1719-1728), so any other instruction ends it (C19
    review F2: vsplit's 64 moves before an s_barrier are not its edge group)."""
    j = len(body) - 2
    while j >= 0 and (moves(body[j]) or body[j].startswith(EDGE_PASS)):
        j -= 1
    mv = [k for k in range(j + 1, len(body) - 1) if copy_moves(body[k])]
    return (mv[0], mv[-1]) if mv else None


def wmma_dep1(body):
    """WMMAs whose accumulator C was written by a WMMA at most one WMMA earlier (D -> C gap 0 or 1), the body run twice so the
    back-edge counts: informational, the D -> C latency is undocumented (C5 R6a; flash2's QK^T 2-accumulator chains: 16)"""
    stream = body + body
    wm = [i for i, x in enumerate(stream) if x.startswith("v_wmma")]
    ops = {i: [regs(o) for o in re.split(r",(?![^\[]*\])", stream[i].partition(" ")[2])[:4]] for i in wm}
    n = 0
    for k, i in enumerate(wm):
        if i < len(body):
            continue
        for m in range(k - 1, -1, -1):
            if ops[wm[m]][0] & ops[i][3]:
                n += k - m - 1 <= 1
                break
    return n


def vopd(r, ins):
    """the hot loop's unformed VOPD pairs as allocated, its back-edge copy block (moves, pairs inside it) and WMMA D -> C gap <= 1
    count; note `vopd` at >= VOPD_MIN unformed pairs (C5 / E551 corrected by E580, E585: flash2_f16o 35 of 129, back-edge 65 moves ->
    31 pairs, dep1 16; m256 GEMM 0 of 1)"""
    if r.get("where") != "loop":
        return
    t, e = r["hot_range"]
    body = ins[t:e + 1]
    r["vopd_pairs"], r["vopd_eligible"] = vopd_pairs(body)
    bb = backedge_block(body)
    r["backedge"] = (sum(copy_moves(x) for x in body[bb[0]:bb[1] + 1]), vopd_pairs(body, *bb)[0], bb) if bb else None
    r["wmma_dep1"] = wmma_dep1(body)
    if r["vopd_pairs"] >= VOPD_MIN:
        r.setdefault("notes", []).append("vopd")


IDX_OPS = {"index.add": lambda a, b: a + b, "index.sub": lambda a, b: a - b, "index.mul": lambda a, b: a * b,
           "index.div": lambda a, b: a // b if b else None, "index.ceildiv": lambda a, b: -(-a // b) if b else None,
           "index.rem": lambda a, b: a % b if b else None, "index.min": min, "index.max": max}
ROUNDS_MAX = 8192    # the largest size searched for the printed one-round bound (Loom's index.assume ranges on tokens / spans)


def idx_eval(lines, env):
    """%name -> value over index.constant / config.get / index.assume / IDX_OPS lines, starting from ENV: an int, ('cfg', key) a size
    the package sets at run time, ('op', name, a, b), or 'conflict' (unknown: any other op, or a name defined twice with different
    text; review r1 R13: an unmodelled op in the launcher is unknown, never a runtime size)"""
    seen, env = {}, dict(env)
    for line in lines:
        m = re.match(r"\s*(%\w+)\s*=\s*([\w.]+)\s+(.*)$", line)
        if not m:
            continue
        name, op, rest = m.groups()
        if name in seen and seen[name] != (op, rest):
            env[name] = "conflict"
            continue
        seen[name] = (op, rest)
        args = re.findall(r"%\w+", rest.split("[", 1)[0].split(":", 1)[0])
        if op == "index.constant":
            c = re.match(r"(-?\d+)\s*:\s*index", rest)
            env[name] = int(c.group(1)) if c else "conflict"
        elif op == "config.get":
            c = re.match(r"(@[\w.]+)\s*:\s*index", rest)
            env[name] = ("cfg", c.group(1)) if c else "conflict"
        elif op == "index.assume" and args and args[0] in env:
            env[name] = env[args[0]]
        elif op in IDX_OPS and len(args) == 2 and all(a in env for a in args):
            env[name] = ("op", op, env[args[0]], env[args[1]])
        else:
            env[name] = "conflict"
    return env


def idx_value(v, size):
    """an idx_eval() value with each size key at SIZE[key] (a dict) or at SIZE (an int, every key alike); None when unknown"""
    if isinstance(v, int):
        return v
    if isinstance(v, tuple) and v[0] == "cfg":
        return size.get(v[1]) if isinstance(size, dict) else size
    if isinstance(v, tuple):
        a, b = idx_value(v[2], size), idx_value(v[3], size)
        return None if a is None or b is None else IDX_OPS[v[1]](a, b)
    return None


def size_keys(v):
    """the config keys an idx_eval() value depends on"""
    if isinstance(v, tuple) and v[0] == "cfg":
        return {v[1]}
    if isinstance(v, tuple):
        return size_keys(v[2]) | size_keys(v[3])
    return set()


def launch_grids(texts):
    """kernel -> [[one value per workgroup axis] or None, per launch site]: each site's grid operands resolved in the definition that
    launches it (through dual-target selectors to the template.apply's arguments) and fed to the kernel's kernel.launch.config
    workgroups(...)"""
    blocks = []
    for t in texts:
        starts = [m.start() for m in re.finditer(r"(?m)^[a-z]\w*\.def\b", t)] + [len(t)]
        blocks += [t[a:b] for a, b in zip(starts, starts[1:])]
    configs, sels = {}, {}
    for b in blocks:
        m = re.match(r'kernel\.def (?:target\(@\w+\) )?export\("(\w+)"\) @\w+\(([^)]*)\)\s*\{', b)
        if m:
            head = b[m.end():b.find("} launch(")]
            configs[m.group(1)] = ([x.split(":")[0].strip() for x in m.group(2).split(",") if x.strip()], head)
        m = re.match(r"template\.def<@(sel\d+_\w+)> target\(@rdna3\) @\w+\(([^)]*)\)", b)
        if m:
            g = re.search(r"kernel\.launch @(\w+)\[([^\]]*)\]", b)
            if g:
                sels[m.group(1)] = ([x.split(":")[0].strip() for x in m.group(2).split(",")], g.group(1),
                                    [x.strip() for x in g.group(2).split(",")])
    out = collections.defaultdict(list)
    for b in blocks:
        if b.startswith("template.def"):
            continue
        sites = [(m.group(1), [x.strip() for x in m.group(2).split(",")]) for m in re.finditer(r"kernel\.launch @(\w+)\[([^\]]*)\]", b)]
        for m in re.finditer(r"template\.apply<@(sel\d+_\w+)>\(([^)]*)\)", b):
            if m.group(1) in sels:
                params, k, grid = sels[m.group(1)]
                args = dict(zip(params, [x.strip() for x in m.group(2).split(",")]))
                sites.append((k, [args.get(g, "?") for g in grid]))
        if not sites:
            continue
        env = idx_eval(b.split("\n"), {})
        for k, ops in sites:
            if k not in configs:
                continue
            params, head = configs[k]
            if len(ops) != len(params):
                out[k].append(None)
                continue
            cenv = idx_eval(head.split("\n"), dict(zip(params, [env.get(o, "conflict") for o in ops])))
            w = re.search(r"workgroups\(([^)]*)\)", head)
            out[k].append([cenv.get(x.strip(), "conflict") for x in w.group(1).split(",")] if w else None)
    return out


def device_domains(target):
    """occupancy domains (WGPs) of the device: simds / simds_per_occupancy_domain from common/devices/<target>.json, or None"""
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "devices", f"{target}.json")
    d = json.load(open(f)) if target and os.path.exists(f) else {}
    s, p = d.get("simds"), d.get("simds_per_occupancy_domain")
    return s // p if isinstance(s, int) and isinstance(p, int) and p and s % p == 0 else None


def grid_rounds(sites, wgs, domains):
    """one round's figures over a kernel's launch sites at the reachable tier: {"resident", "sites": [(size key or None, workgroups
    at size 1, one-round bound)], "unknown"}. The bound is the largest size up to ROUNDS_MAX whose grid still fits one round (None
    when even size 1 does not); a constant grid has key None and bound 1 when it fits. Unknown: an unresolved value, a grid of two
    size keys, a constant empty grid, or one that shrinks as its size grows. No served size is claimed (review r2: Loom 0.2.0 admits
    tokens and spans up to the prefill capacity, model_registration.c / schedule.c loom_serve_text_default_shapes and the package's
    prepare.loom stage configuration)"""
    res = wgs * domains
    out, unk = [], 0
    for s in sites:
        keys = set().union(*[size_keys(x) for x in s]) if s and "conflict" not in s else None
        if keys is None or len(keys) > 1:
            unk += 1
            continue
        total = lambda n: (lambda xs: None if None in xs else math.prod(xs))([idx_value(x, n) for x in s])
        low, top = total(1), total(ROUNDS_MAX)
        if low is None or top is None or top < low or (not keys and low <= 0):
            unk += 1
            continue
        bound = None
        if low <= res and not keys:
            bound = 1
        elif low <= res:
            lo, hi = 1, ROUNDS_MAX
            while lo < hi:
                mid = (lo + hi + 1) // 2
                lo, hi = (mid, hi) if total(mid) <= res else (lo, mid - 1)
            bound = lo
        out.append((next(iter(keys), None), low, bound))
    return {"resident": res, "sites": out, "unknown": unk}


def rounds(rows, texts):
    """r['rounds'] (grid_rounds) for every kernel with launch sites and a verified tier: information only, never a note (review r2,
    option iii: an admitted-size claim is not the linter's to make). E539's question, 'are all workgroups resident at once', is
    answered by the one-round bound in the size variable, to be read against the sizes a served profile shows"""
    grids = launch_grids(texts)
    for r in rows:
        o = r.get("occ")
        if not o or r["kernel"] not in grids:
            continue
        dom = device_domains(o.get("target"))   # also None for an unverified tier: tier_state's {"why"} carries no target
        if dom:
            r["rounds"] = grid_rounds(grids[r["kernel"]], o["wgs"], dom)


def rounds_info(r):
    """the information line of a kernel every site of which fits one round at size 1 with >= 2 workgroups somewhere, or None"""
    g, o = r.get("rounds"), r.get("occ")
    if not g or g["unknown"] or not g["sites"] or any(x[2] is None for x in g["sites"]) or max(x[1] for x in g["sites"]) < 2:
        return None
    by = {}
    for k, _, b in g["sites"]:
        by[k] = min(by.get(k, b), b)
    lim = "; ".join("constant grid" if k is None else f"{k.split('.')[-1]} <= {b}" for k, b in sorted(by.items(), key=str))
    return (f"rounds {r['kernel']} (info): {len(g['sites'])} launch site(s), {max(x[1] for x in g['sites'])} workgroups at size 1, "
            f"{g['resident']} resident ({o['wgs']} per WGP at {tier_text(o)} waves/SIMD{', a bound' if o['bound'] else ''}); one round "
            f"while {lim}")


def rounds_summary(rows):
    """one line naming the kernels with launch sites whose grid rounds could not resolve (review r1 F5)"""
    unk = sorted(r["kernel"] for r in rows if r.get("rounds") and r["rounds"]["unknown"])
    return f"rounds: not resolved ({len(unk)}): {' '.join(unk) or '-'}"


def lesson_lines(rows):
    """the lines of the C18 notes (guardwait, vopd) and rounds' information lines, by kernel, and the rounds summary"""
    out = []
    for r in sorted(rows, key=lambda r: r["kernel"]):
        n = r.get("notes", [])
        line = rounds_info(r)
        if line:
            out.append(line)
        if "guardwait" in n:
            i, b, v = r["guard_waits"][0]
            out.append(f"guardwait {r['kernel']}: {len(r['guard_waits'])} vmcnt wait(s) inside a conditional block drain loads issued "
                       f"before its branch, younger than every load the block needs (first: instruction {i}, branch at {b}, {v} "
                       f"load(s); E520)")
        if "vopd" in n:
            be = r.get("backedge")
            out.append(f"vopd {r['kernel']}: {r['vopd_pairs']} VOPD pairs unformed as allocated ({r['vopd_eligible']} eligible single VALU)"
                       + (f"; back-edge copy block {be[0]} moves -> {be[1]} pairs" if be else "")
                       + f"; WMMA D->C gap <= 1: {r.get('wmma_dep1')} (latency undocumented; C5 / E551 / E580)")
    if any(r.get("rounds") for r in rows):
        out.append(rounds_summary(rows))
    return out


EDGE_PAIRS_MIN = 8   # VOPD pairs in one back-edge copy group at or above which a vopd-edge candidate is printed (C19, fire data 11-25 on
# pkg-e480: 230 groups at 0-3 pairs, 224 at 4-7 (almost all g_head_mv_t16_q8_p40's, a DRAM-bound decode matvec), 23 at >= 8 in 17 kernels;
# the bar sits in a narrow gap: v_norm_ln_1152 and g_gdn_fold at 7 below it, g_dfl_ctx_split at 10 the lowest above it)
LOOM_TIER_DOC = ("loom/docs/src/workflows/compile-report-queries.md:122-124 (v0.2.0: 'the producer's reduction required for the "
                 "jointly selected next residency tier') against Loom's own whole-workgroup ceil (planning/occupancy.c:377 "
                 "wave_limit_for_workgroups)")
LOOM_VOPD_SRC = ("v0.2.0 planning/packet_plan.c:72-77 (the VOPD plan walks schedule packets only); a branch's edge-copy group is "
                 "expanded at emission (encode_branch_packet, emit/native/amdgpu/encoding.c:1719-1728 -> encode_move_range_impl, "
                 "encoding.c:1016-1072) with no VOPD lookup")
CAND_HEAD = ("loom candidates (prompts for a compiler packet, never verdicts: each needs a minimal .loom repro, upstream main and "
             "existing issues checked, and the user's approval before anything is posted):")


def loom_candidates(name, ins, addr, rep):
    """Loom upstream CANDIDATES for one kernel (C19), never flags or notes: ([{"kind", "line", ...}], [why not compared]).
    tier: the compile report's next tier (target_resources.residency.next_better_tier, or its current tier) differs from tier_state()'s
      whole-workgroup arithmetic (HRX-like E533: flash2 384 lanes reported 6 -> 7 at -21 VGPR / -18,944 B; a workgroup seats 3 waves
      per SIMD, so the next tier is 9 at VGPR <= 168 and LDS <= 43,008 B). An unverified tier is counted, never a candidate.
    vopd-edge: a loop's back-edge copy group (edge_group: the moves, swaps and issue hints right before the backward branch) in
      which vopd_pairs(), Loom 0.2.0's planner rules with the order constraint (neither half reads the other's destination, the
      copies being a sequentialized parallel copy), forms >= EDGE_PAIRS_MIN pairs: moves the allocator emits after the VOPD plan
      (HRX-like E592: flash2_f16o 65 moves -> 31 pairs). A details report's move_causes.branch_edge is cited when present."""
    out, skip = [], []
    if rep is None:
        skip.append("no compile report")
    else:
        occ = occ_inputs(rep)
        st = tier_state(occ, device_simds(occ["target"]))
        res = ((rep.get("target_resources") or {}).get("residency") or {})
        cur, nb = res.get("current_tier"), res.get("next_better_tier")
        if "why" in st:
            skip.append(st["why"])
        elif isinstance(cur, int) and isinstance(nb, int):
            rn = nb if nb > cur else None
            if (cur, rn) != (st["tier"], st["next"]):
                red = ", ".join(f"{(x.get('name') or '?').split('.')[-1]} -{x['reduction_units_to_next_better_tier']}"
                                for x in occ["rows"] if x.get("reduction_units_to_next_better_tier"))
                w = -(-occ["wg"] // occ["wave"])
                need = " and ".join(f"{'VGPR' if n == 'vgpr' else n.upper()} <= {mx}" for n, _, _, mx, _ in st["needs"])
                ours = (f"{tier_text(st)} -> " + (f"{st['next']} at {need}" if st["next"] is not None else
                                                  (st["blocked"] or "none (wave cap)")))
                out.append({"kind": "tier", "report": (cur, rn), "whole": (st["tier"], st["next"]),
                            "line": f"loomcand tier {name}: report tier {cur} -> {rn if rn is not None else 'none'}"
                                    + (f" ({red})" if red else "") + f"; whole workgroups ({occ['wg']} lanes = {w} waves, "
                                    f"{st['q']} per SIMD{', bound: placement undocumented' if st['bound'] else ''}): {ours}; "
                                    f"contradicts {LOOM_TIER_DOC}"})
    groups = []
    for t, e in loops(ins, addr):                  # each group ends right before its own branch, and a branch is no move, so two
        body = ins[t:e + 1]                        # branches never share one (r1's dedupe is gone with backedge_block's 8-instruction window)
        bb = edge_group(body)
        if not bb:
            continue
        pairs = vopd_pairs(body, *bb)[0]
        if pairs >= EDGE_PAIRS_MIN:
            groups.append((sum(copy_moves(x) for x in body[bb[0]:bb[1] + 1]), pairs, t + bb[0]))
    if groups:
        bc = [c for c in ((rep or {}).get("move_causes") or {}).get("causes") or [] if c.get("cause") == "branch_edge"]
        ev = (f"report (kernel-wide, all branches): branch_edge {bc[0].get('unit_count')} moves in {bc[0].get('packet_count')} packets" if bc else
              "report: move causes not itemised (summary report; --compile-report=details itemises branch_edge)")
        out.append({"kind": "vopd-edge", "groups": [(m, p) for m, p, _ in groups],
                    "line": f"loomcand vopd-edge {name}: " + "; ".join(f"back-edge copy group at instruction {i}: {m} moves -> {p} "
                                                                         f"VOPD pairs" for m, p, i in groups)
                            + f" by Loom's own pairing rules, order respected; {ev}; {LOOM_VOPD_SRC}"})
    return out, skip


SHAPE_NOTES = ("reread", "addr64", "guardwait", "vopd", "ldsbank")   # the notes lint() sets per kernel (the rest need texts or peers)


def sha256(path):
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


OBJ_EXT = (".hsaco", ".json")    # what a compile writes per kernel (code object, compile report)


def provenance(lc, progs, roots, objects):
    """what compiled a code-object directory: the compiler (real path), its hrx-loom version, each program's sha256, the roots regex
    and the sha256 of each object this compile wrote (OBJECTS: paths); written by compile_programs as OUT/loom-provenance, read by the
    SDK differential (review F1: objects left from another compile must not take this one's provenance)"""
    import loom_suggest
    return {"compiler": os.path.realpath(lc), "version": loom_suggest.version(lc), "roots": roots,
            "programs": {os.path.basename(p): sha256(p) for p in progs},
            "objects": {os.path.basename(o): sha256(o) for o in objects}}


def read_provenance(d):
    """a directory's loom-provenance plus "path" and, when any *.hsaco / *.json in D is unlisted, changed or missing (an object from
    another compile, a kernel that failed under this compiler keeping its old object, an edit), "unverified" naming the first ones;
    None when absent or unreadable"""
    f = os.path.join(d, "loom-provenance") if d and os.path.isdir(d) else None
    try:
        p = json.load(open(f)) if f and os.path.exists(f) else None
    except (OSError, ValueError):
        return None
    if not isinstance(p, dict):
        return None
    p["path"] = f
    rec = p.get("objects")
    if not isinstance(rec, dict):
        p["unverified"] = "no object hashes"
        return p
    have = sorted(x for x in os.listdir(d) if x.endswith(OBJ_EXT))
    bad = ([f"unlisted: {x}" for x in have if x not in rec] + [f"changed: {x}" for x in have if x in rec and sha256(os.path.join(d, x)) != rec[x]]
           + [f"missing: {x}" for x in sorted(rec) if x not in have])
    p["unverified"] = (", ".join(bad[:3]) + (f" (+{len(bad) - 3})" if len(bad) > 3 else "")) if bad else None
    return p


def sdk_diff(cur, base, pc, pb):
    """the SDK differential (C19; by hand so far: HRX-like E177 / E227): kernel -> row of two compiles of the same programs by two
    compilers (provenance pc / pb). Per kernel in both, the shapes (flags, the per-kernel notes SHAPE_NOTES, candidate kinds) that
    appear or disappear and regressions() each way; a kernel compiled by one side only. Fails closed: without both provenances, with
    an object in either directory that its provenance does not list with the same sha256 (read_provenance's "unverified": review
    F1), different programs or roots, one compiler (same path or same version) or an unknown version, one 'not attributable' line
    and nothing else."""
    unv = [f"{x.get('path', '?')}: {x['unverified']}" for x in (pc, pb) if x and x.get("unverified")]
    known = lambda x: x.get("version") and x.get("version") != "version unknown"
    why = ("provenance missing" if not (pc and pb) else f"objects not as recorded: {'; '.join(unv)}" if unv
           else "programs differ" if pc.get("programs") != pb.get("programs") or not pc.get("programs")
           else "roots differ" if pc.get("roots") != pb.get("roots")
           else "one compiler" if pc.get("compiler") == pb.get("compiler") or pc.get("version") == pb.get("version")
           else "compiler version unknown" if not (known(pc) and known(pb)) else None)
    if why:
        return [f"loomcand sdk: not attributable ({why}): an SDK differential needs both directories compiled by kernel_lint "
                f"(--package / --program) from identical programs by two compilers"]
    vb, vc = pb.get("version"), pc.get("version")
    shapes = lambda r: ({f for f in r.get("flags", []) if not f.startswith("regress:")} | {n for n in r.get("notes", []) if n in SHAPE_NOTES}
                        | {"loomcand:" + c["kind"] for c in r.get("loomcand", [])})
    out, n_same = [], 0
    for k in sorted(set(cur) | set(base)):
        if k not in cur or k not in base:
            out.append(f"loomcand sdk {k}: compiled by {vb if k in base else vc} only ({vc if k in base else vb} has no object)")
            continue
        a, b = shapes(base[k]), shapes(cur[k])
        flip = lambda x: re.sub(r"^(\D+)(\d+)->(\d+)$", r"\1\3->\2", x) if "->" in x else x + " gone"
        worse, better = regressions(cur[k], base[k]), [flip(x) for x in regressions(base[k], cur[k])]
        if a == b and not worse and not better:
            n_same += 1
            continue
        out.append(f"loomcand sdk {k}: {vb} -> {vc}: " + " ".join([f"+{x}" for x in sorted(b - a)] + [f"-{x}" for x in sorted(a - b)])
                   + (f" worse {','.join(worse)}" if worse else "") + (f" better {','.join(better)}" if better else ""))
    return out + [f"loomcand sdk: {vb} -> {vc}: {len(out)} kernel line(s), {n_same} kernel(s) with the same shapes; identical programs "
                  f"and objects as recorded in {pb.get('path', '?')} and {pc.get('path', '?')}"]


def lint_listing(name, ins, addr, res=(None, None)):
    """the lint of one kernel's decoded listing (instructions, byte addresses), as lint() reports it"""
    r = {"kernel": name, "instr": len(ins), "flags": []}
    r["vgpr"], r["occ"] = res
    o = r["occ"] if r["occ"] and "why" not in r["occ"] else None
    r["waves"], r["waves_bound"] = (o["tier"], o["bound"]) if o else (None, None)
    r["next_tier"], r["to_next"] = (o["next"], {n: c for n, _, _, _, c in o["needs"]}) if o else (None, None)
    h = hot(ins, addr)
    if h is None or not any(ins[j].startswith(MM) for j in common_path(ins, h[2], h[0], h[1])):
        b = full = ins                                                # no multiply loop: the whole kernel (e.g. a dequant pass)
        r["where"] = "kernel"
    else:
        t, e, L = h
        b = [ins[j] for j in common_path(ins, L, t, e)]              # the hot loop's common path
        full = ins[t:e + 1]
        r["where"] = "loop"
        r["hot_range"] = (t, e)
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
    r["s_barrier"] = sum(1 for x in ins if x.startswith("s_barrier"))
    r["join"] = join_runs(ins)
    if r["join"] >= JOIN_RUN:
        r["flags"].append("join")
    guardwait(r, ins, addr)
    vopd(r, ins)
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


MASK_NEG = -1e4      # a score mask's fill: a float constant at or below this (live: -3.0e38)


def kernel_texts(texts):
    """kernel -> its text in a package's program texts: from its `kernel.def` line to the next top-level definition"""
    out = {}
    for t in texts:
        starts = [m.start() for m in re.finditer(r"(?m)^[a-z]\w*\.def\b", t)] + [len(t)]
        for a, b in zip(starts, starts[1:]):
            m = re.match(r'kernel\.def (?:target\(@\w+\) )?export\("(\w+)"\)', t[a:b])
            if m:
                out[m.group(1)] = t[a:b]
    return out


def kernel_starts(paths):
    """kernel -> (program file, 1-based line of its kernel.def), so text-level findings name lines a reader can open"""
    out = {}
    for p in paths:
        for i, line in enumerate(open(p).read().split("\n")):
            m = re.match(r'kernel\.def (?:target\(@\w+\) )?export\("(\w+)"\)', line)
            if m:
                out[m.group(1)] = (p, i + 1)
    return out


def masked_v(text):
    """the E444 shape in one kernel's Loom text, or None: a key-valid bit masks the score (a select whose other side is a float
    constant <= MASK_NEG) while nothing zeroes a value outside the score chain (no select of a non-score value against a 0.0
    constant) and the compared key is never clamped (no index.min of a compare operand: the clamped-key form reads a live key).
    Zeroing p is not a guard: p = +0 times a NaN V is NaN (E441). Returns the score-mask conditions"""
    defs, ops = {}, []
    for line in text.split("\n"):
        m = re.match(r"\s*(%\w+(?:\s*,\s*%\w+)*)\s*=\s*([\w.]+)(?:<[^>]*>)?\s*(.*)", line)
        if m:
            names = re.findall(r"%\w+", m.group(1))
            ops.append((names, m.group(2), m.group(3)))
            for n in names:
                defs[n] = (m.group(2), m.group(3))
    def const(n):
        op, rest = defs.get(n, ("", ""))
        if op == "vector.splat":
            s = re.match(r"(%\w+)", rest)
            return const(s.group(1)) if s else None
        v = re.match(r"(-?[\d.]+(?:e[-+]?\d+)?)\s*:\s*(\S+)", rest) if op in ("scalar.constant", "vector.splatf") else None
        return float(v.group(1)) if v and re.search(r"\bf\d+|xf\d+", v.group(2)) else None
    sel = [(names, *re.findall(r"%\w+", rest)[:3]) for names, op, rest in ops if op in ("scf.select", "vector.select")]
    score = [s for s in sel if len(s) == 4 and (const(s[3]) or 0.0) <= MASK_NEG]
    if not score:
        return None
    taint = {n for s in score for n in s[0]}                     # the score chain: masked scores and everything computed from them
    soft = False
    for names, op, rest in ops:
        if taint & set(re.findall(r"%\w+", rest)):
            taint.update(names)
            soft |= bool(re.search(r"\.exp2?f$", op))
    if not soft:
        return None                                              # no softmax on the masked score: not attention (a top-k's logit mask)
    if any(len(s) == 4 and const(s[3]) == 0.0 and s[2] not in taint for s in sel):
        return None                                              # a non-score value zeroed under a bit (E444's fix: the V scale)
    conds, seen, keys = [s[1] for s in score], set(), set()
    while conds:
        c = conds.pop()
        if c in seen:
            continue
        seen.add(c)
        op, rest = defs.get(c, ("", ""))
        if re.match(r"(?:scalar|vector)\.(?:andi|ori)$", op):
            conds += re.findall(r"%\w+", rest)[:2]
        elif op.endswith(".cmp"):
            keys.update(re.findall(r"%\w+", rest)[:2])
    if any(op == "index.min" and keys & set(re.findall(r"%\w+", rest)[:2]) for _, op, rest in ops):
        return None                                              # the compared key clamped: the load reads a live key
    return sorted({s[1] for s in score})


# The barrier check's allowlists (Loom 0.2.0 dialect defs; review 11-25: every spelling the r1 model did not recognise read as
# "no access" or "disjoint"). An op not listed here that takes a workgroup region or a view of one is a read-write access of the
# whole region, and its results join the region.
BAR_ACCESS = {"view.load": "r", "vector.load": "r", "vector.fragment.load": "r", "view.store": "w", "vector.store": "w",
              "vector.fragment.store": "w", "view.atomic.rmw": "a"}
BAR_VIEW = ("buffer.view", "view.subview", "view.refine", "view.bitcast", "view.duplicate", "buffer.duplicate", "buffer.assume.alignment",
            "buffer.assume.memory_space", "buffer.assume.noalias", "buffer.assume.same_root", "buffer.restrict.begin", "scf.select")
# uniformity: results the same for every invocation of a workgroup whatever their operands (kernel/defs.py), and ops whose result
# is uniform when all their operands are (arithmetic, views, plain loads; scf.for / scf.if inside a callee). Every other op's result
# may differ: ids, lanes, shuffles, broadcasts, subgroup reductions / votes, scans, fragments / mma (lane-distributed), atomics and
# loads with an attribute such as <volatile> (the value depends on when each invocation ran), and anything unlisted.
UNIFORM_SOURCE = ("kernel.workgroup.id", "kernel.workgroup.size", "kernel.workgroup.count", "kernel.subgroup.count",
                  "kernel.subgroup.size", "kernel.workgroup.reduce")
UNIFORM_PURE = re.compile(r"(?:index|scalar|buffer|view|vector)\.[\w.]+$|scf\.(?:select|for|if)$")
UNIFORM_NOT = re.compile(r"vector\.fragment|vector\.mma|(?:view|vector|buffer)\.atomic")


def uniform_op(op, s):
    """'source' (uniform whatever its operands), 'pure' (uniform when its operands are) or None (may differ: fail closed)"""
    if op.startswith(UNIFORM_SOURCE):
        return "source"
    if UNIFORM_PURE.match(op) and not UNIFORM_NOT.match(op) and not (op.startswith(("view.load", "vector.load", "vector.gather"))
                                                                       and re.search(re.escape(op) + "<", s)):
        return "pure"                                     # a load with an attribute (<volatile> ...) may read another value per wave
    return None


def _calls_nonuniform(text):
    """a callee whose result may differ for uniform arguments: any result-producing op of it outside the uniform allowlist"""
    return any(uniform_op(m.group(1), m.group(0)) is None for m in re.finditer(r"(?m)^\s*%[\w\s,%]*=\s*([\w.]+).*$", text))
OP_LINE = re.compile(r"(?:(%\w+(?:\s*,\s*%\w+)*)\s*=\s*)?([\w.]+)")


class NotAnalysed(Exception):
    """a kernel the barrier check cannot model soundly (the reason is the message)"""


class Divergent(NotAnalysed):
    """a barrier or workgroup collective under control flow that may differ between invocations: a finding, and no further model"""


def func_texts(texts):
    """callee -> its text in a package's program texts (func.def, to the next top-level definition)"""
    out = {}
    for t in texts:
        starts = [m.start() for m in re.finditer(r"(?m)^[a-z]\w*\.def\b", t)] + [len(t)]
        for a, b in zip(starts, starts[1:]):
            m = re.match(r"func\.def(?:\s+\w+)*\s+@(\w+)\(", t[a:b])
            if m:
                out[m.group(1)] = t[a:b]
    return out


def ir_tree(text):
    """the launch body of a kernel's Loom text as a tree: ('op', line, text) | ('for', line, text, body) |
    ('if', line, text, then, else); line numbers count from the kernel.def line (0). Raises NotAnalysed on a region it
    does not know (anything but scf.for / scf.if / else opening a block)"""
    lines = text.split("\n")
    start = next((i for i, l in enumerate(lines) if l.startswith("} launch(")), None)
    if start is None:
        raise NotAnalysed("no launch body")
    while not lines[start].rstrip().endswith("{"):
        start += 1
    root = []
    stack = [root]
    node = None
    for i in range(start + 1, len(lines)):
        s = lines[i].strip()
        if not s or s.startswith("//"):
            continue
        if s == "}":
            if len(stack) == 1:
                return root
            stack.pop()
            continue
        if re.match(r"\}\s*else\s*\{$", s):
            node = stack.pop()
            parent = stack[-1][-1]
            if parent[0] != "if" or parent[4] is not None:
                raise NotAnalysed(f"line {i}: else without its scf.if")
            parent[4] = []
            stack.append(parent[4])
            continue
        m = OP_LINE.match(s)
        op = m.group(2) if m else ""
        if s.endswith("{"):
            if op == "scf.for":
                it = ["for", i, s, []]
                stack[-1].append(it)
                stack.append(it[3])
            elif op == "scf.if":
                it = ["if", i, s, [], None]
                stack[-1].append(it)
                stack.append(it[3])
            else:
                raise NotAnalysed(f"line {i}: unknown region op {op or s[:40]}")
            continue
        stack[-1].append(["op", i, s])
    raise NotAnalysed("unterminated launch body")


def _yields(block):
    """the names the block's scf.yield yields, in order (empty when it has none)"""
    for it in block[::-1]:
        if it[0] == "op" and it[2].startswith("scf.yield"):
            return re.findall(r"%\w+", it[2].split(":", 1)[0])
    return []


def _taint(tree, funcs=None):
    """names whose value may differ between the workgroup's invocations: every result of an op off the uniform allowlist
    (uniform_op) and everything computed from one, a callee's result unless its text is found and all uniform, a loop's
    variable, carried values and results under a non-uniform bound, a region's result under a non-uniform condition or yielding
    such a value: a fixpoint over the tree"""
    T = set()
    def walk(block):
        ch = False
        for it in block:
            m = OP_LINE.match(it[2])
            if not m:
                continue                                  # seq refuses the line
            res = re.findall(r"%\w+", m.group(1) or "")
            new = set()
            if it[0] == "op":
                rest = it[2][m.end():]
                tainted_in = bool(set(re.findall(r"%\w+", rest)) & T)
                if m.group(2) == "func.call":           # a callee reading an invocation id returns a non-uniform value
                    callee = (funcs or {}).get((re.search(r"@(\w+)", rest) or [None, ""])[1])
                    bad = tainted_in or callee is None or _calls_nonuniform(callee)
                else:
                    u = uniform_op(m.group(2), it[2])
                    bad = u is None or (u == "pure" and tainted_in)
                if bad:
                    new.update(res)
            elif it[0] == "for":
                h = re.match(r"(?:.*?=\s*)?scf\.for\s+(%\w+)\s*=\s*\[([^\]]*)\](?:\((.*?)\))?", it[2])
                if not h:
                    raise NotAnalysed(f"line {it[1]}: scf.for header not parsed")
                args = re.findall(r"(%\w+)\s*=\s*(%\w+|[-\w.]+)", h.group(3) or "")
                if set(re.findall(r"%\w+", h.group(2))) & T:      # a non-uniform trip count: everything the loop carries
                    new.update([h.group(1)] + [a for a, _ in args] + res)
                ys = _yields(it[3])
                for k, (a, init) in enumerate(args):
                    if init in T or (k < len(ys) and ys[k] in T) or a in T:
                        new.add(a)
                        if k < len(res):
                            new.add(res[k])
                ch |= walk(it[3])
            else:
                c = re.match(r"(?:.*?=\s*)?scf\.if\s+(%\w+)", it[2])
                cond = c.group(1) if c else None
                ys, ye = _yields(it[3]), _yields(it[4] or [])
                for k, r in enumerate(res):
                    if cond in T or (k < len(ys) and ys[k] in T) or (k < len(ye) and ye[k] in T):
                        new.add(r)
                ch |= walk(it[3])
                ch |= walk(it[4] or [])
            if not new <= T:
                T.update(new)
                ch = True
        return ch
    while walk(tree):
        pass
    return T


ELEM_BYTES = {"f64": 8, "i64": 8, "f32": 4, "i32": 4, "f16": 2, "bf16": 2, "i16": 2, "i8": 1, "i1": 1}


def view_shape(ty):
    """(static dims, element bytes) of a 'view<AxBxT>' type text, or None (a dynamic or unknown shape, a layout)"""
    m = re.fullmatch(r"view<([\dx]+)x(\w+)>", ty)
    if not m or m.group(2) not in ELEM_BYTES:
        return None
    return [int(x) for x in m.group(1).split("x")], ELEM_BYTES[m.group(2)]


def barrier_check(text, funcs=None, races=False):
    """the workgroup-memory hazards each kernel.barrier<workgroup> orders, from the kernel's Loom text (HRX-like E504: one extra
    barrier per tile cost +1.4 % of flash2 time). Model (fail closed throughout: what it does not recognise is an access of the
    whole region or 'not analysed', never no access and never disjoint):
      memory: one region per buffer.alloca<workgroup> / view.alloca<workgroup>; an alloca in a space other than workgroup /
        private / global, or buffer.alloca.init<workgroup> (a collective), is not analysed. Views derive from a region through
        the BAR_VIEW ops only; BAR_ACCESS ops read / write / update it; any other op with a region operand (masked, gather,
        expand, vector atomics, func.call, copies ...) reads and writes the whole region, and its results join the region.
        kernel.async.* / kernel.tensor.* (data lands after issue) are not analysed. Two accesses of one region overlap unless
        their views are buffer.views of the region at constant offsets with static shapes and disjoint byte extents (indices
        assumed in bounds, as Loom requires); there is no per-index model (r1's index lattice bound names regardless of scope
        and decided no live verdict, review 11-25 B3). Names bind in walk order, so a name reused in a sibling scope is rebound
        before its uses there.
      control: the structured body as a graph; a loop has a back-edge and a zero-trip exit, an if both arms (even under a
        constant condition), so its paths cover every executed one. A barrier or workgroup collective under control that may
        differ between invocations (_taint: only allowlisted uniform sources and ops on them are uniform) is 'divergent'.
      ordering: barrier B orders (a, b) when a reaches B and B reaches b with no other barrier on the way, they overlap and one
        writes (RAW / WAR / WAW; an atomic or an unlisted op writes). No such pair: B 'orders nothing'. A workgroup collective
        (kernel.workgroup.reduce, two s_barriers in the binary) is not counted as a barrier, so pairs it also orders stay B's.
      races (only with races=True, --barriers: super-linear, detail only): two different accesses that overlap, one writing,
        reachable from one another with no barrier between them (arms of an if whose condition differs between invocations are
        concurrent with each other); 'collective only' when a workgroup collective lies between them on every such path.
      scope: a kernel.barrier<workgroup> fences workgroup memory only (Loom 0.2.0 kernel/defs.py kernel_barrier: 'fence a named
        memory space'), so global accesses are not its hazards; any other barrier form is 'not analysed'.
    Returns {'status': 'analysed' | 'single wave' | 'no barriers' | 'not analysed: why' | 'divergent: why', 'barriers': [{'line',
    'pairs': [(writer-or-first line, second line, kind, region)], 'verdict'}], 'races': [(line, line, kind, region, label)],
    'collectives': n, 'expected_isa': s_barriers the binary should hold}; lines count from the kernel.def line (0)"""
    out = {"status": "analysed", "barriers": [], "races": [], "collectives": 0, "expected_isa": None}
    try:
        tree = ir_tree(text)
    except NotAnalysed as e:
        out["status"] = f"not analysed: {e}"
        return out
    if "kernel.barrier" not in text:
        out["status"] = "no barriers"
        return out
    head = text.split("\n} launch(", 1)[0]                # the launch config's own constants (the body may reuse a name)
    hc = {k: int(v) for k, v in re.findall(r"(%\w+)\s*=\s*index\.constant\s+(\d+)\s*:", head)}
    ws = re.search(r"workgroup_size\(([^)]*)\)", head)
    if ws:
        dims = [hc.get(x.strip()) for x in ws.group(1).split(",")]
        if all(d is not None for d in dims) and dims[0] * dims[1] * dims[2] <= 32:
            out["status"] = "single wave"               # one wave32 per workgroup: no other wave to order (Loom drops the s_barrier)
            return out
    funcs = funcs or {}
    K, region, vinfo = {}, {}, {}                         # name -> index constant / set of regions / byte extent, bound in walk order
    const = lambda t: int(t) if t.isdigit() else K.get(t)
    def bind(names):
        for d in names:
            K.pop(d, None)
            region.pop(d, None)
            vinfo.pop(d, None)
    def scope():                                          # a block's bindings end with it (Loom lets an inner name shadow an outer one)
        saved = (dict(K), dict(region), dict(vinfo))
        def close():
            for m, v in zip((K, region, vinfo), saved):
                m.clear()
                m.update(v)
        return close
    try:
        T = _taint(tree, funcs)
        nodes = [("entry", None, None)]                   # (kind, line, info): entry | bar | col | head | acc
        succ = collections.defaultdict(set)
        cross = collections.defaultdict(set)              # extra concurrency edges between the arms of a non-uniform if (races only)
        def callee_bad(name, seen=()):
            t = funcs.get(name)
            if t is None:
                return f"func {name}: text not found"
            if re.search(r"kernel\.barrier|kernel\.workgroup\.(?!id)\w+|kernel\.(?:async|tensor)\.|"
                         r"\.alloca(?:\.\w+)*(?![\w.])(?!<(?:private|global)>)", t):
                return f"func {name} synchronises, allocates workgroup memory or copies asynchronously"
            for c in re.findall(r"func\.call(?:\s+\w+)?\s+@(\w+)", t):
                if c not in seen and c != name:
                    r = callee_bad(c, seen + (name,))
                    if r:
                        return r
            return None
        def add(kind, line, info, preds):
            nodes.append((kind, line, info))
            for p in preds:
                succ[p].add(len(nodes) - 1)
            return [len(nodes) - 1]
        expected = [0]
        def seq(block, preds, mult, ctl):
            for it in block:
                if it[0] in ("for", "if") and set(re.findall(r"%\w+", it[2].split("=", 1)[-1])) & set(region):
                    raise NotAnalysed(f"line {it[1]}: a workgroup view flows through an scf.{it[0]}")
                if it[0] == "for":
                    h = re.match(r"(?:(.*?)=\s*)?scf\.for\s+(%\w+)\s*=\s*\[([^\]]*)\](?:\((.*?)\))?", it[2])
                    close = scope()
                    bind([h.group(2)] + re.findall(r"(%\w+)\s*=", h.group(4) or ""))
                    b = re.split(r"\s+(?:to|step)\s+", h.group(3).strip())
                    trip = None
                    if re.search(r"\bunroll\s*\{$", it[2]) and len(b) == 3 and all(const(x) is not None for x in b):
                        lo, hi, st = (const(x) for x in b)
                        trip = max(0, -(-(hi - lo) // st))
                    hd = add("head", it[1], None, preds)
                    ex = seq(it[3], hd, mult * (trip if trip is not None else 1),
                             ctl + ([it[1]] if set(re.findall(r"%\w+", h.group(3))) & T else []))
                    for p in ex:
                        succ[p].add(hd[0])
                    close()
                    bind(re.findall(r"%\w+", h.group(1) or ""))
                    preds = hd
                    continue
                if it[0] == "if":
                    c = re.match(r"(?:(.*?)=\s*)?scf\.if\s+(%\w+)", it[2])
                    if not c:
                        raise NotAnalysed(f"line {it[1]}: scf.if condition not parsed")
                    nonuni = c.group(2) in T
                    c2 = ctl + ([it[1]] if nonuni else [])
                    n0 = len(nodes)
                    close = scope()
                    t_ex = seq(it[3], preds, mult, c2)
                    close()
                    n1 = len(nodes)
                    e_ex = seq(it[4], preds, mult, c2) if it[4] is not None else list(preds)
                    close()
                    bind(re.findall(r"%\w+", c.group(1) or ""))
                    if nonuni and races:                    # one invocation in each arm at once
                        for x in range(n0, n1):
                            cross[x].update(range(n1, len(nodes)))
                    preds = sorted(set(t_ex) | set(e_ex))    # deduplicated: a list doubled per if, 2^k for k ifs (OOM 11-25)
                    continue
                s = it[2]
                m = OP_LINE.match(s)
                if not m:
                    raise NotAnalysed(f"line {it[1]}: op not parsed: {s[:40]}")
                op = m.group(2)
                defs = re.findall(r"%\w+", m.group(1) or "")
                uses = re.findall(r"%\w+", s[m.end():])
                bind(defs)
                if op.startswith(("kernel.async.", "kernel.tensor.")) or op == "kernel.exit":
                    raise NotAnalysed(f"line {it[1]}: {op} not modelled (asynchronous data or an early exit)")
                if op == "kernel.barrier" or op.startswith("kernel.barrier."):
                    if not re.match(r"kernel\.barrier<workgroup>\s+scope\(workgroup\)", s):
                        raise NotAnalysed(f"line {it[1]}: barrier form not modelled: {s[:60]}")
                    if ctl:
                        raise Divergent(f"line {it[1]}: barrier under control flow that may differ between invocations "
                                          f"(line {ctl[-1]})")
                    expected[0] += mult
                    preds = add("bar", it[1], None, preds)
                    continue
                if op == "kernel.return" and it is not tree[-1]:
                    raise NotAnalysed(f"line {it[1]}: a return before the end of the launch body (later barriers not reached by all)")
                if op.startswith("kernel.workgroup.") and op != "kernel.workgroup.reduce" and not op.startswith(
                        ("kernel.workgroup.id", "kernel.workgroup.size", "kernel.workgroup.count")):
                    raise NotAnalysed(f"line {it[1]}: workgroup collective {op} not modelled (its s_barrier count is unknown)")
                if op == "kernel.workgroup.reduce":
                    if ctl:
                        raise Divergent(f"line {it[1]}: workgroup collective under control flow that may differ between "
                                          f"invocations (line {ctl[-1]})")
                    out["collectives"] += mult
                    expected[0] += 2 * mult                 # a workgroup.reduce lowers to two s_barriers (GEMM: 3 + 2, dfl_topk 3 + 8)
                    preds = add("col", it[1], None, preds)
                    continue
                if re.search(r"\.alloca(?:\.|$)", op):
                    sp = re.match(r"[\w.]+<(\w+)>", s[m.start(2):])
                    space = sp.group(1) if sp else None
                    if space == "workgroup" and op in ("buffer.alloca", "view.alloca"):
                        region[defs[0]] = {defs[0]}
                    elif space not in ("private", "global"):
                        raise NotAnalysed(f"line {it[1]}: {op}<{space}> not modelled")
                    continue
                if op == "index.constant":
                    v = re.search(r"index\.constant\s+(\d+)\s*:", s)
                    if v and defs:
                        K[defs[0]] = int(v.group(1))
                    continue
                if op == "func.call":
                    why = callee_bad(re.search(r"@(\w+)", s).group(1))
                    if why:
                        raise NotAnalysed(f"line {it[1]}: {why}")
                if not any(u in region for u in uses):
                    continue
                rs = set().union(*[region[u] for u in uses if u in region])
                if op == "scf.yield":
                    raise NotAnalysed(f"line {it[1]}: a workgroup view is yielded")
                if op in BAR_VIEW:
                    for d in defs:
                        region[d] = rs
                    vm = re.match(r"buffer\.view\s+(%\w+)\[(%?\w+)\]\s*:\s*buffer\s*->\s*(view<[^>]*>)$", s[m.start(2):])
                    if op == "buffer.view" and vm and region.get(vm.group(1)) == {vm.group(1)}:
                        sh, off = view_shape(vm.group(3)), const(vm.group(2))
                        if sh and off is not None:
                            vinfo[defs[0]] = (off, off + sh[1] * math.prod(sh[0]))
                    continue
                mode = BAR_ACCESS.get(op)
                acc = {"regions": frozenset(rs), "mode": mode or "rw", "ext": None}
                if mode:                                    # a listed access: the byte extent of the view it indexes
                    vv = re.search(r"(%\w+)\[", s[m.end():])
                    acc["ext"] = vinfo.get(vv.group(1)) if vv else None
                else:
                    for d in defs:
                        region[d] = rs                      # an unlisted op's results may carry the region (F3: a callee's view)
                preds = add("acc", it[1], acc, preds)
            return preds
        seq(tree, [0], 1, [])
        out["expected_isa"] = expected[0]
    except Divergent as e:
        out["status"] = f"divergent: {e}"
        return out
    except NotAnalysed as e:
        out["status"] = f"not analysed: {e}"
        return out
    pred = collections.defaultdict(set)
    for x, ys in succ.items():
        for y in ys:
            pred[y].add(x)
    def reach(start, nxt, stops):
        seen, stack = set(), list(start)
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            if nodes[x][0] not in stops:
                stack.extend(nxt[x])
        return seen
    def conflict(x, y):
        a, b = nodes[x][2], nodes[y][2]
        if not a["regions"] & b["regions"] or a["mode"] == b["mode"] == "r":
            return False
        ea, eb = a["ext"], b["ext"]
        return not (ea and eb and (ea[1] <= eb[0] or eb[1] <= ea[0]))
    def kind(x, y):
        a, b = nodes[x][2]["mode"], nodes[y][2]["mode"]
        return "WAW" if a != "r" and b != "r" else ("RAW" if a != "r" else "WAR")
    reg = lambda x, y: "/".join(sorted(nodes[x][2]["regions"] & nodes[y][2]["regions"]))
    acc = [i for i, n in enumerate(nodes) if n[0] == "acc"]
    for b in [i for i, n in enumerate(nodes) if n[0] == "bar"]:
        before = [x for x in reach(pred[b], pred, ("bar", "entry")) if nodes[x][0] == "acc"]
        after = [y for y in reach(succ[b], succ, ("bar",)) if nodes[y][0] == "acc"]
        pairs = sorted({(nodes[x][1], nodes[y][1], kind(x, y), reg(x, y)) for x in before for y in after if conflict(x, y)})
        out["barriers"].append({"line": nodes[b][1], "pairs": pairs, "verdict": "orders" if pairs else "orders nothing"})
    if not races:
        return out
    conc = collections.defaultdict(set, {k: v | cross[k] for k, v in succ.items()})
    for k, v in cross.items():
        conc[k] |= v
    strict = {x: reach(conc[x], conc, ("bar",)) for x in acc}
    lenient = {x: reach(conc[x], conc, ("bar", "col")) for x in acc}
    for i, x in enumerate(acc):
        for y in acc[i + 1:]:
            if not conflict(x, y) or nodes[x][2]["mode"] == nodes[y][2]["mode"] == "a":
                continue
            if y in strict[x] or x in strict[y]:
                f, g = (x, y) if y in strict[x] else (y, x)    # named in the order of a barrier-free path (the back-edge's WAR)
                lab = "" if (g in lenient[f] or f in lenient[g]) else "collective only"
                out["races"].append((nodes[f][1], nodes[g][1], kind(f, g), reg(f, g), lab))
    return out


def barriers(rows, texts, funcs, roots=None, races=False):
    """attach barrier_check to each row whose kernel text is known, cross-checked against the binary: when the text's barriers
    (plus two per workgroup collective, times constant unroll counts) are not the binary's s_barrier count, the text is not
    what runs and no verdict is given (fail closed). Notes: 'barrier' (a barrier orders nothing), 'divbar' (a barrier or collective
    under control flow that may differ between invocations); races are computed only when RACES (--barriers) and listed, not
    noted. With texts, a binary with s_barriers whose text is missing is 'not analysed'; without any text (another engine's
    objects) the rule does not apply"""
    for r in rows:
        if roots and not re.search(roots, r["kernel"]):
            continue
        t = texts.get(r["kernel"])
        if t is None:
            if texts and r.get("s_barrier"):
                r["barrier"] = {"status": "not analysed: no Loom text for this kernel", "barriers": [], "races": [],
                                "collectives": 0, "expected_isa": None}
            continue
        b = barrier_check(t, funcs, races)
        if b["status"] == "analysed" and b["expected_isa"] != r.get("s_barrier"):
            b = dict(b, status=f"not analysed: the binary has {r.get('s_barrier')} s_barrier, the text implies {b['expected_isa']}",
                     barriers=[], races=[])
        r["barrier"] = b
        if any(x["verdict"] == "orders nothing" for x in b["barriers"]):
            r.setdefault("notes", []).append("barrier")
        if b["status"].startswith("divergent"):
            r.setdefault("notes", []).append("divbar")


def barrier_report(r, start=None, full=False):
    """the barrier check's lines for one row: every barrier with the hazard pairs it orders (grouped by region and kind:
    first-side lines -> second-side lines), its verdict, and each potential race; lines are the program file's when START
    (file, kernel.def line) is known, else relative to the kernel.def line"""
    b = r.get("barrier")
    if not b:
        return []
    off = start[1] if start else 0                        # relative line 0 is the kernel.def line itself
    L = lambda n: f"L{n + off}"
    head = f"barriers {r['kernel']}: {b['status']}" + (f" ({os.path.basename(start[0])})" if start else "")
    out = []
    if not full and not any(x["verdict"] == "orders nothing" for x in b["barriers"]) and not (
            b["status"].startswith(("not analysed", "divergent"))):
        return out                                        # a kernel the check could not model is always printed (fail closed)
    out.append(head + (f", {len(b['barriers'])} barriers, {b['collectives']} collectives" if b["status"] == "analysed" else ""))
    for x in b["barriers"]:
        g = collections.defaultdict(lambda: (set(), set(), [0]))
        for a, c, k, reg in x["pairs"]:
            e = g[(reg, k)]
            e[0].add(a)
            e[1].add(c)
            e[2][0] += 1
        desc = "; ".join(f"{reg} {k}: {' '.join(L(v) for v in sorted(e[0]))} -> {' '.join(L(v) for v in sorted(e[1]))} ({e[2][0]} pairs)"
                         for (reg, k), e in sorted(g.items()))
        out.append(f"  {L(x['line'])}: {x['verdict']}" + (f"  [{desc}]" if desc else ""))
    for a, c, k, reg, lab in (b["races"] if full else []):   # 0 of 10 sampled races real (11-25): detail only
        out.append(f"  potential race {k} on {reg}: {L(a)} / {L(c)} with no barrier between" + (f" ({lab})" if lab else ""))
    return out


def compile_programs(progs, out, jobs, lc=None, roots=None):
    """every gfx11 kernel root (matching ROOTS, a regex) of the program files compiled into OUT (code object + summary report;
    the details report for a kernel whose bank model has conflicted packets)
    with loom-compile LC (default: the machine's); returns the program texts and the roots that failed to compile standalone
    (named, never hidden)"""
    lc = lc or loom_compile()
    os.makedirs(out, exist_ok=True)
    pv = os.path.join(out, "loom-provenance")
    if os.path.exists(pv):
        os.remove(pv)                                 # a compile that dies half-way leaves no provenance (no SDK differential on it)
    texts, work = [], []
    for prog in progs:
        t = open(prog).read()
        texts.append(t)
        for k in sorted(set(re.findall(r'(?m)^kernel\.def (?:target\(@\w+\) )?export\("(\w+)"\)', t))):
            if not k.endswith("_rdna4") and (roots is None or re.search(roots, k)):
                work.append((prog, k))
    def one(pk):
        prog, k = pk
        r = None
        for rep in ("summary", "details"):            # details (per-packet rows, ~10x larger) only where the bank model conflicts: ldsbank
            r = subprocess.run([lc, prog, "--target=amdgpu:gfx1151", f"--root={k}", f"--output={out}/{k}.hsaco",
                                f"--compile-report={rep}", f"--compile-report-output={out}/{k}.json"], capture_output=True, text=True)
            if r.returncode or not bank(f"{out}/{k}.json")[0]:
                break
        return k, r.returncode, r.stderr
    failed = []
    with concurrent.futures.ThreadPoolExecutor(jobs) as ex:
        results = list(ex.map(one, work))
        for k, rc, err in results:
            if rc:
                failed.append((k, (err.strip().split("\n") or [""])[0][:160]))
    done = [k for k, rc, _ in results if not rc]
    objs = [f"{out}/{k}{x}" for k in done for x in OBJ_EXT if os.path.exists(f"{out}/{k}{x}")]
    json.dump(provenance(lc, progs, roots, objs), open(pv, "w"), indent=1)
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
    ap.add_argument("--source", action="append", default=[], help="a package directory or program file whose texts feed the "
                    "text-level rules (maskedv, barrier) for code objects compiled earlier (repeatable)")
    ap.add_argument("--barriers", action="store_true", help="print every barrier of every analysed kernel with the hazard pairs it "
                    "orders and the potential races (computed only then), not only the kernels with a barrier that orders nothing or "
                    "that the check could not model")
    ap.add_argument("--suggest", action="store_true", help="run `loom-compile-report suggest` (common/loom_suggest.py) on each kernel's "
                    "compile report: its findings as loom:<id> notes (never flags) and one 'loom KERNEL:' line per finding kind; a failed "
                    "run is a printed 'unknown (...)' and the table stays complete")
    ap.add_argument("--suggest-tool", help="with --suggest: the loom-compile-report (default: loom_suggest.tool())")
    ap.add_argument("--livereg", action="store_true", help="RGA's live VGPRs per instruction (rga -s bin --livereg): the 'live' "
                    "column peak/floor/over and a line per kernel whose loop floor exceeds the next tier's VGPR budget")
    ap.add_argument("--rga", help="with --livereg: the rga binary (default: RGA from the environment or the machine env file)")
    ap.add_argument("--loom-candidates", action="store_true", help="print Loom upstream CANDIDATES (C19; never flags or notes, no "
                    "exit code): per kernel the report's next tier vs whole-workgroup arithmetic (E533) and back-edge copy groups VOPD "
                    "could pair (E592); with --baseline, the SDK differential when both directories were compiled from identical "
                    "programs by two compilers (loom-provenance)")
    ap.add_argument("--flagged", action="store_true", help="print flagged kernels only")
    ap.add_argument("--strict", action="store_true", help="exit 1 when any kernel is flagged")
    a = ap.parse_args()
    sites, failed, texts = None, [], []
    progs = sorted(glob.glob(os.path.join(a.package, "program", "*.loom"))) if a.package else []
    progs += a.program
    if progs:
        if not a.out:
            sys.exit("kernel_lint: --package / --program need --out")
        texts, failed = compile_programs(progs, a.out, a.jobs, a.loom_compile, a.roots)
        sites = launch_sites(texts)
        a.paths = [a.out] + a.paths
    srcs = []
    for x in a.source:
        srcs += sorted(glob.glob(os.path.join(x, "program", "*.loom"))) if os.path.isdir(x) else [x]
    texts += [open(x).read() for x in srcs]
    starts = kernel_starts(progs + srcs)
    rga = None
    if a.livereg:
        rga = rga_tool(a.rga) or ""                       # "" : every kernel reports unknown (rga: not found), never a number
    stats, prompt = served_stats(a.profile) if a.profile else ({}, None)
    ms = collections.Counter({k: v[0] for k, v in stats.items()})
    files = []
    for p in a.paths:
        files += sorted(glob.glob(os.path.join(p, "*.hsaco")) + glob.glob(os.path.join(p, "*.co"))) if os.path.isdir(p) else [p]
    if not files:
        sys.exit("kernel_lint: no code objects")
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
        per_file = list(ex.map(lambda f: lint_file(f, rga), files))
    rows = [r for rs in per_file for r in rs]
    single = {id(rs[0]): f for f, rs in zip(files, per_file) if len(rs) == 1}    # a one-kernel object: its report is f's .json
    outl = outliers(rows)
    tiny(rows, stats)
    rounds(rows, texts)
    barriers(rows, kernel_texts(texts), func_texts(texts), a.roots, a.barriers)
    mv = {k: c for k, t in kernel_texts(texts).items() if (a.roots is None or re.search(a.roots, k)) for c in [masked_v(t)] if c}
    for r in rows:
        if r["kernel"] in mv:
            r.setdefault("notes", []).append("maskedv")         # E444: V read unconditionally under a score-only mask
    base_rows = {}
    if a.baseline:
        for r in rows:
            b = os.path.join(a.baseline, r["kernel"] + ".hsaco")
            if os.path.exists(b):
                base_rows[r["kernel"]] = lint(b)
                r["flags"] += ["regress:" + x for x in regressions(r, base_rows[r["kernel"]])]
    cand_lines, cand_skip = [], collections.Counter()
    if a.loom_candidates:
        def cands(f, name=None):
            js = f[:-6] + ".json"
            rep = json.load(open(js)) if f.endswith(".hsaco") and os.path.exists(js) else None
            return loom_candidates(name or os.path.basename(f)[:-6], *disasm(f), rep)
        with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
            got = dict(ex.map(lambda r: (id(r), cands(single[id(r)], r["kernel"]) if id(r) in single
                                         else ([], ["a kernel of a multi-kernel object (no report)"])), rows))
        for r in rows:
            cs, sk = got[id(r)]
            r["loomcand"] = cs
            cand_lines += [c["line"] for c in cs]
            cand_skip.update(sk)
        if a.baseline:
            bfiles = sorted(glob.glob(os.path.join(a.baseline, "*.hsaco")))
            with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
                for f, (cs, _) in zip(bfiles, ex.map(cands, bfiles)):
                    k = os.path.basename(f)[:-6]
                    base_rows.setdefault(k, lint(f))["loomcand"] = cs
            here = a.out if progs else (a.paths[0] if a.paths and os.path.isdir(a.paths[0]) else None)
            cand_lines += sdk_diff({r["kernel"]: r for r in rows}, base_rows, read_provenance(here), read_provenance(a.baseline))
    loom_lines = []
    if a.suggest:
        import loom_suggest
        reps = {}
        for f in files:
            if f.endswith(".hsaco"):
                reps[os.path.basename(f)[:-6]] = f[:-6] + ".json"
        st = loom_suggest.tool(a.suggest_tool)
        sug = loom_suggest.survey({r["kernel"]: reps.get(r["kernel"], "") for r in rows}, st, suggest_occ, a.jobs)
        loom_sum = loom_suggest.summary(sug, st)
        for r in rows:
            x = sug[r["kernel"]]
            r.setdefault("notes", []).extend(n for n in x["notes"] if n not in r["notes"])
            loom_lines += [f"loom {r['kernel']}: {ln}" for ln in x["lines"] if ln != "no findings"]
    rows.sort(key=lambda r: (-bool(r["flags"]), -ms.get(r["kernel"], 0), -len(r["flags"]), -(sites or {}).get(r["kernel"], 0), r["kernel"]))
    print(f"{'kernel':42s} {'where':>6s} {'loop':>5s} {'comm':>5s} {'mm':>4s} {'other/op':>8s} {'sel':>4s} {'pk':>3s} {'bfe':>4s} {'scr':>4s} {'mov':>4s} "
          f"{'run':>4s} {'vgpr':>4s} {'w/S':>8s} {'next':>16s} {'live':>11s} {'pf':>5s} {'sites':>5s} {'ldsW':>5s} {'cov':>4s} {'fly':>3s} {'bank':>5s} {'ms':>8s}  flags")
    for r in rows:
        if a.flagged and not r["flags"]:
            continue
        f = lambda v: "-" if v is None else str(v)
        s = "-" if sites is None else str(sites.get(r["kernel"], 0))
        print(f"{r['kernel']:42s} {r['where']:>6s} {r['loop']:5d} {r['common']:5d} {r['wmma'] or r['dot']:4d} {f(r['other_per_op']):>8s} {r['select']:4d} {r['pk']:3d} "
              f"{r['bfe']:4d} {r['scratch']:4d} {r['moves']:4d} {r['run']:4d} {f(r['vgpr']):>4s} {tier_text(r['occ'] if r.get('waves') is not None else None):>8s} {next_text(r):>16s} {live_text(r):>11s} {r['pf']:>5s} {s:>5s} "
              f"{f(r['lds_waits_per_op']):>5s} {f(r['cover_min']):>4s} {r['in_flight']:3d} {f(r.get('bank_extra')):>5s} "
              f"{('%.1f' % ms[r['kernel']]) if r['kernel'] in ms else '-':>8s}  "
              f"{' '.join(r['flags'])}{('  [' + ' '.join(r['notes']) + ']') if r.get('notes') else ''}")
    if a.suggest:
        print(loom_suggest.ADVICE + ":")
        print("\n".join(sorted(loom_lines)))
        print(loom_sum)
    for r in sorted(rows, key=lambda r: r["kernel"]):
        x = r.get("live")
        if x and "error" in x:
            print(f"livereg {r['kernel']}: unknown ({x['error']})")
        elif x and x.get("over"):
            print(f"livereg {r['kernel']}: peak {x['peak']} live VGPRs; hot loop peak {x['loop_peak']} at +{x['at'][0]} {x['at'][1]}, floor "
                  f"{x['floor']}; {x['over']} of {x['loop'][1] - x['loop'][0] + 1} loop instructions above the {r['live_budget']}-VGPR budget "
                  f"of {tier_text(r['occ'], 'next')} waves/SIMD" + (" (the floor too: the carried values alone exceed it)" if x["floor"] > r["live_budget"] else ""))
    for k, (v, med, n) in sorted(outl.items()):
        print(f"outlier {k}: {v} non-MMA instructions per MMA vs a median of {med} over {n} peers")
    for r in rows:
        if "ldsbank" in r.get("notes", []):
            x, q = r["bank_loop"]
            print(f"ldsbank {r['kernel']}: {x} of {q} modelled LDS rounds in the hot loop are bank-conflict rounds ({100 * x / q:.1f} %; "
                  f"read the report's bank_service_groups for the access)")
    for r in sorted(rows, key=lambda r: r["kernel"]):
        for line in barrier_report(r, starts.get(r["kernel"]), a.barriers):
            print(line)
    for line in lesson_lines(rows):
        print(line)
    if a.loom_candidates:
        print(CAND_HEAD)
        for line in cand_lines:
            print(line)
        kinds = collections.Counter(c["kind"] for r in rows for c in r.get("loomcand", []))
        print("loomcand: " + (", ".join(f"{v} {k}" for k, v in sorted(kinds.items())) or "none") + "; not compared: "
              + (", ".join(f"{v} {k}" for k, v in sorted(cand_skip.items())) or "none"))
    for k, c in sorted(mv.items()):
        print(f"maskedv {k}: {len(c)} key-valid bits ({' '.join(c[:3])}{' ...' if len(c) > 3 else ''}) mask only the score: no value "
              f"outside the score chain is zeroed and no compared key is clamped (E444: V of a masked key times p = +0)")
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
