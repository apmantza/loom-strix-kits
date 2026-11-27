"""GEMM_WAVE_SKIP=1 (HRX-like E635 / shortpf O4): in the 256-row f16 prefill GEMMs, a wave whose 64 rows are all padding skips its
MMAs (and their LDS fragment reads). Exact by construction for every live row; meant for the epoch's last, part-live row tile.

Why: a prefill epoch runs at its ladder shape (a 1,141-token prompt at prefill_capacity 4096 is a 2048-row epoch), and the live-row
skip (gen_gemm GEMM_SKIP_PADDING) drops only whole 256-row tiles whose rows are all padding. The last tile of 1,141 rows holds 117
live rows, so its 256 rows are computed in full: 1,280 GEMM rows for 1,141 tokens. Upstream's 1024 + 128 split computes 1,152 (E635,
static-1127/shortpf/DESIGN.md O4). In the m256 kernels each wave owns 64 rows (waves 4 x 2), so a wave can tell from the rows' scales
(g_row_scale_f16_live writes 0 for a padding row) that it has no live row and leave its WMMA work to the other waves: 1,152 rows of
MMA work for 1,141 tokens, in the same launch, with no host-side change.

What changes, per kernel (gfx11 g_gemm_f16_{store,acc,swiglu}_m256 and _m256_f16o, the kernels GEMM256_FLATADDR rewrites):
- before the K loop, each wave reduces |scale| over its 64 rows (two loads per lane, kernel.subgroup.reduce<maxnumf>: the result
  is uniform across the wave) into %wsk_live, with the tile skip's predicate (scalar.cmpf one: a NaN scale counts as padding there too);
- in the K loop, the LDS fragment reads and the MMAs (a contiguous region after the staging barrier, before the closing barrier) run
  under `scf.if %wsk_live`; the else branch yields the loop-carried accumulators unchanged. Staging, prefetch loads and both barriers
  stay outside the branch, so every wave reaches every barrier.
Exactness: a live wave runs the same ops on the same operands in the same order. A skipped wave keeps its accumulators at the zero
init. Its rows are padding, and padding rows' outputs are never read: the same invariant the existing whole-tile skip already relies on
(gen_llm live_row_skip). For acc, padding rows of y keep their old value; today they get y + acc x 0, which is y unless the stale
activation rows held inf / NaN.
Fails closed: each anchor must match as expected in every rewritten kernel, the region must hold no barrier, store, nested scf op or
global view, and no value defined in it may be used after it except the accumulators the loop yields. gfx12 twins (target(@rdna4),
_rdna4) are skipped by name; any other g_gemm_f16_*m256* kernel stops the build."""
import re

KERNEL = re.compile(r'^kernel\.def (?:target\(@(\w+)\) )?export\("(g_gemm_f16_\w*m256\w*)"\).*?\n\} launch\(.*?\n\}\n', re.S | re.M)
REWRITE = re.compile(r"g_gemm_f16_(acc|store|swiglu)_m256(_f16o)?")
FORBID = ("kernel.barrier", "vector.store", "view.store", "scf.", "buffer.view", "kernel.workgroup")
LIVE = """  %wsk_c32 = index.constant 32 : index
  %wsk_r0 = index.add %row_origin, %wave_row : index
  %wsk_ra_v = index.add %wsk_r0, %lane : index
  %wsk_rb_v = index.add %wsk_ra_v, %wsk_c32 : index
  %wsk_ra = index.assume %wsk_ra_v [lt(%wsk_ra_v, %rows)] : index
  %wsk_rb = index.assume %wsk_rb_v [lt(%wsk_rb_v, %rows)] : index
  %wsk_sa = view.load %s_view[%wsk_ra, %c0] : view<[%rows]x2xf32> -> f32
  %wsk_sb = view.load %s_view[%wsk_rb, %c0] : view<[%rows]x2xf32> -> f32
  %wsk_aa = scalar.absf %wsk_sa : f32
  %wsk_ab = scalar.absf %wsk_sb : f32
  %wsk_m = scalar.maxnumf %wsk_aa, %wsk_ab : f32
  %wsk_any = kernel.subgroup.reduce<maxnumf> %wsk_m : f32
  %wsk_zero = scalar.constant 0.0 : f32
  %wsk_live = scalar.cmpf one, %wsk_any, %wsk_zero : f32
"""


def _fail(name, what):
    raise SystemExit(f"GEMM_WAVE_SKIP: {name}: {what}; the generator's text changed (extend gemm_wave_skip.py)")


def wave_skip(text, name="kernel"):
    """rewrite one gfx11 g_gemm_f16_*_m256 kernel's text (served form: after f16o11 and GEMM256_FLATADDR)"""
    for need in ("  %lane = kernel.subgroup.lane.id : index\n", "  %wave_row = index.assume %wave_row_value [range(%wave_row_value, 0, 192)] : index\n",
                 "  %wave_row3_c = index.constant 48 : index\n", "  %s_view = buffer.view %scales[%origin] : buffer -> view<[%rows]x2xf32>\n",
                 "  %row_origin = index.mul %row_group, %c256 : index\n"):
        if text.count(need) != 1:
            _fail(name, f"anchor {need.strip()!r} matches {text.count(need)} times (want 1: 64-row waves of a 256-row tile)")
    if "%wsk_" in text:
        _fail(name, "already rewritten")
    lines = text.split("\n")
    heads = [i for i, ln in enumerate(lines) if re.match(r"^  %\S.* = scf\.for %step = \[%c0 to %k_steps step %c1\]\(", ln)]
    if len(heads) != 1:
        _fail(name, f"{len(heads)} K loops (want 1)")
    h = heads[0]
    m = re.match(r"^  (?P<res>.*?) = scf\.for %step = \[%c0 to %k_steps step %c1\]\((?P<args>.*)\) -> \((?P<tys>.*)\) \{$", lines[h])
    if not m:
        _fail(name, "K loop header not in the expected form")
    args = [a.strip() for a in m.group("args").split(", %")]
    args = [("%" + a if not a.startswith("%") else a) for a in args]
    iters = []
    for a in args:
        am = re.fullmatch(r"(%\w+) = (%\w+) : (\S+)", a)
        if not am:
            _fail(name, f"loop-carried value {a!r} not in the expected form")
        iters.append(am.groups())
    # the loop body ends at the first line that closes at the header's indent
    end = next((j for j in range(h + 1, len(lines)) if lines[j] == "  }"), None)
    if end is None:
        _fail(name, "K loop end not found")
    body = list(range(h + 1, end))
    ys = [j for j in body if lines[j].startswith("    scf.yield ")]
    if len(ys) != 1 or ys[0] != end - 1:
        _fail(name, "the K loop's yield is not its last line")
    first = next((j for j in body if "= vector.fragment.load<lhs>" in lines[j]), None)
    last = max((j for j in body if "= vector.mma " in lines[j]), default=None)
    if first is None or last is None or not first < last:
        _fail(name, "no fragment-load / MMA region in the K loop")
    region = lines[first:last + 1]
    if not lines[last + 1].startswith("    kernel.barrier<workgroup>") or not any(lines[j].startswith("    kernel.barrier<workgroup>") for j in range(h + 1, first)):
        _fail(name, "the MMA region is not between the staging barrier and the closing barrier")
    for ln in region:
        if any(f in ln for f in FORBID) or not ln.startswith("    "):
            _fail(name, f"region line not pure fragment / MMA work: {ln.strip()[:80]!r}")
    if sum("= vector.mma " in ln for ln in region) != text.count("= vector.mma "):
        _fail(name, "an MMA outside the region")
    defined = set(re.findall(r"^\s*(%[\w]+(?:\s*,\s*%[\w]+)*) = ", "\n".join(region), re.M))
    defs = {d.strip() for g in defined for d in g.split(",")}
    after = "\n".join(lines[last + 1:end - 1]) + "\n" + "\n".join(lines[end:])
    for d in defs:
        if re.search(re.escape(d) + r"\b", after):
            _fail(name, f"{d} defined in the region is used after it")
    yv = [v.strip() for v in lines[end - 1].strip()[len("scf.yield "):].split(" : ")[0].split(",")]
    if len(yv) != len(iters):
        _fail(name, f"yield has {len(yv)} values for {len(iters)} loop-carried values")
    acc = [(k, v) for k, v in enumerate(yv) if v in defs]
    for k, v in acc:
        if iters[k][1] != "%init" or iters[k][2] != "vector<8xf32>":
            _fail(name, f"yielded region value {v} is not an accumulator (carried {iters[k]})")
    if len(acc) != sum(1 for it in iters if it[1] == "%init"):
        _fail(name, "an accumulator is not produced by the region")
    outs = [f"%wsk_o{k}" for k, _ in acc]
    tys = ", ".join("vector<8xf32>" for _ in acc)
    wrapped = ([f"    {', '.join(outs)} = scf.if %wsk_live -> ({tys}) {{"] + ["  " + ln for ln in region]
               + [f"      scf.yield {', '.join(v for _, v in acc)} : {tys}", "    } else {",
                  f"      scf.yield {', '.join(iters[k][0] for k, _ in acc)} : {tys}", "    }"])
    new_y = list(yv)
    for (k, _), o in zip(acc, outs):
        new_y[k] = o
    ytail = lines[end - 1].strip()[len("scf.yield "):].split(" : ", 1)[1]
    lines[end - 1] = "    scf.yield " + ", ".join(new_y) + " : " + ytail
    lines[first:last + 1] = wrapped
    # the wave's liveness, once, before the K loop (after the accumulators' init, which every rewritten kernel defines there)
    ini = [i for i, ln in enumerate(lines) if ln.startswith("  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>")]
    if len(ini) != 1 or not ini[0] < h:
        _fail(name, "accumulator init anchor not found once before the K loop")
    lines[ini[0] + 1:ini[0] + 1] = LIVE.rstrip("\n").split("\n")
    return "\n".join(lines)


def apply(text):
    """rewrite every served gfx11 256-row f16 GEMM in a program; returns (text, names rewritten). Known-good forms only (above)."""
    done, out, last = [], [], 0
    for m in KERNEL.finditer(text):
        target, name = m.group(1), m.group(2)
        if target == "rdna4" and name.endswith("_rdna4"):
            continue   # gfx12 twin: not served on gfx1151
        if target not in (None, "rdna3") or not REWRITE.fullmatch(name):
            raise SystemExit(f"GEMM_WAVE_SKIP: {name} (target {target}) is neither rewritten nor a known skip: extend apply() or skip it by name")
        out += [text[last:m.start()], wave_skip(m.group(0), name)]; last = m.end(); done.append(name)
    if not done:
        raise SystemExit("GEMM_WAVE_SKIP: no g_gemm_f16_*_m256 kernel in the program")
    return "".join(out) + text[last:], done
