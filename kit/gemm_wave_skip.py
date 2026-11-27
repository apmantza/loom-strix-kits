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
Fails closed (HRX-like E690 review: the region check was a deny-list and accepted an LDS fragment.store, a global atomic and lhs
rows off the wave): each anchor must match once in every rewritten kernel; every region line must be one of the five op forms the
served kernels hold there (ALLOW: index.constant, fragment.load<lhs> from %x_stage_view at %wave_row / %wave_row1-3, vector.load from
%w_wave_view, fragment<rhs>, vector.mma), the two views pinned to their workgroup allocas; the lhs loads must use all four
%wave_row* and those must be %wave_row + 0 / 16 / 32 / 48, the rows the predicate reads (row_origin + wave_row + lane, + 32), with
the staged x rows pinned to the tile's rows (x_row = row_origin + x_stage_row, + 128); no value defined in the region may be used
after it except the accumulators the loop yields. gfx12 twins (target(@rdna4), _rdna4) are skipped by name; any other
g_gemm_f16_*m256* kernel stops the build. Every guard reds a case of test_gemm_wave_skip.py when neutered (HRX-like E690).
Not pinned (residuals the E704 verify accepted): the wave geometry is held only by the %wave_row assume (0-192), not by the
workgroup size, the subgroup count or wave32; the predicate's column constant (%c0, scale column 0) is not checked against the tile
skip's. A generator change to those would pass this gate; the kernel test's wave_skip case is the check that remains."""
import re

KERNEL = re.compile(r'^kernel\.def (?:target\(@(\w+)\) )?export\("(g_gemm_f16_\w*m256\w*)"\).*?\n\} launch\(.*?\n\}\n', re.S | re.M)
REWRITE = re.compile(r"g_gemm_f16_(acc|store|swiglu)_m256(_f16o)?")
ALLOW = (re.compile(r"    %\w+ = index\.constant \d+ : index"),
         re.compile(r"    %\w+ = vector\.fragment\.load<lhs> %x_stage_view\[(?P<row>%wave_row[123]?), %\w+\] shape \[%c16, %c16\] : "
                    r"view<256x72xf16> -> vector<16xf16>"),
         re.compile(r"    %\w+ = vector\.load %w_wave_view\[%\w+, %\w+\] : view<\d+x72xf16> -> vector<16xf16>"),
         re.compile(r"    %\w+ = vector\.fragment<rhs> %\w+ shape \[%c16, %c16\] : vector<16xf16>"),
         re.compile(r"    %\w+ = vector\.mma %\w+, %\w+, %\w+ : vector<16xf16>, vector<16xf16>, vector<8xf32>"))
# the rows the region's lhs loads read are the rows the predicate reads: stage row r holds tile row row_origin + r (r < 128 from x_row,
# r + 128 from x_row1), and a wave's four lhs fragments read stage rows wave_row + 0 / 16 / 32 / 48 (+ 0-15 each)
ANCHORS = ("  %lane = kernel.subgroup.lane.id : index\n", "  %wave_row = index.assume %wave_row_value [range(%wave_row_value, 0, 192)] : index\n",
           "  %wave_row1_c = index.constant 16 : index\n", "  %wave_row2_c = index.constant 32 : index\n", "  %wave_row3_c = index.constant 48 : index\n",
           "  %wave_row1 = index.add %wave_row, %wave_row1_c : index\n", "  %wave_row2 = index.add %wave_row, %wave_row2_c : index\n",
           "  %wave_row3 = index.add %wave_row, %wave_row3_c : index\n",
           "  %s_view = buffer.view %scales[%origin] : buffer -> view<[%rows]x2xf32>\n",
           "  %row_origin = index.mul %row_group, %c256 : index\n",
           "  %x_stage = buffer.alloca<workgroup> align(16) %x_bytes : buffer\n",
           "  %x_stage_view = buffer.view %x_stage[%origin] : buffer -> view<256x72xf16>\n",
           "  %w_stage = buffer.alloca<workgroup> align(16) %w_bytes : buffer\n",
           "  %x_stage_row = index.div %workitem, %c2 : index\n", "  %x_stage_row1 = index.add %x_stage_row, %c128 : index\n",
           "  %x_row_value = index.add %row_origin, %x_stage_row : index\n", "  %x_row = index.assume %x_row_value [lt(%x_row_value, %rows)] : index\n",
           "  %x_row1_value = index.add %x_row, %c128 : index\n", "  %x_row1 = index.assume %x_row1_value [lt(%x_row1_value, %rows)] : index\n",
           "    %x_rawp = vector.load %x_view[%x_row, %x_colp] : view<[%rows]x[%kx]xf16> -> vector<32xf16>\n",
           "    %x_rawp1 = vector.load %x_view[%x_row1, %x_colp] : view<[%rows]x[%kx]xf16> -> vector<32xf16>\n",
           "    vector.store %cr_x, %x_stage_view[%x_stage_row, %x_stage_col] : vector<32xf16>, view<256x72xf16>\n",
           "    vector.store %cr_x1, %x_stage_view[%x_stage_row1, %x_stage_col] : vector<32xf16>, view<256x72xf16>\n")
# the in-loop x prefetch reads the same rows, in one of the two forms the generator emits (by row, or GEMM256_FLATADDR's flat lanes)
PREFETCH = (("      %xv = vector.load %x_view[%x_row, %x_coln] : view<[%rows]x[%kx]xf16> -> vector<32xf16>\n",
             "      %xv1 = vector.load %x_view[%x_row1, %x_coln] : view<[%rows]x[%kx]xf16> -> vector<32xf16>\n"),
            ("  %x_lane0_v = index.madd %x_row, %kx, %x_stage_col : index\n", "  %x_lane1_v = index.madd %x_row1, %kx, %x_stage_col : index\n",
             "  %x_lane0 = index.assume %x_lane0_v [range(%x_lane0_v, 0, 1073741823), lt(%x_lane0_v, %x_flat_n)] : index\n",
             "  %x_lane1 = index.assume %x_lane1_v [range(%x_lane1_v, 0, 1073741823), lt(%x_lane1_v, %x_flat_n)] : index\n",
             "      %x_sv = buffer.view %x[%x_kob] : buffer -> view<[%x_flat_np]xf16>\n",
             "      %xv = vector.load %x_sv[%x_lane0] : view<[%x_flat_np]xf16> -> vector<32xf16>\n",
             "      %xv1 = vector.load %x_sv[%x_lane1] : view<[%x_flat_np]xf16> -> vector<32xf16>\n"))
W_VIEW = re.compile(r"^  %w_wave_view = buffer\.view %w_stage\[%wb_off\] : buffer -> view<\d+x72xf16>$", re.M)
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
    for need in ANCHORS:
        if text.count(need) != 1:
            _fail(name, f"anchor {need.strip()!r} matches {text.count(need)} times (want 1: 64-row waves of a 256-row tile)")
    if sum(all(text.count(n) == 1 for n in form) for form in PREFETCH) != 1:
        _fail(name, "the in-loop x prefetch is in neither known form (by row, or flat lanes from x_row / x_row1)")
    if len(W_VIEW.findall(text)) != 1:
        _fail(name, "anchor %w_wave_view = buffer.view %w_stage[%wb_off] not found once")
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
    rows_read = set()
    for ln in region:
        hit = next((m for m in (a.fullmatch(ln) for a in ALLOW) if m), None)
        if not hit:
            _fail(name, f"region line not pure fragment / MMA work: {ln.strip()[:80]!r}")
        rows_read |= {hit.group("row")} if "row" in hit.re.groupindex else set()
    if rows_read != {"%wave_row", "%wave_row1", "%wave_row2", "%wave_row3"}:
        _fail(name, f"the region's lhs loads read rows {sorted(rows_read)}, not the wave's 64 (%wave_row, %wave_row1-3)")
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
