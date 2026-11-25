"""KV8 prefill flash2: keep the K/V loads in flight (HRX-like LEDGER E109/E110).

Runs on the generated program after gen_llm.kv8_readers, on every gfx11 prefill flash kernel whose loop carries the next
tile's K (prefetch_split + FLASH2_PF11_CARRY: g_attention_flash2, g_attention_flash2_partial). Two switches, both off by
default (output unchanged):

FLASH2_KV8_KCARRY=1  the loop carries K(kt+1) as its raw codes + f32 scale instead of the dequantized f16 vector, and the
    dequant runs at the top of the next tile, where its K store consumes it. With the dequantized carry, the dequant must
    finish inside the iteration that issues the loads and Loom places it right after them: K(kt+1)'s codes and scale were
    waited 2-26 instructions after issue (f16: ~355), the whole prefetch lost. Same arithmetic, moved: exact.
FLASH2_KV8_VLATE=1   V(kt)'s dequant moves to its use (the V^T staging after the S MMAs) and is pinned there by a data
    dependency on the last S MMA result: the codes and the scale pass through a select on `bits(x) != 0x7FBADBAD` of one
    S element (a signalling-NaN pattern no arithmetic produces: always true, NaN included; the first version used
    `oeq x, x`, false on the NaN S of dead padding columns, and zeroed live V: E123). Without the pin Loom hoists the moved dequant back above barrier 1 (E110: source order
    is not kept), so the V loads are waited before the S MMAs.
FLASH2_KV8_VLATE=2   the same move, kept after the S MMAs by `scf.schedule.fence` (the SDK's documented issue-order boundary:
    loom/docs/src/guide/functions-and-control.md, workflows/tune-loop-schedules.md; no instruction, no memory wait) instead
    of the data dependency: no value of S can change V, and no extra instructions (HRX-like E144; ROCm/hrx-system#3491/#3500).

Every anchor must match exactly once per kernel and at least one kernel must match, or the build stops (a regex pass
that silently skips a new shape is a contract break: AGENTS.md section 4)."""
import os
import re
import sys

# kernel -> (loop result suffix of the carried K/O, the variants of the V staging / K select: "" = the loop body, "_pt" = the
# peeled last tile)
KERNELS = {"g_attention_flash2": ("_hot", ("", "_pt")), "g_attention_flash2_partial": ("_hot", ("", "_pt"))}


def _one(lines, pred, what, name):
    hits = [i for i, ln in enumerate(lines) if pred(ln)]
    if len(hits) != 1:
        sys.exit(f"kv8_late: {name}: expected one {what}, found {len(hits)}")
    return hits[0]


def _vlate(L, name, fence=False, variants=("", "_pt")):
    for X in variants:
        names = [f"%vh_v{X}_k8f = ", f"%vh_v{X}_k8sh = ", f"%vh_v{X}_k8ss = ", f"%vh_v{X} = vector.mulf"]
        idx = [_one(L, lambda ln, n=n: ln.strip().startswith(n), n, name) for n in names]
        ind = re.match(r"\s*", L[idx[0]]).group(0)
        moved = [L[i] for i in idx]
        for i in sorted(idx, reverse=True):
            del L[i]
        sel = _one(L, lambda ln: ln.strip().startswith(f"%pfv_s{X} = scf.select"), f"%pfv_s{X} select", name)
        _one(L, lambda ln: ln.strip().startswith(f"%s1_7{X} = vector.mma"), f"%s1_7{X}", name)
        # The condition must be true for EVERY value S can hold, NaN included: a partial block's dead query columns (padding
        # whose Q rows hold stale workspace bits in serving) give NaN S, and the lane whose S is NaN stages V for a live key,
        # so `oeq x, x` zeroed live V (E123: spec gate FAIL; red: flash_bench --dead 12, live token 37 changed). Arithmetic
        # never produces this signalling-NaN pattern (results are quiet NaNs), so `bits != 0x7FBADBAD` always holds, and it
        # still depends on the MMA result, which keeps the dequant after the S MMAs.
        pin = [f"{ind}%vd{X}_x = vector.extract %s1_7{X}[7] : vector<8xf32> -> f32",
               f"{ind}%vd{X}_b = scalar.bitcast %vd{X}_x : f32 to i32",
               f"{ind}%vd{X}_k = scalar.constant 2142952365 : i32",
               f"{ind}%vd{X}_c = scalar.cmpi ne, %vd{X}_b, %vd{X}_k : i32",
               f"{ind}%vd{X}_z = vector.splati 0 : vector<32xi8>",
               f"{ind}%vd{X}_q = scf.select %vd{X}_c, %vh_v{X}_k8q, %vd{X}_z : vector<32xi8>",
               f"{ind}%vd{X}_zf = scalar.constant 0.0 : f32",
               f"{ind}%vd{X}_s = scf.select %vd{X}_c, %vh_v{X}_k8s, %vd{X}_zf : f32"]
        if fence:   # mode 2: the fence alone keeps the moved dequant (and the V loads' wait) after the S MMAs
            L[sel:sel] = [f"{ind}scf.schedule.fence"] + moved
            continue
        new = []
        for ln in moved:
            ln2 = ln.replace(f"%vh_v{X}_k8q :", f"%vd{X}_q :").replace(f"scalar.fptrunc %vh_v{X}_k8s :", f"scalar.fptrunc %vd{X}_s :")
            new.append(ln2)
        if new[0] == moved[0] or new[1] == moved[1]:
            sys.exit(f"kv8_late: {name}: V{X} dequant did not take the pinned codes/scale")
        L[sel:sel] = pin + new


def _kcarry(L, name, res="_hot", variants=("", "_pt")):
    h = _one(L, lambda ln: f", %pfk_f{res}, %pfo_f{res} = scf.for %kt" in ln, "carry loop header", name)
    ln = L[h]
    for a, b in ((f", %pfk_f{res}, %pfo_f{res} =", f", %pfk_f{res}, %pfks_f{res}, %pfo_f{res} ="),
                 ("%pfk = %kh_p0 : vector<32xf16>", "%pfk = %kh_p0_k8q : vector<32xi8>, %pfks = %kh_p0_k8s : f32")):
        if ln.count(a) != 1:
            sys.exit(f"kv8_late: {name}: loop header lacks {a!r}")
        ln = ln.replace(a, b)
    tail = "vector<32xf16>, offset) {"
    if not ln.rstrip().endswith(tail):
        sys.exit(f"kv8_late: {name}: loop header result types end {ln.rstrip()[-60:]!r}")
    L[h] = ln[:ln.rstrip().rindex(tail)] + "vector<32xi8>, f32, offset) {"
    y = _one(L, lambda ln: ln.strip().startswith("scf.yield %m_new, %l_new,") and "%kh_n, %k_offset_n :" in ln, "carry yield", name)
    ln = L[y].replace("%kh_n, %k_offset_n :", "%kh_n_k8q, %kh_n_k8s, %k_offset_n :")
    if not ln.rstrip().endswith("vector<32xf16>, offset"):
        sys.exit(f"kv8_late: {name}: yield types end {ln.rstrip()[-60:]!r}")
    L[y] = ln[:ln.rstrip().rindex("vector<32xf16>, offset")] + "vector<32xi8>, f32, offset"
    sels = {"": ("%pfk_s = scf.select %key_valid, %pfk,", "%pfk", "%pfks", "%pfk_dq", "kc"),
            "_pt": (f"%pfk_s_pt = scf.select %key_valid_pt, %pfk_f{res},", f"%pfk_f{res}", f"%pfks_f{res}", "%pfk_dq_pt", "kcp")}
    for sel, q, sc, out, t in (sels[X] for X in variants):
        i = _one(L, lambda ln, sel=sel: ln.strip().startswith(sel), sel, name)
        ind = re.match(r"\s*", L[i]).group(0)
        L[i] = L[i].replace(f", {q},", f", {out},", 1)
        L[i:i] = [f"{ind}%{t}_f = vector.sitofp {q} : vector<32xi8> to vector<32xf16>",
                  f"{ind}%{t}_sh = scalar.fptrunc {sc} : f32 to f16",
                  f"{ind}%{t}_ss = vector.splat %{t}_sh : vector<32xf16>",
                  f"{ind}{out} = vector.mulf %{t}_f, %{t}_ss : vector<32xf16>"]


def apply(text):
    kc = os.environ.get("FLASH2_KV8_KCARRY", "0") == "1"
    vmode = os.environ.get("FLASH2_KV8_VLATE", "0")
    if vmode not in ("0", "1", "2"):
        sys.exit(f"kv8_late: FLASH2_KV8_VLATE={vmode!r}: 0, 1 (data-dependency pin) or 2 (scf.schedule.fence)")
    vl = vmode != "0"
    if not (kc or vl):
        return text
    done = []
    for name, (res, variants) in KERNELS.items():
        m = re.search(r'^kernel\.def (?:target\(@rdna3\) )?export\("' + name + r'"\).*?\n\} launch\(.*?\n\}\n', text, re.S | re.M)
        if not m:
            continue
        L = m.group(0).split("\n")
        if vl:
            _vlate(L, name, fence=vmode == "2", variants=variants)
        if kc:
            _kcarry(L, name, res, variants)
        text = text.replace(m.group(0), "\n".join(L), 1)
        done.append(name)
    if not done:
        sys.exit("kv8_late: FLASH2_KV8_KCARRY/VLATE set but no prefill flash kernel found")
    print(f"KV8 late: {'kcarry ' if kc else ''}{('vlate-fence ' if vmode == '2' else 'vlate ') if vl else ''}on {', '.join(done)}", file=sys.stderr)
    return text
