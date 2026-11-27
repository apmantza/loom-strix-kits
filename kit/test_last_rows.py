"""Static tests of LAST_ROWS=1 (HRX-like static-1127 SWEEP A1, kit/last_rows.py): the program's last layer masks the row scales of its
q, o, gate/up and down GEMMs to the rows the head reads, with the K / V projections launched before the first mask. Recurrence each
case prevents:
- unset (LAST_Q=1 only, the A/B's reference): the whole package program is byte-identical (a switch that leaks into the default
  program changes the reference package);
- set: four masks, all in @llm_advance's last layer, each right before the GEMM block it serves and on that block's scale buffer
  (a mask on the wrong buffer saves nothing or zeroes a buffer another reader needs);
- set: K and V launch before the first mask and the q GEMM after it (a K / V GEMM after the %s_h mask stores zero K / V rows of
  every non-head token into the arena: every later token's attention reads them; inexact everywhere, seen by EXACT only on a
  multi-chunk request);
- set: the program is exactly the unset program plus the masks, the kernel and the q region's move: an independent inverse gives
  the unset program back byte for byte (a rewrite that touched the verify program or another launch would not invert);
- fail closed (rules section 4): an unknown launch after g_qk_rope_store (a DFlash tap), a K block reading the q projection, a
  statement between V and g_qk_rope_store, a %s_h reader between the attention and the FFN norm, a q GEMM on another scale buffer,
  and a half-converted layer stop the rewrite; a second application returns the text unchanged (idempotent).
usage: python3 test_last_rows.py (exit 1 on a failure)."""
import os, re, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FAILS = []
MASK_RE = re.compile(r"(?m)^  scf\.if %mm \{\n    kernel\.launch @g_scale_head_rows\[%tp\]\(%tp, %t, %spans, %metadata, %info, (%s_[hof])\) : [^\n]*\n  \}\n")


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))
    if not ok:
        FAILS.append(name)


def program(out, **kw):
    """The served program as the kit builds it: common/build.sh (kit.env's recipe + EXTRA) into the package dir `out`
    (rules section 2, E230: test the generator as the served program emits it)."""
    env = dict(os.environ)
    for k in ("LAST_Q", "LAST_ROWS", "LAST_OUT", "EXTRA", "LINT_BASE"):
        env.pop(k, None)
    extra = " ".join(f"{k}={v}" for k, v in kw.items() if v is not None)
    if extra:
        env["EXTRA"] = extra
    r = subprocess.run(["bash", os.path.join(HERE, "..", "common", "build.sh"), "qwen38-27b", out], capture_output=True, text=True, env=env)
    log = open(out + ".log").read() if os.path.exists(out + ".log") else ""
    f = os.path.join(out, "program", "llm.loom")
    text = open(f).read() if r.returncode == 0 and os.path.exists(f) else ""
    return r.returncode, text, log + r.stdout + r.stderr


def segment(text, name):
    s = text.index(f"@{name}()")
    e = text.find("\ncommand.def ", s)
    return text[s:e if e > 0 else len(text)]


def last_layer(seg):
    m = list(re.finditer(r"(?m)^  // ---- layer (\d+)[^\n]*$", seg))[-1]
    return int(m.group(1)), seg[m.start():seg.index("\n  %head = command.parameter", m.start())]


def first(block, pat):
    m = re.search(pat, block)
    return m.start() if m else -1


def inverse(text):
    """written apart from the pass: drop the kernel and the four mask statements; move the text from the first mask's place to
    g_qk_rope_store (the q region) back before the K block's `  scf.if %mm {` (the statement that reads %wk_<L>)"""
    t = re.sub(r'(?ms)\n\n^// LAST_ROWS \(HRX-like static-1127 SWEEP shape A\).*?^\}\n', "", text, count=1)
    m = re.search(r"(?m)^command\.def [^\n]*@llm_advance\(", t)
    L = list(re.finditer(r"(?m)^  // ---- layer (\d+)", t[m.start():]))[-1]
    b0 = m.start() + L.start()
    b1 = t.index("\n  %head = command.parameter", b0) + 1
    blk = t[b0:b1]
    first_mask = MASK_RE.search(blk)
    rope = re.search(r"(?m)^  kernel\.launch @g_qk_rope_store\[", blk)
    region = blk[first_mask.end():rope.start()]
    blk = blk[:first_mask.start()] + blk[rope.start():]
    k = blk.rfind("\n  scf.if %mm {\n", 0, blk.index(f"%wk_{L.group(1)},")) + 1
    blk = blk[:k] + region + blk[k:]
    blk = MASK_RE.sub("", blk)
    return t[:b0] + blk + t[b1:]


def to_gen(text):
    """generator-shaped text (kernel.launch, before dual_target) from a served program (the grid is not read by the pass)"""
    return re.sub(r"template\.apply<@sel\d+_(\w+)>\(", r"kernel.launch @\1[%t](", text)


def unit(base_gen):
    import last_rows  # noqa: E402
    t_ok, lay = last_rows.apply(base_gen)
    check("unit: the pass applies to the generator-shaped served program, at layer 63", lay == 63 and t_ok.count("@g_scale_head_rows[") == 4)
    t2, _ = last_rows.apply(t_ok)
    check("unit: a second application returns the text unchanged", t2 == t_ok)
    after63 = base_gen.index("// ---- layer 63", base_gen.index("@llm_advance("))
    l63 = base_gen[after63:]
    down = re.search(r"(?m)^ *kernel\.launch @g_gemm_f16_store_m256\[[^\n]*%f16, %s_f, [^\n]*%y\)[^\n]*\n", l63).group(0)
    rope = re.search(r"(?m)^  kernel\.launch @g_qk_rope_store\[[^\n]*%layer_63,[^\n]*\n", l63).group(0)
    kline = re.search(r"(?m)^ *kernel\.launch @g_gemm_f16_store_m256\[[^\n]*%h16, %s_h, [^\n]*, %k\) :[^\n]*\n", l63).group(0)
    one_mask = MASK_RE.search(t_ok[t_ok.index("// ---- layer 63", t_ok.index("@llm_advance(")):]).group(0)

    def sub(t, a, b, count=1):
        return t[:after63] + t[after63:].replace(a, b, count)
    qs = sorted(set(re.findall(r"(?m)^[^\n]*%h16, %s_h, [^\n]*%qg\)[^\n]*\n", l63)))
    q_on_o = base_gen
    for ln in qs:
        q_on_o = sub(q_on_o, ln, ln.replace("%h16, %s_h,", "%h16, %s_o,"), -1)
    for what, odd in (
            ("a DFlash tap after the down GEMM", sub(base_gen, down, down + "  kernel.launch @g_dfl_tap[%t](%t, %tap_slot_4, %x, %features) : [index](index, index, buffer, buffer)\n")),
            ("a K GEMM reading the q projection", sub(base_gen, kline, kline.replace("%h16, %s_h,", "%q, %s_h,"))),
            ("a statement between V and g_qk_rope_store", sub(base_gen, rope, "  %lr_probe = index.constant 0 : index\n" + rope)),
            ("a %s_h reader between the attention and the FFN norm", sub(base_gen, rope, rope + "  kernel.launch @g_row_scale_f16_live_lv[%t](%t, %info, %nH, %h, %s_h, %h16) : [index](index, buffer, index, buffer, buffer, buffer)\n")),
            ("the q GEMM on another scale buffer", q_on_o),
            ("a half-converted layer (one mask removed)", t_ok.replace(one_mask, "", 1)),
            ("a half-converted layer (a mask, no move)", sub(base_gen, rope, one_mask + rope) + "\n" + last_rows.KERNEL)):
        try:
            last_rows.apply(odd)
            check(f"unit: {what} stops the rewrite", False, "no exit")
        except SystemExit as e:
            check(f"unit: {what} stops the rewrite", str(e).startswith("LAST_ROWS: "), str(e))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        a, b = (os.path.join(d, x) for x in "ab")
        rc0, base, err0 = program(a, LAST_Q="1")
        check("unset: the generator runs (LAST_Q=1)", rc0 == 0, err0[-400:])
        rc1, cand, err1 = program(b, LAST_Q="1", LAST_ROWS="1")
        check("set: the generator runs (LAST_Q=1 LAST_ROWS=1)", rc1 == 0, err1[-400:])
        adv0, adv1 = segment(base, "llm_advance"), segment(cand, "llm_advance")
        check("unset: no mask", "g_scale_head_rows" not in base)
        L, blk = last_layer(adv1)
        masks = MASK_RE.findall(blk)
        check("set: four masks in the last layer (63) on %s_h, %s_o, %s_h, %s_f, none elsewhere", L == 63 and masks == ["%s_h", "%s_o", "%s_h", "%s_f"]
              and cand.count("@g_scale_head_rows[") == 4, (L, masks, cand.count("@g_scale_head_rows[")))
        fm = first(blk, MASK_RE.pattern)
        kpos = [first(blk, rf"(?m)^ *(?:kernel\.launch @|template\.apply<@sel\d+_)g_gemm_\w+[\[>][^\n]*%{w}\)") for w in ("k", "v")]
        qpos = first(blk, r"(?m)^ *(?:kernel\.launch @|template\.apply<@sel\d+_)g_gemm_\w+[\[>][^\n]*%qg\)")
        check("set: K and V GEMMs launch before the first mask, the q GEMM after it", fm > 0 and all(0 < k < fm for k in kpos) and qpos > fm, (kpos, fm, qpos))
        nxt = [blk[m.end():].lstrip().split("\n", 3) for m in re.finditer(MASK_RE.pattern, blk)]
        reads = [re.search(r"%(?:h16|o16|f16), (%s_[hof])\b", "\n".join(x)) for x in nxt]
        check("set: each mask is followed by a GEMM block reading the masked buffer", [r.group(1) if r else None for r in reads] == masks, [r.group(1) if r else None for r in reads])
        check("set: g_scale_head_rows defined once", cand.count('kernel.def export("g_scale_head_rows")') == 1)
        check("set: the verify program is unchanged", segment(cand, "llm_verify") == segment(base, "llm_verify"))
        inv = inverse(cand)
        check("set: the inverse gives the unset program back byte for byte", inv == base,
              next((f"line {i}: {x!r} vs {y!r}" for i, (x, y) in enumerate(zip(inv.splitlines(), base.splitlines())) if x != y), "length"))
        check("set: the build log names the switch and its layer", "LAST_ROWS: layer 63's q, o, gate/up and down GEMMs" in err1, err1[-300:])
        unit(to_gen(base))
    print("FAILED: " + ", ".join(FAILS) if FAILS else "all passed")
    sys.exit(1 if FAILS else 0)
