"""Static tests of LAST_Q=1 (HRX-like E661, kit/last_q.py): the last attention layer of @llm_advance reads g_flash_tiles_out's
work list (the flash tiles holding a selected span's last token) instead of the full one. Recurrence each case prevents:
- unset: the whole package program is byte-identical (a switch that leaks into the default program changes live's next package);
- set: only the program's last attention layer reads the cropped list, and both its flash kernels do (a crop in a non-final layer
  feeds stale rows to the next layers: inexact everywhere; a crop that misses the partial kernel recomputes nothing it drops but
  leaves its tail blocks on the full list, a silent half-lever);
- set: the program is exactly the unset program plus the crop launch, its list buffer and the kernel: an independent inverse gives
  the unset program back byte for byte (a rewrite that touched the verify program or another launch would not invert);
- fail closed (rules section 4: a text pass that finds code by shape must stop on a shape it does not know): a launch after the
  last attention layer's flash kernels that is not a known row-local kernel (a DFlash tap, a drafter feature write), a layer
  after the last attention layer, and a third list reader stop the rewrite; so does a second application (its layer then has no
  reader of the full list).
usage: python3 test_last_q.py (exit 1 on a failure)."""
import os, re, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FAILS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))
    if not ok:
        FAILS.append(name)


def program(out, **kw):
    """The served program as the kit builds it: common/build.sh (kit.env's recipe + EXTRA) into the package dir `out`
    (rules section 2, E230: test the generator as the served program emits it)."""
    env = dict(os.environ)
    for k in ("LAST_Q", "LAST_OUT", "EXTRA", "LINT_BASE"):
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


def readers(seg):
    """(layer, flash kernel, list buffer) of every flash list read in a program segment, as the served text spells it (dual_target:
    template.apply<@selN_K>)"""
    out, layer = [], None
    for ln in seg.splitlines():
        m = re.match(r"  // ---- layer (\d+)", ln)
        if m:
            layer = int(m.group(1))
        m = re.search(r"(?:kernel\.launch @|template\.apply<@sel\d+_)(g_attention_flash2\w*)[\[>].*?(%flash_tiles\w*),", ln)
        if m:
            out.append((layer, m.group(1), m.group(2)))
    return out


def inverse(text):
    """the crop launch, its list buffer and the kernel removed, %flash_tiles_out read as %flash_tiles"""
    t = re.sub(r"(?m)^ *(?:kernel\.launch @|template\.apply<@sel\d+_)g_flash_tiles_out[\[>][^\n]*\n", "", text)
    t = re.sub(r"(?m)^  %flash_tiles_out = buffer\.alloca[^\n]*\n", "", t)
    t = re.sub(r'(?ms)\n\n^// LAST_Q \(HRX-like E661\).*?^func\.def inline @g_block_holds_output.*?^\}\n', "", t)   # apply: "\n" + KERNEL
    t = re.sub(r"(?ms)^template\.decl @sel\d+_g_flash_tiles_out\(.*?\n(?=template\.decl|kernel\.def|command\.def|config\.decl|func\.def|//|\n)", "", t)
    t = re.sub(r"(?ms)^template\.def<@sel\d+_g_flash_tiles_out>.*?^\}\n", "", t)
    return t.replace("%flash_tiles_out,", "%flash_tiles,")


with tempfile.TemporaryDirectory() as d:
    a, b = (os.path.join(d, x) for x in "ab")
    rc0, base, err0 = program(a)
    check("unset: the generator runs", rc0 == 0, err0[-400:])
    rc1, cand, err1 = program(b, LAST_Q="1")
    check("set: the generator runs", rc1 == 0, err1[-400:])
    adv0, adv1 = segment(base, "llm_advance"), segment(cand, "llm_advance")
    r0, r1 = readers(adv0), readers(adv1)
    attn = sorted({x[0] for x in r0})
    check("unset: 16 attention layers read the full list with both flash kernels", len(attn) == 16 and len(r0) == 32
          and all(x[2] == "%flash_tiles" for x in r0), r0)
    last = max(attn) if attn else None
    want = [(x[0], x[1], "%flash_tiles_out" if x[0] == last else "%flash_tiles") for x in r0]
    check("set: exactly the last attention layer's two flash kernels read %flash_tiles_out", r1 == want and last == 63,
          [x for x in r1 if x not in want] or f"last {last}")
    check("set: the program's last layer is that layer", re.findall(r"(?m)^  // ---- layer (\d+)", adv1)[-1:] == [str(last)])
    crop = re.findall(r"(?m)^ *(?:kernel\.launch @|template\.apply<@sel\d+_)g_flash_tiles_out[\[>][^\n]*$", cand)
    check("set: one crop launch, in @llm_advance, after the work-list launch", len(crop) == 1 and crop[0] in adv1
          and adv1.index(crop[0]) > adv1.index("g_attention_fallback_list") and adv1.index(crop[0]) < adv1.index("// ---- layer 0"), crop)
    check("set: g_flash_tiles_out defined once", cand.count('kernel.def export("g_flash_tiles_out")') == 1)
    check("set: the verify program is unchanged", segment(cand, "llm_verify") == segment(base, "llm_verify"))
    inv = inverse(cand)
    check("set: the inverse gives the unset program back byte for byte", inv == base,
          next((f"line {i}: {x!r} vs {y!r}" for i, (x, y) in enumerate(zip(inv.splitlines(), base.splitlines())) if x != y), "length"))
    check("set: the build log names the switch and its layer", "LAST_Q: layer 63's flash kernels read g_flash_tiles_out's list" in err1, err1[-300:])

    # fail closed: the pass on generator-shaped text (kernel.launch, before dual_target) built from the served unset program
    import last_q  # noqa: E402
    gen = re.sub(r"template\.apply<@sel\d+_(\w+)>\(", r"kernel.launch @\1[%t](", base)   # launch spelling (the grid is not read)
    t_ok, lay = last_q.apply(gen)
    check("unit: the pass applies to the generator-shaped served program, at layer 63", lay == 63 and "%flash_tiles_out" in t_ok)
    part = re.search(r"(?m)^ *kernel\.launch @g_attention_flash2_partial_f16o\[[^\n]*%layer_63,[^\n]*\n", gen).group(0)
    tap = part + "  kernel.launch @g_dfl_tap[%t](%t, %tap_slot_4, %x, %features) : [index](index, index, buffer, buffer)\n"
    for what, odd in (("a DFlash tap after the last flash launch", gen.replace(part, tap, 1)),
                      ("a layer after the last attention layer", gen.replace("\n  %head = command.parameter", "\n  // ---- layer 64 (gated delta-net)\n  %head = command.parameter", 1)),
                      ("a third list reader", gen.replace(part, part + part.replace("g_attention_flash2_partial_f16o", "g_attention_flash2_deep_f16o"), 1)),
                      ("a second application", t_ok)):
        try:
            last_q.apply(odd)
            check(f"unit: {what} stops the rewrite", False, "no exit")
        except SystemExit as e:
            check(f"unit: {what} stops the rewrite", str(e).startswith("LAST_Q: "), str(e))
print("FAILED: " + ", ".join(FAILS) if FAILS else "all passed")
sys.exit(1 if FAILS else 0)
