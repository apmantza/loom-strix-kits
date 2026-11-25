#!/usr/bin/env python3
"""FLASH2_SXH (HRX-like E456 / E462; flash2_trim.sxh after sx11): unset leaves gen_flash2s's output byte-identical; set, exactly the gfx11
kernels that carry SX11's S-partial exchange (g_attention_flash2; the split-K g_attention_vsplit) change, and each changed kernel is its
unset text with the exchange split into halves: one [w]x2x2x32x4 view, every 8-wide store as two 4-wide stores, every 8-wide load as two
4-wide loads + one vector.concat<0>, and nothing else (an independent inverse rewrite gives back the unset kernel byte for byte).
Composes with FLASH2_SMX; refuses without FLASH2_SX11 by name.
Recurrences it guards: a generator change that moves the exchange text so sxh silently rewrites part of it (one store split, its load
left on the 32-byte record: the partner reads the wrong half, wrong attention, compiles fine); a dropped concat or half (the 8-vector
then carries 4 values: wrong sums); the half layout leaking into kernels without the exchange (partial, gfx12 twins); a switch that
does nothing because SX11 is off.
usage: python3 test_flash2_sxh.py [KIT]   (KIT's kit.env recipe gives the flash switches, default qwen38-27b; exit 1 on a failure)"""
import json, os, re, shlex, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL = re.compile(r'export\("(\w+)"\).*?\n\} launch\(.*?\n\}\n', re.S)
# geometry: the split-dims generator's head_dim-256 family (gen_flash2s asserts head_dim 256 / 512); two query-head groupings
GEOMS = [{"kv_heads": 4, "heads": 24, "head_dim": 256, "layers": 64}, {"kv_heads": 2, "heads": 16, "head_dim": 256, "layers": 32}]


def recipe(kit):
    for ln in open(os.path.join(HERE, "..", "kits", kit, "kit.env")):
        if ln.startswith("KIT_RECIPE="):
            return dict(kv.split("=", 1) for kv in shlex.split(ln.split("=", 1)[1].split("#")[0]))
    raise SystemExit(f"no KIT_RECIPE in kits/{kit}/kit.env")


def gen(tmp, cfg, base, vsplit, **extra):
    out = os.path.join(tmp, "flash2.loom")
    env = {**os.environ, **base, "LLM_CONFIG": cfg, "FLASH2_TARGET": "llm", "FLASH2_OUT": out, **extra}
    if vsplit:   # as gen_llm.py runs it for g_attention_vsplit
        env.update(FLASH2_ROLE="partial", FLASH2_SPLITK="1", FLASH2_SPLITK_FROM="2048")
    env = {k: v for k, v in env.items() if v is not None}
    r = subprocess.run([sys.executable, os.path.join(HERE, "gen_flash2s.py")], capture_output=True, text=True, env=env)
    text = open(out).read() if r.returncode == 0 else ""
    if os.path.exists(out):
        os.remove(out)
    return r, text


def unsplit(k):
    """Independent inverse of the half layout (not sxh's code): the 32-byte-record text SX11 emits."""
    k = re.sub(r"\n  %sxh_view = buffer\.view %s_tile\[%origin\] : buffer -> view<\d+x2x2x32x4xf32>", "", k)
    k = re.sub(r"^( *)(%\w+)_sxa = vector\.slice \2\[0\] : vector<8xf32> -> vector<4xf32>\n\1\2_sxb = vector\.slice \2\[4\] : vector<8xf32> -> vector<4xf32>\n"
               r"\1vector\.store \2_sxa, %sxh_view\[%wave, (%c[01]), %c0, %lane, %c0\] : vector<4xf32>, view<(\d+)x2x2x32x4xf32>\n"
               r"\1vector\.store \2_sxb, %sxh_view\[%wave, \3, %c1, %lane, %c0\] : vector<4xf32>, view<\4x2x2x32x4xf32>$",
               r"\1vector.store \2, %sx_view[%wave, \3, %lane, %c0] : vector<8xf32>, view<\4x2x32x8xf32>", k, flags=re.M)
    k = re.sub(r"^( *)(%\w+)_sxa = vector\.load %sxh_view\[%partner_id, (%c[01]), %c0, %lane, %c0\] : view<(\d+)x2x2x32x4xf32> -> vector<4xf32>\n"
               r"\1\2_sxb = vector\.load %sxh_view\[%partner_id, \3, %c1, %lane, %c0\] : view<\4x2x2x32x4xf32> -> vector<4xf32>\n"
               r"\1\2 = vector\.concat<0> \2_sxa, \2_sxb : vector<4xf32>, vector<4xf32> -> vector<8xf32>$",
               r"\1\2 = vector.load %sx_view[%partner_id, \3, %lane, %c0] : view<\4x2x32x8xf32> -> vector<8xf32>", k, flags=re.M)
    return k


def main():
    kit = sys.argv[1] if len(sys.argv) > 1 else "qwen38-27b"
    base = {**recipe(kit), "FLASH2_SX11": "1", "FLASH2_SXH": None, "FLASH2_SMX": None, "FLASH2_SMX2": None}
    fails, n = 0, 0

    def check(ok, what):
        nonlocal fails, n
        n += 1; fails += not ok
        print(("ok  " if ok else "FAIL") + "  " + what)

    with tempfile.TemporaryDirectory() as tmp:
        for gi, geom in enumerate(GEOMS):
            cfg = os.path.join(tmp, f"cfg{gi}.json"); json.dump(geom, open(cfg, "w"))
            tag = f"kv{geom['kv_heads']}/h{geom['heads']}"
            for vs in (False, True):
                what = f"{tag} {'vsplit' if vs else 'flash2'}"
                r0, t0 = gen(tmp, cfg, base, vs)
                if r0.returncode != 0:
                    check(False, f"{what}: generator (unset) exit {r0.returncode}: {r0.stderr.strip()[-300:]}"); continue
                rz, tz = gen(tmp, cfg, base, vs, FLASH2_SXH="0")
                check(rz.returncode == 0 and tz == t0 and "%sxh_view" not in t0, f"{what}: unset and 0 give the same text, no half view")
                r1, t1 = gen(tmp, cfg, base, vs, FLASH2_SXH="1")
                if r1.returncode != 0:
                    check(False, f"{what}: generator (SXH=1) exit {r1.returncode}: {r1.stderr.strip()[-300:]}"); continue
                k0 = {m.group(1): m.group(0) for m in KERNEL.finditer(t0)}
                k1 = {m.group(1): m.group(0) for m in KERNEL.finditer(t1)}
                want = {k for k, v in k0.items() if "%sx_view = buffer.view" in v}
                changed = {k for k in k1 if k1[k] != k0.get(k)}
                check(set(k1) == set(k0) and len(want) == 1 and changed == want,
                      f"{what}: exactly the SX11 kernel changes (want {sorted(want)}, changed {sorted(changed)})")
                for k in sorted(want & set(k1)):
                    a, b = k0[k], k1[k]
                    w = re.search(r"%sx_view = buffer\.view %s_tile\[%origin\] : buffer -> view<(\d+)x2x32x8xf32>", a).group(1)
                    s0 = len(re.findall(r"vector\.store %\w+, %sx_view\[", a)); l0 = len(re.findall(r"= vector\.load %sx_view\[", a))
                    counts = (len(re.findall(rf"%sxh_view = buffer\.view %s_tile\[%origin\] : buffer -> view<{w}x2x2x32x4xf32>", b)),
                              len(re.findall(r"vector\.store %\w+_sx[ab], %sxh_view\[", b)), len(re.findall(r"= vector\.load %sxh_view\[", b)),
                              len(re.findall(r"= vector\.concat<0> (%\w+)_sxa, \1_sxb :", b)), b.count("%sx_view["))
                    check(s0 == l0 and s0 == (2 if vs else 4) and counts == (1, 2 * s0, 2 * l0, l0, 0),
                          f"{what} {k}: view / half stores / half loads / concats / left on the record = {counts} "
                          f"(want (1, {2 * s0}, {2 * l0}, {l0}, 0) from {s0} stores, {l0} loads)")
                    check(unsplit(b) == a, f"{what} {k}: the inverse rewrite gives back the unset kernel byte for byte")
                if not vs:
                    rs0, ts0 = gen(tmp, cfg, base, vs, FLASH2_SMX="1")
                    rs1, ts1 = gen(tmp, cfg, base, vs, FLASH2_SMX="1", FLASH2_SXH="1")
                    ks1 = {m.group(1): m.group(0) for m in KERNEL.finditer(ts1)}
                    kk = next(iter(want))
                    check(rs0.returncode == 0 and rs1.returncode == 0 and ts0 != t0 and kk in ks1
                          and ks1[kk].count("%sxh_view[") == 4 * len(re.findall(r"vector\.store %\w+, %sx_view\[", k0[kk]))
                          and "%sx_view[" not in ks1[kk] and unsplit(ks1[kk]) == {m.group(1): m.group(0) for m in KERNEL.finditer(ts0)}.get(kk),
                          f"{what}: composes with FLASH2_SMX (exit {rs0.returncode} / {rs1.returncode}; SXH = SMX's kernel, halves only)")
        cfg = os.path.join(tmp, "cfg0.json")
        rn, _ = gen(tmp, cfg, base, False, FLASH2_SX11="0", FLASH2_SXH="1")
        check(rn.returncode != 0 and "FLASH2_SXH: rewrites FLASH2_SX11's exchange" in rn.stderr, "SXH without SX11 refuses by name")
    print(f"{n - fails}/{n} passed"); return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
