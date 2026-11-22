#!/usr/bin/env python3
"""common/hsaco_run.py without a GPU. Recurrences it prevents:
  - argument metadata read wrong (an offset or kind off silently feeds a kernel garbage): every kernel in every code object
    under DIR is checked against llvm-readelf's own reading of the same note;
  - a kernel picked by prefix (g_gemm_x_m256 matched for g_gemm_x_m256_acc) or a config region evaluated past its own
    `kernel.launch.config`: every kernel.def in a real program is resolved by exact name, and its region either evaluates or
    is refused with the unsupported line, never guessed;
  - an argument missing, given twice or misspelt passing the kernarg packer.
usage: test_hsaco_run.py PROGRAM.loom CODE_OBJECT_DIR"""
import glob, os, re, subprocess, sys

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import hsaco_run as hr  # noqa: E402

fails = 0


def ok(cond, what):
    global fails
    print(("ok  " if cond else "FAIL") + " " + what)
    fails += not cond


def readelf_args(path):
    """kernel -> [(name, offset, size, kind)] from llvm-readelf --notes (the reference reading, its YAML read line by line)."""
    t = subprocess.run(["llvm-readelf-21", "--notes", path], capture_output=True, text=True).stdout.split("\n")
    out, args, cur, name = {}, None, None, None
    for l in t:
        if l.startswith("  - .args:"):
            args, cur, name = [], None, None
        elif args is not None and re.match(r"^      - \.", l):
            cur = {}; args.append(cur); l = l.replace("- ", "  ", 1)
        if args is None:
            continue
        m = re.match(r"^(\s+)(\.\w+):\s+(?:!str )?(\S+)?", l)
        if not m:
            continue
        ind, k, v = len(m.group(1)), m.group(2), m.group(3)
        if ind >= 8 and cur is not None:
            cur[k] = v
        elif ind == 4:
            cur = None
            if k == ".name":
                name = v
                out[name] = [(x.get(".name", ""), int(x[".offset"]), int(x[".size"]), x[".value_kind"]) for x in args]
    return out


def main():
    prog, objdir = sys.argv[1], sys.argv[2]
    # 1. metadata against llvm-readelf, every code object
    objs = sorted(glob.glob(os.path.join(objdir, "**", "*.hsaco"), recursive=True))
    ok(len(objs) > 50, f"{len(objs)} code objects found")
    bad, n = [], 0
    for p in objs:
        ref = readelf_args(p)
        for k, args in ref.items():
            m = hr.kernel_meta(p, k)
            mine = [(a.get(".name", ""), a[".offset"], a[".size"], a[".value_kind"]) for a in m[".args"]]
            n += 1
            if mine != args:
                bad.append((p, k))
    ok(n > 50 and not bad, f"argument layouts of {n} kernels equal llvm-readelf's ({len(bad)} differ: {bad[:2]})")
    # 2. every kernel.def in the program: exact-name lookup; the region evaluates or is refused
    src = open(prog).read()
    names = re.findall(r'^kernel\.def\b[^\n]*?export\("([\w.]+)"\)', src, re.M)
    evaluated, refused, wrong = 0, [], []
    for name in names:
        work, body, dev = hr.loom_kernel(src, name)
        hdr = [l for l in src.split("\n") if l.startswith("kernel.def") and f'export("{name}")' in l]
        if len(hdr) != 1 or not re.search(r"@" + re.escape(name) + r"\(", hdr[0]):
            wrong.append(name)
        try:
            grid, block = hr.eval_config(body, {w: 4096 for w in work})
            evaluated += all(g >= 1 for g in grid) and all(b >= 1 for b in block)
        except SystemExit as e:
            refused.append((name, str(e)[:60]))
    ok(not wrong, f"{len(names)} kernel.defs resolved by exact name ({len(wrong)} not: {wrong[:3]})")
    ok(all("unsupported line" in r for _, r in refused), f"{evaluated} regions evaluated, {len(refused)} refused with the line: {refused[:2]}")
    ok(evaluated >= len(names) - 10, "nearly every region evaluates (index arithmetic only)")
    # 3. the packer refuses incomplete, doubled or unknown arguments
    gemm = [p for p in objs if os.path.basename(p) == "g_gemm_iq4_nl_store_m256.hsaco"]
    if gemm:
        m = hr.kernel_meta(gemm[0], "g_gemm_iq4_nl_store_m256")
        _, _, dev = hr.loom_kernel(src, "g_gemm_iq4_nl_store_m256")
        args, alias = hr.bind_names(m, dev)
        sc, pt = {"rows": 1024, "k": 5120, "n": 17408}, {"x": 16, "scales": 32, "w": 48, "y": 64}
        kb = hr.pack_kernargs(m, alias, sc, pt)
        import struct
        ok(struct.unpack_from("<iii", kb, 0) == (1024, 5120, 17408) and struct.unpack_from("<QQQQ", kb, 16) == (16, 32, 48, 64),
           "kernargs land at the metadata offsets in source order (x, scales, w, y = binding0..3)")
        for what, s2, p2 in (("missing", sc, {k: v for k, v in pt.items() if k != "w"}),
                             ("twice", {**sc, "k_arg": 1}, pt), ("unknown", {**sc, "kk": 1}, pt)):
            try:
                hr.pack_kernargs(m, alias, s2, p2); ok(False, f"packer refuses an argument {what}")
            except SystemExit:
                ok(True, f"packer refuses an argument {what}")
    # 4. file fill (a launch's metadata packets): bytes in order, zero-padded, oversize refused. Recurrence: a packet file
    #    longer than its buffer silently truncated would launch the head kernel with a wrong span count.
    import numpy as np, tempfile
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "m.bin"); np.array([8, 0, 0, 0, 1], np.int32).tofile(p)
        b = hr.fill(32, "file:" + p, np.random.default_rng(0))
        ok(list(b[:20].view(np.int32)) == [8, 0, 0, 0, 1] and not b[20:].any() and len(b) == 32, "file fill keeps the bytes and zero-pads")
        try:
            hr.fill(16, "file:" + p, np.random.default_rng(0)); ok(False, "file fill refuses a file larger than its buffer")
        except SystemExit:
            ok(True, "file fill refuses a file larger than its buffer")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
