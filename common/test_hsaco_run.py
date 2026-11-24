#!/usr/bin/env python3
"""common/hsaco_run.py without a GPU. Recurrences it prevents:
  - argument metadata read wrong (an offset or kind off silently feeds a kernel garbage): every kernel in every code object
    under DIR is checked against llvm-readelf's own reading of the same note;
  - a kernel picked by prefix (g_gemm_x_m256 matched for g_gemm_x_m256_acc) or a config region evaluated past its own
    `kernel.launch.config`: every kernel.def in a real program is resolved by exact name, and its region either evaluates or
    is refused with the unsupported line, never guessed;
  - an argument missing, given twice or misspelt passing the kernarg packer;
  - another engine's code object (no argument names in its metadata) launched with arguments bound by guess: --arg-names
    must name them in order, and a missing or miscounted list is refused;
  - a buffer smaller than the kernel's view of it launched anyway (q104 11-22: a GPU fault naming no buffer): view_needs sizes
    each zero-offset view from the source and the given scalars.
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


class FakeDev:
    """hsaco_run.Hip's boundary in host memory (the device is the one process boundary; no GPU here): a bump allocator over a byte
    array, kernels as Python callables over (memory, kernarg packet), timings from a list. Records each timed launch's packet."""
    def __init__(self, times=()):
        import numpy as np
        self.mem, self.top, self.times, self.timed_packets = np.zeros(1 << 20, np.uint8), 4096, list(times), []

    def alloc(self, n):
        a = self.top; self.top += -(-n // 4096) * 4096
        assert self.top <= len(self.mem), "fake device out of memory"
        return a

    def write(self, addr, data):
        self.mem[addr:addr + len(data)] = data

    def read(self, addr, n):
        return self.mem[addr:addr + n].copy()

    def prepare(self, kargs):
        return kargs

    def launch(self, fn, prep, what):
        fn(self.mem, prep)

    def timed(self, fn, prep):
        fn(self.mem, prep); self.timed_packets.append(prep)
        return self.times.pop(0) if self.times else 1.0


def cold_cache_checks():
    """--rotate / --anchor (measure, rotation_plan, memory_guard, anchor_drift, llc_bytes) on the fake device. Recurrences: the
    harness re-reading one cached copy while claiming cold timing (copies not cycled, or cycled per round so arm B reads what arm A
    just pulled in); copies filled differently (a copy's output then differs for a reason that is not the kernel); a rotation
    that fills the box's memory while live serves ~53 GB; anchor drift computed against the wrong window or rounds silently dropped;
    --rotate auto sized from a figure nobody measured (TensorFold states 64 MiB; this chip's KFD topology says L2 2 + L3 32 MiB)."""
    import numpy as np, struct, tempfile
    meta = {".args": [{".name": "x", ".offset": 0, ".size": 8, ".value_kind": "global_buffer"},
                      {".name": "w", ".offset": 8, ".size": 8, ".value_kind": "global_buffer"},
                      {".name": "y", ".offset": 16, ".size": 8, ".value_kind": "global_buffer"},
                      {".name": "n", ".offset": 24, ".size": 4, ".value_kind": "by_value"}], ".kernarg_segment_size": 32}
    _, alias = hr.bind_names(meta, None)

    def kern(mem, ka, bias=0):                             # y[i] = x[i] + w[i] + BIAS (bytes, mod 256)
        x, w, y = struct.unpack_from("<QQQ", ka, 0); n, = struct.unpack_from("<i", ka, 24)
        mem[y:y + n] = ((mem[x:x + n].astype(np.int32) + mem[w:w + n] + bias) % 256).astype(np.uint8)
    n = 3000
    spec = {"x": (n, "u8"), "w": (n, "u8"), "y": (n, "zero")}
    rng = np.random.default_rng(1)
    host = {k: hr.fill(s, h, rng) for k, (s, h) in spec.items()}
    # 1. plan: copies = fewest reaching the target over the rotated (non write-only) buffers; unknown write-only refused
    rot, per, copies = hr.rotation_plan(spec, ["y"], 5 * 2 * n - 1)
    ok((rot, per, copies) == (["x", "w"], 2 * n, 5), f"plan: x and w rotate, y single, 5 copies reach {5 * 2 * n - 1} B ({rot}, {per}, {copies})")
    ok(hr.rotation_plan(spec, list(spec), 0)[2] == 1 and hr.rotation_plan(spec, ["y"], 0)[2] == 1, "rotation off: one copy")
    try:
        hr.rotation_plan(spec, ["q"], 100); ok(False, "an unknown --write-only name is refused")
    except SystemExit:
        ok(True, "an unknown --write-only name is refused")
    # 2. measure: copies allocated apart and filled identically; launches cycle copies 0..K-1 in order across arms
    dev = FakeDev()
    res, rc = hr.measure(dev, [("A", kern), ("B", kern)], meta, alias, {"n": n}, spec, host, rot, copies, compare=["y:f32"], reps=6, warmup=1)
    xs = [struct.unpack_from("<Q", p, 0)[0] for p in dev.timed_packets]
    ys = [struct.unpack_from("<Q", p, 16)[0] for p in dev.timed_packets]
    xaddr = sorted(set(xs))
    ok(len(xaddr) == copies and all(b - a >= n for a, b in zip(xaddr, xaddr[1:])), f"{copies} distinct, non-overlapping copies of x launched")
    ok(len(set(ys)) == 1, "the write-only output stays single")
    seq = [xaddr.index(v) for v in xs]
    ok(seq == [j % copies for j in range(len(seq))], f"each launch takes the next copy, across arms (sequence {seq[:12]})")
    ok(res["copy_order"]["A"] == [j % copies for j in range(2, 14, 2)] and res["copy_order"]["B"] == [j % copies for j in range(3, 14, 2)],
       f"recorded copy order per arm after the warm-up round ({res['copy_order']})")
    ok(all(np.array_equal(dev.read(a, n), host["x"]) for a in xaddr), "every copy of x holds the same bytes as the host fill")
    ok(rc == 0 and any(f"all {copies} copies equal" in l for l in res["compare"]), f"--compare checks every copy ({res['compare']})")
    # 3. a copy whose result differs is reported and fails (exit 1): every copy but the first adds 1
    first = {}

    def kern2(mem, ka):
        x, = struct.unpack_from("<Q", ka, 0); first.setdefault("x", x)
        kern(mem, ka, bias=int(x != first["x"]))
    res2, rc2 = hr.measure(FakeDev(), [("A", kern2)], meta, alias, {"n": n}, spec, host, rot, copies, compare=["y:f32"], reps=1, warmup=0)
    ok(rc2 == 1 and any("copy 1 !=" in l for l in res2["compare"]), f"a copy with a different result fails the compare ({res2['compare'][:1]})")
    # 4. memory guard: refuses over budget x free memory, passes under it; free = min(MemAvailable, free GTT)
    with tempfile.TemporaryDirectory() as t:
        mi = os.path.join(t, "meminfo"); open(mi, "w").write("MemTotal: 100 kB\nMemAvailable:   1000 kB\n")
        card = os.path.join(t, "card0", "device"); os.makedirs(card)
        open(os.path.join(card, "mem_info_gtt_total"), "w").write(str(900 * 1024)); open(os.path.join(card, "mem_info_gtt_used"), "w").write(str(100 * 1024))
        free = hr.free_memory(mi, os.path.join(t, "card*", "device"))
        ok(free[0] == 800 * 1024 and "GTT free" in free[1], f"free memory = the smaller of MemAvailable and free GTT ({free})")
        big = {"x": (100 * 1024, "u8"), "y": (4096, "zero")}
        try:
            hr.memory_guard(big, ["x"], 5, free, 0.5); ok(False, "memory guard refuses 5 x 100 KiB over 0.5 x 800 KiB")
        except SystemExit as e:
            ok("free memory" in str(e), "memory guard refuses 5 x 100 KiB over 0.5 x 800 KiB")
        ok(hr.memory_guard(big, ["x"], 3, free, 0.5) == 3 * 100 * 1024 + 4096, "memory guard passes 3 copies (304 KiB) and counts the single buffer once")
        # 5. LLC from a KFD topology: the GPU node (simd_count > 0), levels >= 2 summed, L1 left out
        for node, simd, caches in (("0", 0, [(3, 99999)]), ("1", 80, [(1, 32), (2, 2048), (3, 32768)])):
            d = os.path.join(t, "nodes", node); os.makedirs(d)
            open(os.path.join(d, "properties"), "w").write(f"cpu_cores_count 16\nsimd_count {simd}\n")
            for i, (lv, kib) in enumerate(caches):
                os.makedirs(os.path.join(d, "caches", str(i)))
                open(os.path.join(d, "caches", str(i), "properties"), "w").write(f"processor_id_low 1\nlevel {lv}\nsize {kib}\n")
        llc = hr.llc_bytes(os.path.join(t, "nodes"))
        ok(llc[0] == (2048 + 32768) * 1024 and "L2 2048 KiB + L3 32768 KiB" in llc[1], f"LLC from the GPU node, L2 + L3 ({llc})")
    # 6. anchor drift: arm A's window medians against the first window; drifting windows flagged, every round kept
    ta = [1.0] * 4 + [1.02] * 4 + [1.2] * 4 + [1.0, 1.0]
    an = hr.anchor_drift(ta, 4, 0.10)
    ok([round(w[3], 3) for w in an["windows"]] == [0.0, 0.02, 0.2, 0.0] and an["flagged"] == [8, 9, 10, 11],
       f"drift per window against the first, rounds 8-11 flagged ({[round(w[3], 3) for w in an['windows']]}, {an['flagged']})")
    ok(abs(an["max_drift"] - 0.2) < 1e-9 and hr.anchor_drift(ta, 0, 0.1) is None, "max drift reported; no anchor without N")
    dev3 = FakeDev(times=[9.0, 9.0] + [x for v in ta for x in (v, 2.0)])
    res3, _ = hr.measure(dev3, [("A", kern), ("B", kern)], meta, alias, {"n": n}, spec, host, [], 1, reps=len(ta), warmup=1, anchor=4)
    ok(res3["anchor"]["flagged"] == [8, 9, 10, 11] and len(res3["times"]["B"]) == len(ta) and res3["times"]["A"] == ta,
       "measure: the anchor reads arm A's timed rounds (warm-up excluded); flagged rounds stay in both arms' times")


def main():
    cold_cache_checks()
    if len(sys.argv) < 3:                                  # the checks below need a real program and its code objects
        print("(no PROGRAM.loom CODE_OBJECT_DIR: metadata, region and packer checks skipped)")
        sys.exit(1 if fails else 0)
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
    # 5. unnamed arguments (a hipcc build of another engine's kernel): named in order, or refused
    import copy
    meta = {".args": [{".offset": 0, ".size": 8, ".value_kind": "global_buffer"}, {".offset": 8, ".size": 4, ".value_kind": "by_value"},
                      {".offset": 16, ".size": 8, ".value_kind": "hidden_block_count_x"}], ".kernarg_segment_size": 24}
    m = hr.name_args(copy.deepcopy(meta), ["X", "M"])
    ok([x.get(".name") for x in m[".args"]] == ["X", "M", None], "--arg-names names the non-hidden arguments in order")
    for what, names in (("without --arg-names", None), ("with a wrong count", ["X"])):
        try:
            hr.name_args(copy.deepcopy(meta), names); ok(False, f"unnamed arguments refused {what}")
        except SystemExit:
            ok(True, f"unnamed arguments refused {what}")
    # 6. buffer sizes against the kernel's own views (view_needs). Recurrence (q104 11-22): the norm kernel's h16 view is
    #    [tokens]x162x32xf16 (padded rows), the run gave tokens x 5120 f16 and the GPU faulted naming no buffer.
    need, skipped = hr.view_needs(src, "g_post_add_norm_rs_live_lv", {"tokens": 4096})
    if need:
        ok(need.get("h16", (0,))[0] == 4096 * 162 * 32 * 2, f"h16's padded view sized from the source ({need.get('h16')})")
        ok(need.get("info", (0,))[0] == 4096 * 16 and skipped == 0, f"every view of the norm kernel sized ({len(need)} buffers, {skipped} skipped)")
        n2, s2 = hr.view_needs(src, "g_post_add_norm_rs_live_lv", {})
        ok("h16" not in n2 and s2 > 0, "a dimension with no value given is counted unsized, not guessed")
    toy = """kernel.def export("t") @t(%n: index) {
} launch(%n_arg: index, %a: buffer, %b: buffer) {
  %z = index.constant 0 : offset
  %o = index.constant 64 : offset
  %av = buffer.view %a[%z] : buffer -> view<[%n]x4xf32>
  %bv = buffer.view %b[%o] : buffer -> view<8xf32>
  kernel.return
}"""
    n3, s3 = hr.view_needs(toy, "t", {"n_arg": 3})
    ok(n3 == {"a": (48, "view<[%n]x4xf32>")} and s3 == 1, f"_arg names resolve; a non-zero offset is skipped, not sized ({n3}, {s3})")
    # 7. timing refused at a power level other than auto (E300: kernel A/Bs at profile_standard said 2x where serving gave 3-8 %);
    #    counts allowed with HSACO_RUN_ANY_LEVEL=1 (counters.sh); dry runs unaffected. The level comes from a fake file: no GPU is touched
    #    (the refusal precedes device setup).
    if gemm:
        with tempfile.TemporaryDirectory() as t:
            lf = os.path.join(t, "level"); open(lf, "w").write("profile_standard\n")
            base = [sys.executable, os.path.join(H, "hsaco_run.py"), "--kernel", "g_gemm_iq4_nl_store_m256", "--loom", sys.argv[1],
                    "--workload", "rows=1024,n=17408", "--scalar", "rows=1024,k=5120,n=17408", "--buf", "x=1:zero", "--buf", "scales=1:zero",
                    "--buf", "w=1:zero", "--buf", "y=1:zero", f"A={gemm[0]}"]
            env = dict(os.environ, HSACO_RUN_LEVEL_FILE=lf); env.pop("HSACO_RUN_ANY_LEVEL", None)
            r = subprocess.run(base, capture_output=True, text=True, env=env)
            ok(r.returncode != 0 and "not auto" in (r.stdout + r.stderr), "timing at profile_standard refused (exit nonzero, 'not auto')")
            r = subprocess.run(base + ["--dry-run"], capture_output=True, text=True, env=env)
            ok("not auto" not in r.stderr, "a dry run at profile_standard is not refused")
            # the CLI wires the rotation plan and the memory guard ahead of any device (dry run)
            rb = [x if x != "scales=1:zero" else "scales=8192:zero" for x in base]   # scales sized to its view (8 B x rows)
            r = subprocess.run(rb + ["--dry-run", "--rotate-bytes", "30000", "--write-only", "y"], capture_output=True, text=True, env=env)
            ok(r.returncode == 0 and "rotation: 4 copies" in r.stdout and "single: ['y']" in r.stdout, f"--rotate-bytes plans copies in a dry run ({r.stdout[-160:]!r})")
            r = subprocess.run(rb + ["--dry-run", "--rotate-bytes", "30000", "--mem-budget", "1e-12"], capture_output=True, text=True, env=env)
            ok(r.returncode != 0 and "free memory" in r.stderr, "the memory guard refuses through the CLI before any device")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
