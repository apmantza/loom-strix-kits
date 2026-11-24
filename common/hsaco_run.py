#!/usr/bin/env python3
"""Launch any Loom (or other) kernel from its code object through HIP, outside Loom: timing, an A/B of several code objects of
the same kernel (two compilers, two specialisations) on identical inputs, output comparison, and a process rocprofv3 can see
(Loom drives /dev/kfd itself; common/counters.sh -- hsaco_run.py ... gives the kernel's hardware counters).

What it reads instead of being told:
  - the kernel arguments (names, offsets, sizes, kinds, kernarg size, workgroup size) from the code object's AMDGPU metadata
    note; hidden arguments are zero-filled;
  - the workgroup grid from the kernel's `kernel.launch.config` region in the .loom source (--loom), evaluated on the workload
    values (--workload): index.constant / add / sub / mul / div / ceildiv / rem / min / max. A region with anything else (an
    indirect grid read from a buffer, a select) is refused: pass --grid X,Y,Z;
  - the device signature (`} launch(...)`) names scalars and buffers: --scalar and --buf accept the source names (k_arg or k)
    or the metadata names (k, binding2).
Buffers: --buf NAME=BYTES[:FILL], FILL = zero | f16u | f32u | u8 | file:PATH (uniform: f16/f32 in [-1, 1], bytes 0..255; raw file bytes, zero-padded; constant operands
raise clocks on this power-limited chip, E15) | byte:0xNN | f32:VALUE. Every buffer the kernel takes must be given (fail closed).
--save NAME=PREFIX writes buffer NAME after the clean launch to PREFIX.LABEL.bin.
--compare NAME[:f32|f16] after one launch per object: bytes equal, else the max absolute difference (exit 1 when they differ).
Cold cache: --rotate auto|BYTES keeps K identically filled copies of every buffer not --write-only (K copies together reach BYTES;
auto = 2 x the GPU's L2 + L3 from the KFD topology, 2 x 34 MiB on gfx1151) and gives each launch the next copy, so no launch reads
what the one before it left in cache; --compare then checks every copy's output. Refused over --mem-budget x free memory.
--anchor N: arm A's median per N-round window against the first window's; rounds past --drift are flagged, kept. --json FILE: the
result, every figure labelled a hypothesis (harness figures diagnose; the seeded served bench decides).
usage: hsaco_run.py [--loom FILE --workload k=v,...] [--grid X,Y,Z] --kernel NAME [--scalar k=v ...] --buf NAME=BYTES[:FILL] ...
                    [--reps 20] [--warmup 3] [--bytes N (GB/s)] [--compare NAME] [--rotate auto|N [--write-only NAME]]
                    [--anchor N [--drift 0.10]] [--json FILE] [--dry-run] LABEL=OBJ.hsaco [LABEL=OBJ ...]"""
import os
import argparse, ctypes as C, math, re, statistics, struct, sys

import msgpack
import numpy as np


# ------------------------------------------------------------------------------------------------------------ code object
def kernel_meta(path, name):
    """The AMDGPU metadata (NT_AMDGPU_METADATA, msgpack) of kernel NAME in an ELF code object."""
    b = open(path, "rb").read()
    if b[:4] != b"\x7fELF" or b[4] != 2:
        sys.exit(f"{path}: not a 64-bit ELF code object")
    shoff, = struct.unpack_from("<Q", b, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", b, 0x3A)
    for i in range(shnum):
        sh_type, = struct.unpack_from("<I", b, shoff + i * shentsize + 4)
        if sh_type != 7:                                   # SHT_NOTE
            continue
        off, size = struct.unpack_from("<QQ", b, shoff + i * shentsize + 0x18)
        p = off
        while p < off + size:
            nsz, dsz, typ = struct.unpack_from("<III", b, p)
            nm = b[p + 12:p + 12 + nsz].rstrip(b"\0")
            d0 = p + 12 + ((nsz + 3) & ~3)
            if nm == b"AMDGPU" and typ == 32:
                md = msgpack.unpackb(b[d0:d0 + dsz], raw=False)
                for k in md.get("amdhsa.kernels", []):
                    if k[".name"] == name:
                        return k
                sys.exit(f"{path}: no kernel {name} (has {[k['.name'] for k in md.get('amdhsa.kernels', [])]})")
            p = d0 + ((dsz + 3) & ~3)
    sys.exit(f"{path}: no AMDGPU metadata note")


# ------------------------------------------------------------------------------------------------------------ .loom source
OPS = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b, "div": lambda a, b: a // b,
       "ceildiv": lambda a, b: -(-a // b), "rem": lambda a, b: a % b, "min": min, "max": max}


def loom_kernel(src, name):
    """(workload params, config-region lines, device params) of kernel NAME (its export name or symbol)."""
    lines = src.split("\n")
    pat = re.compile(r'^kernel\.def\b.*?(?:export\("' + re.escape(name) + r'"\)|@' + re.escape(name) + r'\()')
    for i, l in enumerate(lines):
        if pat.search(l):
            m = re.search(r"@[\w.]+\((.*?)\)\s*\{\s*$", l)
            work = re.findall(r"%(\w+)\s*:", m.group(1)) if m else []
            body = []
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("} launch("):
                    dev = re.findall(r"%(\w+)\s*:\s*([\w<>x]+)", lines[j][len("} launch("):])
                    return work, body, dev
                body.append(lines[j].strip())
    sys.exit(f"--loom: no kernel.def for {name}")


def power_levels():
    """the GPU power level(s); HSACO_RUN_LEVEL_FILE overrides the sysfs path (tests)"""
    import glob as _g, os
    paths = [os.environ["HSACO_RUN_LEVEL_FILE"]] if os.environ.get("HSACO_RUN_LEVEL_FILE") else \
        _g.glob("/sys/class/drm/card*/device/power_dpm_force_performance_level")
    return sorted({open(f).read().strip() for f in paths}) or ["unknown"]


ESIZE = {"f64": 8, "i64": 8, "f32": 4, "i32": 4, "f16": 2, "bf16": 2, "i16": 2, "i8": 1}


def view_needs(src, name, values):
    """{buffer: (bytes its largest zero-offset view of the kernel spans, the view)} for the dense views whose every dimension resolves
    (a constant, or [%v] with %v / %v_arg given as --scalar or --workload); plus the count of views it could not size. Recurrence
    (q104 11-22): h16 is [tokens]x162x32xf16 (the PF16_PAD row stride), the run allocated tokens x 5120, the kernel wrote past the end
    and the GPU fault ('illegal memory access') named no buffer and stopped the window."""
    lines = src.split("\n")
    pat = re.compile(r'^kernel\.def\b.*?(?:export\("' + re.escape(name) + r'"\)|@' + re.escape(name) + r'\()')
    i = next((j for j, l in enumerate(lines) if pat.search(l)), None)
    if i is None:
        return {}, 0
    c = next(j for j in range(i, len(lines)) if lines[j].startswith("} launch("))
    e = next((j for j in range(c + 1, len(lines)) if lines[j] == "}"), len(lines) - 1)
    text = "\n".join(lines[c:e + 1])
    zero = set(re.findall(r"%(\w+) = index\.constant 0 : offset", text))
    need, skipped = {}, 0
    for buf, off, shape in re.findall(r"buffer\.view %(\w+)\[%(\w+)\] : buffer -> view<([^>]*)>", text):
        dims = shape.split("x")
        ty = dims.pop() if dims else ""
        n = 1
        for d in dims:
            m = re.fullmatch(r"\[%(\w+)\]", d)
            v = int(d) if d.isdigit() else (values.get(m[1], values.get(m[1] + "_arg")) if m else None)
            if v is None:
                n = None
                break
            n *= int(v)
        if off not in zero or "," in shape or ty not in ESIZE or n is None:
            skipped += 1
            continue
        b = n * ESIZE[ty]
        if b > need.get(buf, (0, ""))[0]:
            need[buf] = (b, f"view<{shape}>")
    return need, skipped


def eval_config(body, workload):
    env = dict(workload)
    for l in body:
        if not l or l.startswith("//"):
            continue
        m = re.match(r"%(\w+) = index\.constant (-?\d+) : index$", l)
        if m:
            env[m.group(1)] = int(m.group(2)); continue
        m = re.match(r"%(\w+) = index\.(\w+) %(\w+), %(\w+) : index$", l)
        if m and m.group(2) in OPS:
            try:
                env[m.group(1)] = OPS[m.group(2)](env[m.group(3)], env[m.group(4)])
            except KeyError as e:
                sys.exit(f"config region: {e} has no value (give it in --workload)")
            continue
        m = re.match(r"kernel\.launch\.config workgroups\(%(\w+), %(\w+), %(\w+)\) workgroup_size\(%(\w+), %(\w+), %(\w+)\)", l)
        if m:
            v = [env.get(x) for x in m.groups()]
            if None in v:
                sys.exit(f"config region: unresolved {[x for x, y in zip(m.groups(), v) if y is None]}")
            return tuple(v[:3]), tuple(v[3:])
        sys.exit(f"config region: unsupported line (pass --grid): {l}")
    sys.exit("config region: no kernel.launch.config")


# ------------------------------------------------------------------------------------------------------------ arguments
def kv(items, conv=int):
    out = {}
    for it in items:
        for p in it.split(","):
            k, v = p.split("=", 1)
            out[k.strip()] = conv(v)
    return out


def name_args(meta, names):
    """Give the kernel's non-hidden arguments their names when the metadata has none (hipcc code objects of other engines
    carry no argument names): NAMES in argument order. Refuses a count that does not match and unnamed arguments without
    NAMES, so a kernel is never launched with arguments bound by guess."""
    args = [x for x in meta[".args"] if not x[".value_kind"].startswith("hidden_")]
    unnamed = [x for x in args if not x.get(".name")]
    if names is None:
        if unnamed:
            sys.exit(f"{len(unnamed)} of the kernel's {len(args)} arguments have no name in its metadata: pass --arg-names "
                     f"with {len(args)} names in order")
        return meta
    if len(names) != len(args):
        sys.exit(f"--arg-names gives {len(names)} names; the kernel takes {len(args)} arguments")
    for x, n in zip(args, names):
        x[".name"] = n
    return meta


def bind_names(meta, dev):
    """metadata arg name -> every name it may be given by (metadata name, source name, source name without _arg)."""
    args = [a for a in meta[".args"] if not a[".value_kind"].startswith("hidden_")]
    sc = [a for a in args if a[".value_kind"] == "by_value"]
    bf = [a for a in args if a[".value_kind"] == "global_buffer"]
    alias = {a[".name"]: {a[".name"]} for a in args}
    if dev is not None:
        ds = [n for n, t in dev if t in ("index", "i32", "i64", "f32", "f16", "i1", "i8", "i16")]
        db = [n for n, t in dev if t == "buffer" or t.startswith("view<")]
        if len(ds) != len(sc) or len(db) != len(bf):
            sys.exit(f"device signature {dev} does not match the code object's {len(sc)} scalars and {len(bf)} buffers")
        for a, n in list(zip(sc, ds)) + list(zip(bf, db)):
            alias[a[".name"]] |= {n, re.sub(r"_arg$", "", n)}
    return args, alias


def fill(nbytes, how, rng):
    kind, _, val = how.partition(":")
    if kind == "zero":
        return np.zeros(nbytes, np.uint8)
    if kind == "u8":
        return rng.integers(0, 256, nbytes, dtype=np.uint8)
    if kind == "f16u":
        return rng.uniform(-1, 1, nbytes // 2).astype(np.float16).view(np.uint8)
    if kind == "f32u":
        return rng.uniform(-1, 1, nbytes // 4).astype(np.float32).view(np.uint8)
    if kind == "byte":
        return np.full(nbytes, int(val, 0), np.uint8)
    if kind == "f32":
        return np.full(nbytes // 4, float(val), np.float32).view(np.uint8)
    if kind == "file":                                    # raw bytes (e.g. a launch's metadata packets), zero-padded
        b = np.fromfile(val, np.uint8)
        if len(b) > nbytes:
            sys.exit(f"fill {how}: {len(b)} bytes do not fit the {nbytes}-byte buffer")
        return np.concatenate([b, np.zeros(nbytes - len(b), np.uint8)])
    sys.exit(f"unknown fill {how} (zero | f16u | f32u | u8 | byte:0xNN | f32:VALUE | file:PATH)")


def pack_kernargs(meta, alias, scalars, pointers):
    buf = bytearray(meta[".kernarg_segment_size"])
    given = set()
    for a in meta[".args"]:
        o, s, kind, nm = a[".offset"], a[".size"], a[".value_kind"], a[".name"]
        if kind.startswith("hidden_"):
            continue
        hit = [k for k in (scalars if kind == "by_value" else pointers) if k in alias[nm]]
        if len(hit) != 1:
            sys.exit(f"argument {nm} (aka {sorted(alias[nm])}): {'missing' if not hit else 'given twice: ' + str(hit)}")
        given.add(hit[0])
        v = (scalars if kind == "by_value" else pointers)[hit[0]]
        if kind == "global_buffer":
            struct.pack_into("<Q", buf, o, v)
        elif s in (4, 8):
            struct.pack_into("<i" if s == 4 else "<q", buf, o, v)
        else:
            sys.exit(f"argument {nm}: by_value of size {s} not supported")
    extra = set(scalars) | set(pointers)
    if extra - given:
        sys.exit(f"given but not kernel arguments: {sorted(extra - given)}")
    return bytes(buf)



# ------------------------------------------------------------------------------------------------------------ cold cache
# Every timed repetition used to re-read the same buffers, so inputs smaller than the last-level cache came from cache in the
# harness and from DRAM in serving: a candidate cause of the harness-vs-served gaps (HRX-like E300, E313, E386), untested until
# this option. Rotation keeps K identically filled copies of every buffer the kernel may read and gives each launch the next copy.
def llc_bytes(nodes="/sys/class/kfd/kfd/topology/nodes"):
    """(bytes, source) of the GPU's caches past L1 (every level >= 2 instance, summed), from the KFD topology the driver
    publishes (rocminfo prints the same: on gfx1151 L2 2,048 KiB + L3 / Infinity Cache 32,768 KiB, so not the 64 MiB some
    engines state). Refuses when no GPU node lists them: --rotate auto is then not computable, give --rotate-bytes."""
    import glob as _g
    for node in sorted(_g.glob(os.path.join(nodes, "*"))):
        try:
            props = dict(l.split() for l in open(os.path.join(node, "properties")) if len(l.split()) == 2)
        except OSError:
            continue
        if int(props.get("simd_count", 0)) == 0:
            continue
        lv = {}
        for c in sorted(_g.glob(os.path.join(node, "caches", "*", "properties"))):
            p = dict(l.split() for l in open(c) if len(l.split()) == 2)
            if int(p.get("level", 0)) >= 2:
                lv.setdefault(int(p["level"]), []).append(int(p["size"]))   # KiB
        if lv:
            src = " + ".join(f"L{k} {sum(v)} KiB" + (f" ({len(v)} instances)" if len(v) > 1 else "") for k, v in sorted(lv.items()))
            return sum(sum(v) for v in lv.values()) * 1024, f"KFD topology {os.path.basename(node)}: {src}"
    sys.exit("--rotate auto: no GPU node with level >= 2 caches in the KFD topology; give --rotate-bytes N")


def free_memory(meminfo="/proc/meminfo", drm="/sys/class/drm/card*/device"):
    """(bytes, source): the smaller of MemAvailable and the GPU's free GTT (unified memory: HIP allocations come from GTT, and a
    live server holds ~50 GB of it)."""
    import glob as _g
    avail = next(int(l.split()[1]) * 1024 for l in open(meminfo) if l.startswith("MemAvailable:"))
    out, src = avail, f"MemAvailable {avail / 2**30:.1f} GiB"
    for d in sorted(_g.glob(drm)):
        try:
            g = int(open(os.path.join(d, "mem_info_gtt_total")).read()) - int(open(os.path.join(d, "mem_info_gtt_used")).read())
        except OSError:
            continue
        src += f", GTT free {g / 2**30:.1f} GiB ({os.path.basename(os.path.dirname(d))})"
        out = min(out, g)
    return out, src


ALIGN = 4096


def rotation_plan(bufspec, write_only, rotate_bytes):
    """(rotated names, bytes per copy, copies): every buffer rotates except those named write-only (the kernel never reads them:
    an unknown name is refused, and a buffer not named is assumed read, so a miss costs memory, never a warm read). Copies = the
    fewest whose rotated bytes together reach ROTATE_BYTES (1 when one copy already does, or when rotation is off)."""
    bad = sorted(set(write_only) - set(bufspec))
    if bad:
        sys.exit(f"--write-only names no buffer: {bad} (buffers: {sorted(bufspec)})")
    rot = [k for k in bufspec if k not in write_only]
    per = sum(bufspec[k][0] for k in rot)
    copies = max(1, math.ceil(rotate_bytes / per)) if rotate_bytes and per else 1
    return rot, per, copies


def memory_guard(bufspec, rot, copies, free, budget):
    """refuse when the buffers (rotated ones x COPIES, each copy page-aligned) exceed BUDGET x the free memory"""
    need = sum((-(-n // ALIGN) * ALIGN) * (copies if k in rot else 1) for k, (n, _) in bufspec.items())
    if need > budget * free[0]:
        sys.exit(f"rotation: {copies} copies need {need / 2**30:.2f} GiB, over {budget:g} x free memory ({free[1]}): "
                 f"lower --rotate-bytes, mark write-only buffers, or stop the live server")
    return need


def anchor_drift(times_a, every, threshold):
    """Arm A (the reference, timed every round of the interleave) as its own anchor: its median per window of EVERY rounds against
    the first window's. Rounds in a window whose |drift| exceeds THRESHOLD are flagged, never dropped: under the 85 W cap a clock
    change can be the candidate's own effect, so neither clock nor drift alone discards a round (llama.cpp's tuner discards, cools
    and re-times; that suits a search, not an A/B). Returns {windows: [(first, last, median, drift)], flagged: [round], max_drift}."""
    if not every or not times_a:
        return None
    win = [(i, min(i + every, len(times_a)) - 1) for i in range(0, len(times_a), every)]
    base = statistics.median(times_a[win[0][0]:win[0][1] + 1])
    rows, flagged = [], []
    for s, e in win:
        m = statistics.median(times_a[s:e + 1])
        d = m / base - 1
        rows.append((s, e, m, d))
        if abs(d) > threshold:
            flagged += range(s, e + 1)
    return {"every": every, "threshold": threshold, "windows": rows, "flagged": flagged,
            "max_drift": max((abs(r[3]) for r in rows), default=0.0)}


# ------------------------------------------------------------------------------------------------------------ HIP
class Hip:
    """the device boundary: addresses are ints; prepare() builds a launch's kernarg packet ahead of the timed region (host work
    between the start event and the launch would be timed as GPU idle)"""
    def __init__(self, grid, block):
        self.h, self.grid, self.block = C.CDLL("libamdhip64.so"), grid, block
        self.e0, self.e1 = C.c_void_p(), C.c_void_p()
        self.ck(self.h.hipEventCreate(C.byref(self.e0)), "event"); self.ck(self.h.hipEventCreate(C.byref(self.e1)), "event")

    def ck(self, r, what):
        if r != 0:
            s = self.h.hipGetErrorString; s.restype = C.c_char_p
            sys.exit(f"{what}: HIP error {r} {s(r).decode()}")

    def alloc(self, n):
        p = C.c_void_p()
        self.ck(self.h.hipMalloc(C.byref(p), C.c_size_t(n)), f"hipMalloc {n}")
        return p.value

    def write(self, addr, data):
        self.ck(self.h.hipMemcpy(C.c_void_p(addr), data.ctypes.data_as(C.c_void_p), C.c_size_t(len(data)), 1), "hipMemcpy H2D")

    def read(self, addr, n):
        out = np.empty(n, np.uint8)
        self.ck(self.h.hipMemcpy(out.ctypes.data_as(C.c_void_p), C.c_void_p(addr), C.c_size_t(n), 2), "hipMemcpy D2H")
        return out

    def load(self, path, kernel):
        mod, fn = C.c_void_p(), C.c_void_p()
        self.ck(self.h.hipModuleLoadData(C.byref(mod), C.c_char_p(open(path, "rb").read())), f"load {path}")
        self.ck(self.h.hipModuleGetFunction(C.byref(fn), mod, kernel.encode()), f"get {kernel} from {path}")
        return fn

    def prepare(self, kargs):
        kbuf = C.create_string_buffer(kargs, len(kargs)); ksize = C.c_size_t(len(kargs))
        extra = (C.c_void_p * 5)(C.c_void_p(1), C.cast(kbuf, C.c_void_p), C.c_void_p(2), C.cast(C.pointer(ksize), C.c_void_p), C.c_void_p(3))
        return (kbuf, ksize, extra)                        # kept alive with the packet

    def launch(self, fn, prep, what):
        self.ck(self.h.hipModuleLaunchKernel(fn, *self.grid, *self.block, 0, None, None, prep[2]), "launch")
        self.ck(self.h.hipDeviceSynchronize(), f"run {what}")

    def timed(self, fn, prep):
        h = self.h
        self.ck(h.hipEventRecord(self.e0, None), "rec")
        self.ck(h.hipModuleLaunchKernel(fn, *self.grid, *self.block, 0, None, None, prep[2]), "launch")
        self.ck(h.hipEventRecord(self.e1, None), "rec")
        self.ck(h.hipEventSynchronize(self.e1), "sync")
        ms = C.c_float(); self.ck(h.hipEventElapsedTime(C.byref(ms), self.e0, self.e1), "elapsed")
        return ms.value


def measure(dev, fns, meta, alias, scalars, bufspec, host, rot, copies, compare=(), save=(), reps=20, warmup=3, anchor=0,
            drift=0.10):
    """Allocate (rotated buffers: COPIES page-aligned copies in one block, each filled from the same host bytes), launch every object
    once per copy (outputs of every copy kept for --compare), then time WARMUP + REPS interleaved rounds, launch j on copy j mod
    COPIES. Returns (result dict, exit code)."""
    base, stride = {}, {}
    for k, (n, _) in bufspec.items():
        stride[k] = -(-n // ALIGN) * ALIGN if k in rot else 0
        base[k] = dev.alloc(stride[k] * copies if k in rot else n)
    addr = [{k: base[k] + c * stride[k] for k in bufspec} for c in range(copies)]
    preps = [dev.prepare(pack_kernargs(meta, alias, scalars, addr[c])) for c in range(copies)]
    names = [c.split(":")[0] for c in compare]
    outs, rc, lines = {}, 0, []
    for lab, fn in fns:                                    # clean launches: every copy, outputs kept for --compare
        outs[lab] = []
        for c in range(copies):
            for k, v in host.items():
                dev.write(addr[c][k], v)
            dev.launch(fn, preps[c], lab)
            outs[lab].append({n: dev.read(addr[c][n], len(host[n])) for n in names})
            if c == 0:
                for sv in save:
                    name, _, prefix = sv.partition("=")
                    dev.read(addr[0][name], len(host[name])).tofile(f"{prefix or name}.{lab}.bin")
    for cmp in compare:
        name, _, dt = cmp.partition(":")
        ref = outs[fns[0][0]][0][name]
        for lab, _ in fns:
            for c in range(copies):
                if (lab, c) == (fns[0][0], 0):
                    continue
                o, who = outs[lab][c][name], f"{lab}" + (f" copy {c}" if copies > 1 else "")
                if np.array_equal(o, ref):
                    if c == 0:
                        lines.append(f"compare {name}: {lab} == {fns[0][0]} (bytes equal)")
                    continue
                t = {"f16": np.float16, "f32": np.float32}.get(dt or "f32")
                d = np.abs(o.view(t).astype(np.float64) - ref.view(t).astype(np.float64))
                lines.append(f"compare {name}: {who} != {fns[0][0]}" + (" copy 0" if copies > 1 else "")
                             + f": max |diff| {np.nanmax(d):.4g}, {int((d > 0).sum())} elements differ")
                rc = 1
        if copies > 1 and rc == 0:
            lines.append(f"compare {name}: all {copies} copies equal, every object")
    times = {lab: [] for lab, _ in fns}
    order = {lab: [] for lab, _ in fns}
    j = 0
    for r in range(-warmup, reps):                         # interleaved: every object once per round, each launch the next copy
        for lab, fn in fns:
            c = j % copies
            j += 1
            ms = dev.timed(fn, preps[c])
            if r >= 0:
                times[lab].append(ms); order[lab].append(c)
    anc = anchor_drift(times[fns[0][0]], anchor, drift)
    return {"times": times, "copy_order": order, "anchor": anc, "compare": lines}, rc


HYPOTHESIS = ("hypothesis: an isolated harness figure diagnoses; only the seeded served bench (agent bench) decides "
              "(AGENTS 7; E300, E313, E386)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("objects", nargs="+", help="LABEL=OBJ.hsaco")
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--loom")
    ap.add_argument("--workload", action="append", default=[])
    ap.add_argument("--grid")
    ap.add_argument("--block")
    ap.add_argument("--scalar", action="append", default=[])
    ap.add_argument("--buf", action="append", default=[])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--bytes", type=float)
    ap.add_argument("--compare", action="append", default=[])
    ap.add_argument("--save", action="append", default=[], help="NAME=PREFIX: after the clean launch, write buffer NAME to PREFIX.LABEL.bin "
                    "(e.g. a stamped build's per-wave stage timings, HRX-like E294)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--arg-names", help="comma-separated argument names in order, for code objects whose metadata has none")
    ap.add_argument("--rotate", help="cold-cache timing: 'auto' (2 x the GPU's L2 + L3 from the KFD topology) or bytes the rotated "
                    "copies reach together; each launch reads the next of K identical copies of every buffer not --write-only")
    ap.add_argument("--rotate-bytes", type=float, help="the same as --rotate N")
    ap.add_argument("--write-only", action="append", default=[], help="NAME[,NAME]: buffers the kernel only writes (kept single "
                    "under rotation); every other buffer is assumed read and rotates")
    ap.add_argument("--mem-budget", type=float, default=0.5, help="refuse a rotation needing more than this share of free memory "
                    "(min of MemAvailable and free GTT; default 0.5)")
    ap.add_argument("--anchor", type=int, default=0, help="N: arm A's median per N-round window against the first window's; "
                    "rounds in a window drifting more than --drift are flagged, not dropped")
    ap.add_argument("--drift", type=float, default=0.10, help="anchor drift that flags a window (default 0.10, llama.cpp's tuner figure)")
    ap.add_argument("--json", help="write the result (times, rotation, anchor drift, each figure labelled a hypothesis) to this file")
    a = ap.parse_args()
    objs = [o.split("=", 1) if "=" in o else (o, o) for o in a.objects]
    metas = [name_args(kernel_meta(p, a.kernel), a.arg_names.split(",") if a.arg_names else None) for _, p in objs]
    lay = [[(x[".name"], x[".offset"], x[".size"], x[".value_kind"]) for x in m[".args"]] for m in metas]
    if any(l != lay[0] for l in lay):
        sys.exit("the code objects' argument layouts differ: not an A/B of one kernel")
    dev, grid, block = None, None, None
    if a.loom:
        work, body, dev = loom_kernel(open(a.loom).read(), a.kernel)
        if a.grid is None:
            w = kv(a.workload)
            miss = [x for x in work if x not in w]
            if miss:
                sys.exit(f"--workload needs {miss} (the kernel's workload signature {work})")
            grid, block = eval_config(body, w)
    if a.grid:
        grid = tuple(int(x) for x in a.grid.split(","))
    if a.block:
        block = tuple(int(x) for x in a.block.split(","))
    if grid is None:
        sys.exit("no grid: give --loom and --workload, or --grid")
    if block is None:
        rq = metas[0].get(".reqd_workgroup_size") or [metas[0][".max_flat_workgroup_size"], 1, 1]
        block = tuple(rq)
    if math.prod(block) > metas[0][".max_flat_workgroup_size"]:
        sys.exit(f"workgroup {block} exceeds the kernel's max_flat_workgroup_size {metas[0]['.max_flat_workgroup_size']}")
    args, alias = bind_names(metas[0], dev)
    scalars = kv(a.scalar)
    bufspec = {}
    for b in a.buf:
        n, _, rest = b.partition("=")
        sz, _, how = rest.partition(":")
        bufspec[n] = (int(float(sz)), how or "zero")
    print(f"kernel {a.kernel}: grid {grid} x workgroup {block}; kernarg {metas[0]['.kernarg_segment_size']} B")
    lv = power_levels()
    print(f"  power level {'/'.join(lv)}" + ("" if lv == ["auto"] else "  ** NOT the serving level (auto): times here need not hold in service (HRX-like E300) **"))
    if lv != ["auto"] and not a.dry_run and os.environ.get("HSACO_RUN_ANY_LEVEL") != "1":
        # timings at another level misled two nights of decisions (E300: 2x at profile_standard, 3-8 % served): refuse, unless the
        # caller only wants counts (common/counters.sh sets HSACO_RUN_ANY_LEVEL=1) or says so explicitly
        sys.exit(f"hsaco_run: power level {'/'.join(lv)}, not auto: timings would not hold in service (set HSACO_RUN_ANY_LEVEL=1 for counts)")
    if a.loom:                                            # buffers too small for the kernel's own views refuse before any launch
        need, skipped = view_needs(open(a.loom).read(), a.kernel, {**kv(a.workload), **scalars})
        short = [f"{k}: {bufspec[k][0]} B given, {v[1]} needs {v[0]} B" for k, v in sorted(need.items()) if k in bufspec and bufspec[k][0] < v[0]]
        if short:
            sys.exit("buffers smaller than the kernel's views:\n  " + "\n  ".join(short))
        print(f"  buffer sizes checked against {len(need)} sized views ({skipped} views not sizable statically)")
    for x in args:
        names = sorted(alias[x[".name"]])
        print(f"  {x['.offset']:4d} {x['.size']}B {x['.value_kind']:13s} {'/'.join(names)}")
    for (lab, p), m in zip(objs, metas):
        print(f"  {lab}: {p}  vgpr {m.get('.vgpr_count')} sgpr {m.get('.sgpr_count')} lds {m.get('.group_segment_fixed_size')} B")
    if a.rotate and a.rotate_bytes:
        sys.exit("give --rotate or --rotate-bytes, not both")
    rspec = a.rotate if a.rotate else (str(int(a.rotate_bytes)) if a.rotate_bytes else None)
    llc, rbytes = None, 0
    if rspec == "auto":
        llc = llc_bytes()
        rbytes = 2 * llc[0]
    elif rspec:
        rbytes = int(float(rspec))
    wo = [x for it in a.write_only for x in it.split(",") if x]
    rot, per, copies = rotation_plan(bufspec, wo if rspec else list(bufspec), rbytes)
    rinfo = {"bytes": rbytes, "spec": rspec, "llc_bytes": llc[0] if llc else None, "llc_source": llc[1] if llc else None,
             "copies": copies, "per_copy_bytes": per, "rotated": rot if rspec else [], "single": [k for k in bufspec if k not in rot]}
    if rspec:
        free = free_memory()
        total = memory_guard(bufspec, rot, copies, free, a.mem_budget)
        rinfo.update(total_bytes=total, free=free[1], budget=a.mem_budget)
        print(f"  rotation: {copies} copies x {per / 2**20:.1f} MiB of {rot} (target {rbytes / 2**20:.1f} MiB"
              + (f" = 2 x {llc[0] / 2**20:.0f} MiB, {llc[1]}" if llc else "") + f"); single: {rinfo['single']}; "
              f"{total / 2**30:.2f} GiB of {free[1]}" + ("  (1 copy: the inputs alone reach the target)" if copies == 1 else ""))
    if a.dry_run:
        pack_kernargs(metas[0], alias, scalars, {k: 0 for k in bufspec})   # the same checks, no device
        print("dry run: arguments complete")
        return
    hip, rng = Hip(grid, block), np.random.default_rng(a.seed)
    host = {k: fill(n, how, rng) for k, (n, how) in bufspec.items()}
    fns = [(lab, hip.load(p, a.kernel)) for lab, p in objs]
    res, rc = measure(hip, fns, metas[0], alias, scalars, bufspec, host, rot if rspec else [], copies, a.compare, a.save,
                      a.reps, a.warmup, a.anchor, a.drift)
    for l in res["compare"]:
        print(l)
    times, anc = res["times"], res["anchor"]
    base = statistics.median(times[fns[0][0]])
    arms = {}
    for lab, t in times.items():
        med = statistics.median(t)
        gbs = f"  {a.bytes / (med * 1e-3) / 1e9:7.1f} GB/s" if a.bytes else ""
        arms[lab] = {"median_ms": med, "best_ms": min(t), "ratio": med / base, "times_ms": t, "status": "hypothesis"}
        if anc and anc["flagged"]:
            keep = [x for i, x in enumerate(t) if i not in set(anc["flagged"])]
            arms[lab]["median_unflagged_ms"] = statistics.median(keep) if keep else None
        print(f"{lab:12s} median {med:9.4f} ms  best {min(t):9.4f} ms  ({len(t)} reps){gbs}  ratio {med / base:.3f}"
              + ("  cold (rotated)" if rspec and copies > 1 else ""))
    if anc:
        print(f"anchor {fns[0][0]} every {anc['every']} rounds: drift " + " ".join(f"{w[3]:+.1%}" for w in anc["windows"])
              + f"; {len(anc['flagged'])} of {len(times[fns[0][0]])} rounds flagged (> {anc['threshold']:.0%}, kept)")
    print(HYPOTHESIS)
    if a.json:
        import json
        out = {"status": HYPOTHESIS, "kernel": a.kernel, "grid": grid, "block": block, "power_level": lv, "reps": a.reps,
               "warmup": a.warmup, "rotation": rinfo, "anchor": anc, "arms": arms, "copy_order": res["copy_order"],
               "compare": res["compare"], "exit": rc}
        json.dump(out, open(a.json, "w"), indent=1)
    sys.exit(rc)


if __name__ == "__main__":
    main()
