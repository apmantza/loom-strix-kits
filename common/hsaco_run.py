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
--compare NAME[:f32|f16] after one launch per object: bytes equal, else the max absolute difference (exit 1 when they differ).
usage: hsaco_run.py [--loom FILE --workload k=v,...] [--grid X,Y,Z] --kernel NAME [--scalar k=v ...] --buf NAME=BYTES[:FILL] ...
                    [--reps 20] [--warmup 3] [--bytes N (GB/s)] [--compare NAME] [--dry-run] LABEL=OBJ.hsaco [LABEL=OBJ ...]"""
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


# ------------------------------------------------------------------------------------------------------------ HIP
class Hip:
    def __init__(self):
        self.h = C.CDLL("libamdhip64.so")

    def ck(self, r, what):
        if r != 0:
            s = self.h.hipGetErrorString; s.restype = C.c_char_p
            sys.exit(f"{what}: HIP error {r} {s(r).decode()}")

    def alloc(self, data):
        p = C.c_void_p()
        self.ck(self.h.hipMalloc(C.byref(p), C.c_size_t(len(data))), f"hipMalloc {len(data)}")
        self.ck(self.h.hipMemcpy(p, data.ctypes.data_as(C.c_void_p), C.c_size_t(len(data)), 1), "hipMemcpy H2D")
        return p

    def read(self, p, n):
        out = np.empty(n, np.uint8)
        self.ck(self.h.hipMemcpy(out.ctypes.data_as(C.c_void_p), p, C.c_size_t(n), 2), "hipMemcpy D2H")
        return out


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
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    objs = [o.split("=", 1) if "=" in o else (o, o) for o in a.objects]
    metas = [kernel_meta(p, a.kernel) for _, p in objs]
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
    for x in args:
        names = sorted(alias[x[".name"]])
        print(f"  {x['.offset']:4d} {x['.size']}B {x['.value_kind']:13s} {'/'.join(names)}")
    for (lab, p), m in zip(objs, metas):
        print(f"  {lab}: {p}  vgpr {m.get('.vgpr_count')} sgpr {m.get('.sgpr_count')} lds {m.get('.group_segment_fixed_size')} B")
    if a.dry_run:
        pack_kernargs(metas[0], alias, scalars, {k: 0 for k in bufspec})   # the same checks, no device
        print("dry run: arguments complete")
        return
    hip, rng = Hip(), np.random.default_rng(a.seed)
    host = {k: fill(n, how, rng) for k, (n, how) in bufspec.items()}
    ptrs = {k: hip.alloc(v) for k, v in host.items()}
    kargs = pack_kernargs(metas[0], alias, scalars, {k: p.value for k, p in ptrs.items()})
    fns = []
    for lab, p in objs:
        mod, fn = C.c_void_p(), C.c_void_p()
        img = open(p, "rb").read()
        hip.ck(hip.h.hipModuleLoadData(C.byref(mod), C.c_char_p(img)), f"load {p}")
        hip.ck(hip.h.hipModuleGetFunction(C.byref(fn), mod, a.kernel.encode()), f"get {a.kernel} from {p}")
        fns.append((lab, fn))
    kbuf = C.create_string_buffer(kargs, len(kargs))
    ksize = C.c_size_t(len(kargs))
    extra = (C.c_void_p * 5)(C.c_void_p(1), C.cast(kbuf, C.c_void_p), C.c_void_p(2), C.cast(C.pointer(ksize), C.c_void_p), C.c_void_p(3))

    def launch(fn):
        hip.ck(hip.h.hipModuleLaunchKernel(fn, *grid, *block, 0, None, None, extra), "launch")

    outs = {}
    for lab, fn in fns:                                    # one clean launch each, outputs kept for --compare
        for k, v in host.items():
            hip.ck(hip.h.hipMemcpy(ptrs[k], v.ctypes.data_as(C.c_void_p), C.c_size_t(len(v)), 1), "reset")
        launch(fn)
        hip.ck(hip.h.hipDeviceSynchronize(), f"run {lab}")
        outs[lab] = {c.split(":")[0]: hip.read(ptrs[c.split(":")[0]], len(host[c.split(":")[0]])) for c in a.compare}
    rc = 0
    for c in a.compare:
        name, _, dt = c.partition(":")
        ref = outs[fns[0][0]][name]
        for lab, _ in fns[1:]:
            o = outs[lab][name]
            if np.array_equal(o, ref):
                print(f"compare {name}: {lab} == {fns[0][0]} (bytes equal)")
            else:
                t = {"f16": np.float16, "f32": np.float32}.get(dt or "f32")
                d = np.abs(o.view(t).astype(np.float64) - ref.view(t).astype(np.float64))
                print(f"compare {name}: {lab} != {fns[0][0]}: max |diff| {np.nanmax(d):.4g}, {int((d > 0).sum())} elements differ")
                rc = 1
    e0, e1 = C.c_void_p(), C.c_void_p()
    hip.ck(hip.h.hipEventCreate(C.byref(e0)), "event"); hip.ck(hip.h.hipEventCreate(C.byref(e1)), "event")
    times = {lab: [] for lab, _ in fns}
    for r in range(-a.warmup, a.reps):                     # interleaved: every object once per round
        for lab, fn in fns:
            hip.ck(hip.h.hipEventRecord(e0, None), "rec"); launch(fn); hip.ck(hip.h.hipEventRecord(e1, None), "rec")
            hip.ck(hip.h.hipEventSynchronize(e1), "sync")
            ms = C.c_float(); hip.ck(hip.h.hipEventElapsedTime(C.byref(ms), e0, e1), "elapsed")
            if r >= 0:
                times[lab].append(ms.value)
    base = statistics.median(times[fns[0][0]])
    for lab, t in times.items():
        med = statistics.median(t)
        gbs = f"  {a.bytes / (med * 1e-3) / 1e9:7.1f} GB/s" if a.bytes else ""
        print(f"{lab:12s} median {med:9.4f} ms  best {min(t):9.4f} ms  ({len(t)} reps){gbs}  ratio {med / base:.3f}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
