"""A small simulator of Loom kernel text for the kit's static value tests (HRX-like E275): runs a kernel's launch body for each workgroup
with all its lanes at once (numpy arrays over the lanes), in float32, so a generated kernel can be compared with a reference without a
GPU (loom-run has no CPU device here). It covers the ops the kit's scalar / per-lane kernels use, and refuses anything else (an unknown
line raises): a kernel it cannot run is not silently passed.

Semantics: lanes execute statement by statement together, so a barrier is a no-op (no lane runs ahead); scf.if masks stores and
yields; scf.for runs while any active lane is inside its own bounds (bounds may differ per lane); subgroup ops act on 32-lane groups,
workgroup reductions on all lanes; LDS (buffer.alloca<workgroup>) is per workgroup. Values: f32 math in float32, index math in int64.
Transcendentals use numpy (exp, log, tanh, sqrt, logistic), so results equal the GPU's only to the <afn> functions' error: compare
with a tolerance unless the kernel uses only + - * / (then float32 results are bit-exact for a fixed operation order).

usage: run(text, buffers, scalars, grid, wg_size) where buffers maps launch buffer names to numpy arrays (modified in place),
scalars maps launch index / scalar argument names to ints, grid is (x, y, z) workgroups, wg_size the lanes per workgroup."""
import re

import numpy as np

f32 = np.float32
TY = {"f32": np.float32, "f16": np.float16, "i32": np.int32, "i8": np.int8, "i64": np.int64, "bf16": np.uint16}   # bf16 kept as raw words


def _blocks(lines):
    """nest the lines into (head, body, else_body) blocks by braces"""
    out, i = [], 0

    def parse(i):
        items = []
        while i < len(lines):
            ln = lines[i]
            if ln == "}" or ln.startswith("} else"):
                return items, i
            if ln.endswith("{"):
                body, i = parse(i + 1)
                els = None
                if lines[i].startswith("} else"):
                    els, i = parse(i + 1)
                items.append((ln, body, els))
                i += 1
            else:
                items.append(ln)
                i += 1
        return items, i
    out, _ = parse(0)
    return out


def _dims(spec, env):
    parts = spec.split("x")
    dims, ty = [], parts[-1]
    for p in parts[:-1]:
        dims.append(int(env[p[2:-1]][0]) if p.startswith("[%") else int(p))
    return dims, ty


class Sim:
    def __init__(self, text, buffers, scalars, wg_size):
        launch = text[text.index("} launch(") + len("} launch("):]
        sig, body = launch.split(") {", 1)
        self.args = [a.split(":")[0].strip()[1:] for a in sig.split(",") if a.strip()]
        body = body[:body.rindex("kernel.return")]
        self.tree = _blocks([ln.strip() for ln in body.splitlines() if ln.strip()])
        self.buffers, self.scalars, self.N = buffers, scalars, wg_size

    def run(self, grid):
        for z in range(grid[2]):
            for y in range(grid[1]):
                for x in range(grid[0]):
                    env = {a: np.full(self.N, self.scalars[a]) for a in self.args if a in self.scalars}
                    self.wg, self.lds, self.views = (x, y, z), {}, {}
                    self.block(self.tree, env, np.ones(self.N, bool))

    # -- helpers --
    def val(self, env, tok):
        tok = tok.strip()
        if tok.startswith("%"):
            return env[tok[1:]]
        return np.full(self.N, int(tok))

    def mem(self, name, idx, env):
        """(flat array, flat index per lane) of view `name` at index tokens idx"""
        buf, dims, off = self.views[name]
        flat = np.zeros(self.N, np.int64)
        for d, t in zip(dims, idx):
            flat = flat * d + self.val(env, t)
        return buf, flat + off

    def block(self, items, env, mask):
        for it in items:
            if isinstance(it, tuple):
                self.compound(it, env, mask)
            else:
                self.stmt(it, env, mask)

    def compound(self, it, env, mask):
        head, body, els = it
        m = re.match(r"(?:(%[\w, %]+) = )?scf\.if %(\w+)", head)
        if m:
            c = env[m.group(2)].astype(bool)
            res = [r.strip()[1:] for r in m.group(1).split(",")] if m.group(1) else []
            y1 = self.block_yield(body, env, mask & c)
            y0 = self.block_yield(els, env, mask & ~c) if els is not None else None
            for k, r in enumerate(res):
                env[r] = np.where(c if y1[k].ndim == 1 else c[:, None], y1[k], y0[k])
            return
        m = re.match(r"(?:(%[\w, %]+) = )?scf\.for %(\w+) = \[%(\w+) to %(\w+) step %(\w+)\]\s*(?:\((.*)\) -> \(.*\))?\s*\{$", head)
        if m:
            res = [r.strip()[1:] for r in m.group(1).split(",")] if m.group(1) else []
            iv, lo, hi, st = m.group(2), env[m.group(3)].copy(), env[m.group(4)], env[m.group(5)]
            carried = []
            if m.group(6):
                for a in m.group(6).split(","):
                    n, init = a.split("=")
                    carried.append(n.strip()[1:])
                    env[n.strip()[1:]] = env[init.split(":")[0].strip()[1:]].copy()
            cur = lo
            while True:
                act = mask & (cur < hi)
                if not act.any():
                    break
                env[iv] = cur.copy()
                ys = self.block_yield(body, env, act)
                for k, n in enumerate(carried):
                    env[n] = np.where(act if ys[k].ndim == 1 else act[:, None], ys[k], env[n])
                cur = np.where(act, cur + st, cur)
            for k, r in enumerate(res):
                env[r] = env[carried[k]]
            return
        raise SystemExit(f"loomsim: unknown block {head!r}")

    def block_yield(self, items, env, mask):
        ys = []
        for it in items:
            if isinstance(it, str) and it.startswith("scf.yield"):
                ys = [env[t.strip().split(":")[0].strip()[1:]] for t in it[len("scf.yield"):].split(":")[0].split(",")]
            elif isinstance(it, tuple):
                self.compound(it, env, mask)
            else:
                self.stmt(it, env, mask)
        return ys

    def stmt(self, ln, env, mask):
        N = self.N
        if ln.startswith("kernel.barrier"):
            return
        m = re.match(r"%(\w+) = buffer\.view %(\w+)\[%(\w+)\] : buffer -> view<(.+)>$", ln)
        if m:
            dims, ty = _dims(m.group(4), env)
            src = self.lds[m.group(2)] if m.group(2) in self.lds else self.buffers[m.group(2)]
            off = int(env[m.group(3)][0]) if m.group(3) in env else 0
            arr = src.view(TY[ty]).reshape(-1) if src.dtype != TY[ty] else src.reshape(-1)
            self.views[m.group(1)] = (arr, dims, off // np.dtype(TY[ty]).itemsize)
            return
        m = re.match(r"%(\w+) = buffer\.alloca<workgroup> align\(\d+\) %(\w+) : buffer$", ln)
        if m:
            self.lds[m.group(1)] = np.zeros(int(env[m.group(2)][0]), np.uint8)
            return
        m = re.match(r"%(\w+) = index\.constant (-?\d+) : (index|offset)$", ln)
        if m:
            env[m.group(1)] = np.full(N, int(m.group(2)))
            return
        m = re.match(r"%(\w+) = index\.(add|sub|mul|div|rem|min|max) %(\w+), %(\w+) : index$", ln)
        if m:
            a, b = env[m.group(3)], env[m.group(4)]
            op = m.group(2)
            env[m.group(1)] = {"add": a + b, "sub": a - b, "mul": a * b, "min": np.minimum(a, b), "max": np.maximum(a, b),
                               "div": a // np.maximum(b, 1), "rem": a % np.maximum(b, 1)}[op]
            return
        m = re.match(r"%(\w+) = index\.scale %(\w+), %(\w+) : index, offset -> offset$", ln)
        if m:
            env[m.group(1)] = env[m.group(2)] * env[m.group(3)]
            return
        m = re.match(r"%(\w+) = index\.assume %(\w+) ", ln)
        if m:
            env[m.group(1)] = env[m.group(2)]
            return
        m = re.match(r"%(\w+) = index\.cmp (ult|eq|ne|ule),? %(\w+), %(\w+) : index$", ln)
        if m:
            a, b = env[m.group(3)], env[m.group(4)]
            env[m.group(1)] = {"ult": a < b, "eq": a == b, "ne": a != b, "ule": a <= b}[m.group(2)]
            return
        m = re.match(r"%(\w+) = kernel\.(workgroup\.id<(\w)>|workitem\.id<x>|subgroup\.id|subgroup\.lane\.id) : index$", ln)
        if m:
            k = m.group(2)
            lanes = np.arange(N)
            env[m.group(1)] = (np.full(N, self.wg["xyz".index(m.group(3))]) if k.startswith("workgroup") else lanes if k == "workitem.id<x>"
                               else lanes // 32 if k == "subgroup.id" else lanes % 32)
            return
        m = re.match(r"%(\w+) = scalar\.constant (\S+) : (f32|i32|i8)$", ln)
        if m:
            env[m.group(1)] = np.full(N, f32(float(m.group(2)))) if m.group(3) == "f32" else np.full(N, int(m.group(2)))
            return
        m = re.match(r"%(\w+) = vector\.splatf (\S+) : vector<(\d+)xf32>$", ln)
        if m:
            env[m.group(1)] = np.full((N, int(m.group(3))), f32(float(m.group(2))))
            return
        m = re.match(r"%(\w+) = scalar\.(mulf|addf|subf|divf|maxnumf|minnumf) %(\w+), %(\w+) : f32$", ln)
        if m:
            a, b = env[m.group(3)].astype(f32), env[m.group(4)].astype(f32)
            op = m.group(2)
            with np.errstate(all="ignore"):
                env[m.group(1)] = {"mulf": a * b, "addf": a + b, "subf": a - b, "divf": a / b,
                                   "maxnumf": np.fmax(a, b), "minnumf": np.fmin(a, b)}[op].astype(f32)
            return
        m = re.match(r"%(\w+) = scalar\.fmaf %(\w+), %(\w+), %(\w+) : f32$", ln)
        if m:
            a, b, c = (env[m.group(k)].astype(np.float64) for k in (2, 3, 4))
            env[m.group(1)] = (a * b + c).astype(f32)                        # one rounding, as a fused multiply-add
            return
        m = re.match(r"%(\w+) = scalar\.(sqrtf|rsqrtf|expf<afn>|logf<afn>|logisticf<afn>|tanhf<afn>|siluf<afn>|softplusf<afn>) %(\w+) : f32$", ln)
        if m:
            a = env[m.group(3)].astype(np.float64)
            op = m.group(2).split("<")[0]
            with np.errstate(all="ignore"):
                r = {"sqrtf": np.sqrt(a), "rsqrtf": 1 / np.sqrt(a), "expf": np.exp(a), "logf": np.log(a), "tanhf": np.tanh(a),
                     "logisticf": 1 / (1 + np.exp(-a)), "siluf": a / (1 + np.exp(-a)), "softplusf": np.logaddexp(0, a)}[op]
            env[m.group(1)] = r.astype(f32)
            return
        m = re.match(r"%(\w+) = scalar\.extsi %(\w+) : i8 to i32$", ln)
        if m:
            env[m.group(1)] = env[m.group(2)].astype(np.int8).astype(np.int64)
            return
        m = re.match(r"%(\w+) = vector\.sitofp %(\w+) : vector<(\d+)xi8> to vector<\d+xf32>$", ln)
        if m:
            env[m.group(1)] = env[m.group(2)].astype(np.int8).astype(np.float32)
            return
        m = re.match(r"%(\w+) = scalar\.sitofp %(\w+) : i32 to f32$", ln)
        if m:
            env[m.group(1)] = env[m.group(2)].astype(f32)
            return
        m = re.match(r"%(\w+) = scf\.select %(\w+), %(\w+), %(\w+) : f32$", ln)
        if m:
            env[m.group(1)] = np.where(env[m.group(2)].astype(bool), env[m.group(3)], env[m.group(4)]).astype(f32)
            return
        m = re.match(r"%(\w+) = kernel\.subgroup\.reduce<(addf|maxnumf)> %(\w+) : f32$", ln)
        if m:
            v = env[m.group(3)].astype(np.float64).reshape(-1, 32)
            r = v.sum(1) if m.group(2) == "addf" else v.max(1)
            env[m.group(1)] = np.repeat(r, 32).astype(f32)
            return
        m = re.match(r"%(\w+) = kernel\.workgroup\.reduce<(addf|maxnumf)> %(\w+) : f32$", ln)
        if m:
            v = env[m.group(3)].astype(np.float64)
            env[m.group(1)] = np.full(N, v.sum() if m.group(2) == "addf" else v.max(), f32)
            return
        m = re.match(r"%(\w+) = view\.load %(\w+)\[([^\]]+)\] : view<[^>]+> -> \w+$", ln)
        if m:
            arr, flat = self.mem(m.group(2), m.group(3).split(","), env)
            safe = np.where(mask, flat, 0)
            env[m.group(1)] = arr[np.clip(safe, 0, arr.size - 1)].copy()
            return
        m = re.match(r"view\.store %(\w+), %(\w+)\[([^\]]+)\] : ", ln)
        if m:
            arr, flat = self.mem(m.group(2), m.group(3).split(","), env)
            v = env[m.group(1)]
            for i in np.nonzero(mask)[0]:
                arr[flat[i]] = v[i]
            return
        m = re.match(r"%(\w+) = vector\.load %(\w+)\[([^\]]+)\] : view<[^>]+> -> vector<(\d+)x\w+>$", ln)
        if m:
            arr, flat = self.mem(m.group(2), m.group(3).split(","), env)
            n = int(m.group(4))
            safe = np.where(mask, flat, 0)
            env[m.group(1)] = arr[np.clip(safe[:, None] + np.arange(n), 0, arr.size - 1)].copy()
            return
        m = re.match(r"%(\w+) = vector\.(mulf|addf|subf) %(\w+), %(\w+) : vector<\d+xf32>$", ln)
        if m:
            a, b = env[m.group(3)].astype(f32), env[m.group(4)].astype(f32)
            env[m.group(1)] = {"mulf": a * b, "addf": a + b, "subf": a - b}[m.group(2)].astype(f32)
            return
        m = re.match(r"%(\w+) = vector\.splat %(\w+) : vector<(\d+)xf32>$", ln)
        if m:
            env[m.group(1)] = np.repeat(env[m.group(2)].astype(f32)[:, None], int(m.group(3)), 1)
            return
        m = re.match(r"vector\.store %(\w+), %(\w+)\[([^\]]+)\] : vector<(\d+)x\w+>, ", ln)
        if m:
            arr, flat = self.mem(m.group(2), m.group(3).split(","), env)
            v, n = env[m.group(1)], int(m.group(4))
            for i in np.nonzero(mask)[0]:
                arr[flat[i]:flat[i] + n] = v[i]
            return
        m = re.match(r"%(\w+), %(\w+) = kernel\.subgroup\.shuffle<xor> %(\w+), %(\w+), %(\w+) : f32, i32, i32$", ln)
        if m:
            v, x = env[m.group(3)], env[m.group(4)]
            lanes = np.arange(self.N)
            env[m.group(1)] = v[(lanes & ~31) | ((lanes & 31) ^ x)].astype(f32)
            env[m.group(2)] = np.zeros(self.N, np.int64)
            return
        m = re.match(r"%(\w+) = vector\.extf %(\w+) : vector<(\d+)x(bf16|f16)> to vector<\d+xf32>$", ln)
        if m:
            v = env[m.group(2)]
            env[m.group(1)] = ((v.astype(np.uint32) << 16).view(np.float32) if m.group(4) == "bf16" else v.astype(np.float32)).copy()
            return
        m = re.match(r"%(\w+) = vector\.slice %(\w+)\[(\d+)\] : vector<\d+x\w+> -> vector<(\d+)x\w+>$", ln)
        if m:
            o, n = int(m.group(3)), int(m.group(4))
            env[m.group(1)] = env[m.group(2)][:, o:o + n].copy()
            return
        m = re.match(r"%(\w+) = vector\.dotf<[^>]*> %(\w+), %(\w+), %(\w+) : vector<\d+xf32>, vector<\d+xf32>, f32$", ln)
        if m:
            a, b = env[m.group(2)].astype(np.float64), env[m.group(3)].astype(np.float64)
            env[m.group(1)] = ((a * b).sum(1) + env[m.group(4)]).astype(f32)
            return
        raise SystemExit(f"loomsim: unknown line {ln!r}")


def run(text, buffers, scalars, grid, wg_size):
    Sim(text, buffers, scalars, wg_size).run(grid)
