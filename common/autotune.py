#!/usr/bin/env python3
"""Gated search over a kit generator's knobs for one kernel family: a static stage on the CPU ranks the candidates that survive the
gates, a timed stage (GPU, in a window step) times the finalists through the HIP harness. Neither stage decides: a device-time
ranking only screens (at the 85 W cap a faster kernel can lower the clock of the whole step, HRX-like E351-E356); the finalist goes
to the candidate chain and the seeded agent bench.

search SPACE.json --out DIR: per candidate configuration, generate the program with the knob values (a package .cmd's recipe through
  kit/build.py, or a generator command), compile the family's kernels for gfx1151 (binary_gate.kernels: source closures, so a kernel
  whose closure an earlier candidate already compiled is not compiled again; binary_gate.measure with Loom's summary report) and lint
  them (kernel_lint.lint). A candidate is disqualified on: a generation or compile error; a family kernel missing; scratch above the
  base kernel's (base 0: any spill); a kernel_lint flag the base kernel lacked (a new family kernel: any flag); fewer waves per SIMD
  than the base kernel, unless a knob that differs from the base declares "trades_occupancy": "<why>" (a knob meant to buy registers
  with residency, e.g. a larger tile). A candidate whose family kernels all have the base's closures is a no-op (the knob did not
  reach the family); one identical to an earlier candidate is a duplicate. The base configuration is always evaluated first and
  must pass its own gates (else exit 2: nothing to compare against).
  Static score, lower first: the geometric mean over the family's kernels of (candidate / base) non-MMA instructions per MMA or dot
  in the hot loop's common path (kernel_lint other_per_op; the whole kernel's instruction count when it has no multiply loop),
  rounded to 0.01, then more waves per SIMD (the minimum over the family). Per-op work is what the loop issues per unit of math
  whatever the tile size, and waves hide latency; neither prices bandwidth, prefetch depth or the clock, which is why it screens.
  Verdict: `AUTOTUNE: N survivors, best <config>` (N excludes the base; best may be `base`) exit 0, `AUTOTUNE: NONE` exit 1;
  exit 2 on a refusal (a malformed space, a knob no generator reads, a base that fails, a family regex matching no base kernel,
  a measurement server running without --beside-measurement).
time DIR/result.json [--top N]: the timed stage (GPU; run by the step script). For the top N survivors and each kernel named in the
  space's timed.kernels: common/hsaco_run.py with the base object twice (the A/A pair) and the finalists, interleaved, the order
  rotated every round; outputs compared against the base in round 0 (a mismatch disqualifies); a finalist whose launch config or
  kernel arguments differ is timed in its own call per round (ratio against the same round's base: marked cross-call). Per
  finalist: the weighted geometric mean of its kernels' median-over-rounds ratios against the A/A spread. Writes DIR/timed.json;
  `AUTOTUNE TIMED (hypothesis): best <config>` exit 0 when one is faster beyond its A/A spread, `AUTOTUNE TIMED: NONE` exit 1.
step DIR/result.json --pkg NAME: print the window step's script (not queued): telemetry, the timed stage, then common/build.sh for
  the base recipe and the best configuration, common/binary_gate.py between them (--allow the family), then the space's chain.

Search space (JSON; unknown keys refused):
  {"family": "gemm256", "kernels": "REGEX over kernel export names",
   "kit": "qwen38-27b", "base_cmd": "/abs/WORK/pkg-x.cmd"          # a package's recipe (common/build.sh's PKG.cmd), or:
   "generator": {"argv": ["{python}", "{lsk}/kit/fixtures/x.py", "{out}/program/x.loom"], "sources": ["kit/fixtures/x.py"]},
   "base_env": {"K": "V"},                                         # generator mode only
   "knobs": {"GEMM256_KT": {"values": ["32", "64"], "trades_occupancy": "...", "note": "..."}, "GEMM256_FLATADDR": ["0", "1"]},
   "strategy": "grid" | "coordinate", "budget": 8,
   "timed": {"top": 3, "rounds": 3, "reps": 20, "chain": "bash .../candidate_check.sh {pkg}", "telemetry": "...",
             "kernels": {"KERNEL": {"workload": ["n=4096"], "buf": ["y=16384"], "scalar": [], "compare": ["y:f32"], "weight": 1}}}}
  A knob must be an environment variable a generator source reads with a literal name (os.environ.get / os.environ[...] / `in
  os.environ` / os.getenv; kit/**/*.py without tests and fixtures in .cmd mode, the generator's sources otherwise): anything else
  is refused. Values are strings (an integer is converted); the base's own value (the recipe's, or unset) is always a candidate.
usage: autotune.py search SPACE.json --out DIR [--jobs N] [--beside-measurement] [--loom-compile PATH]
       autotune.py time DIR/result.json [--top N] [--rounds R] [--reps N] [--timeout S]
       autotune.py step DIR/result.json --pkg NAME [--top N] > step.sh"""
import argparse
import ast
import concurrent.futures
import glob
import itertools
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LSK = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import binary_gate as bg  # noqa: E402

SPACE_KEYS = {"family", "kernels", "kit", "base_cmd", "generator", "base_env", "knobs", "strategy", "budget", "timed", "note"}
KNOB_KEYS = {"values", "trades_occupancy", "note"}
TIMED_KEYS = {"top", "rounds", "reps", "chain", "telemetry", "kernels"}
TKERNEL_KEYS = {"workload", "buf", "scalar", "compare", "weight"}
NAME = re.compile(r"[A-Z][A-Z0-9_]*$")



def lint_version():
    """the cache key's lint part: kernel_lint.py and loop_census.py (its disassembly) hashed, so a cached row.json never carries
    figures an older kernel_lint computed (r2 changed 'waves' from a fraction to Loom's integer tier)"""
    h = hashlib.sha256()
    for f in ("kernel_lint.py", "loop_census.py"):
        h.update(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f), "rb").read())
    return h.hexdigest()[:12]

class Refuse(Exception):
    """a usage or premise error: exit 2"""


# ------------------------------------------------------------------------------------------------------------- knobs
def env_reads(path):
    """environment variable names a Python source reads with a literal name; computed, aliased or templated names are not
    recognised (they never make a knob known), and comments or strings that merely quote a name do not count (AST, not text)"""
    out = set()
    def environ(n):
        return isinstance(n, ast.Attribute) and n.attr == "environ" and isinstance(n.value, ast.Name) and n.value.id in ("os", "_os")
    def lit(n):
        return n.value if isinstance(n, ast.Constant) and isinstance(n.value, str) else None
    for n in ast.walk(ast.parse(open(path).read(), path)):
        name = None
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.args:
            if n.func.attr == "get" and environ(n.func.value):
                name = lit(n.args[0])
            elif n.func.attr == "getenv" and isinstance(n.func.value, ast.Name) and n.func.value.id in ("os", "_os"):
                name = lit(n.args[0])
        elif isinstance(n, ast.Subscript) and environ(n.value):
            name = lit(n.slice)
        elif isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.In, ast.NotIn)) and environ(n.comparators[0]):
            name = lit(n.left)
        if name:
            out.add(name)
    return out


def generator_sources(space, kit_src):
    if "generator" in space:
        return [os.path.join(LSK, s) for s in space["generator"]["sources"]]
    return [f for f in glob.glob(os.path.join(kit_src, "**", "*.py"), recursive=True)
            if not os.path.basename(f).startswith("test_") and os.sep + "fixtures" + os.sep not in f]


# ------------------------------------------------------------------------------------------------------------- space
def load_space(path):
    try:
        s = json.load(open(path))
    except (OSError, ValueError) as e:
        raise Refuse(f"search space {path}: {e}")
    if not isinstance(s, dict):
        raise Refuse("search space: a JSON object")
    bad = set(s) - SPACE_KEYS
    if bad:
        raise Refuse(f"search space: unknown keys {sorted(bad)} (known: {sorted(SPACE_KEYS)})")
    for k in ("family", "kernels", "knobs"):
        if k not in s:
            raise Refuse(f"search space: '{k}' missing")
    try:
        re.compile(s["kernels"])
    except (re.error, TypeError) as e:
        raise Refuse(f"search space: kernels {s['kernels']!r}: {e}")
    if ("base_cmd" in s) == ("generator" in s):
        raise Refuse("search space: exactly one of base_cmd (a package recipe) or generator")
    if "base_cmd" in s and ("kit" not in s or "base_env" in s):
        raise Refuse("search space: base_cmd needs 'kit' and takes its base values from the .cmd (no base_env)")
    if "generator" in s:
        g = s["generator"]
        if not (isinstance(g, dict) and set(g) == {"argv", "sources"} and g["argv"] and g["sources"]):
            raise Refuse("search space: generator = {argv: [...], sources: [kit-relative .py files]}")
        s.setdefault("base_env", {})
    s["strategy"] = s.get("strategy", "grid")
    if s["strategy"] not in ("grid", "coordinate"):
        raise Refuse(f"search space: strategy {s['strategy']!r}: grid or coordinate")
    s["budget"] = s.get("budget", 16)
    if not isinstance(s["budget"], int) or s["budget"] < 1:
        raise Refuse("search space: budget, a positive integer (candidates generated, the base included)")
    if not isinstance(s["knobs"], dict) or not s["knobs"]:
        raise Refuse("search space: knobs, a non-empty object")
    knobs = {}
    for k, v in s["knobs"].items():
        if not NAME.match(k):
            raise Refuse(f"knob {k!r}: not an environment variable name")
        v = {"values": v} if isinstance(v, list) else v
        if not isinstance(v, dict) or set(v) - KNOB_KEYS or not isinstance(v.get("values"), list) or not v["values"]:
            raise Refuse(f"knob {k}: a list of values or {{values: [...], trades_occupancy: why, note}}")
        if any(isinstance(x, bool) or not isinstance(x, (str, int)) for x in v["values"]):
            raise Refuse(f"knob {k}: values are strings or integers ({v['values']})")
        if "trades_occupancy" in v and not (isinstance(v["trades_occupancy"], str) and v["trades_occupancy"].strip()):
            raise Refuse(f"knob {k}: trades_occupancy names why the knob may cost waves (a non-empty string)")
        knobs[k] = dict(v, values=list(dict.fromkeys(str(x) for x in v["values"])))
    s["knobs"] = knobs
    t = s.get("timed")
    if t is not None:
        if not isinstance(t, dict) or set(t) - TIMED_KEYS or not isinstance(t.get("kernels"), dict) or not t["kernels"]:
            raise Refuse(f"search space: timed = {{kernels: {{K: {sorted(TKERNEL_KEYS)}}}, {sorted(TIMED_KEYS - {'kernels'})}}}")
        for k, v in t["kernels"].items():
            if not re.search(s["kernels"], k):
                raise Refuse(f"timed kernel {k} is not in the family {s['kernels']!r}")
            if not isinstance(v, dict) or set(v) - TKERNEL_KEYS:
                raise Refuse(f"timed kernel {k}: keys {sorted(TKERNEL_KEYS)}")
    return s


def parse_cmd(text):
    """a package .cmd (`env K=V ... kit/build.py CONFIG PKG ARGS  # comment`) -> (env pairs, config, package, args)"""
    t = shlex.split(text.strip().split("\n")[0], comments=True)
    if not t or t[0] != "env":
        raise Refuse("base_cmd: not a common/build.sh PKG.cmd (`env K=V ... kit/build.py CONFIG PKG ...`)")
    i, env = 1, []
    while i < len(t) and re.match(r"[A-Za-z_]\w*=", t[i]):
        env.append(tuple(t[i].split("=", 1)))
        i += 1
    if i + 2 >= len(t) or not t[i].endswith("build.py"):
        raise Refuse(f"base_cmd: expected kit/build.py CONFIG PKG after the switches, got {t[i:i + 3]}")
    return env, t[i + 1], t[i + 2], t[i + 3:]


def machine(kit):
    """the environment common/build.sh builds in (common/env.sh lsk_load: the machine env file and the kit manifest, exported),
    and MEASURING: a measurement server of ours that runs"""
    sh = (f'. "{HERE}/env.sh"; lsk_load {shlex.quote(kit)} >/dev/null || exit 3; '
          'm=$(lsk_measuring) && export LSK_AUTOTUNE_MEASURING="$m"; env -0')
    r = subprocess.run(["bash", "-c", sh], capture_output=True)
    if r.returncode:
        raise Refuse(f"kit {kit}: common/env.sh lsk_load failed: {r.stderr.decode(errors='replace').strip()[-200:]}")
    e = dict(x.split("=", 1) for x in r.stdout.decode(errors="replace").split("\0") if "=" in x)
    e["MEASURING"] = e.pop("LSK_AUTOTUNE_MEASURING", "")
    return e


class Recipe:
    """how one configuration's program is generated: base values, and run(values, out) -> error text or None"""

    def __init__(self, space, beside=False):
        self.space, self.knobs = space, space["knobs"]
        if "base_cmd" in space:
            self.m = machine(space["kit"])
            if self.m.get("MEASURING") and not beside:
                raise Refuse(f"a measurement server runs ({self.m['MEASURING']}): a build beside it skews it (common/build.sh); "
                             "--beside-measurement to run anyway (niced)")
            try:
                self.cmd_text = open(space["base_cmd"]).read()
            except OSError as e:
                raise Refuse(f"base_cmd: {e}")
            env, cfg, self.base_pkg, self.args = parse_cmd(self.cmd_text)
            self.base_env = dict(env)
            work = self.m["WORK"]
            self.cfg = cfg if os.path.isabs(cfg) else next(
                (p for p in (os.path.join(LSK, cfg), os.path.join(work, cfg)) if os.path.exists(p)), os.path.join(work, cfg))
            self.kit_src = self.m.get("KIT_SRC") or os.path.join(LSK, "kit")
        else:
            self.m, self.base_env, self.kit_src = {}, {k: str(v) for k, v in space["base_env"].items()}, os.path.join(LSK, "kit")
        self.base = {k: self.base_env.get(k) for k in self.knobs}           # None: unset (the generator's default)

    def env(self, values):
        src = {k: v for k, v in self.m.items() if k != "MEASURING"} if self.m else os.environ
        e = {k: v for k, v in src.items() if k not in self.knobs}
        e.update(self.base_env)
        e.update({k: v for k, v in values.items() if v is not None})
        if self.m.get("VENV"):
            e["PATH"] = os.path.join(self.m["VENV"], "bin") + os.pathsep + e.get("PATH", "")
        return e

    def run(self, values, out, timeout=900):
        os.makedirs(out, exist_ok=True)
        if "base_cmd" in self.space:
            pkg = os.path.join(out, "pkg")
            argv = [self.m.get("PYTHON") or sys.executable, os.path.join(self.kit_src, "build.py"), self.cfg, pkg] + self.args
            cwd = self.m["WORK"]
        else:
            pkg = out
            sub = {"{python}": sys.executable, "{lsk}": LSK, "{out}": out}
            argv = [re.sub("|".join(map(re.escape, sub)), lambda m: sub[m.group(0)], a) for a in self.space["generator"]["argv"]]
            cwd = out
        with open(os.path.join(out, "gen.cmd"), "w") as f:
            f.write(" ".join(f"{k}={v}" for k, v in sorted(values.items()) if v is not None) + " " + shlex.join(argv) + "\n")
        try:
            r = subprocess.run(argv, cwd=cwd, env=self.env(values), capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return pkg, f"generation timed out after {timeout} s"
        open(os.path.join(out, "gen.log"), "w").write(r.stdout + r.stderr)
        if r.returncode:
            tail = (r.stderr or r.stdout).strip().split("\n")
            return pkg, f"generation exit {r.returncode}: {tail[-1][:200] if tail else ''}"
        if "base_cmd" in self.space:                                        # keep the programs; the rest is the weights' manifest
            for x in os.listdir(pkg):
                if x != "program":
                    p = os.path.join(pkg, x)
                    shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        return pkg, None


# ------------------------------------------------------------------------------------------------------------- judging
def judge(base_rows, cand_rows, waiver=False):
    """why a candidate is disqualified (empty: it survives). Rows: kernel -> metrics (binary_gate.measure + kernel_lint.lint:
    scratch, flags, waves, other_per_op, instr) or {error}. WAIVER: a differing knob declares trades_occupancy. Pure."""
    why = []
    for k in sorted(set(base_rows) - set(cand_rows)):
        why.append(f"{k}: missing")
    base_waves = [b["waves"] for b in base_rows.values() if b.get("waves") is not None]
    for k, c in sorted(cand_rows.items()):
        if "error" in c:
            why.append(f"{k}: compile error: {c['error']}")
            continue
        b = base_rows.get(k)
        bs = (b or {}).get("scratch") or 0
        if (c.get("scratch") or 0) > bs:
            why.append(f"{k}: scratch {bs} -> {c['scratch']} B")
        new = sorted(set(c.get("flags") or ()) - set((b or {}).get("flags") or ()))
        if new:
            why.append(f"{k}: new lint flag {','.join(new)}")
        bw = b.get("waves") if b else (min(base_waves) if base_waves else None)
        if c.get("waves") is None:
            why.append(f"{k}: no waves in the compile report")
        elif bw is not None and c["waves"] < bw and not waiver:
            why.append(f"{k}: waves {bw} -> {c['waves']} (no knob declares trades_occupancy)")
    return why


def score(base_rows, cand_rows):
    """(rank key, per-op geomean, min waves): lower key first"""
    logs = []
    for k, c in cand_rows.items():
        b = base_rows.get(k)
        if not b or "error" in c or "error" in b:
            continue
        x = "other_per_op" if b.get("other_per_op") and c.get("other_per_op") else "instr"
        if b.get(x) and c.get(x):
            logs.append(math.log(c[x] / b[x]))
    g = math.exp(sum(logs) / len(logs)) if logs else 1.0
    w = min((c["waves"] for c in cand_rows.values() if c.get("waves") is not None), default=0)
    return [round(g, 2), -w, round(g, 4)], round(g, 4), w


def label(values, base):
    d = [f"{k}={v}" for k, v in values.items() if v != base[k]]
    return " ".join(d) if d else "base"


def configs_grid(knobs, base):
    axes = [[base[k]] + [v for v in knobs[k]["values"] if v != base[k]] for k in knobs]
    for combo in itertools.product(*axes):
        yield dict(zip(knobs, combo))


# ------------------------------------------------------------------------------------------------------------- search
class Search:
    def __init__(self, space, out, jobs=2, lc=None, beside=False):
        self.space, self.out, self.jobs = space, os.path.abspath(out), jobs
        self.rx = re.compile(space["kernels"])
        self.recipe = Recipe(space, beside)
        known = set()
        for f in generator_sources(space, self.recipe.kit_src):
            known |= env_reads(f)
        unknown = sorted(set(space["knobs"]) - known)
        if unknown:
            raise Refuse(f"unknown knob(s) {unknown}: no generator source reads them by a literal name "
                         f"({len(known)} names read; sources: {'kit/**/*.py' if 'base_cmd' in space else space['generator']['sources']})")
        if lc is None:
            from kernel_lint import loom_compile
            lc = loom_compile()
        self.lc = lc
        self.cands, self.seen = [], {}
        os.makedirs(os.path.join(self.out, "cache"), exist_ok=True)

    def compile_one(self, prog, k, fp):
        """the kernel's metrics, compiled once per source closure (the same closure: the same code, binary_gate's premise)"""
        import kernel_lint
        d = os.path.join(self.out, "cache", f"{k}.{fp[:16]}.{lint_version()}")   # a kernel_lint change re-lints (review r1 F6)
        meta = os.path.join(d, "row.json")
        if os.path.exists(meta):
            return json.load(open(meta))
        os.makedirs(d, exist_ok=True)
        row = bg.measure(self.lc, prog, k, d, report=True)
        if "error" not in row:
            hs = os.path.join(d, f"{k}.hsaco")
            lint = kernel_lint.lint(hs)
            row.update({x: lint.get(x) for x in ("flags", "waves", "other_per_op", "instr", "loop", "where", "to_next")})
            row["hsaco"], row["prog"], row["fp"] = hs, prog, fp
        json.dump(row, open(meta, "w"), indent=1)
        return row

    def evaluate(self, values):
        key = json.dumps(values, sort_keys=True)
        if key in self.seen:
            return self.seen[key]
        cid = f"c{len(self.cands):02d}"
        c = {"id": cid, "values": values, "label": label(values, self.recipe.base), "why": [], "status": "ok", "rows": {}}
        self.cands.append(c)
        self.seen[key] = c
        pkg, err = self.recipe.run(values, os.path.join(self.out, cid))
        if err:
            c.update(status="DQ", why=[err])
            return c
        try:
            ks = bg.kernels(pkg)
        except SystemExit as e:
            c.update(status="DQ", why=[f"program: {e}"])
            return c
        fam = {k: v for k, v in ks.items() if self.rx.search(k)}
        c["fps"] = {k: fp for k, (_, fp) in fam.items()}
        c["all_fps"] = {k: fp for k, (_, fp) in ks.items()}
        with concurrent.futures.ThreadPoolExecutor(self.jobs) as ex:
            rows = list(ex.map(lambda kv: self.compile_one(kv[1][0], kv[0], kv[1][1]), sorted(fam.items())))
        c["rows"] = dict(zip(sorted(fam), rows))
        if c is self.cands[0]:
            return c
        base = self.cands[0]
        c["outside"] = sorted(k for k in set(base["all_fps"]) | set(c["all_fps"])
                              if not self.rx.search(k) and base["all_fps"].get(k) != c["all_fps"].get(k))
        if c["fps"] == base["fps"]:
            c["status"] = "no-op"
            return c
        dup = next((o for o in self.cands[1:-1] if o.get("fps") == c["fps"]), None)
        if dup:
            c.update(status="dup", why=[f"same family code as {dup['id']}"])
            return c
        waiver = any(values[k] != self.recipe.base[k] and "trades_occupancy" in self.space["knobs"][k] for k in values)
        c["why"] = judge(base["rows"], c["rows"], waiver)
        c["status"] = "DQ" if c["why"] else "ok"
        c["key"], c["perop"], c["minwaves"] = score(base["rows"], c["rows"])
        return c

    def run(self):
        knobs, base = self.space["knobs"], self.recipe.base
        b = self.evaluate(dict(base))
        if b["status"] != "ok" or not b["rows"]:
            raise Refuse(f"the base configuration fails: {b['why'] or 'no kernel matches ' + repr(self.space['kernels'])}")
        bw = judge(b["rows"], b["rows"])
        if bw:
            raise Refuse(f"the base fails its own gates: {bw}")
        b["key"], b["perop"], b["minwaves"] = score(b["rows"], b["rows"])
        budget = self.space["budget"]
        if self.space["strategy"] == "grid":
            for v in itertools.islice(configs_grid(knobs, base), 1, budget):
                self.evaluate(v)
        else:                                                               # coordinate: one knob at a time from the best so far
            cur = b
            for _ in range(3):
                moved = False
                for k in knobs:
                    for v in [base[k]] + knobs[k]["values"]:
                        if len(self.cands) >= budget:
                            break
                        if v != cur["values"][k]:
                            self.evaluate(dict(cur["values"], **{k: v}))
                    ok = [c for c in self.cands if c["status"] == "ok"]
                    best = min(ok, key=lambda c: c["key"])
                    if best is not cur:
                        cur, moved = best, True
                if not moved or len(self.cands) >= budget:
                    break
        return self.cands


def knob_reach(cands, base):
    """knob -> True (a pair of evaluated candidates differing in that knob alone has different family code), False (pairs exist,
    none differs: the knob does not reach the family), None (no such pair was evaluated)"""
    by = {json.dumps(c["values"], sort_keys=True): c for c in cands if c.get("fps") is not None}
    out = {}
    for k in base:
        pairs = [(c, by.get(json.dumps(dict(c["values"], **{k: base[k]}), sort_keys=True))) for c in by.values() if c["values"][k] != base[k]]
        pairs = [(c, o) for c, o in pairs if o is not None]
        out[k] = any(c["fps"] != o["fps"] for c, o in pairs) if pairs else None
    return out


def table(cands):
    lines = [f"{'id':4s} {'status':6s} {'per-op':>7s} {'waves':>5s} {'vgpr':>5s} {'scr':>4s} {'outside':>7s}  config / why"]
    for c in cands:
        rows = [r for r in c["rows"].values() if "error" not in r]
        vg = max((r.get("vgpr") or 0 for r in rows), default=0)
        sc = max((r.get("scratch") or 0 for r in rows), default=0)
        lines.append(f"{c['id']:4s} {c['status']:6s} {c.get('perop', '-')!s:>7s} {c.get('minwaves', '-')!s:>5s} {vg:5d} {sc:4d} "
                     f"{len(c.get('outside') or []):7d}  {c['label']}" + ("".join(f"\n{'':44s}{w}" for w in c["why"])))
    return "\n".join(lines)


def cmd_search(a):
    space = load_space(a.space)
    os.nice(15)                                                             # CPU work beside others' runs: never ahead of them
    s = Search(space, a.out, a.jobs, a.loom_compile, a.beside_measurement)
    cands = s.run()
    base = cands[0]
    surv = sorted([c for c in cands if c["status"] == "ok"], key=lambda c: c["key"])
    print(table(cands))
    for k, v in knob_reach(cands, base["values"]).items():
        if v is False:
            print(f"KNOB NO-OP: {k}: no value evaluated changed a family kernel (the knob does not reach {space['kernels']!r})")
        elif v is None:
            print(f"KNOB UNTESTED: {k}: no evaluated pair differs in {k} alone (raise the budget)")
    outside = {}
    for c in cands:
        for k in c.get("outside") or ():
            outside.setdefault(k, []).append(c["id"])
    for k, ids in sorted(outside.items()):
        print(f"OUTSIDE: {k} (not in the family) changes in {' '.join(ids)}: reported, not judged; the step's binary_gate fails on it")
    commit = subprocess.run(["git", "-C", LSK, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", LSK, "status", "--porcelain", "--untracked-files=no", "--", "kit"],
                           capture_output=True, text=True).stdout.count("\n")
    res = {"space": space, "space_path": os.path.abspath(a.space), "base": base["values"], "lsk": f"{commit} + {dirty} uncommitted kit files",
           "candidates": cands, "ranking": [c["id"] for c in surv]}
    json.dump(res, open(os.path.join(s.out, "result.json"), "w"), indent=1)
    n = sum(1 for c in surv if c is not base)
    if n == 0:
        print("AUTOTUNE: NONE (no candidate besides the base survives the static gates)")
        return 1
    print(f"AUTOTUNE: {n} survivors, best {surv[0]['label']} ({s.out}/result.json; static screen only)")
    return 0


# ------------------------------------------------------------------------------------------------------------- timed stage
HSACO_RUN = os.path.join(HERE, "hsaco_run.py")   # the GPU process boundary (kit/test_autotune.py stands a stub in its place)
MEDIAN = re.compile(r"^(\S+)\s+median\s+([\d.]+) ms", re.M)
MISMATCH = re.compile(r"^compare \S+: (\S+) != (\S+):", re.M)


def finalists(res, top):
    by = {c["id"]: c for c in res["candidates"]}
    return [by[i] for i in res["ranking"] if i != "c00"][:top]


def hsaco_call(objs, k, prog, spec, reps, compare, timeout):
    """one interleaved hsaco_run.py call: (medians by label, mismatching labels, error or None)"""
    argv = [sys.executable, HSACO_RUN, "--kernel", k, "--loom", prog, "--reps", str(reps)]
    for opt in ("workload", "buf", "scalar"):
        argv += [x for v in spec.get(opt, []) for x in (f"--{opt}", v)]
    if compare:
        argv += [x for v in spec.get("compare", []) for x in ("--compare", v)]
    argv += [f"{lab}={p}" for lab, p in objs]
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {}, set(), f"timeout after {timeout} s"
    med = {m.group(1): float(m.group(2)) for m in MEDIAN.finditer(r.stdout)}
    bad = {x for m in MISMATCH.finditer(r.stdout) for x in m.groups()}
    if r.returncode and not bad:
        return med, bad, (r.stderr or r.stdout).strip().split("\n")[-1][:200]
    return med, bad, None


def group_key(row, k):
    import hsaco_run
    _, body, _ = hsaco_run.loom_kernel(open(row["prog"]).read(), k)
    m = hsaco_run.kernel_meta(row["hsaco"], k)
    return json.dumps([body, [(x.get(".offset"), x.get(".size"), x.get(".value_kind")) for x in m.get(".args", [])]])


def cmd_time(a):
    res = json.load(open(a.result))
    space, out = res["space"], os.path.dirname(os.path.abspath(a.result))
    t = space.get("timed") or {}
    if not t:
        raise Refuse("the space has no 'timed' section: nothing to time")
    top, rounds, reps = a.top or t.get("top", 3), a.rounds or t.get("rounds", 3), a.reps or t.get("reps", 20)
    base = res["candidates"][0]
    fin = finalists(res, top)
    if not fin:
        print("AUTOTUNE TIMED: NONE (no static survivor)")
        return 1
    ratios = {c["id"]: {} for c in fin}
    notes = {c["id"]: [] for c in fin}
    dq = {c["id"]: [] for c in fin}
    spread = {}
    for k, spec in sorted(t["kernels"].items()):
        b = base["rows"].get(k)
        if not b or "error" in b:
            raise Refuse(f"timed kernel {k}: not compiled in the base")
        todo = []
        for c in fin:
            r = c["rows"].get(k)
            if r is None or "error" in r:
                dq[c["id"]].append(f"{k}: not compiled")
            elif r["fp"] == b["fp"]:
                ratios[c["id"]][k] = [1.0]                                   # the base's own code
            else:
                todo.append((c["id"], r))
        if not todo:
            continue
        bkey = group_key(b, k)
        groups = {}
        for cid, r in todo:
            groups.setdefault(group_key(r, k), []).append((cid, r))
        same = groups.pop(bkey, [])
        for cid, _ in [x for g in groups.values() for x in g]:
            notes[cid].append(f"{k}: launch config or arguments differ: own call per round (cross-call ratio; outputs not compared)")
        spread[k] = []
        for rnd in range(rounds):
            objs = [("base", b["hsaco"]), ("aa", b["hsaco"])] + [(cid, r["hsaco"]) for cid, r in same]
            if rnd:
                objs = objs[rnd % len(objs):] + objs[:rnd % len(objs)]
            calls = [(objs, b["prog"], rnd == 0)] + [([(cid, r["hsaco"]), (cid + "_aa", r["hsaco"])], r["prog"], False)
                                                      for g in groups.values() for cid, r in g]
            calls = calls[rnd % len(calls):] + calls[:rnd % len(calls)]
            med = {}
            for o, prog, cmp_ in calls:
                m, bad, err = hsaco_call(o, k, prog, spec, reps, cmp_, a.timeout)
                if err and "power level" in err:
                    raise Refuse(f"hsaco_run refused: {err}")
                for lab, _ in o:
                    cid = lab[:-3] if lab.endswith("_aa") else lab
                    if err and cid in dq:
                        dq[cid].append(f"{k}: harness error: {err}")
                for x in bad:
                    if x in dq:
                        dq[x].append(f"{k}: output differs from the base")
                med.update(m)
            if "base" not in med or "aa" not in med:
                raise Refuse(f"{k}: the base did not time in round {rnd} ({sorted(med)})")
            spread[k].append(abs(med["aa"] / med["base"] - 1))
            for cid, _ in todo:
                if cid + "_aa" in med and cid in med:                       # a cross-call finalist's own A/A pair
                    spread.setdefault((k, cid), []).append(abs(med[cid + "_aa"] / med[cid] - 1))
            for cid, _ in todo:
                if cid in med:
                    ratios[cid].setdefault(k, []).append(med[cid] / med["base"])
    rows, best = [], None
    for c in fin:
        cid, w, lg = c["id"], 0.0, 0.0
        per = {}
        for k, spec in t["kernels"].items():
            rs = sorted(ratios[cid].get(k, []))
            if not rs:
                continue
            per[k] = rs[len(rs) // 2]
            wt = float(spec.get("weight", 1))
            w, lg = w + wt, lg + wt * math.log(per[k])
        agg = math.exp(lg / w) if w else None
        noise = max((max(s) for x, s in spread.items() if s and (x in per or (isinstance(x, tuple) and x[1] == cid))), default=0.0)
        verdict = ("DQ" if dq[cid] else "untimed" if agg is None else "faster" if agg < 1 - noise
                   else "slower" if agg > 1 + noise else "within A/A")
        rows.append({"id": cid, "label": c["label"], "values": c["values"], "ratio": agg, "per_kernel": per, "aa_spread": noise,
                     "verdict": verdict, "why": sorted(set(dq[cid])), "notes": sorted(set(notes[cid]))})
        if verdict == "faster" and (best is None or agg < best["ratio"]):
            best = rows[-1]
    for r in rows:
        print(f"  {r['id']} {r['verdict']:10s} ratio {r['ratio'] if r['ratio'] is None else round(r['ratio'], 4)!s:>7s} "
              f"(A/A {r['aa_spread']:.4f})  {r['label']}" + "".join(f"\n      {x}" for x in r["why"] + r["notes"]))
    bv = res["base"]
    out_doc = {"hypothesis": "harness device time: diagnoses and screens; only the seeded served bench decides (rules section 5)",
               "rounds": rounds, "reps": reps, "rows": rows, "best": best and best["id"],
               "best_env": best and " ".join(f"{k}={v}" for k, v in best["values"].items() if v != bv[k])}
    json.dump(out_doc, open(os.path.join(out, "timed.json"), "w"), indent=1)
    if not best:
        print("AUTOTUNE TIMED: NONE (no finalist faster than the base beyond the A/A spread)")
        return 1
    print(f"AUTOTUNE TIMED (hypothesis): best {best['label']} ratio {best['ratio']:.4f} vs A/A {best['aa_spread']:.4f}")
    return 0


# ------------------------------------------------------------------------------------------------------------- step script
def emit_step(res, result_path, pkg, top=None):
    """the window step's bash script (text): telemetry, timed stage, two builds, binary gate, the chain"""
    space = res["space"]
    if "base_cmd" not in space:
        raise Refuse("step: a generator-mode space has no package recipe to build")
    t = space.get("timed") or {}
    if not t:
        raise Refuse("step: the space has no 'timed' section")
    env, cfg, _, args = parse_cmd(open(space["base_cmd"]).read())
    kit = space["kit"]
    pre = f"kits/{kit}/"
    kcfg = cfg[len(pre):] if cfg.startswith(pre) else os.path.abspath(cfg) if os.path.isabs(cfg) else cfg
    for k, v in env:
        if not re.fullmatch(r"[^\s'\"$`\\]*", v):
            raise Refuse(f"step: base switch {k}={v!r} needs quoting; build it by hand")
    base_extra = " ".join(f"{k}={v}" for k, v in env)
    q = shlex.quote
    tele = t.get("telemetry")
    chain = t.get("chain")
    lines = [
        "#!/bin/bash",
        f"# autotune step: family {space['family']} ({space['kernels']}), generated by common/autotune.py step from {result_path}.",
        "# Not queued. Every harness ratio is a hypothesis: device time screens (at the 85 W cap a faster kernel can lower the",
        "# clock of the whole step); the candidate goes forward on the seeded agent bench only. The base is rebuilt from the same kit",
        "# tree as the candidate (kit.env's KIT_RECIPE, then the base .cmd's switches), so the binary gate compares like with like.",
        "set -u -o pipefail",
        f"LSK={q(LSK)}; RES={q(os.path.abspath(result_path))}; OUT=$(dirname \"$RES\")",
        f'. "$LSK/common/env.sh"; lsk_load {q(kit)}',
        f"BASE={q(pkg + '-base')}; CAND={q(pkg)}",
        "# run in a window with the GPU free (the window runner stops and restores live); no build or server beside the timing",
        'bash "$LSK/common/wait_quiet.sh" || { echo "AUTOTUNE STEP: not quiet"; exit 1; }',
    ]
    if tele:
        lines.append(f"{tele} > \"$OUT/telemetry.before.txt\" 2>&1")
    lines += [
        f'python3 "$LSK/common/autotune.py" time "$RES"' + (f" --top {int(top)}" if top else "")
        + ' | tee "$OUT/timed.txt"; rc=${PIPESTATUS[0]}',
    ]
    if tele:
        lines.append(f"{tele} > \"$OUT/telemetry.after.txt\" 2>&1")
    lines += [
        '[ "$rc" -eq 0 ] || { echo "AUTOTUNE STEP: stop at the timed stage (exit $rc)"; exit "$rc"; }',
        "BEST=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[\"best_env\"])' \"$OUT/timed.json\")",
        '[ -n "$BEST" ] || { echo "AUTOTUNE STEP: no best configuration"; exit 1; }',
        f"export KIT_BUILD_ARGS={q(' '.join(args))} KIT_CONFIG={q(kcfg)}",
        f"EXTRA={q(base_extra)} bash \"$LSK/common/build.sh\" {q(kit)} \"$BASE\" || exit 1",
        f"EXTRA={q(base_extra)}\" $BEST\" bash \"$LSK/common/build.sh\" {q(kit)} \"$CAND\" || exit 1",
        f'python3 "$LSK/common/binary_gate.py" "$WORK/$BASE" "$WORK/$CAND" --allow {q(space["kernels"])} || exit 1',
    ]
    if chain:
        lines.append(chain.replace("{pkg}", '"$WORK/$CAND"').replace("{base}", '"$WORK/$BASE"'))
    else:
        lines.append('echo "AUTOTUNE STEP: $CAND built and gated; no chain in the space: hand it to the candidate chain"')
    return "\n".join(lines) + "\n"


def main():
    global HSACO_RUN
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("space")
    s.add_argument("--out", required=True)
    s.add_argument("--jobs", type=int, default=2)
    s.add_argument("--beside-measurement", action="store_true")
    s.add_argument("--loom-compile")
    t = sub.add_parser("time")
    t.add_argument("result")
    t.add_argument("--top", type=int)
    t.add_argument("--rounds", type=int)
    t.add_argument("--reps", type=int)
    t.add_argument("--timeout", type=float, default=300)
    t.add_argument("--hsaco-run", default=HSACO_RUN, help="the harness (default common/hsaco_run.py)")
    p = sub.add_parser("step")
    p.add_argument("result")
    p.add_argument("--pkg", required=True)
    p.add_argument("--top", type=int)
    a = ap.parse_args()
    try:
        if a.cmd == "search":
            sys.exit(cmd_search(a))
        if a.cmd == "time":
            HSACO_RUN = a.hsaco_run
            sys.exit(cmd_time(a))
        sys.stdout.write(emit_step(json.load(open(a.result)), a.result, a.pkg, a.top))
    except Refuse as e:
        print(f"AUTOTUNE: REFUSED: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
