#!/usr/bin/env python3
"""Price static lint hits with served time and hardware counters: per served kernel, join four sources that answer separately
(HRX-like T4, user 11-25): the served per-dispatch events (time share per kernel key and phase), common/kernel_lint.py's flags and
notes (imported read-only), the Loom compile report (limiting resource, residency tier, VGPRs) and rocprof counters of the served
code object launched through HIP (common/hsaco_run.py under rocprof-compute or common/counters.sh). A lint hit is a prompt to read;
this asks whether the pattern it names shows in the counters, and how much served time it could hold.

Inputs (all arguments; no model constants):
  --codes DIR     code objects + Loom summary reports (KERNEL.hsaco + KERNEL.json), e.g. `kernel_lint.py --package PKG --out DIR`
  --package PKG   optional: the package's program texts (kernel_lint's maskedv note)
  --events RUN    a served profile run (events.jsonl + one *.req.json: common/events_summary.py's loader)
  --lanes FILE    the kit's lanes.json (prefill_marker, lanes): the prefill / decode split, as events_summary.classify does it
  --counters SPEC a counter workload (repeatable): DIR[:hsaco=PATH][:level=auto|profile_standard]. DIR is a rocprof-compute
                  workload (results_pmc_perf_*.csv[.gz]) or a counters.sh output (*_counter_collection.csv).
  --device FILE   common/devices/<gfx>.json: the DRAM read ceiling of its default profile (serial's bandwidth guard), and the
                  hardware the lane floors of (d) are computed at (--profile, default its default_profile)
  --noise-floor F the kit's noise_floor.json ({lane substring: pct}, events_summary's format): (d)'s screen per lane
  --top N         (d): levers ranked and sent to amdahl.py per phase (default 15); --amdahl-out PATH keeps amdahl's input JSON

Counters attach to a served kernel only when the workload's code object is byte-identical to the served compile (sha256 of
--codes/KERNEL.hsaco). The code object is the one hsaco path in the workload's log.txt `Command:` line, or hsaco= in the spec;
none, several, or a different one: the workload is listed and not used (a candidate's counters under the live kernel's name,
q184's 'half' flash2, must not price the live kernel). Power level: clock.txt beside or above the workload ('sclk at LEVEL:'),
or level= in the spec; wait shares and GB/s count only at `auto` (the served level): at profile_standard q171's matvec waited on
counters 16 % of wave time and at auto 75 % (q171 vs q171b), the decode matvecs turn compute-bound there (rules 5, E300).
Bank-conflict % is a ratio of counts (E427: counts agree across levels) and counts at any level.
Saturation: common/counter_saturation.check on the workload's raw rows; any figure built on a saturated counter is 'saturated'
and unused (q171 / q171b flash2: SQ_WAVE_CYCLES = 20 x 2^32, every wait share 100 %, E425).

Wait shares (each SQ_WAIT_* / SQ_WAVE_CYCLES, no factor: devices/gfx1151.md): vmem CNT_VMVS, lgkm CNT_LGKM, cnt CNT_ANY (all
counters; rocprof-compute collects this, not the split), alu DELAY_ALU, barrier BARRIER, issue INST_ANY (ready, not issued),
ifetch IFETCH, any ANY. other = any - (cnt or vmem + lgkm) - barrier - ifetch - alu (derived: dependency waits not on a counter,
ALU dependency when DELAY_ALU is not collected). The dominant wait is the largest of vmem / lgkm (or cnt), alu (or other),
barrier, issue, ifetch. A missing counter is 'unknown', never zero.

Backing, per lint rule (a workload backs a hit when its figure passes; a kernel's hit is backed when every usable workload backs
it, so two shapes of one kernel must agree; 'mixed' counts as not backed). The matching share prices a hit:
  rule                         figure (matching share)              backed when             why this bar
  serial, scratch, reread      mem wait: vmem, else cnt              >= 25 % of wave cycles  a quarter of wave time on load counters:
                                                                                             above the WMMA-issue-bound f16 GEMMs (14-15 %,
                                                                                             E61 / E198 say issue, not loads) with margin;
                                                                                             a pure streaming read waits 96.5 % (q98)
   + serial only               MemReadBytes GB/s                     < 70 % of the device's  a bandwidth-bound kernel waits on vmcnt because
                                                                     DRAM read ceiling       the bus is full, not because loads are serial
                                                                                             (q171b's matvec: 75 % cnt at 92 % of 240 GB/s)
  ldsbank                      measured conflict % (SQC_LDS_BANK_    >= 2.0 %                the lint's own model bar (LDSBANK), so model and
                               CONFLICT / SQC_LDS_IDX_ACTIVE);                               measurement are judged on one threshold
                               price share = lds wait (lgkm, else
                               cnt) x conflict fraction
  copies, select, repack,      issue wait (INST_ANY)                 >= 25 % of wave cycles  extra instructions cost time only where waves
  join, addr64, outlier                                                                      wait to issue; the bandwidth-bound matvec
                                                                                             (10.5 %) and the latency-bound vsplit (16 %,
                                                                                             E435) sit below
  maskedv, tiny, regress:*     not priced: a correctness shape (E444), a launch gap a one-kernel harness cannot see, a baseline diff
The thresholds were set by the author from controls whose bound is known by other evidence; the precision figure is backed hits
over hits with usable counters, first pass, author-judged, with its n: PRINCIPLES asks for an outside sample before it is trusted.
A price is served share x matching share: the share of served device time the flagged pattern could hold (an upper bound on what
fixing it returns; a ranking, not a prediction).
(d) levers (user 11-25: 'make the linter rank, not just flag'): per served kernel and lane, a lever for each priced lint rule, or
an 'unnamed' one for a kernel without one, bounded by min(headroom attribution, pattern price). Headroom = the lane's served time
minus its floor (ceiling.lane_floor at --device's profile, the floor amdahl's to_floor uses); the kernel's attribution is its
share of the lane's served time x that headroom: an attribution, not a measurement. Two ranked lists, decode then prefill, each
by the bound as % of that phase's served time (user 11-25: 'two separate ranks'), top N each. Only floor-backed levers are ranked;
a lane without a floor (judged against 0, so its bound is the whole kernel time) is listed after each list, never screened. The
ranked levers go to common/amdahl.py as one phase per lane (wall = the lane's served ms, screen = its noise floor, one
overlap_group per kernel), and a lever below its lane's screen is 'not worth building' (rules 2: price before building). The
noise floors' false-pass sweep is still owed (kit NOTES.md): 'clears the screen' is provisional and the report says so. (d2) lists
occupancy levers unpriced (rules 1: occupancy is latency hiding, not a throughput model): every resource the compile report's
residency_constraints mark limiting with its reduction to the next tier (flash2 is held by VGPR and LDS at once: both must
move), beside the kernel's load-counter wait share at the served level.
usage: lint_counters.py --codes DIR --events RUN --lanes LANES.json [--package PKG] [--counters SPEC ...] [--device DEV.json]
       [--jobs N] [--profile P] [--noise-floor FILE] [--top N] [--amdahl-out PATH]"""
import argparse
import collections
import concurrent.futures
import csv
import glob
import gzip
import hashlib
import json
import os
import re
import sys

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import ceiling  # noqa: E402
import counter_saturation  # noqa: E402
import events_summary  # noqa: E402

MEM_MIN = 25.0       # % of wave cycles on load counters (serial, scratch, reread)
BW_MAX = 0.70        # serial: GB/s below this fraction of the device ceiling (else bandwidth-bound)
BANK_MIN = 2.0       # % measured LDS bank conflicts (kernel_lint.LDSBANK)
ISSUE_MIN = 25.0     # % of wave cycles waiting to issue (instruction-count rules)
RULES = {"serial": "mem", "scratch": "mem", "reread": "mem", "ldsbank": "bank",
         "copies": "issue", "select": "issue", "repack": "issue", "join": "issue", "addr64": "issue", "outlier": "issue"}
UNPRICED = {"maskedv": "correctness (E444)", "tiny": "launch gaps: not visible in a one-kernel harness"}
WAITS = {"vmem": "SQ_WAIT_CNT_VMVS", "lgkm": "SQ_WAIT_CNT_LGKM", "cnt": "SQ_WAIT_CNT_ANY", "alu": "SQ_WAIT_DELAY_ALU",
         "barrier": "SQ_WAIT_BARRIER", "issue": "SQ_WAIT_INST_ANY", "ifetch": "SQ_WAIT_IFETCH", "any": "SQ_WAIT_ANY"}
EA = (32, 64, 96, 128)


# ---------------------------------------------------------------- served time
def served(run, lanes):
    """(rows, total ms, prompt tokens, verify steps or None): rows = {(kernel, phase): {"ms", "calls", "shapes": {workgroups},
    "lanes": {lane: ms}}} from one profiled request, lanes as events_summary.classify assigns them; verify steps as
    events_summary counts them (None for a kit without a decode_kv lane)"""
    ev, prompt = events_summary.load(run)
    d = json.load(open(lanes))
    pre, lane_of = events_summary.classify(ev, d)
    steps = events_summary.verify_steps(ev, pre, lane_of, d["lanes"]) if any(ln.get("floor") == "decode_kv" for ln in d["lanes"]) \
        else None
    out = {}
    for e in ev:
        r = out.setdefault((e["key"], "prefill" if e["submission_id"] in pre else "decode"),
                           {"ms": 0.0, "calls": 0, "shapes": set(), "lanes": collections.Counter()})
        r["ms"] += e["duration_ns"] / 1e6
        r["lanes"][lane_of(e)] += e["duration_ns"] / 1e6
        r["calls"] += 1
        wg = e.get("workgroup_count") or [0]
        r["shapes"].add(wg[0] * (wg[1] if len(wg) > 1 else 1) * (wg[2] if len(wg) > 2 else 1))
    return out, sum(r["ms"] for r in out.values()), prompt, steps


# ---------------------------------------------------------------- lint + compile report
def report_limit(js):
    """(limiting resources joined by '+', waves per SIMD, VGPRs, [every limiting resource], occupancy inputs for occupancy() or None)
    from a Loom summary report, Nones and [] when absent. A tier can be held by several resources at once (flash2: VGPR and LDS both at 6 waves): each row of
    residency_constraints marked limiting is listed with its units and reduction_units_to_next_better_tier; target_resources'
    limiting_resource names only one of them"""
    if not os.path.exists(js):
        return None, None, None, [], None
    doc = json.load(open(js))
    tr = doc.get("target_resources") or {}
    lims = [{"name": r.get("name", "?").split(".")[-1], "units": r.get("units"), "unit": r.get("unit") or "",
             "reduction": r.get("reduction_units_to_next_better_tier"), "tier": r.get("independent_tier")}
            for r in (doc.get("residency_constraints") or {}).get("rows", []) if r.get("limiting")]
    lim = "+".join(x["name"] for x in lims) or (tr.get("limiting_resource") or "").split(".")[-1] or None
    res = tr.get("residency") or {}
    occ = {"wg": ((doc.get("workload") or {}).get("workgroup_size") or {}).get("flat"), "wave": tr.get("subgroup_size"),
           "cap": tr.get("max_subgroups_per_simd"), "tier": res.get("current_tier"),
           "rows": (doc.get("residency_constraints") or {}).get("rows", [])}
    return lim, res.get("current_tier"), ((tr.get("vector") or {}).get("final") or {}).get("register_count"), lims, occ


def occupancy(occ, simds):
    """The next residency tier a workgroup can reach, and what each resource must drop to for it: {"tier", "next", "q" (waves per
    SIMD per workgroup), "needs": [(resource, units, unit, max units, cut)]}; {"why": reason} when it cannot be verified; None at the
    best reachable tier. Tiers move in whole workgroups: a workgroup of W lanes in waves of S spreads over the SIMDS of one occupancy
    domain (Loom runs WGP mode: 4 SIMDs), q = W / S / SIMDS waves per SIMD, so tiers are multiples of q. Loom 0.2.0's report rounds
    LDS and workgroup slots to whole workgroups but not VGPRs (HRX-like E533): its next_better_tier said 7 for flash2 (384 lanes, q = 3:
    tiers 3 / 6 / 9) and 5 for the 256-lane GEMMs (q = 2), so neither it nor its reductions are used. Each pooled resource's tier is
    recomputed (subgroup scope: pool / rounded units waves; workgroup scope: pool / rounded units workgroups x q) and must equal the
    report's independent_tier, and their minimum (with fixed limits and the wave cap) rounded down to q must equal its current tier,
    or the tier is 'unverified'"""
    if not occ:
        return {"why": "no compile report"}
    wg, wave, cap, cur = occ.get("wg"), occ.get("wave"), occ.get("cap"), occ.get("tier")
    if not simds or not all(isinstance(v, int) and v > 0 for v in (wg, wave, cap, cur)):
        return {"why": "tier unverified: workgroup size, wave size, wave cap or current tier missing from the report, or the device "
                       "file has no simds_per_occupancy_domain"}
    if wg % (wave * simds):
        return {"why": f"tier unverified: a {wg}-lane workgroup is {wg / wave:g} wave(s) over {simds} SIMDs, not whole waves per SIMD"}
    q = wg // (wave * simds)
    pooled, fixed = [], []
    for r in occ["rows"]:
        name, kind = r.get("name", "?").split(".")[-1], r.get("kind")
        if kind == "unconstrained_resource":
            continue
        if kind == "fixed_limit":
            fixed.append((name, r.get("independent_tier")))
            continue
        try:
            pool, gran, units, scope = int(r["pool_units"]), int(r["allocation_granularity"]), int(r["units"]), r["allocation_scope"]
        except (KeyError, TypeError, ValueError):
            return {"why": f"tier unverified: {name} ({kind}) lacks pool, granularity, units or scope"}
        if kind != "pooled_resource" or scope not in ("subgroup", "workgroup") or gran <= 0 or units < 0:
            return {"why": f"tier unverified: {name}: kind {kind!r}, scope {scope!r}, granularity {gran}, units {units} not modelled"}
        ru = -(-units // gran) * gran
        # the report caps each tier at the wave cap; a resource the kernel does not use (0 B of LDS) never limits
        got = cap if ru == 0 else min(cap, pool // ru if scope == "subgroup" else pool // ru * q)
        if got != r.get("independent_tier"):
            return {"why": f"tier unverified: {name} computes to tier {got}, the report says {r.get('independent_tier')}"}
        pooled.append((name, units, r.get("unit") or "", pool, gran, scope, got))
    if any(not isinstance(t, int) for _, t in fixed):
        return {"why": "tier unverified: a fixed limit without a tier"}
    raw = min([c for *_, c in pooled] + [t for _, t in fixed] + [cap])
    tier = raw // q * q
    if cur not in (tier, raw):   # raw: the report's own unrounded minimum (E533: flash2_partial, VGPR tier 5 at q = 3 is tier 3)
        return {"why": f"tier unverified: computes to {tier} in whole workgroups ({raw} unrounded), the report says {cur}"}
    nxt = tier + q
    if nxt > cap:
        return None
    block = [f"{n} (tier {t})" for n, t in fixed if t < nxt]
    if block:
        return {"why": f"next tier {nxt} blocked by a fixed limit: {', '.join(block)}"}
    needs = []
    for name, units, unit, pool, gran, scope, got in pooled:
        if got < nxt:
            mx = (pool // nxt if scope == "subgroup" else pool // (nxt // q)) // gran * gran
            needs.append((name, units, unit, mx, units - mx))
    return {"tier": tier, "next": nxt, "q": q, "needs": needs, "report_tier": cur}


def lint_codes(codes, package=None, events=None, jobs=4):
    """kernel -> kernel_lint row for every code object in CODES (kernel_lint imported, not modified), with outlier, tiny (when
    EVENTS) and maskedv (when PACKAGE) notes as kernel_lint's main adds them"""
    import kernel_lint
    files = sorted(glob.glob(os.path.join(codes, "*.hsaco")))
    with concurrent.futures.ThreadPoolExecutor(jobs) as ex:
        rows = [r for rs in ex.map(kernel_lint.lint_file, files) for r in rs]
    kernel_lint.outliers(rows)
    if events:
        kernel_lint.tiny(rows, kernel_lint.served_stats(events)[0])
    if package:
        texts = [open(p).read() for p in sorted(glob.glob(os.path.join(package, "program", "*.loom")))]
        mv = {k for k, t in kernel_lint.kernel_texts(texts).items() if kernel_lint.masked_v(t)}
        for r in rows:
            if r["kernel"] in mv:
                r.setdefault("notes", []).append("maskedv")
    return {r["kernel"]: r for r in rows}


def hits(r):
    """the rule names a lint row raises (flags and notes)"""
    return list(r.get("flags", [])) + list(r.get("notes", []))


# ---------------------------------------------------------------- counters
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_spec(spec):
    d, *kv = spec.split(":")
    o = {"dir": d}
    for x in kv:
        k, _, v = x.partition("=")
        if k not in ("hsaco", "level") or not v:
            sys.exit(f"lint_counters: bad --counters option {x!r} in {spec!r} (hsaco=PATH, level=LEVEL)")
        o[k] = v
    return o


def workload_hsaco(d):
    """the code object a workload launched: the single *.hsaco path on log.txt's Command line, else None and why"""
    log = os.path.join(d, "log.txt")
    if not os.path.exists(log):
        return None, "no log.txt (pass hsaco=)"
    cmd = next((ln for ln in open(log, errors="replace") if ln.startswith("Command:")), "")
    ps = sorted(set(re.findall(r"(?:^|[\s=])(/[^\s=]+\.hsaco)\b", cmd)))
    if len(ps) != 1:
        return None, f"{len(ps)} code objects on log.txt's Command line (pass hsaco=)"
    return ps[0], None


def workload_level(d):
    """the power level from clock.txt in the workload dir or its parent ('sclk at LEVEL:'), else None"""
    for c in (os.path.join(d, "clock.txt"), os.path.join(os.path.dirname(os.path.normpath(d)), "clock.txt")):
        if os.path.exists(c):
            lv = set(re.findall(r"sclk at (\w+):", open(c).read()))
            return lv.pop() if len(lv) == 1 else None
    return None


def read_rows(d):
    """kernel -> {"n": dispatches, "c": {counter: mean per dispatch}, "dur": {counter: mean ns of the dispatches it was read in},
    "wg": {workgroups}}: a counter may come as several rows per dispatch (dimensions) and from different passes"""
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
    dur = collections.defaultdict(lambda: collections.defaultdict(dict))
    wg = collections.defaultdict(set)
    for f in counter_saturation.files([d]):
        with (gzip.open(f, "rt") if f.endswith(".gz") else open(f)) as fh:
            for r in csv.DictReader(fh):
                r = {k.lower(): v for k, v in r.items() if k}
                k, c = r.get("kernel_name", ""), r.get("counter_name")
                try:
                    v = float(r["counter_value"])
                except (KeyError, TypeError, ValueError):
                    continue
                did = (f, r.get("dispatch_id") or r.get("correlation_id"))   # per file: counters.sh presets each number dispatches from 1 (T6)
                acc[k][c][did] += v
                try:
                    dur[k][c][did] = int(r["end_timestamp"]) - int(r["start_timestamp"])
                except (KeyError, TypeError, ValueError):
                    pass
                try:
                    wg[k].add(int(float(r["grid_size"])) // int(float(r["workgroup_size"])))
                except (KeyError, TypeError, ValueError, ZeroDivisionError):
                    pass
    out = {}
    for k, cs in acc.items():
        out[k] = {"n": max(len(v) for v in cs.values()), "wg": wg[k],
                  "c": {c: sum(v.values()) / len(v) for c, v in cs.items()},
                  "dur": {c: sum(v.values()) / len(v) for c, v in dur[k].items() if v}}
    return out


def metrics(k, sat):
    """figures from one kernel's mean counters: {name: value or the reason it is missing ('unknown' / 'saturated')}"""
    c = k["c"]

    def get(name):
        for n in (name, name + "_sum"):
            if n in c:
                return ("saturated" if n in sat else c[n]), n
        return "unknown", None

    def ratio(num, den, scale=100.0):
        a, b = get(num), get(den)
        for v, _ in (a, b):
            if isinstance(v, str):
                return v
        return scale * a[0] / b[0] if b[0] else "unknown"
    m = {w: ratio(n, "SQ_WAVE_CYCLES") for w, n in WAITS.items()}
    sub = [m["cnt"] if not isinstance(m["cnt"], str) else
           (m["vmem"] + m["lgkm"] if not isinstance(m["vmem"], str) and not isinstance(m["lgkm"], str) else m["cnt"]),
           m["barrier"], m["ifetch"]] + ([m["alu"]] if not isinstance(m["alu"], str) else [])
    bad = [v for v in [m["any"]] + sub if isinstance(v, str)]
    m["other"] = bad[0] if bad else max(0.0, m["any"] - sum(sub))
    m["mem"], m["mem_src"] = (m["vmem"], "vmem") if not isinstance(m["vmem"], str) else (m["cnt"], "cnt")
    m["lds"], m["lds_src"] = (m["lgkm"], "lgkm") if not isinstance(m["lgkm"], str) else (m["cnt"], "cnt")
    m["bank"] = ratio("SQC_LDS_BANK_CONFLICT", "SQC_LDS_IDX_ACTIVE")
    raw = [get(f"GL2C_EA_RDREQ_{b}B") for b in EA]
    if all(not isinstance(v, str) for v, _ in raw):
        ns = [k["dur"].get(n) for _, n in raw if k["dur"].get(n)]
        m["gbs"] = sum(v * b for (v, _), b in zip(raw, EA)) / (sum(ns) / len(ns)) if ns else "unknown"
    elif any(v == "saturated" for v, _ in raw):
        m["gbs"] = "saturated"
    else:
        v, n = get("MemReadBytes")
        m["gbs"] = v if isinstance(v, str) else (v / k["dur"][n] if k["dur"].get(n) else "unknown")
    return m


def dominant(m):
    """(category, share) of the largest measured wait, or None"""
    cand = ["vmem", "lgkm"] if not isinstance(m["vmem"], str) else ["cnt"]
    cand += ["alu"] if not isinstance(m["alu"], str) else ["other"]
    cand += ["barrier", "issue", "ifetch"]
    vals = [(w, m[w]) for w in cand if not isinstance(m[w], str)]
    return max(vals, key=lambda x: x[1]) if vals else None


def load_workloads(specs, codes):
    """per workload: label, level, code object identity and its kernels' figures; served kernels matched by sha256"""
    out = []
    for s in specs:
        o = parse_spec(s)
        d = o["dir"]
        w = {"label": "/".join(os.path.normpath(d).split(os.sep)[-2:]), "dir": d, "level": o.get("level") or workload_level(d),
             "kernels": {}, "why": None}
        hs, why = (o["hsaco"], None) if "hsaco" in o else workload_hsaco(d)
        rows = read_rows(d) if counter_saturation.files([d]) else {}
        if not rows:
            w["why"] = "no counter rows"
        sat = counter_saturation.check(list(counter_saturation.rows([d], None, counter_saturation.derived_names()))) if rows else {}
        w["saturated"] = sorted(sat)
        for k, r in rows.items():
            mine = os.path.join(codes, k + ".hsaco")
            if hs is None:
                ident = f"code object unknown: {why}"
            elif not os.path.exists(hs):
                ident = f"code object {hs} missing"
            elif not os.path.exists(mine):
                ident = "kernel not in --codes"
            elif sha(hs) != sha(mine):
                ident = f"code object {os.path.basename(hs)} differs from the served compile"
            else:
                ident = None
            w["kernels"][k] = {"ident": ident, "m": metrics(r, set(sat)), "n": r["n"], "wg": r["wg"]}
        out.append(w)
    return out


def usable(workloads, kernel):
    """[(workload, figures, served_level)] whose code object is the served kernel's"""
    return [(w, w["kernels"][kernel]["m"], w["level"] == "auto") for w in workloads
            if kernel in w["kernels"] and w["kernels"][kernel]["ident"] is None]


def judge(rule, m, at_auto, bw_ceiling):
    """(backed True / False / None, matching share or None, why) for one rule on one workload's figures"""
    kind = RULES[rule]
    if kind == "bank":
        b = m["bank"]
        if isinstance(b, str):
            return None, None, f"bank {b}"
        lds = m["lds"] if at_auto and not isinstance(m["lds"], str) else None
        return b >= BANK_MIN, (lds * b / 100 if lds is not None else None), f"bank {b:.2f} %"
    if not at_auto:
        return None, None, "wait shares not at the served level"
    v = m[kind]
    if isinstance(v, str):
        return None, None, f"{kind} {v}"
    src = m["mem_src"] if kind == "mem" else kind
    if v < (MEM_MIN if kind == "mem" else ISSUE_MIN):
        return False, v, f"{src} {v:.1f} %"
    if rule == "serial":
        g = m["gbs"]
        if isinstance(g, str) or bw_ceiling is None:
            return None, v, f"{src} {v:.1f} %, GB/s {g if isinstance(g, str) else 'ceiling unknown'}"
        if g >= BW_MAX * bw_ceiling:
            return False, v, f"{src} {v:.1f} % at {g:.0f} GB/s (bandwidth-bound)"
        return True, v, f"{src} {v:.1f} % at {g:.0f} GB/s"
    return True, v, f"{src} {v:.1f} %"


def back(rule, workloads, kernel, bw_ceiling):
    """a kernel's verdict on one rule over its usable workloads: (verdict, mean matching share or None, whys); verdict is
    'backed', 'not backed', 'mixed', 'no counters' or 'unusable counters'"""
    us = usable(workloads, kernel)
    if not us:
        return "no counters", None, []
    js = [(w["label"],) + judge(rule, m, auto, bw_ceiling) for w, m, auto in us]
    known = [j for j in js if j[1] is not None]
    whys = [f"{lb}: {why}" for lb, _, _, why in js]
    if not known:
        return "unusable counters", None, whys
    shares = [j[2] for j in known if j[2] is not None]
    share = sum(shares) / len(shares) if shares else None
    if all(j[1] for j in known):
        return "backed", share, whys
    return ("not backed" if not any(j[1] for j in known) else "mixed"), share, whys


# ---------------------------------------------------------------- report
def fmt(v, f="{:.1f}"):
    return v if isinstance(v, str) else f.format(v)


def best_usable(workloads, kernel):
    """usable workloads, best first: at the served level, then the most wait figures"""
    return sorted(usable(workloads, kernel), key=lambda x: (not x[2], -sum(not isinstance(x[1][c], str) for c in WAITS)))


def counters_cell(workloads, kernel):
    """the table's counter cell: dominant wait, bank %, GB/s from the best usable workload (served level, most figures)"""
    us = best_usable(workloads, kernel)
    other = [w["label"] + " (" + w["kernels"][kernel]["ident"] + ")" for w in workloads
             if kernel in w["kernels"] and w["kernels"][kernel]["ident"] is not None]
    if not us:
        return "no counters" + (f"; not used: {'; '.join(other)}" if other else ""), ""
    w, m, auto = us[0]
    d = dominant(m) if auto else None
    if auto:
        dom = f"{d[0]} {d[1]:.1f} %" if d else "wait " + ("saturated" if any(m[c] == "saturated" for c in WAITS) else "unknown")
    else:
        dom = f"waits not at auto ({w['level'] or 'level unknown'})"
    s = f"{dom}; bank {fmt(m['bank'], '{:.2f}')} %; {fmt(m['gbs'], '{:.0f}') if auto else 'n/a'} GB/s [{w['label']}" + \
        (f" +{len(us) - 1}" if len(us) > 1 else "") + "]"
    sat = [x for x, _, _ in us if x["saturated"]]
    return s, ("SATURATED " + "; ".join(f"{x['label']}: {','.join(x['saturated'])}" for x in sat) +
               " (figures on them dropped)") if sat else ""


def build(lint, srv, total, workloads, bw_ceiling):
    """sections (a)-(c) as lists of lines (pure: tested on fixtures; (d) is section_d)"""
    A = ["(a) per served kernel (key, phase): served device time, lint, compile report, counters",
         f"{'kernel':40s} {'phase':7s} {'ms':>9s} {'share%':>6s} {'calls':>6s} {'shp':>3s} {'limit':>9s} {'tier':>4s} {'vgpr':>4s}  "
         f"{'lint':28s} counters | saturation"]
    for (k, ph), r in sorted(srv.items(), key=lambda x: -x[1]["ms"]):
        lr = lint.get(k)
        ln = " ".join(lr.get("flags", [])) + (" [" + " ".join(lr["notes"]) + "]" if lr.get("notes") else "") if lr else "not linted"
        lim, tier, vg = (lr.get("_limit"), lr.get("waves"), lr.get("vgpr")) if lr else (None, None, None)
        cc, sat = counters_cell(workloads, k)
        A.append(f"{k:40s} {ph:7s} {r['ms']:9.1f} {100 * r['ms'] / total:6.2f} {r['calls']:6d} {len(r['shapes']):3d} "
                 f"{lim or '-':>9s} {str(tier if tier is not None else '-'):>4s} {str(vg if vg is not None else '-'):>4s}  {ln or '-':28s} "
                 f"{cc}" + (f" | {sat}" if sat else ""))
    share = collections.Counter()
    for (k, _), r in srv.items():
        share[k] += 100 * r["ms"] / total
    B = ["(b) per lint rule: hits (all linted kernels / served), hits with usable counters, backed; precision = backed / with counters",
         "    (first pass, author-judged thresholds, n = hits with counters; needs an outside sample before it is trusted: PRINCIPLES)",
         f"{'rule':9s} {'hits':>5s} {'served':>6s} {'w/cnt':>5s} {'backed':>6s} {'mixed':>5s}  precision"]
    C, unpriced = [], []
    for rule in sorted({h for r in lint.values() for h in hits(r)}):
        ks = [k for k, r in lint.items() if rule in hits(r)]
        sv = [k for k in ks if share[k] > 0]
        if rule not in RULES:
            B.append(f"{rule:9s} {len(ks):5d} {len(sv):6d}     -      -     -  not priced: " +
                     UNPRICED.get(rule, "no counter backing defined"))
            continue
        res = {k: back(rule, workloads, k, bw_ceiling) for k in ks}
        wc = [k for k in ks if res[k][0] in ("backed", "not backed", "mixed")]
        bk = [k for k in wc if res[k][0] == "backed"]
        mx = [k for k in wc if res[k][0] == "mixed"]
        prec = f"{len(bk)}/{len(wc)} (n={len(wc)}; first pass, author-judged)" if wc else "no hit has usable counters (n=0)"
        B.append(f"{rule:9s} {len(ks):5d} {len(sv):6d} {len(wc):5d} {len(bk):6d} {len(mx):5d}  {prec}")
        quiet = 0
        for k in ks:
            v, s, whys = res[k]
            if share[k] <= 0 and v == "no counters":
                quiet += 1
                continue
            B.append(f"    {k:40s} {share[k]:6.2f} % {v:17s} {'; '.join(whys)}")
            if share[k] <= 0:
                continue
            if s is None:
                unpriced.append((share[k], k, rule, v))
            else:
                C.append((share[k] * s / 100, k, rule, v, share[k], s))
        if quiet:
            B.append(f"    (+{quiet} hits not served in this request and without counters)")
    C.sort(key=lambda x: -x[0])
    out_c = ["(c) hits ranked by price = served share x matching share (% of served device time the flagged pattern could hold)",
             "    an upper bound per hit, not additive: rules priced off one kernel's same figure (issue wait for copies / addr64 / join)",
             "    share one budget, and the whole kernel's wait is charged to each pattern; a ranking of what to read, not a prediction",
             f"{'#':>3s} {'price%':>7s} {'kernel':40s} {'rule':8s} {'verdict':11s} {'share%':>6s} {'match%':>6s}"]
    out_c += [f"{i:3d} {p:7.3f} {k:40s} {rule:8s} {v:11s} {sh:6.2f} {s:6.1f}" for i, (p, k, rule, v, sh, s) in enumerate(C, 1)]
    out_c.append(f"unpriced served hits (no usable counters; served share is the ceiling): {len(unpriced)}")
    out_c += [f"    {sh:6.2f} %  {k:40s} {rule:8s} {v}" for sh, k, rule, v in sorted(unpriced, reverse=True)]
    return A, B, out_c


# ---------------------------------------------------------------- (d) levers: lane headroom, ranking, amdahl's noise screen
UNNAMED = "unnamed (no lint flag)"
EVIDENCE = {"backed": "counter-backed", "not backed": "counters: not backed", "mixed": "counters: mixed"}   # else 'static only'


def lane_noise(path, d):
    """{lane name: noise floor %} from a {lane substring: pct} file (events_summary.noise_floors: a key naming no lane or
    several stops the run)"""
    names = [ln["lane"] for ln in d["lanes"]]
    nf = events_summary.noise_floors(path, names)
    if nf is None:
        sys.exit(f"lint_counters: --noise-floor {path} unusable (see above)")
    return {names[i]: v for i, v in nf.items()}


def lane_headroom(srv, steps, prompt, d, hw):
    """{lane: {"phase", "ms", "floor_ms", "headroom_ms", "why"}} over one request: the lane's served ms against ceiling.lane_floor
    at HW (resource floor or latency chain, the larger; the floor amdahl's to_floor uses), decode floors x verify steps, the
    context and prompt both the served prompt (events_summary's convention). A lane without a floor is judged against 0
    (ceiling.py); a floor that needs a constant lanes.json lacks, a decode floor without verify steps, or a phase this request
    did not serve, is unknown (None). note: the lane's floor kind from lanes.json (e.g. 'design floor; ideal 0')"""
    out, served_ph = {}, {ph for _, ph in srv}
    for ln in d["lanes"]:
        ms = sum(r["lanes"].get(ln["lane"], 0.0) for r in srv.values())
        h = {"phase": ln["phase"], "ms": ms, "floor_ms": None, "note": ln.get("note")}
        dec = ln["phase"] == "decode"
        if ln["phase"] not in served_ph:
            h.update(why=f"{ln['phase']} not served in this request: no floor or headroom", headroom_ms=None, floored=False)
            out[ln["lane"]] = h
            continue
        try:
            f = ceiling.lane_floor(ln, d.get("model") or {}, hw, prompt, prompt)
        except KeyError as e:
            h["why"] = f"floor unknown: lanes.json model has no {e.args[0]!r}"
        else:
            if f is None:
                h["floor_ms"], h["why"] = 0.0, "no floor: a fixed cost, judged against 0 (ceiling.py)"
            elif dec and not steps:
                h["why"] = "no verify steps in this run: a per-step floor cannot be scaled to the request"
            else:
                h["floor_ms"] = f["floor"] * (steps if dec else 1e3)
                h["why"] = (f"floor {f['floor']:.3f} {'ms/step' if dec else 's/prompt'} ({f['binds']} binds)" +
                            (f" x {steps} verify steps" if dec else ""))
        h["headroom_ms"] = None if h["floor_ms"] is None else max(0.0, ms - h["floor_ms"])
        h["floored"] = bool(h["floor_ms"])   # a computed floor above 0: no-floor (judged against 0) and unknown lanes are not
        out[ln["lane"]] = h
    return out


def levers(lint, srv, workloads, bw_ceiling, heads):
    """one lever per (served kernel, lane, priced lint rule), or one 'unnamed' lever for a kernel without a priced rule:
    {"kernel", "phase", "lane", "lever", "evidence", "target_ms", "attr_ms", "price_ms", "bound_ms", "bound_kind", "pct", "floored"},
    ranked by pct (None last): bound_ms as % of the phase's served ms (r2, user 11-25: decode and prefill are ranked separately).
    floored: the lane has a floor above 0. attr_ms = the kernel's share of its lane's served ms x the lane's headroom: an
    attribution, not an upper bound (it assumes every kernel of the lane is equally far from its floor; no kernel was timed at a
    floor). price_ms = the kernel's ms in the lane x the rule's matching share (section c). bound_ms = the price when the counters
    back the rule and it is below the attribution (bound_kind 'price'), else the attribution ('attribution'): an unbacked or mixed
    price never cuts a kernel's bound (r3: vsplit's 697 ms attribution became three 293 ms levers on a pattern the counters
    refuted, ranked first and 'clearing' the screen)"""
    out, phase_ms = [], collections.Counter()
    for (_, ph), r in srv.items():
        phase_ms[ph] += r["ms"]
    for (k, ph), r in srv.items():
        pats = [h for h in hits(lint.get(k) or {}) if h in RULES]
        res = {rule: back(rule, workloads, k, bw_ceiling) for rule in pats}
        for lane, ms in r["lanes"].items():
            hd = heads[lane]
            attr = hd["headroom_ms"] * ms / hd["ms"] if hd["headroom_ms"] is not None and hd["ms"] > 0 else None
            base = {"kernel": k, "phase": ph, "lane": lane, "target_ms": ms, "attr_ms": attr, "floored": hd["floored"]}
            if not pats:
                out.append(dict(base, lever=UNNAMED, evidence="no evidence", price_ms=None))
            for rule in pats:
                v, share, _ = res[rule]
                out.append(dict(base, lever=rule, evidence=EVIDENCE.get(v, "static only"),
                                price_ms=ms * share / 100 if share is not None else None))
    for x in out:
        priced = x["evidence"] == "counter-backed" and x["price_ms"] is not None
        if priced and (x["attr_ms"] is None or x["price_ms"] < x["attr_ms"]):
            x["bound_ms"], x["bound_kind"] = x["price_ms"], "price"
        else:
            x["bound_ms"], x["bound_kind"] = x["attr_ms"], ("attribution" if x["attr_ms"] is not None else None)
        x["pct"] = 100 * x["bound_ms"] / phase_ms[x["phase"]] if x["bound_ms"] is not None and phase_ms[x["phase"]] > 0 else None
    out.sort(key=lambda x: (x["pct"] is None, -(x["pct"] or 0.0), x["kernel"], x["lever"]))
    return out


def ranked(lv, phase, top):
    """the TOP floor-backed levers of PHASE in rank order (LV is sorted by pct): what (d) ranks and amdahl screens. A floor-backed
    lever is always bounded: its lane's headroom is known, so its attribution is"""
    return [x for x in lv if x["phase"] == phase and x["floored"]][:top]


def amdahl_input(lv, heads, noise, top):
    """(amdahl.py input doc, the levers sent): each phase's TOP floor-backed levers (ranked: a no-floor lane's bound is the whole
    kernel time, not a floor, so it is never screened), one amdahl phase per lane (wall = the lane's served ms, screen_pct = its
    noise floor; without one amdahl's default screen applies), one overlap_group per kernel so two levers on one kernel never add.
    A floor above 0 keeps every lane's savings below its wall (each lever <= its attribution; they sum to the headroom)"""
    picked = [x for ph in sorted({x["phase"] for x in lv}) for x in ranked(lv, ph, top)]
    by = {}
    for x in picked:
        by.setdefault(x["lane"], []).append(x)
    doc = {"phases": []}
    for lane, xs in by.items():
        wall = heads[lane]["ms"]
        ph = {"phase": lane, "unit": "ms/request", "wall": wall,
              "candidates": [{"name": f"{x['kernel']}:{x['lever']}", "target": x["target_ms"], "reduction": x["bound_ms"],
                              "overlap_group": x["kernel"], "src": f"lint_counters (d): {x['evidence']}, bound = {x['bound_kind']}"} for x in xs]}
        if lane in noise:
            ph["screen_pct"] = noise[lane]
        doc["phases"].append(ph)
    return doc, picked


def run_amdahl(doc, path):
    """write DOC to PATH, run common/amdahl.py on it; (its output lines, {(lane, lever name): (passes, screen %)}). Only a PASS / FAIL
    run's result is read: an INVALID run writes none, and the file left by an earlier run at PATH must not be taken for it"""
    import subprocess
    json.dump(doc, open(path, "w"), indent=1)
    res = path + ".result.json"
    p = subprocess.run([sys.executable, os.path.join(H, "amdahl.py"), path, "--json", res], capture_output=True, text=True)
    lines = (p.stderr + p.stdout).rstrip("\n").splitlines() or ["amdahl: no output"]   # its verdict line last
    verdicts = {}
    if p.returncode in (0, 1) and os.path.exists(res):
        for ph in json.load(open(res)):
            for r in ph.get("rows", []):
                verdicts[(ph["phase"], r["name"])] = (r["passes"], ph["screen_pct"])
    return lines, verdicts


def latency_wait(workloads, kernel):
    """'X % (src) [workload]': the share of wave cycles waiting on load counters (vmem + lgkm, else cnt) at the served level, from
    the best usable workload, or why it is unknown"""
    for w, m, auto in best_usable(workloads, kernel):
        if not auto:
            continue
        if not isinstance(m["vmem"], str) and not isinstance(m["lgkm"], str):
            return f"{m['vmem'] + m['lgkm']:.1f} % (vmem {m['vmem']:.1f} + lgkm {m['lgkm']:.1f}) [{w['label']}]"
        if not isinstance(m["cnt"], str):
            return f"{m['cnt']:.1f} % (cnt) [{w['label']}]"
        return f"{m['cnt']} [{w['label']}]"
    return "unknown (no usable counters at the served level)"


def screen_cell(x, verdicts, noise):
    """the screen column for a ranked lever: amdahl's verdict, the screen it used (the lane's noise floor or amdahl's default) and
    whether the bound it judged is a backed price or only an attribution (an attribution below the screen is not 'not worth
    building': the kernel's own distance from a floor was never computed)"""
    v = verdicts.get((x["lane"], f"{x['kernel']}:{x['lever']}"))
    if v is None:
        return "amdahl gave no verdict"
    passes, scr = v
    where = f"{'noise' if x['lane'] in noise else 'default'} {scr:g} %"
    if x["bound_kind"] == "price":
        return f"clears {where}" if passes else "not worth building"
    return f"attribution clears {where}" if passes else "attribution below screen (no per-kernel floor)"


def section_d(lv, heads, verdicts, noise, lint, srv, workloads, top, hwname, simds):
    """(d) as lines: the lanes against their floors, then one ranked list per phase (decode first, ranks from 1 in each) with
    amdahl's verdict, a 'no floor' and a 'floor unknown' block after it; then (d2), the occupancy levers (unpriced)"""
    out = [f"(d) levers per phase, ranked by bound as % of the phase's served time; bound = the kernel's attribution of its lane's",
           f"    headroom, or a counter-backed pattern price when smaller. lane floors: ceiling.lane_floor at {hwname}; headroom = lane",
           "    served - floor. attr = the kernel's share of its lane's served time x the lane's headroom: an attribution, not an upper",
           "    bound and not a measurement (it assumes every kernel of a lane is equally far from its floor). price = the kernel's ms",
           "    in the lane x the rule's matching share (section c); it sets the bound only when the counters back the rule. evidence:",
           "    counter-backed / counters: not backed or mixed (section b) / static only (a lint flag, no usable counters) / no evidence",
           "    (no lint flag). screen: amdahl.py, one phase per lane; 'not worth building' only for a backed price below the screen.",
           "    Not additive: amdahl's combined line counts the best lever per kernel. Only floor-backed levers are ranked and",
           "    screened; levers in lanes without a floor (judged against 0, ceiling.py) or with an unknown floor are listed after."]
    if noise:
        out += ["    screen per lane: the kit's noise_floor.json (lanes without an entry: amdahl's default). PROVISIONAL: its false-pass",
                "    sweep is still owed (kit NOTES.md), so a 'clears noise' verdict is provisional."]
    else:
        out += ["    screen per lane: amdahl's default (no --noise-floor given): not an A/B noise floor."]
    out.append(f"{'lane':45s} {'phase':7s} {'served ms':>10s} {'floor ms':>10s} {'headroom':>10s} {'noise%':>6s}  why")
    for lane, h in heads.items():
        out.append(f"{lane:45s} {h['phase']:7s} {h['ms']:10.1f} {fmt(h['floor_ms'], '{:.1f}') if h['floor_ms'] is not None else 'unknown':>10s} "
                   f"{fmt(h['headroom_ms'], '{:.1f}') if h['headroom_ms'] is not None else 'unknown':>10s} "
                   f"{fmt(noise[lane], '{:.2f}') if lane in noise else '-':>6s}  {h['why']}" + (f" [{h['note']}]" if h.get("note") else ""))
    f = lambda y, w=9, p=1: f"{y:{w}.{p}f}" if y is not None else f"{'-':>{w}s}"
    hdr = (f"{'#':>3s} {'bound%':>7s} {'bound ms':>9s} {'attr ms':>9s} {'price ms':>9s}  {'evidence':21s} {'screen':47s} "
           f"{'kernel':36s} {'lever':22s} lane")
    row = lambda i, x, scr: (f"{i:>3s} {f(x['pct'], 7, 2)} {f(x['bound_ms'])} {f(x['attr_ms'])} {f(x['price_ms'])}  "
                             f"{x['evidence']:21s} {scr:47s} {x['kernel']:36s} {x['lever']:22s} {x['lane']}")
    pms = collections.Counter()
    for (_, ph), r in srv.items():
        pms[ph] += r["ms"]
    phases = sorted({h["phase"] for h in heads.values()} | set(pms), key=lambda p: (p != "decode", p))
    for ph in phases:
        if ph not in pms:
            out += ["", f"(d) {ph}: not served in this request"]
            continue
        out += ["", f"(d) {ph}: levers ranked by bound as % of {ph}'s served time ({pms[ph]:.1f} ms), top {top}", hdr]
        for i, x in enumerate(ranked(lv, ph, top), 1):
            out.append(row(str(i), x, screen_cell(x, verdicts, noise)))
        rest = [x for x in lv if x["phase"] == ph and not x["floored"]]
        nofl = [x for x in rest if heads[x["lane"]]["floor_ms"] is not None]
        out += [f"    {ph}: no floor: bound = whole kernel time (an attribution against 0); unranked, not screened; "
                f"{min(top, len(nofl))} of {len(nofl)} shown"] + [row("-", x, "no floor: not screened") for x in nofl[:top]]
        unk = [x for x in rest if heads[x["lane"]]["floor_ms"] is None]
        if unk:
            out += [f"    {ph}: floor unknown (see the lane table): not ranked, not screened; {min(top, len(unk))} of {len(unk)} shown"] + \
                [row("-", x, "floor unknown: not screened") for x in unk[:top]]
    ms = collections.Counter()
    for (k, _), r in srv.items():
        ms[k] += r["ms"]
    total = sum(ms.values()) or 1.0
    out += ["", "(d2) occupancy levers (unpriced: occupancy is latency hiding, not a throughput model, rules 1): the next tier a",
            "    workgroup can reach (tiers move in whole workgroups: waves per SIMD per workgroup = workgroup lanes / wave size /",
            "    SIMDs per occupancy domain; the report's own next tier is not used, HRX-like E533), every resource that must drop for it",
            "    and to what; the tier moves only when ALL of them move. Beside it the kernel's load-counter wait share at the served",
            "    level: the latency a higher tier could hide. Every served kernel is listed except those at the best reachable tier.",
            f"{'kernel':36s} {'ms':>9s} {'share%':>6s}  next reachable tier and what must move  | load waits"]
    for k in sorted(ms, key=lambda k: -ms[k]):
        lr = lint.get(k)
        o = occupancy(lr.get("_occ"), simds) if lr else {"why": "not linted: no compile report"}
        if o is None:
            continue
        if "why" in o:
            txt = o["why"]
        else:
            txt = (f"tier {o['tier']}" + (f" (report: {o['report_tier']}, not whole workgroups, E533)" if o["report_tier"] != o["tier"] else "")
                   + f" -> {o['next']} ({o['q']} waves/SIMD per workgroup): " +
                   ", ".join(f"{n} {u} -> <= {mx} {unit} (-{cut})" for n, u, unit, mx, cut in o["needs"]) + f": needs all {len(o['needs'])}")
        out.append(f"{k:36s} {ms[k]:9.1f} {100 * ms[k] / total:6.2f}  {txt}  | {latency_wait(workloads, k)}")
    return out


def details(workloads, srv):
    out = ["counter workloads (what each one is and whether it was used):"]
    for w in workloads:
        out.append(f"  {w['label']}: level {w['level'] or 'unknown'}; saturation "
                   f"{'FOUND ' + ' '.join(w['saturated']) if w['saturated'] else 'none'}" + (f"; {w['why']}" if w["why"] else ""))
        for k, x in sorted(w["kernels"].items()):
            m = x["m"]
            sv = sorted({s for (kk, _), r in srv.items() if kk == k for s in r["shapes"]})
            out.append(f"    {k}: {x['n']} dispatches, {'used' if x['ident'] is None else 'NOT USED: ' + x['ident']}; workgroups "
                       f"{sorted(x['wg']) or 'unknown'} (served: {'yes' if set(x['wg']) & set(sv) else 'no'}, {len(sv)} served shapes)")
            out.append("      " + ", ".join(f"{c} {fmt(m[c])}" for c in list(WAITS) + ["other"]) +
                       f"; bank {fmt(m['bank'], '{:.2f}')} %; MemRead {fmt(m['gbs'], '{:.1f}')} GB/s")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--codes", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--lanes", required=True)
    ap.add_argument("--package")
    ap.add_argument("--counters", action="append", default=[])
    ap.add_argument("--device", default=os.path.join(H, "devices", "gfx1151.json"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--profile", help="the device file's profile for the lane floors (default: its default_profile)")
    ap.add_argument("--noise-floor", help="(d): {lane substring: pct}, the kit's noise_floor.json: each lane's amdahl screen")
    ap.add_argument("--top", type=int, default=15, help="(d): levers ranked and screened per phase")
    ap.add_argument("--amdahl-out", help="(d): write amdahl's input JSON here (default: a temporary file)")
    a = ap.parse_args()
    if a.top < 1:
        ap.error("--top must be >= 1")
    dev = json.load(open(a.device))
    bw = ((dev.get("profiles") or {}).get(dev.get("default_profile")) or {}).get("bw_practical_gbs")
    srv, total, prompt, steps = served(a.events, a.lanes)
    lint = lint_codes(a.codes, a.package, a.events, a.jobs)
    for k, r in lint.items():
        r["_limit"], tier, vg, r["_limits"], r["_occ"] = report_limit(os.path.join(a.codes, k + ".json"))
        r["waves"] = r.get("waves") if r.get("waves") is not None else tier
        r["vgpr"] = r.get("vgpr") if r.get("vgpr") is not None else vg
    wl = load_workloads(a.counters, a.codes)
    ph = collections.Counter()
    for (_, p), r in srv.items():
        ph[p] += r["ms"]
    print(f"served: {a.events} ({prompt} prompt tokens), {total / 1e3:.2f} s of dispatches: " +
          ", ".join(f"{p} {v / 1e3:.2f} s" for p, v in sorted(ph.items())))
    print(f"lint: {len(lint)} kernels in {a.codes}; served keys {len({k for k, _ in srv})}, of them not linted "
          f"{len({k for k, _ in srv} - set(lint))}; device DRAM ceiling {bw} GB/s ({os.path.basename(a.device)})")
    for sec in build(lint, srv, total, wl, bw):
        print()
        print("\n".join(sec))
    d = json.load(open(a.lanes))
    hw = ceiling.hardware(d, a.device, a.profile)
    heads = lane_headroom(srv, steps, prompt, d, hw)
    noise = lane_noise(a.noise_floor, d) if a.noise_floor else {}
    lv = levers(lint, srv, wl, bw, heads)
    doc, _ = amdahl_input(lv, heads, noise, a.top)
    import tempfile
    if doc["phases"]:
        with tempfile.TemporaryDirectory() as t:
            lines, verdicts = run_amdahl(doc, a.amdahl_out or os.path.join(t, "amdahl_in.json"))
    else:
        lines, verdicts = ["nothing screened: no floor-backed lever"], {}
    print()
    print("\n".join(section_d(lv, heads, verdicts, noise, lint, srv, wl, a.top, hw["name"], dev.get("simds_per_occupancy_domain"))))
    print(f"\n(d3) amdahl.py on each phase's top {a.top} floor-backed levers (no-floor lanes are not sent)" + (f" ({a.amdahl_out})" if a.amdahl_out else "") +
          (f"; lanes without a noise floor use amdahl's default screen" if set(heads) - set(noise) else "") + ":")
    print("\n".join("    " + x for x in lines))
    print()
    print("\n".join(details(wl, srv)))
    unl = sorted({k for k, _ in srv} - set(lint))
    print(f"\nserved, not linted ({len(unl)}): {' '.join(unl)}")


if __name__ == "__main__":
    main()
