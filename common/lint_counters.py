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
  --top N         (d): levers printed and sent to amdahl.py (default 15); --amdahl-out PATH keeps amdahl's input JSON

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
minus its floor (ceiling.lane_floor at --device's profile, the floor amdahl's to_floor uses; a lane without a floor against 0);
the kernel's attribution is its share of the lane's served time x that headroom: an attribution, not a measurement. The top N
go to common/amdahl.py as one phase per lane (wall = the lane's served ms, screen = its noise floor, one overlap_group per
kernel), and a lever below its lane's screen is 'not worth building' (rules 2: price the lever before building it). (d2) lists
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
    """(limiting resources joined by '+', waves per SIMD, VGPRs, [every limiting resource], next tier) from a Loom summary report, Nones
    and [] when absent. A tier can be held by several resources at once (flash2: VGPR and LDS both at 6 waves): each row of
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
    return (lim, res.get("current_tier"), ((tr.get("vector") or {}).get("final") or {}).get("register_count"), lims,
            res.get("next_better_tier"))


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
    (ceiling.py); a floor that needs a constant lanes.json lacks, or a decode floor without verify steps, is unknown (None)"""
    out = {}
    for ln in d["lanes"]:
        ms = sum(r["lanes"].get(ln["lane"], 0.0) for r in srv.values())
        h = {"phase": ln["phase"], "ms": ms, "floor_ms": None}
        dec = ln["phase"] == "decode"
        try:
            f = ceiling.lane_floor(ln, d["model"], hw, prompt, prompt)
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
        out[ln["lane"]] = h
    return out


def levers(lint, srv, workloads, bw_ceiling, heads):
    """one lever per (served kernel, lane, priced lint rule), or one 'unnamed' lever for a kernel without a priced rule:
    {"kernel", "lane", "lever", "evidence", "target_ms", "attr_ms", "price_ms", "upper_ms"}, ranked by upper_ms (None last).
    attr_ms = the kernel's share of its lane's served ms x the lane's headroom (an attribution: no kernel was timed at a floor);
    price_ms = the kernel's ms in the lane x the rule's matching share (section c); upper_ms = the smaller of the known two"""
    out = []
    for (k, _), r in srv.items():
        pats = [h for h in hits(lint.get(k) or {}) if h in RULES]
        res = {rule: back(rule, workloads, k, bw_ceiling) for rule in pats}
        for lane, ms in r["lanes"].items():
            hd = heads[lane]
            attr = hd["headroom_ms"] * ms / hd["ms"] if hd["headroom_ms"] is not None and hd["ms"] > 0 else None
            base = {"kernel": k, "lane": lane, "target_ms": ms, "attr_ms": attr}
            if not pats:
                out.append(dict(base, lever=UNNAMED, evidence="no evidence", price_ms=None))
            for rule in pats:
                v, share, _ = res[rule]
                out.append(dict(base, lever=rule, evidence=EVIDENCE.get(v, "static only"),
                                price_ms=ms * share / 100 if share is not None else None))
    for x in out:
        known = [v for v in (x["attr_ms"], x["price_ms"]) if v is not None]
        x["upper_ms"] = min(known) if known else None
    out.sort(key=lambda x: (x["upper_ms"] is None, -(x["upper_ms"] or 0.0), x["kernel"], x["lever"]))
    return out


def amdahl_input(lv, heads, noise, top):
    """(amdahl.py input doc, the levers sent, [(lane, why) left out]): the TOP bounded levers, one phase per lane (wall = the
    lane's served ms, screen_pct = its noise floor; without one amdahl's default screen applies), one overlap_group per kernel so
    two levers on one kernel never add. A lane whose kernels' best levers claim its whole time (floor 0) is left out: amdahl
    refuses groups that save the whole wall"""
    picked = [x for x in lv if x["upper_ms"] is not None][:top]
    by = {}
    for x in picked:
        by.setdefault(x["lane"], []).append(x)
    doc, omitted = {"phases": []}, []
    for lane, xs in by.items():
        wall = heads[lane]["ms"]
        best = {}
        for x in xs:
            best[x["kernel"]] = max(best.get(x["kernel"], 0.0), x["upper_ms"])
        if sum(best.values()) >= wall:
            omitted.append((lane, f"its levers claim {sum(best.values()):.1f} of {wall:.1f} ms ({heads[lane]['why']})"))
            continue
        ph = {"phase": lane, "unit": "ms/request", "wall": wall,
              "candidates": [{"name": f"{x['kernel']}:{x['lever']}", "target": x["target_ms"], "reduction": x["upper_ms"],
                              "overlap_group": x["kernel"], "src": f"lint_counters (d): {x['evidence']}"} for x in xs]}
        if lane in noise:
            ph["screen_pct"] = noise[lane]
        doc["phases"].append(ph)
    return doc, picked, omitted


def run_amdahl(doc, path):
    """write DOC to PATH, run common/amdahl.py on it; (its output lines, {(lane, lever name): passes})"""
    import subprocess
    json.dump(doc, open(path, "w"), indent=1)
    res = path + ".result.json"
    p = subprocess.run([sys.executable, os.path.join(H, "amdahl.py"), path, "--json", res], capture_output=True, text=True)
    lines = (p.stdout + p.stderr).rstrip("\n").splitlines() or ["amdahl: no output"]
    verdicts = {}
    if p.returncode in (0, 1) and os.path.exists(res):
        for ph in json.load(open(res)):
            for r in ph.get("rows", []):
                verdicts[(ph["phase"], r["name"])] = r["passes"]
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


def section_d(lv, heads, verdicts, omitted, noise, lint, srv, workloads, top, hwname):
    """(d) as lines: the lanes against their floors, the ranked levers with amdahl's verdict, the occupancy levers (unpriced)"""
    out = [f"(d) levers ranked by upper bound = min(headroom attribution, pattern price), ms of served device time in this request",
           f"    lane floors: ceiling.lane_floor at {hwname}; headroom = lane served - floor (a lane without a floor: against 0).",
           "    attr = the kernel's share of its lane's served time x the lane's headroom: an attribution, not a measurement (no kernel",
           "    was timed at its floor). price = the kernel's ms in the lane x the rule's matching share (section c). evidence:",
           "    counter-backed / counters: not backed or mixed (section b) / static only (a lint flag, no usable counters) / no evidence",
           "    (no lint flag: only the lane's headroom). screen: amdahl.py, one phase per lane, the lane's noise floor % of its served",
           "    time; a lever below it is not worth building (an A/B could not tell it from noise). Not additive: amdahl's combined",
           "    line counts the best lever per kernel.",
           f"{'lane':45s} {'phase':7s} {'served ms':>10s} {'floor ms':>10s} {'headroom':>10s} {'noise%':>6s}  why"]
    for lane, h in heads.items():
        out.append(f"{lane:45s} {h['phase']:7s} {h['ms']:10.1f} {fmt(h['floor_ms'], '{:.1f}') if h['floor_ms'] is not None else 'unknown':>10s} "
                   f"{fmt(h['headroom_ms'], '{:.1f}') if h['headroom_ms'] is not None else 'unknown':>10s} "
                   f"{fmt(noise[lane], '{:.2f}') if lane in noise else '-':>6s}  {h['why']}")
    out += ["", f"{'#':>3s} {'upper ms':>9s} {'attr ms':>9s} {'price ms':>9s}  {'evidence':21s} {'screen':28s} {'kernel':36s} "
                f"{'lever':22s} lane"]
    for i, x in enumerate(lv[:top], 1):
        v = verdicts.get((x["lane"], f"{x['kernel']}:{x['lever']}"))
        scr = ("clears the noise screen" if v else "not worth building") if v is not None else \
            ("unbounded: not screened" if x["upper_ms"] is None else "not screened (lane left out)")
        f = lambda y: f"{y:9.1f}" if y is not None else f"{'-':>9s}"
        out.append(f"{i:3d} {f(x['upper_ms'])} {f(x['attr_ms'])} {f(x['price_ms'])}  {x['evidence']:21s} {scr:28s} {x['kernel']:36s} "
                   f"{x['lever']:22s} {x['lane']}")
    out += [f"    lane left out of the screen: {lane}: {why}" for lane, why in omitted]
    ms = collections.Counter()
    for (k, _), r in srv.items():
        ms[k] += r["ms"]
    total = sum(ms.values()) or 1.0
    out += ["", "(d2) occupancy levers (unpriced: occupancy is latency hiding, not a throughput model, rules 1): every resource the",
            "    compile report marks limiting, with the reduction to the next tier; the tier moves only when ALL of them move. Beside",
            "    it the kernel's load-counter wait share at the served level: the latency a higher tier could hide. Ordered by served ms.",
            f"{'kernel':36s} {'ms':>9s} {'share%':>6s} {'tier':>4s}  limiting resources (reduction to the next tier)  | load waits"]
    for k in sorted(ms, key=lambda k: -ms[k]):
        lims = (lint.get(k) or {}).get("_limits") or []
        if all(x["reduction"] is None for x in lims):
            continue   # none, or only fixed limits (the wave cap at the best tier): nothing to trim
        res = ", ".join(f"{x['name']} {x['units']} {x['unit']} -{x['reduction']}" if x["reduction"] is not None
                        else f"{x['name']} (fixed limit, no reduction given)" for x in lims)
        nt = (lint.get(k) or {}).get("_next_tier")
        out.append(f"{k:36s} {ms[k]:9.1f} {100 * ms[k] / total:6.2f} {str(lint[k].get('waves') or '-'):>4s}  {res}: needs all {len(lims)}"
                   + (f" for tier {nt}" if nt else "") + f"  | {latency_wait(workloads, k)}")
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
    ap.add_argument("--top", type=int, default=15, help="(d): levers printed and sent to amdahl")
    ap.add_argument("--amdahl-out", help="(d): write amdahl's input JSON here (default: a temporary file)")
    a = ap.parse_args()
    dev = json.load(open(a.device))
    bw = ((dev.get("profiles") or {}).get(dev.get("default_profile")) or {}).get("bw_practical_gbs")
    srv, total, prompt, steps = served(a.events, a.lanes)
    lint = lint_codes(a.codes, a.package, a.events, a.jobs)
    for k, r in lint.items():
        r["_limit"], tier, vg, r["_limits"], r["_next_tier"] = report_limit(os.path.join(a.codes, k + ".json"))
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
    doc, _, omitted = amdahl_input(lv, heads, noise, a.top)
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        lines, verdicts = run_amdahl(doc, a.amdahl_out or os.path.join(t, "amdahl_in.json"))
    print()
    print("\n".join(section_d(lv, heads, verdicts, omitted, noise, lint, srv, wl, a.top, hw["name"])))
    print(f"\n(d3) amdahl.py on the top {a.top} bounded levers" + (f" ({a.amdahl_out})" if a.amdahl_out else "") +
          (f"; lanes without a noise floor use amdahl's default screen" if set(heads) - set(noise) else "") + ":")
    print("\n".join("    " + x for x in lines))
    print()
    print("\n".join(details(wl, srv)))
    unl = sorted({k for k, _ in srv} - set(lint))
    print(f"\nserved, not linted ({len(unl)}): {' '.join(unl)}")


if __name__ == "__main__":
    main()
