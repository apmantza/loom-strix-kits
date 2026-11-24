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
  --device FILE   common/devices/<gfx>.json: the DRAM read ceiling of its default profile (serial's bandwidth guard)

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
usage: lint_counters.py --codes DIR --events RUN --lanes LANES.json [--package PKG] [--counters SPEC ...] [--device DEV.json]
       [--jobs N]"""
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
    """(rows, total ms): rows = {(kernel, phase): {"ms", "calls", "shapes": {workgroups}}} from one profiled request"""
    ev, prompt = events_summary.load(run)
    pre, _ = events_summary.classify(ev, json.load(open(lanes)))
    out = {}
    for e in ev:
        r = out.setdefault((e["key"], "prefill" if e["submission_id"] in pre else "decode"), {"ms": 0.0, "calls": 0, "shapes": set()})
        r["ms"] += e["duration_ns"] / 1e6
        r["calls"] += 1
        wg = e.get("workgroup_count") or [0]
        r["shapes"].add(wg[0] * (wg[1] if len(wg) > 1 else 1) * (wg[2] if len(wg) > 2 else 1))
    return out, sum(r["ms"] for r in out.values()), prompt


# ---------------------------------------------------------------- lint + compile report
def report_limit(js):
    """(limiting resource, waves per SIMD, VGPRs) from a Loom summary report, Nones when absent"""
    if not os.path.exists(js):
        return None, None, None
    tr = json.load(open(js)).get("target_resources") or {}
    lim = tr.get("limiting_resource")
    return (lim.split(".")[-1] if lim else None, (tr.get("residency") or {}).get("current_tier"),
            ((tr.get("vector") or {}).get("final") or {}).get("register_count"))


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
                did = r.get("dispatch_id") or r.get("correlation_id")
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


def counters_cell(workloads, kernel):
    """the table's counter cell: dominant wait, bank %, GB/s from the best usable workload (served level, most figures)"""
    us = usable(workloads, kernel)
    other = [w["label"] + " (" + w["kernels"][kernel]["ident"] + ")" for w in workloads
             if kernel in w["kernels"] and w["kernels"][kernel]["ident"] is not None]
    if not us:
        return "no counters" + (f"; not used: {'; '.join(other)}" if other else ""), ""
    us.sort(key=lambda x: (not x[2], -sum(not isinstance(x[1][c], str) for c in WAITS)))
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
    """the three report sections as lists of lines (pure: tested on fixtures)"""
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
    a = ap.parse_args()
    dev = json.load(open(a.device))
    bw = ((dev.get("profiles") or {}).get(dev.get("default_profile")) or {}).get("bw_practical_gbs")
    srv, total, prompt = served(a.events, a.lanes)
    lint = lint_codes(a.codes, a.package, a.events, a.jobs)
    for k, r in lint.items():
        r["_limit"], tier, vg = report_limit(os.path.join(a.codes, k + ".json"))
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
    print()
    print("\n".join(details(wl, srv)))
    unl = sorted({k for k, _ in srv} - set(lint))
    print(f"\nserved, not linted ({len(unl)}): {' '.join(unl)}")


if __name__ == "__main__":
    main()
