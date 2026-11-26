#!/usr/bin/env python3
"""Lanes vs hardware floors from served per-dispatch events (every lane against its limit).

Each RUN is a directory holding events.jsonl (iree-profile dispatch --dispatch_events --format=jsonl of a loom-serve
--device_profiling_output bundle; events_ab.sh writes it) and one *.req.json (profile_longctx_req.py: the served prompt
size). Dispatches go to the first lane whose regex matches the kernel name (MODEL.json "lanes"); verify steps are the
distinct submissions that ran the verify-attention lane. Several RUNs print side by side (an ABBA's four runs).

usage: events_summary.py MODEL.json RUN [RUN ...] [--top 12] [--split KERNEL[:CUT_MS]] [--device DEV.json [--profile P]]
         [--gate LANE --ref I,J --cand K,L [--min-drop-pct P] [--guard-others PCT] [--expect-kernel REGEX] [--noise-floor FILE.json]]
  --device: floors at a device file's power profile (common/devices/<gfx>.json; ceiling.py), not MODEL.json's hardware.
  --split prints a kernel's dispatch-time clusters (E143: vsplit's aligned 1.62 ms vs straddling 3.01 ms) per run: below
  and above CUT_MS (default: midway between the run's 10th and 90th percentile), count and median each.
  --gate: a one-sided lane gate over an ABBA (E351, E356, E374): a profile can only STOP a candidate, never pass it to go-live
  (under the 85 W cap device time can drop without a wall-clock gain; the seeded agent bench decides). LANE is a substring of
  exactly one lane name (else exit 2); --ref / --cand are 1-based RUN positions (two or more each). Noise per lane = the larger
  of the reference runs' and the candidate runs' spread, (max - min) / mean in %. PASS-TO-BENCH (exit 0) only when the
  candidate mean is below the reference mean by more than max(noise, --min-drop-pct); --guard-others PCT also stops when any
  other lane's candidate mean is above its reference by more than max(its noise, PCT); a lane that cannot be measured stops.
  Prints the (kernel, workgroups) present only in the candidate runs or only in the reference runs; --expect-kernel REGEX
  stops when no candidate-only kernel name matches (the switch did not reach serving). Exit 1 = STOP.
  --no-slower PCT: for a change whose goal is 'not slower' (a correctness fix, an exact layout change; HRX-like E466: q179's NaN fix
  tied +0.20 % inside 0.38 % noise and the speed gate stopped it): the targeted lane passes unless it rose by more than
  max(noise, PCT). The guards on the other lanes and --expect-kernel apply as before.
  --noise-floor FILE.json: a per-lane noise floor in %, {"LANE substring": PCT, ...} (each key names exactly one lane, else
  exit 2): a lane's noise is max(its spread, its floor). Two runs per arm understate a lane's start-to-start spread (HRX-like
  E485: flash2 4.2 % over 8 starts where one ABBA read 3.19 %); measure the floors with an A/A over several starts. Default: none.
Base and excess (HRX-like E625): a lane whose MODEL.json entry declares a work model, "base": {"group_size": N, "rank": R}, runs
  N equal-work dispatches per (submission, kernel, workgroups) group (flash2: one per attention layer of a prefill chunk). Its
  base is, per group, N x the R-th fastest dispatch; its excess = total - base (machine-state slow dispatches: flash2 read 1.2-1.8x
  at layer >= 6 and context >= ~48k, 0.2 s on some nights and 3.4-7 s on others). Every run prints total / base / excess for such
  lanes. With "slow_ratio": Q, a group is quiet in a run when no dispatch exceeds Q x its R-th fastest; the group positions quiet
  in every run are the lane's quiet groups. The gate (verdict(): the state-space table, lane-gate review r1) reads, per lane, the
  total and, when the lane has a base, the base (its own noise, the lane's floor) against the mode's bar:
    a rise of the total or the base beyond its noise stops in every mode (--no-slower, --guard-others, the speed gate): an excess
    the candidate's starts carry may be the candidate's own regression, so neither measure excuses the other;
    the speed gate passes on a total drop; it passes on the base only when the total did not move, the arms' mean excesses differ
    by more than the larger arm's excess range, and the quiet groups dropped beyond their own noise (added spread lowers the R-th
    fastest with no work saved, and leaves no quiet group); an unmeasured total or base stops.
  The table's last column says what was read and why (no base model; base unavailable: a group not of N dispatches; excess
  alike / differs); the verdict line names the excess and the total for the targeted lane. q205: total -0.57 % inside 1.50 % noise,
  base -6.0 %, quiet groups (below ~48k context) -4.8 %: the candidate's excess was 6.8 / 7.0 s vs 4.2 / 3.4 s.
"""
import argparse
import glob
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))   # realpath: the HRX-like skill links this file, not its siblings
from ceiling import floor, hardware  # noqa: E402


def load(run):
    path = os.path.join(run, "events.jsonl")
    ev = []
    for ln in open(path):
        d = json.loads(ln)
        if d.get("type") == "dispatch_event" and d.get("valid", True):
            ev.append(d)
    if not ev:
        raise SystemExit(f"events_summary: no valid dispatch events in {path}")
    reqs = glob.glob(os.path.join(run, "*.req.json"))
    if len(reqs) != 1:
        raise SystemExit(f"events_summary: want exactly one *.req.json in {run}, found {len(reqs)}")
    try:
        prompt = int(json.loads(open(reqs[0]).read().splitlines()[-1])["requests"][-1]["prompt_tokens"])
    except (ValueError, KeyError, IndexError, TypeError) as e:   # the served size decides every floor: no guessing
        raise SystemExit(f"events_summary: cannot read prompt_tokens from {reqs[0]}: {e}")
    return ev, prompt


def classify(ev, d):
    """(prefill submission ids, lane_of(event)): the one classification events_summary and lane_breakdown share.
    A submission (one queue submission: an epoch) that runs a prefill marker kernel, or whose g_token_info covers more than
    prefill_min_tokens tokens, is a prefill epoch; everything in it is prefill work, the drafter and matvecs included (the
    drafter consumes the prompt: per name, the first real run read the drafter at 3.6x its per-step cost). A lane with a
    "submission_regex" owns every dispatch of a submission of its phase that runs a kernel matching it: kernel names cannot
    split call sites that share a kernel (E163: the drafter's propose pass runs g_mv_q4_k_* like the target, and the target
    runs g_mv_q8_0_* like the drafter; by name, 3.6 ms/step moved between the two lanes). Otherwise a dispatch goes to the
    first lane of its phase whose regex matches; each phase ends with a catch-all lane."""
    lanes = d["lanes"]
    mark = re.compile(d["prefill_marker"])
    big = d.get("prefill_min_tokens", 64)
    pre = {e["submission_id"] for e in ev if mark.search(e["key"])
           or (e["key"] == "g_token_info" and (e.get("workgroup_count") or [0])[0] > big)}
    rx = [(ln, re.compile(ln["regex"])) for ln in lanes]
    owners = [(ln, re.compile(ln["submission_regex"])) for ln in lanes if ln.get("submission_regex")]
    owned = {}
    for e in ev:
        ph = "prefill" if e["submission_id"] in pre else "decode"
        for ln, r in owners:
            if ln["phase"] == ph and r.search(e["key"]):
                owned.setdefault(e["submission_id"], ln["lane"])
    memo = {}

    def lane_of(e):
        if e["submission_id"] in owned:
            return owned[e["submission_id"]]
        ph = "prefill" if e["submission_id"] in pre else "decode"
        key = (e["key"], ph)
        if key not in memo:
            memo[key] = next((ln["lane"] for ln, r in rx if ln["phase"] == ph and r.search(e["key"])), None)
            if memo[key] is None:
                raise SystemExit(f"events_summary: {ph} kernel {e['key']} matches no lane (each phase needs a catch-all)")
        return memo[key]
    return pre, lane_of


def verify_steps(ev, pre, lane_of, lanes):
    """verify steps: the distinct decode submissions that ran the verify-attention lane (floor decode_kv); a kit without such a
    lane has none to count (summarize, lint_counters)"""
    vlane = next(ln["lane"] for ln in lanes if ln.get("floor") == "decode_kv")
    return len({e["submission_id"] for e in ev if e["submission_id"] not in pre and lane_of(e) == vlane})


def summarize(ev, prompt, d):
    m, hw, lanes = d["model"], d["hardware"], d["lanes"]
    pre, lane_of = classify(ev, d)
    per = {ln["lane"]: [0, 0.0] for ln in lanes}
    for e in ev:
        p = per[lane_of(e)]
        p[0] += 1
        p[1] += e["duration_ns"] / 1e9
    steps = verify_steps(ev, pre, lane_of, lanes)
    bases = lane_bases(ev, lane_of, lanes)
    rows, based = [], {}
    for ln in lanes:
        n, s = per[ln["lane"]]
        f = floor(ln["floor"], m, hw, context=prompt, prompt=prompt)
        if ln["phase"] == "decode":
            meas = s * 1e3 / steps if steps else float("nan")
            unit = "ms/step"
        else:
            meas, unit = s, "s"
        rows.append((ln["lane"], n, meas, unit, f, (f / meas * 100) if f is not None and meas else None))
        if ln["lane"] in bases:
            b, why, grp = bases[ln["lane"]]
            k = meas / s if s else float("nan")   # seconds -> the lane's unit
            based[len(rows) - 1] = (None, why, None) if b is None else (b * k, None, [(t * k, qt) for t, qt in grp])
    return steps, sum(e["duration_ns"] for e in ev) / 1e9, rows, based


def base_model(ln):
    """(group_size, rank, slow_ratio or None) of a lane's declared work model, None without one; exit 2 on a malformed one"""
    m = ln.get("base")
    if m is None:
        return None
    n, r, q = (m.get("group_size"), m.get("rank"), m.get("slow_ratio")) if isinstance(m, dict) else (None, None, None)
    if (not all(isinstance(x, int) and not isinstance(x, bool) for x in (n, r)) or not 1 <= r <= n
            or not (q is None or (isinstance(q, (int, float)) and not isinstance(q, bool) and q > 1))):
        print(f"events_summary: lane {ln['lane']!r}: base {m!r}: want {{\"group_size\": N, \"rank\": R[, \"slow_ratio\": Q]}}, "
              "integers 1 <= R <= N, Q > 1", file=sys.stderr)
        sys.exit(2)
    return n, r, q


def lane_bases(ev, lane_of, lanes):
    """{lane name: (base s, None, groups) or (None, why, None)} for the lanes with a work model (see the module doc); groups:
    [(group total s, quiet)] in order of first dispatch, quiet = no dispatch above slow_ratio x the group's R-th fastest
    (False for every group without a slow_ratio)"""
    models = {ln["lane"]: base_model(ln) for ln in lanes}
    groups = {name: {} for name, m in models.items() if m}
    for e in ev:
        g = groups.get(lane_of(e))
        if g is not None:
            g.setdefault((e["submission_id"], e["key"], tuple(e.get("workgroup_count", []))), []).append(e["duration_ns"])
    out = {}
    for name, g in groups.items():
        n, r, q = models[name]
        bad = [(k, len(v)) for k, v in g.items() if len(v) != n]
        if not g:
            out[name] = (None, "no dispatches", None)
        elif bad:
            (_, key, wg), cnt = bad[0]
            out[name] = (None, f"model does not fit: {len(bad)} of {len(g)} (submission, kernel, workgroups) groups hold "
                               f"other than {n} dispatches (first: {key} {list(wg)} x {cnt})", None)
        else:
            srt = [sorted(v) for v in g.values()]   # dicts keep insertion order: groups in order of first dispatch
            out[name] = (sum(n * x[r - 1] for x in srt) / 1e9, None,
                         [(sum(x) / 1e9, q is not None and x[-1] <= q * x[r - 1]) for x in srt])
    return out


def busy_clock(run):
    """(the run's median shader clock while busy in GHz, its source) or (None, None): from the recorded samples (tel.tsv, 1 MHz),
    else from the gpu_telemetry summary (tel.txt; 2 decimals before 11-26, 0.46 % steps against a 0.27 % lane floor: HRX-like E624)"""
    tsv, txt = os.path.join(run, "tel.tsv"), os.path.join(run, "tel.txt")
    if os.path.exists(tsv) and os.path.getsize(tsv):   # imported here: test_events_ab.sh stubs the CLI and leaves tel.tsv empty
        from gpu_telemetry import busy_clock as tel_busy_clock
        c = tel_busy_clock(tsv)
        if c:
            return c, "tel.tsv"
    m = re.search(r"while busy: .*?clock ([0-9.]+) ", open(txt).read()) if os.path.exists(txt) else None
    if not m:
        return None, None
    return float(m.group(1)), ("tel.txt" if len(m.group(1).partition(".")[2]) >= 3 else "coarse tel.txt")


def base_table(res, lanes):
    """total / base / excess per run for every lane with a work model (module doc), or no lines"""
    out = []
    for i, ln in enumerate(lanes):
        if i not in res[0][7]:
            continue
        if not out:
            out = ["\n| lane (base: per group of N equal-work dispatches, N x the R-th fastest; excess = total - base) | "
                   + " | ".join(r[0] for r in res) + " |", "|---|" + "---|" * len(res)]
        n, r, _ = base_model(ln)
        cells = []
        for x in res:
            tot, unit = x[5][i][2], x[5][i][3]
            b, why, _ = x[7][i]
            cells.append(f"{tot:.2f} / {b:.2f} / {tot - b:.2f} {unit}" if b is not None else f"{tot:.2f} {unit}, no base: {why}")
        out.append(f"| {ln['lane']}: total / base / excess (N {n}, R {r}) | " + " | ".join(cells) + " |")
    return out


def cycles_table(runs, res, lanes):
    """Each lane's device time x the run's busy clock (HRX-like E482): at a power cap the clock follows temperature, and a
    compute-bound lane (the f16 GEMMs: 239-241 G cycles at 2.11-2.22 GHz over four starts) keeps its cycles while its seconds move
    5 %. Informational: the gate still judges seconds. The busy clock is the whole run's, prefill-dominated: decode lanes run at
    another clock (~2.5 GHz) and are memory-bound; read their seconds (q187's ABBA: GEMMs 1.8 % -> 0.1 %, matvecs 0.2 % -> 1.7 %). The spread columns compare the runs in seconds and in cycles; a lane whose
    cycle spread is the smaller one is clock-bound across these runs."""
    got = [busy_clock(r) for r in runs]
    clk = [c for c, _ in got]
    missing = [res[i][0] for i, c in enumerate(clk) if c is None]
    if missing:
        return [f"\ncycles: no busy clock (tel.tsv / tel.txt) for {', '.join(missing)}: not printed"]
    coarse = [res[i][0] for i, (_, src) in enumerate(got) if src == "coarse tel.txt"]
    out = ["\n| lane (cycles = time x busy clock) | " + " | ".join(f"{r[0]} @ {c:.3f} GHz" for r, c in zip(res, clk))
           + " | spread s | spread cycles |", "|---|" + "---|" * (len(res) + 2)]
    for i, ln in enumerate(lanes):
        unit = res[0][5][i][3]
        secs = [r[5][i][2] for r in res]
        cyc = [m * c for m, c in zip(secs, clk)]   # s x GHz = G cycles; ms/step x GHz = M cycles / step
        cu = "G" if unit == "s" else "M/step"
        sp = (f"{spread(secs):.1f} %", f"{spread(cyc):.1f} %") if len(res) > 1 and all(x == x for x in secs) else ("-", "-")
        out.append(f"| {ln['lane']} | " + " | ".join(f"{x:.2f} {cu}" for x in cyc) + f" | {sp[0]} | {sp[1]} |")
    if coarse:
        out.append(f"clock from a 2-decimal tel.txt (no tel.tsv) for {', '.join(coarse)}: steps of ~0.46 %, coarser than a 0.3 % lane floor")
    return out


def clusters(ev, kernel, cut):
    xs = sorted(e["duration_ns"] / 1e6 for e in ev if e["key"] == kernel)
    if not xs:
        return f"{kernel}: no dispatches"
    c = cut if cut is not None else (xs[len(xs) // 10] + xs[9 * len(xs) // 10]) / 2   # p10/p90: outliers do not move it
    lo, hi = [x for x in xs if x < c], [x for x in xs if x >= c]
    f = lambda v: f"{len(v)} x {st.median(v):.3f} ms" if v else "0"
    return f"{kernel} (cut {c:.2f} ms): below {f(lo)}, above {f(hi)}, mean {st.mean(xs):.3f} ms"


def spread(xs):
    m = st.mean(xs)
    return (max(xs) - min(xs)) / m * 100 if m > 0 else float("nan")


def noise_floors(path, names):
    """--noise-floor: {lane index: floor %} from {"LANE substring": PCT}; None (with a message) when the file is unreadable or
    a key does not name exactly one lane or a value is not a number >= 0: a floor that matches nothing must not pass silently"""
    try:
        raw = json.load(open(path))
    except (OSError, ValueError) as e:
        print(f"events_summary: --noise-floor {path}: {e}", file=sys.stderr)
        return None
    if not isinstance(raw, dict):
        print(f"events_summary: --noise-floor {path}: want an object {{lane substring: pct}}", file=sys.stderr)
        return None
    out = {}
    for key, v in raw.items():
        hits = [i for i, n in enumerate(names) if key in n]
        if len(hits) != 1 or isinstance(v, bool) or not isinstance(v, (int, float)) or not v >= 0:
            print(f"events_summary: --noise-floor {path}: {key!r}: {v!r} (matches {len(hits)} lanes; want one lane and a number >= 0)",
                  file=sys.stderr)
            return None
        out[hits[0]] = float(v)
    return out


def on_base(res, i, ref, cand, rv, cv):
    """lane i's base, excess and quiet-group figures over the arms (module doc): {"decides", "why", "rb", "cb", "re", "ce", "esp",
    "rq", "cq", "nq", "ng"}; rb/cb (bases per reference / candidate run, the lane's unit) are None when any run has no base.
    decides: the arms' mean excesses differ by more than the larger arm's excess range (the base may then carry a speed-up).
    Quiet groups: the group positions quiet in every run (none when the runs hold different group counts); rq/cq their totals."""
    got = [res[j][7].get(i) for j in ref + cand]
    out = {"decides": False, "rb": None, "cb": None, "re": None, "ce": None, "esp": None, "rq": None, "cq": None, "nq": 0, "ng": 0}
    if any(g is None for g in got):
        return dict(out, why="no base model")
    if any(g[0] is None for g in got):
        return dict(out, why="base unavailable: " + next(g[1] for g in got if g[0] is None))
    rb, cb = [g[0] for g in got[:len(ref)]], [g[0] for g in got[len(ref):]]
    re_, ce = [t - b for t, b in zip(rv, rb)], [t - b for t, b in zip(cv, cb)]
    esp = max(max(re_) - min(re_), max(ce) - min(ce))
    decides = abs(st.mean(ce) - st.mean(re_)) > esp
    grs = [g[2] for g in got]
    qi = [k for k in range(len(grs[0])) if all(x[k][1] for x in grs)] if len({len(x) for x in grs}) == 1 else []
    tq = [sum(x[k][0] for k in qi) for x in grs]
    return dict(out, decides=decides, why="excess differs" if decides else "excess alike", rb=rb, cb=cb, re=re_, ce=ce, esp=esp,
                rq=tq[:len(ref)], cq=tq[len(ref):], nq=len(qi), ng=len(grs[0]))


def verdict(mode, t, b, e, q):
    """The gate's state-space table (review r1, round 2; test_profile_tools SPEC). mode: 'speed' (the targeted lane must drop),
    'no_slower' (the targeted lane must not rise), 'guard' (another lane must not rise). t / b: the total's / the base's move
    beyond its noise, 'up' / 'flat' / 'down' / 'unmeasured' (b None: no base). e: the excess 'alike' / 'differs'; q: the quiet
    groups 'down' or not. Returns (PASS or STOP, the measure that decided, reason).
    A rise of the total or the base stops in every mode: an excess the candidate's starts carry may be the candidate's own (review
    r1 F1). The base passes a speed-up only when the total did not move, the excess differs between the arms, and the groups that
    carry no excess in any run dropped too (review r1 F2: added spread lowers the R-th fastest with no work saved, and leaves no
    quiet group)."""
    if t == "unmeasured" or b == "unmeasured":
        return "STOP", ("total" if t == "unmeasured" else "base"), "unmeasured"
    if t == "up":
        return "STOP", "total", "rose"
    if b == "up":
        return "STOP", "base", "rose"
    if mode != "speed":
        return "PASS", "total", "not slower"
    if t == "down":
        return "PASS", "total", "dropped"
    if b == "down" and e == "differs" and q == "down":
        return "PASS", "base", "dropped"
    return "STOP", "total", "did not move"


def direction(chg, noise, up_bar, down_bar):
    """'unmeasured' (nan), 'up' beyond up_bar, 'down' beyond down_bar, else 'flat'"""
    if not (chg == chg and noise == noise):
        return "unmeasured"
    return "up" if chg > up_bar else ("down" if -chg > down_bar else "flat")


def lane_gate(a, d, res):
    """One-sided lane gate (see the module doc): returns the exit code, 0 = PASS-TO-BENCH, 1 = STOP."""
    names = [ln["lane"] for ln in d["lanes"]]
    hits = [i for i, n in enumerate(names) if a.gate in n]
    if len(hits) != 1:
        print(f"events_summary: --gate {a.gate!r} matches {len(hits)} lanes ({[names[i] for i in hits]}), want exactly one",
              file=sys.stderr)
        return 2
    try:
        ref = [int(x) - 1 for x in a.ref.split(",")]
        cand = [int(x) - 1 for x in a.cand.split(",")]
    except (AttributeError, ValueError):
        print("events_summary: --gate needs --ref and --cand as comma-separated run positions (1-based)", file=sys.stderr)
        return 2
    if (len(ref) < 2 or len(cand) < 2 or set(ref) & set(cand)
            or not all(0 <= i < len(res) for i in ref + cand)):
        print(f"events_summary: --ref/--cand need two or more distinct run positions each, disjoint, within 1..{len(res)}",
              file=sys.stderr)
        return 2
    gi = hits[0]
    floors = noise_floors(a.noise_floor, names) if a.noise_floor else {}
    if floors is None:
        return 2
    stops, notes, passed = [], [], ""
    print(f"\nlane gate: reference {', '.join(res[i][0] for i in ref)}; candidate {', '.join(res[i][0] for i in cand)}")
    print("| lane | reference mean | candidate mean | change | noise | measures |")
    print("|---|---|---|---|---|---|")
    pct = lambda r, c: (st.mean(c) / st.mean(r) - 1) * 100 if st.mean(r) > 0 else float("nan")
    for i, name in enumerate(names):
        rv = [res[r][5][i][2] for r in ref]
        cv = [res[c][5][i][2] for c in cand]
        unit = res[0][5][i][3]
        rm, cm = st.mean(rv), st.mean(cv)
        noise = max(spread(rv), spread(cv))
        floored = i in floors and floors[i] > noise   # False for an unmeasured lane (nan): it stays unmeasured
        if floored:
            noise = floors[i]
        # refused, not passed: an unmeasured lane (nan != nan), or a lane missing from a run (a renamed kernel fell into
        # another lane: its 'drop' would be 100 %)
        ok = rm > 0 and noise == noise and cm == cm and all(res[j][5][i][1] for j in ref + cand)
        chg = (cm - rm) / rm * 100 if ok else float("nan")
        mark = " (gated)" if i == gi else ""
        base = on_base(res, i, ref, cand, rv, cv)
        print(f"| {name}{mark} | {rm:.3f} {unit} | {cm:.3f} {unit} | {chg:+.2f} % | {noise:.2f} %{' (floor)' if floored else ''} "
              f"| {'total and base' if base['rb'] is not None else 'total'} ({base['why']}) |")
        flo = floors.get(i, 0.0)
        bchg = bnoise = qchg = qnoise = float("nan")
        if base["rb"] is not None:
            rb, cb, re_, ce = base["rb"], base["cb"], base["re"], base["ce"]
            bnoise, bchg = max(spread(rb), spread(cb), flo), pct(rb, cb)
            print(f"| {name}: base | {st.mean(rb):.3f} {unit} | {st.mean(cb):.3f} {unit} | {bchg:+.2f} % (ratio {1 + bchg / 100:.4f}) "
                  f"| {bnoise:.2f} % | |")
            print(f"| {name}: excess | {st.mean(re_):.3f} {unit} | {st.mean(ce):.3f} {unit} | {st.mean(ce) - st.mean(re_):+.3f} {unit} "
                  f"| range {base['esp']:.3f} {unit} | |")
            if base["nq"]:
                qnoise, qchg = max(spread(base["rq"]), spread(base["cq"]), flo), pct(base["rq"], base["cq"])
                print(f"| {name}: quiet groups ({base['nq']} of {base['ng']}) | {st.mean(base['rq']):.3f} {unit} | "
                      f"{st.mean(base['cq']):.3f} {unit} | {qchg:+.2f} % | {qnoise:.2f} % | |")
            if i == gi:
                notes.append(f"{name}: excess {st.mean(re_):.2f} -> {st.mean(ce):.2f} {unit} "
                             f"{'differs beyond' if base['decides'] else 'alike within'} its range {base['esp']:.2f} {unit}; total {chg:+.2f} %")
        if i == gi:
            mode = "speed" if a.no_slower is None else "no_slower"
        elif a.guard_others is not None:
            mode = "guard"
            if not ok and rm == 0 and cm == 0:   # a lane absent from both sides has nothing to guard
                continue
        else:
            continue
        lim = {"speed": 0.0, "no_slower": a.no_slower, "guard": a.guard_others}[mode]
        drop = a.min_drop_pct if mode == "speed" else 0.0
        t = direction(chg if ok else float("nan"), noise, max(noise, lim), max(noise, drop))
        b = None if base["rb"] is None else direction(bchg, bnoise, max(bnoise, lim), max(bnoise, drop))
        q = direction(qchg, qnoise, max(qnoise, lim), max(qnoise, drop)) if base["nq"] else "none"
        got, measure, why = verdict(mode, t, b, "differs" if base["decides"] else "alike", q)
        if got == "PASS":
            if mode == "speed" and measure == "base":
                passed = (f"{name} dropped beyond noise on its base ({bchg:+.2f} %, bar -{max(bnoise, drop):.2f} %; quiet groups "
                          f"{qchg:+.2f} %, bar -{max(qnoise, drop):.2f} %)")
            elif mode == "speed":
                passed = f"{name} dropped beyond noise ({chg:+.2f} %, bar -{max(noise, drop):.2f} %)"
            elif mode == "no_slower":
                passed = (f"{name} not slower beyond noise ({chg:+.2f} %, bar +{max(noise, lim):.2f} %"
                          + (f"; base {bchg:+.2f} %, bar +{max(bnoise, lim):.2f} %)" if b is not None else ")"))
            continue
        if mode == "guard":
            stops.append(f"another lane cannot be measured: {name}" if why == "unmeasured" else
                         f"another lane got slower on its base: {name} {bchg:+.1f} %" if measure == "base" else
                         f"another lane got slower: {name} {chg:+.1f} %")
        elif why == "unmeasured":
            stops.append(f"targeted lane {name} cannot be measured in these runs" if measure == "total" else
                         f"targeted lane {name}'s base cannot be measured in these runs")
        elif mode == "no_slower":
            stops.append(f"targeted lane got slower beyond noise on its base: {bchg:+.2f} % (bar +{max(bnoise, lim):.2f} %)"
                         if measure == "base" else f"targeted lane got slower beyond noise: {chg:+.2f} % (bar +{max(noise, lim):.2f} %)")
        elif measure == "base":
            stops.append(f"targeted lane's base rose beyond noise: {bchg:+.2f} % (bar +{bnoise:.2f} %)")
        else:
            msg = f"targeted lane did not move beyond noise: skip the bench ({chg:+.2f} %, bar -{max(noise, drop):.2f} %)"
            if b == "down":
                msg += (f"; base {bchg:+.2f} % not taken: " + ("the total rose" if t == "up" else "excess alike" if not base["decides"]
                        else "no quiet groups" if q == "none" else f"quiet groups {qchg:+.2f} % (bar -{max(qnoise, drop):.2f} %)"))
            stops.append(msg)
    kset = lambda idx: {k for j in idx for k in res[j][6]}
    rk, ck = kset(ref), kset(cand)
    for label, only, idx in (("candidate-only", ck - rk, cand), ("reference-only", rk - ck, ref)):
        print(f"\n{label} (kernel, workgroups): {len(only)}")
        for k, wg in sorted(only, key=lambda kw: -sum(res[j][6].get(kw, (0, 0.0))[1] for j in idx)):
            n = sum(res[j][6].get((k, wg), (0, 0.0))[0] for j in idx)
            ms = sum(res[j][6].get((k, wg), (0, 0.0))[1] for j in idx)
            print(f"  {k:36s} {str(list(wg)):18s} n {n:6d}  {ms / 1e3:8.2f} s (summed over {len(idx)} runs)")
    if a.expect_kernel is not None:
        rx = re.compile(a.expect_kernel)
        if not any(rx.search(k) for k, _ in ck - rk):
            stops.append(f"no candidate-only kernel matches {a.expect_kernel!r}: the switch did not reach serving")
    print("\nLANE GATE: " + (f"STOP ({'; '.join(stops)})" if stops else f"PASS-TO-BENCH ({passed}; the agent bench decides)")
          + (f" [{'; '.join(notes)}]" if notes else ""))
    return 1 if stops else 0


def kernels(ev):
    """(kernel, workgroups) -> [dispatches, total ms]: the top-kernels table's data and the gate's dispatch evidence."""
    agg = {}
    for e in ev:
        v = agg.setdefault((e["key"], tuple(e.get("workgroup_count", []))), [0, 0.0])
        v[0] += 1
        v[1] += e["duration_ns"] / 1e6
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--split", action="append", default=[])
    ap.add_argument("--device", help="device file (common/devices/<gfx>.json): floors at its profile, not MODEL.json's hardware")
    ap.add_argument("--profile", help="the device file's profile (default: its default_profile)")
    ap.add_argument("--gate", help="one-sided lane gate on the lane whose name contains this (see the module doc)")
    ap.add_argument("--ref", help="--gate: reference run positions, 1-based, e.g. 1,4")
    ap.add_argument("--cand", help="--gate: candidate run positions, 1-based, e.g. 2,3")
    ap.add_argument("--min-drop-pct", type=float, default=0.0, help="--gate: the drop must also exceed this (%%)")
    ap.add_argument("--guard-others", type=float, help="--gate: stop when another lane rises by more than max(noise, this %%)")
    ap.add_argument("--no-slower", type=float, help="--gate: pass unless the targeted lane rose by more than max(noise, this %%) (a fix)")
    ap.add_argument("--expect-kernel", help="--gate: stop unless a candidate-only kernel name matches this regex")
    ap.add_argument("--noise-floor", help="--gate: JSON {lane substring: pct}: each lane's noise is at least its floor")
    a = ap.parse_args()
    d = json.load(open(a.model))
    d["hardware"] = hardware(d, a.device, a.profile)   # MODEL.json's block unless --device
    res = []
    for run in a.runs:
        ev, prompt = load(run)
        steps, busy, rows, based = summarize(ev, prompt, d)
        res.append((os.path.basename(os.path.normpath(run)), ev, prompt, steps, busy, rows, kernels(ev), based))
    print("| lane | floor | " + " | ".join(f"{r[0]}: measured (% of floor)" for r in res) + " |")
    print("|---|---|" + "---|" * len(res))
    for i, ln in enumerate(d["lanes"]):
        name, _, _, unit, f, _ = res[0][5][i]
        cells = []
        for r in res:
            _, n, meas, unit, f, pct = r[5][i]
            cells.append(f"{meas:.2f} {unit}" + (f" ({pct:.0f} %)" if pct is not None else "") + f", {n} disp")
        print(f"| {name} | " + (f"{f:.2f} {unit}" if f is not None else "-") + " | " + " | ".join(cells) + " |")
    print("\n".join(base_table(res, d["lanes"])))
    print("\n".join(cycles_table(a.runs, res, d["lanes"])))
    for r in res:
        print(f"\n{r[0]}: prompt {r[2]} tokens, {r[3]} verify steps, device busy {r[4]:.1f} s")
        for spec in a.split:
            k, _, c = spec.partition(":")
            print("  " + clusters(r[1], k, float(c) if c else None))
    if a.top:
        ev = res[0][1]
        agg = {}
        for e in ev:
            key = (e["key"], tuple(e.get("workgroup_count", [])))
            agg.setdefault(key, []).append(e["duration_ns"] / 1e6)
        tot = sum(sum(v) for v in agg.values())
        print(f"\ntop {a.top} (kernel, workgroups) in {res[0][0]}:")
        for (k, wg), v in sorted(agg.items(), key=lambda x: -sum(x[1]))[:a.top]:
            print(f"  {k:36s} {str(list(wg)):18s} n {len(v):6d}  {sum(v) / 1e3:8.2f} s  median {st.median(v):9.3f} ms  {sum(v) / tot * 100:5.1f} %")
    if a.gate is not None:
        sys.exit(lane_gate(a, d, res))


if __name__ == "__main__":
    main()
