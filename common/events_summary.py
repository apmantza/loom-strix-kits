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
"""
import argparse
import glob
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
    rows = []
    for ln in lanes:
        n, s = per[ln["lane"]]
        f = floor(ln["floor"], m, hw, context=prompt, prompt=prompt)
        if ln["phase"] == "decode":
            meas = s * 1e3 / steps if steps else float("nan")
            unit = "ms/step"
        else:
            meas, unit = s, "s"
        rows.append((ln["lane"], n, meas, unit, f, (f / meas * 100) if f is not None and meas else None))
    return steps, sum(e["duration_ns"] for e in ev) / 1e9, rows


def busy_clock(run):
    """The run's median shader clock while busy (GHz) from its gpu_telemetry summary (tel.txt), or None."""
    p = os.path.join(run, "tel.txt")
    m = re.search(r"while busy: .*?clock ([0-9.]+) ", open(p).read()) if os.path.exists(p) else None
    return float(m.group(1)) if m else None


def cycles_table(runs, res, lanes):
    """Each lane's device time x the run's busy clock (HRX-like E482): at a power cap the clock follows temperature, and a
    compute-bound lane (the f16 GEMMs: 239-241 G cycles at 2.11-2.22 GHz over four starts) keeps its cycles while its seconds move
    5 %. Informational: the gate still judges seconds. The busy clock is the whole run's, prefill-dominated: decode lanes run at
    another clock (~2.5 GHz) and are memory-bound; read their seconds (q187's ABBA: GEMMs 1.8 % -> 0.1 %, matvecs 0.2 % -> 1.7 %). The spread columns compare the runs in seconds and in cycles; a lane whose
    cycle spread is the smaller one is clock-bound across these runs."""
    clk = [busy_clock(r) for r in runs]
    missing = [res[i][0] for i, c in enumerate(clk) if c is None]
    if missing:
        return [f"\ncycles: no busy clock (tel.txt) for {', '.join(missing)}: not printed"]
    out = ["\n| lane (cycles = time x busy clock) | " + " | ".join(f"{r[0]} @ {c:.3f} GHz" for r, c in zip(res, clk))
           + " | spread s | spread cycles |", "|---|" + "---|" * (len(res) + 2)]
    for i, ln in enumerate(lanes):
        unit = res[0][5][i][3]
        secs = [r[5][i][2] for r in res]
        cyc = [m * c for m, c in zip(secs, clk)]   # s x GHz = G cycles; ms/step x GHz = M cycles / step
        cu = "G" if unit == "s" else "M/step"
        sp = (f"{spread(secs):.1f} %", f"{spread(cyc):.1f} %") if len(res) > 1 and all(x == x for x in secs) else ("-", "-")
        out.append(f"| {ln['lane']} | " + " | ".join(f"{x:.2f} {cu}" for x in cyc) + f" | {sp[0]} | {sp[1]} |")
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
    stops = []
    print(f"\nlane gate: reference {', '.join(res[i][0] for i in ref)}; candidate {', '.join(res[i][0] for i in cand)}")
    print("| lane | reference mean | candidate mean | change | noise |")
    print("|---|---|---|---|---|")
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
        print(f"| {name}{mark} | {rm:.3f} {unit} | {cm:.3f} {unit} | {chg:+.2f} % | {noise:.2f} %{' (floor)' if floored else ''} |")
        if i == gi:
            bar = max(noise, a.min_drop_pct) if ok else float("nan")
            if not ok:
                stops.append(f"targeted lane {name} cannot be measured in these runs")
            elif a.no_slower is not None:
                tie = max(noise, a.no_slower)
                if chg > tie:
                    stops.append(f"targeted lane got slower beyond noise: {chg:+.2f} % (bar +{tie:.2f} %)")
            elif not -chg > bar:
                stops.append(f"targeted lane did not move beyond noise: skip the bench ({chg:+.2f} %, bar -{bar:.2f} %)")
        elif a.guard_others is not None:
            if not ok:
                if rm != 0 or cm != 0:   # a lane absent from both sides has nothing to guard
                    stops.append(f"another lane cannot be measured: {name}")
            elif chg > max(noise, a.guard_others):
                stops.append(f"another lane got slower: {name} {chg:+.1f} %")
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
    print("\nLANE GATE: " + (f"STOP ({'; '.join(stops)})" if stops else
                              (f"PASS-TO-BENCH ({names[gi]} not slower beyond noise; the agent bench decides)" if a.no_slower is not None
                               else f"PASS-TO-BENCH ({names[gi]} dropped beyond noise; the agent bench decides)")))
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
        res.append((os.path.basename(os.path.normpath(run)), ev, prompt) + summarize(ev, prompt, d) + (kernels(ev),))
    print("| lane | floor | " + " | ".join(f"{r[0]}: measured (% of floor)" for r in res) + " |")
    print("|---|---|" + "---|" * len(res))
    for i, ln in enumerate(d["lanes"]):
        name, _, _, unit, f, _ = res[0][5][i]
        cells = []
        for r in res:
            _, n, meas, unit, f, pct = r[5][i]
            cells.append(f"{meas:.2f} {unit}" + (f" ({pct:.0f} %)" if pct is not None else "") + f", {n} disp")
        print(f"| {name} | " + (f"{f:.2f} {unit}" if f is not None else "-") + " | " + " | ".join(cells) + " |")
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
