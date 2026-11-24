#!/usr/bin/env python3
"""Amdahl screen for speed levers: what each lever can give the whole step or prompt, which clear the noise floor, and
what the non-overlapping set gives together. Model-agnostic: every number comes from the input file.

Input (JSON): one phase object, or {"phases": [phase, ...]}. A phase:
  {"phase": "decode", "unit": "ms/step", "wall": 115.1, "screen_pct": 1.0,
   "candidates": [{"name": "...", "target": 79.4, "to": 69.1 | "reduction": 10.3 | "reduction_pct": 13,
                   "overlap_group": "matvecs", "enabler": false, "src": "LEDGER E301"}, ...],
   "expect": {"wall": 115, "after": 93, "tolerance_pct": 3}}
- wall: the measured time of the whole step / prompt (served events: the sum of every lane, catch-all included).
- target: the time of the part the lever acts on; its saving is target - to, or reduction, or target x reduction_pct / 100.
- overlap_group: levers that claim the same time (two rewrites of one kernel, a fusion and a retune of the same pass).
  The combined gain takes the largest saving per group, never the sum; levers in different groups add.
- enabler: a lever with no saving of its own that others need; it is listed and kept, never screened out.
- screen_pct: a lever passes when it cuts at least this share of the wall (default --screen, 1 %): set it to the A/B's
  noise floor (on the HRX-like agent bench ~1 % decode, ~0.2 % prefill: LEDGER E374), or a real gain cannot be told apart.
- expect (optional): totals the input should reproduce (a limits table's 'now' and 'limit' sums); a miss beyond
  tolerance_pct (default 2) is a discrepancy and fails the run. "after" is checked against every lever (a table takes each
  part to its floor).

Whole-request headroom ("cells", optional, beside or instead of "phases"): per served cell, how far each orthogonal
axis sits from the device's limit at its power profile (--device common/devices/<gfx>.json [--profile], ceiling.py's
loader), each as a multiplier ceiling (limit / measured, >= 1):
  {"cell": "decode 81k", "kind": "decode", "tokens_per_s": 27.3, "tokens_per_step": 3.25,
   "min_bytes_per_step": 2.16e10, "measured_bytes_per_step": null, "busy_share": 0.997,
   "published_tokens_per_step": 4.1,
   "stack": [{"name": "...", "axis": "kernel_bandwidth", "multiplier": 1.08}, ...]}
  {"cell": "prefill 81k", "kind": "prefill", "tokens_per_s": 414, "active_params": 2.435e10, "extra_flops": 1.3e15,
   "prompt_tokens": 81430, "busy_share": 0.997}
- decode: bandwidth efficiency = min_bytes_per_step x steps/s / capped bandwidth (steps/s = tokens_per_s / tokens_per_step),
  split into orthogonal axes whose product is its inverse: traffic (measured / minimum bytes), kernel_bandwidth (capped
  bandwidth / bytes moved per busy second), dispatch (1 / busy_share). Without measured bytes, traffic and kernel_bandwidth
  are reported as one axis (bandwidth). tokens_per_step: published / ours (acceptance, a ceiling only if the content matches).
- prefill: compute = capped TFLOP/s / achieved, achieved = (2 x active_params x tokens/s + extra_flops x tokens/s /
  prompt_tokens); active_params for an MoE is the routed top-k, not the total (make_lanes.py's linear_params). dispatch likewise.
- min_bytes_per_step: the decode floors' bytes (ceiling.py's floors x its bandwidth); a missing input prints unknown.
- stack: candidate multipliers by axis; the best per axis counts (the overlap-group rule of the phases), axes multiply.
  A candidate above its axis's ceiling claims more than the device allows: FAIL.

usage: amdahl.py CANDIDATES.json [--screen 1.0] [--device DEV.json [--profile P]] [--json OUT.json]
Last line `amdahl: PASS` (every phase has a lever that clears its screen, every expect holds, no stack exceeds a
ceiling), `amdahl: FAIL`, or `amdahl: INVALID` (exit 0 / 1 / 2).
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ceiling import hardware  # noqa: E402


class Invalid(Exception):
    pass


def num(c, key, name):
    v = c.get(key)
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
        raise Invalid(f"{name}: {key} must be a finite number, got {v!r}")
    return float(v)


def saving(c, name):
    target = num(c, "target", name)
    forms = [k for k in ("to", "reduction", "reduction_pct") if k in c]
    if c.get("enabler") and not forms:
        return target, 0.0
    if len(forms) != 1:
        raise Invalid(f"{name}: give exactly one of to / reduction / reduction_pct (got {forms or 'none'})")
    k = forms[0]
    v = num(c, k, name)
    s = target - v if k == "to" else v if k == "reduction" else target * v / 100
    if target < 0 or s < 0 or s > target + 1e-9:
        raise Invalid(f"{name}: saving {s:g} outside [0, target {target:g}]")
    return target, s


def best_per_group(items, group, value):
    """{group: the item with the largest value}: levers claiming the same time (or the same axis) count once."""
    best = {}
    for it in items:
        g = group(it)
        if g not in best or value(it) > value(best[g]):
            best[g] = it
    return best


def screen_phase(ph, default_screen):
    wall = num(ph, "wall", ph.get("phase", "phase"))
    if wall <= 0:
        raise Invalid(f"{ph.get('phase')}: wall must be positive")
    scr = float(ph.get("screen_pct", default_screen))
    cands = ph.get("candidates")
    if not isinstance(cands, list) or not cands:
        raise Invalid(f"{ph.get('phase')}: candidates must be a non-empty list")
    rows, names = [], set()
    for c in cands:
        name = str(c.get("name", ""))
        if not name or name in names:
            raise Invalid(f"candidate name {name!r} missing or repeated")
        names.add(name)
        target, s = saving(c, name)
        if target > wall + 1e-9:
            raise Invalid(f"{name}: target {target:g} exceeds the wall {wall:g}")
        group = str(c.get("overlap_group") or name)
        cut = s / wall * 100
        enabler = bool(c.get("enabler"))
        ok = enabler or cut >= scr
        rows.append({"name": name, "group": group, "target": target, "saving": s, "cut_pct": cut,
                     "speedup_pct": (wall / (wall - s) - 1) * 100 if s < wall else math.inf,
                     "enabler": enabler, "passes": ok, "src": c.get("src", "")})
    pick = lambda rs: sum(r["saving"] for r in best_per_group(rs, lambda r: r["group"], lambda r: r["saving"]).values())
    tot_pass, tot_all = pick([r for r in rows if r["passes"]]), pick(rows)
    if tot_all >= wall:
        raise Invalid(f"{ph.get('phase')}: the groups save {tot_all:g} of a {wall:g} wall: overlapping levers in different groups?")
    return {"phase": ph.get("phase", ""), "unit": ph.get("unit", ""), "wall": wall, "screen_pct": scr, "rows": rows,
            "combined_saving": tot_pass, "after": wall - tot_pass, "combined_saving_all": tot_all, "after_all": wall - tot_all,
            "passing": sum(r["passes"] and not r["enabler"] for r in rows)}


AXES = ("traffic", "kernel_bandwidth", "bandwidth", "dispatch", "compute", "tokens_per_step")


def opt(c, key, name, lo=0.0):
    if c.get(key) is None:
        return None
    v = num(c, key, name)
    if v <= lo:
        raise Invalid(f"{name}: {key} must be > {lo:g}")
    return v


def headroom_cell(c, hw):
    """[(axis, multiplier ceiling or None, why)] for one served cell."""
    name, kind = str(c.get("cell", "")), c.get("kind")
    if not name or kind not in ("decode", "prefill"):
        raise Invalid(f"cell {name!r}: needs a name and kind decode | prefill")
    tps = opt(c, "tokens_per_s", name)
    busy = opt(c, "busy_share", name)
    if busy is not None and busy > 1:
        raise Invalid(f"{name}: busy_share above 1")
    bw, tf = hw["bw_practical_gbs"] * 1e9, hw["wmma_f16_tflops"] * 1e12
    ax = []
    if kind == "decode":
        tpstep, minb, meas = opt(c, "tokens_per_step", name), opt(c, "min_bytes_per_step", name), opt(c, "measured_bytes_per_step", name)
        if meas is not None and minb is not None and meas < minb:
            raise Invalid(f"{name}: measured bytes per step below the minimum")
        if None in (tps, tpstep, minb):
            ax.append(("bandwidth", None, "unknown: needs tokens_per_s, tokens_per_step, min_bytes_per_step"))
        else:
            sps = tps / tpstep
            eff = minb * sps / bw
            b = busy or 1.0
            if meas is None:
                ax.append(("bandwidth", 1 / eff / (1 / b), f"minimum bytes at {minb * sps / 1e9:.1f} GB/s wall = "
                           f"{eff * 100:.1f} % of {bw / 1e9:.0f} GB/s; traffic vs kernel bandwidth unknown (no measured bytes)"))
            else:
                ax.append(("traffic", meas / minb, f"{meas / 1e9:.2f} GB moved per step vs {minb / 1e9:.2f} GB minimum"))
                ax.append(("kernel_bandwidth", bw / (meas * sps / b), f"{meas * sps / b / 1e9:.1f} GB/s while busy"))
        pub = opt(c, "published_tokens_per_step", name)
        ax.append(("tokens_per_step", (pub / tpstep if pub and tpstep else None),
                   f"published {pub:g} vs {tpstep:g} (a ceiling only on matching content)" if pub and tpstep
                   else "unknown: needs published_tokens_per_step"))
    else:
        ap_, extra, prompt = opt(c, "active_params", name), c.get("extra_flops") or 0.0, opt(c, "prompt_tokens", name)
        if None in (tps, ap_) or (extra and prompt is None):
            ax.append(("compute", None, "unknown: needs tokens_per_s, active_params (and prompt_tokens with extra_flops)"))
        else:
            ach = 2 * ap_ * tps + (extra * tps / prompt if extra else 0.0)
            ax.append(("compute", tf / (ach / (busy or 1.0)), f"{ach / 1e12:.1f} TFLOP/s wall of {tf / 1e12:.1f}"))
    ax.append(("dispatch", 1 / busy if busy else None, f"busy {busy * 100:.1f} %" if busy else "unknown: needs busy_share"))
    return ax


def stack_cell(c, ceil):
    """Best stack candidate per axis (best_per_group), product across axes; [(axis, item, over ceiling?)]."""
    items = c.get("stack") or []
    for it in items:
        if it.get("axis") not in AXES or num(it, "multiplier", str(it.get("name"))) < 1:
            raise Invalid(f"stack item {it.get('name')!r}: axis in {AXES}, multiplier >= 1")
    best = best_per_group(items, lambda it: it["axis"], lambda it: it["multiplier"])
    out = []
    for a_, it in best.items():
        cap = ceil.get(a_) if a_ in ceil else ceil.get("bandwidth") if a_ in ("traffic", "kernel_bandwidth") else None
        out.append((a_, it, cap is not None and it["multiplier"] > cap * (1 + 1e-6), len([x for x in items if x["axis"] == a_]) - 1))
    return out


def check_expect(ph, res):
    exp, out = ph.get("expect") or {}, []
    tol = float(exp.get("tolerance_pct", 2))
    for key, got in (("wall", res["wall"]), ("after", res["after_all"])):
        if key in exp:
            want = float(exp[key])
            dev = (got - want) / want * 100
            out.append((key, want, got, dev, abs(dev) <= tol))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("candidates")
    ap.add_argument("--screen", type=float, default=1.0, help="default screen: %% of the wall a lever must cut")
    ap.add_argument("--json", help="write the results here")
    ap.add_argument("--device", help="device file for the cells' limits (common/devices/<gfx>.json)")
    ap.add_argument("--profile", help="the device file's profile (default: its default_profile)")
    try:
        a = ap.parse_args()
    except SystemExit:
        print("amdahl: INVALID")
        raise SystemExit(2)
    try:
        doc = json.load(open(a.candidates))
        phases = doc.get("phases", []) if isinstance(doc, dict) and ("phases" in doc or "cells" in doc) else [doc]
        results = [(ph, screen_phase(ph, a.screen)) for ph in phases]
        cells = doc.get("cells", []) if isinstance(doc, dict) else []
        if cells and not a.device:
            raise Invalid("cells need --device (the capped limits live in the device file, not the model)")
        hw = hardware({}, a.device, a.profile) if cells else None
        heads = [(c, headroom_cell(c, hw)) for c in cells]
        stacks = [stack_cell(c, {x: m for x, m, _ in ax if m is not None}) for c, ax in heads]
    except (OSError, ValueError, KeyError, TypeError, AttributeError, Invalid) as e:
        print(f"amdahl: input error: {e}", file=sys.stderr)
        print("amdahl: INVALID")
        sys.exit(2)
    ok, dump = True, []
    for ph, r in results:
        u = r["unit"]
        print(f"## {r['phase']}: wall {r['wall']:.2f} {u}, screen {r['screen_pct']:g} % of the wall")
        print("| lever | group | target | saving | cut % of wall | speedup | verdict | source |")
        print("|---|---|---|---|---|---|---|---|")
        for x in sorted(r["rows"], key=lambda x: -x["saving"]):
            v = "ENABLER" if x["enabler"] else "PASS" if x["passes"] else "below screen"
            print(f"| {x['name']} | {x['group']} | {x['target']:.2f} | {x['saving']:.2f} | {x['cut_pct']:.1f} | "
                  f"+{x['speedup_pct']:.1f} % | {v} | {x['src']} |")
        print(f"combined (largest saving per overlap group, levers that pass): {r['wall']:.2f} -> {r['after']:.2f} {u} "
              f"(-{r['combined_saving'] / r['wall'] * 100:.1f} %, x{r['wall'] / r['after']:.3f})")
        if r["combined_saving_all"] != r["combined_saving"]:
            print(f"combined, every lever: {r['wall']:.2f} -> {r['after_all']:.2f} {u}")
        if not r["passing"]:
            ok = False
            print(f"no lever clears the {r['screen_pct']:g} % screen: a served A/B could not tell any of them from noise")
        for key, want, got, dev, good in check_expect(ph, r):
            ok &= good
            print(f"expect {key}: {want:g} {u}, input gives {got:.2f} ({dev:+.1f} %)"
                  + ("" if good else "  DISCREPANCY: the source's figures do not add up to its stated total"))
        print()
        dump.append(r)
    for (c, ax), st in zip(heads, stacks):
        known = [m for _, m, _ in ax if m is not None]
        print(f"## headroom: {c['cell']} ({c['kind']}) at {hw['name']}: {hw['bw_practical_gbs']} GB/s, {hw['wmma_f16_tflops']} TFLOP/s")
        print("| axis | multiplier ceiling | why |\n|---|---|---|")
        for x, m, why in ax:
            print(f"| {x} | " + (f"x{m:.3f}" if m is not None else "unknown") + f" | {why} |")
        print(f"product of the known axes: x{math.prod(known):.3f}" + ("" if len(known) == len(ax) else " (some axes unknown)"))
        if st:
            print("stack (best candidate per axis, axes multiply):")
            for x, it, over, skipped in st:
                ok &= not over
                print(f"  {x:18s} x{it['multiplier']:.3f}  {it.get('name', '')}" + (f"  ({skipped} on the same axis not counted)" if skipped else "")
                      + ("  EXCEEDS the axis ceiling" if over else ""))
            print(f"  stacked: x{math.prod(it['multiplier'] for _, it, _, _ in st):.3f}")
        print()
        dump.append({"cell": c["cell"], "axes": [{"axis": x, "ceiling": m, "why": w} for x, m, w in ax]})
    if a.json:
        json.dump(dump, open(a.json, "w"), indent=1)
    print("amdahl: " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
