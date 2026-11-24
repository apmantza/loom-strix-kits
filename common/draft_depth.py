#!/usr/bin/env python3
"""Draft depth: expected tokens per verify step, step time and tokens/s per depth, and the best depth. Model-agnostic: the
drafter's kind and cost, the verify cost by width and the acceptance all come from the input (a kit's drafter file).

A verify step at depth d checks d drafted tokens plus the target's own (d + 1 rows) and commits the accepted prefix plus
one: E[tokens/step] = 1 + sum_{k=1..d} prod_{j<=k} a_j, where a_j is the conditional acceptance of position j (accepted
given 1..j-1 accepted). Step time = verify(d + 1 rows) + drafter(d) + fixed.

Input (JSON):
 "acceptance": one of
   {"position_probs": [a_1, a_2, ...]}                   conditional acceptance per position (measured)
   {"accept_hist": {"0": n0, "1": n1, ...}}              accepted drafts per verify step, counted at the served depth
   {"served": [{"depth": 7, "tokens_per_step": 3.25, "src": "..."}, ...]}
       aggregate counters only (loom-serve /metrics draft_proposed / draft_accepted, the agent bench's per-turn rows).
       One depth fits a constant a (a_j = a); two or more fit a_j = a x r^(j - 1) by least squares and print each
       depth's residual. Both are MODELS: a per-position measurement (accept_hist at the served depth) replaces them.
   --replay FILE... --replay-depth D adds a served point from agent-bench rows (out, draft_acc, draft_proposed; or a
       llama.cpp engine's draft_n / draft_n_accepted):
       tokens/step = out / (out - accepted), the formula agent_ab.sh prints.
 "drafter": {"kind": "block", "ms": 10.5, "max_depth": 7, "per_depth_ms": 0}
       one pass proposes the whole block (DFlash-style): cost flat in depth up to max_depth (per_depth_ms optional)
   or {"kind": "autoregressive", "step_ms": 1.2, "fixed_ms": 0.3, "max_depth": 8}
       one drafter step per drafted token (an MTP head, a small draft model): fixed_ms + d x step_ms
   "warm_ms": per-step cost that exists only when drafting (a drafter's context update); 0 when absent.
 "verify": {"ms_by_width": {"1": 85.0, "8": 115.0}}     measured (interpolated linearly between widths, never extrapolated)
   or {"lanes": [{"name": "matvecs", "ms": 79.4, "width": 8, "scaling": "tile", "tile": 8}, ...]}  a model per lane:
       flat      the same at every width (weights read once per step)
       tile      flat up to `tile` rows, then ms x ceil(w / tile) (a width-specialised kernel padded to its tile)
       per_row   ms x w / width (attention, recurrences: work per verified row)
       moe_union dense part flat; the expert share grows with the expected union of the rows' experts:
                 ms x ((1 - expert_share) + expert_share x U(w) / U(width)), U(w) = 1 - (1 - top_k / experts)^w
 "fixed_ms": per-step cost outside verify and drafter (sampling, accept, host); "expect": {"depth": 7, "tokens_per_step": 3.25,
 "tolerance_pct": 1} checks the acceptance model against a served figure.

usage: draft_depth.py DRAFTER.json [--replay FILE ... --replay-depth D] [--max-depth N] [--json OUT]
Last line `draft_depth: BEST d=N` with exit 0, `draft_depth: FAIL` (an expect missed) exit 1, `draft_depth: INVALID` exit 2.
"""
import argparse
import json
import math
import sys


class Invalid(Exception):
    pass


def f(v, what, lo=None, hi=None):
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
        raise Invalid(f"{what} must be a finite number, got {v!r}")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise Invalid(f"{what} = {v} outside [{lo}, {hi}]")
    return float(v)


def expected_tokens(probs, d):
    e, s = 1.0, 1.0
    for a in probs[:d]:
        s *= a
        e += s
    return e


def probs_from_hist(h):
    n = {int(k): f(v, f"accept_hist[{k}]", 0) for k, v in h.items()}
    if not n or sum(n.values()) <= 0:
        raise Invalid("accept_hist is empty")
    top, out = max(n), []
    for k in range(1, top + 1):
        reach = sum(v for a, v in n.items() if a >= k - 1)
        out.append(sum(v for a, v in n.items() if a >= k) / reach if reach else 0.0)
    return out


def solve_const(depth, tokens):   # sum_{k=1..depth} a^k = tokens - 1, monotone in a: bisection
    want = tokens - 1
    if not 0 <= want <= depth:
        raise Invalid(f"tokens_per_step {tokens} impossible at depth {depth}")
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sum(mid ** k for k in range(1, depth + 1)) < want else (lo, mid)
    return lo


def decay(a, r, n):
    return [min(1.0, a * r ** j) for j in range(n)]


def fit(served):
    """(position probs over max served depth, description). One point: constant a; more: a x r^(j-1) least squares."""
    deep = max(p["depth"] for p in served)
    depths = {p["depth"] for p in served}
    if len(depths) == 1:   # one depth (several measurements of it: their mean) fixes one parameter only
        t = sum(p["tokens_per_step"] for p in served) / len(served)
        a = solve_const(deep, t)
        return [a] * deep, f"constant a = {a:.4f} fitted to one served depth (a model: per-position acceptance unmeasured)"
    best = None
    for i in range(1, 501):
        a = i / 500
        for j in range(250, 751):
            r = j / 500
            ps = decay(a, r, deep)
            err = sum((expected_tokens(ps, p["depth"]) / p["tokens_per_step"] - 1) ** 2 for p in served)
            if best is None or err < best[0]:
                best = (err, a, r)
    _, a, r = best
    return decay(a, r, deep), f"a_j = {a:.3f} x {r:.3f}^(j-1) least-squares over {len(depths)} served depths (a model)"


def replay_point(files, depth):
    out = acc = 0.0
    for fn in files:
        for line in open(fn):
            d = json.loads(line)
            for x in d.get("rows", []):
                out += x.get("out") or 0
                if "draft_n" in x:   # a llama.cpp engine: per-request draft_n / draft_n_accepted (server timings)
                    acc += x.get("draft_n_accepted") or 0
                else:
                    acc += (x.get("draft_acc") or 0) * (x.get("draft_proposed") or 0)   # draft_acc is a rate (replay.py)
    if out <= acc or out <= 0:
        raise Invalid("replay rows give no verify steps")
    return {"depth": depth, "tokens_per_step": out / (out - acc), "src": f"{len(files)} replay file(s), {int(out)} tokens"}


def verify_ms(v, w):
    if "ms_by_width" in v:
        pts = sorted((int(k), f(x, f"ms_by_width[{k}]", 0)) for k, x in v["ms_by_width"].items())
        for (w0, t0), (w1, t1) in zip(pts, pts[1:]):
            if w0 <= w <= w1:
                return t0 + (t1 - t0) * (w - w0) / (w1 - w0) if w1 > w0 else t0
        if len(pts) == 1 and pts[0][0] == w:
            return pts[0][1]
        return None   # outside the measured widths: unknown, not extrapolated
    tot = 0.0
    for ln in v.get("lanes", []):
        ms, ref, sc = f(ln["ms"], f"lane {ln.get('name')} ms", 0), int(ln.get("width", 1)), ln.get("scaling")
        if sc == "flat":
            tot += ms
        elif sc == "tile":
            t = int(ln["tile"])
            tot += ms * math.ceil(w / t) / math.ceil(ref / t)
        elif sc == "per_row":
            tot += ms * w / ref
        elif sc == "moe_union":
            share, k, n = f(ln["expert_share"], "expert_share", 0, 1), ln["top_k"], ln["experts"]
            u = lambda x: 1 - (1 - k / n) ** x
            tot += ms * ((1 - share) + share * u(w) / u(ref))
        else:
            raise Invalid(f"lane {ln.get('name')}: scaling {sc!r} not in flat / tile / per_row / moe_union")
    return tot


def drafter_ms(dr, d):
    if d == 0:
        return 0.0
    warm = f(dr.get("warm_ms", 0), "warm_ms", 0)
    if dr.get("kind") == "block":
        return f(dr["ms"], "drafter ms", 0) + f(dr.get("per_depth_ms", 0), "per_depth_ms", 0) * d + warm
    if dr.get("kind") == "autoregressive":
        return f(dr.get("fixed_ms", 0), "fixed_ms", 0) + f(dr["step_ms"], "step_ms", 0) * d + warm
    raise Invalid(f"drafter kind {dr.get('kind')!r} not in block / autoregressive")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("drafter")
    ap.add_argument("--replay", nargs="+", default=[])
    ap.add_argument("--replay-depth", type=int)
    ap.add_argument("--max-depth", type=int)
    ap.add_argument("--json")
    try:
        a = ap.parse_args()
    except SystemExit:
        print("draft_depth: INVALID")
        raise SystemExit(2)
    try:
        doc = json.load(open(a.drafter))
        acc, dr, v = doc.get("acceptance", {}), doc["drafter"], doc["verify"]
        served = [dict(p) for p in acc.get("served", [])]
        if a.replay:
            if not a.replay_depth:
                raise Invalid("--replay needs --replay-depth (the depth the replayed package drafted at)")
            served.append(replay_point(a.replay, a.replay_depth))
        for p in served:
            p["depth"] = int(f(p["depth"], "served depth", 1))
            f(p["tokens_per_step"], "served tokens_per_step", 1)
        if "position_probs" in acc:
            probs, how = [f(x, "position_probs", 0, 1) for x in acc["position_probs"]], "measured per-position acceptance"
        elif "accept_hist" in acc:
            probs, how = probs_from_hist(acc["accept_hist"]), "per-position acceptance from the accepted-length histogram"
        elif served:
            probs, how = fit(served)
        else:
            raise Invalid("acceptance needs position_probs, accept_hist or served points (or --replay)")
        cap = min(x for x in (int(dr.get("max_depth", len(probs))), len(probs), a.max_depth or 10 ** 9))
        fixed = f(doc.get("fixed_ms", 0), "fixed_ms", 0)
        rows = []
        for d in range(cap + 1):
            vm = verify_ms(v, d + 1)
            e = expected_tokens(probs, d)
            step = None if vm is None else vm + drafter_ms(dr, d) + fixed
            rows.append({"depth": d, "a_d": probs[d - 1] if d else None, "tokens_per_step": e, "verify_ms": vm,
                         "drafter_ms": drafter_ms(dr, d), "step_ms": step, "tok_s": e / step * 1e3 if step else None})
    except (OSError, ValueError, KeyError, TypeError, AttributeError, Invalid) as e:
        print(f"draft_depth: input error: {e}", file=sys.stderr)
        print("draft_depth: INVALID")
        sys.exit(2)
    print(f"acceptance: {how}")
    for p in served:
        print(f"  served depth {p['depth']}: {p['tokens_per_step']:.3f} tokens/step, model {expected_tokens(probs, p['depth']):.3f} "
              f"({(expected_tokens(probs, p['depth']) / p['tokens_per_step'] - 1) * 100:+.1f} %)  {p.get('src', '')}")
    print(f"drafter: {dr['kind']}" + (" (one pass proposes the block: cost flat in depth)" if dr["kind"] == "block"
                                      else " (one drafter step per drafted token)"))
    print("| depth | a_d | tokens/step | verify ms | drafter ms | step ms | tok/s |\n|---|---|---|---|---|---|---|")
    for r in rows:
        g = lambda x, p=2: "unknown" if x is None else f"{x:.{p}f}"
        print(f"| {r['depth']} | {g(r['a_d'], 3)} | {r['tokens_per_step']:.3f} | {g(r['verify_ms'])} | {r['drafter_ms']:.2f} | "
              f"{g(r['step_ms'])} | {g(r['tok_s'])} |")
    ok = True
    exp = doc.get("expect")
    if exp:
        got = expected_tokens(probs, int(exp["depth"]))
        dev = (got / exp["tokens_per_step"] - 1) * 100
        good = abs(dev) <= float(exp.get("tolerance_pct", 1))
        ok &= good
        print(f"expect depth {exp['depth']}: {exp['tokens_per_step']} tokens/step, model {got:.3f} ({dev:+.1f} %)" + ("" if good else "  MISS"))
    known = [r for r in rows if r["tok_s"] is not None]
    if a.json:
        json.dump({"acceptance": how, "rows": rows}, open(a.json, "w"), indent=1)
    if not ok:
        print("draft_depth: FAIL")
        sys.exit(1)
    if not known:
        print("draft_depth: unknown (no depth has a verify cost: measure ms_by_width at the widths you serve)")
        print("draft_depth: INVALID")
        sys.exit(2)
    best = max(known, key=lambda r: r["tok_s"])
    print(f"best depth {best['depth']}: {best['tok_s']:.2f} tok/s device ({best['tokens_per_step']:.3f} tokens/step, "
          f"{best['step_ms']:.2f} ms/step)")
    print(f"draft_depth: BEST d={best['depth']}")


if __name__ == "__main__":
    main()
