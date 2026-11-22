#!/usr/bin/env python3
"""Server-side numbers of an agent run from /metrics snapshots (the first runs' figures were computed by hand).
RUN_DIR holds metrics-before.txt, metrics/<epoch>.txt (one per minute, agent-runs/*/README.txt) and optionally metrics-after.txt.
A session's own context is NOT in /metrics: pool residency counts every row and retained session (read the transcript's
per-turn usage for the context). Prints totals over the run (requests, failed / rejected / cancelled, prefill and decode tokens, model busy time, draft acceptance,
peak KV residency) and one row per interval with tokens and busy seconds, and writes RUN_DIR/summary.json.
Rates are per model busy second (the server's model_duration), which mixes prefill and decode inside an interval; an interval
with no decode tokens is a pure prefill rate. usage: agent_run_summary.py RUN_DIR"""
import json
import os
import re
import sys

LINE = re.compile(r'^(loom_serve_\w+)(\{[^}]*\})? ([0-9.eE+-]+)$')


def parse(path):
    """{name or name:field -> value} from a Prometheus text snapshot (observation metrics keyed by their field label)"""
    out = {}
    for l in open(path):
        m = LINE.match(l.strip())
        if not m:
            continue
        name, labels, v = m.group(1), m.group(2) or "", float(m.group(3))
        f = re.search(r'field="(\w+)"', labels)
        if "bucket" in name:
            continue
        out[f"{name}:{f.group(1)}" if f else name] = out.get(f"{name}:{f.group(1)}" if f else name, 0) + v
    return out


OBS, MOD = "loom_serve_model_observation_total", "loom_serve_model"
KEYS = {"requests": f"{MOD}_requests_total", "completed": f"{MOD}_completed_total", "failed": f"{MOD}_failed_total",
        "rejected": f"{MOD}_rejected_total", "cancelled": f"{MOD}_cancelled_total", "input_tokens": f"{MOD}_input_tokens_total",
        "output_tokens": f"{MOD}_output_tokens_total", "prefill_tokens": f"{OBS}:prefill_tokens", "decode_tokens": f"{OBS}:decode_tokens",
        "busy_ns": f"{OBS}:model_duration", "draft_proposed": f"{OBS}:draft_proposed_tokens", "draft_accepted": f"{OBS}:draft_accepted_inputs"}


def delta(a, b):
    return {k: b.get(v, 0) - a.get(v, 0) for k, v in KEYS.items()}


def main(d):
    snaps = [(os.path.getmtime(os.path.join(d, "metrics-before.txt")), parse(os.path.join(d, "metrics-before.txt")))]
    md = os.path.join(d, "metrics")
    for f in sorted(os.listdir(md), key=lambda x: int(x.split(".")[0])) if os.path.isdir(md) else []:
        s = parse(os.path.join(md, f))
        if s:   # a failed curl leaves an empty file
            snaps.append((int(f.split(".")[0]), s))
    if os.path.exists(os.path.join(d, "metrics-after.txt")):
        snaps.append((os.path.getmtime(os.path.join(d, "metrics-after.txt")), parse(os.path.join(d, "metrics-after.txt"))))
    assert len(snaps) >= 2, "agent_run_summary: need metrics-before.txt and at least one later snapshot"
    tot = delta(snaps[0][1], snaps[-1][1])
    resident = [s.get("loom_serve_model_observation:pool_resident", 0) for _, s in snaps]
    busy = tot["busy_ns"] / 1e9
    wall = snaps[-1][0] - snaps[0][0]
    acc = tot["draft_accepted"] / tot["draft_proposed"] if tot["draft_proposed"] else None
    rows = []
    for (t0, a), (t1, b) in zip(snaps, snaps[1:]):
        x = delta(a, b)
        if x["busy_ns"] > 0:
            rows.append({"t": round(t1 - snaps[0][0]), "prefill": int(x["prefill_tokens"]), "decode": int(x["decode_tokens"]),
                         "busy_s": round(x["busy_ns"] / 1e9, 1), "resident": int(b.get("loom_serve_model_observation:pool_resident", 0))})
    pre_only = [r for r in rows if r["decode"] == 0 and r["prefill"] and r["busy_s"]]
    summary = {**{k: int(v) for k, v in tot.items() if k != "busy_ns"}, "busy_s": round(busy, 1), "wall_s": round(wall),
               "busy_share": round(busy / wall, 3) if wall else None, "acceptance": round(acc, 3) if acc is not None else None,
               "peak_pool_resident_all_rows": int(max(resident)),   # NOT this session's context: every row + retained sessions (once misread as a 240k context; the task peaked at 136k)
               "pure_prefill_tok_s": round(sum(r["prefill"] for r in pre_only) / sum(r["busy_s"] for r in pre_only), 1) if pre_only else None,
               "intervals": rows}
    json.dump(summary, open(os.path.join(d, "summary.json"), "w"), indent=1)
    for k, v in summary.items():
        if k != "intervals":
            print(f"{k:22s} {v}")
    print(f"{'t':>6s} {'prefill':>8s} {'decode':>7s} {'busy_s':>7s} {'resident':>9s}")
    for r in rows:
        print(f"{r['t']:6d} {r['prefill']:8d} {r['decode']:7d} {r['busy_s']:7.1f} {r['resident']:9d}")


if __name__ == "__main__":
    main(sys.argv[1])
