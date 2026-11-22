#!/usr/bin/env python3
"""Replay a frozen agent session (freeze.py) against any OpenAI-compatible engine, turn by turn, one request at a time.

Modes
  fixed  every turn sends the RECORDED conversation up to it: byte-identical prompts on every engine (fairest
         engine-to-engine comparison). Pessimistic on caching: the engine's own previous reply is not what the next
         turn carries, so engines that only continue an exact previous request re-prefill (run1 on Loom: turns of
         55-69k new tokens where the live run had a few hundred). --every N samples turns.
  own    each turn's history uses the engine's OWN previous replies (reasoning, text, tool calls); the recorded tool
         results are attached to the engine's tool-call ids in order. Caching behaves as in a live session; outputs
         differ between engines, so compare per-token rates and the cache columns, not totals alone. Runs a contiguous
         range of turns (--every is refused).

Per turn: context, new vs cached prompt tokens (engine-reported) next to the recording, time to first token
(client-side), prefill rate, output tokens, decode rate, wall time. Summary: session wall time, prefill/decode split.

usage: replay.py SESSION URL MODEL [--kind loom|llama|generic] [--mode fixed|own] [--turns A:B] [--every N]
                 [--exact-length] [--effort LEVEL] [--label X] [--log FILE]
"""
import argparse
import json
import os
import statistics
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engines  # noqa: E402
sys.path.insert(0, os.path.join(os.environ.get("LOOM_SKILLS") or sys.exit("set LOOM_SKILLS to a loom-skills checkout"), "skills/loom-benchmark/scripts"))
from bench_common import acceptance_between, server_draft_counters  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("url")
ap.add_argument("model")
ap.add_argument("--kind", default="generic", choices=["loom", "llama", "generic"])
ap.add_argument("--mode", default="fixed", choices=["fixed", "own"])
ap.add_argument("--turns", default="0:")
ap.add_argument("--every", type=int, default=1)
ap.add_argument("--exact-length", action="store_true", help="ask the engine to ignore EOS (llama.cpp) so every engine decodes the recorded length")
ap.add_argument("--effort", default=None)
ap.add_argument("--label", default="")
ap.add_argument("--log", default=None)
ap.add_argument("--sampling", default=None, help="'card' = the Qwen3.8-27B card's set (thinking: T 1.0, top_p 0.95, top_k 20; "
                "effort none: T 0.7, top_p 0.8, top_k 20; E113), or a JSON object of body fields; default greedy (temperature 0)")
ap.add_argument("--seed", type=int, default=None, help="with --sampling: per-turn seed SEED*100000+turn (same seed, same build: same text)")
ap.add_argument("--salt", default="", help="prepended to the system prompt: a second concurrent stream of the same session shares no cached prefix")
a = ap.parse_args()
if a.mode == "own" and a.every != 1:
    sys.exit("--mode own replays a contiguous range of turns (the history is the engine's own output): drop --every")

S = json.load(open(a.session))
msgs, turns, tools = S["messages"], S["turns"], S["tools"]
lo, _, hi = a.turns.partition(":")
sel = list(range(len(turns)))[int(lo or 0):(int(hi) if hi else None)][::a.every]
URL = a.url.rstrip("/")


def history(k, own):
    """Messages for turn k. In own mode the assistant messages before k are replaced by the engine's replies."""
    h = [dict(m) for m in msgs[:turns[k]["upto"]]]
    if a.salt:
        h[0] = dict(h[0], content=a.salt + "\n" + h[0]["content"])
    if not own:
        return h
    out, remap = [], {}
    for i, m in enumerate(h):
        if m["role"] == "assistant" and i in own:
            mine = dict(own[i])
            rec_ids = [t["id"] for t in m.get("tool_calls") or []]
            new_ids = [t["id"] for t in mine.get("tool_calls") or []]
            if rec_ids and not new_ids:          # the engine produced no (parseable) call: keep the recorded calls
                mine["tool_calls"] = m["tool_calls"]
                new_ids = rec_ids
            for j, rid in enumerate(rec_ids):
                remap[rid] = new_ids[min(j, len(new_ids) - 1)] if new_ids else rid
            out.append(mine)
        elif m["role"] == "tool":
            out.append(dict(m, tool_call_id=remap.get(m["tool_call_id"], m["tool_call_id"])))
        else:
            out.append(m)
    return out


CARD = {"thinking": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}, "instruct": {"temperature": 0.7, "top_p": 0.8, "top_k": 20}}


def sampling_for(effort):
    """Body fields for --sampling: the card's set by reasoning state, or the given JSON object."""
    if a.sampling == "card":
        return dict(CARD["instruct" if effort == "none" else "thinking"])
    return json.loads(a.sampling)


def run(k, own):
    t = turns[k]
    body = {"model": a.model, "messages": history(k, own), "tools": tools, "stream": True, "stream_options": {"include_usage": True}}
    # own mode: headroom over the recorded length, so a turn is not cut inside a tool call (a dropped call changes the
    # history the next turn re-sends and breaks continuation); fixed mode keeps the recorded length
    cap = max(1, t["rec_out"]) if own is None else max(2048, 2 * t["rec_out"])
    body.update(engines.request_params(a.kind, a.effort or t["effort"], cap, a.exact_length and own is None))
    if a.sampling:
        body.update(sampling_for(body.get("reasoning_effort")))
        if a.seed is not None:
            body["seed"] = a.seed * 100000 + k
    dc0 = server_draft_counters(a.url) if a.kind == "loom" else {}   # per-turn draft acceptance (E99: same text, different speed)
    t0 = time.perf_counter()
    first, usage, timings = None, {}, None
    content, reasoning, calls = [], [], {}
    with requests.post(URL + "/v1/chat/completions", json=body, stream=True, timeout=7200) as r:
        if r.status_code >= 400:   # keep the server's reason (raise_for_status drops the body)
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
        for line in r.iter_lines():
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            usage = d.get("usage") or usage
            timings = d.get("timings") or timings
            for c in d.get("choices") or []:
                dl = c.get("delta") or {}
                if dl.get("content") or dl.get("reasoning_content") or dl.get("tool_calls"):
                    first = first or time.perf_counter()
                content.append(dl.get("content") or "")
                reasoning.append(dl.get("reasoning_content") or "")
                for tc in dl.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "args": ""})
                    slot["id"] = tc.get("id") or slot["id"]
                    fn = tc.get("function") or {}
                    slot["name"] += fn.get("name") or ""
                    slot["args"] += fn.get("arguments") or ""
    end = time.perf_counter()
    pt, cached = engines.cache_stats(a.kind, usage, timings)
    out = (usage or {}).get("completion_tokens") or (timings or {}).get("predicted_n") or 0
    reply = {"role": "assistant", "content": "".join(content) or None}
    if "".join(reasoning):
        reply["reasoning_content"] = "".join(reasoning)
    if calls:   # arguments re-serialized like the chat template's tojson (json.dumps defaults), as pi sends them:
        # history reuse needs the re-sent turn to render byte-identically to what the engine retained (loom-kb cfg-review)
        def canon(args):
            try:
                return json.dumps(json.loads(args))
            except ValueError:
                return args
        reply["tool_calls"] = [{"id": c["id"] or f"call_r{k}_{i}", "type": "function", "function": {"name": c["name"], "arguments": canon(c["args"])}}
                               for i, c in sorted(calls.items())]
    ttft, gen = (first or end) - t0, end - (first or end)
    row = {"turn": k, "prompt": pt, "cached": cached, "new": pt - cached, "rec_new": t["rec_new"], "rec_cached": t["rec_cached"],
           "ttft": round(ttft, 2), "prefill_tps": round((pt - cached) / ttft, 1) if ttft > 0 else None, "out": out,
           "rec_out": t["rec_out"], "gen_s": round(gen, 2), "decode_tps": round((out - 1) / gen, 1) if gen > 0 and out > 1 else None,
           "wall": round(end - t0, 2), "rec_wall": t["rec_wall_s"], "cap": cap}
    if dc0:
        dc1 = server_draft_counters(a.url)
        row["draft_acc"] = acceptance_between(dc0, dc1)
        row["draft_proposed"] = dc1.get("draft_proposed_tokens", 0) - dc0.get("draft_proposed_tokens", 0)
    return row, reply


own = {} if a.mode == "own" else None
res, t_all = [], time.perf_counter()
prev = None
for k in sel:
    try:
        row, reply = run(k, own)
    except Exception as e:   # a failing turn is a row, not the end of the run (own mode then keeps the recorded reply)
        print(f"turn {k:3d} ERROR {type(e).__name__}: {str(e)[:300]}", flush=True)
        res.append({"turn": k, "error": f"{type(e).__name__}: {str(e)[:300]}", "prompt": 0, "cached": 0, "new": 0, "ttft": 0.0,
                    "out": 0, "gen_s": 0.0, "lost": None})
        prev = None
        continue
    # continuation check: the previous request's prompt + its output is what the engine retained; a live pi session
    # reuses all of it but one token (run1: 49/49 turns). lost > 64 means the history did not extend it exactly.
    row["lost"] = (prev["prompt"] + prev["out"] - row["cached"]) if prev and sel[sel.index(k) - 1] == k - 1 else None
    prev = row
    if own is not None:
        own[turns[k]["upto"]] = reply      # the assistant message at that index becomes the engine's reply
    res.append(row)
    print(f"turn {k:3d} ctx {row['prompt']:6d} new {row['new']:6d} (rec {row['rec_new']:6d}) ttft {row['ttft']:6.1f}s "
          f"{row['prefill_tps'] or 0:6.0f} tok/s | out {row['out']:5d}/{row['rec_out']:5d} {row['decode_tps'] or 0:5.1f} tok/s"
          f" | wall {row['wall']:6.1f}s (rec {row['rec_wall']:.1f})"
          + (f" | lost {row['lost']}" if row["lost"] is not None else "")
          + (f" | acc {row['draft_acc']}" if row.get("draft_acc") is not None else ""), flush=True)
errors = [r for r in res if r.get("error")]
res_ok = [r for r in res if not r.get("error")]
if not res_ok:
    print(f"SUMMARY {{\"errors\": {len(errors)}}}"); sys.exit(1 if errors else 0)
wall = time.perf_counter() - t_all
pre, gen = sum(r["ttft"] for r in res_ok), sum(r["gen_s"] for r in res_ok)
newt, outt = sum(r["new"] for r in res_ok), sum(r["out"] for r in res_ok)
summary = {"label": a.label, "model": a.model, "kind": a.kind, "mode": a.mode, "sampling": a.sampling, "seed": a.seed, "session": S["name"], "turns": len(res_ok), "errors": len(errors),
           "session_wall_s": round(wall, 1), "prefill_s": round(pre, 1), "decode_s": round(gen, 1), "new_tokens": newt,
           "out_tokens": outt, "prefill_tps": round(newt / pre, 1) if pre else None, "decode_tps": round(outt / gen, 1) if gen else None,
           "cache_hit": round(sum(r["cached"] for r in res_ok) / max(1, sum(r["prompt"] for r in res_ok)), 3),
           "max_ctx": max(r["prompt"] for r in res_ok),
           "continuation_misses": sum(1 for r in res_ok if r.get("lost") is not None and r["lost"] > 64), "median_ttft_s": round(statistics.median(r["ttft"] for r in res_ok), 2)}
print("SUMMARY " + json.dumps(summary), flush=True)
if a.log:
    with open(a.log, "a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "summary": summary, "rows": res}) + "\n")
