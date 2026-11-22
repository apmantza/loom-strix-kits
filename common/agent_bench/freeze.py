#!/usr/bin/env python3
"""Freeze a recorded agent session (a JSONL transcript of {type: message, message: OpenAI chat message} rows) into an engine-neutral session file for replay.py.

The file holds the conversation ONCE (OpenAI chat messages: system, user, assistant with reasoning_content and
tool_calls, tool results), the tool definitions pi sent, and one entry per assistant turn: how many messages precede
it (the request is messages[:upto]), the reasoning level, and what the live run recorded (new / cached prompt tokens,
output tokens, wall time). Every engine then gets byte-identical requests.

Fidelity (checked on run1 with Loom's token counter): requests rebuilt this way match the recorded prompt size of every
turn to a constant offset, the preamble the worker saw on its first message (the transcript stores the task
unexpanded). --pad N prepends N filler sentences to the first user message to restore that size (run1: 63 -> 11863 vs
11860 recorded); --calibrate URL MODEL finds N with an OpenAI-compatible engine that counts input tokens
(Loom: /v1/chat/completions/input_tokens).

usage: freeze.py TRANSCRIPT OUT.session.json [--pad N | --calibrate URL MODEL] [--name NAME]
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("transcript")
ap.add_argument("out")
ap.add_argument("--pad", type=int, default=0)
ap.add_argument("--calibrate", nargs=2, metavar=("URL", "MODEL"))
ap.add_argument("--name", default=None)
a = ap.parse_args()

FILLER = "Workspace notes for this delegation: follow the brief, keep the workspace clean, report what you ran. "
rows = [json.loads(l) for l in open(a.transcript) if l.strip()]
src = [r["message"] for r in rows if r.get("type") == "message"]
level = next((r["thinkingLevel"] for r in rows if r.get("type") == "thinking_level_change"), "high")
s = src[0]["sections"]
system = "\n\n".join(s[k] for k in ("preamble", "tools", "rules", "docs", "skills", "cwd") if k in s)
listed = {l.split(":")[0].strip("- ").strip() for l in s.get("tools", "").splitlines() if l.startswith("- ")}
tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("parameters")}}
         for t in src[0].get("toolsAdded", []) if t["name"] in listed]


def text_of(content):
    return "".join(c.get("text", "") for c in content) if isinstance(content, list) else (content or "")


def build(pad):
    msgs, turns, first = [{"role": "system", "content": system}], [], True
    for m in src[1:]:
        if m["role"] == "user":
            msgs.append({"role": "user", "content": (FILLER * pad + "\n\n" if first and pad else "") + text_of(m["content"])})
            first = False
        elif m["role"] == "assistant":
            u = m.get("usage", {})
            turns.append({"upto": len(msgs), "effort": m.get("thinkingLevel", level), "rec_new": u.get("input", 0),
                          "rec_cached": u.get("cacheRead", 0), "rec_out": u.get("output", 0),
                          "rec_wall_s": round(m.get("durationMs", 0) / 1000, 2), "rec_stop": m.get("stopReason")})
            d = {"role": "assistant", "content": "".join(c.get("text", "") for c in m["content"] if c["type"] == "text") or None}
            th = "".join(c.get("thinking", "") for c in m["content"] if c["type"] == "thinking")
            if th:
                d["reasoning_content"] = th
            tcs = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                   for c in m["content"] if c["type"] == "toolCall"]
            if tcs:
                d["tool_calls"] = tcs
            msgs.append(d)
        elif m["role"] == "toolResult":
            msgs.append({"role": "tool", "tool_call_id": m["toolCallId"], "content": text_of(m["content"])})
    return msgs, turns


pad = a.pad
if a.calibrate:
    import requests
    url, model = a.calibrate
    want = None

    def count(p):
        msgs, turns = build(p)
        r = requests.post(url.rstrip("/") + "/v1/chat/completions/input_tokens",
                          json={"model": model, "messages": msgs[:turns[0]["upto"]], "tools": tools}, timeout=300)
        return r.json()["input_tokens"], turns

    c0, turns = count(0)
    want = turns[0]["rec_new"] + turns[0]["rec_cached"]
    lo, hi = 0, 4000
    while lo < hi:
        mid = (lo + hi) // 2
        if count(mid)[0] >= want:
            hi = mid
        else:
            lo = mid + 1
    pad = lo
    print(f"calibrated pad {pad}: turn 0 = {count(pad)[0]} tokens (recorded {want})")

msgs, turns = build(pad)
out = {"format": "agent-session/1", "name": a.name or a.transcript.rsplit("/", 1)[-1].split(".")[0], "source": a.transcript,
       "pad": pad, "reasoning_level": level, "tools": tools, "messages": msgs, "turns": turns}
json.dump(out, open(a.out, "w"), ensure_ascii=False)
print(f"wrote {a.out}: {len(msgs)} messages, {len(turns)} turns, {len(tools)} tools, pad {pad}")
