#!/usr/bin/env python3
"""Requests of one profiled serve session at long context: a warm-up, then one prompt of ~L tokens (real text) followed by a
drafted decode of N tokens. Prefill and decode run different kernels (flash2 / GEMMs vs split-K attention / matvecs), so the
per-kernel statistics separate the verify step at context L from the prefill. Prints one JSON line like profile_requests.py
(requests, draft counter deltas) so profile_stats.py can divide the decode kernels by the verify steps.

usage: profile_req.py URL MODEL LENGTH [--tokens 300]
"""
import argparse
import json
import sys
import time

import os
sys.path.insert(0, os.path.join(os.environ.get("LOOM_SKILLS") or sys.exit("set LOOM_SKILLS to a loom-skills checkout"), "skills/loom-benchmark/scripts"))
from bench_common import doc, server_draft_counters, stream, thinking_extra  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("model")
ap.add_argument("length", type=int)
ap.add_argument("--tokens", type=int, default=300)
a = ap.parse_args()
TH = thinking_extra("off")
nonce = f"{time.time()}"
BIG = doc(32000)
chars = int(a.length * 3.6)
text, i = "", 0
while len(text) < chars:
    text += f"\n\n[part {nonce}-{i}]\n\n" + BIG
    i += 1
text = text[:chars]


def go(content, n):
    s = stream(a.url, a.model, [{"role": "user", "content": content}], n, extra=TH)
    if s["status"] != 200:
        raise SystemExit(f"HTTP {s['status']}: {s.get('error')}")
    return {"prompt_tokens": s["prompt_tokens"], "completion_tokens": s["tokens"], "ttft_s": round(s["ttft"], 4),
            "decode_tps": s["decode_tps"] and round(s["decode_tps"], 2), "total_s": round(s["total"], 3)}


out = {"scenario": f"dec{a.length // 1000}k", "warmup": go(f"[warm-up {nonce}] Say hi.", 2)}
before = server_draft_counters(a.url)
out["requests"] = [go(f"[{nonce}] Read this documentation, then write a long, detailed essay about it, section by section.\n\n" + text,
                      a.tokens)]
after = server_draft_counters(a.url)
out["draft"] = {k: after.get(k, 0) - before.get(k, 0) for k in after}
print(json.dumps(out), flush=True)
