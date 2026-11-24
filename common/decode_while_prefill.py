#!/usr/bin/env python3
"""How does the server schedule one stream's decode while another stream prefills? (per-token: the E59 / L19 method)

Stream B decodes a long reply (thinking off) and records the arrival time of every streamed chunk. After B has decoded for
--lead seconds, stream A sends a long prompt (--prompt tokens of real text) with a two-token reply and records its first token
(its prefill window). Prints B's inter-chunk gaps before, during and after A's prefill (concurrency_audit.gap_stats: one
phase classifier for this probe and the replay audit), the largest gaps during it, and B's tokens/s in each phase. Bursty
stalls (gaps of seconds) mean B's steps wait for whole prefill chunks; evenly slower tokens mean B rides along in mixed epochs.
Model-agnostic: --model is the served id (default $KIT_MODEL_ID); the server must already be up (common/serve_as.sh).

usage: decode_while_prefill.py [--url http://127.0.0.1:8160] [--model ID] [--prompt 40000] [--lead 15] [--b-tokens 3000] [--out FILE]
       exit 0 measured, 1 a stream failed, 2 usage
"""
import argparse
import json
import os
import sys
import threading
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrency_audit import gap_stats  # noqa: E402
sys.path.insert(0, os.path.join(os.environ.get("LOOM_SKILLS") or sys.exit("set LOOM_SKILLS to a loom-skills checkout"), "skills/loom-benchmark/scripts"))
from bench_common import doc, thinking_extra  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8160")
ap.add_argument("--model", default=os.environ.get("KIT_MODEL_ID"))
ap.add_argument("--prompt", type=int, default=40000)
ap.add_argument("--lead", type=float, default=15.0)
ap.add_argument("--b-tokens", type=int, default=3000)
ap.add_argument("--chars-per-token", type=float, default=3.6, help="prompt text length per target token (tokenizer-dependent)")
ap.add_argument("--out", default=None)
a = ap.parse_args()
if not a.model:
    print("decode_while_prefill: pass --model (or set KIT_MODEL_ID)")
    sys.exit(2)
URL = a.url.rstrip("/") + "/v1/chat/completions"
TH = thinking_extra("off")
nonce = f"{time.time()}"


def stream(messages, max_tokens, stamps, label):
    body = {"model": a.model, "messages": messages, "max_tokens": max_tokens, "stream": True, "temperature": 0}
    body.update(TH)
    t0 = time.perf_counter()
    with requests.post(URL, json=body, stream=True, timeout=3600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            for c in d.get("choices") or []:
                dl = c.get("delta") or {}
                if dl.get("content") or dl.get("reasoning_content"):
                    stamps.append(time.perf_counter())
    stamps.append(("end", time.perf_counter() - t0, label))


b_st, a_st = [], []
B = threading.Thread(target=stream, args=([{"role": "user", "content": f"[{nonce}] Write a very long, detailed story about a lighthouse keeper, at least 3000 words."}],
                                          a.b_tokens, b_st, "B"))
B.start()
deadline = time.perf_counter() + 600   # B's prefill is short; a refused or dead stream must not hang the run
while len([x for x in b_st if not isinstance(x, tuple)]) < 5:
    if not B.is_alive() or time.perf_counter() > deadline:
        print("decode_while_prefill: stream B produced no tokens (server down or request failed)")
        sys.exit(1)
    time.sleep(0.05)
time.sleep(a.lead)
text, i, big = "", 0, doc(32000)
while len(text) < a.prompt * a.chars_per_token:
    text += f"\n\n[part {nonce}-{i}]\n\n" + big
    i += 1
text = text[:int(a.prompt * a.chars_per_token)]
a_start = time.perf_counter()
A = threading.Thread(target=stream, args=([{"role": "user", "content": f"[{nonce}] Summarize this in one line.\n\n" + text}], 2, a_st, "A"))
A.start()
A.join()
a_first = next((x for x in a_st if not isinstance(x, tuple)), None)
B.join()
if a_first is None:
    print("decode_while_prefill: stream A produced no token")
    sys.exit(1)
bt = [x for x in b_st if not isinstance(x, tuple)]
seg = [(float("-inf"), a_start, "before"), (a_start, a_first, "during"), (a_first, float("inf"), "after")]
ph = gap_stats(bt, seg)
gaps = [(bt[k] - bt[k - 1], bt[k]) for k in range(1, len(bt))]
res = {"prompt_target": a.prompt, "a_prefill_s": round(a_first - a_start, 2),
       "before": ph.get("before"), "during": ph.get("during"), "after": ph.get("after"),
       "largest_gaps_during": sorted([(round(1000 * d, 1), round(t - a_start, 2)) for d, t in gaps if a_start <= t < a_first], reverse=True)[:12]}
print(json.dumps(res, indent=1))
if a.out:
    json.dump({"result": res, "b_stamps": [t - a_start for t in bt], "a_first": a_first - a_start}, open(a.out, "w"))
