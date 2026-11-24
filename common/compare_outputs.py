#!/usr/bin/env python3
"""Greedy outputs of two models on one server must be identical when a change is meant to be bit-exact (L4b: prefill GEMMs
over an f16 copy produced by the same dequant). Prompts long enough (>= 64 tokens) to take the prefill GEMM path.

usage: compare_outputs.py URL MODEL_A MODEL_B          (two models on one server)
       compare_outputs.py --save OUT.json URL MODEL      (one server at a time: E170, two 27B packages with fixed backing do not fit)
       compare_outputs.py --diff A.json B.json           (exit 1 unless every prompt's output is identical; exit 3 CONTROL
                                                          FAILED when a planted one-character / one-counter change in a copy of A
                                                          is not reported: the comparison is checked before its pass is trusted)
--long adds a ~64k-token prompt (the 32k document twice): long-context prefill GEMMs and attention.
"""
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checks  # noqa: E402  loop flag


def diff(A, B, text_only=False, out=print):
    """True when every prompt's text AND draft counters match (text_only: text alone). Recurrence (retro 11-22): --diff exited 0
    with different draft counters; q81 and q87 checked drafts by hand, and E170's drafter break showed only in drafts."""
    same = dsame = 0
    for name in A:
        x, y = A[name]["text"], B.get(name, {}).get("text")
        eq = x == y
        same += eq
        pfx = next((i for i, (p, q) in enumerate(zip(x, y or "")) if p != q), min(len(x), len(y or ""))) if not eq else "-"
        da, db = A[name].get("draft"), B.get(name, {}).get("draft")
        dsame += da == db
        out(f"{name:6s} prompt {A[name]['prompt_tokens']} tok  identical={eq}  first diff at char {pfx}  draft A {da} B {db}{'' if da == db else '  DRAFTS DIFFER'}{'  LOOP(A)' if checks.is_loop(x) else ''}{'  LOOP(B)' if checks.is_loop(y or '') else ''}")
    out(f"IDENTICAL {same}/{len(A)}  DRAFTS EQUAL {dsame}/{len(A)}")
    return same == len(A) and (text_only or dsame == len(A))


def planted(ref, text_only=False):
    """copies of REF that an exact comparison must call different: per prompt, one character of its text changed (a character
    appended to an empty text), and unless text_only one draft counter changed (a counter added to empty drafts)"""
    for name in ref:
        c = copy.deepcopy(ref)
        t = c[name]["text"]
        i = len(t) // 2
        c[name]["text"] = t[:i] + ("#" if t[i:i + 1] != "#" else "%") + t[i + 1:] if t else "#"
        yield f"{name} text char {i}", c
        if not text_only:
            c = copy.deepcopy(ref)
            d = c[name].get("draft") or {}
            k = next(iter(d), "planted")
            c[name]["draft"] = dict(d, **{k: (d.get(k) or 0) + 1})
            yield f"{name} draft {k}", c


def control(ref, text_only=False):
    """the planted-fault control run before a pass is trusted: the names of plants the comparison did NOT report as different
    (empty = the comparison sees a one-character / one-counter change in every prompt). Recurrence: an EXACT verdict from a
    comparison that cannot fail (a key never compared, a prompt skipped, an always-equal edit) reads as proof of exactness."""
    if not ref:
        return ["no prompts in the reference"]
    return [what for what, c in planted(ref, text_only) if diff(ref, c, text_only, out=lambda *_: None)]


argv = [x for x in sys.argv[1:] if x not in ("--long", "--text-only")]
LONG = "--long" in sys.argv
if argv and argv[0] == "--diff":
    # exit 1 unless every prompt matches; exit 3 (CONTROL FAILED) when the comparison misses a planted difference: then no verdict
    # of it is trusted, a pass least of all. Offline on the saved outputs: no server.
    A, B = (json.load(open(f)) for f in argv[1:3])
    to = "--text-only" in sys.argv
    missed = control(A, to)
    if missed:
        print(f"CONTROL FAILED: the comparison called {len(missed)} planted differences identical ({', '.join(missed[:4])}): no verdict")
        sys.exit(3)
    print(f"control ok: {sum(1 for _ in planted(A, to))} planted differences all reported")
    sys.exit(0 if diff(A, B, to) else 1)

SKILLS = os.path.join(os.environ.get("LOOM_SKILLS") or sys.exit("set LOOM_SKILLS to a loom-skills checkout"), "skills")
sys.path.insert(0, os.path.join(SKILLS, "loom-benchmark", "scripts"))
from bench_common import doc, server_draft_counters, stream, thinking_extra  # noqa: E402
SAVE = None
if argv and argv[0] == "--save":
    SAVE, argv = argv[1], argv[2:]
url, a = argv[0], argv[1]
b = argv[2] if len(argv) > 2 else None
OFF = thinking_extra("off")
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "edit_source.py")).read()
prompts = [
    ("doc1k", "Summarize this documentation in five bullet points.\n\n" + doc(1000), 120),
    ("doc2k", "What are the three most important ideas in this text? Explain each briefly.\n\n" + doc(2000), 160),
    ("doc8k", "List the section headings of this document and give one sentence for each.\n\n" + doc(8000), 160),
    ("code", "Explain what this Python file does, then suggest one improvement.\n\n```python\n" + src + "```", 160),
    ("edit", "Rename `git_commit` to `git_head_commit` everywhere and return the full file only.\n\n```python\n" + src + "```", 400),
] + ([("long64k", "Give a structured outline of these two documents, with one paragraph per major section.\n\n" + doc(32000) + "\n\n---\n\n" + doc(32000), 300)] if LONG else [])
if SAVE:
    res = {}
    for name, text, n in prompts:
        c0 = server_draft_counters(url)
        s = stream(url, a, [{"role": "user", "content": text}], n, extra=OFF)
        c1 = server_draft_counters(url)
        dr = {k: c1.get(k, 0) - c0.get(k, 0) for k in c1}   # E170: equal prompts on an exact candidate give equal drafts
        res[name] = {"text": s.get("text") or "", "prompt_tokens": s.get("prompt_tokens"), "ttft": s.get("ttft"), "draft": dr}
        print(f"{name:6s} prompt {s.get('prompt_tokens')} tok  {len(res[name]['text'])} chars  draft {dr}{'  LOOP' if checks.is_loop(res[name]['text']) else ''}", flush=True)
    json.dump(res, open(SAVE, "w"), indent=1)
    sys.exit(0)
same = 0
for name, text, n in prompts:
    outs = []
    for m in (a, b):
        s = stream(url, m, [{"role": "user", "content": text}], n, extra=OFF)
        outs.append((s.get("text") or "", s.get("prompt_tokens"), s.get("ttft")))
    eq = outs[0][0] == outs[1][0]
    same += eq
    pfx = next((i for i, (x, y) in enumerate(zip(outs[0][0], outs[1][0])) if x != y), min(len(outs[0][0]), len(outs[1][0])))
    print(f"{name:6s} prompt {outs[0][1]} tok  identical={eq}  first diff at char {pfx if not eq else '-'}  "
          f"ttft A {outs[0][2]:.2f}s B {outs[1][2]:.2f}s", flush=True)
print(f"IDENTICAL {same}/{len(prompts)}")
