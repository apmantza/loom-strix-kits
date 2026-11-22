#!/usr/bin/env python3
"""Fail when a long-context profile's label disagrees with the context it actually served (E128/E133: the "dec100k"
profiles served 81,432 prompt tokens, profile_longctx_req.py sizes prompts at 3.6 chars/token and this text tokenizes
denser; E128 then compared serving at "100k" with a bench at 81k and re-framed the gap wrongly).
usage: context_label_check.py REQ.json REQUESTED_TOKENS [--tolerance 0.05]   exit 1 on a mismatch or an unreadable record"""
import argparse
import json
import sys

ap = argparse.ArgumentParser()
ap.add_argument("req")
ap.add_argument("requested", type=int)
ap.add_argument("--tolerance", type=float, default=0.05)
a = ap.parse_args()
try:
    served = int(json.loads(open(a.req).read().splitlines()[-1])["requests"][-1]["prompt_tokens"])
except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
    print(f"CONTEXT LABEL UNKNOWN: {a.req}: {e}")
    sys.exit(1)
off = served / a.requested - 1
line = f"served context {served} prompt tokens (requested {a.requested}, {off:+.1%})"
if abs(off) > a.tolerance:
    print(f"CONTEXT LABEL MISMATCH: {line}: cite the served count, not the label")
    sys.exit(1)
print(line)
