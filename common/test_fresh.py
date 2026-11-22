#!/usr/bin/env python3
"""Tests of fresh.check() with a fake upstream (the network is the true boundary). Each case names the recurrence it prevents.
usage: python3 test_fresh.py (exit 1 on a failure)"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import fresh  # noqa: E402

PINS = {"hrx_loom": "0.2.0", "files": [{"repo": "u/m", "revision": "r" * 40, "path": "m.gguf", "sha256": "a" * 64, "dest": "models"}]}


def upstream(version, sha, missing=False, down=False, pypi_down=False):
    def get(url):
        if down or (pypi_down and "pypi.org" in url):
            raise OSError("network down")
        if "pypi.org" in url:
            return {"info": {"version": version}}
        return [] if missing else [{"path": "m.gguf", "size": 1, "lfs": {"oid": sha}}]
    return get


fails = []


def case(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


_, s, b = fresh.check(PINS, "0.2.0", upstream("0.2.0", "a" * 64))
case("all current -> neither stale nor broken", not s and not b)
# an unnoticed re-upload under the same name kept a server on old weights for weeks
_, s, b = fresh.check(PINS, "0.2.0", upstream("0.2.0", "b" * 64))
case("file re-uploaded upstream -> stale", any("changed upstream" in x for x in s) and not b)
# a Loom release went unnoticed for a day
_, s, b = fresh.check(PINS, "0.2.0", upstream("0.3.0", "a" * 64))
case("newer hrx-loom on PyPI -> stale (installed and kit both named)", len(s) == 2 and not b)
case("never silently green: network down -> broken", bool(fresh.check(PINS, "0.2.0", upstream("", "", down=True))[2]))
case("never silently green: PyPI unreachable alone -> broken", bool(fresh.check(PINS, "0.2.0", upstream("0.2.0", "a" * 64, pypi_down=True))[2]))
case("never silently green: pinned file gone upstream -> broken", bool(fresh.check(PINS, "0.2.0", upstream("0.2.0", "", missing=True))[2]))
case("never silently green: unknown installed version -> broken", bool(fresh.check(PINS, None, upstream("0.2.0", "a" * 64))[2]))
print(f"{7 - len(fails)}/7 passed")
sys.exit(1 if fails else 0)
