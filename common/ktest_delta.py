#!/usr/bin/env python3
"""Judge a candidate's kernel tests by its delta against the base package (PRINCIPLES: a gate judges the actor's delta).
A case that fails on the candidate and also fails on the base package (built from the same kit tree, run with the same generator)
is INHERITED: printed with the case, not a stop. A case that fails only on the candidate, a case the base did not run, a run-level
failure (FAIL compile / select: no cases ran) and a known-defect tag that now passes are NEW: the step stops. Why: hand-tagged
KNOWN-FAIL lists (run_kernel_tests.sh) go stale both ways; E445: q166 stopped on E444's stale-KV defect, which live had too.
usage: ktest_delta.py CAND.run.log BASE.run.log      exit 0 no new failure, 1 a new failure, 2 a log without results"""
import re
import sys

RUN_LEVEL = {"compile", "select", "KNOWN-DEFECT"}   # 'FAIL KNOWN-DEFECT NOW PASSES case': the tag outlived the defect


def results(path):
    """(failed cases, all cases seen, run-level failures); a case is the token after PASS / FAIL / KNOWN-FAIL (id)."""
    failed, seen, run = set(), set(), []
    for line in open(path, errors="replace"):
        m = re.match(r"(PASS|FAIL|KNOWN-FAIL \([^)]*\)) (\S+)", line)
        if not m:
            continue
        word, case = m.group(1), m.group(2).rstrip(":")
        if word == "FAIL" and case in RUN_LEVEL:
            run.append(line.strip()[:200])
            continue
        seen.add(case)
        if word != "PASS":
            failed.add(case)
    return failed, seen, run


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[-2], file=sys.stderr)
        return 2
    cf, cs, crun = results(argv[1])
    bf, bs, _ = results(argv[2])
    if not cs and not crun:
        print(f"ktest_delta: no results in {argv[1]}")
        return 2
    if not bs:
        print(f"ktest_delta: no results in the base log {argv[2]}: cannot judge a delta")
        return 2
    new = sorted(c for c in cf if c not in bf)
    inherited = sorted(c for c in cf if c in bf)
    for c in inherited:
        print(f"INHERITED {c} (fails on the base package too)")
    for c in new:
        print(f"NEW FAIL {c}" + ("" if c in bs else " (the base package did not run this case)"))
    for r in crun:
        print(f"NEW FAIL run-level: {r}")
    print(f"ktest_delta: {len(new) + len(crun)} new, {len(inherited)} inherited, {len(cs)} cases on the candidate, {len(bs)} on the base")
    return 1 if new or crun else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
