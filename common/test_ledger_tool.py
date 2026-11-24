#!/usr/bin/env python3
"""Tests of ledger_tool.py on a small log written here (synthetic rows shaped like a real experiment log's). Each case names the
recurrence it prevents.
usage: python3 test_ledger_tool.py (exit 1 on a failure)"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import ledger_tool as lt  # noqa: E402

LOG = """# log

| # | Date | Seam | Change (switch) | How measured | Result | Verdict |
|---|---|---|---|---|---|---|
| E10 | 11-21 | gemm | workgroup tile order swapped (TILE_ORDER=n) | counters | no gain; DRAM bytes equal | refuted |
| E11 | 11-21 | gemm | SW_A=1 on the candidate | bench | exact; not slower (paired 0.99) | kept |
| E12 | 11-22 | gemm | SW_B=1 on the candidate | bench | 3 % slower served | removed |
| E13 | 11-22 | matvec | MV_PF=2 (q5_k prefetch) | bench | 1.2 % faster decode, exact | kept; owed: iq4_xs, q6_k |
| E14 | 11-23 | matvec | the `owed: a b` syntax quoted in prose (`ported: E13 q6_k`) | - | - | note |
| E15 | 11-24 | matvec | MV_PF on iq4_xs | bench | 0.8 % faster | kept; ported: E13 iq4_xs |
| E16 | 11-24 | port | wins to the 0.3.0 SDK | bench | exact | kept; owed: sdk-0.3.0, inkling |
"""
CARD = """#!/bin/bash
# q200: MV_PF=2 on q6_k (exact by construction)
# prior: E13
# predict: decode ratio 0.97-0.99
# kill: ratio CI includes 1.0
# budget: 45 min
# result: paired decode time \\S+ \\(95 % CI over turns\\): all ([0-9.]+)
set -u
"""
fails = []


def case(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


def cli(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = lt.main(list(argv))
    return rc, out.getvalue()


def main():
    with tempfile.TemporaryDirectory() as t:
        def w(name, text):
            p = os.path.join(t, name)
            open(p, "w").write(text)
            return p
        logp = w("LOG.md", LOG)
        log = lt.Log(logp)

        # preflight. Recurrence: a step re-queued a refuted lever (tile order) as new while a row had refuted it the day before.
        reopen = w("q300.sh", "#!/bin/bash\n# prior: none (searched: tile order)\necho run\n")
        pr, _ = lt.preflight(reopen, log)
        case(f"re-opening a refuted lever without changed: is refused naming E10 ({pr})",
             len(pr) == 1 and "E10 (no gain, refuted)" in pr[0] and "'tile order'" in pr[0])
        rc, out = cli("--ledger", logp, "preflight", reopen)
        case(f"CLI: exit 1 and 'LEDGER: PREFLIGHT_REFUSED' (rc {rc})", rc == 1 and out.strip().endswith("LEDGER: PREFLIGHT_REFUSED"))
        ans = w("q301.sh", "#!/bin/bash\n# prior: none (searched: tile order)\n# changed: E10 measured on the served shape now\necho\n")
        pr, _ = lt.preflight(ans, log)
        rc, out = cli("--ledger", logp, "preflight", ans)
        case(f"answering E10 with '# changed: E10 ...' passes, exit 0 ({pr}, rc {rc})",
             not pr and rc == 0 and "LEDGER: PREFLIGHT_OK" in out)
        cited = w("q302.sh", "#!/bin/bash\n# prior: E10 (searched: tile order)\necho\n")
        case("citing E10 in prior: passes", not lt.preflight(cited, log)[0])
        # Recurrence: 'not slower' counted as a loss would force citations of every kept tie.
        sw = w("q303.sh", '#!/bin/bash\n# prior: none (searched: zzz)\nEXTRA="SW_A=1 SW_B=1" bash build.sh\n')
        pr, _ = lt.preflight(sw, log)
        case(f"'not slower' is not a loss, '3 % slower ... removed' is ({pr})", len(pr) == 1 and "E12" in pr[0])
        pr, _ = lt.preflight(sw, log, exclude={"SW_B"})
        case("a switch the live build carries is not re-litigated", not pr)
        hid = w("q304.sh", '#!/bin/bash\n# prior: none (searched: zzz)\nEXTRA="$SW" bash build.sh\n')
        case("switches hidden in a variable are refused (fail closed)", any("not readable" in x for x in lt.preflight(hid, log)[0]))
        case("a grandfathered step passes with a note", lt.preflight(reopen, log, grandfather=300)[1][0].endswith("written before this check)"))

        # probe card. Recurrence: results believed with no prediction made before the run (a harness 2x where serving gave 3-8 %).
        good = w("q200.sh", CARD)
        rc, out = cli("probe-card", "check", good)
        case(f"a full card: PROBE: OK, exit 0 ({out.strip()[:80]})", rc == 0 and out.startswith("PROBE: OK"))
        for field in ("predict", "kill", "budget"):
            p = w(f"q20{field[0]}.sh", "".join(l + "\n" for l in CARD.split("\n") if not l.startswith(f"# {field}:")))
            rc, out = cli("probe-card", "check", p)
            case(f"no '# {field}:' line: PROBE: MISSING, exit 1", rc == 1 and "PROBE: MISSING" in out and f"'# {field}:'" in out)
        for bad, why in (("# predict: decode ratio 0.99-0.97", "range"), ("# predict: decode ratio about 0.98", "known form"),
                         ("# budget: 0", "not > 0")):
            key = bad.split(":")[0]
            p = w("q210.sh", "\n".join(bad if l.startswith(key + ":") else l for l in CARD.split("\n")))
            rc, out = cli("probe-card", "check", p)
            case(f"{bad!r}: MISSING ({why})", rc == 1 and why in out)
        p = w("q211.sh", CARD.replace("all ([0-9.]+)", "all [0-9.]+"))
        case("a result regex without a capture group: MISSING", cli("probe-card", "check", p)[0] == 1)
        case("grandfathered: PROBE: OK", cli("probe-card", "check", "--grandfather", "200", w("q199.sh", "#!/bin/bash\n"))[0] == 0)

        # verdict. Recurrence: a measured figure outside its prediction was taken as the lever's effect, not checked as the measurement.
        summ = "paired decode time pkg-b/pkg-a (95 % CI over turns): all {} [0.9, 1.0]; h1 0.95\n"
        out_f = w("sum_out.txt", summ.format("0.946"))
        rc, out = cli("verdict", good, out_f)
        case(f"a result outside the range: OUTSIDE, exit 1, debug first ({out.strip()[:90]})",
             rc == 1 and out.startswith("VERDICT: OUTSIDE") and "debug the measurement before believing it" in out)
        rc, out = cli("verdict", good, w("sum_in.txt", summ.format("0.981")))
        case(f"inside: INSIDE, exit 0 ({out.strip()[:70]})", rc == 0 and out.startswith("VERDICT: INSIDE"))
        case("range ends are inside", cli("verdict", good, "0.99")[0] == 0 and cli("verdict", good, "0.97")[0] == 0)
        rc, out = cli("verdict", good, w("sum_two.txt", summ.format("0.981") + summ.format("1.010")))
        case("the last match is the figure (a re-run appended)", rc == 1 and "1.01" in out)
        rc, out = cli("verdict", good, w("sum_none.txt", "no summary here\n"))
        case("a result regex that matches nothing: ERROR, exit 2 (fail closed)", rc == 2 and out.startswith("VERDICT: ERROR"))
        rf = w("q201.sh", CARD.replace("set -u", "# result-file: sum_in.txt\nset -u"))
        case("the header's result-file is read when no RESULT is given", cli("verdict", rf)[0] == 0)
        case("no RESULT and no result-file: exit 2", cli("verdict", good)[0] == 2)

        # owed. Recurrence: a kept win on one target (q5_k) never carried to its siblings (iq4_xs, q6_k) or to the next SDK.
        rc, out = cli("--ledger", logp, "owed")
        case(f"owed listed: E13 -> q6_k (iq4_xs ported by E15), E16 two targets, OWED: 2, exit 1 ({out.split(chr(10))[:3]})",
             rc == 1 and "E13 -> q6_k:" in out and "E16 -> sdk-0.3.0, inkling" in out and "OWED: 2" in out)
        cleared = w("LOG2.md", LOG + "| E17 | 11-25 | matvec | MV_PF on q6_k | bench | tie | kept; ported: E13 q6_k |\n"
                    "| E18 | 11-25 | port | 0.3.0 | bench | exact | ported: E16 sdk-0.3.0; dropped: E16 inkling (no kit yet) |\n")
        rc, out = cli("--ledger", cleared, "owed")
        case(f"every target ported or dropped by a later row: OWED: 0, exit 0 ({out.strip()})", rc == 0 and "OWED: 0" in out)
        early = w("LOG3.md", LOG.replace("| E10 | 11-21 | gemm | workgroup", "| E10 | 11-21 | gemm | ported: E16 inkling; workgroup"))
        rc, out = cli("--ledger", early, "owed")
        case("a ported mark above the owed row clears nothing (and is reported)", rc == 2 and "E16 -> sdk-0.3.0, inkling" in out)
        bad = w("LOG4.md", LOG + "| E19 | 11-25 | x | y | z | kept | owed: the inkling kit |\n")
        rc, out = cli("--ledger", bad, "owed")
        case("an owed list with spaces in a target: exit 2", rc == 2 and "not in a known form" in out)
    print(f"{len(fails)} failed" if fails else "OK")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
