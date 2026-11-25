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
# bound: flash2 60.1 s (limit 29.7 s; best engine none measured), best case 1.2 % of the request (amdahl.py), resolved by served lane ABBA (floor 0.5 %)
# result: paired decode time \\S+ \\(95 % CI over turns\\): all ([0-9.]+)
set -u
"""
# kernel_lint tables (verbatim rows, HRX-like 11-25): the base (pkg-e480, live), pkg-cmb2 (E521: the three-lever flash2 candidate
# after the clash fix) and pkg-cmb (E520: the same levers before it, the V-tile prefetch lost: cover 58 -> 23 while the loop moved)
HDR = ("kernel                                      where  loop  comm   mm other/op  sel  pk  bfe  scr  mov  run vgpr w/S next    pf "
       "sites  ldsW  cov fly  bank       ms  flags\n")
LINT_BASE = HDR + """g_attention_flash2_f16o                      loop   952   952   32     28.8  113   0   16    0   90   65  240   6    -  kept    16   1.5   58   0     0        -  copies  [addr64]
g_attention_flash2_partial_f16o              loop  1003  1003   32     30.3  113   0   16    0   93   50  252   5   12  kept    16  1.62   86   3     0        -  copies  [addr64]
kernel_lint: 2 kernels, 2 flagged
"""
LINT_CMB2 = HDR + """g_attention_flash2_f16o                      loop   903   903   32     27.2  113   0   16    0   72   58  237   6    -  kept    16   1.5  147   1     0        -  copies
g_attention_flash2_partial_f16o              loop   950   950   32     28.7  113   0   16    0   59   43  237   6    -  kept    16  1.59   10   3     0        -  copies
"""
LINT_CMB_E520 = HDR + """g_attention_flash2_f16o                      loop   897   897   32     27.0  113   0   16    0   72   58  237   -    - drain     -   1.5   23   1     -        -  copies
g_attention_flash2_partial_f16o              loop   944   944   32     28.5  113   0   16    0   59   43  237   -    - drain     -  1.59    8   3     -        -  copies
"""   # kernel_lint on W/static-1125/combo/lint/combo.*.hsaco (E520's build)
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
        # Recurrence (11-25, T6 builder): '(searched: ...)' on a continuation line, or a dangling comma, read as no terms and passed
        cont = w("q305.sh", "#!/bin/bash\n# prior: E11,\n#   (searched: tile order)\necho\n")
        case("'(searched:' on a continuation line is refused (fail closed)", any("must be on the '# prior:' line" in x for x in lt.preflight(cont, log)[0]))
        dangle = w("q306.sh", "#!/bin/bash\n# prior: E11,\necho\n")
        case("a dangling comma after the rows is refused", any("not a parenthetical" in x for x in lt.preflight(dangle, log)[0]))
        note = w("q307.sh", "#!/bin/bash\n# prior: E10 (a note on why) (searched: tile order)\necho\n")
        case("a parenthetical note after the rows is admitted and its terms searched", not lt.preflight(note, log)[0])
        note2 = w("q308.sh", "#!/bin/bash\n# prior: E11 (a note on why) (searched: tile order)\necho\n")
        case("... and the terms in it still find E10", any("E10" in x for x in lt.preflight(note2, log)[0]))
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
        # Recurrence (HRX-like E508): q201 ran a timed measurement as `events_ab.sh ... && events_summary.py ...; sed ...`: a failed
        # measurement skipped its own judgement and the step went on green. A measurement in an && chain must stop the step on failure.
        silent = w("q212.sh", CARD + "bash $E/scripts/events_ab.sh $LA 100000 a b b a > abl.log 2>&1 \\\n  && python3 events_summary.py $LA/1-a > g.txt; sed -n 1p g.txt\n")
        rc, out = cli("probe-card", "check", silent)
        case(f"a measurement in an && chain with no stop on failure: MISSING ({out.strip()[:70]})", rc == 1 and "&& chain" in out)
        stops = w("q213.sh", CARD + "bash $E/scripts/agent_ab.sh a b out && python3 s.py out || { echo 'STOP'; exit 1; }\n")
        case("... the same chain ending in '|| { ...; exit 1; }': OK", cli("probe-card", "check", stops)[0] == 0)
        plain = w("q214.sh", CARD + "bash $K/common/counters.sh o waits -- x > o.log 2>&1 || fail counters\n[ -f a ] && echo ok\n")
        case("a measurement with its own stop, and an && chain with no measurement: OK", cli("probe-card", "check", plain)[0] == 0)
        quoted = w("q215.sh", CARD + "# events_ab.sh a && b (a comment)\necho 'hsaco_run.py && x'\n")
        case("a comment or a quoted string naming a tool is not a measurement", cli("probe-card", "check", quoted)[0] == 0)

        # bound line. Recurrence (HRX-like E536 / E538, rules section 2 'Price the lever'): levers were built and timed with no
        # stated price (the part's time, its limit, the best case for the whole request); exactly one well-formed bound line.
        nob = w("q216.sh", "".join(l + "\n" for l in CARD.split("\n") if not l.startswith("# bound:")))
        rc, out = cli("probe-card", "check", nob)
        case(f"no '# bound:' line: MISSING ({out.strip()[:60]})", rc == 1 and "0 '# bound:' lines" in out)
        bl = [l for l in CARD.split("\n") if l.startswith("# bound:")][0]
        for alt, ok_, what in (("# bound: n/a (diagnosis)", True, "n/a (diagnosis)"),
                               (bl.replace("flash2 60.1", "flash2 now 60.1"), True, "with 'now'"),
                               (bl.replace("none measured", "llama.cpp 41.2 s"), True, "a measured best engine"),
                               ("# bound: n/a", False, "n/a without (diagnosis)"),
                               ("# bound: n/a (tool)", False, "n/a (tool)"),
                               ("# bound: flash2 is slow", False, "free text"),
                               (bl.replace(" (amdahl.py)", ""), False, "no amdahl.py source"),
                               (bl.replace("(amdahl.py)", "(estimate)"), False, "a source other than amdahl.py"),
                               (bl.replace(" (floor 0.5 %)", ""), False, "no floor"),
                               (bl.replace("limit 29.7 s; ", ""), False, "no limit"),
                               (bl + "\n" + bl, False, "two bound lines"),
                               # review C16 F5 (r3): the rule the line records, not only its form. Recurrence: HRX-like E536 (q205's
                               # flash2 lever, 0.3-1.8 % of prefill, was queued against a 7 % bench floor; rules: 'not built')
                               (bl.replace("1.2 %", "0.3-1.8 %"), True, "f3: a range, judged by its upper end 1.8 >= 0.5"),
                               (bl.replace("1.2 %", "0.3-1.8 %").replace("floor 0.5 %", "floor 1.8 %"), True, "a range whose upper end is the floor"),
                               (bl.replace("1.2 %", "0.3-1.8 %").replace("floor 0.5 %", "floor 2 %"), False, "a range wholly below the floor"),
                               (bl.replace("1.2 %", "1.8-0.6 %"), False, "a range written backwards (both ends above the floor)"),
                               (bl.replace("served lane ABBA (floor 0.5 %)", "agent bench (floor 7 %)").replace("1.2 %", "0.3 %"), False,
                                "f4: E536's shape, best 0.3 % below the deciding bench's 7 % floor"),
                               (bl.replace("1.2 %", "0.5 %"), True, "best case equal to the floor"),
                               (bl.replace("1.2 %", "0.4 %"), False, "best case just below the floor"),
                               (bl.replace("(amdahl.py)", "(not amdahl.py)"), False, "f6: '(not amdahl.py)' names the tool, never ran it"),
                               (bl.replace("(amdahl.py)", "(estimated, cf. amdahl.py)"), False, "amdahl.py named after other text"),
                               (bl.replace("(amdahl.py)", "(common/amdahl.py --device cap85w W/x/events.json)"), True, "an amdahl.py command line"),
                               (bl.replace("(amdahl.py)", "(W/static-1125/q205/amdahl-flash2.txt)"), True, "a saved amdahl output path"),
                               (bl.replace("(amdahl.py)", "(W/static-1125/q205/guess.txt)"), False, "an output path not amdahl's"),
                               (bl.replace(" of the request", ""), False, "V17: no 'of the request'"),
                               (bl.replace("1.2 % of the request", "1.2 % of flash2"), False, "V17: a share of the part, not of the request"),
                               (bl.replace("1.2 %", "1.2"), False, "a best case without %"),
                               (bl.replace("floor 0.5 %", "floor 0.5"), False, "a floor without % (cannot be compared)"),
                               (bl.replace("floor 0.5 %", "floor 0.5 % decode"), True, "a floor with a note after the %"),
                               (bl + " and more", False, "text after the floor's paren")):
            p = w("q217.sh", CARD.replace(bl, alt))
            rc, out = cli("probe-card", "check", p)
            case(f"bound {what}: {'OK' if ok_ else 'MISSING'} ({out.strip()[:70]})", (rc == 0) == ok_)
        # compiler lines. Recurrence (E110, E230, E318; HRX-like E520): a predicted static delta was never checked in the built
        # binary, so a timing tested a change that was not there; each line must parse with a metric kernel_lint prints.
        cl = "# compiler: g_attention_flash2_f16o loop_instructions 952 -> 903"
        for alt, ok_, what in ((cl, True, "a well-formed line"),
                               (cl + "\n# compiler: g_attention_flash2(_partial)?_f16o vgpr 252 -> 237 (partial)", True, "two lines, a note"),
                               (cl.replace("loop_instructions", "valu_per_tile"), False, "an unknown metric (VALU per tile is not printed)"),
                               (cl.replace("loop_instructions", "s_barrier"), False, "s_barrier (not in kernel_lint's table)"),
                               (cl.replace("g_attention_flash2_f16o", "g_attention_flash2_(f16o"), False, "a regex that does not compile"),
                               (cl.replace(" -> 903", " 903"), False, "no arrow"),
                               (cl.replace("952 -> 903", "952 -> fewer"), False, "a word for a number"),
                               (cl + " junk", False, "V18: trailing text after TO"),
                               (cl + " (note) junk", False, "V18: trailing text after the note")):
            p = w("q218.sh", CARD.replace("# result:", alt + "\n# result:"))
            rc, out = cli("probe-card", "check", p)
            case(f"compiler {what}: {'OK' if ok_ else 'MISSING'} ({out.strip()[:70]})", (rc == 0) == ok_)
        # compiler-check against the built and base tables (E520's shape: the loop moved as predicted, the cover did not)
        base, cmb2, e520 = w("lint-base.txt", LINT_BASE), w("lint-cmb2.txt", LINT_CMB2), w("lint-e520.txt", LINT_CMB_E520)
        q205 = w("q219.sh", CARD.replace("# result:", cl + "\n# compiler: g_attention_flash2_f16o load_cover 58 -> 147\n# result:"))
        rc, out = cli("compiler-check", q205, cmb2, base)
        case(f"compiler-check, pkg-cmb2 vs live: both deltas present, COMPILER: OK ({out.strip()[-40:]})", rc == 0 and "COMPILER: OK" in out)
        rc, out = cli("compiler-check", q205, e520, base)
        case(f"compiler-check, E520's pkg-cmb vs live: the cover delta ABSENT, the loop line ok ({[l for l in out.split(chr(10)) if 'ABSENT' in l][:1]})",
             rc == 1 and "load_cover" in out and "COMPILER: ABSENT" in out and out.count("COMPILER ABSENT") == 1)
        rc, out = cli("compiler-check", q205, base, base)
        case("compiler-check, the base against itself: both ABSENT (nothing moved)", rc == 1 and out.count("COMPILER ABSENT") == 2)
        for line, what in (("# compiler: g_attention_flash2_f16o loop_instructions 952 -> 940", "a delta half as large (12 predicted, 49 read)"),
                           ("# compiler: g_attention_flash2_f16o loop_instructions 952 -> 1000", "the wrong direction"),
                           ("# compiler: g_attention_flash2_f16o loop_instructions 800 -> 903", "a base value far from FROM"),
                           ("# compiler: g_attention_flash2_f16o vgpr 240 -> 240", "'unchanged' predicted, 240 -> 237 read"),
                           ("# compiler: g_attention_flash2 loop_instructions 952 -> 903", "a regex matching no kernel (full match)"),
                           ("# compiler: g_attention_flash2.*_f16o loop_instructions 952 -> 903", "one of two matched kernels off (partial 1003 -> 950)"),
                           ("# compiler: g_attention_flash2_f16o pf 1 -> 2", "a metric not in the list")):
            p = w("q220.sh", CARD.replace("# result:", line + "\n# result:"))
            rc, out = cli("compiler-check", p, cmb2, base)
            case(f"compiler-check, {what}: refused ({out.strip().splitlines()[-1]})", rc == 1)
        rc, out = cli("compiler-check", w("q221.sh", CARD.replace("# result:", "# compiler: g_attention_flash2_f16o vgpr 240 -> 237\n# result:")), cmb2, base)
        case("compiler-check, an exact delta (VGPR 240 -> 237): OK", rc == 0)
        rc, out = cli("compiler-check", w("q223.sh", CARD.replace("# result:", "# compiler: g_attention_flash2_f16o loop_instructions 952 -> 910\n# result:")), cmb2, base)
        case("compiler-check, a loop count within 2 % of TO (910 predicted, 903 read): OK", rc == 0)
        rc, out = cli("compiler-check", w("q224.sh", CARD.replace("# result:", "# compiler: g_attention_flash2_f16o vgpr 240 -> 238\n# result:")), cmb2, base)
        case("compiler-check, VGPR is exact (238 predicted, 237 read): refused", rc == 1)
        rc, out = cli("compiler-check", w("q225.sh", CARD.replace("# result:", "# compiler: g_attention_flash2_f16o waves 6 -> 6\n# result:")), e520, base)
        case(f"compiler-check, a '-' value (no residency in the E520 table): refused ({out.strip()[-60:]})", rc == 1 and "no value" in out)
        near = w("lint-951.txt", LINT_BASE.replace("   952   952", "   951   951", 1))
        rc, out = cli("compiler-check", w("q226.sh", CARD.replace("# result:", "# compiler: g_attention_flash2_f16o loop_instructions 952 -> 945\n# result:")), near, base)
        case("compiler-check, 952 -> 945 predicted, 951 read (inside both 2 % bands, 1 of 7 moved): refused", rc == 1)
        def pair(b_loop, a_loop, line, n):
            bt = w(f"lint-b{n}.txt", LINT_BASE.replace("   952   952", f"   {b_loop}   {b_loop}", 1))
            at = w(f"lint-a{n}.txt", LINT_BASE.replace("   952   952", f"   {a_loop}   {a_loop}", 1))
            return cli("compiler-check", w(f"q23{n}.sh", CARD.replace("# result:", line + "\n# result:")), bt, at)
        rc, out = pair(903, 952, "# compiler: g_attention_flash2_f16o loop_instructions 1000 -> 903", 0)
        case("compiler-check, base 952 far from FROM 1000, built at TO, moved in the direction: refused", rc == 1)
        rc, out = pair(958, 952, "# compiler: g_attention_flash2_f16o loop_instructions 952 -> 945", 1)
        case("compiler-check, 952 -> 945 predicted, 952 -> 958 read (both inside their bands, moved the other way): refused", rc == 1)
        rc, out = pair(940, 960, "# compiler: g_attention_flash2_f16o loop_instructions 952 -> 952", 2)
        case("compiler-check, unchanged predicted, 960 -> 940 read (each inside 952's band, moved 20 > 19): refused", rc == 1)
        rc, out = pair(950, 954, "# compiler: g_attention_flash2_f16o loop_instructions 952 -> 952", 3)
        case("compiler-check, unchanged predicted, 954 -> 950 read (both within 2 of 952): OK", rc == 0)
        # review C16 F6 (r3): h1-h4 and the band edges. Recurrence: the 2 % band was relative to the value, so any delta under ~4 %
        # of the count passed at half its size, and 'unchanged' passed a 1.9 % move (E520's whole lever was 49 instructions, 5 %)
        for b_, a_, line, ok_, what in (
                (936, 952, "952 -> 920", True, "h1: half the predicted delta moved (936 = 920 + 16, the band edge): ok by the >= half rule"),
                (937, 952, "952 -> 920", False, "15 of 32 moved (under half)"),
                (1677, 1700, "1677 -> 1660", False, "h2: built reads FROM, base 1700 not FROM: ABSENT"),
                (952, 952, "952 -> 920", False, "built == FROM == base: ABSENT"),
                (934, 952, "952 -> 952", False, "h4: 'unchanged' predicted, 952 -> 934 read (1.9 %)"),
                (955, 952, "952 -> 952", False, "'unchanged', built 3 off (band 2)"),
                (921, 954, "952 -> 920", True, "base 2 off FROM (the band edge), built 1 off TO"),
                (920, 955, "952 -> 920", False, "base 3 off FROM (band 2)"),
                (900, 952, "952 -> 920", False, "an overshoot: 52 moved for 32 predicted (built 20 past TO, half-delta band 16)"),
                (936, 950, "952 -> 920", False, "base 2 toward TO, built at TO's band edge: 14 moved, under half of 32"),
                (953, 954, "952 -> 954", False, "a small delta (+2) read backwards: base 954, built 953 (each inside its band)")):
            rc, out = pair(b_, a_, "# compiler: g_attention_flash2_f16o loop_instructions " + line, 4)
            case(f"compiler-check, {what}: {'OK' if ok_ else 'refused'} ({out.strip().splitlines()[-1]})", (rc == 0) == ok_)
        # V13: the regex must fully match a kernel name ('flash2_f16' is a prefix of a real kernel: search would match it)
        rc, out = cli("compiler-check", w("q240.sh", CARD.replace("# result:", "# compiler: g_attention_flash2_f16 loop_instructions 952 -> 903\n# result:")), cmb2, base)
        case(f"compiler-check V13: a prefix of a kernel name matches nothing (full match): refused ({out.strip()[-50:]})", rc == 1 and "matches no kernel" in out)
        # review C16 F6 h3: an alternation branch matching no kernel (claimed, never checked); an optional group likewise
        for line, ok_, what in (
                ("g_attention_flash2(_f16o|_nope_f16o) loop_instructions 952 -> 903", False, "h3: alternative '_nope_f16o' matches no kernel"),
                ("g_attention_flash2_(?:f16o|nope) loop_instructions 952 -> 903", False, "a non-capturing alternation with a dead branch"),
                ("g_attention_flash2_f16o|g_attention_flash2_nope loop_instructions 952 -> 903", False, "a top-level dead alternative"),
                ("g_attention_flash2(_nope)?_f16o loop_instructions 952 -> 903", False, "an optional group no kernel takes"),
                ("g_attention_flash2(_partial)?_f16o vgpr 240 -> 237", False, "both branches checked: partial's vgpr 252 is not FROM"),
                ("g_attention_flash2_(f16o)\\1? loop_instructions 952 -> 903", False, "a back-reference"),
                ("g_attention_flash2_[f|x]16o loop_instructions 952 -> 903", True, "a '|' inside a character class is not an alternation"),
                ("g_attention_flash2_f16o|g_attention_flash2_partial_f16o selects 113 -> 113", True, "two live alternatives, both checked")):
            rc, out = cli("compiler-check", w("q241.sh", CARD.replace("# result:", "# compiler: " + line + "\n# result:")), cmb2, base)
            case(f"compiler-check, {what}: {'OK' if ok_ else 'refused'} ({out.strip().splitlines()[-2][-70:] if not ok_ else out.strip()[-12:]})", (rc == 0) == ok_)
        # V14: kernel-level rows (no loop) are read like loop rows
        krow = "g_rmsnorm_f16                                kernel   120   120    0      0.0    4   0    0    0   10    4   48  16    -     -     0     0    0   0     0        -\n"
        rc, out = cli("compiler-check", w("q242.sh", CARD.replace("# result:", "# compiler: g_rmsnorm_f16 vgpr 48 -> 40\n# result:")),
                      w("lint-kb.txt", LINT_CMB2 + krow.replace("   48  16", "   40  16")), w("lint-ka.txt", LINT_BASE + krow))
        case(f"compiler-check V14: a kernel-level row (where = kernel) is read: OK ({out.strip()[-14:]})", rc == 0)
        # V16: compiler-check judges the compiler lines whatever the bound line says (a malformed bound is probe-card's refusal)
        nb = CARD.replace(bl, "# bound: flash2 is slow").replace("# result:", cl + "\n# result:")
        rc, out = cli("compiler-check", w("q243.sh", nb), cmb2, base)
        case(f"compiler-check V16: a malformed bound line does not stop the check: OK ({out.strip()[-14:]})", rc == 0 and "base 952 built 903" in out)
        # F7 / V15: verdict filters problems by field, not by message text
        rc, out = cli("verdict", w("q244.sh", CARD.replace("# budget: 45 min", "# budget: soon '# bound:'")), "0.98")
        case(f"verdict F7 g2: a malformed budget line quoting '# bound:' still errors ({out.strip()[:40]})", rc == 2 and "VERDICT: ERROR" in out)
        rc, out = cli("verdict", w("q245.sh", CARD.replace("# result:", "# compiler: g_x valu_per_tile 1 -> 2\n# result:")), "0.98")
        case(f"verdict V15: a malformed compiler line (an admission line) does not stop the verdict: INSIDE ({out.strip()[:30]})", rc == 0)
        rc, out = cli("verdict", w("q246.sh", CARD.replace(bl, bl.replace("1.2 %", "0.1 %"))), "0.98")
        case(f"verdict: a bound below its floor (an admission problem) does not stop the verdict: INSIDE ({out.strip()[:30]})", rc == 0)
        rc, out = cli("verdict", w("q227.sh", "".join(l + "\n" for l in CARD.split("\n") if not l.startswith("# bound:"))), "0.98")
        case(f"verdict does not need the bound line (an admission line, checked before the run): INSIDE ({out.strip()[:40]})", rc == 0)
        only = w("lint-only.txt", LINT_CMB2.split("g_attention_flash2_partial")[0])
        rc, out = cli("compiler-check", w("q222.sh", CARD.replace("# result:", "# compiler: g_attention_flash2.*_f16o vgpr 252 -> 237\n# result:")), only, base)
        case("compiler-check, a matched kernel in one table only: refused", rc == 1 and "table only" in out)
        mixed = w("lint-mixed.txt", LINT_CMB2 + LINT_BASE.split("\n", 1)[1])
        rc, out = cli("compiler-check", q205, mixed, base)
        case(f"compiler-check, a table holding two builds (a kernel twice): error, exit 2 ({out.strip()[:60]})", rc == 2 and "twice" in out)
        rc, out = cli("compiler-check", q205, w("notlint.txt", "nothing\n"), base)
        case("compiler-check, a file without kernel_lint's header: error, exit 2", rc == 2 and "no kernel_lint header" in out)
        d = os.path.join(t, "lintdir")
        os.makedirs(d)
        open(os.path.join(d, "lint.txt"), "w").write(LINT_CMB2)
        open(os.path.join(d, "run.log"), "w").write("build log\n")
        rc, out = cli("compiler-check", q205, d, base)
        case("compiler-check, a directory holding one table: read", rc == 0)
        open(os.path.join(d, "lint2.txt"), "w").write(LINT_CMB2)
        rc, out = cli("compiler-check", q205, d, base)
        case("compiler-check, a directory holding two tables: error, exit 2", rc == 2 and "2 kernel_lint tables" in out)
        rc, out = cli("compiler-check", good, cmb2, base)
        case("compiler-check, a step without compiler lines: OK, nothing predicted", rc == 0 and "nothing predicted" in out)

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

        # archive. Recurrence: a log trimmed to its active rows (closed rows moved to an archive file) would let a refuted lever
        # be re-queued as new, and would report a ported win as still owed once the owed row and its port sat in different files.
        rows = LOG.split("\n")
        head, body = rows[:4], [r for r in rows[4:] if r]
        moved = [r for r in body if r.startswith(("| E10 ", "| E13 ", "| E18 "))]
        kept = [r for r in body if r not in moved]
        hist = w("HIST.md", "# archive\n\n" + "\n".join(head[2:] + moved) + "\n")
        active = w("ACTIVE.md", "\n".join(head[:2] + ["Archive: HIST.md", ""] + head[2:] + kept) + "\n")
        alog = lt.Log(active)
        pr, _ = lt.preflight(reopen, alog)
        case(f"a loss row moved to the archive still refuses the repeat ({pr})", len(pr) == 1 and "E10 (no gain, refuted)" in pr[0])
        bare = w("BARE.md", open(active).read().replace("Archive: HIST.md\n", ""))
        case("without the Archive line the moved row is not seen (the line is what carries it)",
             not lt.preflight(reopen, lt.Log(bare))[0])
        rc, out = cli("--ledger", bare, "--ledger", hist, "preflight", reopen)
        case(f"--ledger given twice reads both files (rc {rc})", rc == 1 and "E10" in out)
        rc, out = cli("--ledger", active, "preflight", cited)
        case(f"a prior citing a moved row is satisfied (rc {rc})", rc == 0)
        gone = w("GONE.md", open(active).read().replace("Archive: HIST.md", "Archive: NOPE.md"))
        rc, out = cli("--ledger", gone, "preflight", reopen)
        case(f"a named archive that cannot be read: exit 2, never a skip ({out.strip()[:60]})", rc == 2)
        rc, out = cli("--ledger", active, "owed")
        case(f"owed across files: E13 (archived) keeps q6_k after E15 (active) ported iq4_xs; OWED: 2 ({out.split(chr(10))[:3]})",
             rc == 1 and "E13 -> q6_k:" in out and "OWED: 2" in out)
        a2 = w("ACTIVE2.md", open(active).read() + "| E17 | 11-25 | matvec | MV_PF on q6_k | bench | tie | kept; ported: E13 q6_k |\n")
        h2 = w("HIST.md", open(hist).read() + "| E18 | 11-25 | port | 0.3.0 | bench | exact | ported: E16 sdk-0.3.0; dropped: E16 inkling (x) |\n")
        rc, out = cli("--ledger", a2, "owed")
        case(f"a port archived while its owed row stays active still clears it (ids, not files, set the order): OWED: 0 ({out.strip()})",
             rc == 0 and "OWED: 0" in out)
        dup = w("DUP.md", open(bare).read().replace("| E11 | 11-21 | gemm | SW_A=1 on the candidate | bench | exact; not slower (paired 0.99) | kept |",
                                                   "| E11 | 11-21 | gemm | SW_A=1 on the candidate | bench | exact; not slower (paired 0.99) | kept |\n"
                                                   "| E10 | 11-21 | gemm | tile order | bench | exact | kept |"))
        dl = lt.Log([hist, dup])   # the refuted copy read first: a later clean copy must not hide it
        case(f"an id in two files: both rows are searched (the refuted copy still refuses) ({dl.dupes})",
             dl.dupes == ["E10"] and len(lt.preflight(reopen, dl)[0]) == 1)
    print(f"{len(fails)} failed" if fails else "OK")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
