#!/usr/bin/env python3
"""Checks between an experiment log and the steps queued against it: a step must cite the rows that already judged its lever,
declare what it predicts before it runs, and be judged against that prediction after; kept wins owed to other targets stay listed
until a row marks them ported. Model- and repo-agnostic: any log kept as a markdown table with one row per experiment
(`| ID | ... | result | decision |`) and steps kept as script files whose leading comment block is the header.

Recurrences (a real log, 11-23): a step re-queued workgroup tile order as a new hypothesis while a row had refuted it the day before;
another was queued from a static count while a row had shown the path it trimmed is not the cost. Both followed a written rule
("search the log for the lever and cite the rows") that nothing checked. Results were believed without a prior prediction to
compare them with (a harness said 2x where serving gave 3-8 %).

Row contract: a line starting `| ID |` (ID matches --id, default [A-Z]+\\d+). Its result / decision text is the cells from
--result-from on (default -2: the last two). The whole row is matched for terms; only the result / decision text for loss words.

Archive: a log kept short (active rows only) names the file its closed rows moved to on a line of its own, `Archive: HISTORY.md`
(a path relative to the log). Every subcommand then reads both, rows ordered by id, so a loss moved to the archive still blocks a
repeat and an owed row stays matched to its ported marks. A named archive that cannot be read is an error (exit 2), never a skip.
--ledger may also be given more than once. An id found in two files is searched with both rows' text (the log's own check should
refuse the duplicate).

Header contract (lines of the leading comment block, after a shebang):
  # prior: E175, E305 [free text, optionally (searched: term, term)]    exactly one prior line, in one of these two forms
  # prior: none (searched: term1, term2)                              (search terms are required with none)
  # changed: E175 <what changed since that row>                       answers a row instead of citing it (any number)
  # predict: <metric> <lo>-<hi>          e.g. 'decode ratio 0.97-0.99' (also lo..hi, lo to hi); lo <= hi
  # kill: <condition>                    what result stops the lever (free text)
  # budget: <minutes>                    the step's time budget, > 0 (an optional 'min' after the number)
  # bound: PART [now] X (limit Y; best engine Z | none measured), best case B % of the request (amdahl.py ...), resolved by M (floor F %)
                                       exactly one: the lever's price (rules section 2, 'Price the lever'; HRX-like E536 / E538);
                                       a diagnosis or tool step: '# bound: n/a (diagnosis)'. Anything else is refused. B may be a
                                       range lo-hi (judged by hi); B and F in %; the source starts with an amdahl.py command or
                                       its saved output path; B below F is refused (name the finer measurement that decides).
  # compiler: KERNEL_REGEX METRIC FROM -> TO [(note)]   optional, one per predicted static delta in the built package: the regex
                                       compiles, METRIC is a kernel_lint table column (COMPILER_METRICS), FROM / TO numbers
  # result: <regex>                      optional: one capture group, the result figure (the last match in the result file wins)
  # result-file: <path>                  optional: where verdict reads it (relative to the step's directory)

Owed ports (a kept win that must still be carried to other targets): the row carries `owed: target, target` (targets are tokens
without spaces, ended by ';', '|', '(', ')' or a sentence end); a later row carries `ported: ID target[, target]` or
`dropped: ID target <why>`. Text inside backticks is ignored, so the syntax can be quoted.

Subcommands and verdict lines (exit 0 ok, 1 refused / outside / owed, 2 usage or unreadable input):
  preflight STEP            LEDGER: PREFLIGHT_OK | PREFLIGHT_REFUSED   every row naming one of the step's terms with a loss word
                            must be cited in prior: or answered by changed:. Terms: the step's switches (NAME=value tokens in
                            --switch-var="..." values and `env A=1 cmd` prefixes, minus --exclude-switches-from's) plus the
                            searched terms (switches: exact name; searched: case-insensitive, from a word start). Loss words are
                            matched after negated forms ('not slower') are removed. It matches words, not diagnoses.
  probe-card check STEP     PROBE: OK | MISSING                       predict, kill, budget and bound present and well formed;
                                                                      compiler lines (if any) well formed
  compiler-check STEP BUILT BASE   COMPILER: OK | ABSENT            each '# compiler:' line against kernel_lint's table (its stdout,
                            saved; or a directory holding exactly one such file) of the built package and of the base: every kernel
                            the regex fully matches, present in both tables, must read FROM in BASE and TO in BUILT: exact for
                            vgpr, waves, scratch, in_flight, selects; otherwise BASE within the band of FROM, BUILT within half the
                            predicted delta of TO, having moved at least half of it in its direction; FROM == TO: both within an
                            absolute band (counts 2, per-MMA ratios 0.05). Every alternative of every '|' and every capturing
                            group must match a kernel (a claimed branch nothing checks: ABSENT). A character class is not split:
                            'k[xy]' checks only the kernels that exist (kx), so name each claimed kernel by an alternative
                            ('k(x|y)') when it must exist. A regex matching no kernel, a kernel in one table only, a '-' value, a table with
                            a kernel twice (two builds mixed) or without kernel_lint's header: ABSENT / error. Recurrence: E520's
                            three-lever flash2 build lost the V-tile prefetch (cover 58 -> 23) while its loop count moved as
                            predicted; the delta a hypothesis predicts is checked before any GPU minute (E110, E230, E318).
  verdict STEP [RESULT]     VERDICT: INSIDE | OUTSIDE                 RESULT: a file (the header's result regex, last match) or a
                            number; default the header's result-file. OUTSIDE: debug the measurement before believing it.
  owed                      OWED: N                                   kept wins with targets no later row marked ported / dropped
--grandfather-list FILE (preflight, probe-card): steps written before the checks, pinned by content, in sha256sum format
('<sha256>  <step file name>', one per line; '#' comments). A step passes with a note only when its file name is listed and its
content still hashes to the listed sha256: a listed step rewritten in place is checked like a new one (HRX-like C16 verify r3: a
digit-run cut let 'q1_flash.sh' skip both checks). A list that cannot be read or has a malformed line: exit 2, never a skip.
usage: ledger_tool.py [--ledger LOG.md [--ledger ARCHIVE.md]] [--id REGEX] [--result-from K] {preflight,probe-card check,compiler-check,verdict,owed} ..."""
import argparse
import hashlib
import os
import re
import sys

# loss words, counted over a real log's result / decision cells (refuted 33, slower 27, removed 26, dropped 19, failed 19, rejected 16,
# lost 14, no gain 11, parked 9, worse 8, reverted 7, not worth 7, ...). Not loss words: 'tie' (a tie can be kept), 'held' (waiting).
LOSS = re.compile(r"\b(refuted|refutes|lost|loses|slower|removed|reverted|no gain|gained nothing|not the cost|dropped|could not help|"
                  r"worse|regressed|regresses|not worth|parked|failed|does not pay|no effect|not a win|rejected|discarded|shelved)\b", re.I)
NEGATED = re.compile(r"\b(not|no|never|nor)\s+(slower|worse|regressed|lost|failed)\b", re.I)
SWITCH = re.compile(r"\b([A-Z][A-Z0-9_]{2,})=([A-Za-z0-9_.:-]+)(?=\s|$|[\"'])")
NUM = r"[-+]?\d+(?:\.\d+)?"
PREDICT = re.compile(r"^#\s*predict:\s*(?P<metric>.*?\S)\s+(?P<lo>" + NUM + r")\s*(?:-|\.\.|to)\s*(?P<hi>" + NUM + r")\s*%?"
                     r"(?:\s*\(.*\))?\s*$")
KILL = re.compile(r"^#\s*kill:\s*\S")
BUDGET = re.compile(r"^#\s*budget:\s*(" + NUM + r")\s*(?:min|minutes)?(?:\s*\(.*\))?\s*$")
TARGET = re.compile(r"^[\w.:+/-]+$")
UNUM = r"\d+(?:\.\d+)?"
BOUND = re.compile(r"^#\s*bound:\s*(?P<part>\S.*?)\s+(?:now\s+)?(?P<now>" + NUM + r")\s*[^\s(),;]*\s*"
                   r"\(limit\s+(?P<limit>" + NUM + r")[^;()]*;\s*best engine\s+(?P<engine>[^()]*?\S)\s*\)\s*,\s*"
                   r"best case\s+(?P<best>" + UNUM + r")(?:\s*-\s*(?P<best_hi>" + UNUM + r"))?\s*%\s+of the request\s*"
                   r"\((?P<tool>[^()]*)\)\s*,\s*"
                   r"resolved by\s+(?P<by>[^()]*?\S)\s*\(floor\s+(?P<floor>" + UNUM + r")\s*%(?:[\s,;][^()]*)?\)\s*$")
# the source of the best case: amdahl.py run (its command, e.g. 'amdahl.py', 'common/amdahl.py --device cap85w') or its saved
# output (a path whose file name starts with 'amdahl'); text that merely names it ('not amdahl.py') is refused (review C16 F5)
BOUND_TOOL = re.compile(r"^(?:\S*/)?amdahl\.py(?:\s.*)?$|^\S*/amdahl[\w.-]*\.(?:txt|json|md|log)(?:\s.*)?$")
BOUND_NA = re.compile(r"^#\s*bound:\s*n/a \(diagnosis\)\s*$")
COMPILER = re.compile(r"^#\s*compiler:\s*(?P<rx>\S+)\s+(?P<metric>\S+)\s+(?P<frm>" + NUM + r")\s*->\s*(?P<to>" + NUM +
                      r")\s*(?:\([^()]*\))?\s*$")
# metric -> kernel_lint table column (the columns kernel_lint.py prints; 'VALU per tile' and s_barrier are not printed: refused)
COMPILER_METRICS = {"loop_instructions": "loop", "common_path": "comm", "non_mma_per_mma": "other/op", "selects": "sel",
                    "scratch": "scr", "moves": "mov", "move_run": "run", "vgpr": "vgpr", "waves": "w/S", "load_cover": "cov",
                    "in_flight": "fly", "lds_waits_per_mma": "ldsW"}
# exact metrics: one unit is the effect (registers, waves, spills, loads in flight, selects): base == FROM and built == TO.
# Other metrics (review C16 F6): a predicted change FROM -> TO passes when base reads FROM within the absolute band, built within
# |TO - FROM| / 2 of TO, and built - base moved at least half the delta in its direction; a predicted 'unchanged' (FROM == TO)
# passes when both read within the band of FROM. Band: instruction counts 2, per-MMA ratios 0.05 (FROM is a base-table reading).
COMPILER_EXACT = {"vgpr", "waves", "scratch", "in_flight", "selects"}
COMPILER_BAND = {m: (0.0 if m in COMPILER_EXACT else 0.05 if m in ("non_mma_per_mma", "lds_waits_per_mma") else 2.0)
                 for m in COMPILER_METRICS}
END = r"(?=[;|()`]|\.(?:\s|$)|$)"
ARCHIVE = re.compile(r"^Archive:\s*(\S+)\s*$")


class Log:
    """the experiment log: rows in file order (by id when several files are read), id -> (row text, result / decision text with
    negated loss words removed, what). path: a file or a list of files; a file's `Archive: PATH` line adds that file."""

    def __init__(self, path, id_re=r"[A-Z]+\d+", result_from=-2, what_cell=None):
        paths = [path] if isinstance(path, str) else list(path)
        self.id_re, self.rows, self.order, self.dupes = id_re, {}, [], []
        row = re.compile(r"^\|\s*(" + id_re + r")\s*\|")
        self.paths, seen, named = [], set(), set()
        for p in paths:   # grows while read: a log's Archive line appends its archive
            if os.path.realpath(p) in seen:
                continue
            seen.add(os.path.realpath(p))
            self.paths.append(p)
            for line in open(p, encoding="utf-8"):
                a = ARCHIVE.match(line)
                if a and p not in named:   # one level: an archive's own Archive line is not followed
                    named.add(os.path.join(os.path.dirname(p), a.group(1)))
                    paths.append(os.path.join(os.path.dirname(p), a.group(1)))
                m = row.match(line)
                if m:
                    cells = [c.strip() for c in line.strip().strip("|").split("|")]
                    what = cells[what_cell] if what_cell is not None and what_cell < len(cells) else " ".join(cells[1:result_from])
                    entry = (line, NEGATED.sub("", " ".join(cells[result_from:])), what)
                    i = m.group(1)
                    if i in self.rows:
                        self.dupes.append(i)
                        old = self.rows[i]
                        entry = (old[0] + line, old[1] + " " + entry[1], old[2])
                    else:
                        self.order.append(i)
                    self.rows[i] = entry
        self.path = self.paths[0]
        self.name = " / ".join(os.path.basename(p) for p in self.paths)
        if len(self.paths) > 1:
            self.order.sort(key=idkey)


def idkey(i):
    m = re.match(r"(\D*)(\d+)", i)
    return (m.group(1), int(m.group(2))) if m else (i, 0)


def header(text):
    """the leading comment block (after the shebang)"""
    out = []
    for line in text.split("\n")[1 if text.startswith("#!") else 0:]:
        if line.startswith("#"):
            out.append(line.rstrip())
        elif line.strip():
            break
    return out


def switches(text, var="EXTRA"):
    """NAME=value switches the step builds or runs with: tokens inside VAR="..." / '...' and `env A=1 B=2 cmd` prefixes (comments
    dropped). A VAR whose value is not a literal string (VAR=$X, VAR="$X ...") cannot be read: returned as unreadable."""
    code = "\n".join(l.split(" #")[0] for l in text.split("\n") if not l.lstrip().startswith("#"))
    found, unreadable = set(), []
    for m in re.finditer(r"\b" + re.escape(var) + r"=(\"[^\"]*\"|'[^']*'|\S*)", code):
        v = m.group(1)
        if "$" in v or "`" in v:
            unreadable.append(m.group(0))
        found.update(n for n, _ in SWITCH.findall(v.strip("\"'")))
    for m in re.finditer(r"(?:^|[;&|({]\s*|\s)env((?:\s+[A-Z][A-Z0-9_]*=\S+)+)", code, re.M):
        found.update(n for n, _ in SWITCH.findall(m.group(1)))
    return found, unreadable


def switches_of_file(path):
    """switch names in a build command file (text before the first '#'): the live build's switches, judged by its go-live"""
    return {n for n, _ in SWITCH.findall(open(path).read().split("#")[0])}


def grandfathered(step, listfile):
    """True when STEP's file name is in LISTFILE (sha256sum format) with the sha256 of its current content; None LISTFILE:
    False. Raises ValueError on a malformed line (OSError on an unreadable list): fail closed, never a skip"""
    if listfile is None:
        return False
    want = {}
    for n, line in enumerate(open(listfile), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = re.fullmatch(r"([0-9a-f]{64}) [ *](\S+)\s*", line)
        if not m or "/" in m.group(2):
            raise ValueError(f"{listfile}:{n}: not '<sha256>  <file name>': {line.strip()[:80]!r}")
        want[m.group(2)] = m.group(1)
    name = os.path.basename(step)
    return name in want and hashlib.sha256(open(step, "rb").read()).hexdigest() == want[name]


def preflight(step, log, exclude=frozenset(), exclude_note=None, grandfather=None, switch_var="EXTRA"):
    """(problems, notes) for one step against the log"""
    name = os.path.basename(step)
    if grandfathered(step, grandfather):
        return [], [f"{name}: grandfathered (listed with its sha256 in {os.path.basename(grandfather)}, written before this check)"]
    text = open(step).read()
    hd = header(text)
    rows, idr = log.rows, log.id_re
    probs, notes = [], []
    pl = [l for l in hd if re.match(r"^#\s*prior:", l)]
    if len(pl) != 1:
        return [f"{name}: {len(pl)} '# prior:' lines in the header (need exactly one: '# prior: E123, E456' or "
                f"'# prior: none (searched: term1, term2)')"], notes
    me = re.match(r"^#\s*prior:\s*((?:" + idr + r")(?:\s*,\s*(?:" + idr + r"))*)(.*)$", pl[0])
    mn = re.match(r"^#\s*prior:\s*none\s*\(searched:\s*([^)]*\S[^)]*)\)\s*$", pl[0])
    cited, terms = set(), []
    # fail closed on what the parser cannot read (T6 builder, 11-25): '(searched: ...)' on a continuation line, or a prior line
    # ending in a dangling comma, passed with no terms searched while the same terms on one line gave 17 refusals
    stray = [l for l in hd if "(searched:" in l and l is not pl[0]]
    if stray:
        return [f"{name}: '(searched: ...)' must be on the '# prior:' line itself, found on {stray[0]!r}"], notes
    if me:
        cited = set(re.findall(idr, me.group(1)))
        rest = me.group(2).strip()
        # known-good tails: nothing, or parentheticals ('(searched: a, b)', a note); a dangling comma or bare words are not
        if rest and not rest.startswith("("):
            return [f"{name}: '# prior:' line has text after its rows that is not a parenthetical: {rest!r}"], notes
        s = re.search(r"\(searched:\s*([^)]*\S[^)]*)\)", rest)
        if "searched:" in rest and not s:
            return [f"{name}: '# prior:' line has a 'searched:' clause the parser cannot read: {rest!r}"], notes
        terms = [t.strip() for t in s.group(1).split(",") if t.strip()] if s else []
    elif mn:
        terms = [t.strip() for t in mn.group(1).split(",") if t.strip()]
    else:
        return [f"{name}: '# prior:' line not in a known form: {pl[0]!r}"], notes
    for e in sorted(cited - set(rows), key=idkey):
        probs.append(f"{name}: prior cites {e}, which is not in {log.name}")
    answered = set()
    for l in hd:
        if re.match(r"^#\s*changed:", l):
            c = re.match(r"^#\s*changed:\s*(" + idr + r")\b\s*\S", l)
            if not c:
                probs.append(f"{name}: '# changed:' line must name a row and what changed: {l!r}")
            else:
                answered.add(c.group(1))
    sw, unreadable = switches(text, switch_var)
    for u in unreadable:
        probs.append(f"{name}: switches not readable from {u!r} (write them literally in {switch_var})")
    notes.append(exclude_note or ("excluded switches: " + (", ".join(sorted(exclude)) or "none")))
    new_sw = sorted(sw - set(exclude))
    notes.append(f"{name}: switches checked {new_sw or 'none'}; searched terms {terms or 'none'}")
    hits = {}
    for t in new_sw:
        rx = re.compile(r"\b" + re.escape(t) + r"\b")
        for e, (row, res, _) in rows.items():
            if rx.search(row) and LOSS.search(res):
                hits.setdefault(e, []).append(t)
    for t in terms:
        rx = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(t), re.I)
        for e, (row, res, _) in rows.items():
            if rx.search(row) and LOSS.search(res):
                hits.setdefault(e, []).append(t)
    for e in sorted(hits, key=idkey):
        if e not in cited and e not in answered:
            marks = sorted({x.lower() for x in LOSS.findall(rows[e][1])})
            probs.append(f"{name}: {e} ({', '.join(marks)}) names {', '.join(repr(x) for x in hits[e])} and is not cited: "
                         f"{rows[e][2][:90]!r}; cite it in '# prior:' or answer it with '# changed: {e} <what changed since>'")
    return probs, notes


def card(step):
    """(fields, problems) of the probe-card lines: predict (metric, lo, hi), kill, budget minutes, bound, compiler, result regex,
    result file"""
    f, tagged = card_tagged(step)
    return f, [m for _, m in tagged]


class _Tagged(list):
    """problems tagged with the header key they belong to (review C16 F7: callers filter by key, never by message text)"""
    key = None

    def append(self, msg):
        super().append((self.key, msg))


def card_tagged(step):
    """(fields, [(key, problem)]): card() with each problem tagged by its header key"""
    hd = header(open(step).read())
    f, probs = {}, _Tagged()
    for key, rx in (("predict", PREDICT), ("kill", KILL), ("budget", BUDGET)):
        probs.key = key
        ls = [l for l in hd if re.match(r"^#\s*" + key + r":", l)]
        if len(ls) != 1:
            probs.append(f"{len(ls)} '# {key}:' lines (need exactly one)")
            continue
        m = rx.match(ls[0])
        if not m:
            probs.append(f"'# {key}:' line not in a known form: {ls[0]!r}")
        elif key == "predict":
            lo, hi = float(m.group("lo")), float(m.group("hi"))
            if lo > hi:
                probs.append(f"predict range {lo} > {hi}")
            f["predict"] = (m.group("metric"), lo, hi)
        elif key == "budget":
            if float(m.group(1)) <= 0:
                probs.append(f"budget {m.group(1)} is not > 0 minutes")
            f["budget"] = float(m.group(1))
        else:
            f["kill"] = ls[0].split(":", 1)[1].strip()
    probs.key = "bound"
    ls = [l for l in hd if re.match(r"^#\s*bound:", l)]
    if len(ls) != 1:
        probs.append(f"{len(ls)} '# bound:' lines (need exactly one: '# bound: PART now X (limit Y; best engine Z | none measured), "
                     "best case B % of the request (amdahl.py ...), resolved by M (floor F %)' or '# bound: n/a (diagnosis)')")
    elif BOUND_NA.match(ls[0]):
        f["bound"] = None
    elif BOUND.match(ls[0]):
        b = f["bound"] = BOUND.match(ls[0]).groupdict()
        lo, hi = float(b["best"]), float(b["best_hi"] or b["best"])
        if not BOUND_TOOL.match(b["tool"].strip()):
            probs.append(f"'# bound:' best case source {b['tool']!r} is not an amdahl.py run or its saved output "
                         "('amdahl.py ...', 'common/amdahl.py ...', '<dir>/amdahl-....txt')")
        elif lo > hi:
            probs.append(f"'# bound:' best case range {lo:g}-{hi:g} % runs backwards")
        elif hi < float(b["floor"]):
            probs.append(f"'# bound:' best case {hi:g} % is below the floor {float(b['floor']):g} % of '{b['by']}': a lever below "
                         "every measurement's floor is not built; name the finer measurement that decides (floor at or below the "
                         "best case) in 'resolved by' (rules section 2, 'Price the lever'; HRX-like E536)")
    else:
        probs.append(f"'# bound:' line not in a known form: {ls[0]!r}")
    probs.key = "compiler"
    f["compiler"] = []
    for l in (l for l in hd if re.match(r"^#\s*compiler:", l)):
        m = COMPILER.match(l)
        if not m:
            probs.append(f"'# compiler:' line not in a known form (KERNEL_REGEX METRIC FROM -> TO): {l!r}")
            continue
        if m.group("metric") not in COMPILER_METRICS:
            probs.append(f"'# compiler:' metric {m.group('metric')!r} is not one kernel_lint prints "
                         f"({', '.join(sorted(COMPILER_METRICS))}): {l!r}")
            continue
        try:
            rx = re.compile(m.group("rx"))
        except re.error as e:
            probs.append(f"'# compiler:' regex does not compile ({e}): {l!r}")
            continue
        f["compiler"].append((l, rx, m.group("metric"), float(m.group("frm")), float(m.group("to"))))
    for key in ("result", "result-file"):
        probs.key = key
        ls = [l.split(":", 1)[1].strip() for l in hd if re.match(r"^#\s*" + key + r":", l)]
        if len(ls) > 1:
            probs.append(f"{len(ls)} '# {key}:' lines (at most one)")
        elif ls:
            f[key] = ls[0]
    probs.key = "result"
    if "result" in f:
        try:
            if re.compile(f["result"]).groups != 1:
                probs.append(f"result regex needs exactly one capture group: {f['result']!r}")
        except re.error as e:
            probs.append(f"result regex does not compile ({e}): {f['result']!r}")
    return f, list(probs)


# a timed or judged GPU measurement (HRX-like E508: q201's `events_ab.sh ... && events_summary.py ...; sed` skipped its judgement when
# the measurement failed and the step went on green): in an && chain, the logical line must end the step on failure
MEASURE = re.compile(r"\b(events_ab\.sh|agent_ab\.sh|counters\.sh|hsaco_run\.py|replay\.py|eq\.sh|gates\.sh|ktest\.sh|serve_as\.sh)\b")
STOPS = re.compile(r"\|\|\s*(\{[^}]*\b(exit|fail|die)\b|exit\b|fail\b|die\b)")


def chains(text):
    """Problems: logical lines (continuations joined; comments and quoted strings blanked) that run a measurement inside an && chain
    and do not end the step when it fails. Known-good form only: a '|| exit / fail / die' (or '|| { ...; exit ...; }') on the line."""
    probs = []
    for line in re.sub(r"\\\n", " ", text).split("\n"):
        code = re.sub(r"'[^']*'|\"[^\"]*\"", "''", line)
        code = re.sub(r"(^|\s)#.*", "", code)
        if MEASURE.search(code) and "&&" in code and not STOPS.search(code):
            probs.append(f"a measurement in an && chain with no stop on failure (a failed run skips its judgement, E508): {line.strip()[:100]!r}")
    return probs


def verdict(step, result=None):
    """(line, exit code, detail lines)"""
    f, tagged = card_tagged(step)
    # bound and compiler are admission lines (probe-card check's job, before the run): the verdict judges the rest, by key
    probs = [m for k, m in tagged if k in ("predict", "kill", "budget", "result", "result-file")]
    if "predict" not in f or probs:
        return "VERDICT: ERROR", 2, ["probe card not valid: " + "; ".join(probs or ["no predict line"])]
    metric, lo, hi = f["predict"]
    src = result or (os.path.join(os.path.dirname(os.path.abspath(step)), f["result-file"]) if "result-file" in f else None)
    if src is None:
        return "VERDICT: ERROR", 2, ["no RESULT given and no '# result-file:' in the header"]
    if re.fullmatch(NUM, src.strip()):
        val, where = float(src), "given"
    else:
        if "result" not in f:
            return "VERDICT: ERROR", 2, ["a result file needs a '# result: <regex>' line in the header"]
        try:
            ms = re.findall(f["result"], open(src, encoding="utf-8", errors="replace").read(), re.M)
        except OSError as e:
            return "VERDICT: ERROR", 2, [f"result file unreadable: {e}"]
        if not ms:
            return "VERDICT: ERROR", 2, [f"result regex {f['result']!r} matches nothing in {src}"]
        try:
            val = float(ms[-1])
        except ValueError:
            return "VERDICT: ERROR", 2, [f"result {ms[-1]!r} is not a number"]
        where = os.path.basename(src)
    inside = lo <= val <= hi
    line = f"VERDICT: {'INSIDE' if inside else 'OUTSIDE'} ({metric} {val:g} from {where}; predicted {lo:g}-{hi:g})"
    return line, 0 if inside else 1, [] if inside else ["debug the measurement before believing it (then the lever); kill: "
                                                         + f.get("kill", "?")]


def owed(log):
    """(items [(id, remaining targets, what)], problems): `owed:` targets with no later `ported:` / `dropped:` mark"""
    debt, probs = {}, []
    tok = r"((?:[^;|()`.]|\.(?!\s|$))*)"
    for e in log.order:
        text = re.sub(r"`[^`]*`", "", log.rows[e][0])
        for m in re.finditer(r"\b(ported|dropped):\s*(" + log.id_re + r")\b\s*" + tok + END, text):
            src = m.group(2)
            ts = [t.strip().split()[0] for t in m.group(3).split(",") if t.strip()]
            if src not in debt:
                probs.append(f"{e}: {m.group(1)}: {src}, which carries no earlier owed: list")
                continue
            for t in ts:
                if t in debt[src]:
                    debt[src].remove(t)
                else:
                    probs.append(f"{e}: {m.group(1)}: {src} {t}: not an owed target of {src} ({sorted(debt[src]) or 'none left'})")
        for m in re.finditer(r"\bowed:\s*" + tok + END, text):
            ts = [t.strip() for t in m.group(1).split(",")]
            bad = [t for t in ts if not TARGET.match(t)]
            if bad or not ts:
                probs.append(f"{e}: owed: list not in a known form (targets without spaces, comma-separated): {m.group(0)!r}")
                continue
            debt.setdefault(e, [])
            debt[e] += [t for t in ts if t not in debt[e]]
    items = [(e, ts, log.rows[e][2]) for e, ts in sorted(debt.items(), key=lambda x: idkey(x[0])) if ts]
    return items, probs


def lint_table(path):
    """{kernel: {column: float or None}} from kernel_lint's saved stdout (PATH, or the one file in a directory that holds its
    header); raises ValueError on no header, a kernel twice, or a row that does not split into the header's columns"""
    if os.path.isdir(path):
        hits = [os.path.join(path, n) for n in sorted(os.listdir(path)) if os.path.isfile(os.path.join(path, n))
                and open(os.path.join(path, n), errors="replace").readline().split()[:3] == ["kernel", "where", "loop"]]
        if len(hits) != 1:
            raise ValueError(f"{path}: {len(hits)} kernel_lint tables in the directory (need exactly one; pass the file)")
        path = hits[0]
    cols, rows = None, {}
    for line in open(path, errors="replace"):
        w = line.split()
        if w[:3] == ["kernel", "where", "loop"]:
            if cols is not None:
                raise ValueError(f"{path}: two kernel_lint headers (two runs in one file)")
            cols = w[:w.index("flags")] if "flags" in w else w
            continue
        if cols is None or len(w) < len(cols) or w[1] not in ("loop", "kernel"):
            continue
        if w[0] in rows:
            raise ValueError(f"{path}: kernel {w[0]} twice (two builds in one table)")
        vals = {}
        for c, v in zip(cols[2:], w[2:len(cols)]):
            try:
                vals[c] = None if v == "-" else float(v)
            except ValueError:
                vals[c] = v
        rows[w[0]] = vals
    if cols is None:
        raise ValueError(f"{path}: no kernel_lint header ('kernel where loop ...')")
    return rows


def alternatives(pat):
    """[(label, variant)]: for each alternation (top level or inside any group), one variant of PAT per alternative, keeping that
    alternative alone; a variant that fully matches no kernel is a branch the line claims and nothing checks (review C16 F6 h3)"""
    groups, stack, bars, i = [], [(0, [])], [], 0
    while i < len(pat):
        c = pat[i]
        if c == "\\":
            i += 2
            continue
        if c == "[":
            j = i + 1
            if j < len(pat) and pat[j] == "^":
                j += 1
            if j < len(pat) and pat[j] == "]":
                j += 1
            while j < len(pat) and pat[j] != "]":
                j += 2 if pat[j] == "\\" else 1
            i = j + 1
            continue
        if c == "(":
            m = re.match(r"\((?:\?(?::|P<\w+>|<\w+>|[=!]|<[=!]|[aiLmsux-]+:))?", pat[i:])
            stack.append((i + m.end(), []))
            i += m.end()
            continue
        if c == ")" and len(stack) > 1:
            start, bs = stack.pop()
            groups.append((start, i, bs))
        elif c == "|":
            stack[-1][1].append(i)
        i += 1
    groups.append((0, len(pat), stack[0][1]))
    out = []
    for start, end, bs in groups:
        if not bs:
            continue
        cuts = [start - 1] + bs + [end]
        for a, b in zip(cuts, cuts[1:]):
            alt = pat[a + 1:b]
            out.append((alt or "(empty)", pat[:start] + alt + pat[end:]))
    return out


def compiler_check(step, built, base):
    """(lines, problems): each '# compiler:' line against the built and base kernel_lint tables (state table: the C16 hand-off,
    r3 D)"""
    f, tagged = card_tagged(step)
    probs = [m for k, m in tagged if k == "compiler"]
    if not f["compiler"] and not probs:
        return ["no '# compiler:' lines: nothing predicted"], []
    tb, ta = lint_table(built), lint_table(base)
    out = []
    for line, rx, metric, frm, to in f["compiler"]:
        col, band = COMPILER_METRICS[metric], COMPILER_BAND[metric]
        if re.search(r"\\[1-9]|\(\?P=", rx.pattern):
            probs.append(f"{line!r}: a back-reference in the regex (one line per kernel instead)")
            continue
        names = sorted(k for k in set(tb) | set(ta) if rx.fullmatch(k))
        if not names:
            probs.append(f"{line!r}: the regex matches no kernel in either table")
            continue
        for alt, variant in alternatives(rx.pattern):
            if not any(re.fullmatch(variant, k) for k in names):
                probs.append(f"{line!r}: the alternative {alt!r} matches no kernel")
        for g in range(1, rx.groups + 1):
            if all(rx.fullmatch(k).group(g) is None for k in names):
                probs.append(f"{line!r}: group {g} of the regex takes part in no matched kernel (an optional part nothing checks)")
        for k in names:
            if k not in tb or k not in ta:
                probs.append(f"{line!r}: {k} is in the {'base' if k in ta else 'built'} table only")
                continue
            a, b = ta[k].get(col), tb[k].get(col)
            if not isinstance(a, float) or not isinstance(b, float):
                probs.append(f"{line!r}: {k} {col} has no value (base {a!r}, built {b!r})")
                continue
            d = to - frm
            if frm == to:
                moved, rule = abs(a - frm) <= band and abs(b - frm) <= band, f"both within {band:g} of {frm:g}"
            elif metric in COMPILER_EXACT:
                moved, rule = a == frm and b == to, "exact"
            else:
                h = abs(d) / 2
                moved = abs(a - frm) <= band and abs(b - to) <= h and (b - a) * d > 0 and abs(b - a) >= h
                rule = f"base within {band:g} of FROM, built within {h:g} of TO, moved >= {h:g} in its direction"
            out.append(f"  {k} {metric}: base {a:g} built {b:g} (predicted {frm:g} -> {to:g}): {'ok' if moved else 'ABSENT'}")
            if not moved:
                probs.append(f"{line!r}: {k} {metric} read {a:g} -> {b:g}, predicted {frm:g} -> {to:g} ({rule}): go back to "
                             "source, not to a timing")
    return out, probs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ledger", action="append", help="the experiment log (markdown table; default LEDGER.md); repeat to add "
                    "an archive (a log's own `Archive: PATH` line adds it too)")
    ap.add_argument("--id", default=r"[A-Z]+\d+", help="row id regex (default [A-Z]+\\d+)")
    ap.add_argument("--result-from", type=int, default=-2, help="first result / decision cell (default -2: the last two)")
    ap.add_argument("--what-cell", type=int, help="the cell quoted in refusals (default: the cells before the result)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("step")
    p.add_argument("--grandfather-list", dest="grandfather", help="steps written before the check, pinned by sha256")
    p.add_argument("--switch-var", default="EXTRA")
    p.add_argument("--exclude-switches-from", help="a build command file: its NAME=value switches are not re-litigated")
    p = sub.add_parser("probe-card")
    p.add_argument("action", choices=["check"])
    p.add_argument("step")
    p.add_argument("--grandfather-list", dest="grandfather", help="steps written before the check, pinned by sha256")
    p = sub.add_parser("compiler-check")
    p.add_argument("step")
    p.add_argument("built", help="kernel_lint's saved stdout for the built package (or a directory holding exactly one)")
    p.add_argument("base", help="the same for the base package")
    p = sub.add_parser("verdict")
    p.add_argument("step")
    p.add_argument("result", nargs="?")
    sub.add_parser("owed")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "probe-card":
            name = os.path.basename(a.step)
            if grandfathered(a.step, a.grandfather):
                print(f"PROBE: OK ({name}: grandfathered, listed with its sha256 in {a.grandfather})")
                return 0
            f, probs = card(a.step)
            probs += chains(open(a.step).read())
            for x in probs:
                print(f"PROBE MISSING {name}: {x}")
            if probs:
                print("PROBE: MISSING")
                return 1
            m, lo, hi = f["predict"]
            print(f"PROBE: OK ({name}: predict {m} {lo:g}-{hi:g}; kill {f['kill'][:60]!r}; budget {f['budget']:g} min"
                  + ("; result regex" if "result" in f else "; no result regex: judge by hand") + ")")
            return 0
        if a.cmd == "compiler-check":
            try:
                lines, probs = compiler_check(a.step, a.built, a.base)
            except ValueError as e:
                print(f"COMPILER: ERROR {e}")
                return 2
            for x in lines:
                print(x)
            for x in probs:
                print(f"COMPILER ABSENT {os.path.basename(a.step)}: {x}")
            print("COMPILER: " + ("ABSENT" if probs else "OK"))
            return 1 if probs else 0
        if a.cmd == "verdict":
            line, rc, detail = verdict(a.step, a.result)
            print(line)
            for d in detail:
                print(d)
            return rc
        log = Log(a.ledger or ["LEDGER.md"], a.id, a.result_from, a.what_cell)
        if a.cmd == "preflight":
            ex = switches_of_file(a.exclude_switches_from) if a.exclude_switches_from else set()
            probs, notes = preflight(a.step, log, ex, None, a.grandfather, a.switch_var)
            return report(probs, notes)
        items, probs = owed(log)
        for e, ts, what in items:
            print(f"  {e} -> {', '.join(ts)}: {what[:90]}")
        for x in probs:
            print("OWED PROBLEM " + x)
        print(f"OWED: {len(items)}")
        return 2 if probs else (1 if items else 0)
    except (OSError, ValueError) as e:     # ValueError: a malformed grandfather list (fail closed)
        print(f"LEDGER: ERROR {e}")
        return 2


def report(probs, notes, prefix="ledger_tool"):
    for n in notes:
        print(f"{prefix}: {n}")
    for p in probs:
        print("LEDGER PREFLIGHT REFUSED " + p)
    print("LEDGER: " + ("PREFLIGHT_REFUSED" if probs else "PREFLIGHT_OK"))
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
