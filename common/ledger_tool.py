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
  probe-card check STEP     PROBE: OK | MISSING                       predict, kill and budget present and well formed
  verdict STEP [RESULT]     VERDICT: INSIDE | OUTSIDE                 RESULT: a file (the header's result regex, last match) or a
                            number; default the header's result-file. OUTSIDE: debug the measurement before believing it.
  owed                      OWED: N                                   kept wins with targets no later row marked ported / dropped
--grandfather N: steps whose name carries a number <= N (first digit run of the basename) pass with a note (written before the check).
usage: ledger_tool.py [--ledger LOG.md [--ledger ARCHIVE.md]] [--id REGEX] [--result-from K] {preflight,probe-card check,verdict,owed} ..."""
import argparse
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


def grandfathered(step, cut):
    m = re.search(r"(\d+)", os.path.basename(step))
    return cut is not None and m is not None and int(m.group(1)) <= cut


def preflight(step, log, exclude=frozenset(), exclude_note=None, grandfather=None, switch_var="EXTRA"):
    """(problems, notes) for one step against the log"""
    name = os.path.basename(step)
    if grandfathered(step, grandfather):
        return [], [f"{name}: grandfathered (number <= {grandfather}, written before this check)"]
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
    if me:
        cited = set(re.findall(idr, me.group(1)))
        s = re.search(r"\(searched:\s*([^)]*)\)", me.group(2))
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
    """(fields, problems) of the probe-card lines: predict (metric, lo, hi), kill, budget minutes, result regex, result file"""
    hd = header(open(step).read())
    f, probs = {}, []
    for key, rx in (("predict", PREDICT), ("kill", KILL), ("budget", BUDGET)):
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
    for key in ("result", "result-file"):
        ls = [l.split(":", 1)[1].strip() for l in hd if re.match(r"^#\s*" + key + r":", l)]
        if len(ls) > 1:
            probs.append(f"{len(ls)} '# {key}:' lines (at most one)")
        elif ls:
            f[key] = ls[0]
    if "result" in f:
        try:
            if re.compile(f["result"]).groups != 1:
                probs.append(f"result regex needs exactly one capture group: {f['result']!r}")
        except re.error as e:
            probs.append(f"result regex does not compile ({e}): {f['result']!r}")
    return f, probs


def verdict(step, result=None):
    """(line, exit code, detail lines)"""
    f, probs = card(step)
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
    p.add_argument("--grandfather", type=int)
    p.add_argument("--switch-var", default="EXTRA")
    p.add_argument("--exclude-switches-from", help="a build command file: its NAME=value switches are not re-litigated")
    p = sub.add_parser("probe-card")
    p.add_argument("action", choices=["check"])
    p.add_argument("step")
    p.add_argument("--grandfather", type=int)
    p = sub.add_parser("verdict")
    p.add_argument("step")
    p.add_argument("result", nargs="?")
    sub.add_parser("owed")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "probe-card":
            name = os.path.basename(a.step)
            if grandfathered(a.step, a.grandfather):
                print(f"PROBE: OK ({name}: grandfathered, number <= {a.grandfather})")
                return 0
            f, probs = card(a.step)
            for x in probs:
                print(f"PROBE MISSING {name}: {x}")
            if probs:
                print("PROBE: MISSING")
                return 1
            m, lo, hi = f["predict"]
            print(f"PROBE: OK ({name}: predict {m} {lo:g}-{hi:g}; kill {f['kill'][:60]!r}; budget {f['budget']:g} min"
                  + ("; result regex" if "result" in f else "; no result regex: judge by hand") + ")")
            return 0
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
    except OSError as e:
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
