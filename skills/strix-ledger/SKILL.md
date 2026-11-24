---
name: strix-ledger
description: Keep an experiment log honest - before a step is queued, cite the rows that already judged its lever and declare a prediction, kill condition and time budget; after it ends, judge the result against the prediction; list kept wins still owed to sibling targets. Use for "queue a step", "has this been tried", "what did we predict", "is this result believable", "what is still owed / not ported". Model- and repo-agnostic: any project that logs one row per experiment (id, lever, result, decision) as a markdown table.
---

# Ledger: prior rows, probe cards, verdicts, owed ports

Plain markdown for any coding agent; commands run from the repository root. Tool: `common/ledger_tool.py` (tests:
`common/test_ledger_tool.py`). Companion to loom-strix-optimise (rules.md section 5, measurement). Nothing here is model- or
kit-specific: the log, the step scripts and the result files are the project's.

## When to load
- **Before a step is queued** (a script the queue or window will run): `preflight` and `probe-card check` on it.
- **After it ends**: `verdict` on its result, then the log row (with `owed:` if it is a kept win other targets still need).
- At a regroup or before a release: `owed`, to see which kept wins were never carried over.

## The log (row contract)
One markdown table row per thing tried, losers included: `| ID | ... | result | decision |`. The tool needs only the id at the
start (`--id`, default `[A-Z]+\d+`; use a narrow pattern such as `E\d+` when other tables share the file) and which cells hold the
result and decision (`--result-from K`, default the last two). Losses are recognised by the log's own words in those cells
(refuted, slower, removed, dropped, failed, rejected, lost, no gain, parked, worse, reverted, not worth, ...; negated forms such as
`not slower` are removed first; `tie` and `held` are not losses). Write the loss word when a lever loses; a row that loses in
other words is invisible to the check.

A log kept to active work can move closed rows to an archive file and name it on a line of its own in the log, `Archive:
HISTORY.md` (relative to the log). Every subcommand then reads both, rows ordered by id: a loss moved to the archive still blocks
a repeat, and an owed row is still matched to its ported marks across files. A named archive that cannot be read is an error
(exit 2), never a skip. `--ledger` can also be repeated. Keep ids unique across the two files (the tool searches both copies of a
duplicate, but the project's own check should refuse it).

## The step header (leading comment block)
```
# prior: E175, E305 (searched: tile order, DRAM read bytes)   exactly one; or: # prior: none (searched: term1, term2)
# changed: E175 grouped order now measured on the served shape   answers a row instead of citing it (any number of lines)
# predict: decode ratio 0.97-0.99      metric words, then lo-hi (also lo..hi, lo to hi); lo <= hi
# kill: paired ratio CI includes 1.0   what result ends the lever
# budget: 45 min                       the step's time budget
# result: all ([0-9.]+) \[             optional: regex, one capture group; the last match in the result file is the figure
# result-file: out/summary.txt         optional: where verdict reads it, relative to the step's directory
```
- `preflight STEP` (`LEDGER: PREFLIGHT_OK | PREFLIGHT_REFUSED`): every row that names one of the step's terms and carries a
  loss word must be cited in `prior:` or answered by `changed:`. Terms are the switches the step sets (`NAME=value` inside
  `EXTRA="..."` (`--switch-var`) and `env A=1 cmd` prefixes; a non-literal value is refused) minus those the live build already
  carries (`--exclude-switches-from BUILD.cmd`), plus the searched terms. Cited rows must exist.
- `probe-card check STEP` (`PROBE: OK | MISSING`): predict, kill and budget present and well formed; a result regex must have one
  group. Write the card before the run: a prediction written after the figure is not one.
- `verdict STEP [RESULT]` (`VERDICT: INSIDE | OUTSIDE`): RESULT is a file (the header's regex) or a number; default the
  header's result-file. OUTSIDE prints `debug the measurement before believing it` and exits 1: first check the harness
  reproduces a known figure, the configuration matches what is served, the clocks and power state; only then the lever.
  A regex that matches nothing is an error (exit 2), never a pass.
- `--grandfather N` on preflight and probe-card: steps whose name carries a number <= N pass with a note (written before the
  check was adopted). Exit codes everywhere: 0 ok, 1 refused / missing / outside / owed, 2 usage or unreadable input.

## Owed ports
A kept win on one target is often owed to others: the same kernel change for a sibling quant format, a win on one model to the
next model's kit, a win on one SDK version to the next. Write it in the kept row, and clear it in a later row:
- kept row: `owed: iq4_xs, q6_k` (targets: tokens without spaces, comma-separated, ended by `;`, `|`, a bracket or a sentence end);
- later row: `ported: E388 iq4_xs` (done) or `dropped: E388 q6_k <why>` (decided not needed).
`owed` lists every row with targets no later row cleared (`OWED: N`). Marks above the owed row, or naming a target the row does
not owe, are reported (exit 2). Text in backticks is ignored, so the syntax can be quoted in prose.

Examples from the Qwen3.8-27B log (HRX-like LEDGER rows): a matvec prefetch built for one quant format is owed to the others
(MV_PF on q5_k, E388: `owed: iq4_xs, q6_k`); wins kept on the Qwen kit are owed to the next model's kit
(`owed: inkling`); kept switches built on SDK 0.2.0 are owed to the 0.3.0 port (`owed: sdk-0.3.0`), each cleared by the row that
measured it there.

## Limits
- It matches words, not diagnoses: a row that refuted a lever under another name, or a loss written without a loss word,
  is not found; a cited row is not checked for whether it actually answers the question. The search terms you declare decide
  what it can find: search for the mechanism, not only the switch name.
- A verdict compares one figure with one range: it does not judge noise (use the bench's CI and A/A floor), nor whether the
  metric was the right one. INSIDE means the measurement agrees with the prediction, not that the lever is kept.
- The budget and the kill condition are declared, not enforced: the runner compares elapsed time with the budget, and the
  kill condition is read by whoever writes the row.
