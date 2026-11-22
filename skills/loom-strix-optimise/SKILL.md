---
name: loom-strix-optimise
description: Push a Loom-served LLM on AMD Strix Halo (gfx1151) toward the hardware limits, kernel by kernel, with served evidence. Use for decode or prefill speed work on a kit in this repository: finding where time goes (served per-dispatch events against per-lane floors), diagnosing a kernel, changing the generator, judging a candidate package against the current one.
---

# Loom on Strix Halo: toward the hardware limits

Plain markdown for any coding agent. The rules are in rules.md (read sections 1, 2 and 6 first), the tools by step in tools.md, the device facts in
devices/gfx1151.md, the traps in gotchas.md, what the Loom SDK offers our kernels in loom-0.2-seams.md, and the model's own
knowledge in kits/<kit>/NOTES.md. Commands below run from the repository root; `KIT` is a directory under kits/.
Keep a log: one entry per try (what ran, where the data is, the result, the decision), refuted tries included.

## The loop

0. **Weights and floors first.** `common/kit_check.sh KIT`: local files match pins.json, upstream freshness, and lanes.json
   reproduces from the weights (`common/make_lanes.py`); a stale floor misranks every lane.
1. **Scoreboard from serving, not from a bench.** `common/events_ab.sh KIT OUT LENGTH PKG` serves the package with a
   per-dispatch profile at a long prompt, then `common/events_summary.py kits/KIT/lanes.json RUN` prints every lane
   against its floor (common/ceiling.py). Cite the served prompt size it prints, not the label. Work on the lane with the
   largest (measured - floor) x likelihood.
2. **Distributions, not means.** `--split KERNEL` shows a kernel's time clusters; the top-kernels table splits by
   workgroup count. One kernel's mean hid two populations (aligned and straddling verify tokens) at 1.62 and 3.01 ms.
3. **Classify the bound** (rules section 1): bytes / time against bandwidth, FLOPs / time against the WMMA peak, latency
   (a timing ablation without loads and one with cache-resident loads), or registers (the compile report).
   Before designing, run `common/kernel_lint.py` over the package's compiled kernels: it flags the defect shapes already
   found (a scalarised lookup, byte-vector repacking, a hot-loop spill) and prints instructions per MMA and the registers to
   the next residency tier.
4. **Read the SDK and this repository's generator before designing around a limit** (rules section 6): grep an hrx-system
   checkout at the installed tag (loom/docs/src, loom/py/loom/dialect) and kit/.
5. **Prior art at the diagnosis step** (rules section 3), with the measured diagnosis in the brief; spot-check key claims.
6. **Cheapest evidence first** (rules section 2): the compile report, `common/loop_census.py` (the hot loop from its
   backward branch: length, moves, waits), `common/load_cover.py` (where each load is waited), `common/kernel_lint.py`
   (every root of a package or program; against another build or compiler with `--baseline`), `loom-compile-report show/suggest` on a details report (wait reasons, copy waits, the LDS
   bank model), Radeon GPU Analyzer for live registers; then a timing ablation; then the served binary (`common/serve_code_dump.py`) timed through
   HIP, with rocprofv3 counters or thread trace on it when the question is about caches, LDS or stalls; then a build.
7. **Change test-first.** Kernel cases that straddle every new boundary (`kit/make_kernel_tests.py`), then deliberate
   breaks of each new guard in the built program: a break that survives is a missing case.
8. **Candidate check.** `common/build.sh KIT PKG` (EXTRA="SWITCH=1" for the candidate; the exact command lands in PKG.cmd),
   `common/ktest.sh KIT PKG DIR` (kernel tests stop the chain), `common/eq.sh KIT PKG REF OUT` when the change is meant to
   be exact (texts and draft counters), `common/gates.sh KIT PKG OUT`, then the served ABBA
   (`common/events_ab.sh KIT OUT LENGTH REF CAND CAND REF`) and the agent bench
   (`common/agent_ab.sh KIT REF CAND OUT SESSION`: its paired CI must exclude 1 and its neighbour rule must ACCEPT).
9. **Adopt** only after both pass, with the rollback package named.
10. **New weights** (another quant, a re-upload): `common/quant_eval.sh` against the reference quant, a `variants/<name>/` in the
   kit, `KIT_VARIANT=<name> common/kit_check.sh`, then the same candidate check.

## Running long jobs

One GPU job at a time; a measurement on a quiet machine (`common/wait_quiet.sh`; no build beside it, `common/build.sh`
refuses during one). Write a chain as one script that restores the production server on every exit (common/env.sh's
lsk_restore), start it in the background, and chain the next with `common/wait_pid.sh PID "EXACT ARGS"`. Never find
processes with `pgrep -f`.

## Setup

`common/env.example` -> `~/.config/loom-strix-kits/env` (venv, loom-skills checkout, llama.cpp for the gates, work dir,
IREE_PROFILE, and hooks to stop / restore a server another tool manages). Each kit's `kit.env` holds its recipe, build
and serve flags, reference GGUF and lanes file.

## Using this with an agent

Point the agent at this file (e.g. from your AGENTS.md or system prompt). Agents that load skill folders can link this
directory into their skills path (for Claude Code: `.claude/skills/loom-strix-optimise`).
