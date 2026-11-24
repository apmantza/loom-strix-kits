---
name: strix-critical-path
description: Where served time goes on a Loom kit on AMD Strix Halo, what limits each part at the machine's power setting, and which lever to take next. Use for "where does the time go", "what is the limit", "which lever next", Amdahl screens of candidate levers, whole-request headroom (bandwidth, traffic, compute, dispatch, tokens per step), and regenerating a kit's standing limits table. Model-agnostic: works for any kit (dense, hybrid, MoE).
---

# Critical path: served time, its limits, the next lever

Plain markdown for any coding agent; commands run from the repository root, `KIT` is a directory under kits/. Companion to
loom-strix-optimise (rules.md sections 1, 2 and 5 first). Nothing here is model-specific: shapes, lanes and floors come from
`kits/KIT/lanes.json` (common/make_lanes.py from the weights), the machine's limits from a device file
(`common/devices/<gfx>.json`, one profile per power setting), and measured times from served events.

## When to load
- Before choosing what to work on next, or when two candidate levers compete for one queue slot.
- When a kernel got faster in a harness and the served A/B did not move (is the part on the critical path at all? is the
  machine at its power cap, so the floor you compared against is out of reach?).
- When a limits table (scoreboard of every part vs its floor) must be built or refreshed.

## Procedure
1. **Served per-dispatch accounting.** `common/events_ab.sh KIT OUT LENGTH PKG` serves the package with dispatch events at a
   long prompt; `common/events_summary.py kits/KIT/lanes.json OUT/1-PKG --device common/devices/gfx1151.json` prints every
   lane against its floor at the device's default profile. Lanes are disjoint (a catch-all lane takes the rest), so their sum is
   the device step / prompt. `common/lane_breakdown.py` splits a lane by shape with GB/s. Cite the served prompt size it prints.
2. **Floors at the power the machine runs at.** `common/ceiling.py kits/KIT/lanes.json --device common/devices/gfx1151.json`
   (`--profile nominal` for the uncapped figures, `--clock-ghz X` to ask what a clock would give). The device file derives the
   WMMA peak from the clock (SIMDs x FLOP per WMMA / cycles x clock). Record the power, clock and temperature beside every
   served figure (the A/B's telemetry): a floor at a clock the chip cannot hold under its power cap ranks levers wrongly.
3. **Amdahl screen.** Write each lever as `{name, target, to | reduction | reduction_pct, overlap_group, enabler, src}` in
   a phase `{phase, unit, wall, screen_pct, candidates}` and run `common/amdahl.py FILE`. It prints each lever's share of the
   whole step, which clear the screen, and the combined gain counting the largest saving per overlap group (two rewrites of one
   kernel claim the same time; parts in different lanes add). Set `screen_pct` to your A/B's measured noise floor: a lever below
   it cannot be confirmed by a served A/B, whatever its kernel-level speedup. `enabler: true` keeps a lever with no gain of its
   own that others need. `expect` checks that the input reproduces a table's stated totals.
4. **Whole-request headroom.** Add `cells` (one per served measurement: a decode bench, a prefill profile) and run with
   `--device`: each orthogonal axis becomes a multiplier ceiling at the capped limits: bandwidth (minimum bytes per step x
   steps/s vs capped bandwidth; split into traffic = measured / minimum bytes and kernel bandwidth when measured bytes exist),
   dispatch (1 / busy share), compute for prefill (2 x ACTIVE params x tok/s + attention FLOPs vs capped TFLOP/s: for an MoE
   the routed top-k, make_lanes.py's linear_params), tokens per step vs the drafter's published figure (a ceiling only on
   matching content). `stack` lists candidate multipliers by axis: the best per axis counts, axes multiply, and a candidate
   above its axis's ceiling fails the run (it claims more than the device allows).
5. **Idle share.** Before working on launch gaps, read the profiled run's busy share (`iree-profile explain`: active over
   visible) and the gap between back-to-back dispatches from the events: on a queue that is busy ~99.7 % of the span, host idle
   bounds at ~0.3 %; summed launch gaps inside a step are a separate, measured figure. Neither is worth a lever below the screen.
6. **Pick** the lever with the largest saving x likelihood that clears the screen and has a measured cause (loom-strix-optimise
   rules section 1); a lever whose part is power-bound pays only if it cuts bytes moved or work done.

## Regenerating the standing limits table
1. Serve the live package and its predecessor as an ABBA at the table's prompt size (`common/events_ab.sh KIT OUT LENGTH REF
   LIVE LIVE REF`), with telemetry.
2. `common/events_summary.py kits/KIT/lanes.json OUT/* --device common/devices/<gfx>.json`: one row per lane (measured,
   floor, % of floor). Parts without a ceiling floor keep their documented floor (with its row) or 0 (fixed costs).
3. Write the table's phases as an amdahl input (each part a lever to its floor, its own overlap group; `expect` = the table's
   totals) and run it: `amdahl: PASS` means the parts add up to the totals the table states; a DISCREPANCY names which total
   does not. Keep the input beside the kit (`kits/KIT/limits.json`) so the next refresh diffs against it.
4. Each part's cause cell names the rows that diagnosed it; a part without one says NOT RESEARCHED.

## Worked example (Qwen3.8-27B, gfx1151 at the 85 W cap; HRX-like LEDGER rows)
`python3 common/amdahl.py kits/qwen38-27b/limits.json --device common/devices/gfx1151.json`:
- Device profile cap85w: 2.14 GHz served at 84-85 W (E356, E358), 43.8 TFLOP/s (E369), 240 GB/s cool and hot (E380).
- Decode per verify step at 81,430 positions: 115.1 ms (limits table with HEADPACK's verify attention, E350) -> 90.1 ms at the
  capped floors, matvecs (10.3 ms, E301) and verify attention (8.1 ms) first. E369 stated ~92-95 ms: it kept 228 GB/s for
  attention, drafter and GDN; at 240 GB/s on every DRAM-bound lane the sum is 90.1.
- Prefill at 81,430 tokens: 196.75 s -> 129.4 s (E369: ~197 -> ~128-132), flash2 attention (30.5 s) then GEMMs (21.3 s).
- Headroom: decode bandwidth x1.32 on the seeded agent bench (E381; its turns span 34-108k context, so the 81k bytes approximate),
  tokens per step x1.26 against DFlash2's published low end (E181, other content: E188), dispatch x1.003 (busy 99.7 %, E354);
  prefill compute x1.63.
- Screens: ~1 % decode, ~0.2 % prefill, the seeded agent bench's A/A floor (E374).

## What this does not cover
Concurrent streams (aggregate tok/s rises, per-stream falls; lanes scale differently): they need their own served events.
Energy per token at the cap (energy per WMMA and per byte) is not modelled; under a power cap a lever that only cuts latency or
issue density may not pay (HRX-like E356): judge every lever by the seeded wall-clock bench.
