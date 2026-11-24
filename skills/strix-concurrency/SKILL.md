---
name: strix-concurrency
description: Serve a Loom package to several concurrent clients on AMD Strix Halo and judge it. Load it when two or more agents (or users) share one server, when time-to-first-token spikes or a turn re-prefills its whole history (eviction), when sizing the KV pool or context per session, or when a candidate package needs its concurrent cell checked beside the single-stream bench.
---

# Concurrent serving on Strix Halo

Plain markdown for any coding agent. This skill does not depend on the model: every quantity comes from the package (model.json
preset, the storage program), the kit (`kits/<KIT>/kit.env`, `lanes.json`) or an argument. A tool that cannot find a quantity
prints `unknown` and exits 2. It never falls back to another kit's number. Commands run from the repository root. Speed
rules, measurement and the agent bench are in skills/loom-strix-optimise. Worked examples below cite the Qwen3.8-27B kit's
ledger rows (Exx). They show the method and are not defaults.

## Procedure

1. **Budget first (static, no GPU).** `common/kv_budget.py PKG --kit KIT --streams CTX1,CTX2[,...] [--completion N]`
   works out whether the streams' histories fit the KV pool as loom-serve sizes it. It then prints the largest history
   that 1, 2 or 3 streams can each keep.
   - **SDK 0.1.x:** the pool is the preset's `pool_capacity`.
   - **SDK 0.2 and 0.3, fixed backing:** the pool is `context_capacity`, one context shared by every resident row.
   - **Elastic backing:** the pool comes from physical memory. Pass `--pool` from the server's
     `text_memory.pool.capacity_tokens`.

   Each active request holds its history, plus a page-rounded completion credit (the request's max_tokens), plus draft
   depth + 1 positions. A drafter's K/V uses the target's page ids, so it costs bytes but not positions.
   `--miss-tolerant` adds one continuation miss: a history held twice until the old copy is evicted.
   If the budget does not fit, stop here. Every turn past the limit is a full re-prefill.
2. **The concurrent cell.** Run N agent replays at once against one server, served exactly as production serves.
   - Use `common/agent_bench/replay.py` in own mode.
   - Give each stream its own salt (`--salt`) and `--label`.
   - Use seeded sampling (`--sampling card --seed 1`) or greedy decoding, so the replies can be reproduced.
   - Run the same streams one at a time on fresh servers (the solo arm).
   - Record GPU telemetry (clock, power, temperature, busy %) beside every arm.

   The rows carry `t_send / t_first / t_end` (epoch) and `out_sha`, which is all the audit needs. Rows written before those
   fields existed get a timeline rebuilt from turn walls, and their exactness is unknown.
3. **Interference by phase.** `common/concurrency_audit.py CELL.jsonl --solo SOLO.jsonl [--ref REF_CELL.jsonl --floor F]`
   reports, per stream:
   - decode tok/s split by what the other streams were doing (prefilling / decoding / idle; E49's turn-level method);
   - the share of decode time that actually overlapped another stream's prefill (sliced);
   - TTFT p50 and p95;
   - re-prefill events, each classed as eviction, or as a continuation miss when the previous reply hit its cap.

   Per cell it reports the aggregate decode tok/s, one stream alone against the N-stream total (share), and the cell's
   wall time against the solo runs back to back. For per-token gaps (does the decoding stream freeze during the other's
   prefill?), run `common/decode_while_prefill.py --model ID --prompt N` on a running server (E59's method).
4. **Exactness.** Each stream's replies must equal its solo replies turn by turn. The audit first checks that it catches a
   planted difference (CONTROL FAILED, exit 2, otherwise). In own mode, one divergent turn changes every later history, so
   the first divergent turn is the finding.

## Verdict lines (exit 0 pass, 1 fail, 2 usage / invalid / unknown)

- `KV_BUDGET: FIT | NO_FIT | UNKNOWN` (common/kv_budget.py)
- `CONCURRENCY: OK | FAIL | UNKNOWN` (common/concurrency_audit.py). It fails on any of these:
  - a reply that differs from solo;
  - an eviction re-prefill;
  - failed turns;
  - streams that never overlapped;
  - a first turn that reuses a cached prefix (prompts not salted apart);
  - a paired per-turn decode or prefill ratio against `--ref` whose 95 % CI is below 1 and beyond `--floor`.

  The verdict is UNKNOWN without `--solo` or reply hashes.
- `common/decode_while_prefill.py` prints its phase table (exit 0 measured, 1 a stream failed, 2 usage).

## What counts as a regression

A candidate's concurrent cell that is slower than the reference's cell on the same streams. "Slower" means the paired
per-turn ratio's CI excludes 1 and the drop exceeds the kit's A/A noise floor, which is measured per kit with the agent bench
(Qwen: ~1 % decode, ~0.2 % prefill). An exactness break or a new eviction counts as a regression whatever the speed.
A faster single stream does not excuse a slower concurrent cell. Run the cell at every N you serve: on a MoE kit, extra rows
activate extra experts per step, so per-stream cost grows with N rather than staying flat.

## Levers recorded (worked examples, Qwen3.8-27B kit)

- **Pool size against eviction (E47, E51).** On SDK 0.1.0, two own-mode streams at 140,439 and 122,812 positions
  re-prefilled whole histories (TTFT 1,579 s and 881 s).
  - kv_budget says those two fit the 327,680-position pool with exact continuations (headroom 56,208). They overflow it
    only with a continuation miss held twice, and the old run had cap-induced misses (E48).
  - On SDK 0.2 with fixed backing, the pool is the 262,144 context itself: the same two streams are short by 9,328, and two
    streams can each keep at most 126,968 positions.
  - The levers are 8-bit KV (bytes, not positions), a larger `context_capacity` or elastic backing (positions), and
    replies that do not hit their cap (no misses).
- **Prefill chunk size against interference (E59).** A decoding stream advanced 2 times in 79 s while another stream
  prefilled 40k tokens with 8,192-token chunks, 4 times with 4,096 and 7 times with 2,048. Single-stream TTFT cost +0.4-3.4 %
  at 4,096 and +1.3-6.8 % at 2,048. loom-serve packs mixed epochs, but one epoch is one verify step for the decoding row.
- **Two streams on one chip (E48, E49).** Per stream, decode ran at ~20 tok/s while the other stream decoded and
  9.6-11.4 tok/s while it prefilled, and 60-80 % of decode turns overlapped a prefill. At < 30k, two streams combined
  (~22 tok/s) were below one stream alone (37-43).
- **Consistency (E30).** 4 concurrent short prompts gave outputs identical to running them alone. Checking this on real agent
  turns is the job of step 4.

## Shared power

Strix Halo decode is power-capped (85 W on the reference machine). Concurrent streams split one power and bandwidth budget, so
a cell can lower the clock for every row. Record clock, power and temperature beside every arm, and start each arm from a cool
GPU, as the agent bench does. A share figure measured hot and compared against a solo run measured cold is a thermal
artifact.
