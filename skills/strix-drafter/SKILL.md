---
name: strix-drafter
description: Speculative decoding on a Loom kit on AMD Strix Halo - tokens per verify step, draft depth, the drafter's cost, sampled vs greedy drafts, pricing a drafter change before building it. Use for "tokens per step", "which draft depth", "is the drafter worth it at long context", "would rejection sampling / a sampling drafter help", "fine-tune the drafter". Model-agnostic: block drafters (DFlash-style) and step-per-token drafters (MTP, small draft models), dense or MoE targets.
---

# Drafter: tokens per step, depth and cost

Plain markdown for any coding agent; commands run from the repository root. Companion to loom-strix-optimise (rules.md
section 4, correctness, before any change here). Nothing is model-specific: the drafter's kind and cost, the verify cost by
width and the acceptance come from the kit's drafter file (`kits/KIT/drafter.json`), from served events and from the bench.

## When to load
- Decode speed is judged in tokens per verify step, or a depth / drafter / drafter-quant change is proposed.
- Someone asks whether sampled drafts, rejection sampling or a fine-tuned drafter would raise acceptance.
- Long context: does the drafter's own attention or the wider verify cost eat the gain?

## The model (common/draft_depth.py)
A verify step at depth d checks d drafts plus the target's own row (width d + 1) and commits the accepted prefix plus one:
E[tokens/step] = 1 + sum_k prod_{j<=k} a_j with a_j the conditional acceptance of position j. Step time = verify(d + 1)
+ drafter(d) + fixed. `python3 common/draft_depth.py kits/KIT/drafter.json` prints tokens/step, step ms and tok/s per depth
and `draft_depth: BEST d=N`.
- **Drafter cost.** `kind: block`: one pass proposes the whole block (DFlash2), so the cost is flat in depth up to the block
  (its kernels are width-specialised); `kind: autoregressive`: one drafter step per drafted token (MTP, a small model):
  fixed + d x step. A drafter's per-step context update (warm pass) is `warm_ms`: it exists only when drafting.
- **Verify cost by width.** Measured (`ms_by_width`, interpolated, never extrapolated) or a lane model: weight matvecs flat
  up to their kernel's token tile (on gfx1151 the decode matvecs are 8-token kernels: near flat in width), attention and
  recurrences per row, MoE `moe_union` (extra rows activate extra experts: the expected union 1 - (1 - k/n)^w, as
  make_lanes.py --verify-tokens counts it). Lane scalings are assumptions until served events at other depths measure them.
- **Acceptance.** Best: the accepted-length histogram per step at the served depth (`accept_hist`) or per-position rates.
  What loom-serve exposes is aggregate (/metrics draft_proposed_tokens / draft_accepted_inputs, and the agent bench's
  per-turn draft_acc x draft_proposed): `--replay h1-PKG.jsonl h2-PKG.jsonl --replay-depth D` turns bench rows into served
  tokens/step (out / (out - accepted), agent_ab.sh's figure). One served depth fits a constant a; two or more depths fit a
  decaying a_j = a x r^(j-1). Both are models: on Qwen (E126: d7 4.013, d3 3.023 tokens/step on the same 20 turns) the
  constant fit from d7 predicts d3 at 2.90 (-4 %), so per-position acceptance falls with depth. Measure two depths before
  trusting a depth change from the model.

## Procedure
1. Served tokens per step from the bench (`common/agent_ab.sh` prints it) and the drafter lane from served events
   (`common/events_summary.py`, `lane_breakdown.py`): never from a short prompt (acceptance and the drafter's attention change
   with context and content).
2. Fill `kits/KIT/drafter.json` (`expect` = the served tokens/step), run draft_depth.py, read the best depth AND how flat the
   curve is near it: a difference below the bench's A/A noise floor is not a reason to change depth.
3. Measure what the model assumed: served events at the candidate depth (verify lanes by width), the bench at that depth
   (tokens/step). A depth change goes forward on the seeded bench like any switch.
4. Drafter cost levers (laggard kernels, the proposal head) are judged as any lane: floor, served time, power cap.

## Exactness
The kit's acceptance rule keeps output identical to plain decode with the same seed, whatever the drafter: the target samples
every drafted position with the draw plain decode would make there and accepts the longest prefix where draft == sample
(HRX-like E189). So a drafter change can be gated on identical text and draft counters. Rejection sampling with GREEDY drafts
gives the same acceptance as this rule (q is one-hot: accept with p(x), E391). A SAMPLING drafter (drafts drawn from q) can
accept more (sum_x min(p, q) per position) but is exact only in distribution: outputs then depend on the drafter and every
exact-equality gate is lost. Adopting it is the user's decision, after pricing.

## Pricing a sampling drafter before building it
Loom 0.2.0 exports neither verify-stage logits nor the drafter's candidate scores (E393), so the price is measured outside
the kit first: llama.cpp's HIP build has `--spec-draft-sampling {greedy,probabilistic}` with rejection-sampling verify and
per-request draft_n / draft_n_accepted (llama-server timings). Run the same two GGUFs (target, drafter) greedy vs probabilistic
over the bench session's turns at the card sampling, depth matched to the kit; record each response's draft_n /
draft_n_accepted per turn (draft_depth.py --replay reads them). The ratio of tokens per step prices the change for llama.cpp's DFlash2 (not our kernels): feed both
figures to draft_depth.py as served points. A Loom probe would need a host change to dump the feedback slot per verify epoch.

## Later: on-policy data for a drafter fine-tune
Only after pricing shows the content gap is the drafter's (not context or the verify rule). It needs: the drafter's
tokenizer identical to the target's (compare vocab and merges, not the name); target completions harvested from live
serving under the served sampling (the session turns the bench replays, with consent for the content); the target's
hidden states / taps the drafter conditions on, captured at the served precision; a held-out bench session to show the
tokens per step gain on content the fine-tune did not see; and the same exactness gates afterwards (a greedy drafter keeps
outputs identical, whatever its weights).

## Worked example (Qwen3.8-27B + DFlash2, live depth 7; HRX-like LEDGER rows)
`python3 common/draft_depth.py kits/qwen38-27b/drafter.json` reproduces the agent bench's 3.25 tokens per step at depth 7 (E381;
from the raw bench rows with `--replay`: 3.248) and a 115.1 ms step (the limits table's lanes: matvecs 79.4 ms E301, verify
attention 19.6 E350, drafter 10.49 E367, GDN 3.6 E185, other 2.0), 28.2 tok/s device; best depth 7 under this model.
Published DFlash2 figures are 4.1-5.5 tokens per step on other content (E181, E188); drafter precision does not explain the gap
(E188).
