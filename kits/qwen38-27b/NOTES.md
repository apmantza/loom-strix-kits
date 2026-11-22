# Qwen3.8-27B kit: model knowledge for optimisation work

What is specific to this model and recipe. The method is in skills/loom-strix-optimise/, device facts in
skills/loom-strix-optimise/devices/gfx1151.md. E-numbers point into the authors' experiment log (not published).

## Model and serving shape
- 64 layers: 48 Gated DeltaNet (linear attention, recurrent state per row) + 16 full attention (24 query heads, 4 KV heads,
  head_dim 256). FFN 17,408, hidden 5,120, vocabulary 248,320.
- DFlash2 block-diffusion drafter (z-lab GGUF, Q4_K_M), depth 7: each decode step verifies 8 tokens (anchor + 7). At
  temperature 1.0 on agent work it accepts ~3.1-3.4 tokens per step.
- Served shapes at one stream (from served per-dispatch events): every verify step launches 8 token slots (the verify
  attention grid is (8 tokens, 4 KV heads)); prefill runs 4,096-token chunks; Loom's controller ABI 1 already picks the
  prepared shape that matches the live count, so there is no dead-row work to remove at one stream (E213).
- Half of the served verify steps straddle two 16-position KV blocks and cost ~1.8x the aligned ones (E141/E143): benches
  must sample positions across a block's offsets.
- Prefix reuse on a hybrid model needs checkpoints (a recurrent row cannot roll back): `prefix_cache_capacity` 8 (E105/E106).

## Lanes (kits/qwen38-27b/lanes.json) and the kernels behind them
| Lane | Kernels | Notes |
|---|---|---|
| decode: target weight matvecs | g_mv_*, g_head_mv_* (not *_pN) | bandwidth floor; big shapes run 204-216 GB/s |
| decode: verify attention | g_attention_vsplit (+ combine) | issue-bound, not memory (E168); ~19 % of the loop is allocator copies (#3542), ~13 % mask selects that interior tiles do not need (E208) |
| decode: drafter | g_dfl_*, g_head_mv_*_pN, drafter GEMMs | the proposal head reads a 715 MB q4_0 copy of the vocabulary head per step |
| decode: GDN | g_gdn_* | fold pass at bandwidth; recurrence ~2.2x its read floor |
| prefill: GEMMs | g_gemm_* | f16 WMMA on a per-chunk f16 dequant scratch (PREFILL_F16_SCRATCH) |
| prefill: attention | g_attention_flash2 | latency-bound (0.57x without K/V loads, E52) |
| prefill: dequant | g_deq16_* | at the scratch design's floor |

Two head matvecs per verify step have two owners: `g_head_mv_t8_q8` is the target's verify head; `g_head_mv_t8_q8_p40` is the
drafter's proposal head. Attribute a kernel by the program that launches it (events_summary does), not by its name.

## Where the time is (one request, 81,430-token prompt, device time; Nov 2026)
| Part | Measured | Floor |
|---|---|---|
| decode weight matvecs | 80.9 ms/step | 74.3 |
| decode verify attention | 27.3 ms/step | 12.1 (KV8 bytes) |
| decode drafter | 10.5 ms/step | 8.7 |
| decode GDN | 3.6 ms/step | 1.4 |
| prefill GEMMs | 111.5 s | 88.1 |
| prefill attention | 60.1 s | 29.0 |
| prefill dequant | 5.8 s | 5.7 |
| prefill GDN | 9.4 s | ~3.8 |

## Recipe switches
The README's table lists what each switch in kit.env's KIT_RECIPE does. Off-recipe switches under evaluation: `EPI_FUSE=1`
(residual add in the output GEMMs' 256-row epilogue + skipping unread f32 norm outputs; exact; kernel-tested, served gates
pending), `F16O11=1` (the f16 swiglu GEMM writes the row-scaled f16 copy itself).
