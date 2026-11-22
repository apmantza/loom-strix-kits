# strix-halo-engine-lab: Inkling 2 Mini prefill on Strix Halo

What is behind the engine-lab maintainer's post (Inkling 2 Mini on one Strix Halo: 4K 1,658 tok/s, 8K 1,635, 16K 1,645,
32K 1,518, 64K 1,277, "using the official GGUF"), and which of its techniques matter for a Loom port. Read 2026-11-23 from
the public repository (github.com/hyukudan/strix-halo-engine-lab, main at 2026-11-22 ~23:00 UTC). All numbers below are
the maintainer's, measured on their gfx1151 nodes; none were reproduced here.

## Verdict on the claim

- **Prefill, not decode.** The 8K figure matches the measured prefill time after the merged HIP prefill PRs: 8,192 tokens in
  about 5.0 s (PR #990: 4,998 / 5,018 ms) is 1,635 tok/s. Decode on the same engine is 22-26 tok/s (below).
- **Not a GGUF.** The engine loads the original Thinking Machines safetensors checkpoint (revision ed40e7d): NVFP4
  experts (E2M1, E4M3 scale per 16, per-expert FP32 outer scale) in physical layers 0-19, MXFP8 experts (E4M3, E8M0 scale per
  32) in layers 20-23, BF16 everything else. No GGUF reader for Inkling exists in the repository. "Official GGUF" in the post
  most likely means the official checkpoint; unknown whether a separate GGUF build exists.
- **Tier.** The PR evidence is the EXACT tier (BF16 dense, checkpoint formats, logits byte-equal across every change).
  A FAST tier (dense rows and LM head quantised to Q8_0 at load) is in open PR #1009 / #1038 and adds ~1-2 % to prefill.
- **The contributor.** No PR, commit or comment by the GitHub user `italianclownz` exists in the repository (searched
  author, commits, PR text). The only external PR in the last ~1,000 is #1094 by another user (host sampler, landed as #1101),
  which does not touch prefill. The `italianclownz` account's public commits are to a llama.cpp fork (ROCmFPX /
  hasegaw/RDNA-FAST: Vulkan tiled MUL_MAT_ID prefill for IQ3_S experts with per-32 int8 activation quantisation; a
  contiguous KDA conv input for prefill). Which PR the post credits: unknown (possibly landed under the maintainer's
  authorship, or the contributor uses another handle).

## License

Apache-2.0 (`gh api repos/hyukudan/strix-halo-engine-lab/license`). Technique notes are fine; any derived code would need the
Apache notice and attribution. Nothing here is copied code.

## Benchmark setup (as stated in the PRs)

- Binary `strix-inkling2-mini-hip` (HIP, gfx1151, Release, warnings as errors), kernel 7.4, one quiet node, prose prompts,
  one process per run, ABBA or ABAB arms in one turn; prefill time from the engine's `prefill_ms`; tok/s = prompt tokens /
  prefill seconds. Prefill chunk default 8192 rows on HIP (PR #963), 2048 on CUDA.
- Batch: one sequence; the chunk is the ubatch analogue. No server batching in the figures.
- Correctness: every speed PR shows final logits bytes equal to main (EXACT). Model-level: KL vs the CPU reference on a short
  prompt 1.3e-4 (HIP), first ids match (PR #910). The quality gate "gate v2" for the FAST tier: decode KL 0.00012, top-1
  0.996; prefill KL 0.00010, top-1 0.998, dNLL +0.00042 (PR #1009). No perplexity on a corpus seen.

Prefill progression at 8K (ms, from the PR bodies; one line per merged change, each bit-identical):

| step | PR | 8K prefill ms | tok/s |
|---|---|---|---|
| bring-up, chunk 2048 | #910 / #963 | 7,269-7,331 | ~1,125 |
| chunk 8192 | #963 | 6,223-6,269 | ~1,310 |
| attention 64-position tile | #964 | 5,872 | ~1,395 |
| NVFP4 decode via block table | #965 | 5,857-5,876 | ~1,400 |
| (other merged work in between) | | 5,396-5,412 | ~1,515 |
| controller fused into gate-pair GEMM | #975 | 5,285-5,300 | ~1,548 |
| SwiGLU as gate-up epilogue | #978 | 5,150-5,155 | ~1,590 |
| relative bias loaded one tile ahead | #990 | 4,998-5,018 | ~1,635 |
| QK scores kept in registers | #991 | (bit-identical, no 8K delta stated) | |
| gate-up reads token rows through the assignment map | #1002 | (185 ms gather removed per 8K, figure not stated) | |

4K at bring-up: 3.46 s (1,184 tok/s); the claim's 4K 1,658 tok/s is ~2.47 s. 32K was neutral to the chunk change (35.6 s at
chunk 2048 and 8192, ~920 tok/s at PR #963 time); the claim's 1,518 at 32K implies later attention work not identified here.

Decode (one row): 23.6 tok/s at a 7-token prompt, 22.6 at 4K (#910); 24.3-24.4 after #955; 8K EXACT TG 22.2 vs FAST TG 26
(#1009). RTX PRO 6000: 72 tok/s short, 42.6 at 4K.

## Techniques

"New to us" is against what we hold from Strata v0.2.19 (relative-bias table per token and layer; NVFP4 via LDS byte->f16-pair
tables with f16 dots; prefill f16 WMMA 64x64; controllers BF16 at load, q/k/v in one launch; attention-output conv fused into
the output projection; FFN conv into the expert reduce) and vLLM (conv history keyed by position; K/V conv in the attention
prologue; expert activations quantised per token).

| technique | where (PR, file) | new to us? | relevance to a Loom port |
|---|---|---|---|
| Large prefill chunk (8192 rows, was 2048); logits chunk-invariant at 2K/4K/8K chunk | #963, `apps/inkling2_mini_gpu_main.hpp` | partly (a schedule choice, not a kernel) | High and cheap: -10 to -14 % at 4K-16K, neutral at 32K. Check our chunk size and that our outputs are chunk-invariant first. |
| Flash prefill tile of 64 query positions (12 waves, each staged 16-key K/V tile serves 192 rows); decode/short chunks keep 16 | #964, `src/inkling2/inkling2_mini_attention_hip.hpp`, shared `attn_core/dense_flash_prefill.hpp` | Partly (Strata uses 64x64 WMMA; the point here is K/V reuse across 4x more query rows per staged tile) | High: -6 % of prefill. Fits a Loom attention tile parameter. |
| Relative-bias logits for the next key tile prefetched with the K/V tile (bias table is FP32 per row x head x 1024 distances, ~805 MB per layer at 8K) | #990, `attn_core/dense_flash_prefill.hpp` | Yes (the prefetch; the table itself is the Strata form) | High: -2.4 %. Counters: flash prefill 913 ms of 5.28 s, 57 % of wave cycles waiting on vector memory. The table's size is the real cost; a Loom port should weigh computing the bias from the 16-wide relative projection in-kernel against reading a materialised table. |
| QK scores kept in registers with one head-dim slice: K as WMMA operand A, Q as B, half-wave exchange of four registers; one barrier less per key tile | #991, `attn_core/dense_flash_prefill.hpp` | Yes | Medium: a WMMA operand-order trick for gfx1151; applies to any head-128 flash tile we generate. |
| NVFP4 B-tile decode through a per-block table (one build per 16-value block, four quad lookups, two byte permutes per quad) instead of a scalar decode per weight; VALU per GEMM 5.7e8 -> 2.0e8 | #965, `src/core/f16_tile_gemm_core.inc`, `src/core/nvfp4_bf16_table.hpp` | No (same idea as Strata's LDS byte->f16-pair table) | Confirms the approach: -7.4 % of prefill; decode VALU competes with WMMA on RDNA3.5. |
| Exact NVFP4 "exponent shift" decode: E2M1 magnitudes are M x 2^E, so (code x scale) x outer can be produced by adding E to the exponent of a rounded M x scale x outer; byte permute plus sign bit per code; fallback outside the proven range | #1029, `include/strix/weight/formats/microscaled.hpp` (CUDA lowering only so far) | Yes | Medium: cuts decode ALU further (303 -> 190 instructions per k16 step on sm_120). HIP not bound yet. |
| Expert GEMMs in BF16 (NVFP4/MXFP8 decoded to BF16 in the tile), expert-major plan, routed rows gathered; FP32 accumulate | #910, `moe_core/grouped_*`, `src/inkling2/inkling2_mini_moe_hip.hpp` | No (Strata uses f16 dots) | Baseline only. BF16 keeps EXACT vs the reference. |
| Gate-up reads token rows in place through the assignment map (one BF16 image of the tokens instead of top-k gathered copies); removes 3.86 ms per layer, 185 ms per 8K | #1002, `core/f16_tile_gemm` (`MappedRows`), `moe_core/grouped_microscaled` | Yes (as an explicit step; vLLM-style kernels index by sorted ids too) | High for a Loom MoE: no gather pass, A-stager uses a per-thread row pointer that stays fixed over K. |
| Softcapped SwiGLU as the gate-up GEMM epilogue (CTA computes matching gate and up panels, exchanges halves in LDS) | #978, `moe_core/grouped_microscaled` | Partly (common fusion; Inkling-specific caps 4 / 25 and sigmoid on the raw gate) | Medium: -1.8 %. |
| Tall slab expert schedule: 128 routed rows x 128 outputs per CTA, weight slab decoded once per CTA and stage into a shared BF16 slab, 4-slot async ring, one barrier per 32-input stage | #1097, `src/moe_core/grouped_tall_plan.hpp` (CUDA body; HIP not yet) | Yes | High in principle: every 64-row CTA used to re-decode the same expert weights. Worth designing into a Loom expert GEMM (decode once, reuse across M). |
| Low-rank controller fused as the epilogue of the [forget, gain] projection GEMM: forget waves take gain sums via LDS and store `(1 - tanh f) * cur + tanh g * msg` directly; FP32 forget/gain images never hit memory (prefill only, rows > 16) | #975, `src/core/controller_mix.hpp`, `f16_tile_gemm` `kGated` | Yes | High for prefill: at 8K the two FP32 images cost 151 MB stored + 377 MB moved per controller, 96 controllers per prefill; trace 231 -> 123 ms. Direct fit for Loom: one projection kernel with a two-operand epilogue. |
| Decode bank mix in a register form: loads issued before FMA chains, bank pointers hoisted; one block per row | #955, `src/core/residual_mix.hpp` | Partly | Medium for decode: mix 2.0 -> 1.2 ms/token (+1.7 % tok/s). Same latency-chain shape as our E297 controller finding. |
| Residual bank batch: all one-row bank readouts of a step in one launch, grid (1, jobs); controller "tail" on a precomputed message | open #1089 (HIP), #1069 (CUDA) | Yes | Medium for decode: fewer tiny launches. |
| Controllers per path (q, k, v, mlp) run as separate chains: normalize pair (grid rows x 2), rank-64 down projection, gate-pair projection, pointwise controller; controller weights are BF16 in the checkpoint (no conversion), queries FP32. On pass 0 of sliding layers q/k/v share one input norm | `src/inkling2/inkling2_mini_controller_device.hpp`, `inkling2_mini_resident_forward.hpp` | No (Strata batches q/k/v in one launch; engine-lab does not on HIP main) | Strata's form is ahead here. |
| MoE decode GEMV: block scales loaded with the codes, partial last lane iteration packed across rows; cold gate-up 158 -> 122 us, down 84 -> 61 us | #987, `src/moe_core/microscaled_r1.hpp` | Partly | Medium for decode. |
| MoE decode GEMV serving up to 8 rows per launch (verify windows) | #1024 | Partly | Low now (speculation). |
| Route + plan inside the gate-up launch for one decode row | open #1099 (CUDA) | Yes | Medium for decode launch count. |
| Channel-safe row pitch: BF16 row pitch a multiple of 2048 B makes concurrent rows collide on DRAM channels (121-193 GB/s vs 219-230 GB/s at pitch + 256 B) | #1010, #1011, `include/strix/weight/row_pitch.hpp` | Check against our gfx1151 notes | High for any GEMV on this chip: pad weight rows when the pitch is a multiple of 2 KiB (Inkling `wo`: K 3072 x 2 B = 6144 B). |
| Attention-output and MoE-output short convs as separate kernels; the conv's input rounding folded into its reads (opt-in) | #910, #1053, `src/core/short_convolution.hpp` | No (Strata and vLLM fuse further) | Low. |
| Decode attention lanes for multi-request decode: one store, one flash and one combine launch for up to 16 lanes | #1021 | Partly | Low for single-stream. |
| FAST tier: dense projections and LM head as Q8_0 at load, gated by KL / top-1 against EXACT | open #1009 / #1038, #981 | Partly | Decode +17 % (22.2 -> 26 tok/s at 8K), prefill +1-2 %. A quality gate model worth copying for any lossy lever. |

## Open questions

- Which PR from the credited contributor; whether "official GGUF" means anything other than the official safetensors.
- What raised 32K from ~920 tok/s (PR #963 era) to the claimed 1,518 and 64K to 1,277: no merged PR found that states it;
  possibly the relative-bias prefetch and register scores, or unpublished work.
- Where the remaining 5.0 s at 8K goes now (last split stated: flash prefill 913 ms, grouped expert GEMMs ~1.7 s, controllers
  123 ms); not published per kernel after #990.
- Whether the HIP prefill experts will move to the tall slab schedule (#1097 is CUDA-only) or to native low-precision WMMA
  (the bring-up PR lists "native block-scaled FP4 expert GEMM for prefill" as a follow-up).

## Sources

- Repository and license: https://github.com/hyukudan/strix-halo-engine-lab
- Bring-up and model doc: https://github.com/hyukudan/strix-halo-engine-lab/pull/910,
  https://github.com/hyukudan/strix-halo-engine-lab/blob/main/docs/models/inkling2-mini/README.md
- Prefill PRs: https://github.com/hyukudan/strix-halo-engine-lab/pull/963, /pull/964, /pull/965, /pull/975, /pull/978,
  /pull/990, /pull/991, /pull/1002, /pull/1097, /pull/1029
- Decode PRs: /pull/955, /pull/987, /pull/1024, /pull/1021, /pull/1089, /pull/1099
- Row pitch: /pull/1010, /pull/1011
- FAST tier and gate: /pull/1009, /pull/1038
- External PR: /pull/1094 (landed as /pull/1101)
- Contributor's public commits (other project): https://github.com/ROCmFPX/ROCmFPX
