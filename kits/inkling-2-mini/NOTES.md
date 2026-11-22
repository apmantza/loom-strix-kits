# Inkling 2 Mini: model knowledge and port scope

Status: scoping (Nov 2026). Nothing built or measured. Sources: the reference implementation (Hugging Face Transformers
`models/inkling2_loop/modeling_inkling2_loop.py`), llama.cpp's merged support (`src/models/inkling2.cpp`, `inkling2-loop.cpp`),
vLLM, the native checkpoint's safetensors header and `hf_quant_config.json`; Loom 0.3.0 docs and ROCm/hrx-loom-library main for
the patterns below. Strata 0.2.19 (an engine with an Inkling backend on gfx1151) is a speed baseline, not a code source.

## The model
- 26.4B parameters, ~3.4B active per token; 24 decoder layers; **layers 6-17 run three times per token** with shared
  weights (48 applications: 0-5, [6-17] x3, 18-23). Every pass restarts from the prelude output; passes 2 and 3 receive
  gated "messages" (attention over the previous pass's block states) before q/k/v and the MLP. KV and conv state are per
  pass: 48 cache slots.
- Attention on every layer: 24 query heads, 8 KV heads, head_dim 128, **no RoPE**. Layers 5, 11, 17, 23 are global; the
  rest slide over 1,024 positions. Per head: q and k L2-clipped (22.6), a learned relative-position bias (16 dims x 1,024
  distances), a per-token attention sink, log-scaled queries on global layers (1 + 0.1 ln(max(pos / 8192, 1))), value
  RMS-norm without gamma, sigmoid output gate.
- Four short causal convolutions per layer and pass (width 4, residual inside): on k and v (dynamic, input-dependent
  weights on global layers), after attention, after the MoE.
- Block attention residuals: the layer input is a learned attention mix over [embedding, completed 6-layer block sums,
  the running partial], with gamma-free normalised keys and values.
- MoE: 128 experts, top-8 by sigmoid score plus bias, weights renormalised x 8; expert FFN 1,152 with softcapped gate (4)
  and up (25). No shared expert. Logit softcap 30; vocabulary 200,058 used of 201,024 rows; untied embedding.
- Weights (native): BF16 except experts, NVFP4 (e2m1, an e4m3 scale per 16 values, a per-expert fp32 factor) in layers 0-19
  and MXFP8 (e4m3, an e8m0 scale per 32) in layers 20-23; gate and up rows interleaved. The MTP head (`mtp.safetensors`,
  BF16) is a DSpark block drafter: 3 dense layers drafting 8-token blocks from four target feature taps plus a Markov head.

- **Vision** (`vision_config`, `inkling2_vision` [src, config.json]): a 23-layer ViT, hidden 1,024, 16 heads, MLP ratio 4, native
  patch 20 with a 32x32 learned position grid, temporal patch 2 (video), a 2x2 merge adapter and a bias-free projector
  (hidden 4,096) into the 2,304-wide text model. Projector GGUFs: Unsloth mmproj BF16 / F16 0.64 GB, F32 1.27 GB; ggml-org
  BF16 0.64 GB, Q8_0 0.35 GB. Vision is a goal for every kit (kit/vision), so the encoder is in scope after plain decode.

## KV cache (22 Nov)
- As released every loop pass keeps its own cache: 48 slots, 8 global (layers 5 and 23, plus 11 and 17 on each of 3 passes: 8 KV
  heads x 128 x K,V x 2 B = 32 KiB per token at f16) and 40 sliding-window slots (1,024 positions, 160 MiB fixed) [comp, config].
  At f16: 1.2 / 4.2 / 8.2 GiB at 32k / 128k / 256k. Sharing the global cache across passes would halve the per-token figure
  (16 KiB), but no variant of sharing is exact: the model was trained with per-pass caches, the blog shows no KV ablation, and
  SGLang / vLLM offer only full sharing and refuse it on this fixed-loop model; llama.cpp and Strata keep the 48 slots.
- At f16 the global slots' reads equal the 5.55 GB weight read near 169k tokens, so long-context decode is KV-bound sooner
  than on Qwen3.8-27B. Plan: per-pass caches (exact), 8-bit KV read in-tile inside every attention variant (relative bias,
  sink, sliding window), as the Qwen kit's KV8.
- llama.cpp: Unsloth's fork's relative-bias attention kernel took only F16/BF16 K/V, so q8_0 KV fell back to a dense path
  with an extra F16 copy (more memory than F16); fixed in their fork (unslothai/llama.cpp PRs 386 / 399: in-tile q8_0 / q4_0),
  not upstream. Reference runs on a build without it use F16 KV.

## Floors (lanes.json from common/make_lanes.py on the native header; 228 GB/s, 45 TFLOP/s)
- Decode reads **5.55 GB per token** (one token) but **15.6 GB per 8-token verify step**: eight tokens touch ~40 % of each
  layer's experts. Speculation buys far less here than on a dense model: a step costs ~2.8x a token.
- Weight floor 24 ms per plain token (~41 tok/s), 68 ms per verify step. BF16 attention and gates are ~2.6 GB of the 5.55.
- KV: 8 global slots read the context, 40 sliding slots read 1,024 positions each. Prefill attention floor at 81k: 8.2 s.
- Strata 0.2.19 reports 47-48 tok/s on Strix Halo (quant and drafter setting not stated).
- lanes.json still carries the Qwen kit's lane regexes (the drafter and GDN lanes have no constants): adapt them to this
  kit's kernel names when the first kernels exist.

## What Loom already has, and what is new (Loom 0.3.0, hrx-loom-library, loom-skills starter)
| Piece | In Loom | New work |
|---|---|---|
| NVFP4 weights | `encoding.define #encoding.f4e2m1<scale_format=f8e4m3fn, scale_group_elements=16>` + `vector.decode` (corpus `numeric/matrix/nvfp4.loom`); e2m1 decode lowers on gfx11 by table lookups | first: compile the scaled schema for gfx1151; expert matvec and grouped GEMM kernels; the per-expert fp32 factor |
| MXFP8 weights | `#encoding.f8e4m3fn<scale_format=e8m0>`; gfx11 decodes fp8 in software, `vector.fragment.load<rhs>` straight from f8 storage (`gfx11/fp8_finite_storage_fragments.loom`) | expert kernels for layers 20-23 |
| MoE routing, grouped GEMMs, decode matvecs | library Flash-Next: `expert_buckets.loom` (expert-major route records, no host readback), `expert_matrix.loom`, `experts.loom`, `routing.loom`; starter `gen_moe.py` | sigmoid + bias + x8 routing; softcapped activation; FP4/FP8 operands |
| Looped layers | `command.parameter "blk.{}..."[%index]` with a computed weight index (docs guide/command-programs.md); no model uses it yet | weight index = f(application), state index = application (48 slots) |
| Sliding + global attention | starter `%sliding` / window in gen_llm and gen_flash2 | relative bias, sink, log scaling, q/k clipping, value norm inside flash decode and prefill (nothing in Loom) |
| Short convolutions | `reference/lfm2moe-example/gen_sconv.py` (history in recurrent slots, speculative replay) | four per layer; dynamic weights on global k/v |
| Block attention residuals | library `motif/tensor/layer_mix.loom` (a 12-layer mix) as a starting point | the query mix over block states; the loop messages |
| MTP / DSpark | library Flash-Next MTP + adaptive DFlash proposal programs (0.3.0) | the DSpark head (3 layers + Markov head) |
| Checkpoint | safetensors read in place by key (loom-kb facet/safetensors) | an hf_names map for Inkling's keys |

## Weights: what the header and the GGUFs show (11-22)
- Names: kit/hf_names/inkling2.json maps the native text keys (model.llm.*, 822) to llama.cpp's inkling2 GGUF names (734),
  checked by kit/test_hf_names_inkling2.py. Two matrices are transposed by the converter (relative-bias table, global layers'
  static conv weights); the loop layers 6-17 add 12 controller matrices each; block-attention-residual queries live outside
  the layers (blk.i takes attn_queries.(i-1); [3, 2304] on layers 5, 11, 17, 23).
- Unsloth UD-Q8_K_XL keeps the native NVFP4 experts of layers 0-19 (llama.cpp has an NVFP4 type), with each expert's amax and
  outer denominator folded into one f32 `.scale`; the MXFP8 experts of layers 20-23 become **BF16** (2x their native bytes).
  So only the native route keeps those experts 8-bit.
- Checked on values (byte ranges of the native file, ~220 KB, against UD-Q8_K_XL; exact everywhere): norms are copied as stored
  (no +1); the GGUF's NVFP4 `.scale` = amax / outer denominator (2688 = 6 x 448); native w13 rows interleave gate and up (even =
  gate) and the GGUF de-interleaves them, for NVFP4 and MXFP8 alike; the GGUF's BF16 MXFP8 experts are a lossless upcast.
- Unsloth UD-Q4_K_XL (17.6 GB) is not a Q4_K file: experts of layers 0-19 keep the **native NVFP4 bytes** (11.47 GB), the
  MXFP8 experts of layers 20-23 become Q8_0 (4.33 GB), attention / gates / other Q8_0 (0.68 GB), embedding and output Q8_0
  (0.98 GB) [comp, tensor table]. UD-Q8_K_XL holds native values everywhere (NVFP4 experts, lossless BF16 upcasts, native BF16),
  so it is an exact local reference: Unsloth's KLD 0.0107 / 98.0 % same top token for UD-Q4_K_XL is effectively against native.
- **Route chosen (22 Nov): UD-Q4_K_XL** (route b): native NVFP4 experts, no MXFP8 software decode (Q8_0 is a served kit format),
  Q8_0 instead of BF16 attention (fewer bytes per token than native); quality checked with common/quant_eval.sh against
  UD-Q8_K_XL. Fallback if that KLD is too high: NVFP4 experts + Q8_0 MXFP8 experts + BF16 attention.
- Prior art on this ROCm: hipBLASLt 10.2 ships hand-written gfx1151 kernels for 4-bit-weight GEMMs with bf16/fp16 activations.
- References: Unsloth's official gfx1151 build of its fork (b12588-mix-c0c3de8, release asset, digest checked).

## Plan, cheapest evidence first
1. Static: compile the NVFP4 and MXFP8 decode schemas for gfx1151 (the one unproven Loom capability). **Done 11-22**: both
   compile on Loom 0.2.0 and 0.3.0 for gfx1151 (one-lane probes: `vector.decode` with `#encoding.f4e2m1<scale_format=f8e4m3fn,
   scale_group_elements=16>` and `#encoding.f8e4m3fn<scale_format=e8m0, scale_group_elements=32>`). NVFP4 to f16: ~5.3 VALU per
   value including the store (e2m1 by `v_perm_b32` table, scale by `v_pk_mul_f16`). MXFP8 to f16 or f32: ~12.7 VALU per value
   (gfx11 has no fp8 convert: software e4m3 decode, `v_bfi_b32` / packed 16-bit arithmetic). For decode matvecs, kit/gen_mxfp4s.py's
   int8 path (e2m1 as doubled integer codes for dot4) is the nearest existing route for NVFP4: it differs in the scale (e4m3 per 16,
   plus the per-expert fp32 factor, against e8m0 per 32).
2. Reference: the Transformers implementation on CPU for per-layer outputs at short prompts; Unsloth's llama.cpp build for
   GGUF references and `common/quant_eval.sh` (UD-Q4_K_XL against UD-Q8_K_XL, which keeps the native experts).
3. Generator (kit/, behind an `inkling2` architecture path): plain decode first, no speculation; then prefill; then DSpark.
   **Started 11-22**: kit/gen_nvfp4.py, the NVFP4 expert codes for the int8 decode path (@g_nvfp4_icodes: e2m1 as doubled
   integer codes by a v_perm table, two e4m3 half scales decoded on integers, the per-expert factor applied per row) and a
   single-token probe matvec: compiles on Loom 0.2.0 and 0.3.0 for gfx1151, 211 instructions, 35 VGPRs, no lint flag
   (kit/test_nvfp4.py).
   **Experts in the shared generator (22 Nov)**: the GGUF route's NVFP4 is ggml's block layout (36 bytes per 64 values, scale bytes
   first, nibble j / j+8 per 16), not the native split planes: format "nvfp4" (kit/formats.py) with @g_nvfp4_icodes
   (gen_nvfp4.gguf_icodes), the SITU activation (moe.act "situ") and the per-expert gate_up scale (gen_moe.expert_mv gu_scale).
   Inkling's fused gate+up (K 2304, 2 x 1152) and down (K 1152) expert kernels compile lint-clean at 16 waves; a GPU value test
   against ggml's decode is written (kit/test_nvfp4_gguf.py --gpu). Semantics from Unsloth's llama.cpp fork (src/models/inkling2.cpp,
   inkling2-loop.cpp, llama-graph.cpp build_moe_ffn).
   **Loop (22 Nov)**: every pass restarts from the prelude output; passes 2-3 mix a message (depth attention over the previous
   pass's block states) into each core layer's q, k, v and FFN inputs through a controller, out = cur - tanh(F z) cur +
   tanh(G z) msg, z = D [rms(cur); rms(msg)] (rank 64, f32 weights, ~226 MB per token). kit/gen_loop.py has the controller kernel
   (weights read once per token, 176 VGPRs, lint-clean; kit/test_loop.py) and the depth-attention mix shared by the block attention
   residuals, the loop messages and the output query (gen_loop.depth_mix: states RMS-normalised, softmax over learned-query scores,
   the normalised states mixed; each state read once, two barriers per call). Still to write: the 48-slot schedule in the program, a
   multi-token controller for verify spans.
   **Controller weights are BF16 in disguise (22 Nov)**: all 28.3M f32 values of the GGUF's 144 loop-controller tensors have zero low
   16 bits (exact BF16 upcasts; the native checkpoint stores them in BF16), so storing them as BF16 halves the ~226 MB per token with
   bit-identical results.
   **From vLLM's Inkling backend (read via github.com/sxuff/inkling-2-mini-spark, 22 Nov)**: conv history keyed by absolute position
   like KV (speculative rollback free); K / V convolutions fused into the attention prologue with the clips and the KV write; decode
   attention with one workgroup per (KV head, all verify tokens x its GQA group); the expert activations are quantised per token as
   the checkpoint's hf_quant_config asks (NVFP4 layers 0-19, MXFP8 20-23), which llama.cpp does not do. Its GB10 step time suggests
   an 8-token verify reads ~1.9x one token's bytes, not the 2.8x estimated above: measure the expert union on real windows.
   **Short convolutions (22 Nov)**: kit/gen_sconv.py, the four per-layer causal convolutions with residual and 1/sqrt(n) scale, the
   global layers' K / V with per-head dynamic taps; one lane per channel, history read and written once, a decode or verify span
   per call. kit/test_sconv.py executes the generated kernel text in float32 and matches ggml's convolve() bit for bit. As
   separate kernels these are ~190 launches per token (4 x 48 layer applications): a fusion candidate (into the K/V projection
   and residual epilogues) once the program exists. Speculative rollback (per-step history) is still to add.
   **Attention inputs (22 Nov)**: relative_extent is 1024 on every layer (global included; distances >= 1024 hit the zero pad row,
   bias = -sink), so the bias is r . rel_bias[min(d, 1024)] - sink, 16 multiply-adds per key from a 64 KB table: computed inside the
   attention kernel, nothing per distance materialised. kit/gen_inkattn.py prep: the per-token clips (q: per-channel scale + L2 clip
   22.63; k: clip; v: RMS norm; r: scale + clip 8) and, on global layers, tau = 1 + 0.1 log(max(1, (pos + 1) / 8192)) on q and r;
   matched to the fork's math (kit/test_inkattn.py). gen_inkattn.decode_swa: the sliding-window decode attention (one workgroup per
   query head: scores with the bias into LDS, softmax with the zero-logit sink, 2 x 128 lanes for the output dims, sigmoid gate;
   f32 K / V for now): 630 instructions, 68 VGPRs, 16 waves, lint-clean; equal to the fork's attention within float32 accumulation
   (window cut at 1,024, sink-dominated heads, GQA) on kit/loomsim.py. gen_inkattn.decode_global: the global layers' split-K form
   (256-key chunks, then a combine with the zero-logit sink; the bias 0 past distance 1024), checked the same way. Next: KV8
   in-tile, verify spans.
   **Strata 0.2.19 (MIT; read 22 Nov, techniques only)**: it materialises the bias as a 24 x 1024 f32 table per token and layer
   visit (ours: 16 multiply-adds per key in the kernel); decode attention splits keys 4 ways, 8 waves per query head, the 3 heads
   of a KV group reading K / V separately; prefill f16 WMMA 64 x 64; KV paged f16 (FP8 only at >= 16 parallel requests), per-pass
   caches; controller weights converted to BF16 at load (half the bytes; changes numerics), q/k/v controllers in one launch; the
   depth mix reads each state twice; the attention-output convolution fused into the output-projection matvec and the FFN one
   into the expert reduce; NVFP4 through LDS byte -> f16-pair tables with f16 dots; DSpark verify size from cost(M) = 1 + 4.2 (1 -
   (15/16)^M). Vendor numbers: 47.3 tok/s (native-weight pack) and 48.0 (smallest pack) with drafting, prompt 1,466 tok/s; the
   near-equal speed of packs 2x apart suggests decode there is not bandwidth-bound.
4. Weights route: UD-Q4_K_XL chosen 22 Nov (see Weights); confirm with our own KLD run against UD-Q8_K_XL (step 2).
