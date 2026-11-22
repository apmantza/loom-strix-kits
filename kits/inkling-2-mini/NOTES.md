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
- Prior art on this ROCm: hipBLASLt 10.2 ships hand-written gfx1151 kernels for 4-bit-weight GEMMs with bf16/fp16 activations.
- References: Unsloth's official gfx1151 build of its fork (b12588-mix-c0c3de8, release asset, digest checked).

## Plan, cheapest evidence first
1. Static: compile the NVFP4 and MXFP8 decode schemas for gfx1151 (the one unproven Loom capability).
2. Reference: the Transformers implementation on CPU for per-layer outputs at short prompts; Unsloth's llama.cpp build for
   GGUF references and `common/quant_eval.sh` (UD-Q4_K_XL against UD-Q8_K_XL, which keeps the native experts).
3. Generator (kit/, behind an `inkling2` architecture path): plain decode first, no speculation; then prefill; then DSpark.
4. Decide the weights route on numbers: native (exact, 5.55 GB/token) or a GGUF quant (fewer attention bytes; KLD from step 2).
