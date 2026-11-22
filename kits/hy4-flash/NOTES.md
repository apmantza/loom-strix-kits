# Hy4 Flash: model knowledge and port scope

Status: scoping (Nov 2026). Nothing built, downloaded beyond metadata, or measured here. Numbers are tagged **[src]** (from
a source as quoted), **[comp]** (computed from the config and the 148 safetensors shard headers, read by HTTP range
requests) or **[meas]** (measured by someone else, named). Sources: the model card https://huggingface.co/tencent/Hy4-flash
(config.json, safetensors headers), llama.cpp PR https://github.com/ggml-org/llama.cpp/pull/32517 (`src/models/hy-v4-flash.cpp`),
AngelSlim's GGUFs https://huggingface.co/AngelSlim/Hy4-flash-GGUF, and Strata's pack
https://huggingface.co/strata-engine/Hy4-Flash-Strata (a speed baseline, not a code source).

## The model
- Tencent's Hy team (the Hunyuan family, renamed "Hy"); released 16-17 Nov 2026; Tencent Hy License (free internally; a
  separate license above US$50M revenue or 100M monthly users). Also `Hy4-flash-FP8` and `Hy4-flash-base`.
- **220B parameters: 155.5B backbone + 63.5B Engram tables + 0.43B vision + 0.85B DFly2 drafter; 11.3B active per token**
  [src; 11.27B from the headers, comp]. 42 layers, hidden 3,072, vocabulary 120,832; layer 0 a dense FFN (13,824), layers
  1-41 MoE. Context 1,048,576; logits in fp32.
- **Residual: identity hyper-connections, 4 streams** (`hc_mult` 4): a pre-mix per sublayer (`hc_attn_layer` /
  `hc_mlp_layer`) and `hc_head` collapsing the streams before the final norm.
- **Attention: MLA on every layer**, two kinds. 12 sparse layers (0, 1, 5, 9, ..., 41): 32 heads, DeepSeek-style sparse
  attention with a compressed indexer (16 heads of 128, keys pooled over groups of 4 tokens with gated pooling and an
  intra-group APE, group top-512 then up to 2,048 tokens), RoPE base 1e7. 30 sliding-window layers: 64 heads, window 512,
  RoPE base 1e4. Both: q_lora 1,024, kv_lora 512, qk head 256 (192 nope + 64 rope), v head 256, qk-norm, interleaved
  M-RoPE (sections 12/10/10), a learnable per-head sink, an elementwise sigmoid output gate, a sandwich norm.
  Attention is 4.99B of the 11.27B active [comp].
- **MoE ("LatentMoE")**: 512 routed experts, top-16, sigmoid scores + selection bias, renormalised, scaling 3.998, one
  shared SwiGLU expert (1,536). Routed experts run in a 1,536-wide latent (a down projection 3,072 -> 1,536 in, an up
  projection out, then `combine_output_norm`); SwiGLU clamp 10 on routed experts; 7.08M parameters per expert.
- **Engram**: hashed bigram + trigram embeddings on layers 1, 11, 21, 31 (8 hash heads each, ~10.3M buckets per head,
  96 dims): 63.4B parameters, **118 GiB in bf16, 59 GiB at 8 bits** [comp]. Ids depend only on token ids (llama.cpp hashes
  on the host before the first gather); per token 64 rows of 96 values (~6 KB at 8 bits).
- **DFly2 drafter** (0.85B, 5 layers): each draft layer reuses the KV and the indexer's selected positions of target sparse
  layers 25, 29, 33, 37, 41, plus a hidden correction, an MTP fusion and a confidence head; 5 speculative tokens.
- **Vision**: SigLIP2-So400m-derived (27 layers, hidden 1,152, patch 16, 2x2 merge), an RMSNorm-MLP projector to 3,072.

## Weights and quants
- Native BF16, 148 shards, 440.5 GB; an official FP8 variant.
- AngelSlim (Tencent) GGUFs, arch `hy_v4_flash`, the Engram table in its own shard (63.4 GB): Q4_K_M 156.8 GiB (experts
  q4_k / q6_k, **non-expert tensors q8_0**), Q3_K 144.7 GiB, NVFP4 "SPARK" ~150 GiB, IQ2-AH 125.7 GiB (non-expert bf16).
  No Unsloth or bartowski quants yet (11-22). Other GGUFs on Hugging Face by 22 Nov (sizes as listed, GiB; whether the
  Engram table is included differs by repo): Xylolabs (imatrix, IQ2_XXS 77.4 ... Q8_0 213.1, incl. NVFP4 144.3 and Q4_K_M
  150.1), nace-ai (IQ3_XXS 81.8, Q3_K_M 100.1, IQ4_XS 114.2, Q4_K_M 118.5), AtomicChat (IQ2_S/IQ2_XXS 78.9), zephel01
  (86.5, sized for 32 GB VRAM + 96 GB RAM), Baekpica (Q5/Q8 mixes with the Engram table kept in FP8 on SSD, 96.8-98.9, with
  DFly2 drafter GGUFs). Plus EXL3, MLX, NVFP4 / W4A16 / INT8 safetensors variants.
- llama.cpp: PR #32517 open, not merged (new KV cache and memory classes for the pooled index); main already has the
  neighbours (hy-v4: MLA + indexer + HC; deepseek41: Engram + compressed indexer + HC). Transformers, vLLM and SGLang
  support: open PRs plus vendor images serving FP8 at tensor parallel 8.

## Floors (228 GB/s, 45 TFLOP/s) [comp]
Decode bytes per token (weights read once; Engram rows and KV below 0.3 GB up to 256k):

| recipe | GB/token | floor | resident backbone (no Engram, vision, drafter) |
|---|---:|---:|---:|
| AngelSlim Q4_K_M as published | 9.84 | 43 ms, 23 tok/s | 90.3 GiB |
| AngelSlim Q3_K as published | 9.43 | 41 ms, 24 tok/s | 78.2 GiB |
| experts q4_k, everything else q5_k | 7.22 | 32 ms, 32 tok/s | 82.4 GiB |
| experts iq4_xs, everything else q4_k | 6.29 | 28 ms, 36 tok/s | 77.4 GiB |

- **The non-expert weights dominate decode**: 6.6B of the 11.3B active parameters are read every token, and the published
  GGUFs keep them at q8_0 (7.0 of 9.8 GB). Requantising them is the largest decode lever; expert width matters less than on
  Qwen-style MoEs.
- KV is small: the MLA latent is 576 values per token per layer (12 sparse layers ~13.8 KB/token, plus pooled index keys;
  sliding layers hold a fixed 512-token window). AngelSlim reports 5.7 GiB for 256k tokens [src].
- A 6-token verify touches ~89 distinct experts per layer under uniform routing (5.6x one token's expert bytes), while the
  non-expert bytes are shared: a verify step costs ~2.2-2.6x a token, so speculation pays more than on Inkling (2.8x).
- Prefill: 22.5 GFLOP/token, a ~2,000 tok/s compute ceiling.
- Others on Strix Halo: Strata reports 51-60 tok/s decode and 728 tok/s prompt with DFly2 on [meas, Strata]. Mac M5 Max,
  llama.cpp Q4_K_M: 48 tok/s [meas, AngelSlim].

## Fit on this machine
- ~110 GB GPU-visible: experts q4_k + rest q5_k needs ~94 GiB with vision, drafter, 256k KV and workspace [comp]. It fits,
  but **not beside another served model**. The iq4_xs / Q3_K-class recipes (~77-78 GiB) leave room for long context.
- The Engram table cannot be resident: it is read on demand from NVMe, as every other engine does. Lookups are known from
  the token ids before the step (64 random rows per token, ~262k rows for a 4k prefill chunk).

## What exists, and what is new
| Piece | In the kit or Loom | New work |
|---|---|---|
| Sigmoid + bias routing, renormalisation, scaling, shared expert | kit/gen_moe.py | 512 experts at top-16 (the workgroup top-k at that size) |
| SwiGLU clamp | kit/gen_moe.py (`swiglu_oai`) | the plain-silu clamp variant |
| LatentMoE | nothing | latent projections around the experts; expert GEMMs at K = 1,536 |
| Hyper-connections | hrx-loom-library Qwen3.8-Flash-Next (`hyperconnection*.loom`) | port or reuse; not in the kit |
| Hashed n-gram tables streamed from disk | library Flash-Next PLE (hash, lookup, row cache fed from the checkpoint file) | Hy4's hash, four Engram layers, gating into the HC streams |
| Indexer + top-k sparse attention | library Flash-Next `qsa_indexer`, `topk.loom` | gated 4-token pooling with APE, group top-512 -> 2,048, over an MLA latent cache |
| MLA attention | nothing in the kit or the library | absorbed decode over a 576-wide latent (32 and 64 heads); decompression for prefill (head 256) |
| Sliding window | kit/gen_flash2.py | window 512 on MLA |
| Sink + elementwise output gate together | each alone in kit/gen_llm.py (which refuses the pair) | lift the restriction; the gate is per element |
| Interleaved M-RoPE | kit/rope_qg.py | on MLA's 64-dim rope slice |
| DFly2 | DFlash / MTP drafters in kit/draft | KV and index sharing with target layers, confidence head |
| Vision | kit/vision | SigLIP2-derived tower + Hy projector |
| Names | kit/hf_names pattern | a hy_v4_flash map from the shard headers |

No missing Loom capability was found outright. The risk is shapes Loom has no example of: MLA latent attention, 512-expert
routing at top-16, and on-demand file reads inside the serving loop on this ROCm.

## Compared with Inkling 2 Mini (kits/inkling-2-mini/NOTES.md)
| | Inkling 2 Mini | Hy4 Flash |
|---|---|---|
| total / active | 26.4B / 3.4B | 220B (155.5B backbone + 63.5B Engram) / 11.3B |
| decode bytes, plain floor | 5.55 GB, ~41 tok/s (native) | 6.3-9.8 GB, 23-36 tok/s by recipe |
| memory | small | 77-94 GiB + Engram on NVMe; needs the whole GPU |
| new subsystems | looped layers, attention variants, short convolutions, block residuals, NVFP4 / MXFP8 | MLA, compressed-indexer sparse attention, Engram streaming, 4-stream HC, LatentMoE, sink + gate |
| others on Strix Halo | Strata 47-48 tok/s | Strata 51-60 tok/s (DFly2 on) |

## Verdict
Worth a kit, as the second large port after Inkling's plain decode: the stronger model, with decode floors in the same
band, but five new subsystems, a memory footprint that needs the whole GPU, and references that are not merged yet. Strata
already reaches 51-60 tok/s with DFly2, so a competitive kit needs the drafter.

## From a DGX Spark recipe (github.com/sxuff/Hy4-Flash-Spark-recipe, MIT; read 22 Nov)
- llama.cpp + AngelSlim's patch with the SPARK-4Bit-Fast GGUF (NVFP4 experts: Blackwell FP4, not usable on gfx1151; non-expert Q8_0;
  Engram Q8_0 62.7 GiB on the SSD via mmap; research/nvme-streaming.md). ~87 GiB on the GPU, 256K context, f16 KV, MLA as absorbed
  MQA over the 576-wide latent even in prefill; the sparse layers run a dense masked pass (prefill 765 tok/s at 8K -> 542 at 200K)
  [theirs, GB10]. Decode 40.6 tok/s at temperature 0 with MTP depth 4 on short prompts [theirs].
- **Drafter**: these GGUFs carry the 1-layer MTP block, not DFly2: MTP (depth 3-4: +17 % over depth 1 at T 0 [theirs]) is the first
  drafter to plan for. The non-expert weights stayed Q8_0 there too, so the largest decode lever (above) is untested by anyone.

## Plan, cheapest evidence first
1. Static, no weights: a hy_v4_flash name map from the shard headers and the PR's converter; one MLA absorbed-decode probe
   (576-wide latent, 64 heads, sink + elementwise gate) compiled for gfx1151, to see its register and LDS cost.
2. Read the library's Flash-Next hyper-connection, hashed-table and indexer programs against Hy4's definitions; decide
   reuse versus kit generators before writing a kernel.
3. Reference numbers on the machine (a ~145 GB download and a GPU window): the PR built for gfx1151 with AngelSlim Q3_K,
   decode / prefill speed and logits, then the non-expert tensors requantised to q4_k / q5_k and their KLD measured.
