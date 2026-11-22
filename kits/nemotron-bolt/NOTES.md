# Nemotron 3.5 Bolt 74B-A9B: model knowledge and port scope

Status: scoping (22 Nov 2026). Nothing built or measured here yet; the shipping GGUF, its vision projector and the LiLiCorr drafter are being
downloaded for a llama.cpp reference run (plan step 2). Numbers are tagged **[src]** (from
a source as quoted), **[comp]** (computed here from config.json, the 63 BF16 safetensors shard headers and the GGUF tensor
tables, all read by HTTP range requests) or **[meas]** (measured by someone
else, named).

Name check: "Nvidia Bolt 74B MoE" is unambiguous. It is **NVIDIA Nemotron 3.5 Bolt 74B-A9B** (released on Hugging Face
15 Oct 2026 [src]). The only other "Bolt" is **Nemotron 3.5 Bolt E4B** (a post-trained Gemma 4 E4B, `model_type: gemma4`
[src, its config.json]), a different architecture and not scoped here.

Sources:
- BF16 card + config + headers: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16
- the shipping 2-bit model: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF (and the safetensors twin
  `...-24GB`); drafter: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr(-GGUF)
- NVIDIA blog: https://research.nvidia.com/labs/nemotron/Nemotron-3.5-Bolt-24GB/ ; technical report (not read):
  https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-Technical-Report.pdf
- Unsloth GGUFs: https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF
- llama.cpp main (2b66bfbf7, 10 Nov 2026): `src/models/nemotron-h.cpp`, `mamba-base.cpp`,
  `dflash.cpp`; XDA test https://www.xda-developers.com/i-tested-nvidias-nemotron-35-bolt-and-it-changed-what-i-thought-my-16gb-gpu-could-do/

## The model
- NVIDIA; compressed from Nemotron 3.5 Super VL 120B-A12B with Star Elastic (router-guided architecture search: per-layer
  expert counts and widths, top-k fixed at 9, down from 22) and 120B tokens of distillation [src, card]. Vision-language,
  reasoning on/off, tool calling. License **OpenMDW-1.1** (commercial use allowed) [src].
- **74.9B parameters, 9.06B active per token** [comp; "74B (9B active)" src]. Split [comp]: routed experts 66.14B, Mamba-2
  mixers 4.39B, shared experts 1.76B, vision + projector 0.84B, embedding 0.54B, lm_head 0.54B (untied), latent projections
  0.34B, attention 0.29B, routers 0.06B. Of the 9.06B active, **only 1.69B are routed experts; 7.36B are read every token**
  (Mamba alone 4.39B).
- 88 blocks, hidden 4,096, vocabulary 131,072, context 262,144 [src, config]. Each block is one pre-norm (RMSNorm) mixer on
  a plain residual: **40 Mamba-2, 40 MoE, 8 attention** (attention at blocks 7, 16, 25, 36, 47, 58, 69, 78 [comp from
  `layers_block_type`]). Pattern: mamba / moe alternating, one attention roughly every 11 blocks.
- **Mamba-2** (40 layers): 128 heads x head dim 64 (d_inner 8,192), state 128, 8 B/C groups, conv1d width 4 over x,B,C
  (10,240 channels, with bias), dt bias, A_log, D skip, SiLU-gated **grouped RMSNorm (8 groups of 1,024)** before out_proj
  [src, config + llama.cpp mamba-base.cpp]. in_proj 4,096 -> 18,560 (z 8,192 + xBC 10,240 + dt 128). Chunk size 128 for the
  SSD prefill. State 128 x 64 x 128 per layer; config `mamba_ssm_cache_dtype` float32, NVIDIA's memory budget assumes
  16-bit SSM state [src].
- **Attention** (8 layers): GQA 32 query heads / 2 KV heads, head 128, **no RoPE, no qk-norm, no bias, no sliding window**
  [src, config + nemotron-h.cpp]. KV is tiny: 8 KiB per token at f16 [comp: 8 layers x 2 heads x (K,V) x 128 x 2 B].
- **LatentMoE** (40 layers): router on the 4,096 hidden, **sigmoid + e_score_correction_bias, top-9, renormalised, scaling
  5.0**, one group [src]. Routed experts run in a **1,024-wide latent** (`fc1_latent_proj` 4,096 -> 1,024 before, `fc2_latent_proj`
  1,024 -> 4,096 after); experts are **non-gated relu^2** (up 1,024 -> I, relu squared, down I -> 1,024). One shared relu^2
  expert (5,376) on the full 4,096 hidden. **Heterogeneous per layer**: 240 / 304 / 368 / 432 / 496 experts and widths
  1,536 / 2,048 / 2,304 / 2,560 (12 distinct (E, I) pairs over 40 layers; 13,824 experts in all) [src, config].
- **No MTP heads** (`num_nextn_predict_layers` 0). Speculation is an external drafter, **LiLiCorr** (1.11B, 5 dense layers,
  GGUF arch `dflash`): a DFlash2-style block drafter (block 8, non-causal in-block attention, sliding window 1,024, attention
  sink bias, 2-tap grouped conv modulation, context from target layers 2/19/36/52/69/86) with a 2-layer LiLiCorr head
  re-scoring the top-8 target-head candidates jointly over the block instead of DFlash2's rank-256 selector; NVFP4 weights
  [src, drafter config + GGUF header]. Acceptance length at draft 7: 4.30 average, coding 5.26, writing 3.41 (SPEED-Bench,
  T 1.0) [src, drafter card].
- **Vision**: C-RADIO v4-H (ViT-H/16, 32 blocks, hidden 1,280, 16 heads, CPE up to 2,048 px, 10 cls/register tokens), 2x2
  pixel-shuffle, MLP projector 5,120 -> 20,480 -> 4,096 [src, config + headers]. mmproj `clip.projector_type = nemotron_omni`
  [src, mmproj header]. Video input with temporal patch 2.

## Weights and quants
- Native BF16, 63 shards, 149.8 GB [comp, shard sizes]. The BF16 release is "for customization and post-training" [src].
- **The shipping model is the 2-bit one**, trained for it by quantisation-aware distillation (so its weights differ from the
  BF16 release): **routed and shared experts IQ2_XXS (2.0625 bpw), Mamba in/out projections NVFP4 (e4m3 per 16 + a per-tensor
  f32 scale), attention / latent projections / embedding / lm_head Q8_0, routers F32** [src, card; confirmed on the tensor
  table, comp]. GGUF 22.02 GB (20.50 GiB, arch `nemotron_h_moe`) + mmproj 1.73 GB; the same packing as safetensors
  (`...-24GB`, 24.1 GB, ggml-packed IQ2_XXS inside safetensors) for vLLM [comp].
- Unsloth from the BF16 release (imatrix): IQ1_S 18.5 GB ... UD-Q4_K_XL 50.2 GB, Q8_0 78.9 GB, BF16 [comp, file sizes].
  UD-Q4_K_XL: experts mostly Q5_K (up) and IQ4_XS / Q5_K (down), shared experts Q5_K-Q8_0, Mamba Q6_K / Q8_0, attention and
  latent Q8_0, router F32 [comp, tensor tables]. Its IQ2_XXS (22.02 GB) is byte-size-identical to NVIDIA's and likely a re-host
  (unchecked). Many community MLX / EXL3 / NVFP4-FP8 variants.
- The benchmark table puts the 24GB IQ2 a few points under BF16 (e.g. Terminal-Bench 2.1 50.98 vs 53.93, GPQA 86.43 vs
  88.83) [src, card]: the QAD 2-bit model is the intended product, not a degraded fallback.

## Engine support
- **llama.cpp: merged** (arch `nemotron_h_moe`, `src/models/nemotron-h.cpp`: per-layer expert count and width, latent
  projections, relu^2, NVFP4 with `.scale` tensors; Mamba-2 in `mamba-base.cpp` storing per-token states for speculative
  rollback; DFlash + LiLiCorr in `dflash.cpp` / `common/speculative.cpp`) [src, local checkout 11-10]. XDA: "build b11927,
  which added Bolt support" [src]. HIP build on gfx1151 for IQ2_XXS + NVFP4 + ssm_scan: expected, **unknown** (not built).
- vLLM: the official runtime (`--mamba-backend flashinfer`, LiLiCorr with 7 draft tokens) [src]. SGLang / TensorRT-LLM:
  unknown (not found).
- Strata: no Bolt pack on HF (strata-engine has 2 repos) [comp, HF API 11-22]. strix-halo-engine-lab supports only Bolt E4B
  (the Gemma 4 one) [src, docs/models/nemotron-3.5-bolt-e4b]. **No Strix Halo measurement of the 74B found.**
- Other hardware: RTX Spark, llama.cpp, IQ2_XXS + LiLiCorr: 83.99 tok/s on SPEED-Bench-Coding [meas, NVIDIA blog]. RTX 4070
  Ti Super 16 GB with expert offload, no drafter, 64k: 22-32 tok/s decode, 110 tok/s prompt [meas, XDA]. RTX 5080 16 GB reader
  report: tg128 80 tok/s, pp512 979 tok/s [meas, an XDA reader comment, flags unknown; weak].

## Floors (228 GB/s, 45 TFLOP/s) [comp]
Decode bytes per token = every non-routed-expert weight (embedding: one row) + 9 routed experts' up+down per MoE layer at the
layer's real width + KV read at the context + Mamba state read and written (f32: 2 x 172.7 MB; bf16: 2 x 88.8 MB).

| recipe | non-expert GB/tok | experts GB/tok | GB/token at 32k (f32 state) | floor 4k / 32k / 128k | resident weights |
|---|---:|---:|---:|---:|---:|
| NVIDIA 24GB as published (QAD) | 4.39 | 0.44 | 5.44 | 43.8 / 41.9 / 36.5 tok/s | 20.5 GiB |
| same, Q8_0 tensors -> Q4_K, router F16 | 3.70 | 0.44 | 4.75 | 50.5 / 48.0 / 41.1 tok/s | 19.6 GiB |
| Unsloth UD-Q4_K_XL as published | 7.00 | 1.11 | 8.72 | 26.9 / 26.1 / 23.9 tok/s | 46.8 GiB |
| BF16-derived: experts Q4_K, rest Q5_K, router F16 | 5.15 | 0.95 | 6.71 | 35.2 / 34.0 / 30.3 tok/s | 39.8 GiB |

- **The non-expert weights dominate decode even more than on Hy4**: in the shipping recipe the routed experts are 0.44 of
  4.83 GB of weights per token. Mamba in/out (NVFP4) are 2.47 GB, lm_head 0.57, shared experts 0.45, latent projections 0.36,
  attention 0.30, and the **F32 routers 0.23 GB** (a free 0.11 GB at f16) [comp]. The 2-bit experts already did their job.
- Mamba state traffic is a real lane: 0.35 GB per token at f32 (7 % of the shipping recipe's bytes), half at bf16. The KV
  read is small (0.27 GB at 32k, 1.07 GB at 128k). llama.cpp also writes one state snapshot per verified token for rollback:
  for an 8-token verify that is ~1.4 GB at f32 [comp], unless a kernel recomputes from the accepted prefix instead.
- Speculation pays well here: an 8-token verify (anchor + 7 drafts) touches E(1-(1-9/E)^8) experts per layer under uniform
  routing, **1.57x one token's weight bytes** on the shipping recipe (Hy4: 2.2-2.6x, Inkling: 2.8x), because the shared
  non-expert bytes dominate. With LiLiCorr's 4.3 mean acceptance the weight floor is ~4.3 / (1.57 x 22.8 ms + drafter) ->
  on the order of 100 tok/s at short context before state snapshots and drafter cost [comp, rough].
- Prefill: 2 x 9.06 GFLOP + attention (0.27 at 4k avg, 2.15 at 32k, 8.6 at 128k) + SSD (~0.26) = 18.6 / 20.5 / 27.0 GFLOP
  per token -> a 2,400 / 2,200 / 1,700 tok/s compute ceiling at 45 TFLOP/s [comp]. IQ2_XXS and NVFP4 decode cost in the
  prefill GEMMs is not counted.

## Fit on this machine
- Shipping recipe: 20.5 GiB weights + mmproj 1.6 GiB + LiLiCorr 0.9 GiB + KV 2.0 GiB at 256k (0.25 GiB at 32k, 1.0 at 128k)
  + SSM state 0.16 GiB (+1.3 GiB of verify snapshots at f32) = **~26.5 GiB at the full 256k** [comp]. It **fits beside the live
  Qwen model**, unlike Hy4. UD-Q4_K_XL ~53 GiB, Q8_0 ~80 GiB (alone). BF16 (139.5 GiB) does not fit.

## What exists, and what is new
| Piece | In the kit or Loom | New work |
|---|---|---|
| Sigmoid + bias routing, renormalise, scale, shared expert | kit/gen_moe.py (gating 2) | top-9 over 240-496 experts; **per-layer expert count and width** (12 shapes) |
| relu^2 non-gated experts | nothing (gen_moe: SwiGLU / swiglu_oai; gen_llm: SwiGLU / GEGLU) | up -> relu^2 -> down; shared expert also relu^2 |
| LatentMoE | nothing (also new for Hy4) | latent down/up around routed experts; expert GEMMs at K = 1,024; shared expert on the full hidden |
| **Mamba-2 (SSD)** | kit/gen_gdn.py (recurrent slots, conv history, origins/fork for speculation), gen_gdn_chunk.py (gfx12 chunked WMMA) | the scalar-decay SSD recurrence (state 64 x 128 per head, B/C shared by 16 heads per group), dt softplus + bias, D skip, grouped gated RMSNorm; a chunked SSD prefill for gfx1151 |
| Conv1d width 4 + SiLU | kit/gen_gdn.py (qkv conv) | over xBC (10,240 channels) with bias |
| GQA attention without RoPE | kit/gen_llm.py (`nope_layers`), gen_flash2.py | none expected (8 layers, all NoPE) |
| **IQ2_XXS weights** | kit/formats.py has q3_k / iq3_s (grid codebook) / iq4_xs, no 2-bit type | IQ2_XXS decode (256-entry 8-value grid + 7-bit sign index, 66 B per 256) for expert matvecs and grouped GEMMs |
| NVFP4 weights (Mamba projections) | kit/gen_nvfp4.py (e4m3 per 16 + per-tensor/expert fp32 factor; probe only, not wired) | dense matvec / GEMM for in_proj and out_proj |
| LiLiCorr drafter | kit/draft/gen_dflash.py (DFlash2: conv modulation, sliding window, context taps) | the LiLiCorr head (2 layers, joint top-8 rescoring), attention sink bias, NVFP4 drafter weights; Mamba rollback on rejection |
| Vision | kit/vision (gemma4v, qwen VL, muse-glimmer families) | C-RADIO v4-H tower + `nemotron_omni` projector (pixel shuffle + MLP); video optional |
| Names | kit/hf_names pattern | a nemotron_h_moe map (88 blocks, three mixer kinds) |

No missing Loom capability was found outright. The risk is in kernels the kit has no example of: a Mamba-2 recurrence and
chunked SSD prefill on gfx1151, IQ2_XXS decode at speed, and per-layer heterogeneous MoE shapes in one generator.

## Compared with Hy4 Flash and Inkling 2 Mini
| | Inkling 2 Mini | Hy4 Flash | Nemotron 3.5 Bolt 74B-A9B |
|---|---|---|---|
| total / active | 26.4B / 3.4B | 220B (155.5B + 63.5B Engram) / 11.3B | 74.9B / 9.06B (routed experts only 1.69B of it) |
| decode bytes, plain floor | 5.55 GB, ~41 tok/s (native) | 6.3-9.8 GB, 23-36 tok/s by recipe | 4.8-8.7 GB at 32k, 26-48 tok/s by recipe (shipping: 5.4 GB, ~42) |
| verify step vs one token | 2.8x (8 tokens) | 2.2-2.6x (6 tokens) | 1.57x (8 tokens, shipping) |
| memory | small | 77-94 GiB + Engram on NVMe; whole GPU | ~26.5 GiB at 256k (shipping); fits beside another model |
| new subsystems | looped layers, attention variants, short convs, block residuals, NVFP4 / MXFP8 | MLA, compressed-indexer sparse attention, Engram, 4-stream HC, LatentMoE, sink + gate | Mamba-2 SSD, IQ2_XXS, LatentMoE, relu^2, heterogeneous MoE, LiLiCorr head, RADIO vision |
| others on Strix Halo | Strata 47-48 tok/s | Strata 51-60 tok/s (DFly2 on) | none found; RTX Spark 84 tok/s with LiLiCorr [meas, NVIDIA] |

## Verdict
Worth a kit, and the cheapest of the three to make useful: a small footprint that coexists with the live model, a plain
decode floor (~42 tok/s shipping, ~48 with Q4_K non-experts) above Hy4's, and the best speculation economics (1.57x verify cost,
4.3 mean acceptance). The attention side is trivial (8 NoPE GQA layers). The work concentrates in four pieces: a Mamba-2
recurrence and chunked prefill (the GDN machinery is the template), an IQ2_XXS expert path, LatentMoE with relu^2 and per-layer
shapes (LatentMoE is shared with Hy4, so it is built once for both), and the LiLiCorr head on the existing DFlash2 drafter.
Run the shipping QAD 2-bit file, not a requant of the BF16 release: it is NVIDIA's intended product and the BF16-derived
Q4 recipes read more bytes for no measured gain.

## Plan, cheapest evidence first
1. Static, no weights: **done 22 Nov**, all compiled for gfx1151 on Loom 0.2.0 and 0.3.0 and lint-clean.
   - kit/gen_iq2xxs.py: IQ2_XXS codes for the int8 decode path (grid magnitudes with their signs applied on 32-bit words, the
     sign table computed from parity, one scale per 32 values) and a single-token probe matvec, the 2 KB grid in constant
     memory or staged in LDS: 225 / 230 instructions, 37 / 30 VGPRs, 8 dot4 (kit/test_iq2xxs.py). Which placement is
     faster is a GPU question.
   - kit/gen_mamba2.py: a Mamba-2 decode step in llama.cpp's operation order (softplus, a scalar decay per head, each state row
     split over 8 lanes with a xor-shuffle sum, D added after): 186 instructions, 80 VGPRs, 16 waves, the state read and
     written once (4 MiB each way per layer: ~37 us per layer, ~1.5 ms per token at f32) (kit/test_mamba2.py).
   - kit/hf_names/nemotron_h_moe.json: the GGUF <-> BF16-release name map for all 88 blocks (three mixer kinds, experts
     stacked, A from A_log, the NVFP4 per-tensor factors GGUF-only), checked against both real tensor tables
     (kit/test_hf_names_nemotron_h_moe.py: every tensor once, every pairing by shape).
2. Reference on the machine (the GGUF, mmproj and drafter are downloaded and the run is scripted; it waits for a GPU window): llama.cpp main built for gfx1151 with the
   NVIDIA 24GB GGUF + LiLiCorr: decode with and without the drafter at 4k / 32k, prefill at 2k / 32k, per-op time
   (`GGML_SCHED` / events) to see whether Mamba, IQ2_XXS matvecs or the routers dominate, and logits for later equivalence.
3. Generator (kit/, behind a `nemotron_h_moe` path): plain decode first (Mamba-2 + IQ2_XXS LatentMoE + NoPE attention),
   then chunked SSD prefill, then LiLiCorr with state rollback, then the C-RADIO vision tower (vision is a goal for every
   kit, so its encoder pieces are built to be shared: kit/vision); requantise the Q8_0 non-expert tensors to Q4_K only after a
   KLD measurement against the shipping file.
