# Inkling 2 Mini: the decode program (plan, 23 Nov)

What one decode step (1 token) and one verify step (a span of T tokens) launch, in order, with the kernel that does each part,
whether it exists, and what is still open. Model facts and sources are in NOTES.md; kernels are in kit/ (gen_*.py) and are
value-checked on kit/loomsim.py unless marked otherwise. Weights route: Unsloth UD-Q4_K_XL GGUF (dense rows in K-quants, experts
in ggml NVFP4 / MXFP8 blocks).

## Applications and slots

48 layer applications per token: prelude 0-5, core 6-17 three times, coda 18-23. Application a uses weights of layer
`w(a)` (0-5, then 6-17 repeated, then 18-23) and cache / conv slot `a` (per-pass state, exact). In Loom the weight index is a
computed `command.parameter "blk.{}..."[%index]` (docs guide/command-programs.md); the state index is the application.

## One application (decode, T = 1; verify spans use the `_t{T}` / span forms)

Before application 0: embedding lookup, then an RMS norm with weight `token_embd_norm.weight` (F32 [2304], eps 1e-6; found in the GGUF
23 Nov, not in gen_llm: the NORMS kernel with that weight, one launch per token). Its output is the depth mix's embedding input.

| # | step | kernel | state |
|---|---|---|---|
| 1 | layer input: block attention residual mix over [embedding, finished 6-layer block sums, running partial] | gen_loop.depth_mix | built |
| 2 | (passes 2-3, core layers) message = depth mix over the previous pass's block states | gen_loop.depth_mix | built |
| 3 | attention norm | gen_llm NORMS (RMS) | in the Qwen kit; checked 23 Nov: weighted (blk.N.attn_norm / ffn_norm F32 [2304]), eps 1e-6 (rms_eps in config.draft.json, read by gen_llm) |
| 4 | (passes 2-3) controllers on the q, k, v inputs | gen_loop.controller_split (z + out), BF16 | built; batching q/k/v in one launch open |
| 5 | q, k, v, r, gate projections (dense, K-quant rows) | gen_llm mv (Qwen formats) | in the Qwen kit; Inkling shapes (2304 in) to generate |
| 6 | K / V short convolutions (dynamic taps on global layers) + prep (clips, value norm, tau) + KV write | gen_sconv.sconv_pos + gen_inkattn.prep | built separately; fusion into one prologue planned (NOTES) |
| 7 | attention: sliding (40 slots) or global (8 slots), KV8 | gen_inkattn.decode_swa / decode_global (span: decode_swa_span / decode_global_span) | built |
| 8 | output projection + attention-output convolution + residual | gen_llm mv + sconv_pos epilogue | conv epilogue fusion planned |
| 9 | FFN norm; (passes 2-3) controller on the FFN input | NORMS, controller_split | built |
| 10 | router: sigmoid scores + bias, top-8, weights renormalised x 8 | gen_moe.route_decode / route_prefill (gating 2, bias, norm, scale 8) | exists: configuration only |
| 11 | experts: gate_up (K 2304, 2 x 1152, SITU softcaps 4 / 25) and down (K 1152); in the UD-Q4_K_XL GGUF NVFP4 in layers 0-19 and Q8_0 in 20-23 (the native MXFP8 is requantised; per-expert F32 scales beside) | gen_moe expert_mv (formats nvfp4, q8_0) | NVFP4 built (GPU value test q106 pending); Q8_0 is an existing kit format |
| 12 | expert combine + MoE-output convolution + residual | gen_moe combine + sconv_pos epilogue | conv epilogue fusion planned |

After application 47: final norm, LM head (200,058 used rows of 201,024), logit softcap 30, sampling. The Qwen kit's head and
sampler carry over (softcap to add).

## Launch count and the floor

Separate kernels as listed are ~20 launches per application, ~960 per token: at a few microseconds each that alone is several
milliseconds against a 24 ms weight floor (NOTES Floors). The fusions planned above (conv into epilogues, prologue fusion of
step 6, q/k/v controllers in one launch, depth mix with the norm) aim at ~10 per application. Measure per-launch overhead on this
machine before deciding how far to fuse (the Qwen kit's served profile gives the gap between back-to-back launches).
Measured 23 Nov (HRX-like E328, the Qwen kit's served decode events): 0.72 us from one dispatch's end to the next one's start inside
a command buffer (p10 0.72, p90 0.76; 1,196 dispatches per decode step, 0.87 ms of a 115 ms step). For ~960 launches that is ~0.7 ms
against the 24 ms floor: launch count alone does not justify fusion here. What does is each small kernel's fill and drain (a grid that
cannot fill the chip, q112) and the bytes a fusion saves (a conv epilogue that does not re-read its input); judge each fusion by those.

## Controller and single-workgroup kernels

controller() ran one workgroup per token (E297); every per-token kernel in this list needs a grid that fills the chip at T = 1.
Check each new kernel's grid at T = 1 against q112's bandwidth-vs-workgroups figures before it enters the program.

## Verify spans (DSpark, T = 8)

Span forms exist for the controller, sliding and global attention (K / V read once per span). Still per token: the depth mix
(cheap), the convolutions (sconv_pos handles spans), the dense matvecs (the Qwen kit's 8-token matvecs), the experts (union of
the span's routed experts: NOTES estimates ~40 % of each layer's experts for 8 tokens; engine-lab's MoE decode GEMV serves up to 8
rows per launch, research/engine-lab-inkling.md).

## Config (23 Nov)

kits/inkling-2-mini/config.draft.json comes from kit/inspect_gguf.py: hidden 2304, 24 / 8 heads x 128, sliding window 1024 on 40 of the
48 applications (global every 6th), attention gate, MoE 128 experts top-8, gating 2 (sigmoid + exp_probs_b selection bias),
weights renormalised x 8, SITU caps 4 / 25 (read from the GGUF's activation.situ_beta / situ_linear_beta), fused gate_up with
per-expert scales. Dense projections are Q8_0 and the head Q8_0 (kit formats); experts NVFP4 in layers 0-19, Q8_0 in 20-23.
Not yet buildable: 'layers' 48 counts applications over 24 blocks, so gen_llm needs the application -> block map for weights
and the application index for caches; the chat block is filled by hand.

## Where it plugs into gen_llm (23 Nov, read)

gen_llm.layer_text(i) builds trunk layer i from tensors `blk.{i}.*` (`command.parameter %weights, "blk.{i}..."`) and gives it a KV-plane
ordinal (`%layer_{i}`). For the loop the change is small: an application index a (48) for ordinals, caches and conv slots, and a block index
w(a) (24: 0-5, 6-17 three times, 18-23) for the parameter names. The large part is the layer body: layer_text emits Qwen-style layers
(norm, projections, flash / GDN, dense or MoE FFN); Inkling needs its own body (steps 1-12 above: depth mix, controllers, the four
convolutions, the relative-bias attention with sink and gate, the sigmoid router and SITU experts) wired through the same program
skeleton (embedding, head, sampling, drafter hooks). Plan: an `inkling2_layer_text(a)` beside layer_text, selected by arch.

## Prefill attention: scores in registers (23 Nov, prior art)

strix-halo-engine-lab PR #991 (Apache-2.0; research/engine-lab-inkling.md): when one wave covers a whole head (head_dim 128 here, one slice),
use K as WMMA operand A and Q as operand B so a query row sits in the lane and its keys in registers; exchange four registers between
half-waves, keep the same products and accumulation order, and drop the LDS round trip and one barrier per key tile (bit-identical).
Qwen3.8's 256-wide heads split over two waves cannot use it (the partial sums cross waves: Qwen's flash2 exchanges them through LDS,
FLASH2_SX11); Inkling's 128-wide heads can, so the Inkling flash prefill starts from this layout.

## Order of work

1. Generate the dense matvecs and norms at Inkling's shapes from the Qwen kit's generators, and the router / experts from gen_moe with
   Inkling's configuration (gating 2, scale 8, SITU, nvfp4 + q8_0); no new expert format is needed on the GGUF route.
2. A program for one application end to end on the GPU against the llama.cpp reference (per-layer outputs, short prompts).
3. All 48 applications + head: first token equal to llama.cpp's top-1, then KL on a short corpus.
4. Fusions (launch count), then verify spans and DSpark, then prefill (engine-lab's PRs are the map: chunk, attention tile,
   controller in the GEMM epilogue, SwiGLU epilogue, row mapping instead of gathers).
