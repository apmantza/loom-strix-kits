# Streaming from NVMe on Strix Halo: n-gram tables and MoE experts

Status: collected 22 Nov 2026; nothing here is measured on Strix Halo yet. Sources are linked per item; **[theirs]** marks a number
measured on someone else's hardware, **[comp]** one computed here, **[hyp]** a hypothesis.

## Why it matters here
Strix Halo has one memory pool (~110 GB GPU-visible LPDDR5X at ~228 GB/s, CPU and GPU share it), so the discrete-GPU question
"VRAM or RAM?" does not exist: the only tier below memory is NVMe. Two kinds of data end up there:
- **hashed n-gram embedding tables**, which are large and read sparsely by token ids: Hy4 Flash's Engram (63.4B parameters, ~59 GiB
  at 8 bits, 64 rows of 96 values per token; kits/hy4-flash/NOTES.md) and Qwen3.8 Flash Next's per-layer embeddings (a hashed table
  with a row cache fed from the checkpoint file in AMD's hrx-loom-library);
- **MoE experts** of quants that exceed memory, which the user runs on purpose for the largest models (DeepSeek, Flash Next).

## n-gram tables (Engram)
- Ids depend only on the token ids, so every lookup of a step is known before the step: hash on the host, batch the reads, prefetch
  before the layers that consume them (llama.cpp's Hy4 support hashes on the host before the first gather; kits/hy4-flash/NOTES.md).
- A DGX Spark recipe for Hy4 Flash (https://github.com/sxuff/Hy4-Flash-Spark-recipe, MIT; AngelSlim's llama.cpp patch) keeps the
  Engram table (Q8_0, 62.7 GiB) on the SSD through mmap: all four Engram modules are hashed on the host, one batched prefetch,
  then a CPU gather; Engram reads cost 8-11 % of prefill time **[theirs, GB10]**.
- Page-granular reads fetch far more than the rows used: 96-byte rows in 4 KiB pages, roughly 16x over-read for scattered rows
  **[comp, by the study of that recipe]**. Measure the real row locality on a long prompt before building a row cache.
- Decode needs ~6 KB of Engram rows per token at 8 bits (kits/hy4-flash/NOTES.md): latency, not bandwidth, is the risk; a
  prefill chunk of 4k tokens touches ~262k rows.

## MoE experts beyond memory
From "Everything I learned offloading MoE experts on 4x RTX 3090" (an X article by @superalesha, 12 Nov 2026; four months on
DeepSeek V4.1 Flash, Qwen3.8 Flash Next, GLM-5.3 Flash, MiniMax M3.1 Flash with Strata, llama.cpp, vLLM and forks). All numbers
**[theirs]**; on their rig the miss path is PCIe from RAM, here it would be NVMe.
- Decode time per token = a constant (dense weights, attention, launches) + misses x cost per miss. Measure the miss share before
  buying residency: once misses are cheap the constant dominates.
- Expert popularity differs by model. DeepSeek V4.1 Flash: the top 200 of 384 experts get 89 % of picks. Qwen3.8 Flash Next: 53 % of
  experts resident gives 92-95 % hits. GLM-5.3 Flash and MiniMax M3.1 Flash: a profile beats random placement by only 3-8 points, so
  a static hot set is nearly useless for them.
- A live cache with decaying counters is the main result: 87-96 % hits on MiniMax against 97-99 % for an oracle (Belady); the policy
  matters more than the cache size. Short answers (tool calls) do not warm it (71-76 % vs 85-93 % on long answers).
- Prefetch did not work: the previous token's "almost picked" experts did nothing; predicting layer L+1 from layer L is right 84-90 %
  of the time but too late to hide a copy (an expert copy took longer than a decode layer).
- Prefill touches 70-80 % of a layer's experts per 1,300-row chunk, so cold experts are re-read every chunk; bigger chunks helped.
- Speculative decoding multiplies misses (more distinct experts per verify step); drafting by cost recovered the losses on MiniMax.
- Lower the bits rather than drop experts.

## What transfers to this machine
- The time model and the per-model popularity results transfer; the CPU-compute path, RAM channel and multi-GPU results do not (one
  pool, no PCIe hop).
- A miss is an NVMe read: an ~11 MB expert at ~7 GB/s is ~1.6 ms **[comp; our drive's speed is unmeasured]**, close to their PCIe 3.0
  copies, so their cache and prefetch conclusions should hold.
- Design implied **[hyp]**: a validated static placement plus a decaying live cache, whole experts read into a pinned slot pool (not
  page faults), and prefill as the bottleneck (perhaps layer by layer). An expert predictor (Inkling 2's, assessed 22 Nov) only pays
  in an NVMe expert-streaming kit; on models that fit it is not useful.
- Before streaming at all, check whether a smaller quant fits: for Flash Next a ~94 GB IQ4_XS that nearly fits probably beats
  streaming a 177 GB Q4 **[hyp]**.

## Next evidence (cheapest first)
1. Our NVMe's random 4 KiB and 1 MiB read rates and latency (fio), read-only.
2. Router traces of one real agent session per model (which experts, per layer, per token) to place each model on the popularity
   scale above.
3. Engram row locality on a long prompt (distinct rows and distinct pages per 4k-token chunk).
