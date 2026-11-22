# Qwen3.8-27B on Strix Halo (Loom)

> **Work in progress.** This is the recipe our daily-driver server runs. Its numbers come from one Strix Halo box.

## Weights (pinned in fetch.sh)

| File | Repository @ revision | License |
|---|---|---|
| Qwen3.8-27B-UD-Q4_K_XL.gguf (17.7 GB) | unsloth/Qwen3.8-27B-GGUF @ 4af51b35 (UD3.1) | Apache-2.0 (Qwen) |
| mmproj-F16.gguf (vision) | unsloth/Qwen3.8-27B-GGUF @ 4af51b35 | Apache-2.0 |
| Qwen3.8-27B-DFlash2-Q4_K_M.gguf (drafter) | z-lab/Qwen3.8-27B-DFlash2-GGUF @ 2d9571f8 | Apache-2.0 |
| tokenizer.json | Qwen/Qwen3.8-27B @ 37561816 | Apache-2.0 |

## Quick start

```bash
./fetch.sh                                     # ~19.7 GB into ./models, ./tok
cd ../.. && common/build.sh qwen38-27b pkg     # ~20 s: writes kits/qwen38-27b/pkg (programs + model.json)
common/serve.sh qwen38-27b pkg                 # OpenAI-compatible API on 127.0.0.1:8160
```

The machine settings (venv, loom-skills checkout) go in `~/.config/loom-strix-kits/env`; see `common/env.example`.
The recipe and flags are in `kit.env`.

The server holds about 53 GB resident: a 262,144-token
context, a KV pool shared by all rows and the prefix cache, 4 initial rows, and card sampling (temperature 1.0,
top-p 0.95, top-k 20) when a request sets none.

## Numbers (one request alone, 81,430-token prompt, device time from the served profile)

| | Now | Hardware floor (sum of lanes) |
|---|---|---|
| Prefill | ~409 tok/s (198.9 s) | ~520 tok/s practical |
| Decode | ~25 tok/s (124 ms per verify step, ~3.1 tokens per step) | ~32 tok/s at today's acceptance |

On real agent tasks served by this recipe (contexts up to ~136k, temperature 1.0), decode averaged 23 to 26 tok/s.

## What the recipe switches do

Each switch is read in the file named (under `kit/`). Unset, a switch leaves the generated program unchanged.

| Switch | Where | What |
|---|---|---|
| PREFILL_F16_SCRATCH | gen_llm.py, gen_gemm.py | prefill chunks of 2048+ rows dequantize each weight once into an f16 scratch, then run f16 WMMA GEMMs |
| PF16_PAD=64 | pf16_pad.py | pads f16 operand rows so that no row stride is a multiple of 4 KB (memory channel camping) |
| KV8=1 KV8_LAYOUT=2 | gen_llm.py | 8-bit KV cache (int8 codes + scales). It changes numerics; it passed our needle and 26k/81k/130k continuation gate against f16 |
| ATTN_L17_VS, FLASH2_VS_KVH, FLASH2_SPLITK_ANY | gen_llm.py, gen_flash2s.py | verify attention for the 8-token speculative step (split over keys, KV-head hoist, blocks that straddle pages) |
| FLASH2_PF11*, FLASH2_KVH11 | gen_flash2s.py, flash2_prefetch.py | prefill flash attention prefetch and carry scheduling |
| FLASH2_KV8_KCARRY, FLASH2_KV8_VLATE=2 | kv8_late.py | KV8 prefill attention: carry raw K across iterations, fence V loads |
| GDN_FOLD, GDN_AB_R | gen_gdn.py, gen_llm.py | Gated DeltaNet decode state fold; alpha/beta projection rows |
| DFL_SPLIT_W, DRAFT_MM_FROM=17 | draft/gen_dflash.py, gen_llm.py | DFlash2 drafter weight split and matmul threshold |

## Correctness

`common/ktest.sh qwen38-27b pkg TESTDIR` writes and runs the kernel tests for the switches the package was built with. Every recipe change here passed these kernel tests,
greedy output equality against the previous package (texts and draft counters), and a served A/B before going live.

## Requirements and limits

- gfx1151 only, with ~53 GB free: this preset does not fit a 64 GB machine.
- Serve with `--pool_backing=fixed` (`common/serve.sh` does this). KV8 is only correct with fixed backing on Loom 0.2.0.
- Unlimited memlock is required.
