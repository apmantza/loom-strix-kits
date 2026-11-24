# loom-strix-kits

> **Work in progress.** These kits change daily. Every number comes from one machine, and nothing here is a release
> yet. Expect rough edges, and open an issue if something does not build.

These are [Loom](https://pypi.org/project/hrx-loom/) (hrx-loom) model kits tuned for **AMD Strix Halo**: Ryzen AI
Max+ 395 with the Radeon 8060S (gfx1151, RDNA 3.5) and 128 GB of unified LPDDR5X. Each kit pushes one model's decode
and prefill toward this chip's hardware limits, and every change is measured on the served program.

The repository has three parts:
- `kit/` is the shared generator that writes the Loom programs for every kit. It is forked from the `loom-port-llm` starter
  of [loom-fun/loom-skills](https://github.com/loom-fun/loom-skills). Each optimization is a switch, and an unset switch
  leaves the generated program unchanged, so one model's tuning cannot change another's programs.
- `kits/<model>/` holds what is specific to one model:
  - a manifest (`kit.env`: model id, recipe switches, build and serve flags, reference GGUF);
  - the model config;
  - pins (`pins.json`: the hrx-loom version measured, each file's repository, revision and sha256) and a fetch script;
- `common/` holds tools that work for any kit: build, serve, kernel tests, output equality, gates, an agent-session A/B,
  and code-object analysis.
- `skills/` holds agent-readable procedures (plain markdown, any coding agent): `skills/loom-strix-optimise/` is how we
  do speed work on a kit, with the rules (each with its evidence), the gfx1151 device facts and the traps. A kit's own
  model knowledge is in `kits/<kit>/NOTES.md`.

The weights are never redistributed; each kit downloads them from their original repositories.

## Kits

| Kit | Model | Weights | State |
|---|---|---|---|
| [qwen38-27b](kits/qwen38-27b/) | Qwen3.8-27B (hybrid Gated DeltaNet + attention, DFlash2 speculative decoding) | unsloth UD-Q4_K_XL GGUF | used daily for agent work; tuning ongoing |
| [inkling-2-mini](kits/inkling-2-mini/) | Thinking Machines Inkling 2 Mini (26B MoE, ~3.4B active, looped layers) | native NVFP4/MXFP8 release; Unsloth GGUFs as variants | scoping (NOTES.md) |
| [hy4-flash](kits/hy4-flash/) | Tencent Hy4 Flash (220B MoE incl. 63.5B Engram tables, ~11.3B active, MLA + sparse indexer) | AngelSlim GGUFs (non-expert q8_0); native BF16 / FP8 | scoping (NOTES.md) |
| [nemotron-bolt](kits/nemotron-bolt/) | NVIDIA Nemotron 3.5 Bolt 74B-A9B (Mamba-2 / LatentMoE / NoPE attention hybrid, ~9.1B active, C-RADIO vision) | NVIDIA's QAD 2-bit GGUF (experts IQ2_XXS, Mamba NVFP4) + LiLiCorr drafter | scoping (NOTES.md) |

## Requirements

- Strix Halo (gfx1151). Other RDNA 3 parts may build, but they are untested and the tuning assumes this chip.
- Linux with ROCm-capable amdgpu, and **unlimited memlock** for loom-serve (io_uring pinning).
- Python 3.12, `pip install hrx-loom==0.2.0 gguf numpy tokenizers huggingface_hub`.
- A loom-skills checkout for its knowledge base (`LOOM_KB_ROOT=<checkout>/skills/loom-kb`), tested at 7952d7f.
- Note that the BIOS power mode matters: on some boards the 85 W default caps prefill noticeably below the 120 W mode.

## Usage

```bash
cp common/env.example ~/.config/loom-strix-kits/env   # then edit it: venv, loom-skills checkout, llama.cpp, work dir
kits/qwen38-27b/fetch.sh                              # pinned weights + tokenizer
common/build.sh qwen38-27b pkg                        # the kit's recipe; EXTRA="SWITCH=1" adds switches
common/serve.sh qwen38-27b pkg                        # OpenAI-compatible API, the kit's serve flags
```

| Tool | What it does |
|---|---|
| `common/build.sh KIT PKG` | builds with the kit's recipe; writes PKG.cmd (the exact command) and refuses while a measurement runs |
| `common/serve.sh KIT PKG` | serves in the foreground with the kit's flags |
| `common/state.sh KIT [--pkg\|--idle]` | shows what serves on the port, read-only |
| `common/ktest.sh KIT PKG DIR` | runs the kernel tests for exactly the switches PKG was built with |
| `common/serve_as.sh KIT PKG` | serves one package for a measurement: stops other servers, waits for a quiet machine, checks the port serves PKG |
| `common/eq.sh KIT PKG REF OUT` | greedy outputs and draft counters against a reference, one server at a time; exits 1 on any difference, 3 (CONTROL FAILED) when the comparison misses a planted one-character / one-counter difference |
| `common/gates.sh KIT PKG OUT` | runs loom-skills' quick gates (llama.cpp reference outputs, chat, tools, vision, concurrency) |
| `common/agent_ab.sh KIT A B OUT SESSION` | A/B on a frozen agent session (`agent_bench/freeze.py`), ABBA halves |
| `common/loop_census.py`, `kernel_lint.py`, `load_cover.py`, `load_events.py` | code-object analysis: hot loops, register moves, load-to-wait cover; the lint over a package, against another build or compiler |
| `common/binary_gate.py BASE CAND --allow REGEX` | untouched-kernel gate between two packages: per kernel, the source closure (own text + every helper it reaches); changed closures compiled and compared on ISA, VGPR, SGPR, scratch, LDS; fails on an unallowed change, a missing kernel, new scratch, or an `--allow` that changed nothing |
| `common/serve_code_dump.py` | dumps the kernels a running loom-serve compiled, which are not always the ones a benchmark compiles |
| `common/agent_run_summary.py` | summarizes server-side numbers of an agent run from /metrics snapshots |
| `common/events_ab.sh KIT OUT LEN PKG...` | served per-dispatch profiles (ABBA order) at one prompt size, then every lane against its hardware floor (`kits/<kit>/lanes.json`) |
| `common/events_summary.py`, `lane_breakdown.py`, `ceiling.py` | the lane table, one lane kernel by kernel (with GB/s for matvecs), the floors themselves |
| `common/fresh.py KIT` | is the kit still current: installed and measured hrx-loom against PyPI, each pinned file's sha256 against its repository's main (exit 1 stale, 2 check failed) |
| `common/new_kit.sh NAME MODEL_ID` | scaffolds `kits/NAME/` (manifest, pins, lanes, notes, fetch wrapper) |
| `common/make_lanes.py WEIGHTS` | the floors' model constants from a GGUF or safetensors header (local or remote): bytes per decode step (MoE: the experts a verify step's tokens touch; looped layers; MTP and vision excluded), multiply-adds, KV slots for hybrid attention |
| `common/kit_check.sh KIT` | the weights gate: files match the pins, upstream freshness, lanes.json reproduces from the weights |
| `common/quant_eval.sh REF CAND OUT` | KL divergence, same top token and perplexity of a quant against a reference quant (llama.cpp) |
| `common/hsaco_run.py` | Launch any Loom kernel's code object through HIP (arguments from its metadata, grid from the .loom source): timing, A/B, output comparison, counters |
| `common/counters.sh` | Hardware counters on a HIP process (ROCm 10.2): wait reasons, memory bytes, L2, load latency, with derived metrics per GPU |
| `common/bw_probe.sh` | DRAM read, write and copy bandwidth (HIP), the measured ceiling behind the decode floors |
| `common/fetch.py KIT` | downloads the kit's pinned files (`kits/<kit>/pins.json`) and verifies every sha256 |
| `common/checks.py` | A/B verdict checks (paired bootstrap CI, ABBA neighbour rule), a loop flag, GPU memory per process from fdinfo |

`build.sh`, `state.sh`, `ktest.sh` and `events_summary.py` (on a real served profile) have run in this layout. `serve_as.sh`, `eq.sh`,
`gates.sh`, `agent_ab.sh` and `events_ab.sh` are ports of the scripts we use daily; their first runs in this layout are still pending.
`events_ab.sh` needs an `iree-profile` built from the installed hrx-system tag (`IREE_PROFILE` in the machine env).

## License

MIT (see LICENSE). The generators derive from loom-skills (MIT, Copyright (c) 2026 Loom Fun contributors); their
notice is kept in LICENSE. Model weights carry their own licenses (named in each kit's README).
