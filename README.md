# loom-strix-kits

> **Work in progress.** These kits change daily. Every number comes from one machine, and nothing here is a release
> yet. Expect rough edges, and open an issue if something does not build.

These are [Loom](https://pypi.org/project/hrx-loom/) (hrx-loom) model kits tuned for **AMD Strix Halo**: Ryzen AI
Max+ 395 with the Radeon 8060S (gfx1151, RDNA 3.5) and 128 GB of unified LPDDR5X. Each kit pushes one model's decode
and prefill toward this chip's hardware limits, and every change is measured on the served program.

A kit contains:
- a generator that writes the Loom programs, forked from the `loom-port-llm` starter of
  [loom-fun/loom-skills](https://github.com/loom-fun/loom-skills);
- a model config;
- a fetch script that pins the weights by revision and sha256;
- the build recipe (generator switches) and serve flags;
- the kernel tests.

The weights are never redistributed; each kit downloads them from their original repositories.

## Kits

| Kit | Model | Weights | State |
|---|---|---|---|
| [qwen38-27b](kits/qwen38-27b/) | Qwen3.8-27B (hybrid Gated DeltaNet + attention, DFlash2 speculative decoding) | unsloth UD-Q4_K_XL GGUF | used daily for agent work; tuning ongoing |
| inkling-2-mini | Thinking Machines Inkling 2 Mini (26B MoE, ~3.4B active, looped layers) | to be decided: native NVFP4/MXFP8 or unsloth GGUF | planned |

## Requirements

- Strix Halo (gfx1151). Other RDNA 3 parts may build, but they are untested and the tuning assumes this chip.
- Linux with ROCm-capable amdgpu, and **unlimited memlock** for loom-serve (io_uring pinning).
- Python 3.12, `pip install hrx-loom==0.2.0 gguf numpy tokenizers huggingface_hub`.
- A loom-skills checkout for its knowledge base (`LOOM_KB_ROOT=<checkout>/skills/loom-kb`), tested at 7952d7f.
- Note that the BIOS power mode matters: on some boards the 85 W default caps prefill noticeably below the 120 W mode.

## Layout

```
kits/<kit>/kit/        the generator (Python) and its kernel tests
kits/<kit>/config.json the model config read by kit/build.py
kits/<kit>/fetch.sh    pinned downloads into kits/<kit>/models, sha256-checked
kits/<kit>/build.sh    the current recipe: kit/build.py with its switches
kits/<kit>/serve.sh    loom-serve with the flags the kit was measured with
common/                (coming) shared tooling: A/B harness, served-profile tools, gates
```

## License

MIT (see LICENSE). The generators derive from loom-skills (MIT, Copyright (c) 2026 Loom Fun contributors); their
notice is kept in LICENSE. Model weights carry their own licenses (named in each kit's README).
