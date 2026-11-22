# Tools by step of the work

Search this list, Loom's own tools (`loom-*`, their `--help`, the docs' workflows), installed Python packages and the ROCm
and llama.cpp builds before writing a tool: a hand-written remote header reader, a hand-derived bank analysis and a planned
KL-divergence path each turned out to exist already (rules.md section 2). Add a tool here only with the step that uses it.

| Step (SKILL.md) | Question | Tool |
|---|---|---|
| Weights in | Are the files the pinned ones? Is anything newer upstream? | `common/fetch.py KIT --verify-only`, `common/fresh.py KIT`, both inside `common/kit_check.sh KIT` |
| Names | Does a kit's name map cover the checkpoint, and pair each tensor with its GGUF twin? | `kit/hf_names/<family>.json`; `kit/test_hf_names_inkling2.py` (every native key once, every GGUF name, shapes; fixture lists in kit/fixtures/) |
| Floors | What would each lane cost at the hardware limit? | `common/make_lanes.py` (GGUF, local or `hf:` safetensors headers via huggingface_hub); `common/kit_check.sh` fails when lanes.json drifts from the weights; `common/ceiling.py` |
| Device limits | Is the bandwidth floor itself right? | `common/bw_probe.sh` (read / write / copy GB/s with 128-bit accesses; the read figure bounds decode), against lanes.json `hardware` and devices/<gfx>.md. rocm-bandwidth-test did not build here (Boost's nested submodules; its pinned TransferBench uses a hardware register gfx1151 lacks) |
| Scoreboard | Where does served time go? | `common/events_ab.sh`, `events_summary.py` (lanes against floors), `lane_breakdown.py` (per shape, GB/s) |
| Static: the binary | What does the loop issue and wait on? | `common/loop_census.py`, `compiler_ab.py` (two compilers, every root), `load_cover.py`, `load_events.py`; `llvm-objdump-21 -d --mcpu=gfx1151` |
| Static: the compiler's view | Why did it place / wait / spill? | `loom-compile --compile-report=details` then `loom-compile-report show / diff / suggest` (wait reasons, `suggest` for pipeline copy waits, spill and allocation rows, schedule bands) |
| Static: LDS | Do these LDS accesses conflict? | the report's Bank service view (`source_low.memory.bank_service_groups`: silicon-calibrated gfx1151 model for b16/b32/b64/b128 and u16_d16). When a kernel's lane predicate keeps a group "unknown", model the access pattern alone in a small probe kernel |
| Static: registers | Where is register pressure, live range by live range? | Radeon GPU Analyzer (`rga`, binary-analysis mode on the .hsaco: live-register analysis, CFG); the compile report's allocation high-water rows |
| Served binary | Is the bench's binary the served one? | `common/serve_code_dump.py` (the launch-site specializations loom-serve compiled), then time them through a HIP harness |
| HIP harness | How fast is this code object, do two compilers' objects agree, and what do its counters say? | `common/hsaco_run.py` (any kernel: arguments from the code object's metadata, grid from the .loom `kernel.launch.config` region, interleaved A/B, `--compare` outputs, `--dry-run`); under `common/counters.sh ... -- hsaco_run.py ...` for counters |
| Counters | What do the caches, LDS and waves do while it runs, and what does each wave wait on? | `common/counters.sh OUT waits,bytes,l2,latency -- <HIP harness>` (ROCm 10.2 rocprofv3, one pass per preset, derived metrics in common/counters/<gfx>.yaml: WaitVmemPct, WaitDelayAluPct, MemReadBytes, VmemLoadLatency, VmemLoadsInFlight; `counters.sh OUT calibrate` first, on bw_probe) on the served code object through a HIP harness (Loom drives /dev/kfd itself; devices/gfx1151.md). No PC sampling on gfx1151 (rocprofv3-avail lists no agent); `rocprofv3` ATT thread trace and PC sampling for per-instruction stalls (gfx1151 support: test first) |
| Correctness | Same results? | `kit/make_kernel_tests.py` + `common/ktest.sh` (cases and deliberate breaks), `common/eq.sh` (texts and draft counters), `common/gates.sh` |
| Weights quality | Is this quant good enough? | `common/quant_eval.sh REF CAND OUT` (llama.cpp KL divergence, same top token, perplexity); a kit holds several quants as `variants/<name>/` (KIT_VARIANT) |
| Verdict | Is the candidate faster? | `common/agent_ab.sh` (paired CI, neighbour rule, loop flags), `common/checks.py` |
| Memory | What does the server hold? | `common/state.sh` (GPU memory from /proc/PID/fdinfo) |
| SDK changes | What did a Loom release change in our code? | `common/compiler_ab.py` over every root; `git bisect run` over a source build with a probe like E216's (first bad commit in ~10 builds) |

Not useful here: `llvm-mca` (no gfx1151 scheduling model), rocprof against loom-serve or loom-benchmark (sees nothing).
