# Rules: LLM inference on Strix Halo with Loom

How to diagnose and change kernels on AMD Strix Halo (Ryzen AI Max+ 395, Radeon 8060S, gfx1151 / RDNA3.5, unified LPDDR5X),
whatever the model. Each rule carries the measurement that taught it; E-numbers point into the authors' experiment log (not
published), the numbers next to them are the evidence. Device facts: devices/gfx1151.md. Order of work and tools: SKILL.md.

## 1. Find what each tile waits on before choosing what to cut

A kernel is bound by one of: DRAM bytes, issue rate (VALU/MMA instructions), **memory latency (the number of
dependent or scattered reads each tile waits on)**, or registers/LDS (occupancy). Fixes for the wrong one do nothing or
make it worse. Classify first, with numbers:

- **bytes/time vs bandwidth** and **instructions/time vs issue rate** (E40: a design assumed DRAM-bound at 3.7x the
  read-once floor; it was VALU-bound and the caches already shared the data, so the load-sharing kernel was slower).
- **bytes from counters are fabric requests, not DRAM**: the L2-to-fabric and GCEA read counters include what the memory-side
  MALL serves (HRX-like E497: a grouped GEMM read 345 GB/s of them, above the 240 GB/s DRAM ceiling). Check a counter's byte rate
  against the device ceiling before calling a kernel DRAM-bound (counters_summary.py flags it), and price a traffic lever by an
  order or ablation's time, not its bytes: two tile orders moved the f16 GEMMs' fabric reads by -28 % and +52 % and their time by
  under 1 % (E305, E494).
- **latency**: a timing ablation that removes the loads (zeros staged instead) and one that keeps them cache-resident.
  L18/E52: prefill attention ran 0.57x without K/V loads and 0.85x with cache-resident loads: latency, not bandwidth.
- **registers**: final VGPRs and the occupancy tier from the compile report; spills share the load counter on gfx11.
- **counters first, once they are calibrated on the machine** (ROCm 10.2 rocprofv3; `common/counters.sh OUT calibrate` must
  show MemReadBytes matching the bytes a probe reads): wait reasons (vmem, ALU dependency, LDS), load latency and loads in flight
  answer the classification directly for any kernel a HIP harness can launch (common/hsaco_run.py); the ablations above stay the
  check when counters are not available or disagree.
- **Occupancy is latency hiding, not a throughput model**: read the limiting resource first (VGPRs, LDS, workgroup slots or the wave
  cap: the compile report names it; devices/gfx1151.md has the VGPR tiers), then the VGPRs and the VGPRs to free for the next residency tier from the
  compile report (`common/kernel_lint.py` prints both); a rewrite that frees registers is worth a tier only if the kernel waits on
  latency. Loom kernels run in WGP mode (a workgroup over 4 SIMDs). A model of part-empty last rounds of workgroups predicted
  14-29 % idle on decode matvecs measured at the bandwidth ceiling (E234): do not rank work by it.

What each kernel class was bound by here, and the number to watch (one model, Qwen3.8-27B; re-classify for another):

| Kernel class | Bound by (evidence) | Watch |
|---|---|---|
| Decode weight matvecs, bulk shapes | DRAM bytes: 204-216 GB/s of a 235 GB/s read ceiling (E192) | GB/s against the floor; bytes per token |
| Decode matvecs that lag their format's peers | open: per-wave issue suspected (q6_k: 616 loop instructions vs q5_k's 261, E233/E235) | instructions per weight, wait reasons |
| Prefill GEMMs | WMMA issue; VALU does not overlap WMMA on gfx11, so dequant VALU competes with the MMAs (E61, E198) | non-MMA instructions per MMA |
| Prefill attention | memory latency: reads waited right after issue (E52, E80, E109); instruction cuts did nothing (E77) | load-to-wait distance (load_cover.py) |
| Decode / verify attention (split-K) | issue: moves and staging dominate the common path (E168, E184) | instructions per key; moves |


**On a latency-bound kernel, count reads, not bytes or instructions.** What costs is how many separate memory
transactions a tile waits on: distinct cache lines, extra address streams, loads whose address depends on another load
(page-table lookups), and waits the compiler inserts. Corollaries seen here:
- Halving bytes does not help (E75: 8-bit KV halved K/V bytes; decode and prefill attention got no faster).
- Cutting instructions does not help (E77: two thirds of the extra dequant VALU removed; prefill time unchanged).
- Adding a second, scattered stream hurts, and can cost as much as everything else saved (E80): in 8-bit KV prefill
  attention the per-key scale reads, one cache line per key, were the whole penalty: 82.0 s with them, 66.9 s with a
  constant scale, f16 67.2 s at 81.4k, while halving the code bytes and cutting two thirds of the dequant VALU had
  changed nothing. Every engine surveyed keeps a tile's side data next to the data or packed contiguously (Strata,
  SGLang, llama.cpp, vLLM). E109 re-diagnosed it: the scale reads cost because they were waited right after
  issue (exposed), not because of their line count (page-local's 8x fewer lines changed nothing).
- A loop that carries a converted value forces the conversion, and the wait on its loads, into the iteration that loads
  it: carry the raw loaded data across the back-edge and convert at its use (E109: KV8 prefill carried K dequantized and
  lost the whole K prefetch, 2-26 instructions of cover vs f16's ~355).
- Prefetch only helps if the load really stays in flight: register allocation can force an early wait
  (ROCm/hrx-system#3409); verify in the disassembly (`s_waitcnt vmcnt`) that overlap survived.
- Design side data (scales, indices, masks, page maps) for the reader's tile: contiguous per tile, aligned, fetched
  with the data or once per tile, never one line per element or per key.

## 2. Cheapest evidence first

Order: static count -> timing ablation -> per-kernel device profile -> build and A/B. Each step can kill a hypothesis
for a fraction of the next one's cost.
- **Static**: compile report and disassembly (VALU, conversions, loads by width, VGPRs, waits) in about a minute
  (E76: a 50-minute build plus profile plus gates refuted an idea the report would have redirected). But a static
  report can miss a runtime cliff (E16: same registers and instruction count, 8.5x slower): profile any loser.
- **A source reorder or prefetch is a claim about the binary**: before timing it, check the moved op and its loads'
  waits with common/load_cover.py (instructions from each load to the wait that retires it) and load_events.py. Loom
  places a pending load's consumer at its earliest legal point; only a data dependency keeps it late (E110: E76's
  deferred conversion never reached the binary, so its "no effect" tested nothing).
- **Count a loop from its backward branch with a script (common/loop_census.py, kernel_lint.py), never a hand-cut line range, and place
  each run of moves in its block (branch join, back-edge, rare path) before naming the mechanism** (E168 -> E179: "264
  v_mov, a back-edge copy" came from a range wider than the loop; the copies sat at branch joins).
- **Compile a generator change as the served program emits it** (E230): splice it into the served program (recipe switches,
  post-passes) before counting. The IQ4 lookup rewrite compiled clean from the bare generator and spilled 2.4 KB under the
  served recipe's padding-skip `scf.if`.
- **Check a model against the measured rows it should also explain before naming it a cause** (E233 -> E234): 'q6_k is slow
  because its last round is part-empty' was written down, then the same model put 14-29 % idle on kernels measured at the
  bandwidth ceiling. A model that mispredicts the controls explains nothing.
- **Ablation**: timing-only builds that remove one ingredient (loads, barriers, dequant, a scale read) rank levers
  before any rewrite (E15, E52). Results are garbage by design; label them so, and check the ablation's binary before timing it:
  the loads still issue in a cache-resident build, the removed work is gone and nothing else is, the barrier and publication stay in
  a dispatch-only build. A build the compiler emptied bounds nothing (E137 kept the stores live with a data-dependent never-true guard;
  E318 stopped on an ablation pattern that matched nothing in live's kernel; Loom: the proxy's native artifact must prove it kept the mechanism).
- **Profile per kernel, both sides of a loss** (E10: an average hid +84 % in one kernel and -11/-23 % in two). Split a
  shared kernel name by call site before attributing time (E13). Attribute by call shape, not by name (E43). A change that moves
  work between kernels (fusion, split, deferral) names the cut it spans first (inputs, outputs, exactness, every reader of what it
  writes) and is judged on the summed served time of every lane in that cut, never on the lane it shrinks; the launch count is not
  the objective (HRX-like E314: a deferred GDN fold cut its lane 3.61 -> 2.86 ms/step and moved the work into the matvecs, +0.6-0.8
  ms/step, bench tie; E329: an epilogue fusion paid 2.75 s).
- **Price the lever before building it** (E15/E24): the price goes in the step's `# bound:` line (strix-ledger: the part now, its
  limit and the best engine, the lever's best case as a share of the request, the measurement that resolves it); a best case below
  every measurement's floor is not built. A best case below the deciding bench's floor names the finer measurement that decides
  (the served lane ABBA, section 5), never 'the bench will tell' (HRX-like E536: q205's flash2 lever was 0.3-1.8 % of prefill against a
  7 % bench floor). Loom requires the same record (agent-driven-kernel-development.md, 'Admit only optimization that can close the gap').
- **Call a cause a hypothesis until a count or an ablation shows it** (E76-E77: two causes stated as likely, both
  refuted by the next measurement). Report what was measured, what is suspected, and the test that will tell. A source
  read is a hypothesis too: follow the value from its producer to the binding the kernel reads before naming its memory,
  and title a row "root cause" only after the direct test (E139 called the page map host-private from its allocation
  site; serving uploads it into a device buffer, E140).
- **Cite the served context, not the profile's label** (E128/E133: "dec100k" served 81,432 tokens and reversed a
  framing). common/events_ab.sh prints it on a `###` line and stops on a >5 % mismatch (common/context_label_check.py); a check's
  status read inside an `echo "$(...)"` is discarded: run it on its own line (HRX-like E421: the size label 81430 served 66,304).
- **A ratio compares equal work** (HRX-like E449: two concurrent agents' '1.7x' wall throughput was mostly 31 % less output, since
  own-mode replies differ; work-adjusted ~1.15-1.2x): before quoting a speedup between arms, check that they generated the same
  tokens, or price each arm's work per unit (common/concurrency_audit.py throughput()).
- **Measure on a quiet machine**: no build, compile or second server beside a profile (a CPU build during a profile moved GEMM
  time 123 -> 131 s). common/wait_quiet.sh waits builds out; common/build.sh refuses during a measurement.

- **Search before building a tool**: Loom's own tools and docs workflows, installed packages, the ROCm and llama.cpp builds
  (tools.md lists them by step). In one session a remote safetensors reader, a hand bank-conflict analysis and a KL-divergence
  path were each about to be written while huggingface_hub, Loom's compile report and llama-perplexity already did them.
- **Plateau rule**: after three levers in a row land inside noise, stop and write a ranked status of the remaining levers, each with
  its price and the evidence behind it, before starting a fourth (E212, from sybil-solutions/inference-optimisation-skill). The same
  holds on a part below about two thirds of its limit: after three exact trims of under 2 % of the part each, the next step for that
  part is a design note for its execution shape (tile ownership, staging, reuse, split or fusion), priced against the limit and the
  best engine, before a fourth trim is queued; kept trims stay kept (exact and not slower). HRX-like: flash2 stayed at 49 % of its
  limit (60.1 s vs 29.7 s) through E468, E496, E511 and E516, each real, none moving the 30 s gap; Loom calls repeated 1-2 % wins
  against a large gap non-evidence for the execution shape.
- **Never park an experiment without its root cause**. "Slower, unexplained" or "held back by X" is not a
  verdict: dig until the mechanism is shown (a counter, an ablation, the disassembly line that waits), then decide. An
  experiment may wait for GPU time, but its log entry names the open question and the next probe, and it is
  investigated before the lane moves on.
- **A win is a pattern: sweep for its siblings before the next lever.** When a change is kept (or a diagnosis names what bounds
  a kernel), name its shape in one line and list every other served kernel or site it could apply to, each priced from the served
  profile (ms per step or per request) with the precondition it needs, before picking the next lever. Shapes seen: compute only
  the rows a consumer reads (HRX-like E697: the last attention layer on the head's rows, -3.9 s per 81k; the rest of that layer is
  the next sibling); skip work on padding rows (E684); fold a layout-only kernel into its producer's store (E695); cut issue in an
  issue-bound kernel (E689). The sweep records what was searched and what was found clean; a sibling inherits the original's
  exactness and grid conditions as hypotheses to re-check, not as facts (HRX-like E691: a fusion that shrank the grid lost 23 %).

**Serving vs isolated gaps (E128-E141: fourteen experiments to find a cause the bench hid by construction).**
- **Judge the gap against the binary serving runs, on serving-shaped inputs.** loom-serve compiles launch-site
  specializations that loom-benchmark never runs (E132 inferred "same binary"; E138 dumped six others). Dump them from the
  live process (common/serve_code_dump.py) and time them outside Loom through HIP (a small harness per kernel). Comparing served time
  with the bench's generic compile overstated a fix (E139). Show a gap on the served binary before calling it upstream's.
- **Bench inputs sample serving's distribution, not one canonical case**: positions across a block's offsets, chunk
  starts after a prefix-cache resume, live-row counts. Report each case and the served-weighted mix. Every bench put the 8
  verify tokens block-aligned; 7 of 16 served steps straddle two blocks and cost 1.8x (E141). Boundary cases apply to timing,
  not only to correctness.
- **A candidate is refuted only when its predicted share of the served absolute number is small.** "kvh still gains with
  it" answers a different question: E133 tested straddling that way, called it refuted, and it was the main cause. A
  model that cannot fit the served numbers (E133: base implied 20 % straddling, kvh 56 %) is the lead, not a footnote.
- **An ablation changes one ingredient; when it changes two, split it before attributing** (E139's nomap removed the
  map reads and the paged addresses at once; E140's mapident showed the reads were cheap).

## 3. Survey prior art at the diagnosis step

Before designing a fix (not after a refuted try), ask how other engines met the same bottleneck. Sources,
all of them every time: llama.cpp HIP and Vulkan, Strata (github.com/Niko1221/Strata), DwarfStar, strix-halo-engine-lab
(github.com/hyukudan/strix-halo-engine-lab, Strix-specific), strix-llama.cpp (github.com/halo-box/strix-llama.cpp, MIT,
Strix-specific llama.cpp fork with measured gfx1151 findings), Gufo (github.com/gufo-org/gufo, MIT, Strix-only engine: ROCm
kernels for Qwen3.8-27B, Flash Next and DeepSeek-V4 Flash, serves Loom packages too; its prefill figures differ across harnesses, so
compare it on equal work and power), TensorFold (github.com/ashhart/TensorFold, Apache-2.0: CUDA, Metal
and a HIP backend with kernel cost tables measured on gfx1151 (zig/src/hip/tuning/gfx1151.zon); GDN, attention and drafter paths for a
Qwen3.8 hybrid), Unsloth (github.com/unslothai/unsloth: the library is Apache-2.0, its Studio is AGPL-3.0, so ideas only from studio/;
Studio serves GGUFs through llama.cpp with per-model launch tuning, e.g. micro-batch and MoE expert placement for Inkling 2), vLLM and
SGLang, and loom-fun/loom-skills' knowledge base.
Read a closed-licence engine for technique notes only, never derived code. Give each surveyor the measured diagnosis
(ablation numbers, register/LDS budget) and ask for file:line evidence plus any measured numbers; spot-check the key
claims yourself before relying on them. Record the table in the design note. Surveys have repeatedly found the decisive
detail: exactness rules (E38), layout choices (8-bit KV), and what nobody has solved (E53).
A comparative claim (theirs vs ours) is checked on both sides before it reaches the user: E188 relayed 'our GEMM moves ~40 %
more bytes than Strata's' after checking only Strata's file; our generator's docstring had the same per-wave tile (E190).

## 4. Correctness and exactness

- **Every reader of shared data sees identical values.** Decode, verify and prefill kernels that read the same cache must
  agree bit for bit, or concurrency and speculative-decoding gates fail (E72: one reader rounded through f16, one did not).
  When you change a format, list every reader and writer first (E70, E74) and change them together.
- **Red first, mutate every guard, test the boundaries.** New kernel cases must fail on the old code and on a deliberate
  break (E4, E10); every threshold or mode switch gets a case straddling it (E45). Keep the unchanged path's tests
  byte-identical as a regression check.
- **Text rewrites of generated code are contracts.** A regex pass that finds code by its exact shape silently skips new
  shapes; make the build fail when a site is left half-converted, and prove that guard with a mutant (E31). Make passes
  terminate and idempotent (E76: a move pass swapped two blocks forever).
- **Shared helpers carry every caller's assumptions** (E27: a target-only rewrite reached the drafter). Scope rewrites.
- **Drafter changes are invisible to greedy-equality checks** (E27, D1): judge them on acceptance and ms per accepted token.
  The same holds for changes meant to be exact: compare the reference's and the candidate's draft counters on the same prompts
  (common/compare_outputs.py --save records them), not only the outputs. E170's padding overran a drafter buffer: outputs
  identical 6/6, drafts accepted 0 at 8k+; only the served step count showed it.
- **Before attributing output behaviour to sampling (or to its absence), read the effective chain**: the client's
  fields, the proxy's profile, the package default (kit build.py --temperature/--top-p/--top-k, greedy
  unless set). The 11-20 trial ran greedy while E104 called its loop a sampling effect (E112), and a replay without the
  client's injected text is not the live input (E104).
- **Quality-affecting formats need a quality gate on long context** (agreement and needles at 26k-200k, E75), not only
  short prompts. Precision changes go live only by the owner's decision.
- **Judge a non-exact change by its divergence against the reference's own noise, never an absolute bar** (strix-llama.cpp review
  practice): the candidate's mean KL against the reference counts only beyond the reference's KL against itself under a change that
  should be harmless (the same weights at another verify width or batch: t1 vs t8). GGUF-level changes: common/quant_eval.sh (llama.cpp
  --kl-divergence); a kernel change needs the serving engine's logits. A pass still needs the long-context gate and the owner.

- **A switch is a test fixture, not a product.** An env switch in a generator exists to A/B one change, and it leaves when its
  verdict lands: adopted (live, or exact and not slower) -> make the on path unconditional, delete the off path and the switch, and
  rebase the unchanged-path tests on every kit the generator serves; refuted -> delete its code (the ledger row keeps the commit for
  reference). A switch stays only while its A/B is open or a kit's recipe needs the off path, and its comment says which. Every off
  path is a branch each later generator change, compile and kernel test must carry: the 11-25 census found 220 env names read under
  kit/, with 9 switches on top of live's recipe (HRX-like E497 follow-up).

## 5. Measurement hygiene

- Compare within one interleaved A/B (ABBA) on one server; absolute numbers drift between sessions (the same package measured 21.5 and 26-34 tok/s on two harnesses).
- An A/B verdict needs common/agent_ab.sh's paired bootstrap CI to exclude 1 and its neighbour rule to ACCEPT (the candidate beats the
  reference in both ABBA halves; prefill and TTFT no more than 3 % worse); a looping output (8-gram repeat share >= 0.5) is not a speed
  result (common/checks.py, E212; from sybil-solutions/inference-optimisation-skill: a 117 tok/s 'record' there was a list stuck in a loop).
- Decide sub-noise changes on device time per verify step, not tokens/s (sub-noise gains tie in a 5-round A/B; the per-kernel profile shows them in minutes).
- **An A/A floor counts server starts, not turns.** A floor from one start per arm (prefill 0.2 %) called an identical copy's
  prefill 6.1 % slower on another night (1.061 [1.031, 1.089]; the swapped-label rerun read 1.004), and the same package's late long
  turns moved +6.9 % decode / -18 % prefill between two starts of different memory state (HRX-like E473 / E475). Measure the floor over
  >= 4 starts per arm; until then agent_ab's prefill floor is 7 % and prefill kernels are judged on served device time (events_ab).
- **A shape-dependent switch owns a shape region only where it wins on every measured shape in it, with >= 3 shapes per region**
  (strix-llama.cpp's dispatch tables): a threshold set from one shape hands the shapes beside it to a variant that loses there
  (HRX-like E11 / E12: a row threshold extended to every format was rejected).
- Ablate with realistic data on a power-limited chip; constant operands raise clocks (E15 vs E24).
- **At a power cap the clock follows temperature, and a compute-bound kernel's time follows the clock.** At 85 W the prefill clock
  read 2.22 / 2.20 / 2.17 / 2.11 GHz at 74 / 78 / 83 / 89 C across four server starts, while the GEMM lane's cycles stayed at
  239-241 G (+-0.5 %): its seconds moved 5 % with nothing changed. A latency-bound kernel (flash2, 126-136 G cycles) does not scale
  this way. Compare a compute-bound lane across starts in cycles (device time x the run's busy clock), record temperature beside every
  run, and check a 'memory state' explanation against temperature before believing it (HRX-like E482: the 'fragmented' run was also
  the hottest). The 0.1 s telemetry cannot attribute clock to lanes that interleave faster than that.
- An isolated kernel A/B is a hypothesis until the harness reproduces the live kernel's served time at the served shape (same
  kernel timed both ways). A flash2 trim measured 0.93-0.96 in isolation at every chunk and context, and 1.039 served in every
  decile of an 81k prefill (HRX-like E313 / E316); scattered pages explained part of it. A candidate goes forward on the served
  events A/B (device time per lane) only.
- Time kernels at the power level the server runs at (auto), never at profile_standard (the level for counters): at
  profile_standard decode matvecs turned compute-bound and two rewrites looked 2x faster (54 -> 102 GB/s); at auto both ran
  at 196-211 GB/s and the gain was 3-8 %, which is what serving showed. Before a harness figure feeds a decision, check that it
  reproduces the kernel's served bandwidth (common/hsaco_run.py prints the level; HRX-like E300 / E302).
- Wait for memory to be released between server runs (common/wait_quiet.sh, E69); the first request after a
  restart is cold.
- Judge agent-facing changes at agent context lengths (50-130k) and on the replay, not on 2-16k prompts (E34). Use a real
  long agent session frozen with common/agent_bench/freeze.py (ours: a 56-turn review, 34k -> 114k, E183) through
  common/agent_ab.sh (ABBA halves, served with the kit's flags, the model card's sampling, a fixed seed).
- Read the per-model label on shared /metrics (E22).
- A shared box (other agents and builds; load ~13 on 32 threads seen). CPU load competes with the GPU for the
  one DRAM bus, so it moves device times too. Trust interleaved comparisons (round-robin benches, ABBA halves of a
  replay) over back-to-back absolute numbers, and keep the load line wait_quiet.sh records with every result.
  Interleaving cancels CPU load, not another GPU client: no timing while another server is serving (E100).

## 6. Loom on this chip (device facts: devices/gfx1151.md)

- No hardware async copy on gfx11: prefetch is registers only, and the allocator decides whether it survives (#3409).
  VALU and WMMA do not overlap; dequant VALU in a GEMM tile competes with the MMAs (256-row IQ4_XS tiles lost 84 % to dequant VALU, E10).
- Spills and global loads share `vmcnt`: any spill drains the prefetch.
- Loom drives /dev/kfd itself (no HSA runtime), and 0.2.0 has no GPU hardware counters: rocprofv3 sees no served kernel. Count on the
  served code objects launched through HIP, under rocprofv3 (devices/gfx1151.md: install and power level; E209).
- Loom elastic backing needs power-of-two text-cache pages (E73-E74: 16,640-B pages faulted the GPU; 16,384-B pages
  pass). Fixed backing (`--pool_backing=fixed`) ties elastic on speed here (E69).
- loom-serve needs unlimited memlock (io_uring pinning; E1).
- Check freshness at the start of a session and before blaming the SDK or the model (`common/fresh.py KIT`: hrx-loom on
  PyPI, each pinned file at its repository's main): we ran two weeks behind hrx-loom once, and two months behind Unsloth's UD3.1
  re-upload (E113). Read upstream source at the installed tag (E105 read v0.1.0 source against a
  0.2.0 server). hrx-loom 0.3.0 (11-21) went unnoticed for a day when the check was skipped.
- Loom's prefix cache on a hybrid (GDN) model: a row cannot roll back, so reuse below the tip needs a checkpoint;
  automatic `prefix_cache_capacity` is 2 and a full table refuses deeper captures, so agent sessions fell back to their
  first prefill's 32,768 (E105). Packages set it explicitly (8: +1.9 GB, E106).
- Budget for SDK bugs on paths published ports never took; timebox and park with a compiler packet (Loom agent-driven-kernel-development.md, 'Reduce compiler blockers to standalone packets':
  minimized .loom, exact command, expected vs observed, provenance, report or ISA), checked against upstream main and existing issues,
  and a draft only until the owner approves posting (the published 27B port used the proposal-head copy with MTP only; DFlash plus our kernels hit a parser bug).
- **Always read the SDK and its docs first**. Before designing around a compiler or runtime limit, and
  always before filing an upstream issue or feature request, search the installed tag's docs (loom/docs/src: guide,
  workflows), op definitions (loom/py/loom/dialect) and source for the capability: `git grep` in an hrx-system checkout at the
  installed tag. ROCm/hrx-system#3491 asked for a scheduling fence that 0.2.0 already documents
  (`scf.schedule.fence`, docs/workflows/tune-loop-schedules.md); we had built a NaN-fragile workaround instead (E124, E144). Search our own generator too: it already used the fence.
