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
- **latency**: a timing ablation that removes the loads (zeros staged instead) and one that keeps them cache-resident.
  L18/E52: prefill attention ran 0.57x without K/V loads and 0.85x with cache-resident loads: latency, not bandwidth.
- **registers**: final VGPRs and the occupancy tier from the compile report; spills share the load counter on gfx11.

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
- **Count a loop from its backward branch with a script (common/loop_census.py, compiler_ab.py), never a hand-cut line range, and place
  each run of moves in its block (branch join, back-edge, rare path) before naming the mechanism** (E168 -> E179: "264
  v_mov, a back-edge copy" came from a range wider than the loop; the copies sat at branch joins).
- **Compile a generator change as the served program emits it** (E230): splice it into the served program (recipe switches,
  post-passes) before counting. The IQ4 lookup rewrite compiled clean from the bare generator and spilled 2.4 KB under the
  served recipe's padding-skip `scf.if`.
- **Check a model against the measured rows it should also explain before naming it a cause** (E233 -> E234): 'q6_k is slow
  because its last round is part-empty' was written down, then the same model put 14-29 % idle on kernels measured at the
  bandwidth ceiling. A model that mispredicts the controls explains nothing.
- **Ablation**: timing-only builds that remove one ingredient (loads, barriers, dequant, a scale read) rank levers
  before any rewrite (E15, E52). Results are garbage by design; label them so.
- **Profile per kernel, both sides of a loss** (E10: an average hid +84 % in one kernel and -11/-23 % in two). Split a
  shared kernel name by call site before attributing time (E13). Attribute by call shape, not by name (E43).
- **Price the lever before building it** (E15/E24): if the ablation's best case is below the A/B noise, do not build.
- **Call a cause a hypothesis until a count or an ablation shows it** (E76-E77: two causes stated as likely, both
  refuted by the next measurement). Report what was measured, what is suspected, and the test that will tell. A source
  read is a hypothesis too: follow the value from its producer to the binding the kernel reads before naming its memory,
  and title a row "root cause" only after the direct test (E139 called the page map host-private from its allocation
  site; serving uploads it into a device buffer, E140).
- **Cite the served context, not the profile's label** (E128/E133: "dec100k" served 81,432 tokens and reversed a
  framing). common/events_ab.sh prints it on a `###` line (common/context_label_check.py, exit 1 on a >5 % mismatch).
- **Measure on a quiet machine**: no build, compile or second server beside a profile (a CPU build during a profile moved GEMM
  time 123 -> 131 s). common/wait_quiet.sh waits builds out; common/build.sh refuses during a measurement.

- **Search before building a tool**: Loom's own tools and docs workflows, installed packages, the ROCm and llama.cpp builds
  (tools.md lists them by step). In one session a remote safetensors reader, a hand bank-conflict analysis and a KL-divergence
  path were each about to be written while huggingface_hub, Loom's compile report and llama-perplexity already did them.
- **Plateau rule**: after three levers in a row land inside noise, stop and write a ranked status of the remaining levers, each with
  its price and the evidence behind it, before starting a fourth (E212, from sybil-solutions/inference-optimisation-skill).
- **Never park an experiment without its root cause**. "Slower, unexplained" or "held back by X" is not a
  verdict: dig until the mechanism is shown (a counter, an ablation, the disassembly line that waits), then decide. An
  experiment may wait for GPU time, but its log entry names the open question and the next probe, and it is
  investigated before the lane moves on.

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
(github.com/hyukudan/strix-halo-engine-lab, Strix-specific), vLLM and SGLang, and loom-fun/loom-skills' knowledge base.
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

## 5. Measurement hygiene

- Compare within one interleaved A/B (ABBA) on one server; absolute numbers drift between sessions (the same package measured 21.5 and 26-34 tok/s on two harnesses).
- An A/B verdict needs common/agent_ab.sh's paired bootstrap CI to exclude 1 and its neighbour rule to ACCEPT (the candidate beats the
  reference in both ABBA halves; prefill and TTFT no more than 3 % worse); a looping output (8-gram repeat share >= 0.5) is not a speed
  result (common/checks.py, E212; from sybil-solutions/inference-optimisation-skill: a 117 tok/s 'record' there was a list stuck in a loop).
- Decide sub-noise changes on device time per verify step, not tokens/s (sub-noise gains tie in a 5-round A/B; the per-kernel profile shows them in minutes).
- Ablate with realistic data on a power-limited chip; constant operands raise clocks (E15 vs E24).
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
- Budget for SDK bugs on paths published ports never took; timebox and park with a reproducer (the published 27B port used the proposal-head copy with MTP only; DFlash plus our kernels hit a parser bug).
- **Always read the SDK and its docs first**. Before designing around a compiler or runtime limit, and
  always before filing an upstream issue or feature request, search the installed tag's docs (loom/docs/src: guide,
  workflows), op definitions (loom/py/loom/dialect) and source for the capability: `git grep` in an hrx-system checkout at the
  installed tag. ROCm/hrx-system#3491 asked for a scheduling fence that 0.2.0 already documents
  (`scf.schedule.fence`, docs/workflows/tune-loop-schedules.md); we had built a NaN-fragile workaround instead (E124, E144). Search our own generator too: it already used the fence.
