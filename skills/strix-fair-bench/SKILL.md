---
name: strix-fair-bench
description: Judge whether a candidate Loom package is faster, slower or tied against a reference on AMD Strix Halo, with a seeded agent bench that is fair to both - cool start per half, ABBA halves, paired per-turn confidence intervals, an A/A noise floor, GPU and CPU telemetry beside every half. Load it before any speed verdict, when a bench result looks too good or too bad, when two benches of the same package disagree, or when a profile or harness says faster and you want to know whether serving agrees.
---

# Fair benchmarking on Strix Halo

Plain markdown for any coding agent. Nothing here depends on the model: the session, the packages, the port and the model id
come from the kit (`kits/<KIT>/kit.env`), the machine env (`~/.config/loom-strix-kits/env` or `$LSK_ENV`) or an argument.
Commands run from the repository root. The speed method as a whole is skills/loom-strix-optimise. This skill is its last step:
the measurement that decides. Worked examples below cite the Qwen3.8-27B kit's ledger rows (Exx). They show the method and
are not defaults: measure your own floor.

## Why a fair bench is needed on this chip

Strix Halo decode runs pinned at its power cap (85 W, ~90 C on the reference machine, E356). The CPU and the GPU share one
DRAM system and one power budget. So the same package measured twice can differ by more than most candidates change it:

- **Heat:** a half run straight after other GPU work decodes slower than one started from a cool chip (7 %, E368).
- **Drift:** the reference itself drifts over a session (E356: 7 % between two benches of the same package hours apart).
- **CPU work:** a build or a numpy job beside a half steals bandwidth and power from the timed run (E415).
- **Device timers:** they cannot see a lower clock under the power limit, so a kernel that is faster on every profile can
  serve slower (E351-E353).
- **The software stack:** a driver, firmware or library update changes the machine under an unchanged package. In
  strix-llama.cpp #241 a Mesa update turned a measured speedup into a loss.

A verdict therefore comes from one interleaved, seeded, wall-clock run of both packages, judged against the noise of a
package against itself.

## Procedure

1. **Freeze a real agent session** at agent context lengths (common/agent_bench/freeze.py; 50-130k tokens; a short prompt
   does not show long-context costs). Fixed mode replays byte-identical prompts; the model card's sampling with a fixed seed
   makes the replies reproducible.
2. **Measure the A/A floor once per kit and machine:** `common/agent_ab.sh KIT REF REF_COPY OUT SESSION`, where REF_COPY is a
   copy of the reference package (`cp -a`, same program bytes). Its paired ratios are the noise floor. Set
   `AB_FLOOR_DECODE` / `AB_FLOOR_PREFILL` (kit or machine env) to a margin above them. The per-turn bootstrap CI excludes 1.0
   even in an A/A run (E374), so the CI alone never decides.
3. **Run the bench:** `common/agent_ab.sh KIT REF CAND OUT SESSION`. It does, for each of four runs:
   - **Cool start (E368):** stop the bench server (`AB_STOP`), wait until the GPU edge temperature is at most `T_START`
     (default 55 C) or `COOL_MAX` seconds (default 900) have passed; log the start temperature (`OUT/<tag>.start.txt`); a
     half that starts hot says so.
   - **Serve exactly as production serves** (`AB_SERVE`, default common/serve_as.sh with the kit's flags; it checks that the
     port serves the requested package). A harness default is a different configuration.
   - **Telemetry beside the half:** common/gpu_telemetry.py records busy %, power, shader clock, temperature, fabric / memory /
     SoC clocks, CPU Tctl, the load average and the CPU busy outside the bench's own server, every 0.1 s.
   - **ABBA halves:** half 1 runs REF then CAND on turns 0, 4, 8, ...; half 2 runs CAND then REF on turns 2, 6, 10, ....
     Each package is first in one half, so a run-order effect shows as a half-dependent ratio instead of a verdict.
4. **Read the summary** (`OUT/summary.txt`). Per package: decode and prefill tok/s, tokens per verify step, mean TTFT, how
   many turns produced equal output lengths. Then the paired per-turn time ratios CAND/REF with a 2,000-sample bootstrap
   95 % CI over turns, overall and per half; the start temperatures and telemetry of each half in run order; and the verdict.
5. **Compare results only from one environment.** common/env_block.py prints a fixed block of `key: value` lines: kernel,
   GPU boot parameters (amdgpu.*, ttm.*, iommu), vbios and GPU firmware versions, linux-firmware, Mesa, libdrm, ROCm, Loom,
   the kit commit (`-dirty` when a tracked file is edited), each package's model.json sha256 and build line, GPU / platform
   power settings, the device file and profile, CPU governor and EPP, and the Python versions. A value that cannot be read is
   `unknown`. agent_ab.sh records it before every half and prints it once under `environment:` in the summary. A half whose
   block differs from the first half's makes the bench `UNTRUSTED` and names the keys. events_ab.sh does the same for its
   runs (exit 2). Before you compare two results from different runs (this bench against last week's, a profile against
   a bench), run `common/env_block.py --diff A/env.txt B/env.txt`. Exit 0 means the environments match. Exit 1 prints each
   differing key with both values: name each one in the comparison and judge it, or rerun both sides in one environment.
   The package keys differ by design between benches of different candidates. A kernel, firmware, Mesa or Loom difference
   is not judged away: rerun. Two `unknown` values compare equal, so the check cannot clear a key that neither side could
   read. The reason is #241: a stack update turned a measured speedup into a loss.
6. **Telemetry is read beside the verdict, never instead of it.** A candidate at the same power and a lower clock is drawing
   more energy per step. A half at a lower fabric clock was a different machine state. A half whose start was hot or whose CPU
   was busy is not comparable.
7. **A profile can only stop a candidate, never pass it.** Device-time A/Bs, served per-dispatch events, isolated kernel
   harnesses and counters diagnose. They can stop a candidate before it reaches the bench (a lane that got slower, a kernel
   that is not exact). Only this bench sends a candidate forward. Each switch gets its own bench before it joins a package.

## Verdict line (exit 0 pass, 1 fail, 2 invalid / untrusted)

Last line of `OUT/summary.txt`:

- `BENCH: FASTER` (exit 0): at least one of decode time and prefill time is faster, and none is slower.
- `BENCH: TIE` (exit 0): neither differs. A tie keeps an exact change that does not cost speed.
- `BENCH: SLOWER` (exit 1): decode time or prefill time is slower, whatever the other does.
- `BENCH: UNTRUSTED` (exit 2): a half was CONTAMINATED by CPU work (other CPU beside the server above `AB_CPU_MAX` cores,
  default 2, in more than 10 % of its samples), or has no CPU record. Rerun on a quiet machine. Also UNTRUSTED when the
  environment block changed between halves or could not be read (procedure step 5). Rerun in one environment.
- `BENCH: INVALID` (exit 2): fewer than 3 turns with equal output lengths, or a half missing. Decode time is compared only on
  turns with equal output lengths; a candidate that changes the replies needs an exactness verdict first.
- A failed run (a server that did not come up, replay errors) exits 2 with no BENCH line.

A metric differs only when **both** hold:
- its paired 95 % CI excludes 1.0, and
- its ratio is beyond the A/A floor (`AB_FLOOR_DECODE`, default 0.01; `AB_FLOOR_PREFILL`, default 0.002: the reference machine's).

The summary also prints, for information only: the median of per-turn decode rates with its CI, the ABBA neighbour rule
(the candidate beats the reference in both halves) and loop flags (8-gram repeat share >= 0.5: a looping reply is not a speed
result). A looping turn is a finding of its own.

## Machine specifics

None are in the script. A machine with a managed server (a launcher, a systemd unit) sets them in its env or in a thin
wrapper that calls common/agent_ab.sh:

| Variable | What | Default |
|---|---|---|
| `AB_SERVE` | command prefix: `AB_SERVE PKG` serves PKG, `AB_SERVE --restore` hands the port back | `bash common/serve_as.sh KIT` |
| `AB_STOP` | stops the bench server before each cooling wait | `lsk_stop` (our servers, then `LSK_STOP_HOOK`) |
| `AB_REPLAY` | the replay script | common/agent_bench/replay.py |
| `T_START`, `COOL_MAX` | start temperature (edge, C) and the longest cooling wait (s) | 55, 900 |
| `AB_FLOOR_DECODE`, `AB_FLOOR_PREFILL` | the A/A floor as a ratio | 0.01, 0.002 |
| `AB_CPU_MAX` | other CPU, in cores, above which a sample counts toward CONTAMINATED | 2 |

## Worked examples (Qwen3.8-27B kit)

- **E368, the cool start.** Half 1 holds most of the decode time and always ran the reference first. The first half after an
  idle gap decoded 601-602 s; the same half straight after other GPU work took 642-661 s, for every package alike. Shader
  clock and edge temperature medians did not separate them; prefill was unaffected (bandwidth-bound decode, compute-bound
  prefill). Every candidate in that period was biased by ~7 % against. Fix: the thermal start gate above.
- **E356, drift and the power cap.** One switch measured -8 % decode before the gate; -2.5 % of that was the switch and the
  rest heat soak: the reference decoded 7 % slower in a later bench of the same seed and turns. Telemetry for every half: busy
  100 %, power 84-85 W (at the cap), 90-91 C. Device-time cuts can vanish at the cap; only the served bench shows it.
- **E374, the A/A floor.** The reference against a byte-identical copy, every half started at 54-55 C: paired decode time
  1.004 [1.003, 1.004], prefill 1.000 [1.000, 1.001]. The CI excluded 1.0 although nothing differed, so the rule became: CI
  excludes 1.0 AND beyond the floor (~1 % decode, ~0.2 % prefill). This bench reads `BENCH: TIE`. A switch with decode time
  1.026 [1.020, 1.031] reads `BENCH: SLOWER` (E371); one with 0.946 [0.940, 0.951] decode and 0.994 prefill reads
  `BENCH: FASTER` (E373).

## Proposal: a fair harness A/B mode for common/hsaco_run.py (not implemented)

Ideas from TensorFold's fair-bench (github.com/ashhart/skills, Apache-2.0; ideas only, no code). hsaco_run.py already
interleaves every code object once per round, after `--warmup` rounds, and checks the reference's drift (`--anchor`). It
prints ratios of medians, but no verdict. A proposed `--verdict` mode would add:

1. **Warm-up**, unscored (exists: `--warmup`).
2. **An A/A floor from the same run:** `--aa-rounds M` rounds of the reference against itself (two labels, one object), in the
   same mirrored order as the comparison. The floor is the largest A/A gap between per-round values.
3. **Mirrored rounds:** each comparison round launches A, B, B, A, and an arm's round value is the mean of its two launches,
   so a trend inside a round (clock ramp, heat) cancels instead of favouring the arm that runs first.
4. **The decision:** FASTER only if the candidate wins every round and the median gap is larger than the floor; SLOWER by the
   mirror rule; otherwise WITHIN_NOISE. Fewer than ~5 rounds is reported as too few to decide (one arm wins every one of
   few rounds by chance).
5. **Output:** `HARNESS: FASTER | SLOWER | WITHIN_NOISE (hypothesis)` with exit 0 / 1 / 0 and 2 for usage, beside the power
   level and the anchor drift. Every harness figure stays a hypothesis (E300, E313): this mode can stop a candidate (SLOWER)
   or send it to the agent bench, never send it live.
