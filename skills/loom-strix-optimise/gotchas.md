# Gotchas (each cost a session; E-numbers point into the authors' log)

## Measuring
- **The profile label is not the served size**: "dec100k" served 81,432 tokens (E128/E133). Cite `prompt_tokens`
  (common/context_label_check.py; events_summary reads it and refuses without it).
- **Means hide populations**: vsplit 2.32 ms mean = 1.62 ms aligned + 3.01 ms straddling, 50/50 (E143). Use `--split`.
- **Benches sample one input; serving samples a distribution**: every vsplit bench put the verify tokens block-aligned;
  half of served steps straddle (E141/E143). Chunk starts, offsets, live-row counts: report the served-weighted mix.
- **Constant-filled K/V reads ~10 % fast** (E116, E143: 1.45 vs 1.58 ms aligned). Benches judged against serving use
  random data.
- **loom-serve runs launch-site specializations loom-benchmark never compiles** (E138/E142): a generic compile can be
  0.3 ms off in decode attention and flips prefetch verdicts. Time the served binary (common/serve_code_dump.py + a HIP harness).
- **Prefill epochs run decode-looking kernels** (drafter attention, matvecs): attribute per submission, not per name
  (events_summary does; per-name it read the drafter at 3.6x its per-step cost).
- **Floors drift with the model file**: a weight-bytes floor from an earlier quant was off by 8 % (18.3 vs 16.93 GB).
  Recompute constants from the served GGUF (gguf.GGUFReader) when the quant changes.
- **Two head matvecs per verify step, two owners** (Qwen3.8 kit): g_head_mv_t8_q8 is @llm_verify on the target Q6_K head;
  g_head_mv_t8_q8_p40 is @llm_dflash_propose on the drafter's q4_0 head copy (715 MB). Attribute a kernel by the command
  that launches it, not by its name.

## Kernels and the SDK
- **Read the SDK and our own kit before a workaround or an upstream request** (E145): the fence existed in 0.2.0 docs
  and in our own generator while we shipped a NaN-fragile pin and filed an upstream feature request for it.
- **Loom places a pending load's consumer at its earliest legal point**; source order is not kept (E110). Keep a consumer
  late with `scf.schedule.fence`; check with common/load_events.py / load_cover.py that the wait moved.
- **A data-dependency pin must hold for every value, NaN included**: dead padding columns read stale workspace (E123).
- **A loop-carried converted value drags its loads' wait into the issuing iteration** (E109): carry raw codes.
- **Spills and global loads share vmcnt**: any spill drains the prefetch (gfx11).
- **I2: a token's result must not depend on its span** (alone == in span, atol 0). Partitions (split-K boundaries) must
  depend on the token's position only (E144: absolute key splits sized by a 4096-position bucket).
- **Generated-text rewrites are contracts**: every anchor exactly once, build fails otherwise (kit/kv8_late.py, gen_flash2s.py
  FLASH2_SPLITK_ANY), proven with a mutant.

## Running
- **The harness must serve as production serves**: a harness default ran ELASTIC backing while production ran
  `--pool_backing=fixed` (common/serve_as.sh takes the kit's flags). KV8 corrupts on elastic only (E146: a day of 'live bug' hunting was a harness configuration).
- **Never `pgrep -f`**; waiters verify the PID's exact args (common/wait_pid.sh). An awk field error gave an empty PID
  and a waiter that returned at once; a reaped-late child is a zombie that `kill -0` still sees.
- **No build or second server beside a profile** (GEMMs 123 -> 131 s). common/build.sh refuses during one.
- **loom-serve needs unlimited memlock**; elastic backing needs power-of-two pages (E73/E74).
- **Restore the production server on every exit path** (trap; LSK_RESTORE_HOOK); a manual run once left it down ~20 min.
