# Parameter Golf Tracker

Last updated: 2026-04-03 09:27 UTC

For the self-contained research and implementation plan, read `PLAN.md`. The collaborator-facing PDF render lives under `outputs/plan_report/parameter_golf_plan.pdf`. This file is the compact state-of-record and run ledger.

## State Of Record

### Proved

- Public 10-minute baseline:
  - architecture: `9 x 512`, tied embeddings, `8` query heads, `4` KV heads, `MLP_MULT=2`
  - final quantized metric: `final_int8_zlib_roundtrip_exact val_bpb:1.22436570`
  - total submission bytes: `15,863,489`
  - code bytes: `47,642`
  - timed stop: `13,780` steps in `600.038` seconds
- Unlimited-compute reference on the same stem:
  - `1.20737944` after `4` hours, so the objective still has real headroom without changing the tokenizer
- Remote run surface:
  - `/data/scratch/murphy/parameter-golf` now exists on the node at `origin/main` commit `08ee8ba`
  - the full published `fineweb10B_sp1024` dataset plus `fineweb_1024_bpe.model` tokenizer are materialized under that checkout
  - the working CUDA env is `/home/murphy/miniforge3/envs/swe311`
  - run artifacts now land under `outputs/run_jobs/<run_id>/`
- Execution-stage rule:
  - idea-testing and code-testing may use `1` GPU with minimal CPU and host-memory requests
  - full `8`-GPU runs are reserved for later challenge-honest comparison, not for the first code-path validation
- Lean idea-testing anchor:
  - job `1993` completed cleanly on `1` GPU / `2` CPUs / `12G`
  - `final_int8_zlib_roundtrip_exact val_bpb:1.57745762`
  - stopped at `1111` steps in `120.101s` with `step_avg:108.10ms`
  - peak GPU memory `1335 MiB allocated / 1678 MiB reserved`
  - total int8+zlib submission size `12,697,534` bytes
- First recurrence smoke:
  - job `2000` ran `NUM_UNIQUE_LAYERS=3` with logical depth `9`
  - `final_int8_zlib_roundtrip_exact val_bpb:1.60109017`
  - stopped at `1454` steps in `120.007s` with `step_avg:82.54ms`
  - peak GPU memory `1173 MiB allocated / 1448 MiB reserved`
  - total int8+zlib submission size `4,963,934` bytes
  - delta versus the lean baseline smoke: `+0.02363255` worse on `val_bpb`
- Widened recurrence smoke:
  - job `2003` (`rec_u3_r3_d576_kv4_mlp2_smoke1gpu`) completed cleanly
  - `final_int8_zlib_roundtrip_exact val_bpb:1.59122937`
  - stopped at `1380` steps in `120.075s` with `step_avg:87.01ms`
  - peak GPU memory `1025 MiB allocated / 1650 MiB reserved`
  - total int8+zlib submission size `6,069,746` bytes
  - delta versus the lean baseline smoke: `+0.01377175` worse on `val_bpb`
  - delta versus the first recurrence smoke: `-0.00986080` better on `val_bpb`
- Best cheap-stage recurrence anchor so far:
  - job `2004` (`rec_u3_r3_d640_kv4_mlp2_smoke1gpu`) completed cleanly
  - `final_int8_zlib_roundtrip_exact val_bpb:1.58541212`
  - stopped at `1348` steps in `120.039s` with `step_avg:89.05ms`
  - peak GPU memory `1291 MiB allocated / 2072 MiB reserved`
  - total int8+zlib submission size `7,265,478` bytes
  - delta versus the lean baseline smoke: `+0.00795450` worse on `val_bpb`
  - delta versus the first recurrence smoke: `-0.01567805` better on `val_bpb`
- Cheap KV-head reallocation check:
  - job `2007` (`rec_u3_r3_d640_kv2_mlp2_smoke1gpu`) completed cleanly
  - `final_int8_zlib_roundtrip_exact val_bpb:1.59932364`
  - stopped at `1261` steps in `120.016s` with `step_avg:95.18ms`
  - peak GPU memory `1283 MiB allocated / 1858 MiB reserved`
  - total int8+zlib submission size `6,602,540` bytes
  - delta versus the `d=640, KV4` anchor: `+0.01391152` worse on `val_bpb`
  - delta versus the lean baseline smoke: `+0.02186602` worse on `val_bpb`
- Full-contract baseline gate:
  - job `2008` (`baseline_sp1024_localcheck`) is queued on `8` GPUs
  - current external blockers on `wth-gpu-01`: job `1965` holds `1` GPU until `2026-04-03 23:08 UTC`, and job `2002` holds `1` GPU until `2026-04-03 12:28 UTC`
  - consequence: the honest baseline is queued, but it cannot start until the node clears
- Static recurrence size check at init:
  - `d=512` total about `1.77 MB`
  - `d=576` total about `2.15 MB`
  - `d=640` total about `2.61 MB`
  - inference: the current recurrence line is still underusing the byte budget
- Cheap-stage verdict on recurrence:
  - widening `512 -> 576 -> 640` improved quantized `val_bpb` monotonically from `1.60109017 -> 1.59122937 -> 1.58541212`
  - recurrence no longer looks "dead"; it looks neutral-but-live, with `d=640, KV4` as the current cheap anchor
  - the `KV4 -> KV2` reallocation failed cleanly, so there is no more cheap ambiguity left in this line
  - the next honest object is the queued `8`-GPU baseline, then the promoted `d=640, KV4` recurrence run
- Athena-backed ranking for this repo and constraint set:
  - `1)` shared-depth recurrence / aggressive tying
  - `2)` bounded test-time adaptation / streaming memory
  - `3)` `MTP-lite`
  - `MQA/GQA` is a cheap supporting sweep
  - deeper MLA / latent-KV and stored low-rank factorization stay behind that under the current exporter
  - small float tensors with `<= 65,536` elements stay in fp16 passthrough under the current int8 export path, so naive stored low-rank compression is not obviously byte-positive here
- AttnRes verdict:
  - do not move full Kimi Attention Residuals or Block AttnRes ahead of recurrence or `MTP-lite`
  - if the idea gets touched later, only steal a tiny depth-mix component

### Unproved

- No reproduced local full-contract baseline exists yet from the current surviving fork state.
- No `MTP-lite`, AttnRes-lite, or bounded test-time adaptation run has been executed from the current fork state.
- No recurrence run has beaten even the lean baseline smoke yet, and there is still no Murphy-run full-contract baseline anchor for any "beats baseline" claim.
- The first working runtime surface is now lean idea-testing rather than `8`-GPU reproduction:
  - the earlier queued jobs `1989` and `1990` were intentionally canceled after the collaborator redirected the first stage
  - `1993` validated the warmed substrate and jobs `2000`, `2003`, `2004`, and `2007` completed the cheap recurrence selection path
  - `/data` still has only `219G` free of `7.0T`, so scratch cleanup will matter once later runs start accumulating

## Exact Next Execution Path

1. Remote substrate
   - Done: `/data/scratch/murphy/parameter-golf` is the working checkout, caches point at `/data/scratch/murphy/cache`, and the SP1024 dataset/tokenizer are on disk.
2. Idea-testing surface
   - Done: lean smoke job `1993` validated the warmed path and artifact capture.
3. First code change
   - Done: added shared-depth recurrence with `NUM_UNIQUE_LAYERS` while keeping the existing tokenizer, exporter, and optimizer split intact.
4. First lean recurrence smoke
   - Done: `rec_u3_r3_d512_kv4_mlp2_smoke1gpu`.
5. Width-selection sweep
   - Done: `rec_u3_r3_d576_kv4_mlp2_smoke1gpu`.
   - Done: `rec_u3_r3_d640_kv4_mlp2_smoke1gpu`.
   - Outcome: promote `d=640` as the cheap recurrence anchor; do not spend more time on `d=512` or `d=576` unless the promoted line becomes unstable.
6. Cheap KV-head falsification
   - Done: `rec_u3_r3_d640_kv2_mlp2_smoke1gpu`.
   - Outcome: worse than the `d=640, KV4` anchor on both loss and achieved steps; drop the `KV2` branch.
7. Baseline comparison gate
   - Reproduce the published `9 x 512` run from this fork on the full challenge surface before claiming any baseline improvement.
   - Treat that later run as the local comparison anchor even if it differs slightly from the published record.
   - Current queue state: job `2008` is pending on resources behind external jobs `1965` and `2002`.
8. Promoted full-contract recurrence sweep
   - `baseline_9x512_kv4_mlp2`
   - `rec_u3_r3_d640_kv4_mlp2`
9. Post-sweep decision rule
   - Positive: any recurrence run beats the local baseline by `>= 0.002` quantized `val_bpb`
   - Negative: all recurrence runs lose by `>= 0.010`
   - Ambiguous: keep the `d=640, KV4` recurrence anchor and move the next budget to `MTP-lite` rather than reopening the dead `KV2` branch

## Experiment Ledger

| Run ID | Family | Shape | Objective | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| `baseline_sp1024_smoke_1gpu_lean` | smoke | `9 x 512`, `KV4`, `MLP2`, `1` GPU, `2` CPUs, `12G`, `TRAIN_BATCH_TOKENS=32768`, `120s` | validate repo/data/env/logging path on the warmed substrate with the collaborator's minimal-resource constraint | completed (`1993`) | exact `val_bpb=1.57745762`, `1111` steps, `12,697,534` bytes |
| `baseline_9x512_kv4_mlp2` | baseline | `9 x 512`, `KV4`, `MLP2` | reproduce local anchor under the full challenge contract before claiming improvement | queued (`2008`) | run is pending on resources behind external jobs `1965` and `2002` |
| `rec_u3_r3_d512_kv4_mlp2_smoke1gpu` | recurrence-smoke | `3` unique blocks, `3` repeats, `d=512`, `KV4`, `MLP2`, lean `1`-GPU envelope | first architecture comparison on the cheap validated surface | completed (`2000`) | exact `val_bpb=1.60109017`; worse than baseline smoke by `0.0236`, but faster and much smaller |
| `rec_u3_r3_d576_kv4_mlp2_smoke1gpu` | recurrence-smoke | `3` unique blocks, `3` repeats, `d=576`, `KV4`, `MLP2`, lean `1`-GPU envelope | widen the recurrence line toward the byte budget | completed (`2003`) | exact `val_bpb=1.59122937`; gap to baseline smoke shrank to `0.0138`; `6,069,746` bytes |
| `rec_u3_r3_d640_kv4_mlp2_smoke1gpu` | recurrence-smoke | `3` unique blocks, `3` repeats, `d=640`, `KV4`, `MLP2`, lean `1`-GPU envelope | push the same family closer to the byte ceiling | completed (`2004`) | exact `val_bpb=1.58541212`; current best cheap recurrence anchor; `7,265,478` bytes |
| `rec_u3_r3_d640_kv2_mlp2_smoke1gpu` | recurrence-smoke | `3` unique blocks, `3` repeats, `d=640`, `KV2`, `MLP2`, lean `1`-GPU envelope | test cheap KV-head reallocation on the promoted recurrence anchor | completed (`2007`) | exact `val_bpb=1.59932364`; slower and worse than the `d=640, KV4` anchor |
| `rec_u3_r3_d512_kv4_mlp2` | recurrence | `3` unique blocks, `3` repeats, `d=512`, `KV4`, `MLP2` | cheapest recurrence test against same width as baseline | deferred | cheap-stage smoke lost too cleanly to justify a full-contract run first |
| `rec_u3_r3_d576_kv4_mlp2` | recurrence | `3` unique blocks, `3` repeats, `d=576`, `KV4`, `MLP2` | test whether width reallocation helps recurrence under cap | deferred | cheap-stage result improved, but `d=640` dominates it on the same family |
| `rec_u3_r3_d640_kv4_mlp2` | recurrence | `3` unique blocks, `3` repeats, `d=640`, `KV4`, `MLP2` | promoted full-contract recurrence anchor after the honest baseline gate | pending | best cheap-stage recurrence line so far |
| `rec_u3_r3_d640_kv2_mlp2` | recurrence | `3` unique blocks, `3` repeats, `d=640`, `KV2`, `MLP2` | promote cheap KV-head reallocation if the smoke stays live | dropped | cheap-stage `KV2` check lost cleanly to `d=640, KV4` |

## Run Recording Contract

For every run, record the following in one place:

- `run_id`
- `git_rev`
- `model_shape`
- recurrence knobs or auxiliary-head knobs
- `code_bytes`
- `quantized_model_bytes`
- `total_submission_bytes`
- final quantized `val_bpb`
- `step_count`
- `step_avg_ms`
- `peak_memory`
- any divergence, NaNs, scheduler interference, or suspicious seed sensitivity

## Family Order After The Recurrence Verdict

1. Honest local baseline, then the promoted `d=640, KV4` recurrence run
2. `MTP-lite` on the winning stem, or on the baseline stem if recurrence fails cleanly
3. Bounded test-time adaptation or streaming memory only if training-side changes stall
4. AttnRes-lite only as a later falsification branch for depth-state mixing
