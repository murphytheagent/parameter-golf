# Parameter Golf Tracker

Last updated: 2026-04-03 08:01 UTC

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
- Static recurrence size check at init:
  - `d=512` total about `1.77 MB`
  - `d=576` total about `2.15 MB`
  - `d=640` total about `2.61 MB`
  - inference: the current recurrence line is still underusing the byte budget
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

- No reproduced local baseline exists yet from the current surviving fork state.
- No recurrence, `MTP-lite`, AttnRes-lite, or bounded test-time adaptation run has been executed from the current fork state.
- There is still no Murphy-run baseline anchor for any "beats baseline" claim.
- The first working runtime surface is now lean idea-testing rather than `8`-GPU reproduction:
  - the earlier queued jobs `1989` and `1990` were intentionally canceled after the collaborator redirected the first stage
  - `1993` validated the warmed substrate and `2000` produced the first architecture comparison
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
5. Next cheap sweep
   - Run `rec_u3_r3_d576_kv4_mlp2_smoke1gpu`.
   - Run `rec_u3_r3_d640_kv4_mlp2_smoke1gpu`.
   - Do not call the family dead until one of the wider runs has been checked; the `d=512` line is too far under the byte cap.
6. Baseline comparison gate
   - Reproduce the published `9 x 512` run from this fork on the full challenge surface before claiming any baseline improvement.
   - Treat that later run as the local comparison anchor even if it differs slightly from the published record.
7. First recurrence sweep
   - `baseline_9x512_kv4_mlp2`
   - `rec_u3_r3_d512_kv4_mlp2`
   - `rec_u3_r3_d576_kv4_mlp2`
   - `rec_u3_r3_d640_kv4_mlp2`
8. Post-sweep decision rule
   - Positive: any recurrence run beats the local baseline by `>= 0.002` quantized `val_bpb`
   - Negative: all recurrence runs lose by `>= 0.010`
   - Ambiguous: keep one stable recurrence anchor and run exactly one cheap `KV4 -> KV2` reallocation before switching families

## Experiment Ledger

| Run ID | Family | Shape | Objective | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| `baseline_sp1024_smoke_1gpu_lean` | smoke | `9 x 512`, `KV4`, `MLP2`, `1` GPU, `2` CPUs, `12G`, `TRAIN_BATCH_TOKENS=32768`, `120s` | validate repo/data/env/logging path on the warmed substrate with the collaborator's minimal-resource constraint | completed (`1993`) | exact `val_bpb=1.57745762`, `1111` steps, `12,697,534` bytes |
| `baseline_9x512_kv4_mlp2` | baseline | `9 x 512`, `KV4`, `MLP2` | reproduce local anchor under the full challenge contract before claiming improvement | pending | later comparison gate, no longer the first runtime action |
| `rec_u3_r3_d512_kv4_mlp2_smoke1gpu` | recurrence-smoke | `3` unique blocks, `3` repeats, `d=512`, `KV4`, `MLP2`, lean `1`-GPU envelope | first architecture comparison on the cheap validated surface | completed (`2000`) | exact `val_bpb=1.60109017`; worse than baseline smoke by `0.0236`, but faster and much smaller |
| `rec_u3_r3_d576_kv4_mlp2_smoke1gpu` | recurrence-smoke | `3` unique blocks, `3` repeats, `d=576`, `KV4`, `MLP2`, lean `1`-GPU envelope | widen the recurrence line toward the byte budget | pending | next cheap recurrence test |
| `rec_u3_r3_d640_kv4_mlp2_smoke1gpu` | recurrence-smoke | `3` unique blocks, `3` repeats, `d=640`, `KV4`, `MLP2`, lean `1`-GPU envelope | push the same family closer to the byte ceiling | pending | likely the strongest cheap follow-up |
| `rec_u3_r3_d512_kv4_mlp2` | recurrence | `3` unique blocks, `3` repeats, `d=512`, `KV4`, `MLP2` | cheapest recurrence test against same width as baseline | pending | first recurrence anchor after baseline verdict |
| `rec_u3_r3_d576_kv4_mlp2` | recurrence | `3` unique blocks, `3` repeats, `d=576`, `KV4`, `MLP2` | test whether width reallocation helps recurrence under cap | pending | keep exporter unchanged |
| `rec_u3_r3_d640_kv4_mlp2` | recurrence | `3` unique blocks, `3` repeats, `d=640`, `KV4`, `MLP2` | push depth-per-byte line near cap | pending | only live if dry size check stays under cap |

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

1. Cheap `KV4 -> KV2` reallocation on the best stable recurrence anchor if recurrence is neutral but not dead
2. `MTP-lite` on the winning stem, or on the baseline stem if recurrence fails cleanly
3. Bounded test-time adaptation or streaming memory only if training-side changes stall
4. AttnRes-lite only as a later falsification branch for depth-state mixing
