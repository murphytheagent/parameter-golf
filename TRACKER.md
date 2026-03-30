# Parameter Golf Tracker

Last updated: 2026-03-30 10:20 UTC

For the self-contained research and implementation plan, read `PLAN.md`. This file is the compact state-of-record and run ledger.

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
- Athena-backed ranking for this repo and constraint set:
  - `1)` shared-depth recurrence / aggressive tying
  - `2)` bounded test-time adaptation / streaming memory
  - `3)` `MTP-lite`
  - `MQA/GQA` is a cheap supporting sweep
  - deeper MLA / latent-KV and stored low-rank factorization stay behind that under the current exporter
- AttnRes verdict:
  - do not move full Kimi Attention Residuals or Block AttnRes ahead of recurrence or `MTP-lite`
  - if the idea gets touched later, only steal a tiny depth-mix component

### Unproved

- No reproduced local baseline exists yet from the current surviving fork state.
- No recurrence, `MTP-lite`, AttnRes-lite, or bounded test-time adaptation run has been executed from the current fork state.
- The remote substrate is not warm:
  - on `2026-03-30 09:17 UTC`, the node was occupied by Slurm job `1892` (`flashvla`, user `zekaili`) with projected end `2026-04-01 02:29 UTC`
  - `/data/scratch/murphy` exists
  - `/data/users/murphy` does not exist
  - no ready Parameter Golf checkout or FineWeb SP1024 dataset/tokenizer cache was visible under the checked Murphy paths

## Exact Next Execution Path

1. Remote substrate
   - Use `/data/scratch/murphy/parameter-golf` as the working directory.
   - Route `HF_HOME`, Hugging Face hub, transformers, torch, and `XDG_CACHE_HOME` to `/data/scratch/murphy/cache`.
   - Materialize the `fineweb10B_sp1024` dataset and `fineweb_1024_bpe.model` on the same data volume.
2. Baseline reproduction
   - Reproduce the published `9 x 512` run from this fork before changing the model.
   - Treat that run as the local comparison anchor even if it differs slightly from the published record.
3. First code change
   - Add shared-depth recurrence with a small number of unique blocks and a repeat loop.
   - Keep exporter logic, optimizer split, tokenizer, dataset, and evaluation code unchanged.
   - Keep repeat-specific parameters tiny and separate from the shared matrices.
4. First recurrence sweep
   - `baseline_9x512_kv4_mlp2`
   - `rec_u3_r3_d512_kv4_mlp2`
   - `rec_u3_r3_d576_kv4_mlp2`
   - `rec_u3_r3_d640_kv4_mlp2`
5. Post-sweep decision rule
   - Positive: any recurrence run beats the local baseline by `>= 0.002` quantized `val_bpb`
   - Negative: all recurrence runs lose by `>= 0.010`
   - Ambiguous: keep one stable recurrence anchor and run exactly one cheap `KV4 -> KV2` reallocation before switching families

## Experiment Ledger

| Run ID | Family | Shape | Objective | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| `baseline_9x512_kv4_mlp2` | baseline | `9 x 512`, `KV4`, `MLP2` | reproduce local anchor under `600s` and `<16 MB` | pending | published target is `1.22436570`, `15,863,489` bytes |
| `rec_u3_r3_d512_kv4_mlp2` | recurrence | `3` unique blocks, `3` repeats, `d=512`, `KV4`, `MLP2` | cheapest recurrence test against same width as baseline | pending | first recurrence anchor |
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
