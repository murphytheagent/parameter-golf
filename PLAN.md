# Parameter Golf Plan

Last updated: 2026-04-01 11:18 UTC

## Objective

Beat the current held-out FineWeb baseline under the actual challenge contract:

- fixed FineWeb validation objective (`val_bpb`)
- strict artifact cap: `< 16,000,000` bytes for code plus compressed weights
- strict training budget: `600` seconds on `8` GPUs
- no upstream PR yet; work stays in the local fork until there is a result worth publishing

This file is the self-contained research and implementation plan for the current fork state. `TRACKER.md` is the compact run ledger and stage-gate file.

## What Is Established

### Baseline object

- Public 10-minute baseline:
  - architecture: `9 x 512`, tied embeddings, `8` query heads, `4` KV heads, `MLP_MULT=2`
  - final quantized metric: `final_int8_zlib_roundtrip_exact val_bpb:1.22436570`
  - total submission bytes: `15,863,489`
  - code bytes: `47,642`
  - timed stop: `13,780` steps in `600.038` seconds
- Unlimited-compute reference on the same stem:
  - `1.20737944` after `4` hours, so the objective still has real headroom without changing the tokenizer

### Athena-backed ranking

For this exact repo and constraint set, the current ranking is:

1. shared-depth recurrence / aggressive parameter tying
2. bounded test-time adaptation or streaming memory
3. `MTP-lite`

Supporting notes:

- `MQA/GQA` is a cheap supporting sweep, not the primary first branch
- deeper MLA / latent-KV is lower priority under the current exporter
- stored low-rank factorization is not attractive under the current tensor serialization behavior

### AttnRes verdict

- Do not move full Kimi Attention Residuals or Block AttnRes ahead of recurrence or `MTP-lite`
- If anything is borrowed later, it should only be a tiny depth-mix component, not a full branch rewrite

## What Is Not Established

- No reproduced local baseline exists yet from the current surviving fork state
- No recurrence run exists yet from the current surviving fork state
- No `MTP-lite`, AttnRes-lite, or bounded test-time adaptation run exists yet from the current surviving fork state
- The remote substrate is not warm:
  - on `2026-04-01 11:05 UTC`, `wth-gpu-01` was `mixed`, not fully blocked, with only `zhijianliu`'s 1-GPU job `1942` scheduled through `2026-04-01 14:27 UTC`
  - `/data/scratch/murphy` exists and `/data/scratch/murphy/cache` exists
  - `/data/users/murphy` does not exist
  - no ready Parameter Golf checkout or FineWeb SP1024 dataset/tokenizer cache was visible under the checked Murphy paths (`/data/scratch/murphy/parameter-golf` and `/data/scratch/murphy/projects/parameter-golf` were both absent)

## Why Recurrence Is First

- The byte cap rewards effective depth per stored weight more directly than bigger attention-side mechanisms
- The exporter path is already stable for the baseline stem, so recurrence can attack the depth-per-byte problem without introducing an evaluation-side ambiguity immediately
- `MTP-lite` is still attractive, but it is cleaner to decide whether it helps after one architectural stem is established
- Bounded test-time adaptation remains live, but it is harder to keep challenge-legible and should not be the first implementation line
- Full AttnRes is too large a conceptual jump relative to what the baseline already gets from its existing mix/skip structure

## One Clean Execution Path

### 1. Warm the remote substrate

- Use `/data/scratch/murphy/parameter-golf` as the working directory
- Route `HF_HOME`, Hugging Face hub, transformers, torch, and `XDG_CACHE_HOME` to `/data/scratch/murphy/cache`
- Materialize the `fineweb10B_sp1024` dataset and `fineweb_1024_bpe.model` on the same data volume

### 2. Reproduce the published baseline in our environment

Purpose:

- establish the real local comparison anchor before changing the model
- verify logging, artifact capture, and size accounting

Pinned command shape:

```bash
NCCL_IB_DISABLE=1 \
RUN_ID=baseline_sp1024_localcheck \
DATA_PATH=<remote fineweb10B_sp1024 path> \
TOKENIZER_PATH=<remote fineweb_1024_bpe.model path> \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Acceptance rule:

- if the reproduced run lands within about `+-0.005` `val_bpb` of `1.22436570` and stays under the byte cap, it becomes the local anchor

### 3. Make exactly one first code change: shared-depth recurrence

- Add a smaller set of unique blocks and repeat them to recover effective depth
- Keep exporter logic, tokenizer, dataset, optimizer split, and evaluation path unchanged
- Keep repeat-specific parameters tiny:
  - repeat-indexed `resid_mix`
  - `attn_scale`
  - `mlp_scale`
  - only any additional skip-mix coefficients that prove necessary
- Do not mix in `MTP-lite`, test-time adaptation, or full AttnRes in the first recurrence commit

### 4. Run the first recurrence sweep

- `baseline_9x512_kv4_mlp2`
- `rec_u3_r3_d512_kv4_mlp2`
- `rec_u3_r3_d576_kv4_mlp2`
- `rec_u3_r3_d640_kv4_mlp2`

Interpretation:

- `u3`: `3` unique blocks
- `r3`: `3` repeats
- widths `512 / 576 / 640` test whether width reallocation helps recurrence under the same byte cap

### 5. Decide the next family with a hard rule

- Positive:
  - if any recurrence run beats the local baseline by `>= 0.002` quantized `val_bpb`, recurrence stays live
- Negative:
  - if all recurrence runs lose by `>= 0.010`, recurrence is dead for now and the next budget goes to `MTP-lite`
- Ambiguous:
  - if recurrence is neutral but stable, keep one recurrence anchor and run exactly one cheap `KV4 -> KV2` reallocation test before switching families

## What Comes After The Recurrence Verdict

Priority order:

1. cheap `KV4 -> KV2` sweep on the best stable recurrence anchor if recurrence is neutral
2. `MTP-lite` on the winning stem, or on the baseline stem if recurrence fails cleanly
3. bounded test-time adaptation or streaming memory only if training-side changes stall
4. AttnRes-lite only as a later falsification branch for depth-state mixing

## Run Recording Contract

For every run, record the following in one place:

- `run_id`
- `git_rev`
- exact model shape
- recurrence knobs or auxiliary-head knobs
- `code_bytes`
- `quantized_model_bytes`
- `total_submission_bytes`
- final quantized `val_bpb`
- `step_count`
- `step_avg_ms`
- `peak_memory`
- divergence, NaNs, scheduler interference, or suspicious seed sensitivity

## Current Blocker

There is still no honest performance claim beyond the public baseline because the remote substrate is not staged yet. The old full-node blocker note is no longer current: the node is partially free now, but only `7 / 8` GPUs are available and there is still no ready checkout/data surface for Murphy under `/data/scratch`. The next executable step, once the substrate is materialized and the full 8-GPU window is real, is baseline reproduction first, not model editing first.
