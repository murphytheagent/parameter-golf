# Parameter Golf Plan

Last updated: 2026-04-03 08:01 UTC

## Objective

Beat the current held-out FineWeb baseline under the actual challenge contract:

- fixed FineWeb validation objective (`val_bpb`)
- strict artifact cap: `< 16,000,000` bytes for code plus compressed weights
- strict training budget: `600` seconds on `8` GPUs
- no upstream PR yet; work stays in the local fork until there is a result worth publishing

This file is the self-contained research and implementation plan for the current fork state. `TRACKER.md` is the compact run ledger and stage-gate file. A collaborator-facing PDF render of this plan lives under `outputs/plan_report/parameter_golf_plan.pdf`.

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

### Remote run surface

- Scratch checkout now exists at `/data/scratch/murphy/parameter-golf` on `origin/main` commit `08ee8ba`
- The full published `fineweb10B_sp1024` dataset and `fineweb_1024_bpe.model` tokenizer are materialized under that checkout
- The working CUDA environment on the node is `/home/murphy/miniforge3/envs/swe311`, which already has `torch 2.9.1+cu128`, `numpy`, `datasets`, `sentencepiece`, and `huggingface_hub`
- Run artifacts are now organized under `/data/scratch/murphy/parameter-golf/outputs/run_jobs/<run_id>/`
- Lean smoke anchor:
  - job `1993` completed cleanly on `1` GPU / `2` CPUs / `12G`
  - exact result: `final_int8_zlib_roundtrip_exact val_bpb:1.57745762`
  - stopped at `1111` steps in `120.101s`
  - total int8+zlib submission size: `12,697,534` bytes

### Execution-stage correction

- The collaborator explicitly relaxed the early-stage resource rule on `2026-04-03 07:57 UTC`: idea-testing and code-testing do not need `8` GPUs yet
- Current rule:
  - use the smallest available GPU slice that can honestly exercise the path
  - keep CPU and host-memory requests minimal in that stage
  - reserve the full `8`-GPU baseline only for the later challenge-honest comparison gate

### Athena-backed ranking

For this exact repo and constraint set, the current ranking is:

1. shared-depth recurrence / aggressive parameter tying
2. bounded test-time adaptation or streaming memory
3. `MTP-lite`

Supporting notes:

- `MQA/GQA` is a cheap supporting sweep, not the primary first branch
- deeper MLA / latent-KV is lower priority under the current exporter
- stored low-rank factorization is not attractive under the current tensor serialization behavior
- small float tensors with `<= 65,536` elements stay in fp16 passthrough under the current int8 export path, which weakens naive stored low-rank compression ideas

### AttnRes verdict

- Do not move full Kimi Attention Residuals or Block AttnRes ahead of recurrence or `MTP-lite`
- If anything is borrowed later, it should only be a tiny depth-mix component, not a full branch rewrite

## What Is Not Established

- No reproduced local baseline exists yet from the current surviving fork state
- No recurrence run exists yet from the current surviving fork state
- No `MTP-lite`, AttnRes-lite, or bounded test-time adaptation run exists yet from the current surviving fork state
- There is still no challenge-honest baseline comparison yet:
  - the earlier queued jobs `1989` and `1990` were intentionally canceled after the collaborator redirected the first stage away from `8` GPUs
  - `1993` proved the lean runtime path, but it is only a code-testing anchor, not a baseline-comparison anchor
  - `/data` still has only `219G` free of `7.0T`, so later iterations should stay disciplined about artifact cleanup

## Why Recurrence Is First

- The byte cap rewards effective depth per stored weight more directly than bigger attention-side mechanisms
- The exporter path is already stable for the baseline stem, so recurrence can attack the depth-per-byte problem without introducing an evaluation-side ambiguity immediately
- `MTP-lite` is still attractive, but it is cleaner to decide whether it helps after one architectural stem is established
- Bounded test-time adaptation remains live, but it is harder to keep challenge-legible and should not be the first implementation line
- Full AttnRes is too large a conceptual jump relative to what the baseline already gets from its existing mix/skip structure

## One Clean Execution Path

### 1. Remote substrate

- Current state at `2026-04-03 07:54 UTC`: done
- Use `/data/scratch/murphy/parameter-golf` as the working directory
- Route `HF_HOME`, Hugging Face hub, transformers, torch, and `XDG_CACHE_HOME` to `/data/scratch/murphy/cache`
- Materialize the `fineweb10B_sp1024` dataset and `fineweb_1024_bpe.model` on the same data volume
- Keep run logs and copied artifacts under `outputs/run_jobs/<run_id>/`

### 2. Reproduce the published baseline in our environment

Purpose:

- establish the real local comparison anchor before changing the model
- verify logging, artifact capture, and size accounting

Important stage note:

- this remains the first honest comparison anchor before any claim of "beats baseline"
- it is no longer the required first runtime action in the idea-testing stage

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

There is still no honest performance claim beyond the public baseline because no Murphy-run baseline has finished yet. But there is no longer a blocker to architecture idea-testing: the checkout, cache routing, dataset, tokenizer, and lean `1`-GPU run surface all work, and `1993` already produced the first cheap anchor. The remaining blocker applies only to challenge-honest comparison, not to the next coding step: before claiming an actual baseline improvement, the project still needs a Murphy-run `8`-GPU anchor, but before that the next executable move is a real architecture change under the lean `1`-GPU envelope.
