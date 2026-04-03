# Remote Runbook

Last updated: 2026-04-03 08:48 UTC

This note pins the current remote execution surface for Parameter Golf so later sessions can resume from disk instead of reconstructing the node state from Slack.

## Pinned Paths

- Node SSH alias: `tianhaowang-gpu0`
- Scratch checkout: `/data/scratch/murphy/parameter-golf`
- Cache root: `/data/scratch/murphy/cache`
- CUDA Python env: `/home/murphy/miniforge3/envs/swe311`
- Dataset path: `/data/scratch/murphy/parameter-golf/data/datasets/fineweb10B_sp1024`
- Tokenizer path: `/data/scratch/murphy/parameter-golf/data/tokenizers/fineweb_1024_bpe.model`

## Required Cache Exports

Every queued run should export:

```bash
export HF_HOME=/data/scratch/murphy/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/data/scratch/murphy/cache/huggingface/hub
export TRANSFORMERS_CACHE=/data/scratch/murphy/cache/huggingface/transformers
export TORCH_HOME=/data/scratch/murphy/cache/torch
export XDG_CACHE_HOME=/data/scratch/murphy/cache
```

## Run Directory Contract

Use `outputs/run_jobs/<run_id>/` for queued-node runs. Each run directory should contain:

- `job.sbatch` — exact submitted Slurm script
- `slurm-<jobid>.out` — scheduler stdout/stderr
- `train.log` — tee'd trainer log
- `train_gpt.py` — copied code snapshot used for the run
- `final_model.pt` — copied final checkpoint when the run produces one

## Current Launch Shapes

- Lean idea-testing smoke:
  - `RUN_ID=baseline_sp1024_smoke_1gpu_lean`
  - `--gres=gpu:1`
  - `--cpus-per-task=2`
  - `--mem=12G`
  - `TRAIN_BATCH_TOKENS=32768`
  - `MAX_WALLCLOCK_SECONDS=120`
  - purpose: validate repo/data/env/logging path with minimal resource pressure
  - validated once already as job `1993`: `val_bpb=1.57745762`, `1111` steps, `12,697,534` total int8+zlib bytes
- Lean recurrence smoke:
  - `RUN_ID=rec_u3_r3_d512_kv4_mlp2_smoke1gpu`
  - same `1` GPU / `2` CPU / `12G` envelope as the baseline smoke
  - add `NUM_UNIQUE_LAYERS=3` while keeping `NUM_LAYERS=9`
  - validated once already as job `2000`: `val_bpb=1.60109017`, `1454` steps, `4,963,934` total int8+zlib bytes
- Widened lean recurrence smokes:
  - `RUN_ID=rec_u3_r3_d576_kv4_mlp2_smoke1gpu`
  - same `1` GPU / `2` CPU / `12G` envelope
  - keep `NUM_UNIQUE_LAYERS=3`, `NUM_LAYERS=9`, `NUM_KV_HEADS=4`, `MLP_MULT=2`, set `MODEL_DIM=576`
  - validated once already as job `2003`: `val_bpb=1.59122937`, `1380` steps, `6,069,746` total int8+zlib bytes
  - `RUN_ID=rec_u3_r3_d640_kv4_mlp2_smoke1gpu`
  - same lean envelope, set `MODEL_DIM=640`
  - validated once already as job `2004`: `val_bpb=1.58541212`, `1348` steps, `7,265,478` total int8+zlib bytes
- Next cheap recurrence discriminator:
  - `RUN_ID=rec_u3_r3_d640_kv2_mlp2_smoke1gpu`
  - same `1` GPU / `2` CPU / `12G` envelope
  - keep `NUM_UNIQUE_LAYERS=3`, `NUM_LAYERS=9`, `MODEL_DIM=640`, `MLP_MULT=2`
  - change `NUM_KV_HEADS=2`
  - purpose: test whether KV-head reallocation can close the remaining cheap-stage gap on the promoted `d=640` recurrence anchor before an honest `8`-GPU comparison run
- Real baseline:
  - `RUN_ID=baseline_sp1024_localcheck`
  - `--gres=gpu:8`
  - `MAX_WALLCLOCK_SECONDS=600`
  - `VAL_LOSS_EVERY=200`
  - purpose: reproduce the published `9 x 512` anchor later, at the challenge-honest comparison gate

## Parsing Logs

After a run finishes, extract the compact metrics with:

```bash
python tools/parse_train_log.py outputs/run_jobs/<run_id>/train.log
```

This parser is intended to feed the tracker, roadmap, and Slack updates without hand-copying the log tail.
