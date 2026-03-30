# Round 1 Plan

This note has been promoted into root `PLAN.md`. Keep `PLAN.md` as the current self-contained plan; use this file as supporting detail only if needed.

## What Is Actually Established

- Public baseline:
  - `9 x 512`, tied embeddings, `8` query heads, `4` KV heads, `MLP_MULT=2`
  - `final_int8_zlib_roundtrip_exact val_bpb:1.22436570`
  - total submission bytes: `15,863,489`
  - code bytes: `47,642`
  - timed stop at `13,780` steps in `600.038` seconds
- Unlimited-compute reference:
  - same architecture reaches `1.20737944` after `4` hours, so there is real headroom in the objective even without changing the tokenizer
- Athena-backed ranking for this repo:
  - `1)` shared-depth recurrence / aggressive parameter tying
  - `2)` bounded test-time adaptation / streaming memory
  - `3)` `MTP-lite`
  - `MQA/GQA` is a cheap supporting sweep
  - deeper MLA / latent-KV and stored low-rank factorization are lower priority under the current exporter
- AttnRes verdict:
  - do not move full AttnRes or Block AttnRes ahead of recurrence or `MTP-lite`
  - if the idea is touched at all in this repo, only steal a tiny depth-mix component later

## What Is Not Established

- No surviving implementation of the earlier recurrence plan exists in the current fork.
- No surviving project-local experiment tracker existed before this reset.
- No first-wave run has been launched from the current fork state.
- The remote substrate is not preloaded with a Murphy-owned Parameter Golf checkout or a confirmed FineWeb SP1024 cache.

## Why The First Implementation Should Be Recurrence

- The current exporter and byte cap reward effective depth per stored weight more directly than flashy attention-side mechanisms.
- The baseline already includes cheap depth-mixing elements (`resid_mix` inside each block and `skip_weights` across the encoder/decoder split), so a full AttnRes transplant is partly redundant.
- `MTP-lite` is still attractive, but it is cleaner to establish one architectural stem first and only then decide whether an auxiliary objective helps that stem.
- Bounded test-time adaptation remains live, but it is harder to keep challenge-legible and should not be the very first code path.

## Exact Next-Run Handoff

1. Remote substrate
   - Use `/data/scratch/murphy/parameter-golf` as the working directory.
   - Use `/data/scratch/murphy/cache` for all framework caches.
   - Materialize the SP1024 dataset and tokenizer on the same data volume.
2. Baseline reproduction
   - Reproduce the published `9 x 512` run from this fork with the upstream command shape before changing the model.
   - Treat that run as the local comparison anchor even if it differs slightly from the published record.
3. First code change
   - Add shared-depth recurrence with a small number of unique blocks and a repeat loop.
   - Keep exporter logic, optimizer split, tokenizer, dataset, and evaluation code unchanged.
   - Keep repeat-specific parameters tiny and clearly separated from the shared matrices.
4. First recurrence sweep
   - `rec_u3_r3_d512_kv4_mlp2`
   - `rec_u3_r3_d576_kv4_mlp2`
   - `rec_u3_r3_d640_kv4_mlp2`
5. Decision rule
   - Positive result: any recurrence run beats the local baseline by `>= 0.002` quantized `val_bpb`
   - Negative result: all recurrence runs lose by `>= 0.010`
   - Ambiguous result: keep one stable recurrence anchor and run exactly one cheap KV-head reallocation before switching families

## Run-Recording Rule

For every run, record the following in one place:
- exact model shape and recurrence knobs
- code bytes
- quantized model bytes
- total submission bytes
- final quantized `val_bpb`
- step count and average step time
- peak memory
- any divergence, NaNs, or suspicious seed sensitivity
