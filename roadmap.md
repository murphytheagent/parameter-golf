# Parameter Golf Roadmap

Last updated: 2026-03-30 09:39 UTC

## Current Status

- Proved:
  - A root `TRACKER.md` now exists and is the single-entry state-of-record for proved vs unproved status, exact next runs, and the experiment ledger.
  - The public 10-minute baseline is the `9 x 512`, tied-embedding, Muon-trained GPT at `final_int8_zlib_roundtrip_exact val_bpb:1.22436570` and total submission size `15,863,489` bytes.
  - The current upstream-root `train_gpt.py` is still small enough to leave code-budget headroom: `47,642` bytes and `1,126` lines.
  - Athena's ranking for this exact challenge stayed stable across the successful retries: `1)` shared-depth recurrence / aggressive tying, `2)` bounded test-time adaptation or streaming memory, `3)` `MTP-lite`, with `MQA/GQA` as a cheap supporting sweep and deeper MLA / latent-KV behind that.
  - Athena's later AttnRes check did not promote Kimi Attention Residuals into round one. The recommendation remained: recurrence first, `MTP-lite` second, AttnRes only as a later falsification branch.
  - The fork checkout is healthy again: `projects/parameter-golf` is a clean repo on `main` at `0c0ea98`, with only untracked local outputs left behind.
- Unproved:
  - None of the earlier project-local tracker docs or first-wave implementation commits survived in this checkout. The current fork contains only the challenge code plus `outputs/literature/2026-03-16-kimi-attention-residuals.pdf`.
  - No recurrence, `MTP-lite`, AttnRes-lite, or bounded test-time adaptation experiment has actually been run from this surviving fork state.
  - The remote substrate is not warm: on 2026-03-30 09:17 UTC the node was occupied by `zekaili`'s Slurm job `1892` until `2026-04-01 02:29 UTC`, and no Murphy-owned Parameter Golf checkout or FineWeb SP1024 dataset was visible under the expected remote project/data paths.

## Milestone 1 - Restore Tracker And Remote Substrate

Success gate:
- Durable tracker exists in the fork.
- Baseline command is pinned.
- Remote cache/data paths are pinned outside `$HOME`.
- Baseline reproduction can be launched as soon as the node is free.

Activity log:
- 2026-03-19 01:21 UTC: Forked the upstream repo, read the public baseline and record folders, and initially added local planning docs plus a latent-KV prototype path. That implementation branch is no longer present in the surviving checkout.
- 2026-03-19 19:16 UTC: Athena deep consult succeeded after earlier timeouts and reordered the model-side ranking toward shared-depth recurrence first and `MTP-lite` ahead of MLA / latent-KV. Full consult log: `.agent/runtime/consult_history/1773871418.194959.jsonl`.
- 2026-03-22 00:47 UTC: Athena deep consult on Kimi Attention Residuals said not to promote AttnRes or Block AttnRes into round one ahead of recurrence or `MTP-lite`; only the tiny depth-mix idea is worth stealing later. Full consult log: `.agent/runtime/consult_history/1773871418.194959.jsonl`.
- 2026-03-30 09:31 UTC: Restored the missing project tracker after confirming that the earlier docs and implementation commits are absent from the current fork. Rechecked the remote execution surface: node blocked by job `1892` through `2026-04-01 02:29 UTC`; `/data/scratch/murphy` exists; `/data/users/murphy` and a ready-made FineWeb SP1024 cache/checkpoint surface do not currently exist.
- 2026-03-30 09:39 UTC: Added a root `TRACKER.md` so the next session no longer has to reconstruct the state from `roadmap.md`, `backlog.md`, and `docs/round1-plan.md`. The tracker now carries the proved vs unproved ledger, the exact baseline-plus-recurrence handoff, and the pending experiment table.

## Milestone 2 - Reproduce The Published Baseline In Our Environment

Success gate:
- One clean run from this fork lands within `+-0.005` `val_bpb` of the published `1.22436570`.
- Total bytes remain below `16,000,000`.
- Remote logging and artifact capture are working.

Pinned reference command:

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

Run ledger fields:
- `run_id`
- `git_rev`
- `model_shape`
- `code_bytes`
- `quantized_model_bytes`
- `total_submission_bytes`
- `final_int8_zlib_roundtrip_exact val_bpb`
- `step_count`
- `step_avg_ms`
- `peak_memory`
- notes on instability, divergence, or scheduler interference

## Milestone 3 - Round 1 Shared-Depth Recurrence

Success gate:
- At least one recurrence variant beats the local reproduced baseline on final quantized `val_bpb`.
- Or the family is killed with clear evidence that it loses on the actual 10-minute objective.

One clean implementation line:
- Add block sharing over a smaller set of unique blocks, repeated to recover effective depth.
- Keep the exporter and Muon path unchanged.
- Add only tiny repeat-specific control parameters at first: repeat-indexed `resid_mix`, `attn_scale`, `mlp_scale`, and any skip-mix coefficients that prove necessary.
- Do not mix in `MTP-lite`, test-time adaptation, or AttnRes-style full-history mixing in the first recurrence commit.

Exact first sweep to run after baseline reproduction:
- `baseline_9x512_kv4_mlp2`
- `rec_u3_r3_d512_kv4_mlp2`
- `rec_u3_r3_d576_kv4_mlp2`
- `rec_u3_r3_d640_kv4_mlp2`

Stage gate after the first recurrence sweep:
- If any recurrence run improves the local baseline by at least `0.002` quantized `val_bpb`, keep the family live and widen or simplify KV heads next.
- If all recurrence runs are worse by `>= 0.010`, pause recurrence scaling and move the next budget to `MTP-lite`.
- If the family is neutral but stable, keep one recurrence anchor and test a cheap `KV4 -> KV2` sweep before abandoning it.

## Milestone 4 - Secondary Branches Only After Recurrence Verdict

Priority order:
1. `MTP-lite` on the winning recurrence or baseline stem if recurrence fails cleanly
2. bounded test-time adaptation or streaming memory only if training-side changes stall
3. AttnRes-lite as a later ablation, not a first-wave branch

Branch-specific gates:
- `MTP-lite` only if the code diff stays small and the added head does not break the byte cap.
- Bounded test-time adaptation only if the evaluation protocol stays challenge-honest and the stored state is explicitly budgeted.
- AttnRes-lite only if recurrence gains plateau and the bottleneck looks like depth-state mixing rather than basic depth-per-byte.
