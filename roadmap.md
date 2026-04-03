# Parameter Golf Roadmap

Last updated: 2026-04-03 08:01 UTC

## Current Status

- Proved:
  - A root `PLAN.md` now exists as the self-contained research and implementation plan, so the main plan no longer lives only in `docs/round1-plan.md` or in thread history.
  - A collaborator-facing PDF render now exists at `outputs/plan_report/parameter_golf_plan.pdf`, so the shareable plan surface is back in the repo instead of living only as an old Slack attachment.
  - A root `TRACKER.md` now exists and is the single-entry state-of-record for proved vs unproved status, exact next runs, and the experiment ledger.
  - The public 10-minute baseline is the `9 x 512`, tied-embedding, Muon-trained GPT at `final_int8_zlib_roundtrip_exact val_bpb:1.22436570` and total submission size `15,863,489` bytes.
  - The current upstream-root `train_gpt.py` is still small enough to leave code-budget headroom: `47,642` bytes and `1,126` lines.
  - Athena's ranking for this exact challenge stayed stable across the successful retries: `1)` shared-depth recurrence / aggressive tying, `2)` bounded test-time adaptation or streaming memory, `3)` `MTP-lite`, with `MQA/GQA` as a cheap supporting sweep and deeper MLA / latent-KV behind that.
  - Athena's later AttnRes check did not promote Kimi Attention Residuals into round one. The recommendation remained: recurrence first, `MTP-lite` second, AttnRes only as a later falsification branch.
  - The current exporter still keeps float tensors with `<= 65,536` elements in fp16 passthrough, so naive stored low-rank factorization is still a weak byte-saving story here.
  - The fork checkout is healthy again: `projects/parameter-golf` is a clean repo on `main` at `0c0ea98`, with only untracked local outputs left behind.
  - The remote run surface is finally warm: `/data/scratch/murphy/parameter-golf` exists on `origin/main` commit `08ee8ba`, the full SP1024 dataset/tokenizer are present under `data/`, and the working CUDA env is `/home/murphy/miniforge3/envs/swe311`.
  - The collaborator explicitly relaxed the early-stage resource rule: idea-testing and code-testing may use `1` GPU with minimal CPU and host-memory requests before the later `8`-GPU comparison gate.
  - First lean run result now exists under `outputs/run_jobs/baseline_sp1024_smoke_1gpu_lean/`: job `1993` completed cleanly at `val_bpb=1.57745762`, `1111` steps, `120.101s`, and `12,697,534` total int8+zlib bytes.
  - First recurrence comparison now exists too: job `2000` (`rec_u3_r3_d512_kv4_mlp2_smoke1gpu`) landed at `val_bpb=1.60109017`, `1454` steps, `120.007s`, and `4,963,934` total int8+zlib bytes.
- Unproved:
  - None of the earlier project-local tracker docs or first-wave implementation commits survived in this checkout. The current fork contains only the challenge code plus `outputs/literature/2026-03-16-kimi-attention-residuals.pdf`.
  - No Murphy-run baseline has finished yet, so there is still no local comparison anchor for recurrence.
  - The recurrence family has only been checked at `d=512` so far, and static size estimates show that even `d=640` still leaves a lot of byte headroom. There is still no later full-contract baseline anchor.

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
- 2026-03-30 10:20 UTC: Promoted the self-contained round-one plan into root `PLAN.md` after the collaborator clarified that the intended deliverable was the narrative plan file itself, not the compact tracker.
- 2026-04-02 23:09 UTC: Rebuilt the collaborator-facing plan surface as `outputs/plan_report/parameter_golf_plan.pdf`, verified the render, and refreshed the blocker note with a live remote recheck. Current state: jobs `1960` (`1` GPU), `1965` (`1` GPU), and `1967` (`4` GPUs) still block the required `8`-GPU launch, while `/data/scratch/murphy` exists but no Murphy-owned Parameter Golf checkout or SP1024 dataset/tokenizer surface is materialized there yet.
- 2026-04-03 07:54 UTC: Warmed the real run surface on the node instead of leaving the plan blocked on missing infrastructure. `/data/scratch/murphy/parameter-golf` now exists at `origin/main` commit `08ee8ba`, the full SP1024 dataset/tokenizer are present under that checkout, and the working CUDA env is `/home/murphy/miniforge3/envs/swe311`.
- 2026-04-03 08:01 UTC: The collaborator explicitly redirected the first stage away from `8` GPUs: use minimal CPU and memory and only `1` GPU for idea-testing/code-testing. The earlier queued jobs `1989` and `1990` were canceled on that basis. After one bad submit caused by an unexpanded `PYBIN` variable, the corrected lean smoke job `1993` started successfully on `1` GPU / `2` CPUs / `12G` host memory and is now the active first experiment.
- 2026-04-03 08:04 UTC: Lean smoke job `1993` completed cleanly. Exact result: `final_int8_zlib_roundtrip_exact val_bpb:1.57745762`, `1111` steps in `120.101s`, `step_avg:108.10ms`, peak GPU memory `1335 MiB allocated / 1678 MiB reserved`, and `12,697,534` total int8+zlib submission bytes. This does not beat the public baseline, but it proves the cheap idea-testing loop is real; the next move is the first recurrence code change under the same lean envelope.
- 2026-04-03 08:18 UTC: Implemented the first shared-depth recurrence path by adding `NUM_UNIQUE_LAYERS` while keeping logical depth `9` and the existing exporter/optimizer split. Static size checks immediately showed why `d=512` is not a fair family verdict: `rec_u3_r3_d512` is only about `1.77 MB` total at init, while `d=576` and `d=640` are still only about `2.15 MB` and `2.61 MB`.
- 2026-04-03 08:18 UTC: First recurrence smoke job `2000` completed cleanly. Exact result: `final_int8_zlib_roundtrip_exact val_bpb:1.60109017`, `1454` steps in `120.007s`, `step_avg:82.54ms`, peak GPU memory `1173 MiB allocated / 1448 MiB reserved`, and `4,963,934` total int8+zlib submission bytes. Relative to the lean baseline smoke, this first recurrence patch is worse by about `0.0236` on `val_bpb`, but it buys clear systems wins: more steps, lower memory, and a much smaller artifact. Current inference: the family is undertrained or underparameterized, not dead; the next cheap sweep should widen to `576` and `640`.

## Milestone 2 - Reproduce The Published Baseline In Our Environment

Success gate:
- One clean run from this fork lands within `+-0.005` `val_bpb` of the published `1.22436570`.
- Total bytes remain below `16,000,000`.
- Remote logging and artifact capture are working.

Stage note:
- This is still the first honest comparison gate before any "beats baseline" claim, but it is no longer the required first runtime action during idea-testing.

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
