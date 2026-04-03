# Parameter Golf Backlog

## Immediate

- Keep `PLAN.md`, the PDF render in `outputs/plan_report/`, and `TRACKER.md` synced whenever the stage rule, baseline anchor, or experiment verdicts change.
- Turn the successful `1993` smoke into a repeatable idea-testing loop with minimal CPU and memory.
- Wait for queued honest baseline job `2008` (`baseline_sp1024_localcheck`) to start once the node clears.
- If `2008` lands cleanly, run the promoted full-contract recurrence anchor `rec_u3_r3_d640_kv4_mlp2`.
- Keep the log parser and run-directory layout current so later experiments do not require manual tail parsing.

## Completed This Round

- Materialized the remote run surface under `/data/scratch/murphy/parameter-golf` instead of `$HOME`.
- Routed `HF_HOME`, hub, transformers, torch, and `XDG_CACHE_HOME` to `/data/scratch/murphy/cache`.
- Downloaded the full `fineweb10B_sp1024` dataset and `fineweb_1024_bpe.model` onto the remote data volume.
- Proved that a corrected minimal-resource `1`-GPU smoke can actually launch on the warmed substrate.
- Ran the widened recurrence smokes: `rec_u3_r3_d576_kv4_mlp2_smoke1gpu` and `rec_u3_r3_d640_kv4_mlp2_smoke1gpu`.
- Narrowed the recurrence follow-up path to one anchor: keep `d=640`, then try `KV4 -> KV2`.
- Ran the cheap `KV4 -> KV2` falsification check and dropped that branch after it lost cleanly to `d=640, KV4`.
- Queued the honest `8`-GPU baseline gate as job `2008`.

## First Implementation

- Add a recurrence path with explicit knobs for unique blocks and repeat count.
- Keep the initial recurrence commit architecture-only:
  no `MTP-lite`, no test-time adaptation, no full AttnRes.
- Make repeat-specific control vectors optional and cheap.
- Add a small local byte estimator or dry export path so candidate shapes can be screened before a full remote run.

## First Experiment Wave

- Reproduce `baseline_9x512_kv4_mlp2` under the honest `8`-GPU contract.
- Run `rec_u3_r3_d640_kv4_mlp2`.
- If recurrence still looks only neutral after that, switch the next clean branch to `MTP-lite`.

## Later Branches

- `MTP-lite` only after the recurrence branch gets a clean verdict.
- AttnRes-lite only as a narrow depth-mix ablation.
- Bounded test-time adaptation only if the training-side branches stall and the evaluation protocol remains challenge-legible.
