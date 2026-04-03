# Parameter Golf Backlog

## Immediate

- Keep `PLAN.md`, the PDF render in `outputs/plan_report/`, and `TRACKER.md` synced whenever the stage rule, baseline anchor, or experiment verdicts change.
- Turn the successful `1993` smoke into a repeatable idea-testing loop with minimal CPU and memory.
- Implement shared-depth recurrence and run the first lean `1`-GPU recurrence smoke against the `1993` anchor.
- Reproduce the published `9 x 512` baseline from this fork later as the honest comparison gate before claiming improvement.
- Keep the log parser and run-directory layout current so later experiments do not require manual tail parsing.

## Completed This Round

- Materialized the remote run surface under `/data/scratch/murphy/parameter-golf` instead of `$HOME`.
- Routed `HF_HOME`, hub, transformers, torch, and `XDG_CACHE_HOME` to `/data/scratch/murphy/cache`.
- Downloaded the full `fineweb10B_sp1024` dataset and `fineweb_1024_bpe.model` onto the remote data volume.
- Proved that a corrected minimal-resource `1`-GPU smoke can actually launch on the warmed substrate.

## First Implementation

- Add a recurrence path with explicit knobs for unique blocks and repeat count.
- Keep the initial recurrence commit architecture-only:
  no `MTP-lite`, no test-time adaptation, no full AttnRes.
- Make repeat-specific control vectors optional and cheap.
- Add a small local byte estimator or dry export path so candidate shapes can be screened before a full remote run.

## First Experiment Wave

- Run `rec_u3_r3_d512_kv4_mlp2`.
- Run `rec_u3_r3_d576_kv4_mlp2`.
- Run `rec_u3_r3_d640_kv4_mlp2`.
- If recurrence looks neutral but stable, run one `KV4 -> KV2` reallocation test before switching families.

## Later Branches

- `MTP-lite` only after the recurrence branch gets a clean verdict.
- AttnRes-lite only as a narrow depth-mix ablation.
- Bounded test-time adaptation only if the training-side branches stall and the evaluation protocol remains challenge-legible.
