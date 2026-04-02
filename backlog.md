# Parameter Golf Backlog

## Immediate

- Keep `PLAN.md`, the PDF render in `outputs/plan_report/`, and `TRACKER.md` synced whenever the baseline anchor, experiment verdicts, or branch ordering changes.
- Materialize the remote run surface under `/data/scratch/murphy/parameter-golf` instead of `$HOME`.
- Route `HF_HOME`, hub, transformers, torch, and `XDG_CACHE_HOME` to `/data/scratch/murphy/cache`.
- Download or copy the `fineweb10B_sp1024` dataset and `fineweb_1024_bpe.model` onto the remote data volume.
- Reproduce the published `9 x 512` baseline from this fork and record the exact local reference metrics.
- Add a compact run ledger template so every later experiment records the same byte and metric fields.

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
