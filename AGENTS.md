# Project: parameter-golf

Local fork for OpenAI's 16 MB / 10-minute Parameter Golf challenge. Current focus is one clean first-wave path: follow the self-contained root `PLAN.md`, reproduce the published baseline in our environment, then test shared-depth recurrence before MTP-lite or AttnRes-style branches.

## Key Docs
- `PLAN.md` — self-contained research and implementation plan; read this first
- `outputs/plan_report/parameter_golf_plan.pdf` — shareable PDF render of the current plan surface
- `TRACKER.md` — compact progress and experiment tracker
- `roadmap.md` — current status, milestones, activity log, stage gates
- `backlog.md` — open tasks and pending experiment/setup work
- `docs/README.md` — index of durable notes for the fork
- Upstream `README.md` — challenge contract, public baseline, submission rules

## Sub-Session Instructions
- Read `PLAN.md` first, then `TRACKER.md`, then `roadmap.md`.
- Keep `train_gpt.py` under the upstream `1500` line limit.
- Treat the published 10-minute record as the baseline object:
  `val_bpb=1.22436570`, total bytes `15,863,489`.
- For every run, record:
  `run_id`, model shape, total bytes, quantized-model bytes, code bytes, final quantized `val_bpb`, steps, wallclock, and any divergence/instability notes.
- No upstream PR yet. Work in the local fork until a run is worth publishing.
