# ML Operations Handbook

This handbook defines the implementation baseline for the LLM/ML expansion stream.
It complements `docs/dev-handbook.md` and `docs/llm/overview.md`.

## Scope

- LLM runtime hardening (`flowgrid-serve`, `flowgrid-model`, `flowgrid-checkpoint`)
- ML core pipeline (`flowgrid-data`, `flowgrid-train`, `flowgrid-eval`)
- Classical ML baseline path (`flowgrid-ml`)
- Deployment profiles:
  - `local`
  - `cloud`
  - `hybrid`

## Deployment profiles

### Serve profile selection

Use:

```bash
FLOWGRID_DEPLOYMENT_PROFILE=local|cloud|hybrid
```

Default profile knobs in `flowgrid-serve`:

- `local`: `RPS=32`, `WORKERS=1`, `QUEUE_DEPTH=64`
- `cloud`: `RPS=256`, `WORKERS=4`, `QUEUE_DEPTH=512`
- `hybrid`: `RPS=96`, `WORKERS=2`, `QUEUE_DEPTH=256`

Explicit `FLOWGRID_SERVE_*` env vars override profile defaults.

### CLI profile selection

Use:

```bash
flowgrid-llm --profile local|cloud|hybrid <subcommand> ...
```

Behavior:

- Sets a device default when `FLOWGRID_DEVICE` is not set:
  - `local` -> `cpu`
  - `cloud` / `hybrid` -> `wgpu`
- Persists selected profile in run reports.

## Runtime hardening contract

- Scheduler supports configurable worker model and bounded per-worker queues.
- `FLOWGRID_SERVE_MAX_NEW_TOKENS` caps generation length at scheduling boundary.
- Stream buffering is configurable via `FLOWGRID_SERVE_STREAM_BUFFER`.
- For local checkpoint inference, worker count currently degrades safely to 1 due backend thread-safety constraints.

## ML core contract

- Dataset split semantics are standardized via `flowgrid-data`:
  - `DatasetSplit::{Train, Val, Test}`
  - `SplitSpec { train_frac, val_frac }`
- Eval supports full-dataset and split-range scoring via:
  - `perplexity`
  - `perplexity_in_range`
- Training path supports:
  - epochs (`--epochs`)
  - micro-batch width (`--batch-size`)
  - gradient accumulation (`--grad-accum`)

## MLOps artifact contract

Run metadata is persisted as JSON artifacts:

- Train: `flowgrid-llm train --run-report-out <path>`
- Eval: `flowgrid-llm eval --run-report-out <path>`

`flowgrid-ui` template jobs write default reports to:

- `target/mlops/train_tiny_report.json`
- `target/mlops/eval_smoke_report.json`

## CI gates

ML-specific smoke gate:

- Job: `ml-core-smoke` in `.github/workflows/ci.yml`
- Checks:
  - `cargo check -p flowgrid-data -p flowgrid-eval -p flowgrid-ml -p flowgrid-cli -p flowgrid-serve`
  - `cargo test -p flowgrid-data -p flowgrid-eval -p flowgrid-ml`

Local equivalent:

```bash
just check-ml-core
```

Reproducibility smoke gate:

```bash
just repro-ml-smoke
```

## Baseline KPI + loadtest matrix

Use the profile matrix in `docs/loadtest-matrix.md` and generate reports with:

```bash
just kpi-serve-local
just kpi-serve-hybrid
just kpi-serve-cloud
```

Each run produces a `serve_kpi_smoke_v1` JSON artifact under `target/mlops/`.

## Golden paths (release candidates)

### 1) LLM path

```bash
just golden-llm-path
```

Artifacts:

- `target/mlops/golden_llm_train.json`
- `target/mlops/golden_llm_eval.json`

### 2) Classical ML path

```bash
just golden-classical-ml-path
```

Artifact:

- `target/mlops/golden_classical_ml.json`

## Operations-ready pack

Bundle all required artifacts:

```bash
just ops-release-pack
```

Runbook for incidents and bring-up:

- `docs/runbook-quickstart.md`
- `docs/profile-pack.md`

Runtime resilience program and drill cadence:

- `docs/runtime-resilience-program.md`

Workload template portfolio:

- `docs/workload-templates.md`

Long-horizon ops readiness and deprecation lifecycle:

- `docs/ops-readiness-30m.md`

## Phase delivery baseline (12 weeks)

### Weeks 1-4

- Runtime: scheduler/backpressure/profile wiring
- Data/Eval: split-aware evaluation
- Artifacting: train/eval run reports

### Weeks 5-8

- Training quality: stronger train loops and reproducible presets
- Classical ML: extend `flowgrid-ml` beyond baseline linear/classification metrics
- Compare/reporting API in `flowgrid-ui`

### Weeks 9-12

- Profile A/B/C hardening in docs + CI
- Capacity and latency benchmarking with representative checkpoints
- Release hardening and migration notes

## KPIs

- Runtime: p95 latency, tokens/s, queue saturation rate
- Quality: perplexity trend, regression pass rate
- Operability: successful profile bring-up rate across local/cloud/hybrid
- Productivity: median time from dataset prep to evaluated artifact
