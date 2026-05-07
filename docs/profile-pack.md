# Deployment Profile Pack

This profile pack defines the operational defaults for the 6-month roadmap release line.

## Profiles

| Profile | Default intent | Serve defaults |
|---------|----------------|----------------|
| `local` | laptop / single-node preview | `RPS=32`, `WORKERS=1`, `QUEUE_DEPTH=64` |
| `hybrid` | mixed workstation + shared infra | `RPS=96`, `WORKERS=2`, `QUEUE_DEPTH=256` |
| `cloud` | multi-tenant hosted runtime | `RPS=256`, `WORKERS=4`, `QUEUE_DEPTH=512` |

> Explicit `FLOWGRID_SERVE_*` env vars override these defaults.

## Recommended rollout order

1. Capture KPI baseline with `kpi-serve-local`.
2. Promote and capture `kpi-serve-hybrid`.
3. Promote and capture `kpi-serve-cloud`.
4. Build combined KPI report with `tools/build_kpi_achievement_report.py`.

## Release bundle

Attach these artifacts to each operations-ready release candidate:

- `target/mlops/kpi_local.json`
- `target/mlops/kpi_hybrid.json`
- `target/mlops/kpi_cloud.json`
- `target/mlops/golden_llm_train.json`
- `target/mlops/golden_llm_eval.json`
- `target/mlops/golden_classical_ml.json`
- `target/mlops/kpi_achievement_report.md`
