# KPI Baseline Authority (Deployment Profiles)

This document is the **canonical reference** for Phase A baseline expectations across `local`, `hybrid`, and `cloud`.

## Binding sources

| Artifact | Purpose |
|----------|---------|
| [docs/loadtest-matrix.md](loadtest-matrix.md) | Mandatory matrix rows, error-rate gates, smoke shapes |
| [docs/profile-pack.md](profile-pack.md) | Default RPS/workers/queue per profile |
| [docs/runbook-quickstart.md](runbook-quickstart.md) | Bring-up, KPI capture, incident triage |

## Profile KPI capture

For each profile, generate machine-readable reports with `tools/serve_kpi_smoke.py` (see `just kpi-serve-local|hybrid|cloud`).

Outputs must conform to `serve_kpi_smoke_v1` JSON (see loadtest matrix).

## SLO bands (initial)

Bands are **starting points**; tighten quarterly using real hardware and checkpoint data.

- **Latency:** track `latency_ms_p95` per profile; alert when sustained above the frozen baseline band.
- **Throughput:** track `tokens_per_sec`; regressions require investigation before release.
- **Reliability:** `error_rate` must stay within the gates in [loadtest-matrix.md](loadtest-matrix.md).

## Release binding rule

No operations-ready release candidate ship without:

- `kpi_local.json`, `kpi_hybrid.json`, `kpi_cloud.json` (or documented waiver with owner sign-off).
- Golden-path train/eval/classical artifacts validated by `tools/validate_release_gates.py` (`--require-kpi` when runtime proof is mandatory).
