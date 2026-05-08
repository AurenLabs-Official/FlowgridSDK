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

Capture both traffic shapes per profile:

- steady: `kpi_local.json`, `kpi_hybrid.json`, `kpi_cloud.json`
- burst: `kpi_local_burst.json`, `kpi_hybrid_burst.json`, `kpi_cloud_burst.json`

## Frozen baseline contract

- Canonical in-repo baseline path: `docs/ops-artifacts/baselines/latest/`.
- Each release candidate may reference release assets instead, but must map one-to-one to the six filenames above.
- `tools/check_kpi_regression.py` compares `target/mlops/` against the frozen set and enforces gate bands + regression tolerances.
- Baseline refresh must include owner + date note in the release notes / review doc.

## SLO bands (initial)

Bands are **starting points**; tighten quarterly using real hardware and checkpoint data.

- **Latency:** track `latency_ms_p95` per profile; alert when sustained above the frozen baseline band.
- **Throughput:** track `tokens_per_sec`; regressions require investigation before release.
- **Reliability:** `error_rate` must stay within the gates in [loadtest-matrix.md](loadtest-matrix.md).

## Release binding rule

No operations-ready release candidate ship without:

- `kpi_local.json`, `kpi_hybrid.json`, `kpi_cloud.json` (or documented waiver with owner sign-off).
- `kpi_local_burst.json`, `kpi_hybrid_burst.json`, `kpi_cloud_burst.json` (or documented waiver with owner sign-off).
- Golden-path train/eval/classical artifacts validated by `tools/validate_release_gates.py` (`--require-kpi` when runtime proof is mandatory).
- KPI regression check passes: `python tools/check_kpi_regression.py --current-dir target/mlops --baseline-dir docs/ops-artifacts/baselines/latest --require-current --require-burst`.
