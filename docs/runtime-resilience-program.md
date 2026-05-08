# Runtime Resilience Program

This program operationalizes runtime hardening for months 7-12 and keeps the same loop as a recurring quality track afterwards.

## Objectives

- Keep request handling predictable under load and partial failures.
- Prove overload behavior is safe (`429` + fast recovery).
- Ensure each release includes resilience evidence.

## Resilience pillars

- **Fairness:** scheduler distributes work without persistent starvation.
- **Overload handling:** bounded queues + explicit overload responses.
- **Degraded mode:** safe profile fallback when latency/error gates fail.
- **Failure drills:** repeated test scenarios for timeout, queue stress, restart safety.

## Drill cadence

- Weekly: one smoke drill (single profile).
- Bi-weekly: one stress drill (burst/overload).
- Monthly: cross-profile drill (`local`, `hybrid`, `cloud`) with artifacts.

## Required drill scenarios

1. **Queue saturation drill**
   - Constrain queue/worker (`FLOWGRID_SERVE_QUEUE=1`, `FLOWGRID_SERVE_WORKERS=1`).
   - Verify overload maps to OpenAI-style `429` code `server_overloaded`.
2. **Timeout drill**
   - Lower `FLOWGRID_SERVE_REQUEST_TIMEOUT_MS`.
   - Verify timeout behavior is explicit and does not deadlock worker loops.
3. **Recovery drill**
   - Trigger overload, then reduce pressure.
   - Verify service recovers without restart and error-rate returns to baseline band.

## Recovery KPI targets (initial)

Targets refine each quarter; tie-break using frozen baseline JSON from [kpi-baseline-authority.md](kpi-baseline-authority.md).

| Signal | Target |
|--------|--------|
| Post-drill `error_rate` | Within the profile gate band in [loadtest-matrix.md](loadtest-matrix.md) |
| Time to recover stable p95 | Trend down quarter-over-quarter vs Phase A baseline |
| Repeated overload oscillation | None sustained across two drill cycles |

## Release gate evidence

Each release candidate must include:

- `target/mlops/kpi_local.json`
- `target/mlops/kpi_hybrid.json`
- `target/mlops/kpi_cloud.json`
- Drill notes linked from `docs/runbook-quickstart.md`

### Versioning drill + KPI artifacts

- Attach or archive **`kpi_*.json`** (and optional drill notes) as **GitHub Release assets** alongside the git tag, or commit pinned copies under `docs/ops-artifacts/<tag>/` when your org requires an in-repo paper trail.
- Keep the weekly / bi-weekly / monthly cadence above in a team calendar or project board; copy **drill scenario outcomes** into the quarterly review doc (see [`docs/cycle-review-35m.md`](cycle-review-35m.md) template).

## Ownership loop

- Runtime owner runs drills and uploads artifacts.
- Reviewer signs off gates in PR.
- Release captain checks artifacts before tag/cut.
