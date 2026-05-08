# Runbook Quickstart

This page is the operational start point for `flowgrid-serve` and the LLM/ML preview stack.

## 1) Choose deployment profile

Pick one profile and export it before starting serve:

```bash
export FLOWGRID_DEPLOYMENT_PROFILE=local   # or cloud|hybrid
```

Override specific knobs only when needed:

- `FLOWGRID_SERVE_RPS`
- `FLOWGRID_SERVE_BURST`
- `FLOWGRID_SERVE_QUEUE_DEPTH` (legacy alias: `FLOWGRID_SERVE_QUEUE`)
- `FLOWGRID_SERVE_WORKERS`
- `FLOWGRID_SERVE_STREAM_BUFFER`
- `FLOWGRID_SERVE_MAX_NEW_TOKENS`

## 2) Bring up service

```bash
cargo run -p flowgrid-serve
```

Health endpoint:

```bash
curl -s http://127.0.0.1:9000/healthz
```

## 3) Run KPI smoke

With serve running, collect baseline reports for both shapes:

```bash
python tools/serve_kpi_smoke.py --base-url http://127.0.0.1:9000 --requests 32 --max-tokens 32 --out target/mlops/kpi_local.json
python tools/serve_kpi_smoke.py --base-url http://127.0.0.1:9000 --requests 96 --max-tokens 64 --out target/mlops/kpi_local_burst.json
```

Repeat profile by profile:

1. `FLOWGRID_DEPLOYMENT_PROFILE=local`: generate `kpi_local*.json`.
2. `FLOWGRID_DEPLOYMENT_PROFILE=hybrid`: generate `kpi_hybrid*.json`.
3. `FLOWGRID_DEPLOYMENT_PROFILE=cloud`: generate `kpi_cloud*.json`.

## 4) Incident triage (fast path)

When requests fail or latency spikes:

1. Check `error_rate`, `latency_ms_p95`, and `tokens_per_sec` in the latest KPI report.
2. Confirm profile/env overrides in the serve process.
3. If queue pressure is high, lower `FLOWGRID_SERVE_MAX_NEW_TOKENS` and raise `FLOWGRID_SERVE_QUEUE_DEPTH` cautiously.
4. If overload persists, return `429` to callers and throttle upstream traffic until the queue stabilizes.

## 5) Failure drills

Validate overload behavior with a constrained queue:

```bash
FLOWGRID_SERVE_QUEUE_DEPTH=1 FLOWGRID_SERVE_WORKERS=1 cargo run -p flowgrid-serve
```

Then run the KPI smoke with higher request count and confirm:

- Overload is surfaced as OpenAI-style `429` (`server_overloaded`).
- Service stays responsive for subsequent requests.

## 6) Release artifacts checklist

For each release candidate, attach:

- `target/mlops/kpi_local.json`
- `target/mlops/kpi_cloud.json`
- `target/mlops/kpi_hybrid.json`
- `target/mlops/kpi_local_burst.json`
- `target/mlops/kpi_hybrid_burst.json`
- `target/mlops/kpi_cloud_burst.json`
- train/eval run reports for golden paths

Regression gate:

```bash
python tools/check_kpi_regression.py --current-dir target/mlops --baseline-dir docs/ops-artifacts/baselines/latest --require-current --require-burst
```
