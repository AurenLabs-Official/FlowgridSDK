# Loadtest Matrix (Baseline)

This matrix defines the required baseline runs for `local`, `cloud`, and `hybrid`.

## KPI gates

- `latency_ms_p95` (serve KPI smoke)
- `tokens_per_sec` (serve KPI smoke)
- `error_rate` (serve KPI smoke)
- `requests_ok / requests_total`

## Mandatory matrix

| Profile | Traffic shape | Requests | Max tokens | Expected gate |
|---------|----------------|----------|------------|---------------|
| local | steady smoke | 32 | 32 | `error_rate <= 0.05` |
| local | burst smoke | 96 | 64 | `error_rate <= 0.10` |
| hybrid | steady smoke | 64 | 64 | `error_rate <= 0.05` |
| hybrid | burst smoke | 128 | 64 | `error_rate <= 0.10` |
| cloud | steady smoke | 128 | 64 | `error_rate <= 0.03` |
| cloud | burst smoke | 256 | 128 | `error_rate <= 0.08` |

> These are baseline gates for the current preview stack. Tighten per release once hardware and checkpoint are fixed.

## Output contract

Each run must output:

```json
{
  "kind": "serve_kpi_smoke_v1",
  "base_url": "http://127.0.0.1:9000",
  "requests_total": 32,
  "requests_ok": 32,
  "requests_error": 0,
  "error_rate": 0.0,
  "latency_ms_p95": 8.2,
  "tokens_per_sec": 12345.6
}
```

## Recommended command

```bash
python tools/serve_kpi_smoke.py --base-url http://127.0.0.1:9000 --requests 64 --max-tokens 64 --out target/mlops/kpi_<profile>.json
```
