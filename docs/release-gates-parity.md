# Release Gates: CI and Local Parity

Local delivery commands should mirror CI so developers reproduce the same gates before merge.

## Release candidates (parity)

For a **release candidate** or **tagged preview**, the following local gates are **MUST**-pass (same commands as CI / [`justfile`](../justfile)):

| Gate | Command(s) |
|------|------------|
| ML core | `just check-ml-core` |
| Reproducibility | `just repro-ml-smoke` |
| Templates | `just template-train-lora-smoke` and `just template-eval-val-gate` |
| Golden LLM path | `just golden-llm-path` |
| Classical ML report | `just golden-classical-ml-path` |
| Artifact validation | `python tools/validate_release_gates.py` |
| KPI regression | `just kpi-regression-check` |

The aggregated loop is **`just ops-release-pack`** (includes gate validation; KPI JSON files require a running `flowgrid-serve` for profile smoke scripts).

## CI jobs (ML / gates)

| CI job | Local equivalent |
|--------|------------------|
| `ml-core-smoke` | `just check-ml-core` |
| `ml-repro-smoke` | `just repro-ml-smoke` |
| `ml-template-smoke` | `just template-train-lora-smoke` and `just template-eval-val-gate` |
| `release-gates-smoke` | `just golden-llm-path`, `just golden-classical-ml-path`, `python tools/validate_release_gates.py` |

## Full bundle

Run `just ops-release-pack` for the aggregated artifact loop (includes gate validation; KPI JSON files require a running `flowgrid-serve` for profile smoke).

## Strict KPI validation (release candidates)

```bash
python tools/validate_release_gates.py --require-kpi --kpi-regression-check --kpi-require-burst
```
Fails if required `target/mlops/kpi_*.json` / `target/mlops/kpi_*_burst.json` are missing or if regression/gate checks fail.
