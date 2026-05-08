# Frozen KPI baseline (`latest`)

This directory contains the canonical in-repo KPI baseline used by `tools/check_kpi_regression.py`.

Required files:

- `kpi_local.json`
- `kpi_local_burst.json`
- `kpi_hybrid.json`
- `kpi_hybrid_burst.json`
- `kpi_cloud.json`
- `kpi_cloud_burst.json`

Refresh policy:

1. Re-capture all six files from one measurement runbook cycle.
2. Update this directory atomically (no partial refresh).
3. Record owner/date/reason in release notes or quarterly review notes.
