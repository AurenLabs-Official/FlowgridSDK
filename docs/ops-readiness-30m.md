# Ops Readiness (30-Month Track)

This document defines long-horizon operations readiness for FlowgridSDK.

## Lifecycle policy

### Backward compatibility

- Public SDK surface is semver-governed.
- Breaking changes require migration notes and staged rollout.

### Deprecation policy

- Mark deprecations in docs and changelog before removal.
- Keep deprecated behavior for at least one planned release cycle.
- Provide explicit replacement path in migration docs.

### Upgrade safety

- Every major dependency bump gets a compatibility note.
- Runtime profile defaults (`local|hybrid|cloud`) must be re-validated after upgrades.

## Maintenance calendar

- Monthly: KPI review and incident trend review.
- Quarterly: roadmap scope and release predictability review.
- Semiannual: architecture and dependency risk review.

## 30-month review package

At end-of-cycle, publish:

- KPI trend summary (runtime + ML + delivery)
- Change-log based release narrative
- Deprecation/compatibility summary
- Next-cycle priority proposal

## Required source documents

- `docs/ml-operations-handbook.md`
- `docs/profile-pack.md`
- `docs/runbook-quickstart.md`
- `docs/migration.md`
- `CHANGELOG.md`
