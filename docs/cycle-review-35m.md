# 35-Month Cycle Review (Template)

Use at end of the 35-month roadmap cycle (or any major horizon) to close the loop and plan the next cycle.

## Executive summary

- Cycle dates:
- Primary themes delivered:
- Missed themes / rationale:

## KPI outcomes

### Runtime

- p95 latency trend vs Phase A baseline:
- Throughput (`tokens_per_sec`) trend:
- Error / timeout rate trend:
- Overload handling (`429` / recovery) outcomes:

### Quality / reliability

- Reproducibility gate stability:
- Regression rate (eval/train):
- Incident count / severity trend:

### Operations / delivery

- Release predictability (on-time releases, gate pass rate):
- Lead time to first validated artifact:

## Deliverables audit

- [ ] Baseline KPI JSON for all profiles archived
- [ ] Golden paths and gate scripts still green in CI
- [ ] Template portfolio adoption (share of new workloads using templates)
- [ ] Lifecycle policy adherence ([lifecycle-policy.md](lifecycle-policy.md))

## Decisions for next cycle

- Top 5 priorities:
- Explicit non-goals:
- Risk register updates:

## Sources

- [ops-readiness-30m.md](ops-readiness-30m.md)
- [ml-operations-handbook.md](ml-operations-handbook.md)
- [CHANGELOG.md](../CHANGELOG.md)
