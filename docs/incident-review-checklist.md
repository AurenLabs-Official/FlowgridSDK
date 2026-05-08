# Incident Review Checklist

Use after any production-impacting incident or quarterly drill review.

## Summary

- Incident id / date / severity
- Affected profile (`local` | `hybrid` | `cloud`) and scope (serve / train / eval)

## Signals reviewed

- [ ] `p95` latency trend vs baseline band
- [ ] Error rate / timeout rate vs baseline
- [ ] Overload / `429` share (`server_overloaded`)
- [ ] Retry pressure (SDK `flowgrid.retry_count` where applicable)
- [ ] Queue saturation indicators from KPI reports

## Actions

- [ ] Root cause category (config, load, bug, dependency)
- [ ] Runbook gap identified? Update [runbook-quickstart.md](runbook-quickstart.md)
- [ ] Alert threshold adjustment? Update [observability.md](observability.md)
- [ ] Owner and due date for follow-ups

## Links

- Resilience program: [runtime-resilience-program.md](runtime-resilience-program.md)
- Observability conventions: [observability.md](observability.md)
