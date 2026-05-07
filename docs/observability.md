# Observability runbook

This document describes how outbound HTTP calls from `flowgrid` appear in logs and OpenTelemetry metrics when the **`tracing`** and/or **`opentelemetry`** features are enabled (see **`enterprise`** in the crate README).

## Finding a request in logs or APM

1. **Provider** — filter on `flowgrid.provider` (`openai` or `anthropic`) or on the log field the exporter maps from `tracing` (often the span field with the same name).
2. **Route / path** — use **`flowgrid.api.path`** on the span (logical relative path, e.g. `chat/completions`), not a full URL. Align dashboards with **`flowgrid.http.path`** on metrics (same string).
3. **HTTP method** — **`http.request.method`** on the span; **`http.request.method`** on the `flowgrid.http.request.duration_ms` histogram.
4. **Retries** — after the attempt finishes, the span records **`flowgrid.retry_count`** (number of *retry* rounds after the first attempt; `0` when the first response was final). Metrics include the same value as attribute **`flowgrid.retry_count`**.
5. **Request correlation** — on success, when the provider sends an id, the span records **`flowgrid.request_id`** (OpenAI: `x-request-id`; Anthropic: `request-id` / `x-request-id`). Prefer correlating user errors and support tickets with this id rather than logging response bodies.
6. **Rate limits (non-secret)** — when present, the span records **`flowgrid.ratelimit.requests.remaining`** and **`flowgrid.ratelimit.requests.reset`** (OpenAI: `x-ratelimit-*`; Anthropic: `anthropic-ratelimit-requests-*`).
7. **Trace linkage** — if your app installs a `tracing` → OpenTelemetry bridge, use your backend’s **`trace_id`** / **`span_id`** as usual; the span name below is stable for facet filters.

Structured **`debug!`** logs on target **`flowgrid_http`** still emit **`provider`**, **`method`**, **`path`**, **`elapsed_ms`**, **`status`** (when successful), and **`flowgrid.retry_count`**.

## Span and metric conventions

| Kind | Name / metric | Required or usual attributes | Notes |
|------|----------------|--------------------------------|--------|
| Span | `flowgrid.http.request` | `flowgrid.provider`, `http.request.method`, `flowgrid.api.path` | Parent span for one logical outbound call (includes retries). Dynamic fields recorded when known: `flowgrid.retry_count`, `flowgrid.request_id`, `flowgrid.ratelimit.requests.*`. |
| Metric | `flowgrid.http.request.duration_ms` | `flowgrid.provider`, `http.request.method`, `flowgrid.http.path`, `http.response.status_class`, `flowgrid.http.error`, `flowgrid.retry_count` | Histogram; duration in milliseconds. |

### High cardinality

- **`flowgrid.http.path`** must stay a **small, logical** route (what the SDK passes as the path segment), not per-user URLs. New unique paths increase time series cost.
- **Do not** add **`request_id`** (or similar) as a metric label—the SDK does not; recording it on the **span** only is intentional.
- If you add custom dimensions in your app, follow the same rules: prefer bounded enum-like values over raw URLs or ids.

## OpenTelemetry setup

Install a **meter provider** (and optionally a **tracer provider** if you bridge `tracing`) in your binary before making requests. Without a provider, the histogram API is a no-op. See the [OpenTelemetry Rust docs](https://opentelemetry.io/docs/languages/rust/) for initialization.

## Dashboard / alerts (hints)

These patterns stay vendor-neutral; map them to your backend’s query language (Grafana, Datadog, etc.):

1. **Latency SLO** — from `flowgrid.http.request.duration_ms`, group or filter by **`flowgrid.provider`**, **`flowgrid.http.path`**, and **`http.request.method`**. Use percentiles (p95/p99) per path, not per request id.
2. **Retry pressure** — alert when **`flowgrid.retry_count`** is often `> 0` for a path, or when a high share of spans show non-zero retry counts (can indicate throttling or upstream instability; pair with rate-limit attributes on the span when present).
3. **Error class** — use **`http.response.status_class`** and **`flowgrid.http.error`** on the metric; surface 4xx/5xx jumps per **`flowgrid.http.path`**. Keep labels bounded (see *High cardinality* above).
4. **Trace drill-down** — for user-visible failures, jump from metric/path to traces filtered by **`flowgrid.request_id`** on spans when the provider returned one.

Snippet-style pseudocode for “slow outbound LLM calls” (your backend may rename the histogram—often dots become underscores in Prometheus exporters):

```text
histogram_quantile(0.95,
  sum(rate(<histogram_name>_bucket{job="your_service"}[5m])) by (le, flowgrid_provider, flowgrid_http_path))
)
```
