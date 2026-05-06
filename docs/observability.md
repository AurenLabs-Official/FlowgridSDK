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
