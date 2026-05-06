//! Optional HTTP client metrics (feature `opentelemetry`).
//!
//! **Cardinality:** Treat `flowgrid.http.path` as a small set of logical routes (what the
//! transport passes as the URL path segment). Do not attach unbounded values such as full URLs or
//! per-request ids to metrics. Prefer correlating individual requests via **`tracing`** spans
//! (see the repository `docs/observability.md` runbook).

use opentelemetry::{global, KeyValue};
use reqwest::StatusCode;
use std::sync::OnceLock;

fn status_class(status: StatusCode) -> &'static str {
    match status.as_u16() / 100 {
        2 => "2xx",
        3 => "3xx",
        4 => "4xx",
        5 => "5xx",
        _ => "other",
    }
}

/// Record request duration (milliseconds) for an outbound API call.
///
/// Keep labels low-cardinality: `flowgrid.http.path` is the logical API path (e.g. `chat/completions`),
/// not a full URL. Avoid adding `request_id` here—it can explode series count in Prometheus.
pub(crate) fn record_duration_ms(
    elapsed_ms: f64,
    provider: &'static str,
    method: &str,
    path: &str,
    status: Option<StatusCode>,
    retry_count: u32,
) {
    let (status_class, error_flag) = match status {
        Some(s) => (status_class(s), "false"),
        None => ("error", "true"),
    };
    static HIST: OnceLock<opentelemetry::metrics::Histogram<f64>> = OnceLock::new();
    let hist = HIST.get_or_init(|| {
        global::meter("flowgrid")
            .f64_histogram("flowgrid.http.request.duration_ms")
            .with_description("HTTP client request duration in milliseconds")
            .build()
    });
    hist.record(
        elapsed_ms,
        &[
            KeyValue::new("flowgrid.provider", provider),
            KeyValue::new("http.request.method", method.to_string()),
            KeyValue::new("flowgrid.http.path", path.to_string()),
            KeyValue::new("http.response.status_class", status_class),
            KeyValue::new("flowgrid.http.error", error_flag),
            KeyValue::new("flowgrid.retry_count", i64::from(retry_count)),
        ],
    );
}
