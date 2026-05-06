//! Optional HTTP client metrics (feature `opentelemetry`).

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
pub(crate) fn record_duration_ms(
    elapsed_ms: f64,
    provider: &'static str,
    method: &str,
    path: &str,
    status: Option<StatusCode>,
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
        ],
    );
}
