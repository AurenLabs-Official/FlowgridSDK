//! Shared retry helpers: `Retry-After` parsing and backoff merging.

use reqwest::header::HeaderMap;
use std::time::{Duration, SystemTime};

/// Parse `Retry-After` as delta-seconds or HTTP-date ([RFC 9110](https://www.rfc-editor.org/rfc/rfc9110.html#name-retry-after)).
pub(crate) fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    let raw = headers.get("retry-after")?.to_str().ok()?.trim();
    if raw.is_empty() {
        return None;
    }
    if let Ok(secs) = raw.parse::<u64>() {
        return Some(Duration::from_secs(secs));
    }
    let t = httpdate::parse_http_date(raw).ok()?;
    let now = SystemTime::now();
    t.duration_since(now).ok().or(Some(Duration::ZERO))
}

pub(crate) fn sleep_before_retry(
    headers: &HeaderMap,
    attempt: u32,
    exponential_backoff: impl Fn(u32) -> Duration,
    cap: Duration,
) -> Duration {
    parse_retry_after(headers)
        .unwrap_or_else(|| exponential_backoff(attempt))
        .min(cap)
}

/// Safe, length-capped excerpt for error `Display` / [`crate::internal::error`] fields.
pub(crate) fn body_snippet(text: &str) -> Option<String> {
    if text.is_empty() {
        None
    } else {
        Some(text.chars().take(512).collect())
    }
}
