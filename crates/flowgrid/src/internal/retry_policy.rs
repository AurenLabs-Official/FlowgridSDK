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
    // Stale or equal-to-now HTTP-dates are treated as absent so callers fall back to exponential
    // backoff instead of spinning with a zero delay.
    if t <= now {
        return None;
    }
    t.duration_since(now).ok()
}

/// Parse `Retry-After` first; otherwise exponential backoff capped by `cap`.
#[allow(dead_code)] // Convenience wrapper; transports use [`retry_backoff_duration_openai`] / Anthropic variant.
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

#[cfg(all(feature = "openai", feature = "rate-aware-retry"))]
fn openai_rate_limit_reset_delay(headers: &HeaderMap) -> Option<Duration> {
    let rem = headers
        .get("x-ratelimit-remaining-requests")?
        .to_str()
        .ok()?;
    if rem.trim() != "0" {
        return None;
    }
    let reset = headers
        .get("x-ratelimit-reset-requests")?
        .to_str()
        .ok()?;
    let ts = reset.parse::<u64>().ok()?;
    let now = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?;
    let now_s = now.as_secs();
    if ts <= now_s {
        return Some(Duration::ZERO);
    }
    Some(Duration::from_secs(ts - now_s))
}

#[cfg(all(feature = "anthropic", feature = "rate-aware-retry"))]
fn anthropic_rate_limit_reset_delay(headers: &HeaderMap) -> Option<Duration> {
    let rem = headers
        .get("anthropic-ratelimit-requests-remaining")?
        .to_str()
        .ok()?;
    if rem.trim() != "0" {
        return None;
    }
    let reset = headers
        .get("anthropic-ratelimit-requests-reset")?
        .to_str()
        .ok()?;
    if let Ok(ts) = reset.parse::<u64>() {
        let now = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?;
        let now_s = now.as_secs();
        if ts > now_s {
            return Some(Duration::from_secs(ts - now_s));
        }
        return Some(Duration::ZERO);
    }
    let t = httpdate::parse_http_date(reset).ok()?;
    let now = SystemTime::now();
    if t > now {
        t.duration_since(now).ok()
    } else {
        Some(Duration::ZERO)
    }
}

#[cfg(feature = "openai")]
pub(crate) fn retry_backoff_duration_openai(
    headers: &HeaderMap,
    attempt: u32,
    exponential_backoff: impl Fn(u32) -> Duration,
    cap: Duration,
    #[allow(unused_variables)] rate_limit_aware: bool,
) -> Duration {
    if let Some(d) = parse_retry_after(headers) {
        return d.min(cap);
    }
    #[cfg(feature = "rate-aware-retry")]
    if rate_limit_aware {
        if let Some(d) = openai_rate_limit_reset_delay(headers) {
            return d.min(cap);
        }
    }
    exponential_backoff(attempt).min(cap)
}

#[cfg(feature = "anthropic")]
pub(crate) fn retry_backoff_duration_anthropic(
    headers: &HeaderMap,
    attempt: u32,
    exponential_backoff: impl Fn(u32) -> Duration,
    cap: Duration,
    #[allow(unused_variables)] rate_limit_aware: bool,
) -> Duration {
    if let Some(d) = parse_retry_after(headers) {
        return d.min(cap);
    }
    #[cfg(feature = "rate-aware-retry")]
    if rate_limit_aware {
        if let Some(d) = anthropic_rate_limit_reset_delay(headers) {
            return d.min(cap);
        }
    }
    exponential_backoff(attempt).min(cap)
}

/// Safe, length-capped excerpt for error `Display` / [`crate::internal::error`] fields.
pub(crate) fn body_snippet(text: &str) -> Option<String> {
    if text.is_empty() {
        None
    } else {
        Some(text.chars().take(512).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::HeaderValue;

    #[test]
    fn retry_after_delta_seconds() {
        let mut h = HeaderMap::new();
        h.insert("retry-after", HeaderValue::from_static("120"));
        assert_eq!(parse_retry_after(&h), Some(Duration::from_secs(120)));
    }

    #[test]
    fn retry_after_http_date_in_past_is_treated_as_absent() {
        let mut h = HeaderMap::new();
        h.insert(
            "retry-after",
            HeaderValue::from_static("Thu, 01 Jan 1970 00:00:00 GMT"),
        );
        assert!(parse_retry_after(&h).is_none());
    }

    #[test]
    fn sleep_before_retry_uses_exponential_when_no_retry_after_header() {
        let h = HeaderMap::new();
        let delay = sleep_before_retry(
            &h,
            2,
            |n| Duration::from_millis(50 * n as u64),
            Duration::from_secs(2),
        );
        assert_eq!(delay, Duration::from_millis(100));
    }
}
