//! Internal path helpers for resource clients.

/// `{base}/{tail}` with single slashes (no leading slash on `tail` required).
pub(crate) fn join_path(base: &str, tail: &str) -> String {
    format!(
        "{}/{}",
        base.trim_end_matches('/'),
        tail.trim_start_matches('/')
    )
}
