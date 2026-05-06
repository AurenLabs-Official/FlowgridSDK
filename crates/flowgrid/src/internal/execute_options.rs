//! Per-request execution overrides (timeouts, future knobs).

use std::time::Duration;

/// Optional settings for a single HTTP call (does not replace the provider client `timeout` field).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExecuteOptions {
    /// Overrides the HTTP client timeout for this attempt only (including retries).
    pub timeout: Option<Duration>,
}

impl ExecuteOptions {
    /// No overrides (defaults only).
    pub const fn new() -> Self {
        Self { timeout: None }
    }
}
