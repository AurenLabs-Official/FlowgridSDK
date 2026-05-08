//! OpenAI-only helpers; compiled only with `feature = "openai"` (see `internal/mod.rs`).

use serde::Deserialize;
use serde_json::Value;

/// Hard caps for [`AssistantsClient::list_all_typed`](crate::AssistantsClient::list_all_typed)
/// and similar **cursor walks** over [`ListPage`].
///
/// **`max_pages`** bounds how many HTTP list calls are made (each page is one request).
/// **`max_items`**, when set, stops collection once that many rows have been appended (across pages).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ListPagesLimits {
    /// Upper bound on list requests (must be ≥ 1).
    pub max_pages: u32,
    /// Optional cap on total items returned; `None` means only `max_pages` limits volume.
    pub max_items: Option<u32>,
}

impl Default for ListPagesLimits {
    fn default() -> Self {
        Self {
            max_pages: 100,
            max_items: Some(10_000),
        }
    }
}

impl ListPagesLimits {
    /// No per-item cap (still bounded by `max_pages`).
    pub fn pages_only(max_pages: u32) -> Self {
        Self {
            max_pages: max_pages.max(1),
            max_items: None,
        }
    }
}

/// Generic list/cursor page matching common OpenAI list responses.
#[derive(Debug, Clone, Deserialize)]
pub struct ListPage<T> {
    /// Listed items.
    pub data: Vec<T>,
    /// Object type marker.
    pub object: Option<String>,
    /// First id in page when present.
    #[serde(rename = "first_id")]
    pub first_id: Option<String>,
    /// Last id in page when present.
    #[serde(rename = "last_id")]
    pub last_id: Option<String>,
    /// Whether more pages exist.
    #[serde(rename = "has_more")]
    pub has_more: Option<bool>,
    #[serde(flatten)]
    #[serde(default)]
    pub extra: serde_json::Map<String, Value>,
}

impl<T> ListPage<T> {
    /// Returns true when API reports additional pages.
    pub fn has_more(&self) -> bool {
        self.has_more.unwrap_or(false)
    }

    /// Cursor value to pass as `after` for ascending list order when more pages exist.
    ///
    /// Uses `last_id` when [`Self::has_more`] is true; returns `None` if the API omits `last_id`.
    pub fn after_cursor(&self) -> Option<String> {
        if !self.has_more() {
            return None;
        }
        self.last_id.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_page_after_cursor() {
        let p: ListPage<()> = serde_json::from_value(serde_json::json!({
            "object": "list",
            "data": [],
            "last_id": "x",
            "has_more": true
        }))
        .unwrap();
        assert_eq!(p.after_cursor(), Some("x".into()));
    }

    #[test]
    fn list_page_unknown_field_in_extra() {
        let p: ListPage<String> = serde_json::from_value(serde_json::json!({
            "data": [],
            "future_meta": 1
        }))
        .unwrap();
        assert_eq!(p.extra.get("future_meta").and_then(|v| v.as_u64()), Some(1));
    }
}
