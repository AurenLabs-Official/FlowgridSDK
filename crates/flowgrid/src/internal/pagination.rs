use serde::Deserialize;

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
}

impl<T> ListPage<T> {
    /// Returns true when API reports additional pages.
    pub fn has_more(&self) -> bool {
        self.has_more.unwrap_or(false)
    }
}
