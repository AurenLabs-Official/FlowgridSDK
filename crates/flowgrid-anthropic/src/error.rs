use reqwest::header::HeaderMap;
use reqwest::StatusCode;
use serde::Deserialize;
use thiserror::Error;

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Top-level error type.
#[derive(Debug, Error)]
pub enum Error {
    /// HTTP or API error response.
    #[error(transparent)]
    Api(#[from] ApiError),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Http(#[from] reqwest::Error),

    #[error(transparent)]
    Url(#[from] url::ParseError),

    #[error("SSE parse error: {0}")]
    Sse(String),

    #[error("{0}")]
    Config(String),
}

/// Structured API error (Anthropic JSON).
#[derive(Debug, Clone)]
pub struct ApiError {
    /// HTTP status.
    pub status: StatusCode,
    /// Parsed body when recognizable.
    pub body: Option<ErrorBody>,
    pub raw_body: Option<String>,
    pub request_id: Option<String>,
    pub headers: HeaderMap,
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Anthropic API error HTTP {}", self.status)?;
        if let Some(ref id) = self.request_id {
            write!(f, " (request_id={id})")?;
        }
        if let Some(ref b) = self.body {
            write!(f, ": {b:?}")?;
        } else if let Some(ref raw) = self.raw_body {
            let snippet: String = raw.chars().take(512).collect();
            write!(f, ": {snippet}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ApiError {}

/// Common error envelope: `{"type":"error","error":{...}}`.
#[derive(Debug, Clone, Deserialize)]
pub struct ErrorBody {
    #[serde(rename = "type")]
    pub type_: Option<String>,
    pub error: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiErrorKind {
    BadRequest,
    Authentication,
    PermissionDenied,
    NotFound,
    RateLimit,
    Overloaded,
    InternalServerError,
    Other,
}

impl ApiError {
    pub fn kind(&self) -> ApiErrorKind {
        let s = self.status.as_u16();
        match s {
            400 => ApiErrorKind::BadRequest,
            401 => ApiErrorKind::Authentication,
            403 => ApiErrorKind::PermissionDenied,
            404 => ApiErrorKind::NotFound,
            429 => ApiErrorKind::RateLimit,
            529 => ApiErrorKind::Overloaded,
            n if n >= 500 => ApiErrorKind::InternalServerError,
            _ => ApiErrorKind::Other,
        }
    }
}
