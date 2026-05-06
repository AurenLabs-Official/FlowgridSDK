#[cfg(feature = "openai")]
pub mod oai {
    use reqwest::header::HeaderMap;
    use reqwest::StatusCode;
    use serde::Deserialize;
    use thiserror::Error;

    /// Crate-wide result alias.
    pub type Result<T> = std::result::Result<T, Error>;

    /// Top-level error type (transport, HTTP, JSON, etc.).
    #[derive(Debug, Error)]
    pub enum Error {
        /// HTTP layer or non-success response from the API.
        #[error(transparent)]
        Api(#[from] ApiError),

        /// JSON encode/decode error.
        #[error(transparent)]
        Json(#[from] serde_json::Error),

        /// Reqwest error (includes URL parse, TLS, etc.).
        #[error(transparent)]
        Http(#[from] reqwest::Error),

        /// URL construction error.
        #[error(transparent)]
        Url(#[from] url::ParseError),

        /// SSE parsing error.
        #[error("SSE parse error: {0}")]
        Sse(String),

        /// WebSocket / realtime error (feature `realtime`).
        #[cfg(feature = "realtime")]
        #[error(transparent)]
        Ws(#[from] tokio_tungstenite::tungstenite::Error),

        /// Invalid webhook signature or payload (feature `webhooks`).
        #[cfg(feature = "webhooks")]
        #[error("webhook error: {0}")]
        Webhook(String),

        /// Generic client configuration error.
        #[error("{0}")]
        Config(String),
    }

    /// Structured API error from OpenAI responses (4xx/5xx with body).
    /// `Display` truncates raw body text to 512 characters if no structured body is present.
    #[derive(Debug, Clone)]
    pub struct ApiError {
        /// HTTP status.
        pub status: StatusCode,
        /// Parsed error payload when present.
        pub body: Option<ErrorObject>,
        /// Raw response body string (truncated in Display).
        pub raw_body: Option<String>,
        /// `x-request-id` when present.
        pub request_id: Option<String>,
        /// Response headers.
        pub headers: HeaderMap,
    }

    impl std::fmt::Display for ApiError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "OpenAI API error HTTP {}", self.status)?;
            if let Some(ref id) = self.request_id {
                write!(f, " (request_id={id})")?;
            }
            if let Some(ref o) = self.body {
                write!(f, ": {o:?}")?;
            } else if let Some(ref raw) = self.raw_body {
                let snippet: String = raw.chars().take(512).collect();
                write!(f, ": {snippet}")?;
            }
            Ok(())
        }
    }

    impl std::error::Error for ApiError {}

    /// Common OpenAI error JSON object.
    #[derive(Debug, Clone, Deserialize)]
    pub struct ErrorObject {
        /// Error envelope with nested fields.
        pub error: Option<ErrorDetail>,
    }

    /// Inner error details.
    #[derive(Debug, Clone, Deserialize)]
    pub struct ErrorDetail {
        /// Machine-readable error code.
        pub code: Option<String>,
        /// Human-readable message.
        pub message: Option<String>,
        /// Parameter name when validation fails.
        pub param: Option<String>,
        /// Error type string from API.
        #[serde(rename = "type")]
        pub type_: Option<String>,
    }

    /// Typed classification of HTTP failures (mirrors openai-node error subclasses).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ApiErrorKind {
        /// 400
        BadRequest,
        /// 401
        Authentication,
        /// 403
        PermissionDenied,
        /// 404
        NotFound,
        /// 422
        UnprocessableEntity,
        /// 429
        RateLimit,
        /// >=500
        InternalServerError,
        /// Other 4xx/5xx.
        Other,
    }

    impl ApiError {
        /// Classify by status code.
        pub fn kind(&self) -> ApiErrorKind {
            let s = self.status.as_u16();
            match s {
                400 => ApiErrorKind::BadRequest,
                401 => ApiErrorKind::Authentication,
                403 => ApiErrorKind::PermissionDenied,
                404 => ApiErrorKind::NotFound,
                422 => ApiErrorKind::UnprocessableEntity,
                429 => ApiErrorKind::RateLimit,
                s if s >= 500 => ApiErrorKind::InternalServerError,
                _ => ApiErrorKind::Other,
            }
        }
    }
}

#[cfg(feature = "anthropic")]
pub mod clu {
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
    /// `Display` truncates raw body text to 512 characters when no structured body is present.
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
}
