use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::openai_compat::openai_error_value;

#[derive(Debug, thiserror::Error)]
#[allow(dead_code)]
pub enum ServeError {
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("internal error: {0}")]
    Internal(String),
}

impl IntoResponse for ServeError {
    fn into_response(self) -> Response {
        let (status, typ, code, msg) = match &self {
            Self::BadRequest(m) => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "bad_request",
                m.as_str(),
            ),
            Self::Internal(m) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "internal",
                m.as_str(),
            ),
        };
        (status, Json(openai_error_value(typ, code, msg))).into_response()
    }
}
