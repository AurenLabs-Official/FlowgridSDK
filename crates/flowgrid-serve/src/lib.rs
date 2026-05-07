//! HTTP router and [`AppState`] for `flowgrid-serve` (library surface for tests and embedding).

use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::Router;
use std::sync::Arc;

mod auth;
mod backend;
pub mod completion;
pub mod engine;
pub mod error;
mod handlers;
pub mod openai_compat;
pub mod ratelimit;
pub mod scheduler;
mod sse;
mod types;

pub use auth::AuthConfig;

pub use completion::{CompletionMeta, PlainOutput, StreamPart};

#[derive(Clone)]
pub struct AppState {
    pub scheduler: scheduler::Scheduler,
    pub auth: auth::AuthConfig,
    pub rate: ratelimit::RateLimitState,
}

async fn health() -> impl IntoResponse {
    (axum::http::StatusCode::OK, "ok")
}

async fn ready() -> impl IntoResponse {
    (axum::http::StatusCode::OK, "ready")
}

/// Core routes (add tracing / body limit layers in the binary).
pub fn build_app(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route(
            "/v1/chat/completions",
            post(handlers::chat::chat_completions),
        )
        .route("/v1/responses", post(handlers::responses::responses))
        .with_state(state)
}
