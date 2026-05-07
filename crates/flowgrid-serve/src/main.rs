//! OpenAI-shaped HTTP server (`/v1/chat/completions` + `/v1/responses`).
//!
//! ```text
//! cargo run -p flowgrid-serve
//! ```

use axum::response::IntoResponse;
use axum::routing::post;
use axum::Router;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::TraceLayer;

mod auth;
mod engine;
mod error;
mod handlers;
mod ratelimit;
mod scheduler;
mod sse;
mod types;

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

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let auth_cfg = auth::AuthConfig::from_env();
    let max_body = std::env::var("FLOWGRID_SERVE_MAX_BODY_BYTES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1024 * 1024);
    let rps = std::env::var("FLOWGRID_SERVE_RPS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(32);
    let rate_state = ratelimit::RateLimitState::new(rps);
    let llm = match engine::LocalLlm::from_env() {
        Ok(o) => o,
        Err(e) => {
            tracing::warn!("FLOWGRID_SERVE_CHECKPOINT: {e}; falling back to tokenizer/echo mode");
            None
        }
    };
    let scheduler = scheduler::Scheduler::start(
        scheduler::SchedulerConfig {
            queue_depth: std::env::var("FLOWGRID_SERVE_QUEUE_DEPTH")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(64),
            request_timeout_ms: std::env::var("FLOWGRID_SERVE_REQUEST_TIMEOUT_MS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(10_000),
        },
        llm,
    );
    let state = Arc::new(AppState {
        scheduler,
        auth: auth_cfg,
        rate: rate_state,
    });
    let app = Router::new()
        .route("/health", axum::routing::get(health))
        .route("/ready", axum::routing::get(ready))
        .route(
            "/v1/chat/completions",
            post(handlers::chat::chat_completions),
        )
        .route("/v1/responses", post(handlers::responses::responses))
        .with_state(state)
        .layer(RequestBodyLimitLayer::new(max_body))
        .layer(TraceLayer::new_for_http());
    let addr: SocketAddr = "127.0.0.1:9000".parse().expect("addr");
    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind");
    tracing::info!("flowgrid-serve listening on http://{addr}/v1/chat/completions");
    axum::serve(listener, app).await.expect("serve");
}
