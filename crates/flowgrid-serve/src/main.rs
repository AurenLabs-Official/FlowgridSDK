//! OpenAI-shaped HTTP server (`/v1/chat/completions` + `/v1/responses`).
//!
//! ```text
//! cargo run -p flowgrid-serve
//! ```

use flowgrid_serve::scheduler::{Scheduler, SchedulerConfig};
use flowgrid_serve::{build_app, AppState, AuthConfig};
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::TraceLayer;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    #[cfg(not(feature = "gpu-wgpu"))]
    if flowgrid_device::gpu_requested_in_env() {
        tracing::warn!(
            "FLOWGRID_DEVICE requests GPU; rebuild with `cargo build -p flowgrid-serve --features gpu-wgpu` to enable Burn Wgpu"
        );
    }
    let auth_cfg = AuthConfig::from_env();
    let max_body = std::env::var("FLOWGRID_SERVE_MAX_BODY_BYTES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1024 * 1024);
    let rps = std::env::var("FLOWGRID_SERVE_RPS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(32);
    let burst = std::env::var("FLOWGRID_SERVE_BURST")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(rps.max(1));
    let rate_state = flowgrid_serve::ratelimit::RateLimitState::with_capacity(rps, burst.max(1));
    let llm = match flowgrid_serve::engine::LocalLlm::from_env() {
        Ok(o) => o,
        Err(e) => {
            tracing::warn!("FLOWGRID_SERVE_CHECKPOINT: {e}; falling back to tokenizer/echo mode");
            None
        }
    };
    let scheduler = Scheduler::start(
        SchedulerConfig {
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
    let app = build_app(state)
        .layer(RequestBodyLimitLayer::new(max_body))
        .layer(TraceLayer::new_for_http());
    let addr: SocketAddr = "127.0.0.1:9000".parse().expect("addr");
    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind");
    tracing::info!("flowgrid-serve listening on http://{addr}/v1/chat/completions");
    axum::serve(listener, app).await.expect("serve");
}
