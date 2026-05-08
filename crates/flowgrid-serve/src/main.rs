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

#[derive(Debug, Clone, Copy)]
enum DeploymentProfile {
    Local,
    Cloud,
    Hybrid,
}

impl DeploymentProfile {
    fn from_env() -> Self {
        match std::env::var("FLOWGRID_DEPLOYMENT_PROFILE")
            .unwrap_or_else(|_| "local".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "cloud" => Self::Cloud,
            "hybrid" => Self::Hybrid,
            _ => Self::Local,
        }
    }
}

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
    let profile = DeploymentProfile::from_env();
    let (default_rps, default_burst, default_workers, default_queue_depth) = match profile {
        DeploymentProfile::Local => (96_u64, 128_u64, 1_usize, 64_usize),
        DeploymentProfile::Cloud => (320_u64, 512_u64, 4_usize, 512_usize),
        DeploymentProfile::Hybrid => (160_u64, 224_u64, 2_usize, 256_usize),
    };
    let max_body = std::env::var("FLOWGRID_SERVE_MAX_BODY_BYTES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1024 * 1024);
    let rps = std::env::var("FLOWGRID_SERVE_RPS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default_rps);
    let burst = std::env::var("FLOWGRID_SERVE_BURST")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default_burst);
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
                .or_else(|| {
                    std::env::var("FLOWGRID_SERVE_QUEUE")
                        .ok()
                        .and_then(|v| v.parse::<usize>().ok())
                })
                .unwrap_or(default_queue_depth),
            worker_threads: std::env::var("FLOWGRID_SERVE_WORKERS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(default_workers),
            request_timeout_ms: std::env::var("FLOWGRID_SERVE_REQUEST_TIMEOUT_MS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(10_000),
            stream_buffer: std::env::var("FLOWGRID_SERVE_STREAM_BUFFER")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(64),
            max_new_tokens: std::env::var("FLOWGRID_SERVE_MAX_NEW_TOKENS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|v| *v > 0),
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
    tracing::info!(
        profile = ?profile,
        "flowgrid-serve listening on http://{addr}/v1/chat/completions"
    );
    axum::serve(listener, app).await.expect("serve");
}
