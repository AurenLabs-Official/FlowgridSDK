//! Minimal OpenAI-shaped HTTP server (SSE line compatible with `flowgrid` streaming tests).
//!
//! ```text
//! cargo run -p flowgrid-serve
//! ```

use axum::http::header;
use axum::response::IntoResponse;
use axum::routing::post;
use axum::Json;
use axum::Router;
use serde::Deserialize;
use serde_json::json;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::trace::TraceLayer;
use uuid::Uuid;

#[derive(Clone, Default)]
struct AppState {
    _placeholder: (),
}

#[derive(Debug, Deserialize)]
struct ChatReq {
    model: String,
    stream: Option<bool>,
}

async fn chat_completions(
    axum::extract::State(_st): axum::extract::State<Arc<AppState>>,
    Json(body): Json<ChatReq>,
) -> axum::response::Response {
    if body.stream == Some(true) {
        let id = Uuid::new_v4();
        let sse = format!(
            "event: delta\ndata: {{\"id\":\"{}\",\"object\":\"chat.completion.chunk\"}}\n\n",
            id
        );
        ([(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")], sse).into_response()
    } else {
        (
            axum::http::StatusCode::OK,
            Json(json!({
                "id": Uuid::new_v4().to_string(),
                "object": "chat.completion",
                "model": body.model,
                "choices": [{ "index": 0, "message": { "role": "assistant", "content": "flowgrid-serve stub" } }]
            })),
        )
            .into_response()
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let state = Arc::new(AppState::default());
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
        .layer(TraceLayer::new_for_http());
    let addr: SocketAddr = "127.0.0.1:9000".parse().expect("addr");
    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind");
    tracing::info!("flowgrid-serve listening on http://{addr}/v1/chat/completions");
    axum::serve(listener, app).await.expect("serve");
}
