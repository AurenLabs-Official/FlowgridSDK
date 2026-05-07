//! Streaming chat completions: final chunk includes `usage` from scheduler metadata.
use axum::body::to_bytes;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use flowgrid_serve::ratelimit::RateLimitState;
use flowgrid_serve::scheduler::{Scheduler, SchedulerConfig};
use flowgrid_serve::{build_app, AppState, AuthConfig};
use std::sync::Arc;
use tower::ServiceExt;

fn test_state() -> Arc<AppState> {
    let scheduler = Scheduler::start(
        SchedulerConfig {
            queue_depth: 4,
            request_timeout_ms: 5_000,
        },
        None,
    );
    Arc::new(AppState {
        scheduler,
        auth: AuthConfig {
            required: false,
            keys: vec![],
        },
        rate: RateLimitState::new(9999),
    })
}

#[tokio::test]
async fn chat_stream_ends_with_usage_in_sse() {
    let app = build_app(test_state());
    let body = serde_json::json!({
        "model": "m",
        "stream": true,
        "messages": [{ "role": "user", "content": "ab" }],
        "max_tokens": 4
    });
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = to_bytes(res.into_body(), 2 * 1024 * 1024).await.unwrap();
    let text = String::from_utf8_lossy(&bytes);
    assert!(
        text.contains("\"usage\"") && text.contains("prompt_tokens"),
        "expected usage in stream, got: {text}"
    );
    assert!(
        text.contains("[DONE]"),
        "expected SSE done sentinel, got: {text}"
    );
}
