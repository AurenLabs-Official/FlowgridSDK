//! Contract-style checks for OpenAI-adjacent JSON from `flowgrid-serve`.

use axum::body::to_bytes;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use flowgrid_serve::ratelimit::RateLimitState;
use flowgrid_serve::scheduler::{Scheduler, SchedulerConfig};
use flowgrid_serve::{build_app, AppState, AuthConfig};
use serde_json::Value;
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
async fn chat_completion_includes_usage_and_finish_reason() {
    let app = build_app(test_state());
    let body = serde_json::json!({
        "model": "test-model",
        "stream": false,
        "messages": [{ "role": "user", "content": "hello" }],
        "max_tokens": 8
    });
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = to_bytes(res.into_body(), 1024 * 1024).await.unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert!(v["usage"]["prompt_tokens"].as_u64().is_some());
    assert!(v["usage"]["completion_tokens"].as_u64().is_some());
    assert!(v["usage"]["total_tokens"].as_u64().is_some());
}

#[tokio::test]
async fn unauthorized_returns_openai_error_json() {
    let scheduler = Scheduler::start(
        SchedulerConfig {
            queue_depth: 4,
            request_timeout_ms: 5_000,
        },
        None,
    );
    let state = Arc::new(AppState {
        scheduler,
        auth: AuthConfig {
            required: true,
            keys: vec!["secret".to_string()],
        },
        rate: RateLimitState::new(9999),
    });
    let app = build_app(state);
    let body = serde_json::json!({
        "model": "m",
        "stream": false,
        "messages": [{ "role": "user", "content": "x" }],
    });
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
    let bytes = to_bytes(res.into_body(), 1024 * 1024).await.unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["error"]["type"], "authentication_error");
    assert!(v["error"]["message"].as_str().is_some());
}

#[tokio::test]
async fn api_key_header_authorizes_like_bearer() {
    let scheduler = Scheduler::start(
        SchedulerConfig {
            queue_depth: 4,
            request_timeout_ms: 5_000,
        },
        None,
    );
    let state = Arc::new(AppState {
        scheduler,
        auth: AuthConfig {
            required: true,
            keys: vec!["secret".to_string()],
        },
        rate: RateLimitState::new(9999),
    });
    let app = build_app(state);
    let body = serde_json::json!({
        "model": "m",
        "stream": false,
        "messages": [{ "role": "user", "content": "x" }],
        "max_tokens": 4
    });
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("api-key", "secret")
        .body(Body::from(body.to_string()))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[tokio::test]
async fn response_non_stream_has_status_and_usage() {
    let app = build_app(test_state());
    let body = serde_json::json!({
        "model": "m",
        "stream": false,
        "input": "ping",
        "max_tokens": 4
    });
    let req = Request::builder()
        .method("POST")
        .uri("/v1/responses")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = to_bytes(res.into_body(), 1024 * 1024).await.unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["object"], "response");
    assert_eq!(v["status"], "completed");
    assert_eq!(v["finish_reason"], "stop");
    assert!(v["usage"]["input_tokens"].as_u64().is_some());
}
