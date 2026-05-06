#![cfg(feature = "anthropic")]

use flowgrid::AnthropicClientConfig;
use flowgrid::AnthropicHttpTransport;
use flowgrid::Message;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[tokio::test]
async fn retry_after_large_value_is_capped_messages() {
    use serde_json::json;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_cb = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "test-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_cb.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                ResponseTemplate::new(429)
                    .insert_header("retry-after", "600")
                    .set_body_string(r#"{"type":"error","error":{"message":"rate limit"}}"#)
            } else {
                ResponseTemplate::new(200).set_body_json(json!({
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude",
                    "stop_reason": "end_turn"
                }))
            }
        })
        .mount(&server)
        .await;

    let mut base = url::Url::parse(&server.uri()).unwrap();
    base.set_path("/v1");
    let config = AnthropicClientConfig {
        api_key: "test-key".into(),
        base_url: base,
        anthropic_version: "2023-06-01".into(),
        anthropic_beta: None,
        timeout: Duration::from_secs(30),
        max_retries: 2,
        user_agent_suffix: None,
        request_hook: None,
        retry_after_max: Duration::from_millis(400),
    };
    let t = AnthropicHttpTransport::new(config).unwrap();
    let body =
        json!({"model":"claude","max_tokens":10,"messages":[{"role":"user","content":"hi"}]});
    let t0 = Instant::now();
    let (_msg, meta): (Message, _) = t.post_json("messages", &body).await.unwrap();
    assert!(meta.status.is_success());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    let elapsed = t0.elapsed();
    assert!(
        elapsed < Duration::from_millis(1500),
        "expected capped retry sleep (~400ms), got {elapsed:?}"
    );
}
