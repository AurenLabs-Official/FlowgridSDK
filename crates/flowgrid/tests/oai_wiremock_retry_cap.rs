#![cfg(feature = "openai")]

use flowgrid::OpenAiClientConfig;
use flowgrid::OpenAiHttpTransport;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[tokio::test]
async fn retry_after_large_value_is_capped() {
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_cb = calls.clone();
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_cb.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                ResponseTemplate::new(429)
                    .insert_header("retry-after", "600")
                    .set_body_string(r#"{"error":{"message":"rate limited"}}"#)
            } else {
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "object": "list",
                    "data": []
                }))
            }
        })
        .mount(&server)
        .await;

    let mut base = url::Url::parse(&server.uri()).unwrap();
    base.set_path("/v1");
    let config = OpenAiClientConfig {
        api_key: "test-key".into(),
        base_url: base,
        use_api_key_header: false,
        default_query: Vec::new(),
        org_id: None,
        project_id: None,
        timeout: Duration::from_secs(30),
        max_retries: 2,
        user_agent_suffix: None,
        request_hook: None,
        retry_after_max: Duration::from_millis(400),
        retry_if_response_status: None,
        #[cfg(feature = "webhooks")]
        webhook_secret: None,
    };
    let t = OpenAiHttpTransport::new(config).unwrap();
    let t0 = Instant::now();
    let (_body, meta): (serde_json::Value, _) = t.get_json("models").await.unwrap();
    assert!(meta.status.is_success());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    let elapsed = t0.elapsed();
    assert!(
        elapsed < Duration::from_millis(1500),
        "expected capped retry sleep (~400ms), got {elapsed:?}"
    );
}
