#![cfg(feature = "openai")]

use flowgrid::OpenAiClientConfig;
use flowgrid::OpenAiHttpTransport;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[tokio::test]
async fn retries_on_429_then_succeeds() {
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
                    .insert_header("retry-after", "0")
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
        timeout: std::time::Duration::from_secs(5),
        max_retries: 2,
        user_agent_suffix: None,
        request_hook: None,
        #[cfg(feature = "webhooks")]
        webhook_secret: None,
    };
    let t = OpenAiHttpTransport::new(config).unwrap();
    let (body, meta): (serde_json::Value, _) = t.get_json("models").await.unwrap();
    assert_eq!(body["object"], "list");
    assert!(meta.status.is_success());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}
