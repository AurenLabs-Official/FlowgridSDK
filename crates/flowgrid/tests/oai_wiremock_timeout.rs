#![cfg(feature = "openai")]

use flowgrid::{ExecuteOptions, OpenAiClientConfig, OpenAiHttpTransport};
#[cfg(feature = "anthropic")]
use flowgrid::OpenAiError;
#[cfg(not(feature = "anthropic"))]
use flowgrid::Error as OpenAiError;
use std::time::Duration;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn per_call_timeout_fires_before_slow_response() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(30)))
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
        timeout: Duration::from_secs(60),
        max_retries: 0,
        user_agent_suffix: None,
        request_hook: None,
        retry_after_max: Duration::from_millis(2000),
        #[cfg(feature = "webhooks")]
        webhook_secret: None,
    };
    let t = OpenAiHttpTransport::new(config).unwrap();
    let opts = ExecuteOptions {
        timeout: Some(Duration::from_millis(200)),
    };
    let err = t
        .get_json_with_options::<serde_json::Value>("models", opts)
        .await
        .expect_err("expected timeout");
    assert!(matches!(err, OpenAiError::Http(_)));
}
