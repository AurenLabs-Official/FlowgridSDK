#![cfg(feature = "openai")]

use flowgrid::{ClientBuilder, CreateChatCompletionRequest};

#[tokio::test]
async fn request_pre_send_hook_adds_header() {
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer k"))
        .and(header("x-flowgrid-test", "1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "c1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": []
        })))
        .mount(&server)
        .await;

    let base = format!("{}/v1", server.uri());
    let client = ClientBuilder::new()
        .api_key("k")
        .base_url(base)
        .max_retries(0)
        .request_pre_send_hook(|rb| rb.header("x-flowgrid-test", "1"))
        .build()
        .unwrap();

    let body = CreateChatCompletionRequest {
        model: "m".into(),
        messages: vec![],
        stream: Some(false),
        extra: Default::default(),
    };
    let completion = client.chat().completions().create(&body).await.unwrap();
    assert_eq!(completion.id, "c1");
}
