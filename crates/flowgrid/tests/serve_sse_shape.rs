//! Ensures the HTTP SDK parses SSE shaped like `flowgrid-serve` chat streaming.
#![cfg(feature = "openai")]

use flowgrid::{ClientBuilder, CreateChatCompletionRequest};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn wiremock_flowgrid_serve_style_chat_stream() {
    let server = MockServer::start().await;
    let chunk = r#"{"id":"u","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"content":"he"}}]}"#;
    let sse = format!("data: {chunk}\n\ndata: [DONE]\n\n");

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer k"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream; charset=utf-8")
                .set_body_string(sse),
        )
        .mount(&server)
        .await;

    let base = format!("{}/v1", server.uri());
    let client = ClientBuilder::default()
        .api_key("k")
        .base_url(base)
        .max_retries(0)
        .build()
        .unwrap();

    let body = CreateChatCompletionRequest {
        model: "m".into(),
        messages: vec![],
        stream: Some(true),
        extra: Default::default(),
    };

    let (mut sse_dec, _) = client
        .chat()
        .completions()
        .create_stream(&body)
        .await
        .unwrap();

    let mut saw_he = false;
    while let Ok(Some(ev)) = sse_dec.next_event().await {
        if ev.data.trim() == "[DONE]" {
            break;
        }
        if ev.data.contains("\"he\"") || ev.data.contains("he") {
            saw_he = true;
        }
    }
    assert!(saw_he, "expected delta content in SSE payload");
}
