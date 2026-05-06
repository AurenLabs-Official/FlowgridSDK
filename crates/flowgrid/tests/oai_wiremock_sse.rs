#![cfg(feature = "openai")]

use flowgrid::{ClientBuilder, CreateChatCompletionRequest};
use futures::TryStreamExt;

#[tokio::test]
async fn wiremock_post_stream_returns_sse_bytes() {
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    let sse = "event: delta\ndata: {\"id\":\"1\"}\n\n";
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

    let transport = client.transport();
    let (stream, meta) = transport
        .post_stream_bytes("chat/completions", &body)
        .await
        .unwrap();
    assert!(meta.status.is_success());

    let collected: Vec<u8> = stream
        .try_fold(Vec::new(), |mut acc, chunk| async move {
            acc.extend_from_slice(&chunk);
            Ok(acc)
        })
        .await
        .unwrap();
    let text = String::from_utf8(collected).unwrap();
    assert!(text.contains("data:"));
    assert!(text.contains(r#""id":"1""#));

    let (mut sse_dec, _) = client
        .chat()
        .completions()
        .create_stream(&body)
        .await
        .unwrap();
    let ev = sse_dec.next_event().await.unwrap().unwrap();
    assert_eq!(ev.event, "delta");
    assert!(ev.data.contains(r#""id":"1"#));
}
