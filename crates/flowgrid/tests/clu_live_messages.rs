#![cfg(feature = "anthropic")]

use flowgrid::{AnthropicBuilder, CreateMessageRequest};
use serde_json::json;
use std::env;

#[tokio::test]
#[ignore]
async fn live_messages_smoke() {
    env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY");
    let client = AnthropicBuilder::default()
        .api_key(env::var("ANTHROPIC_API_KEY").unwrap())
        .build()
        .unwrap();
    let req = CreateMessageRequest {
        model: env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-3-5-haiku-20241022".into()),
        max_tokens: 32,
        messages: vec![json!({"role":"user","content":"Reply with exactly: ok"})],
        stream: Some(false),
        extra: serde_json::Map::new(),
    };
    let msg = client.messages().create(&req).await.unwrap();
    assert!(!msg.id.is_empty());
}
