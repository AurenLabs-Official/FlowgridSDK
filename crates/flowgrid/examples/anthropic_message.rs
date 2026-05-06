//! Non-stream Anthropic message when `ANTHROPIC_API_KEY` is set.
//!
//! Run: `ANTHROPIC_API_KEY=sk-ant-... cargo run -p flowgrid --example anthropic_message --features anthropic`

use flowgrid::{Anthropic, AnthropicBuilder, CreateMessageRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let client: Anthropic = AnthropicBuilder::from_env()?.build()?;
    let body = CreateMessageRequest {
        model: "claude-3-5-haiku-20241022".into(),
        max_tokens: 64,
        messages: vec![serde_json::json!({"role": "user", "content": "Say hi in one sentence."})],
        stream: None,
        extra: Default::default(),
    };
    let msg = client.messages().create(&body).await?;
    println!("{}", msg.text_concat().unwrap_or_default());
    Ok(())
}
