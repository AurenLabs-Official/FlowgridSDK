//! Stream chat completions when `OPENAI_API_KEY` is set.
//!
//! Run: `OPENAI_API_KEY=sk-... cargo run -p flowgrid --example openai_chat_stream --features openai`

use flowgrid::{ClientBuilder, CreateChatCompletionRequest, OpenAI};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let client: OpenAI = ClientBuilder::from_env()?.build()?;
    let req = CreateChatCompletionRequest {
        model: "gpt-4o-mini".into(),
        messages: vec![CreateChatCompletionRequest::user_message(
            "Count from 1 to 3 slowly.",
        )],
        stream: Some(true),
        extra: Default::default(),
    };
    let (sse, _meta) = client.chat().completions().create_stream(&req).await?;
    let mut events = sse.into_unpin_event_stream();
    while let Some(item) = events.next().await {
        let ev = item?;
        if ev.data.trim() == "[DONE]" {
            break;
        }
        println!("{}", ev.data);
    }
    Ok(())
}
