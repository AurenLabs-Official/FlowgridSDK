//! Stream the OpenAI **Responses** API and accumulate **visible** output text with hard limits.
//!
//! Set **`OPENAI_API_KEY`**. This uses [`OpenAiStreamTextLimits`] so memory stays bounded
//! (`max_chars`, `max_events`); tune them for your workload.
//!
//! Run:
//! ```text
//! OPENAI_API_KEY=sk-... cargo run -p flowgrid --example openai_responses_stream_accumulate \
//!   --features openai,stream-types
//! ```

use flowgrid::{
    accumulate_openai_response_visible_text_into, parse_openai_response_stream_json,
    ClientBuilder, CreateResponseRequest, OpenAiStreamTextLimits,
};
use futures::StreamExt;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let client = ClientBuilder::from_env()?.build()?;

    let req = CreateResponseRequest {
        model: "gpt-4o-mini".into(),
        instructions: Some("Reply with a single short greeting.".into()),
        input: Some(json!("Hello")),
        stream: Some(true),
        extra: Default::default(),
    };

    let (sse, _meta) = client.responses().create_stream(&req).await?;
    let mut events = sse.into_unpin_event_stream();

    let limits = OpenAiStreamTextLimits {
        max_chars: 256 * 1024,
        max_events: 50_000,
    };
    let mut accumulated = String::new();
    let mut parsed_events = 0usize;

    while let Some(item) = events.next().await {
        let ev = item?;
        if ev.data.trim() == "[DONE]" {
            break;
        }
        if let Some(line) = parse_openai_response_stream_json(&ev.data)? {
            accumulate_openai_response_visible_text_into(
                &mut accumulated,
                &line,
                &limits,
                &mut parsed_events,
            )?;
        }
    }

    if accumulated.is_empty() {
        println!("(no visible text deltas collected; try another model or prompt)");
    } else {
        println!("{}", accumulated);
    }
    Ok(())
}
