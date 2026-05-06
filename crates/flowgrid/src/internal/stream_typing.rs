//! Optional typed parsing for OpenAI chat completion **streaming** JSON lines (`data:` payloads).
//!
//! Requires Cargo features **`openai`** and **`stream-types`**. Complements raw `SseEvent.data`
//! strings from the OpenAI SSE decoder.

use serde::Deserialize;

/// One chunk from a streaming chat completion (`object` is usually `chat.completion.chunk`).
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiChatCompletionChunk {
    pub id: Option<String>,
    pub object: Option<String>,
    #[serde(default)]
    pub choices: Vec<OpenAiChatChunkChoice>,
}

/// Choice delta in a streaming chunk.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiChatChunkChoice {
    pub index: Option<u32>,
    #[serde(default)]
    pub delta: serde_json::Value,
    pub finish_reason: Option<serde_json::Value>,
}

/// Parse a single `data:` line body from an SSE event (not including the `data:` prefix).
///
/// Returns `Ok(None)` for empty data, `[DONE]`, or whitespace-only payloads.
pub fn parse_openai_chat_stream_json(
    data: &str,
) -> Result<Option<OpenAiChatCompletionChunk>, serde_json::Error> {
    let t = data.trim();
    if t.is_empty() || t == "[DONE]" {
        return Ok(None);
    }
    serde_json::from_str(t).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_done_yields_none() {
        assert!(parse_openai_chat_stream_json("[DONE]").unwrap().is_none());
    }

    #[test]
    fn parse_chunk_json() {
        let raw = r#"{"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hi"}}]}"#;
        let v = parse_openai_chat_stream_json(raw).unwrap().unwrap();
        assert_eq!(v.id.as_deref(), Some("c1"));
        assert_eq!(v.choices.len(), 1);
    }
}
