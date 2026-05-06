//! Optional typed parsing for OpenAI **streaming** `data:` JSON: chat completions and Responses API.
//! For Anthropic messages streaming, see `stream_typing_clu` (features **`anthropic`** +
//! **`stream-types`**).
//!
//! Requires Cargo features **`openai`** and **`stream-types`**.

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

/// Parse a single `data:` line from an OpenAI **Responses** SSE stream.
///
/// Returns `Ok(None)` for empty data, `[DONE]`, or whitespace-only payloads.
pub fn parse_openai_response_stream_json(
    data: &str,
) -> Result<Option<OpenAiResponseStreamLine>, serde_json::Error> {
    let t = data.trim();
    if t.is_empty() || t == "[DONE]" {
        return Ok(None);
    }
    serde_json::from_str(t).map(Some)
}

/// Minimal typed envelope for Responses streaming events (`type` + forward-compatible extras).
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiResponseStreamLine {
    /// Event type, e.g. `response.output_item.added`.
    #[serde(rename = "type")]
    pub line_type: String,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
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

    #[test]
    fn parse_response_stream_line() {
        let raw = r#"{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1"}}"#;
        let v = parse_openai_response_stream_json(raw).unwrap().unwrap();
        assert_eq!(v.line_type, "response.output_item.added");
        assert_eq!(
            v.extra.get("output_index").and_then(|x| x.as_u64()),
            Some(0)
        );
    }
}
