//! Optional typed parsing for OpenAI **streaming** `data:` JSON: chat completions and Responses API.
//! For Anthropic messages streaming, see `stream_typing_clu` (features **`anthropic`** +
//! **`stream-types`**).
//!
//! Requires Cargo features **`openai`** and **`stream-types`**.

use serde::Deserialize;
use thiserror::Error;

/// Limits for [`accumulate_openai_chat_visible_text_into`] and
/// [`accumulate_openai_response_visible_text_into`].
///
/// Defaults are **not** unbounded: adjust explicitly for long streams.
#[derive(Debug, Clone)]
pub struct OpenAiStreamTextLimits {
    /// Maximum UTF-8 characters appended in total.
    pub max_chars: usize,
    /// Maximum parsed stream events processed (each successful parse counts once).
    pub max_events: usize,
}

impl Default for OpenAiStreamTextLimits {
    fn default() -> Self {
        Self {
            max_chars: 256 * 1024,
            max_events: 50_000,
        }
    }
}

/// Bounded streaming text accumulation error.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum StreamTextAccumulateError {
    #[error("exceeded max_chars ({0})")]
    MaxChars(usize),
    #[error("exceeded max_events ({0})")]
    MaxEvents(usize),
}

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

/// Append visible assistant **string** `delta.content` fragments from one chat completion chunk.
pub fn accumulate_openai_chat_visible_text_into(
    out: &mut String,
    chunk: &OpenAiChatCompletionChunk,
    limits: &OpenAiStreamTextLimits,
    events_seen: &mut usize,
) -> Result<(), StreamTextAccumulateError> {
    *events_seen = events_seen.saturating_add(1);
    if *events_seen > limits.max_events {
        return Err(StreamTextAccumulateError::MaxEvents(limits.max_events));
    }
    for ch in &chunk.choices {
        if let Some(s) = ch.delta.get("content").and_then(|v| v.as_str()) {
            if out.len().saturating_add(s.len()) > limits.max_chars {
                return Err(StreamTextAccumulateError::MaxChars(limits.max_chars));
            }
            out.push_str(s);
        }
    }
    Ok(())
}

/// Append visible text from a Responses stream line (e.g. `response.output_text.delta` with `delta`).
pub fn accumulate_openai_response_visible_text_into(
    out: &mut String,
    line: &OpenAiResponseStreamLine,
    limits: &OpenAiStreamTextLimits,
    events_seen: &mut usize,
) -> Result<(), StreamTextAccumulateError> {
    *events_seen = events_seen.saturating_add(1);
    if *events_seen > limits.max_events {
        return Err(StreamTextAccumulateError::MaxEvents(limits.max_events));
    }
    let take_delta = matches!(
        line.line_type.as_str(),
        "response.output_text.delta" | "response.text.delta"
    );
    if take_delta {
        if let Some(s) = line.extra.get("delta").and_then(|v| v.as_str()) {
            if out.len().saturating_add(s.len()) > limits.max_chars {
                return Err(StreamTextAccumulateError::MaxChars(limits.max_chars));
            }
            out.push_str(s);
        }
    }
    Ok(())
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

    #[test]
    fn accumulate_chat_respects_max_chars() {
        let raw = r#"{"choices":[{"delta":{"content":"abcd"}}]}"#;
        let chunk = serde_json::from_str::<OpenAiChatCompletionChunk>(raw).unwrap();
        let limits = OpenAiStreamTextLimits {
            max_chars: 3,
            max_events: 10,
        };
        let mut out = String::new();
        let mut n = 0;
        let e = accumulate_openai_chat_visible_text_into(&mut out, &chunk, &limits, &mut n)
            .unwrap_err();
        assert_eq!(e, StreamTextAccumulateError::MaxChars(3));
    }

    #[test]
    fn accumulate_response_text_delta() {
        let raw = r#"{"type":"response.output_text.delta","delta":"x"}"#;
        let line = serde_json::from_str::<OpenAiResponseStreamLine>(raw).unwrap();
        let limits = OpenAiStreamTextLimits::default();
        let mut out = String::new();
        let mut n = 0;
        accumulate_openai_response_visible_text_into(&mut out, &line, &limits, &mut n).unwrap();
        assert_eq!(out, "x");
    }
}
