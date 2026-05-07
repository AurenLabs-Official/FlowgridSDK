//! Map scheduler stream items to SSE `Bytes` chunks (OpenAI-shaped errors use `event: error`).

use bytes::Bytes;
use serde_json::json;

use crate::completion::StreamPart;
use crate::openai_compat::{chat_usage_tokens, openai_error_value, responses_usage_tokens};
use crate::sse;

/// Turn one scheduler item into one or more SSE byte chunks. On inference error: single
/// `event: error` frame, **no** trailing `[DONE]`. On successful completion: usage chunk + `[DONE]`.
pub fn chat_completion_sse_chunks(
    id: &str,
    model: &str,
    item: Result<StreamPart, anyhow::Error>,
) -> Vec<Bytes> {
    match item {
        Ok(StreamPart::Delta(piece)) => {
            let chunk = json!({
                "id": id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{ "index": 0, "delta": { "content": piece } }]
            });
            vec![Bytes::from(sse::frame(&chunk.to_string()))]
        }
        Ok(StreamPart::Done(meta)) => {
            let chunk = json!({
                "id": id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": meta.finish_reason,
                }],
                "usage": chat_usage_tokens(meta.prompt_tokens, meta.completion_tokens),
            });
            vec![
                Bytes::from(sse::frame(&chunk.to_string())),
                Bytes::from(sse::done()),
            ]
        }
        Err(e) => {
            let err = openai_error_value("server_error", "inference_error", e.to_string());
            vec![Bytes::from(sse::frame_event("error", &err.to_string()))]
        }
    }
}

/// Responses API stream mapping (same error / done rules as chat).
pub fn responses_sse_chunks(
    id: &str,
    model: &str,
    item: Result<StreamPart, anyhow::Error>,
) -> Vec<Bytes> {
    match item {
        Ok(StreamPart::Delta(piece)) => {
            let delta = json!({
                "id": id,
                "object": "response.output_text.delta",
                "model": model,
                "delta": piece
            });
            vec![Bytes::from(sse::frame(&delta.to_string()))]
        }
        Ok(StreamPart::Done(meta)) => {
            let usage = responses_usage_tokens(meta.prompt_tokens, meta.completion_tokens);
            let evt = json!({
                "id": id,
                "object": "response.completed",
                "model": model,
                "status": "completed",
                "finish_reason": meta.finish_reason,
                "usage": usage,
            });
            vec![
                Bytes::from(sse::frame(&evt.to_string())),
                Bytes::from(sse::done()),
            ]
        }
        Err(e) => {
            let err = openai_error_value("server_error", "inference_error", e.to_string());
            vec![Bytes::from(sse::frame_event("error", &err.to_string()))]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::completion::CompletionMeta;

    #[test]
    fn chat_error_emits_event_error_without_done() {
        let chunks =
            chat_completion_sse_chunks("id1", "m", Err(anyhow::anyhow!("boom")));
        assert_eq!(chunks.len(), 1);
        let s = String::from_utf8_lossy(&chunks[0]);
        assert!(s.contains("event: error"), "{s}");
        assert!(s.contains("data:"), "{s}");
        assert!(!s.contains("[DONE]"), "{s}");
    }

    #[test]
    fn chat_done_appends_done_sentinel() {
        let chunks = chat_completion_sse_chunks(
            "id1",
            "m",
            Ok(StreamPart::Done(CompletionMeta {
                prompt_tokens: 1,
                completion_tokens: 2,
                finish_reason: "stop",
            })),
        );
        assert_eq!(chunks.len(), 2);
        let tail = String::from_utf8_lossy(&chunks[1]);
        assert!(tail.contains("[DONE]"), "{tail}");
    }

    #[test]
    fn responses_error_emits_event_error_without_done() {
        let chunks = responses_sse_chunks("id1", "m", Err(anyhow::anyhow!("fail")));
        assert_eq!(chunks.len(), 1);
        let s = String::from_utf8_lossy(&chunks[0]);
        assert!(s.contains("event: error"), "{s}");
        assert!(!s.contains("[DONE]"), "{s}");
    }
}
