//! OpenAI-adjacent JSON helpers. When a tokenizer is available, [`crate::engine::LocalLlm`] reports
//! exact prompt/generation token counts; echo paths without a checkpoint still use the **byte
//! heuristic** (~4 chars per token) via [`approx_tokens_from_text`].

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use std::fmt::Display;

/// ~4 bytes per token for Latin text; documented limitation vs production OpenAI tokenizers.
pub fn approx_tokens_from_text(s: &str) -> u32 {
    ((s.len() as f64 / 4.0).ceil() as u32).max(1)
}

pub fn approx_tokens_from_chars(n: usize) -> u32 {
    ((n as f64 / 4.0).ceil() as u32).max(1)
}

pub fn openai_error_value(typ: &str, code: &str, message: impl Display) -> Value {
    json!({
        "error": {
            "message": message.to_string(),
            "type": typ,
            "code": code,
        }
    })
}

pub fn openai_error_response(
    status: StatusCode,
    typ: &str,
    code: &str,
    message: impl Display,
) -> Response {
    (status, Json(openai_error_value(typ, code, message))).into_response()
}

pub fn chat_usage(prompt: &str, completion: &str) -> Value {
    chat_usage_tokens(
        approx_tokens_from_text(prompt),
        approx_tokens_from_text(completion),
    )
}

pub fn chat_usage_tokens(prompt_tokens: u32, completion_tokens: u32) -> Value {
    json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens.saturating_add(completion_tokens),
    })
}

pub fn responses_usage(prompt_flat: &str, output: &str) -> Value {
    responses_usage_tokens(
        approx_tokens_from_text(prompt_flat),
        approx_tokens_from_text(output),
    )
}

pub fn responses_usage_tokens(input_tokens: u32, output_tokens: u32) -> Value {
    json!({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens.saturating_add(output_tokens),
    })
}
