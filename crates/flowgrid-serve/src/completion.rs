//! Completion metadata for OpenAI-shaped responses (`usage`, `finish_reason`).

/// Token accounting and finish reason for one completion (local LLM or fallback).
#[derive(Debug, Clone)]
pub struct CompletionMeta {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// OpenAI-style: `stop` (EOS or non-LLM complete), `length` (hit `max_tokens`).
    pub finish_reason: &'static str,
}

impl CompletionMeta {
    pub fn heuristic_echo(prompt: &str, completion: &str, finish_reason: &'static str) -> Self {
        use crate::openai_compat::approx_tokens_from_text;
        let prompt_tokens = approx_tokens_from_text(prompt);
        let completion_tokens = approx_tokens_from_text(completion);
        Self {
            prompt_tokens,
            completion_tokens,
            finish_reason,
        }
    }
}

/// Plain (non-streaming) scheduler result.
#[derive(Debug, Clone)]
pub struct PlainOutput {
    pub text: String,
    pub meta: CompletionMeta,
}

/// Streaming events from the inference worker.
#[derive(Debug, Clone)]
pub enum StreamPart {
    Delta(String),
    Done(CompletionMeta),
}
