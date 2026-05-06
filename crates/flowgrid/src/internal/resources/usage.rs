//! Typed `usage` sub-objects for OpenAI-style JSON (forward-compatible via `extra` maps).

use serde::{Deserialize, Serialize};

/// Token usage on embedding responses (`CreateEmbeddingResponse.usage`).
#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq, Eq)]
pub struct EmbeddingUsage {
    #[serde(default)]
    pub prompt_tokens: Option<u32>,
    #[serde(default)]
    pub total_tokens: Option<u32>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Token usage on legacy completion responses (`Completion.usage`).
#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq, Eq)]
pub struct CompletionUsage {
    #[serde(default)]
    pub prompt_tokens: Option<u32>,
    #[serde(default)]
    pub completion_tokens: Option<u32>,
    #[serde(default)]
    pub total_tokens: Option<u32>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Usage block on Responses API objects (`ResponseObject.usage`).
#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq, Eq)]
pub struct ResponseObjectUsage {
    #[serde(default)]
    pub input_tokens: Option<u32>,
    #[serde(default)]
    pub output_tokens: Option<u32>,
    #[serde(default)]
    pub total_tokens: Option<u32>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}
