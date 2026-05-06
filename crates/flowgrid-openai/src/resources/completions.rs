use crate::client::{OpenAI, WithResponse};
use crate::error::Result;
use serde::{Deserialize, Serialize};

/// Legacy completions (`client.completions`).
pub struct CompletionsClient<'a> {
    inner: &'a OpenAI,
}

impl<'a> CompletionsClient<'a> {
    pub(crate) fn new(inner: &'a OpenAI) -> Self {
        Self { inner }
    }

    /// `POST /completions`
    pub async fn create(&self, body: &CreateCompletionRequest) -> Result<Completion> {
        let (v, _) = self.inner.transport.post_json("completions", body).await?;
        Ok(v)
    }

    /// `POST /completions` with metadata.
    pub async fn create_with_response(
        &self,
        body: &CreateCompletionRequest,
    ) -> Result<WithResponse<Completion>> {
        let (data, meta) = self.inner.transport.post_json("completions", body).await?;
        Ok(WithResponse { data, meta })
    }
}

/// Legacy completion request.
#[derive(Debug, Clone, Serialize)]
pub struct CreateCompletionRequest {
    /// Model id.
    pub model: String,
    /// Prompt string or token list per API.
    pub prompt: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(flatten)]
    #[serde(skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Legacy completion.
#[derive(Debug, Clone, Deserialize)]
pub struct Completion {
    /// Id when present.
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub choices: Vec<CompletionChoice>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub usage: Option<serde_json::Value>,
}

/// Legacy completion choice.
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionChoice {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub index: Option<u32>,
    #[serde(default)]
    pub finish_reason: Option<String>,
}
