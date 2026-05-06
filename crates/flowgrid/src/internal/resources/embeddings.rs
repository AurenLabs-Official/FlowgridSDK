use crate::internal::client::oai::{OpenAI, WithResponse};
use crate::internal::error::oai::Result;
use crate::internal::resources::usage::EmbeddingUsage;
use serde::{Deserialize, Serialize};

/// Embeddings API (`client.embeddings`).
pub struct EmbeddingsClient<'a> {
    inner: &'a OpenAI,
}

impl<'a> EmbeddingsClient<'a> {
    pub(crate) fn new(inner: &'a OpenAI) -> Self {
        Self { inner }
    }

    /// `POST /embeddings`
    pub async fn create(&self, body: &CreateEmbeddingRequest) -> Result<CreateEmbeddingResponse> {
        let (v, _) = self.inner.transport.post_json("embeddings", body).await?;
        Ok(v)
    }

    /// `POST /embeddings` with metadata.
    pub async fn create_with_response(
        &self,
        body: &CreateEmbeddingRequest,
    ) -> Result<WithResponse<CreateEmbeddingResponse>> {
        let (data, meta) = self.inner.transport.post_json("embeddings", body).await?;
        Ok(WithResponse { data, meta })
    }
}

/// Embeddings request.
#[derive(Debug, Clone, Serialize)]
pub struct CreateEmbeddingRequest {
    /// Embedding model id.
    pub model: String,
    /// Input string, array, or token list per API.
    pub input: serde_json::Value,
    #[serde(flatten)]
    #[serde(skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Embeddings response.
#[derive(Debug, Clone, Deserialize)]
pub struct CreateEmbeddingResponse {
    #[serde(default)]
    pub object: Option<String>,
    #[serde(default)]
    pub data: Vec<Embedding>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub usage: Option<EmbeddingUsage>,
}

/// Single embedding vector payload.
#[derive(Debug, Clone, Deserialize)]
pub struct Embedding {
    #[serde(default)]
    pub object: Option<String>,
    #[serde(default)]
    pub embedding: Vec<f32>,
    #[serde(default)]
    pub index: Option<u32>,
}
