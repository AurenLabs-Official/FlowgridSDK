use crate::internal::client::oai::{OpenAI, WithResponse};
use crate::internal::error::oai::Result;
use crate::internal::sse::oai::SseStream;
use bytes::Bytes;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::result::Result as StdResult;

/// Boxed byte stream from the HTTP client.
pub type BoxedByteStream = Pin<Box<dyn Stream<Item = StdResult<Bytes, std::io::Error>> + Send>>;

/// Typed Responses API client (`client.responses`).
pub struct ResponsesClient<'a> {
    inner: &'a OpenAI,
}

impl<'a> ResponsesClient<'a> {
    pub(crate) fn new(inner: &'a OpenAI) -> Self {
        Self { inner }
    }

    /// `POST /responses`
    pub async fn create(&self, body: &CreateResponseRequest) -> Result<ResponseObject> {
        let (v, _) = self.inner.transport.post_json("responses", body).await?;
        Ok(v)
    }

    /// `POST /responses` with response metadata.
    pub async fn create_with_response(
        &self,
        body: &CreateResponseRequest,
    ) -> Result<WithResponse<ResponseObject>> {
        let (data, meta) = self.inner.transport.post_json("responses", body).await?;
        Ok(WithResponse { data, meta })
    }

    /// `POST /responses` with `stream: true`, returning an SSE decoder.
    pub async fn create_stream(
        &self,
        body: &CreateResponseRequest,
    ) -> Result<(
        SseStream<BoxedByteStream>,
        crate::internal::transport::oai::ResponseMeta,
    )> {
        let (stream, meta) = self
            .inner
            .transport
            .post_stream_bytes("responses", body)
            .await?;
        let boxed: BoxedByteStream = Box::pin(stream);
        Ok((SseStream::new(boxed), meta))
    }

    /// `GET /responses/{id}`
    pub async fn retrieve(&self, id: impl AsRef<str>) -> Result<ResponseObject> {
        let path = format!("responses/{}", id.as_ref());
        let (v, _) = self.inner.transport.get_json(&path).await?;
        Ok(v)
    }

    /// `DELETE /responses/{id}`
    pub async fn delete(&self, id: impl AsRef<str>) -> Result<ResponseDeleted> {
        let path = format!("responses/{}", id.as_ref());
        let (v, _) = self.inner.transport.delete_json(&path).await?;
        Ok(v)
    }
}

/// Request body for creating a response.
#[derive(Debug, Clone, Serialize)]
pub struct CreateResponseRequest {
    /// Model id.
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Developer/system style instructions when supported.
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Structured `input` payload (string or JSON per API).
    pub input: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Enable SSE streaming.
    pub stream: Option<bool>,
    #[serde(flatten)]
    #[serde(skip_serializing_if = "serde_json::Map::is_empty")]
    /// Forward-compatible extra fields.
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Response object (minimal typed surface; extra fields are kept in `extra`).
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseObject {
    /// Response id.
    pub id: String,
    /// Output items (abridged typing).
    #[serde(default)]
    pub output: Vec<serde_json::Value>,
    /// Model name.
    #[serde(default)]
    pub model: Option<String>,
    /// Usage information when present.
    #[serde(default)]
    pub usage: Option<serde_json::Value>,
    #[serde(flatten)]
    /// Any additional keys returned by the API.
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl ResponseObject {
    /// Best-effort extraction of textual output similar to `output_text` helpers in Node examples.
    pub fn output_text(&self) -> Option<String> {
        let mut parts = Vec::new();
        for item in &self.output {
            if let Some(content) = item.get("content").and_then(|c| c.as_array()) {
                for c in content {
                    if let Some(t) = c.get("text").and_then(|t| t.as_str()) {
                        parts.push(t);
                    }
                }
            } else if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                parts.push(text);
            }
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(""))
        }
    }
}

/// Response to delete calls.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseDeleted {
    /// Id deleted.
    pub id: String,
    /// Whether deletion succeeded.
    pub deleted: Option<bool>,
}
