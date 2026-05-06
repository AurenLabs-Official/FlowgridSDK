use crate::internal::client::clu::{Anthropic, WithResponse};
use crate::internal::error::clu::Result;
use crate::internal::execute_options::ExecuteOptions;
use crate::internal::sse::clu::SseStream;

pub use crate::internal::stream_types::BoxedByteStream;
use serde::{Deserialize, Serialize};

pub struct MessagesClient<'a> {
    inner: &'a Anthropic,
}

impl<'a> MessagesClient<'a> {
    pub(crate) fn new(inner: &'a Anthropic) -> Self {
        Self { inner }
    }

    /// `POST /messages`
    pub async fn create(&self, body: &CreateMessageRequest) -> Result<Message> {
        self.create_with_options(body, ExecuteOptions::default())
            .await
    }

    pub async fn create_with_options(
        &self,
        body: &CreateMessageRequest,
        opts: ExecuteOptions,
    ) -> Result<Message> {
        let (v, _) = self
            .inner
            .transport
            .post_json_with_options("messages", body, opts)
            .await?;
        Ok(v)
    }

    pub async fn create_with_response(
        &self,
        body: &CreateMessageRequest,
    ) -> Result<WithResponse<Message>> {
        self.create_with_response_and_options(body, ExecuteOptions::default())
            .await
    }

    pub async fn create_with_response_and_options(
        &self,
        body: &CreateMessageRequest,
        opts: ExecuteOptions,
    ) -> Result<WithResponse<Message>> {
        let (data, meta) = self
            .inner
            .transport
            .post_json_with_options("messages", body, opts)
            .await?;
        Ok(WithResponse { data, meta })
    }

    /// Streaming message (`stream: true`); yields SSE events.
    ///
    /// Requires [`CreateMessageRequest::stream`] `== Some(true)`. Use
    /// [`SseStream::into_unpin_event_stream`](crate::internal::sse::clu::SseStream::into_unpin_event_stream)
    /// for a [`futures::Stream`] of [`SseEvent`](crate::internal::sse::clu::SseEvent) without `pin_mut`.
    pub async fn create_stream(
        &self,
        body: &CreateMessageRequest,
    ) -> Result<(
        SseStream<BoxedByteStream>,
        crate::internal::transport::clu::ResponseMeta,
    )> {
        self.create_stream_with_options(body, ExecuteOptions::default())
            .await
    }

    pub async fn create_stream_with_options(
        &self,
        body: &CreateMessageRequest,
        opts: ExecuteOptions,
    ) -> Result<(
        SseStream<BoxedByteStream>,
        crate::internal::transport::clu::ResponseMeta,
    )> {
        if body.stream != Some(true) {
            return Err(crate::internal::error::clu::Error::Config(
                "create_stream requires CreateMessageRequest { stream: Some(true), .. }"
                    .to_string(),
            ));
        }
        let (stream, meta) = self
            .inner
            .transport
            .post_stream_bytes_with_options("messages", body, opts)
            .await?;
        Ok((SseStream::new(Box::pin(stream)), meta))
    }

    /// `POST /messages/count_tokens`
    pub async fn count_tokens(
        &self,
        body: &CountMessageTokensRequest,
    ) -> Result<MessageTokensCount> {
        let (v, _) = self
            .inner
            .transport
            .post_json("messages/count_tokens", body)
            .await?;
        Ok(v)
    }
}

/// Non-streaming create request (set `stream: Some(false)` or omit).
#[derive(Debug, Clone, Serialize)]
pub struct CreateMessageRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(flatten)]
    #[serde(skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Token counting request (same shape as much of Messages API).
#[derive(Debug, Clone, Serialize)]
pub struct CountMessageTokensRequest {
    pub model: String,
    pub messages: Vec<serde_json::Value>,
    #[serde(flatten)]
    #[serde(skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Message response (minimal typed surface).
#[derive(Debug, Clone, Deserialize)]
pub struct Message {
    pub id: String,
    #[serde(rename = "type")]
    pub type_: Option<String>,
    pub role: Option<String>,
    pub content: Option<Vec<serde_json::Value>>,
    pub model: Option<String>,
    #[serde(rename = "stop_reason")]
    pub stop_reason: Option<String>,
    pub usage: Option<serde_json::Value>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl Message {
    /// Concatenate simple `text` blocks from `content`.
    pub fn text_concat(&self) -> Option<String> {
        let blocks = self.content.as_ref()?;
        let mut out = String::new();
        for b in blocks {
            if let Some(t) = b.get("text").and_then(|x| x.as_str()) {
                out.push_str(t);
            }
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }
}

#[cfg(feature = "batches")]
impl<'a> MessagesClient<'a> {
    /// `client.messages.batches` namespace.
    pub fn batches(&self) -> crate::internal::resources::batches::MessageBatchesClient<'a> {
        crate::internal::resources::batches::MessageBatchesClient::new(self.inner)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct MessageTokensCount {
    #[serde(rename = "input_tokens")]
    pub input_tokens: Option<u32>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}
