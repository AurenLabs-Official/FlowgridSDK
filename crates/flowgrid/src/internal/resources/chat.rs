use crate::internal::client::oai::{OpenAI, WithResponse};
use crate::internal::error::oai::Result;
use crate::internal::oai::pagination::ListPage;
use crate::internal::sse::oai::SseStream;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::responses::BoxedByteStream;

/// Chat namespace (`client.chat`).
pub struct ChatClient<'a> {
    inner: &'a OpenAI,
}

impl<'a> ChatClient<'a> {
    pub(crate) fn new(inner: &'a OpenAI) -> Self {
        Self { inner }
    }

    /// Stored chat completions (`client.chat.completions`).
    pub fn completions(&self) -> ChatCompletionsClient<'_> {
        ChatCompletionsClient { inner: self.inner }
    }
}

/// Stored chat completions client.
pub struct ChatCompletionsClient<'a> {
    inner: &'a OpenAI,
}

impl<'a> ChatCompletionsClient<'a> {
    /// `POST /chat/completions`
    pub async fn create(&self, body: &CreateChatCompletionRequest) -> Result<ChatCompletion> {
        let (v, _) = self
            .inner
            .transport
            .post_json("chat/completions", body)
            .await?;
        Ok(v)
    }

    /// `POST /chat/completions` with metadata.
    pub async fn create_with_response(
        &self,
        body: &CreateChatCompletionRequest,
    ) -> Result<WithResponse<ChatCompletion>> {
        let (data, meta) = self
            .inner
            .transport
            .post_json("chat/completions", body)
            .await?;
        Ok(WithResponse { data, meta })
    }

    /// Streaming chat completions (SSE).
    pub async fn create_stream(
        &self,
        body: &CreateChatCompletionRequest,
    ) -> Result<(
        SseStream<BoxedByteStream>,
        crate::internal::transport::oai::ResponseMeta,
    )> {
        let (stream, meta) = self
            .inner
            .transport
            .post_stream_bytes("chat/completions", body)
            .await?;
        Ok((SseStream::new(Box::pin(stream)), meta))
    }

    /// `GET /chat/completions/{id}`
    pub async fn retrieve(&self, id: impl AsRef<str>) -> Result<ChatCompletion> {
        let path = format!("chat/completions/{}", id.as_ref());
        let (v, _) = self.inner.transport.get_json(&path).await?;
        Ok(v)
    }

    /// `PATCH /chat/completions/{id}`
    pub async fn update(
        &self,
        id: impl AsRef<str>,
        body: &serde_json::Value,
    ) -> Result<ChatCompletion> {
        let path = format!("chat/completions/{}", id.as_ref());
        let (v, _) = self.inner.transport.patch_json(&path, body).await?;
        Ok(v)
    }

    /// `GET /chat/completions`
    pub async fn list(
        &self,
        params: &ChatCompletionListParams,
    ) -> Result<ListPage<ChatCompletion>> {
        let mut path = String::from("chat/completions");
        let mut ser = url::form_urlencoded::Serializer::new(String::new());
        if let Some(limit) = params.limit {
            ser.append_pair("limit", &limit.to_string());
        }
        if let Some(ref after) = params.after {
            ser.append_pair("after", after);
        }
        if let Some(ref order) = params.order {
            ser.append_pair("order", order);
        }
        let q = ser.finish();
        if !q.is_empty() {
            path.push('?');
            path.push_str(&q);
        }
        let (v, _) = self.inner.transport.get_json(&path).await?;
        Ok(v)
    }

    /// `DELETE /chat/completions/{id}`
    pub async fn delete(&self, id: impl AsRef<str>) -> Result<ChatCompletionDeleted> {
        let path = format!("chat/completions/{}", id.as_ref());
        let (v, _) = self.inner.transport.delete_json(&path).await?;
        Ok(v)
    }

    /// `GET /chat/completions/{completion_id}/messages`
    pub async fn list_messages(
        &self,
        completion_id: impl AsRef<str>,
        params: &ChatCompletionMessagesListParams,
    ) -> Result<ListPage<serde_json::Value>> {
        let mut path = format!("chat/completions/{}/messages", completion_id.as_ref());
        let mut ser = url::form_urlencoded::Serializer::new(String::new());
        if let Some(limit) = params.limit {
            ser.append_pair("limit", &limit.to_string());
        }
        if let Some(ref after) = params.after {
            ser.append_pair("after", after);
        }
        if let Some(ref order) = params.order {
            ser.append_pair("order", order);
        }
        let q = ser.finish();
        if !q.is_empty() {
            path.push('?');
            path.push_str(&q);
        }
        let (v, _) = self.inner.transport.get_json(&path).await?;
        Ok(v)
    }
}

/// Chat completion creation request.
#[derive(Debug, Clone, Serialize)]
pub struct CreateChatCompletionRequest {
    /// Model id.
    pub model: String,
    /// Chat messages (typed loosely as JSON values).
    pub messages: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(flatten)]
    #[serde(skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl CreateChatCompletionRequest {
    /// Minimal user message helper.
    pub fn user_message(text: impl Into<String>) -> serde_json::Value {
        json!({ "role": "user", "content": text.into() })
    }
}

/// Chat completion object.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletion {
    /// Completion id.
    pub id: String,
    #[serde(default)]
    pub choices: Vec<ChatCompletionChoice>,
    #[serde(default)]
    pub created: Option<u64>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub object: Option<String>,
    #[serde(default)]
    pub usage: Option<serde_json::Value>,
}

/// One choice in a chat completion.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionChoice {
    #[serde(default)]
    pub index: Option<u32>,
    #[serde(default)]
    pub message: Option<ChatCompletionMessage>,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// Assistant message payload.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionMessage {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<serde_json::Value>,
}

impl ChatCompletion {
    /// Convenience for `choices[0].message.content` string content.
    pub fn message_content(&self) -> Option<String> {
        let choice = self.choices.first()?;
        let msg = choice.message.as_ref()?;
        match &msg.content {
            Some(serde_json::Value::String(s)) => Some(s.clone()),
            Some(other) => Some(other.to_string()),
            _ => None,
        }
    }
}

/// List parameters for stored chat completions.
#[derive(Debug, Clone, Default)]
pub struct ChatCompletionListParams {
    /// Page size.
    pub limit: Option<u32>,
    /// Pagination cursor.
    pub after: Option<String>,
    /// Sort order (`asc` / `desc`).
    pub order: Option<String>,
}

/// List parameters for stored messages.
#[derive(Debug, Clone, Default)]
pub struct ChatCompletionMessagesListParams {
    /// Page size.
    pub limit: Option<u32>,
    /// Pagination cursor.
    pub after: Option<String>,
    /// Sort order (`asc` / `desc`).
    pub order: Option<String>,
}

/// Delete response for chat completions.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionDeleted {
    /// Id removed.
    pub id: String,
    #[serde(default)]
    pub object: Option<String>,
    #[serde(default)]
    pub deleted: Option<bool>,
}
