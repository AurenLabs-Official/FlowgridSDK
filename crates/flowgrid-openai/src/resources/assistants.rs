use crate::OpenAI;
use crate::Result;
use serde_json::Value;

/// Assistants API (`client.assistants`, feature `assistants`).
pub struct AssistantsClient<'a> {
    inner: &'a OpenAI,
}

impl<'a> AssistantsClient<'a> {
    pub(crate) fn new(inner: &'a OpenAI) -> Self {
        Self { inner }
    }

    /// `POST /assistants`
    pub async fn create(&self, body: &Value) -> Result<Value> {
        let (v, _) = self.inner.transport.post_json("assistants", body).await?;
        Ok(v)
    }

    /// `GET /assistants/{id}`
    pub async fn retrieve(&self, id: impl AsRef<str>) -> Result<Value> {
        let path = format!("assistants/{}", id.as_ref());
        let (v, _) = self.inner.transport.get_json(&path).await?;
        Ok(v)
    }

    /// `GET /assistants`
    pub async fn list(&self) -> Result<Value> {
        let (v, _) = self.inner.transport.get_json("assistants").await?;
        Ok(v)
    }

    /// `POST /assistants/{id}`
    pub async fn update(&self, id: impl AsRef<str>, body: &Value) -> Result<Value> {
        let path = format!("assistants/{}", id.as_ref());
        let (v, _) = self.inner.transport.post_json(&path, body).await?;
        Ok(v)
    }

    /// `DELETE /assistants/{id}`
    pub async fn delete(&self, id: impl AsRef<str>) -> Result<Value> {
        let path = format!("assistants/{}", id.as_ref());
        let (v, _) = self.inner.transport.delete_json(&path).await?;
        Ok(v)
    }
}
