use crate::OpenAI;
use crate::Result;
use serde_json::Value;

/// Batches API (`client.batches`, feature `batches`).
pub struct BatchesClient<'a> {
    inner: &'a OpenAI,
}

impl<'a> BatchesClient<'a> {
    pub(crate) fn new(inner: &'a OpenAI) -> Self {
        Self { inner }
    }

    /// `POST /batches`
    pub async fn create(&self, body: &Value) -> Result<Value> {
        let (v, _) = self.inner.transport.post_json("batches", body).await?;
        Ok(v)
    }

    /// `GET /batches/{id}`
    pub async fn retrieve(&self, id: impl AsRef<str>) -> Result<Value> {
        let path = format!("batches/{}", id.as_ref());
        let (v, _) = self.inner.transport.get_json(&path).await?;
        Ok(v)
    }

    /// `GET /batches`
    pub async fn list(&self) -> Result<Value> {
        let (v, _) = self.inner.transport.get_json("batches").await?;
        Ok(v)
    }

    /// `POST /batches/{id}/cancel`
    pub async fn cancel(&self, id: impl AsRef<str>) -> Result<Value> {
        let path = format!("batches/{}/cancel", id.as_ref());
        let (v, _) = self
            .inner
            .transport
            .post_json(&path, &serde_json::json!({}))
            .await?;
        Ok(v)
    }
}
