use crate::Anthropic;
use crate::Result;
use serde_json::Value;

pub struct MessageBatchesClient<'a> {
    inner: &'a Anthropic,
}

impl<'a> MessageBatchesClient<'a> {
    pub(crate) fn new(inner: &'a Anthropic) -> Self {
        Self { inner }
    }

    pub async fn create(&self, body: &Value) -> Result<Value> {
        let (v, _) = self
            .inner
            .transport
            .post_json("messages/batches", body)
            .await?;
        Ok(v)
    }

    pub async fn retrieve(&self, id: impl AsRef<str>) -> Result<Value> {
        let p = format!("messages/batches/{}", id.as_ref());
        let (v, _) = self.inner.transport.get_json(&p).await?;
        Ok(v)
    }

    pub async fn list(&self) -> Result<Value> {
        let (v, _) = self.inner.transport.get_json("messages/batches").await?;
        Ok(v)
    }

    pub async fn delete(&self, id: impl AsRef<str>) -> Result<Value> {
        let p = format!("messages/batches/{}", id.as_ref());
        let (v, _) = self.inner.transport.delete_json(&p).await?;
        Ok(v)
    }

    pub async fn cancel(&self, id: impl AsRef<str>) -> Result<Value> {
        let p = format!("messages/batches/{}/cancel", id.as_ref());
        let (v, _) = self
            .inner
            .transport
            .post_json(&p, &serde_json::json!({}))
            .await?;
        Ok(v)
    }

    pub async fn results_bytes(&self, id: impl AsRef<str>) -> Result<Vec<u8>> {
        let p = format!("messages/batches/{}/results", id.as_ref());
        let (b, _) = self.inner.transport.get_bytes(&p).await?;
        Ok(b)
    }
}
