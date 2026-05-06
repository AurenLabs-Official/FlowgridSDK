use crate::OpenAI;
use crate::Result;
use serde_json::Value;

/// Containers API (`client.containers`, feature `containers`).
pub struct ContainersClient<'a> {
    inner: &'a OpenAI,
}

impl<'a> ContainersClient<'a> {
    pub(crate) fn new(inner: &'a OpenAI) -> Self {
        Self { inner }
    }

    /// `POST /containers`
    pub async fn create(&self, body: &Value) -> Result<Value> {
        let (v, _) = self.inner.transport.post_json("containers", body).await?;
        Ok(v)
    }

    /// `GET /containers/{id}`
    pub async fn retrieve(&self, id: impl AsRef<str>) -> Result<Value> {
        let path = format!("containers/{}", id.as_ref());
        let (v, _) = self.inner.transport.get_json(&path).await?;
        Ok(v)
    }

    /// `GET /containers`
    pub async fn list(&self) -> Result<Value> {
        let (v, _) = self.inner.transport.get_json("containers").await?;
        Ok(v)
    }

    /// `DELETE /containers/{id}`
    pub async fn delete(&self, id: impl AsRef<str>) -> Result<Value> {
        let path = format!("containers/{}", id.as_ref());
        let (v, _) = self.inner.transport.delete_json(&path).await?;
        Ok(v)
    }
}
