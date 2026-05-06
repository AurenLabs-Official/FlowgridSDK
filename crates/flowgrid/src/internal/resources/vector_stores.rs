use crate::internal::oai::OpenAI;
use crate::internal::oai::Result;
use serde_json::Value;

/// Vector stores API (`client.vector_stores`, feature `vector_stores`).
pub struct VectorStoresClient<'a> {
    inner: &'a OpenAI,
}

impl<'a> VectorStoresClient<'a> {
    pub(crate) fn new(inner: &'a OpenAI) -> Self {
        Self { inner }
    }

    /// `POST /vector_stores`
    pub async fn create(&self, body: &Value) -> Result<Value> {
        let (v, _) = self
            .inner
            .transport
            .post_json("vector_stores", body)
            .await?;
        Ok(v)
    }

    /// `GET /vector_stores/{id}`
    pub async fn retrieve(&self, id: impl AsRef<str>) -> Result<Value> {
        let path = format!("vector_stores/{}", id.as_ref());
        let (v, _) = self.inner.transport.get_json(&path).await?;
        Ok(v)
    }

    /// `GET /vector_stores`
    pub async fn list(&self) -> Result<Value> {
        let (v, _) = self.inner.transport.get_json("vector_stores").await?;
        Ok(v)
    }

    /// `DELETE /vector_stores/{id}`
    pub async fn delete(&self, id: impl AsRef<str>) -> Result<Value> {
        let path = format!("vector_stores/{}", id.as_ref());
        let (v, _) = self.inner.transport.delete_json(&path).await?;
        Ok(v)
    }
}
