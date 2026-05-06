use crate::OpenAI;
use crate::Result;
use serde_json::Value;

/// Evals API (`client.evals`, feature `evals`).
pub struct EvalsClient<'a> {
    inner: &'a OpenAI,
}

impl<'a> EvalsClient<'a> {
    pub(crate) fn new(inner: &'a OpenAI) -> Self {
        Self { inner }
    }

    /// `POST /evals`
    pub async fn create(&self, body: &Value) -> Result<Value> {
        let (v, _) = self.inner.transport.post_json("evals", body).await?;
        Ok(v)
    }

    /// `GET /evals/{id}`
    pub async fn retrieve(&self, id: impl AsRef<str>) -> Result<Value> {
        let path = format!("evals/{}", id.as_ref());
        let (v, _) = self.inner.transport.get_json(&path).await?;
        Ok(v)
    }

    /// `GET /evals`
    pub async fn list(&self) -> Result<Value> {
        let (v, _) = self.inner.transport.get_json("evals").await?;
        Ok(v)
    }
}
