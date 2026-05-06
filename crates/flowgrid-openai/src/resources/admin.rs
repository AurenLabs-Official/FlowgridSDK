use crate::OpenAI;
use crate::Result;
use serde_json::Value;

/// Administration helpers (`client.admin`, feature `admin`).
///
/// Paths are relative to the configured `v1` base (for example `organization/projects`).
pub struct AdminClient<'a> {
    inner: &'a OpenAI,
}

impl<'a> AdminClient<'a> {
    pub(crate) fn new(inner: &'a OpenAI) -> Self {
        Self { inner }
    }

    /// `GET /{path}`
    pub async fn get(&self, path: impl AsRef<str>) -> Result<Value> {
        let p = path.as_ref().trim_start_matches('/');
        let (v, _) = self.inner.transport.get_json(p).await?;
        Ok(v)
    }

    /// `POST /{path}`
    pub async fn post(&self, path: impl AsRef<str>, body: &Value) -> Result<Value> {
        let p = path.as_ref().trim_start_matches('/');
        let (v, _) = self.inner.transport.post_json(p, body).await?;
        Ok(v)
    }

    /// `DELETE /{path}`
    pub async fn delete(&self, path: impl AsRef<str>) -> Result<Value> {
        let p = path.as_ref().trim_start_matches('/');
        let (v, _) = self.inner.transport.delete_json(p).await?;
        Ok(v)
    }
}
