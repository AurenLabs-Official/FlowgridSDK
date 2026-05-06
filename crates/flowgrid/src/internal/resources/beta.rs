use crate::internal::clu::Anthropic;
use crate::internal::clu::Result;
use serde_json::Value;

/// Thin `beta/*` paths (requires appropriate `anthropic-beta` header on the client builder).
pub struct BetaClient<'a> {
    inner: &'a Anthropic,
}

impl<'a> BetaClient<'a> {
    pub(crate) fn new(inner: &'a Anthropic) -> Self {
        Self { inner }
    }

    pub async fn get(&self, path: impl AsRef<str>) -> Result<Value> {
        let p = path.as_ref().trim_start_matches('/');
        let full = format!("beta/{p}");
        let (v, _) = self.inner.transport.get_json(&full).await?;
        Ok(v)
    }

    pub async fn post(&self, path: impl AsRef<str>, body: &Value) -> Result<Value> {
        let p = path.as_ref().trim_start_matches('/');
        let full = format!("beta/{p}");
        let (v, _) = self.inner.transport.post_json(&full, body).await?;
        Ok(v)
    }

    pub async fn delete(&self, path: impl AsRef<str>) -> Result<Value> {
        let p = path.as_ref().trim_start_matches('/');
        let full = format!("beta/{p}");
        let (v, _) = self.inner.transport.delete_json(&full).await?;
        Ok(v)
    }

    pub fn models(&self) -> BetaModelsClient<'_> {
        BetaModelsClient { inner: self.inner }
    }
}

pub struct BetaModelsClient<'a> {
    inner: &'a Anthropic,
}

impl<'a> BetaModelsClient<'a> {
    pub async fn retrieve(&self, model_id: impl AsRef<str>) -> Result<Value> {
        let p = format!("beta/models/{}", model_id.as_ref());
        let (v, _) = self.inner.transport.get_json(&p).await?;
        Ok(v)
    }

    pub async fn list(&self) -> Result<Value> {
        let (v, _) = self.inner.transport.get_json("beta/models").await?;
        Ok(v)
    }
}
