use crate::Anthropic;
use crate::Result;
use serde_json::Value;

pub struct ModelsClient<'a> {
    inner: &'a Anthropic,
}

impl<'a> ModelsClient<'a> {
    pub(crate) fn new(inner: &'a Anthropic) -> Self {
        Self { inner }
    }

    pub async fn retrieve(&self, model_id: impl AsRef<str>) -> Result<Value> {
        let p = format!("models/{}", model_id.as_ref());
        let (v, _) = self.inner.transport.get_json(&p).await?;
        Ok(v)
    }

    pub async fn list(&self) -> Result<Value> {
        let (v, _) = self.inner.transport.get_json("models").await?;
        Ok(v)
    }
}
