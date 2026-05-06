use crate::internal::oai::OpenAI;
use crate::internal::oai::Result;
use serde_json::Value;

/// Moderations API (`client.moderations`, feature `moderations`).
pub struct ModerationsClient<'a> {
    inner: &'a OpenAI,
}

impl<'a> ModerationsClient<'a> {
    pub(crate) fn new(inner: &'a OpenAI) -> Self {
        Self { inner }
    }

    /// `POST /moderations`
    pub async fn create(&self, body: &Value) -> Result<Value> {
        let (v, _) = self.inner.transport.post_json("moderations", body).await?;
        Ok(v)
    }
}
