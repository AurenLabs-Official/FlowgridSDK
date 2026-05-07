use crate::internal::oai::OpenAI;
use crate::internal::oai::Result;
use serde_json::Value;

/// Fine-tuning API (`client.fine_tuning`, feature `fine_tuning`).
pub struct FineTuningClient<'a> {
    inner: &'a OpenAI,
}

impl<'a> FineTuningClient<'a> {
    pub(crate) fn new(inner: &'a OpenAI) -> Self {
        Self { inner }
    }

    /// `client.fine_tuning.jobs`
    pub fn jobs(&self) -> FineTuningJobsClient<'_> {
        FineTuningJobsClient { inner: self.inner }
    }
}

/// Fine-tuning jobs resource.
pub struct FineTuningJobsClient<'a> {
    inner: &'a OpenAI,
}

impl<'a> FineTuningJobsClient<'a> {
    /// `POST /fine_tuning/jobs`
    pub async fn create(&self, body: &Value) -> Result<Value> {
        let (v, _) = self
            .inner
            .transport
            .post_json("fine_tuning/jobs", body)
            .await?;
        Ok(v)
    }

    /// `GET /fine_tuning/jobs/{id}`
    pub async fn retrieve(&self, id: impl AsRef<str>) -> Result<Value> {
        let path = format!("fine_tuning/jobs/{}", id.as_ref());
        let (v, _) = self.inner.transport.get_json(&path).await?;
        Ok(v)
    }

    /// `GET /fine_tuning/jobs`
    pub async fn list(&self) -> Result<Value> {
        let (v, _) = self.inner.transport.get_json("fine_tuning/jobs").await?;
        Ok(v)
    }

    /// `POST /fine_tuning/jobs/{id}/cancel`
    pub async fn cancel(&self, id: impl AsRef<str>) -> Result<Value> {
        let path = format!("fine_tuning/jobs/{}/cancel", id.as_ref());
        let (v, _) = self.inner.transport.post_empty(&path).await?;
        Ok(v)
    }

    /// `GET /fine_tuning/jobs/{id}/events`
    pub async fn list_events(&self, id: impl AsRef<str>) -> Result<Value> {
        let path = format!("fine_tuning/jobs/{}/events", id.as_ref());
        let (v, _) = self.inner.transport.get_json(&path).await?;
        Ok(v)
    }

    /// `GET /fine_tuning/jobs/{id}/checkpoints`
    pub async fn list_checkpoints(&self, id: impl AsRef<str>) -> Result<Value> {
        let path = format!("fine_tuning/jobs/{}/checkpoints", id.as_ref());
        let (v, _) = self.inner.transport.get_json(&path).await?;
        Ok(v)
    }
}
