#[cfg(feature = "openai")]
mod oai_impl {
    use crate::internal::oai::OpenAI;
    use crate::internal::oai::Result;
    use serde_json::Value;

    /// Batches API (`client.batches`, feature `batches`).
    pub struct BatchesClient<'a> {
        pub(super) inner: &'a OpenAI,
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
}

#[cfg(feature = "openai")]
pub use oai_impl::BatchesClient;

#[cfg(feature = "anthropic")]
mod clu_impl {
    use crate::internal::clu::Anthropic;
    use crate::internal::clu::Result;
    use serde_json::Value;

    pub struct MessageBatchesClient<'a> {
        pub(super) inner: &'a Anthropic,
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
}

#[cfg(feature = "anthropic")]
pub use clu_impl::MessageBatchesClient;
