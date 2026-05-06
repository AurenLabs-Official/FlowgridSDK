use crate::internal::oai::OpenAI;
use crate::internal::oai::Result;
use reqwest::multipart::Form;
use serde_json::Value;

/// Images API (`client.images`, feature `images`).
pub struct ImagesClient<'a> {
    inner: &'a OpenAI,
}

impl<'a> ImagesClient<'a> {
    pub(crate) fn new(inner: &'a OpenAI) -> Self {
        Self { inner }
    }

    /// `POST /images/generations`
    pub async fn generate(&self, body: &Value) -> Result<Value> {
        let (v, _) = self
            .inner
            .transport
            .post_json("images/generations", body)
            .await?;
        Ok(v)
    }

    /// `POST /images/edits` (multipart).
    pub async fn edit(&self, form: Form) -> Result<Value> {
        let (v, _) = self
            .inner
            .transport
            .post_multipart_json("images/edits", form)
            .await?;
        Ok(v)
    }

    /// `POST /images/variations` (multipart).
    pub async fn create_variation(&self, form: Form) -> Result<Value> {
        let (v, _) = self
            .inner
            .transport
            .post_multipart_json("images/variations", form)
            .await?;
        Ok(v)
    }
}
