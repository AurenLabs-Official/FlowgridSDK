use crate::OpenAI;
use crate::Result;
use reqwest::multipart::Form;
use serde_json::Value;

/// Files API (`client.files`, feature `files`).
pub struct FilesClient<'a> {
    inner: &'a OpenAI,
}

impl<'a> FilesClient<'a> {
    pub(crate) fn new(inner: &'a OpenAI) -> Self {
        Self { inner }
    }

    /// `POST /files` (multipart).
    pub async fn create(&self, form: Form) -> Result<Value> {
        let (v, _) = self.inner.transport.post_multipart_json("files", form).await?;
        Ok(v)
    }

    /// Convenience: upload bytes with a given filename and purpose.
    pub async fn create_bytes(
        &self,
        filename: impl Into<String>,
        purpose: impl Into<String>,
        bytes: Vec<u8>,
    ) -> Result<Value> {
        let file =
            crate::multipart::part_from_bytes(filename.into(), "application/octet-stream", bytes)?;
        let purpose_part = reqwest::multipart::Part::text(purpose.into());
        let form = Form::new().part("file", file).part("purpose", purpose_part);
        self.create(form).await
    }

    /// `GET /files/{file_id}`
    pub async fn retrieve(&self, file_id: impl AsRef<str>) -> Result<Value> {
        let path = format!("files/{}", file_id.as_ref());
        let (v, _) = self.inner.transport.get_json(&path).await?;
        Ok(v)
    }

    /// `GET /files`
    pub async fn list(&self) -> Result<Value> {
        let (v, _) = self.inner.transport.get_json("files").await?;
        Ok(v)
    }

    /// `DELETE /files/{file_id}`
    pub async fn delete(&self, file_id: impl AsRef<str>) -> Result<Value> {
        let path = format!("files/{}", file_id.as_ref());
        let (v, _) = self.inner.transport.delete_json(&path).await?;
        Ok(v)
    }

    /// `GET /files/{file_id}/content` (raw bytes).
    pub async fn content_bytes(&self, file_id: impl AsRef<str>) -> Result<Vec<u8>> {
        let path = format!("files/{}/content", file_id.as_ref());
        let (b, _) = self.inner.transport.get_bytes(&path).await?;
        Ok(b)
    }
}
