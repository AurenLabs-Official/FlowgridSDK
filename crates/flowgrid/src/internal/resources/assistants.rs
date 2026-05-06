use crate::internal::oai::OpenAI;
use crate::internal::oai::Result;
use crate::internal::pagination::ListPage;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Query parameters for [`AssistantsClient::list_with_params`] / [`AssistantsClient::list_typed`].
#[derive(Clone, Default, Debug)]
pub struct AssistantsListParams {
    pub limit: Option<u32>,
    pub order: Option<String>,
    pub after: Option<String>,
    pub before: Option<String>,
}

impl AssistantsListParams {
    /// Builds URL query pairs (skips unset fields).
    pub fn query_pairs(&self) -> Vec<(String, String)> {
        let mut v = Vec::new();
        if let Some(n) = self.limit {
            v.push(("limit".to_string(), n.to_string()));
        }
        if let Some(ref s) = self.order {
            if !s.is_empty() {
                v.push(("order".to_string(), s.clone()));
            }
        }
        if let Some(ref s) = self.after {
            if !s.is_empty() {
                v.push(("after".to_string(), s.clone()));
            }
        }
        if let Some(ref s) = self.before {
            if !s.is_empty() {
                v.push(("before".to_string(), s.clone()));
            }
        }
        v
    }
}

/// One OpenAI assistant object (`GET/POST /assistants` …).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct Assistant {
    pub id: String,
    #[serde(default)]
    pub object: Option<String>,
    #[serde(default)]
    pub created_at: Option<i64>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub tool_resources: Option<serde_json::Value>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Assistants API (`client.assistants`, feature `assistants`).
pub struct AssistantsClient<'a> {
    inner: &'a OpenAI,
}

impl<'a> AssistantsClient<'a> {
    pub(crate) fn new(inner: &'a OpenAI) -> Self {
        Self { inner }
    }

    /// `POST /assistants`
    pub async fn create(&self, body: &Value) -> Result<Value> {
        let (v, _) = self.inner.transport.post_json("assistants", body).await?;
        Ok(v)
    }

    /// `POST /assistants` with typed response body.
    pub async fn create_typed(&self, body: &Value) -> Result<Assistant> {
        let (v, _) = self.inner.transport.post_json("assistants", body).await?;
        Ok(v)
    }

    /// `GET /assistants/{id}`
    pub async fn retrieve(&self, id: impl AsRef<str>) -> Result<Value> {
        let path = format!("assistants/{}", id.as_ref());
        let (v, _) = self.inner.transport.get_json(&path).await?;
        Ok(v)
    }

    /// `GET /assistants/{id}` as structured JSON.
    pub async fn retrieve_typed(&self, id: impl AsRef<str>) -> Result<Assistant> {
        let path = format!("assistants/{}", id.as_ref());
        let (v, _) = self.inner.transport.get_json(&path).await?;
        Ok(v)
    }

    /// `GET /assistants` (no extra query parameters).
    pub async fn list(&self) -> Result<Value> {
        self.list_with_params(&AssistantsListParams::default())
            .await
    }

    /// `GET /assistants` with pagination / sort query parameters.
    pub async fn list_with_params(&self, params: &AssistantsListParams) -> Result<Value> {
        let q = params.query_pairs();
        let (v, _) = self
            .inner
            .transport
            .get_json_query("assistants", &q)
            .await?;
        Ok(v)
    }

    /// `GET /assistants` as [`ListPage`] of [`Assistant`].
    pub async fn list_typed(&self, params: &AssistantsListParams) -> Result<ListPage<Assistant>> {
        let q = params.query_pairs();
        let (v, _) = self
            .inner
            .transport
            .get_json_query("assistants", &q)
            .await?;
        Ok(v)
    }

    /// `POST /assistants/{id}`
    pub async fn update(&self, id: impl AsRef<str>, body: &Value) -> Result<Value> {
        let path = format!("assistants/{}", id.as_ref());
        let (v, _) = self.inner.transport.post_json(&path, body).await?;
        Ok(v)
    }

    /// `POST /assistants/{id}` with typed response body.
    pub async fn update_typed(&self, id: impl AsRef<str>, body: &Value) -> Result<Assistant> {
        let path = format!("assistants/{}", id.as_ref());
        let (v, _) = self.inner.transport.post_json(&path, body).await?;
        Ok(v)
    }

    /// `DELETE /assistants/{id}`
    pub async fn delete(&self, id: impl AsRef<str>) -> Result<Value> {
        let path = format!("assistants/{}", id.as_ref());
        let (v, _) = self.inner.transport.delete_json(&path).await?;
        Ok(v)
    }
}
