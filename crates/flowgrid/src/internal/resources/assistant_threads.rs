//! Threads, messages, and runs under `/v1/threads/*` (OpenAI Assistants workflow; feature `assistants`).

use crate::internal::error::oai::Error;
use crate::internal::oai::OpenAI;
use crate::internal::oai::Result;
use crate::internal::pagination::{ListPage, ListPagesLimits};
use crate::internal::resources::assistants::AssistantsListParams;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// List parameters for messages on a thread (cursor fields + optional `run_id` filter).
#[derive(Clone, Default, Debug)]
pub struct ThreadMessagesListParams {
    pub cursor: AssistantsListParams,
    pub run_id: Option<String>,
}

impl ThreadMessagesListParams {
    pub fn query_pairs(&self) -> Vec<(String, String)> {
        let mut q = self.cursor.query_pairs();
        if let Some(ref r) = self.run_id {
            if !r.is_empty() {
                q.push(("run_id".to_string(), r.clone()));
            }
        }
        q
    }
}

/// OpenAI thread object.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct Thread {
    pub id: String,
    #[serde(default)]
    pub object: Option<String>,
    #[serde(default)]
    pub created_at: Option<i64>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_resources: Option<serde_json::Value>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Message on a thread (`object` is usually `thread.message`).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ThreadMessage {
    pub id: String,
    #[serde(default)]
    pub object: Option<String>,
    #[serde(default)]
    pub created_at: Option<i64>,
    #[serde(default)]
    pub thread_id: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<Vec<serde_json::Value>>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Run on a thread (`object` is usually `thread.run`).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ThreadRun {
    pub id: String,
    #[serde(default)]
    pub object: Option<String>,
    #[serde(default)]
    pub created_at: Option<i64>,
    #[serde(default)]
    pub thread_id: Option<String>,
    #[serde(default)]
    pub assistant_id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// One step in an assistant run (`GET …/runs/{id}/steps`).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ThreadRunStep {
    pub id: String,
    #[serde(default)]
    pub object: Option<String>,
    #[serde(default)]
    pub created_at: Option<i64>,
    #[serde(default)]
    pub run_id: Option<String>,
    #[serde(default)]
    pub assistant_id: Option<String>,
    #[serde(default)]
    pub thread_id: Option<String>,
    #[serde(rename = "type")]
    #[serde(default)]
    pub step_type: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Top-level threads API (`client.threads()`, feature `assistants`).
pub struct ThreadsClient<'a> {
    inner: &'a OpenAI,
}

impl<'a> ThreadsClient<'a> {
    pub(crate) fn new(inner: &'a OpenAI) -> Self {
        Self { inner }
    }

    /// Bind a thread id for messages / runs sub-clients.
    pub fn thread(&self, thread_id: impl Into<String>) -> ThreadClient<'a> {
        ThreadClient {
            inner: self.inner,
            thread_id: thread_id.into(),
        }
    }

    /// `POST /threads`
    pub async fn create(&self, body: &Value) -> Result<Value> {
        let (v, _) = self.inner.transport.post_json("threads", body).await?;
        Ok(v)
    }

    pub async fn create_typed(&self, body: &Value) -> Result<Thread> {
        let (v, _) = self.inner.transport.post_json("threads", body).await?;
        Ok(v)
    }

    /// `GET /threads/{id}`
    pub async fn retrieve(&self, thread_id: impl AsRef<str>) -> Result<Value> {
        let path = format!("threads/{}", thread_id.as_ref());
        let (v, _) = self.inner.transport.get_json(&path).await?;
        Ok(v)
    }

    pub async fn retrieve_typed(&self, thread_id: impl AsRef<str>) -> Result<Thread> {
        let path = format!("threads/{}", thread_id.as_ref());
        let (v, _) = self.inner.transport.get_json(&path).await?;
        Ok(v)
    }

    /// `POST /threads/{id}` (same as [`ThreadClient::update`](ThreadClient::update)).
    pub async fn update(&self, thread_id: impl AsRef<str>, body: &Value) -> Result<Value> {
        let path = format!("threads/{}", thread_id.as_ref());
        let (v, _) = self.inner.transport.post_json(&path, body).await?;
        Ok(v)
    }

    pub async fn update_typed(&self, thread_id: impl AsRef<str>, body: &Value) -> Result<Thread> {
        let path = format!("threads/{}", thread_id.as_ref());
        let (v, _) = self.inner.transport.post_json(&path, body).await?;
        Ok(v)
    }

    /// `DELETE /threads/{id}` (same as [`ThreadClient::delete`](ThreadClient::delete)).
    pub async fn delete(&self, thread_id: impl AsRef<str>) -> Result<Value> {
        let path = format!("threads/{}", thread_id.as_ref());
        let (v, _) = self.inner.transport.delete_json(&path).await?;
        Ok(v)
    }

    /// `GET /threads`
    pub async fn list(&self) -> Result<Value> {
        self.list_with_params(&AssistantsListParams::default())
            .await
    }

    pub async fn list_with_params(&self, params: &AssistantsListParams) -> Result<Value> {
        let q = params.query_pairs();
        let (v, _) = self.inner.transport.get_json_query("threads", &q).await?;
        Ok(v)
    }

    pub async fn list_typed(&self, params: &AssistantsListParams) -> Result<ListPage<Thread>> {
        let q = params.query_pairs();
        let (v, _) = self.inner.transport.get_json_query("threads", &q).await?;
        Ok(v)
    }

    /// Cursor walk over [`ThreadsClient::list_typed`]; see [`AssistantsClient::list_all_typed`](crate::internal::resources::assistants::AssistantsClient::list_all_typed).
    pub async fn list_all_typed(
        &self,
        initial: &AssistantsListParams,
        limits: ListPagesLimits,
    ) -> Result<Vec<Thread>> {
        let max_pages = limits.max_pages.max(1);
        let mut out = Vec::new();
        let mut params = initial.clone();
        for _ in 0..max_pages {
            let page = self.list_typed(&params).await?;
            let next_after = page.after_cursor();
            let has_more_flag = page.has_more();
            for item in page.data {
                if let Some(cap) = limits.max_items {
                    if out.len() >= cap as usize {
                        return Ok(out);
                    }
                }
                out.push(item);
            }
            match next_after {
                Some(after) => params.after = Some(after),
                None => {
                    if has_more_flag {
                        return Err(Error::Config(
                            "OpenAI list: has_more is true but last_id is missing; cannot advance after cursor"
                                .into(),
                        ));
                    }
                    break;
                }
            }
        }
        Ok(out)
    }
}

/// Operations for a single thread (`/threads/{id}` and delegated message/run paths).
pub struct ThreadClient<'a> {
    inner: &'a OpenAI,
    thread_id: String,
}

impl<'a> ThreadClient<'a> {
    /// `GET /threads/{id}`
    pub async fn retrieve(&self) -> Result<Value> {
        let path = format!("threads/{}", self.thread_id);
        let (v, _) = self.inner.transport.get_json(&path).await?;
        Ok(v)
    }

    pub async fn retrieve_typed(&self) -> Result<Thread> {
        let path = format!("threads/{}", self.thread_id);
        let (v, _) = self.inner.transport.get_json(&path).await?;
        Ok(v)
    }

    /// `POST /threads/{id}`
    pub async fn update(&self, body: &Value) -> Result<Value> {
        let path = format!("threads/{}", self.thread_id);
        let (v, _) = self.inner.transport.post_json(&path, body).await?;
        Ok(v)
    }

    pub async fn update_typed(&self, body: &Value) -> Result<Thread> {
        let path = format!("threads/{}", self.thread_id);
        let (v, _) = self.inner.transport.post_json(&path, body).await?;
        Ok(v)
    }

    /// `DELETE /threads/{id}`
    pub async fn delete(&self) -> Result<Value> {
        let path = format!("threads/{}", self.thread_id);
        let (v, _) = self.inner.transport.delete_json(&path).await?;
        Ok(v)
    }

    pub fn messages(&self) -> ThreadMessagesClient<'a> {
        ThreadMessagesClient {
            inner: self.inner,
            thread_id: self.thread_id.clone(),
        }
    }

    pub fn runs(&self) -> ThreadRunsClient<'a> {
        ThreadRunsClient {
            inner: self.inner,
            thread_id: self.thread_id.clone(),
        }
    }
}

/// `GET/POST …/threads/{id}/messages`
pub struct ThreadMessagesClient<'a> {
    inner: &'a OpenAI,
    thread_id: String,
}

impl<'a> ThreadMessagesClient<'a> {
    fn base_path(&self) -> String {
        format!("threads/{}/messages", self.thread_id)
    }

    /// `GET …/messages`
    pub async fn list(&self) -> Result<Value> {
        self.list_with_params(&ThreadMessagesListParams::default())
            .await
    }

    pub async fn list_with_params(&self, params: &ThreadMessagesListParams) -> Result<Value> {
        let q = params.query_pairs();
        let (v, _) = self
            .inner
            .transport
            .get_json_query(&self.base_path(), &q)
            .await?;
        Ok(v)
    }

    pub async fn list_typed(
        &self,
        params: &ThreadMessagesListParams,
    ) -> Result<ListPage<ThreadMessage>> {
        let q = params.query_pairs();
        let (v, _) = self
            .inner
            .transport
            .get_json_query(&self.base_path(), &q)
            .await?;
        Ok(v)
    }

    /// `POST …/messages`
    pub async fn create(&self, body: &Value) -> Result<Value> {
        let (v, _) = self
            .inner
            .transport
            .post_json(&self.base_path(), body)
            .await?;
        Ok(v)
    }

    pub async fn create_typed(&self, body: &Value) -> Result<ThreadMessage> {
        let (v, _) = self
            .inner
            .transport
            .post_json(&self.base_path(), body)
            .await?;
        Ok(v)
    }

    /// `GET …/messages/{message_id}`
    pub async fn retrieve(&self, message_id: impl AsRef<str>) -> Result<Value> {
        let path = format!("{}/{}", self.base_path(), message_id.as_ref());
        let (v, _) = self.inner.transport.get_json(&path).await?;
        Ok(v)
    }

    pub async fn retrieve_typed(&self, message_id: impl AsRef<str>) -> Result<ThreadMessage> {
        let path = format!("{}/{}", self.base_path(), message_id.as_ref());
        let (v, _) = self.inner.transport.get_json(&path).await?;
        Ok(v)
    }
}

/// `GET/POST …/threads/{id}/runs` (+ cancel, tool output submission)
pub struct ThreadRunsClient<'a> {
    inner: &'a OpenAI,
    thread_id: String,
}

impl<'a> ThreadRunsClient<'a> {
    fn base_path(&self) -> String {
        format!("threads/{}/runs", self.thread_id)
    }

    /// `POST …/runs`
    pub async fn create(&self, body: &Value) -> Result<Value> {
        let (v, _) = self
            .inner
            .transport
            .post_json(&self.base_path(), body)
            .await?;
        Ok(v)
    }

    pub async fn create_typed(&self, body: &Value) -> Result<ThreadRun> {
        let (v, _) = self
            .inner
            .transport
            .post_json(&self.base_path(), body)
            .await?;
        Ok(v)
    }

    /// `GET …/runs`
    pub async fn list(&self) -> Result<Value> {
        self.list_with_params(&AssistantsListParams::default())
            .await
    }

    pub async fn list_with_params(&self, params: &AssistantsListParams) -> Result<Value> {
        let q = params.query_pairs();
        let (v, _) = self
            .inner
            .transport
            .get_json_query(&self.base_path(), &q)
            .await?;
        Ok(v)
    }

    pub async fn list_typed(&self, params: &AssistantsListParams) -> Result<ListPage<ThreadRun>> {
        let q = params.query_pairs();
        let (v, _) = self
            .inner
            .transport
            .get_json_query(&self.base_path(), &q)
            .await?;
        Ok(v)
    }

    /// `GET …/runs/{run_id}`
    pub async fn retrieve(&self, run_id: impl AsRef<str>) -> Result<Value> {
        let path = format!("{}/{}", self.base_path(), run_id.as_ref());
        let (v, _) = self.inner.transport.get_json(&path).await?;
        Ok(v)
    }

    pub async fn retrieve_typed(&self, run_id: impl AsRef<str>) -> Result<ThreadRun> {
        let path = format!("{}/{}", self.base_path(), run_id.as_ref());
        let (v, _) = self.inner.transport.get_json(&path).await?;
        Ok(v)
    }

    /// `POST …/runs/{run_id}/cancel`
    pub async fn cancel(&self, run_id: impl AsRef<str>) -> Result<Value> {
        let path = format!("{}/{}/cancel", self.base_path(), run_id.as_ref());
        let (v, _) = self.inner.transport.post_empty(&path).await?;
        Ok(v)
    }

    pub async fn cancel_typed(&self, run_id: impl AsRef<str>) -> Result<ThreadRun> {
        let path = format!("{}/{}/cancel", self.base_path(), run_id.as_ref());
        let (v, _) = self.inner.transport.post_empty(&path).await?;
        Ok(v)
    }

    /// `GET …/runs/{run_id}/steps`
    pub async fn list_steps_typed(
        &self,
        run_id: impl AsRef<str>,
        params: &AssistantsListParams,
    ) -> Result<ListPage<ThreadRunStep>> {
        let path = format!("{}/{}/steps", self.base_path(), run_id.as_ref());
        let q = params.query_pairs();
        let (v, _) = self.inner.transport.get_json_query(&path, &q).await?;
        Ok(v)
    }

    /// `POST …/runs/{run_id}/submit_tool_outputs`
    pub async fn submit_tool_outputs(
        &self,
        run_id: impl AsRef<str>,
        body: &Value,
    ) -> Result<Value> {
        let path = format!(
            "{}/{}/submit_tool_outputs",
            self.base_path(),
            run_id.as_ref()
        );
        let (v, _) = self.inner.transport.post_json(&path, body).await?;
        Ok(v)
    }

    pub async fn submit_tool_outputs_typed(
        &self,
        run_id: impl AsRef<str>,
        body: &Value,
    ) -> Result<ThreadRun> {
        let path = format!(
            "{}/{}/submit_tool_outputs",
            self.base_path(),
            run_id.as_ref()
        );
        let (v, _) = self.inner.transport.post_json(&path, body).await?;
        Ok(v)
    }
}
