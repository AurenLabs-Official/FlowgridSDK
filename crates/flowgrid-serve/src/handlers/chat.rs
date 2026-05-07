use axum::body::Body;
use axum::extract::State;
use axum::http::header;
use axum::http::HeaderMap;
use axum::response::IntoResponse;
use axum::Json;
use bytes::Bytes;
use futures::stream::{self, StreamExt};
use serde_json::json;
use std::io::{Error, ErrorKind};
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::sse;
use crate::types::ChatReq;
use crate::AppState;

pub async fn chat_completions(
    State(st): State<std::sync::Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<ChatReq>,
) -> axum::response::Response {
    if crate::auth::authorize(&headers, &st.auth).is_err() {
        return axum::http::StatusCode::UNAUTHORIZED.into_response();
    }
    if !st.rate.allow() {
        return axum::http::StatusCode::TOO_MANY_REQUESTS.into_response();
    }
    let prompt = body
        .messages
        .as_ref()
        .and_then(|m| m.last())
        .map(|m| m.content.clone())
        .unwrap_or_else(|| "flowgrid-serve".to_string());
    let max_new = body.max_tokens.unwrap_or(128).min(4096) as usize;

    if body.stream == Some(true) {
        let rx = match st.scheduler.submit_stream(prompt, max_new).await {
            Ok(r) => r,
            Err(_) => return axum::http::StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        };
        let id = Uuid::new_v4().to_string();
        let model = body.model.clone();
        let mapped = ReceiverStream::new(rx).map(move |res| -> Result<Bytes, Error> {
            let inner = res.map_err(|e| std::io::Error::other(e.to_string()))?;
            let v: serde_json::Value =
                serde_json::from_str(&inner).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
            let piece = v["delta"].as_str().unwrap_or("");
            let chunk = json!({
                "id": id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{ "index": 0, "delta": { "content": piece } }]
            });
            Ok(Bytes::from(sse::frame(&chunk.to_string())))
        });
        let tail = stream::once(async move { Ok::<Bytes, Error>(Bytes::from(sse::done())) });
        let stream = mapped.chain(tail);
        let body = Body::from_stream(stream);
        return (
            [(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")],
            body,
        )
            .into_response();
    }

    let content = st
        .scheduler
        .submit_plain(prompt, max_new)
        .await
        .unwrap_or_else(|_| "scheduler-error".to_string());
    (
        axum::http::StatusCode::OK,
        Json(json!({
            "id": Uuid::new_v4().to_string(),
            "object": "chat.completion",
            "model": body.model,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": content }
            }]
        })),
    )
        .into_response()
}
