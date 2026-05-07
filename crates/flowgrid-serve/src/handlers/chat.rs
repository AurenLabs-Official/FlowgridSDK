use axum::body::Body;
use axum::extract::State;
use axum::http::header;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::Json;
use bytes::Bytes;
use futures::stream::{self, StreamExt};
use serde_json::json;
use std::io::{Error, ErrorKind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::openai_compat::{chat_usage, openai_error_response};
use crate::sse;
use crate::types::ChatReq;
use crate::AppState;

pub async fn chat_completions(
    State(st): State<std::sync::Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<ChatReq>,
) -> axum::response::Response {
    if crate::auth::authorize(&headers, &st.auth).is_err() {
        return openai_error_response(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "invalid_api_key",
            "Invalid or missing API key",
        );
    }
    if !st.rate.allow() {
        return openai_error_response(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            "rate_limit_exceeded",
            "Too many requests",
        );
    }
    let prompt = body
        .messages
        .as_ref()
        .and_then(|m| m.last())
        .map(|m| m.content.clone())
        .unwrap_or_else(|| "flowgrid-serve".to_string());
    let max_new = body.max_tokens.unwrap_or(128).min(4096) as usize;
    let finish_reason = if st.local_llm_loaded {
        "length"
    } else {
        "stop"
    };

    if body.stream == Some(true) {
        let rx = match st.scheduler.submit_stream(prompt.clone(), max_new).await {
            Ok(r) => r,
            Err(e) => {
                return openai_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "server_error",
                    "scheduler_error",
                    e.to_string(),
                );
            }
        };
        let id = Uuid::new_v4().to_string();
        let model = body.model.clone();
        let id_chunk = id.clone();
        let model_chunk = model.clone();
        let chars_out = Arc::new(AtomicUsize::new(0));
        let co = Arc::clone(&chars_out);
        let mapped = ReceiverStream::new(rx).map(move |res| -> Result<Bytes, Error> {
            let inner = res.map_err(|e| std::io::Error::other(e.to_string()))?;
            let v: serde_json::Value =
                serde_json::from_str(&inner).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
            let piece = v["delta"].as_str().unwrap_or("");
            co.fetch_add(piece.chars().count(), Ordering::Relaxed);
            let chunk = json!({
                "id": id_chunk,
                "object": "chat.completion.chunk",
                "model": model_chunk,
                "choices": [{ "index": 0, "delta": { "content": piece } }]
            });
            Ok(Bytes::from(sse::frame(&chunk.to_string())))
        });
        let id_done = id.clone();
        let model_done = model.clone();
        let prompt_done = prompt.clone();
        let final_ev = stream::once(async move {
            use crate::openai_compat::{approx_tokens_from_chars, approx_tokens_from_text};
            let pt = approx_tokens_from_text(&prompt_done);
            let ct = approx_tokens_from_chars(chars_out.load(Ordering::Relaxed));
            let chunk = json!({
                "id": id_done,
                "object": "chat.completion.chunk",
                "model": model_done,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "total_tokens": pt + ct
                }
            });
            Ok::<Bytes, Error>(Bytes::from(sse::frame(&chunk.to_string())))
        });
        let stream = mapped.chain(final_ev).chain(stream::once(async move {
            Ok::<Bytes, Error>(Bytes::from(sse::done()))
        }));
        let body = Body::from_stream(stream);
        return (
            [(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")],
            body,
        )
            .into_response();
    }

    let content = match st.scheduler.submit_plain(prompt.clone(), max_new).await {
        Ok(s) => s,
        Err(e) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "scheduler_error",
                e.to_string(),
            );
        }
    };
    (
        StatusCode::OK,
        Json(json!({
            "id": Uuid::new_v4().to_string(),
            "object": "chat.completion",
            "model": body.model,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": content },
                "finish_reason": finish_reason,
            }],
            "usage": chat_usage(&prompt, &content),
        })),
    )
        .into_response()
}
