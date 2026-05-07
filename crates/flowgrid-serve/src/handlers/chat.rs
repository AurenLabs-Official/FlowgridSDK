use axum::body::Body;
use axum::extract::State;
use axum::http::header;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::Json;
use bytes::Bytes;
use futures::stream::{self, StreamExt};
use serde_json::json;
use std::io::Error;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::completion::StreamPart;
use crate::openai_compat::{chat_usage_tokens, openai_error_response, openai_error_value};
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
        let mapped = ReceiverStream::new(rx).map(
            move |item: Result<StreamPart, anyhow::Error>| -> Result<Bytes, Error> {
                match item {
                    Ok(StreamPart::Delta(piece)) => {
                        let chunk = json!({
                            "id": id_chunk,
                            "object": "chat.completion.chunk",
                            "model": model_chunk,
                            "choices": [{ "index": 0, "delta": { "content": piece } }]
                        });
                        Ok(Bytes::from(sse::frame(&chunk.to_string())))
                    }
                    Ok(StreamPart::Done(meta)) => {
                        let chunk = json!({
                            "id": id_chunk,
                            "object": "chat.completion.chunk",
                            "model": model_chunk,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": meta.finish_reason,
                            }],
                            "usage": chat_usage_tokens(meta.prompt_tokens, meta.completion_tokens),
                        });
                        Ok(Bytes::from(sse::frame(&chunk.to_string())))
                    }
                    Err(e) => {
                        let err =
                            openai_error_value("server_error", "inference_error", e.to_string());
                        Ok(Bytes::from(sse::frame(&err.to_string())))
                    }
                }
            },
        );
        let stream = mapped.chain(stream::once(async move {
            Ok::<Bytes, Error>(Bytes::from(sse::done()))
        }));
        let body = Body::from_stream(stream);
        return (
            [(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")],
            body,
        )
            .into_response();
    }

    let out = match st.scheduler.submit_plain(prompt.clone(), max_new).await {
        Ok(o) => o,
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
                "message": { "role": "assistant", "content": out.text },
                "finish_reason": out.meta.finish_reason,
            }],
            "usage": chat_usage_tokens(out.meta.prompt_tokens, out.meta.completion_tokens),
        })),
    )
        .into_response()
}
