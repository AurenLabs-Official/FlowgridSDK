use axum::body::Body;
use axum::extract::State;
use axum::http::header;
use axum::http::HeaderMap;
use axum::response::IntoResponse;
use axum::Json;
use bytes::Bytes;
use futures::stream::{self, StreamExt};
use serde_json::json;
use std::io::Error;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::completion::StreamPart;
use crate::openai_compat::{
    openai_error_response, openai_error_value, responses_usage_tokens,
};
use crate::sse;
use crate::types::ResponsesReq;
use crate::AppState;

fn flatten_input(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(a) => a
            .iter()
            .filter_map(|x| x.as_str().map(|s| s.to_string()))
            .collect::<Vec<_>>()
            .join(" "),
        _ => v.to_string(),
    }
}

pub async fn responses(
    State(st): State<std::sync::Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<ResponsesReq>,
) -> axum::response::Response {
    use axum::http::StatusCode;
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
    let text_in = flatten_input(&body.input);
    let max_new = body.max_tokens.unwrap_or(128).min(4096) as usize;

    if body.stream == Some(true) {
        let rx = match st.scheduler.submit_stream(text_in.clone(), max_new).await {
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
                        let delta = json!({
                            "id": id_chunk,
                            "object": "response.output_text.delta",
                            "model": model_chunk,
                            "delta": piece
                        });
                        Ok(Bytes::from(sse::frame(&delta.to_string())))
                    }
                    Ok(StreamPart::Done(meta)) => {
                        let usage = responses_usage_tokens(
                            meta.prompt_tokens,
                            meta.completion_tokens,
                        );
                        let evt = json!({
                            "id": id_chunk,
                            "object": "response.completed",
                            "model": model_chunk,
                            "status": "completed",
                            "finish_reason": meta.finish_reason,
                            "usage": usage,
                        });
                        Ok(Bytes::from(sse::frame(&evt.to_string())))
                    }
                    Err(e) => {
                        let err = openai_error_value(
                            "server_error",
                            "inference_error",
                            e.to_string(),
                        );
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

    let out = match st.scheduler.submit_plain(text_in.clone(), max_new).await {
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
            "object": "response",
            "status": "completed",
            "model": body.model,
            "output_text": out.text,
            "finish_reason": out.meta.finish_reason,
            "usage": responses_usage_tokens(out.meta.prompt_tokens, out.meta.completion_tokens),
        })),
    )
        .into_response()
}
