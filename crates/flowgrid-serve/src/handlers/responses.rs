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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::openai_compat::{openai_error_response, responses_usage};
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
    let finish_reason = if st.local_llm_loaded {
        "length"
    } else {
        "stop"
    };

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
        let chars_out = Arc::new(AtomicUsize::new(0));
        let co = Arc::clone(&chars_out);
        let mapped = ReceiverStream::new(rx).map(move |res| -> Result<Bytes, Error> {
            let inner = res.map_err(|e| std::io::Error::other(e.to_string()))?;
            let v: serde_json::Value =
                serde_json::from_str(&inner).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
            let piece = v["delta"].as_str().unwrap_or("");
            co.fetch_add(piece.chars().count(), Ordering::Relaxed);
            let delta = json!({
                "id": id_chunk,
                "object": "response.output_text.delta",
                "model": model_chunk,
                "delta": piece
            });
            Ok(Bytes::from(sse::frame(&delta.to_string())))
        });
        let id_done = id.clone();
        let model_done = model.clone();
        let prompt_done = text_in.clone();
        let fr = finish_reason;
        let completed = stream::once(async move {
            use crate::openai_compat::{approx_tokens_from_chars, approx_tokens_from_text};
            let pt = approx_tokens_from_text(&prompt_done);
            let ct = approx_tokens_from_chars(chars_out.load(Ordering::Relaxed));
            let usage = json!({
                "input_tokens": pt,
                "output_tokens": ct,
                "total_tokens": pt + ct
            });
            let evt = json!({
                "id": id_done,
                "object": "response.completed",
                "model": model_done,
                "status": "completed",
                "finish_reason": fr,
                "usage": usage,
            });
            Ok::<Bytes, Error>(Bytes::from(sse::frame(&evt.to_string())))
        });
        let stream = mapped.chain(completed).chain(stream::once(async move {
            Ok::<Bytes, Error>(Bytes::from(sse::done()))
        }));
        let body = Body::from_stream(stream);
        return (
            [(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")],
            body,
        )
            .into_response();
    }

    let text = match st.scheduler.submit_plain(text_in.clone(), max_new).await {
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
            "object": "response",
            "status": "completed",
            "model": body.model,
            "output_text": text,
            "finish_reason": finish_reason,
            "usage": responses_usage(&text_in, &text),
        })),
    )
        .into_response()
}
