use axum::body::Body;
use axum::extract::State;
use axum::http::header;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::Json;
use futures::stream::StreamExt;
use serde_json::json;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;
use crate::openai_compat::{chat_usage_tokens, openai_error_response};
use crate::scheduler::SchedulerSubmitError;
use crate::stream_sse;
use crate::types::ChatReq;
use crate::AppState;

fn scheduler_error_response(err: anyhow::Error) -> axum::response::Response {
    let msg = err.to_string();
    match err.downcast_ref::<SchedulerSubmitError>() {
        Some(SchedulerSubmitError::Overloaded) => openai_error_response(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            "server_overloaded",
            msg,
        ),
        Some(SchedulerSubmitError::Closed) | None => openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            "scheduler_error",
            msg,
        ),
    }
}

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
                return scheduler_error_response(e);
            }
        };
        let id = Uuid::new_v4().to_string();
        let model = body.model.clone();
        let id_chunk = id.clone();
        let model_chunk = model.clone();
        let stream = ReceiverStream::new(rx).flat_map(move |item| {
            let chunks = stream_sse::chat_completion_sse_chunks(&id_chunk, &model_chunk, item);
            futures::stream::iter(chunks.into_iter().map(Ok::<_, std::io::Error>))
        });
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
            return scheduler_error_response(e);
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
