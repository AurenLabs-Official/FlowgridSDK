use axum::http::header;
use axum::http::HeaderMap;
use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use serde_json::json;
use uuid::Uuid;

use crate::AppState;
use crate::sse;
use crate::types::ChatReq;

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
    let content = st
        .scheduler
        .submit(prompt, 128)
        .await
        .unwrap_or_else(|_| "scheduler-error".to_string());
    if body.stream == Some(true) {
        let id = Uuid::new_v4().to_string();
        let chunk = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "model": body.model,
            "choices": [{ "index": 0, "delta": { "content": content } }]
        });
        let stream = format!("{}{}", sse::frame(&chunk.to_string()), sse::done());
        ([(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")], stream).into_response()
    } else {
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
}
