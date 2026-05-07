use axum::http::header;
use axum::http::HeaderMap;
use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use serde_json::json;
use uuid::Uuid;

use crate::AppState;
use crate::sse;
use crate::types::ResponsesReq;

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
    if crate::auth::authorize(&headers, &st.auth).is_err() {
        return axum::http::StatusCode::UNAUTHORIZED.into_response();
    }
    if !st.rate.allow() {
        return axum::http::StatusCode::TOO_MANY_REQUESTS.into_response();
    }
    let text = st
        .scheduler
        .submit(flatten_input(&body.input), 128)
        .await
        .unwrap_or_else(|_| "scheduler-error".to_string());
    if body.stream == Some(true) {
        let id = Uuid::new_v4().to_string();
        let delta = json!({
            "id": id,
            "object": "response.output_text.delta",
            "model": body.model,
            "delta": text
        });
        let stream = format!("{}{}", sse::frame(&delta.to_string()), sse::done());
        ([(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")], stream).into_response()
    } else {
        (
            axum::http::StatusCode::OK,
            Json(json!({
                "id": Uuid::new_v4().to_string(),
                "object": "response",
                "model": body.model,
                "output_text": text
            })),
        )
            .into_response()
    }
}
