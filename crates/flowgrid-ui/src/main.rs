//! Local dashboard shell (REST + SQLite). Replace HTML with Leptos SSR when you need richer UX.

use anyhow::Result;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use rusqlite::Connection;
use serde_json::json;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

struct UiState {
    db: Mutex<Connection>,
}

fn init_db(path: &PathBuf) -> Result<Connection> {
    let conn = Connection::open(path)?;
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            created_ms INTEGER NOT NULL,
            note TEXT
        );
        "#,
    )?;
    Ok(conn)
}

async fn index() -> impl IntoResponse {
    (
        StatusCode::OK,
        [("content-type", "text/html; charset=utf-8")],
        "<!doctype html><title>flowgrid-ui</title><p>REST: <code>/api/runs</code>, health <code>/health</code></p>",
    )
}

async fn health() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

async fn api_runs(State(st): State<Arc<UiState>>) -> impl IntoResponse {
    let conn = st.db.lock().expect("db");
    let mut stmt = conn
        .prepare("SELECT id, created_ms, note FROM runs ORDER BY created_ms DESC LIMIT 20")
        .unwrap();
    let rows = stmt
        .query_map([], |r| {
            Ok(json!({
                "id": r.get::<_, String>(0)?,
                "created_ms": r.get::<_, i64>(1)?,
                "note": r.get::<_, Option<String>>(2)?,
            }))
        })
        .unwrap();
    let mut out = Vec::new();
    for row in rows {
        out.push(row.unwrap());
    }
    axum::Json(json!({ "runs": out }))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let db_path = std::env::var("FLOWGRID_UI_DB").unwrap_or_else(|_| "flowgrid_ui.sqlite".into());
    let conn = init_db(&PathBuf::from(&db_path))?;
    let state = Arc::new(UiState {
        db: Mutex::new(conn),
    });
    let app = Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/api/runs", get(api_runs))
        .with_state(state);
    let addr: std::net::SocketAddr = "127.0.0.1:9010".parse().unwrap();
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("flowgrid-ui http://{addr}/");
    axum::serve(listener, app).await?;
    Ok(())
}
