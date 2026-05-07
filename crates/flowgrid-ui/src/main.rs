//! Local dashboard shell with job control and checkpoint browser.

mod jobs;

use anyhow::{Context, Result};
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::Json;
use axum::Router;
use jobs::{JobRecord, StartJobReq};
use rusqlite::Connection;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::process::Command;

struct UiState {
    db: Mutex<Connection>,
    children: Mutex<HashMap<String, u32>>,
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
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
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            status TEXT NOT NULL,
            started_ms INTEGER NOT NULL,
            log_path TEXT NOT NULL,
            child_pid INTEGER
        );
        "#,
    )?;
    Ok(conn)
}

async fn index() -> impl IntoResponse {
    (
        StatusCode::OK,
        [("content-type", "text/html; charset=utf-8")],
        "<!doctype html><title>flowgrid-ui</title><h1>Run Control Center</h1><p>Use /api/jobs, /api/checkpoints, /api/runs/compare</p>",
    )
}

async fn health() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

async fn ready(State(st): State<Arc<UiState>>) -> impl IntoResponse {
    let conn = st.db.lock().expect("db");
    let ok = conn
        .prepare("SELECT 1")
        .and_then(|mut s| s.query_row([], |_r| Ok(())))
        .is_ok();
    if ok {
        (StatusCode::OK, "ready")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "not ready")
    }
}

async fn api_runs(State(st): State<Arc<UiState>>) -> impl IntoResponse {
    let conn = st.db.lock().expect("db");
    let mut stmt = conn
        .prepare("SELECT id, created_ms, note FROM runs ORDER BY created_ms DESC LIMIT 50")
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
    let out: Vec<_> = rows.filter_map(|r| r.ok()).collect();
    Json(json!({ "runs": out }))
}

async fn api_jobs(State(st): State<Arc<UiState>>) -> impl IntoResponse {
    let conn = st.db.lock().expect("db");
    let mut stmt = conn
        .prepare("SELECT id, kind, status, started_ms, log_path, child_pid FROM jobs ORDER BY started_ms DESC LIMIT 100")
        .unwrap();
    let rows = stmt
        .query_map([], |r| {
            Ok(JobRecord {
                id: r.get(0)?,
                kind: r.get(1)?,
                status: r.get(2)?,
                started_ms: r.get(3)?,
                log_path: r.get(4)?,
                child_pid: r.get::<_, Option<u32>>(5)?,
            })
        })
        .unwrap();
    let out: Vec<_> = rows.filter_map(|r| r.ok()).collect();
    Json(json!({ "jobs": out }))
}

async fn api_start_job(
    State(st): State<Arc<UiState>>,
    Json(req): Json<StartJobReq>,
) -> axum::response::Response {
    const ALLOWED_KINDS: &[&str] = &["train", "eval", "generate", "prepare", "merge-lora"];
    if !ALLOWED_KINDS.contains(&req.kind.as_str()) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "kind not allowlisted", "allowed": ALLOWED_KINDS })),
        )
            .into_response();
    }
    if !req.command.is_empty() {
        let prog = req.command[0].as_str();
        if prog != "cargo" && !prog.ends_with("flowgrid-llm") && prog != "flowgrid-llm" {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "command must start with cargo or flowgrid-llm" })),
            )
                .into_response();
        }
    }
    let id = uuid::Uuid::new_v4().to_string();
    let log_path = format!("flowgrid_job_{id}.log");

    let mut cmd = if req.command.is_empty() {
        #[cfg(windows)]
        {
            let mut c = Command::new("cmd");
            c.arg("/C").arg("echo no command");
            c
        }
        #[cfg(not(windows))]
        {
            let mut c = Command::new("sh");
            c.arg("-c").arg("echo no command");
            c
        }
    } else {
        Command::new(&req.command[0])
    };
    if !req.command.is_empty() && req.command.len() > 1 {
        cmd.args(&req.command[1..]);
    }
    let log = std::fs::File::create(&log_path).ok();
    if let Some(log) = log {
        if let Ok(cloned) = log.try_clone() {
            cmd.stdout(cloned);
            cmd.stderr(log);
        }
    }
    let child = cmd.spawn().ok();
    let pid = child.as_ref().and_then(|c| c.id());
    {
        let conn = st.db.lock().expect("db");
        let _ = conn.execute(
            "INSERT INTO jobs (id, kind, status, started_ms, log_path, child_pid) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![id, req.kind, "running", now_ms(), log_path, pid],
        );
    }
    if let Some(pid) = pid {
        st.children
            .lock()
            .expect("children")
            .insert(id.clone(), pid);
    }
    (StatusCode::OK, Json(json!({ "id": id, "pid": pid }))).into_response()
}

async fn api_stop_job(State(st): State<Arc<UiState>>, Path(id): Path<String>) -> impl IntoResponse {
    let pid = st.children.lock().expect("children").remove(&id);
    if let Some(pid) = pid {
        #[cfg(windows)]
        {
            let _ = Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/T", "/F"])
                .spawn();
        }
        #[cfg(unix)]
        {
            let _ = Command::new("kill")
                .args(["-TERM", &pid.to_string()])
                .spawn();
        }
        #[cfg(not(any(windows, unix)))]
        {
            let _ = pid;
        }
    }
    let conn = st.db.lock().expect("db");
    let _ = conn.execute(
        "UPDATE jobs SET status = 'stopped' WHERE id = ?1",
        rusqlite::params![id],
    );
    (StatusCode::OK, Json(json!({ "ok": true })))
}

async fn api_job_log(Path(id): Path<String>) -> impl IntoResponse {
    let path = format!("flowgrid_job_{id}.log");
    match tokio::fs::read_to_string(&path).await {
        Ok(body) => (StatusCode::OK, body).into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "log not found".to_string()).into_response(),
    }
}

async fn api_checkpoints() -> impl IntoResponse {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(".") {
        for ent in rd.flatten() {
            if ent.path().join("manifest.json").exists() {
                out.push(ent.path().display().to_string());
            }
        }
    }
    Json(json!({ "checkpoints": out }))
}

async fn api_checkpoint(Path(id): Path<String>) -> impl IntoResponse {
    let path = PathBuf::from(id).join("manifest.json");
    match std::fs::read_to_string(&path) {
        Ok(body) => (StatusCode::OK, body).into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "missing manifest".to_string()).into_response(),
    }
}

#[derive(Deserialize)]
struct CompareQ {
    a: String,
    b: String,
}

async fn api_compare(Query(q): Query<CompareQ>) -> impl IntoResponse {
    Json(json!({
        "a": q.a,
        "b": q.b,
        "summary": "compare endpoint scaffolded; integrate run metrics stream in next patch"
    }))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let db_path = std::env::var("FLOWGRID_UI_DB").unwrap_or_else(|_| "flowgrid_ui.sqlite".into());
    let conn = init_db(&PathBuf::from(&db_path)).context("init db")?;
    let state = Arc::new(UiState {
        db: Mutex::new(conn),
        children: Mutex::new(HashMap::new()),
    });
    let app = Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/api/runs", get(api_runs))
        .route("/api/runs/compare", get(api_compare))
        .route("/api/jobs", get(api_jobs))
        .route("/api/jobs/start", post(api_start_job))
        .route("/api/jobs/:id/stop", post(api_stop_job))
        .route("/api/jobs/:id/log", get(api_job_log))
        .route("/api/checkpoints", get(api_checkpoints))
        .route("/api/checkpoints/:id", get(api_checkpoint))
        .with_state(state);
    let addr: std::net::SocketAddr = "127.0.0.1:9010".parse().unwrap();
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("flowgrid-ui http://{addr}/");
    axum::serve(listener, app).await?;
    Ok(())
}
