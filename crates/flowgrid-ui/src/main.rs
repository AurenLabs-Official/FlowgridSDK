//! Local dashboard shell with job control and checkpoint browser.

mod jobs;

use anyhow::{Context, Result};
use axum::body::to_bytes;
use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::header;
use axum::http::{HeaderMap, StatusCode};
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
    /// Optional bearer key required on mutating routes (e.g. `/api/jobs/start`).
    job_admin_key: Option<String>,
    allow_advanced_jobs: bool,
    train_tokens_path: String,
    prepare_input_path: String,
    serve_url: Option<String>,
    serve_api_key: Option<String>,
    http: reqwest::Client,
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
    let body = r#"<!doctype html><meta charset="utf-8"><title>flowgrid-ui</title>
<h1>Run Control Center</h1>
<p>API: <code>/api/jobs/start</code> with JSON <code>{"kind":"prepare","template":"prepare-readme"}</code>
or <code>{"kind":"train","template":"train-tiny"}</code>. Free-form <code>command</code> requires
<code>advanced: true</code> and server env <code>FLOWGRID_UI_ALLOW_ADVANCED=1</code>.</p>
<p>Proxy to the LLM server: <code>POST /api/llm/v1/chat/completions</code> when <code>FLOWGRID_UI_SERVE_URL</code> is set
(use <code>FLOWGRID_UI_SERVE_API_KEY</code> for outbound auth).</p>"#;
    (
        StatusCode::OK,
        [("content-type", "text/html; charset=utf-8")],
        body,
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

fn require_job_admin(headers: &HeaderMap, expected: Option<&String>) -> Result<(), StatusCode> {
    let Some(exp) = expected else {
        return Ok(());
    };
    let auth = headers
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .unwrap_or_default();
    if auth == format!("Bearer {exp}") {
        Ok(())
    } else {
        Err(StatusCode::UNAUTHORIZED)
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
    headers: HeaderMap,
    Json(req): Json<StartJobReq>,
) -> axum::response::Response {
    if let Err(code) = require_job_admin(&headers, st.job_admin_key.as_ref()) {
        return (
            code,
            Json(json!({ "error": "missing or invalid FLOWGRID_UI_JOB_ADMIN_KEY" })),
        )
            .into_response();
    }

    const ALLOWED_KINDS: &[&str] = &["train", "eval", "generate", "prepare", "merge-lora"];
    if !ALLOWED_KINDS.contains(&req.kind.as_str()) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "kind not allowlisted", "allowed": ALLOWED_KINDS })),
        )
            .into_response();
    }

    let argv = match jobs::resolve_job_argv(
        &req,
        st.allow_advanced_jobs,
        &st.train_tokens_path,
        &st.prepare_input_path,
    ) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": e.as_ref() })),
            )
                .into_response();
        }
    };

    let prog = argv[0].as_str();
    if prog != "cargo" && !prog.ends_with("flowgrid-llm") && prog != "flowgrid-llm" {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "command must start with cargo or flowgrid-llm" })),
        )
            .into_response();
    }

    let id = uuid::Uuid::new_v4().to_string();
    let log_path = format!("flowgrid_job_{id}.log");

    let mut cmd = Command::new(&argv[0]);
    if argv.len() > 1 {
        cmd.args(&argv[1..]);
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

/// Forward JSON body to `flowgrid-serve` chat completions (adds outbound API key when configured).
async fn api_llm_chat(
    State(st): State<Arc<UiState>>,
    headers: HeaderMap,
    body: Body,
) -> axum::response::Response {
    let Some(ref base) = st.serve_url else {
        return (
            StatusCode::NOT_IMPLEMENTED,
            Json(json!({ "error": "FLOWGRID_UI_SERVE_URL is not set" })),
        )
            .into_response();
    };
    let Ok(bytes) = to_bytes(body, 512 * 1024).await else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "body too large" })),
        )
            .into_response();
    };
    let url = format!("{}/v1/chat/completions", base.trim_end_matches('/'));
    let mut rb = st.http.post(url).body(bytes.to_vec());
    rb = rb.header(header::CONTENT_TYPE, "application/json");
    if let Some(ref k) = st.serve_api_key {
        rb = rb.header(header::AUTHORIZATION, format!("Bearer {k}"));
    } else if let Some(h) = headers.get(header::AUTHORIZATION) {
        if let Ok(s) = h.to_str() {
            rb = rb.header(header::AUTHORIZATION, s);
        }
    }
    match rb.send().await {
        Ok(res) => {
            let status =
                StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            let ct = res.headers().get(header::CONTENT_TYPE).cloned();
            let body = res.bytes().await.unwrap_or_default();
            let mut resp = axum::response::Response::builder().status(status);
            if let Some(ct) = ct {
                resp = resp.header(header::CONTENT_TYPE, ct);
            }
            resp.body(Body::from(body)).unwrap_or_else(|_| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": "response build failed" })),
                )
                    .into_response()
            })
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({ "error": format!("upstream: {e}") })),
        )
            .into_response(),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let db_path = std::env::var("FLOWGRID_UI_DB").unwrap_or_else(|_| "flowgrid_ui.sqlite".into());
    let conn = init_db(&PathBuf::from(&db_path)).context("init db")?;
    let job_admin_key = std::env::var("FLOWGRID_UI_JOB_ADMIN_KEY").ok();
    let allow_advanced_jobs = std::env::var("FLOWGRID_UI_ALLOW_ADVANCED").as_deref() == Ok("1");
    let train_tokens_path =
        std::env::var("FLOWGRID_UI_TRAIN_TOKENS").unwrap_or_else(|_| "target/readme.bin".into());
    let prepare_input_path =
        std::env::var("FLOWGRID_UI_PREPARE_INPUT").unwrap_or_else(|_| "README.md".into());
    let serve_url = std::env::var("FLOWGRID_UI_SERVE_URL").ok();
    let serve_api_key = std::env::var("FLOWGRID_UI_SERVE_API_KEY").ok();
    let http = reqwest::Client::builder()
        .build()
        .context("reqwest client")?;

    let state = Arc::new(UiState {
        db: Mutex::new(conn),
        children: Mutex::new(HashMap::new()),
        job_admin_key,
        allow_advanced_jobs,
        train_tokens_path,
        prepare_input_path,
        serve_url,
        serve_api_key,
        http,
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
        .route("/api/llm/v1/chat/completions", post(api_llm_chat))
        .with_state(state);
    let addr: std::net::SocketAddr = "127.0.0.1:9010".parse().unwrap();
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("flowgrid-ui http://{addr}/");
    axum::serve(listener, app).await?;
    Ok(())
}
