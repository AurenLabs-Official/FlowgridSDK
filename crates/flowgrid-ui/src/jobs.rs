use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRecord {
    pub id: String,
    pub kind: String,
    pub status: String,
    pub started_ms: i64,
    pub log_path: String,
    pub child_pid: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StartJobReq {
    pub kind: String,
    pub command: Vec<String>,
}
