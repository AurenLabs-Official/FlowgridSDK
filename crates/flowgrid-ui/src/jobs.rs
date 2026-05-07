use serde::{Deserialize, Serialize};
use std::borrow::Cow;

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
    /// Allowlisted template name. Omit only when `advanced` is true.
    pub template: Option<String>,
    /// When true, `command` is used directly (requires `FLOWGRID_UI_ALLOW_ADVANCED=1` on the server).
    #[serde(default)]
    pub advanced: bool,
    #[serde(default)]
    pub command: Vec<String>,
}

/// Build argv for `tokio::process::Command` from a template or advanced `command`.
pub fn resolve_job_argv(
    req: &StartJobReq,
    allow_advanced: bool,
    train_tokens: &str,
    prepare_input: &str,
) -> Result<Vec<String>, Cow<'static, str>> {
    if req.advanced {
        if !allow_advanced {
            return Err(Cow::Borrowed(
                "advanced jobs disabled; set FLOWGRID_UI_ALLOW_ADVANCED=1 on the UI server",
            ));
        }
        if req.command.is_empty() {
            return Err(Cow::Borrowed("advanced job requires non-empty command"));
        }
        return Ok(req.command.clone());
    }

    let name = req
        .template
        .as_deref()
        .ok_or(Cow::Borrowed("template is required unless advanced=true"))?;

    match (name, req.kind.as_str()) {
        ("prepare-readme", "prepare") => Ok(vec![
            "cargo".into(),
            "run".into(),
            "-p".into(),
            "flowgrid-cli".into(),
            "--".into(),
            "prepare".into(),
            "-i".into(),
            prepare_input.into(),
            "-o".into(),
            train_tokens.into(),
        ]),
        ("train-tiny", "train") => Ok(vec![
            "cargo".into(),
            "run".into(),
            "-p".into(),
            "flowgrid-cli".into(),
            "--".into(),
            "train".into(),
            "--tokens".into(),
            train_tokens.into(),
            "--steps".into(),
            "8".into(),
            "--vocab".into(),
            "256".into(),
            "--block".into(),
            "32".into(),
            "--layers".into(),
            "2".into(),
            "--embd".into(),
            "64".into(),
        ]),
        ("prepare-readme", _) | ("train-tiny", _) => Err(Cow::Borrowed(
            "template/kind mismatch: use prepare-readme with kind prepare, or train-tiny with kind train",
        )),
        _ => Err(Cow::Owned(format!("unknown template '{name}'"))),
    }
}
