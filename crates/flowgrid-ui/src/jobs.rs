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

/// Paths and defaults wired from `FLOWGRID_UI_*` environment variables.
pub struct JobEnv<'a> {
    pub train_tokens: &'a str,
    pub prepare_input: &'a str,
    pub eval_dataset: &'a str,
    pub eval_load: &'a str,
    pub generate_load: &'a str,
    pub generate_prompt: &'a str,
}

/// Build argv for `tokio::process::Command` from a template or advanced `command`.
pub fn resolve_job_argv(
    req: &StartJobReq,
    allow_advanced: bool,
    env: &JobEnv<'_>,
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
            env.prepare_input.into(),
            "-o".into(),
            env.train_tokens.into(),
        ]),
        ("train-tiny", "train") => Ok(vec![
            "cargo".into(),
            "run".into(),
            "-p".into(),
            "flowgrid-cli".into(),
            "--".into(),
            "train".into(),
            "--tokens".into(),
            env.train_tokens.into(),
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
            "--run-report-out".into(),
            "target/mlops/train_tiny_report.json".into(),
        ]),
        ("eval-smoke", "eval") => {
            if env.eval_dataset.is_empty() {
                return Err(Cow::Borrowed(
                    "set FLOWGRID_UI_EVAL_DATASET for template eval-smoke",
                ));
            }
            let mut argv = vec![
                "cargo".into(),
                "run".into(),
                "-p".into(),
                "flowgrid-cli".into(),
                "--".into(),
                "eval".into(),
                "--dataset".into(),
                env.eval_dataset.into(),
                "--block".into(),
                "32".into(),
                "--stride".into(),
                "32".into(),
            ];
            if !env.eval_load.is_empty() {
                argv.push("--load".into());
                argv.push(env.eval_load.into());
            }
            argv.push("--run-report-out".into());
            argv.push("target/mlops/eval_smoke_report.json".into());
            Ok(argv)
        }
        ("generate-demo", "generate") => {
            if env.generate_load.is_empty() {
                return Err(Cow::Borrowed(
                    "set FLOWGRID_UI_GENERATE_CHECKPOINT for template generate-demo",
                ));
            }
            Ok(vec![
                "cargo".into(),
                "run".into(),
                "-p".into(),
                "flowgrid-cli".into(),
                "--".into(),
                "generate".into(),
                "--prompt".into(),
                env.generate_prompt.into(),
                "--load".into(),
                env.generate_load.into(),
                "--max-new".into(),
                "16".into(),
            ])
        }
        ("prepare-readme", _) | ("train-tiny", _) | ("eval-smoke", _) | ("generate-demo", _) => {
            Err(Cow::Borrowed(
                "template/kind mismatch: prepare-readme+prepare, train-tiny+train, eval-smoke+eval, generate-demo+generate",
            ))
        }
        _ => Err(Cow::Owned(format!("unknown template '{name}'"))),
    }
}
