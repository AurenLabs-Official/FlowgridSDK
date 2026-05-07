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
        ("train-lora-smoke", "train") => Ok(vec![
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
            "--lora".into(),
            "--lora-targets".into(),
            "q,v,o".into(),
            "--run-report-out".into(),
            "target/mlops/train_lora_smoke_report.json".into(),
        ]),
        ("train-golden-llm", "train") => Ok(vec![
            "cargo".into(),
            "run".into(),
            "-p".into(),
            "flowgrid-cli".into(),
            "--profile".into(),
            "local".into(),
            "--".into(),
            "train".into(),
            "--tokens".into(),
            env.train_tokens.into(),
            "--steps".into(),
            "8".into(),
            "--epochs".into(),
            "2".into(),
            "--batch-size".into(),
            "2".into(),
            "--learn".into(),
            "--seed".into(),
            "7".into(),
            "--run-report-out".into(),
            "target/mlops/golden_llm_candidate_train.json".into(),
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
        ("eval-val-gate", "eval") => {
            if env.eval_dataset.is_empty() {
                return Err(Cow::Borrowed(
                    "set FLOWGRID_UI_EVAL_DATASET for template eval-val-gate",
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
                "--split".into(),
                "val".into(),
                "--train-frac".into(),
                "0.8".into(),
                "--val-frac".into(),
                "0.1".into(),
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
            argv.push("target/mlops/eval_val_gate_report.json".into());
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
        ("prepare-readme", _)
        | ("train-tiny", _)
        | ("train-lora-smoke", _)
        | ("train-golden-llm", _)
        | ("eval-smoke", _)
        | ("eval-val-gate", _)
        | ("generate-demo", _) => {
            Err(Cow::Borrowed(
                "template/kind mismatch: prepare-readme+prepare, train-tiny+train, train-lora-smoke+train, train-golden-llm+train, eval-smoke+eval, eval-val-gate+eval, generate-demo+generate",
            ))
        }
        _ => Err(Cow::Owned(format!("unknown template '{name}'"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_env() -> JobEnv<'static> {
        JobEnv {
            train_tokens: "target/mlops/tokens.bin",
            prepare_input: "README.md",
            eval_dataset: "target/mlops/tokens.bin",
            eval_load: "",
            generate_load: "target/mlops/ckpt",
            generate_prompt: "hello",
        }
    }

    #[test]
    fn train_lora_template_contains_lora_flags() {
        let req = StartJobReq {
            kind: "train".into(),
            template: Some("train-lora-smoke".into()),
            advanced: false,
            command: vec![],
        };
        let argv = resolve_job_argv(&req, false, &test_env()).expect("must resolve");
        assert!(argv.contains(&"--lora".to_string()));
        assert!(argv.contains(&"target/mlops/train_lora_smoke_report.json".to_string()));
    }

    #[test]
    fn eval_val_gate_template_sets_split_val() {
        let req = StartJobReq {
            kind: "eval".into(),
            template: Some("eval-val-gate".into()),
            advanced: false,
            command: vec![],
        };
        let argv = resolve_job_argv(&req, false, &test_env()).expect("must resolve");
        assert!(argv.windows(2).any(|w| w == ["--split", "val"]));
        assert!(argv.contains(&"target/mlops/eval_val_gate_report.json".to_string()));
    }
}
