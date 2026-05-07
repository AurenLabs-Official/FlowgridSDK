use anyhow::{anyhow, Result};
use flowgrid_tokenizer::FgTokenizer;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::engine::{serve_sampling_from_env, serve_seed_from_env, LocalLlm};

#[derive(Clone)]
pub struct Scheduler {
    req_tx: Arc<mpsc::SyncSender<SchedulerReq>>,
}

enum SchedulerReq {
    Plain {
        prompt: String,
        max_new: usize,
        respond: mpsc::Sender<Result<String>>,
    },
    Stream {
        prompt: String,
        max_new: usize,
        chunk_tx: mpsc::Sender<Result<String>>,
    },
}

#[derive(Debug)]
pub struct SchedulerConfig {
    pub queue_depth: usize,
    pub request_timeout_ms: u64,
}

impl Scheduler {
    pub fn start(cfg: SchedulerConfig, llm: Option<LocalLlm>) -> Self {
        let tokenizer = std::env::var("FLOWGRID_SERVE_TOKENIZER")
            .ok()
            .and_then(|p| FgTokenizer::from_file(&p).ok());
        let timeout_ms = cfg.request_timeout_ms.max(1);
        let (req_tx, req_rx) = mpsc::sync_channel::<SchedulerReq>(cfg.queue_depth.max(1));
        let req_tx = Arc::new(req_tx);
        let dispatch = Arc::clone(&req_tx);
        thread::spawn(move || {
            Self::inference_loop(req_rx, llm, tokenizer, timeout_ms);
        });
        Self { req_tx: dispatch }
    }

    fn inference_loop(
        req_rx: mpsc::Receiver<SchedulerReq>,
        llm: Option<LocalLlm>,
        tokenizer: Option<FgTokenizer>,
        timeout_ms: u64,
    ) {
        let timeout = Duration::from_millis(timeout_ms.max(1));
        while let Ok(req) = req_rx.recv() {
            let deadline = Instant::now() + timeout;
            let deadline = Some(deadline);
            match req {
                SchedulerReq::Plain {
                    prompt,
                    max_new,
                    respond,
                } => {
                    let r = Self::run_plain(&llm, tokenizer.as_ref(), &prompt, max_new, deadline);
                    let _ = respond.send(r);
                }
                SchedulerReq::Stream {
                    prompt,
                    max_new,
                    chunk_tx,
                } => {
                    if let Some(ref engine) = llm {
                        let sampling = serve_sampling_from_env();
                        let seed = serve_seed_from_env();
                        let r = engine.complete_stream(
                            &prompt,
                            max_new.max(1),
                            sampling,
                            seed,
                            deadline,
                            |piece| {
                                let chunk = serde_json::json!({
                                    "object": "text.delta",
                                    "delta": piece,
                                })
                                .to_string();
                                let _ = chunk_tx.send(Ok(chunk));
                            },
                        );
                        if let Err(e) = r {
                            let _ = chunk_tx.send(Err(anyhow!("{e}")));
                        }
                    } else {
                        let r = Self::echo_fallback_timed(
                            tokenizer.as_ref(),
                            &prompt,
                            max_new,
                            deadline,
                        );
                        let chunk = serde_json::json!({
                            "object": "text.delta",
                            "delta": r,
                        })
                        .to_string();
                        let _ = chunk_tx.send(Ok(chunk));
                    }
                }
            }
        }
    }

    /// Echo / tokenizer path bounded by the same deadline as LLM inference.
    fn echo_fallback_timed(
        tokenizer: Option<&FgTokenizer>,
        prompt: &str,
        max_new: usize,
        deadline: Option<Instant>,
    ) -> String {
        if let Some(d) = deadline {
            if Instant::now() > d {
                return "inference timeout".to_string();
            }
        }
        Self::echo_fallback(tokenizer, prompt, max_new)
    }

    fn echo_fallback(tokenizer: Option<&FgTokenizer>, prompt: &str, max_new: usize) -> String {
        let max_new = max_new.max(1);
        if let Some(tok) = tokenizer {
            match tok.encode(prompt, true) {
                Ok(ids) => {
                    let start = ids.len().saturating_sub(max_new);
                    tok.decode(&ids[start..], true).unwrap_or_default()
                }
                Err(_) => format!("echo: {}", prompt.chars().take(max_new).collect::<String>()),
            }
        } else {
            format!("echo: {}", prompt.chars().take(max_new).collect::<String>())
        }
    }

    fn run_plain(
        llm: &Option<LocalLlm>,
        tokenizer: Option<&FgTokenizer>,
        prompt: &str,
        max_new: usize,
        deadline: Option<Instant>,
    ) -> Result<String> {
        let max_new = max_new.max(1);
        if let Some(engine) = llm {
            let sampling = serve_sampling_from_env();
            let seed = serve_seed_from_env();
            engine
                .complete(prompt, max_new, sampling, seed, deadline)
                .map_err(|e| anyhow!("{e}"))
        } else if let Some(tok) = tokenizer {
            if let Some(d) = deadline {
                if Instant::now() > d {
                    return Err(anyhow!("inference timeout"));
                }
            }
            let ids = tok
                .encode(prompt, true)
                .map_err(|e| anyhow!("tokenize: {e}"))?;
            if let Some(d) = deadline {
                if Instant::now() > d {
                    return Err(anyhow!("inference timeout"));
                }
            }
            let start = ids.len().saturating_sub(max_new);
            tok.decode(&ids[start..], true)
                .map_err(|e| anyhow!("decode: {e}"))
        } else {
            Ok(format!(
                "{}{}",
                "echo: ",
                prompt.chars().take(max_new).collect::<String>()
            ))
        }
    }

    pub async fn submit_plain(&self, prompt_text: String, max_new: usize) -> Result<String> {
        let (tx, rx) = mpsc::channel::<Result<String>>();
        self.req_tx
            .send(SchedulerReq::Plain {
                prompt: prompt_text,
                max_new,
                respond: tx,
            })
            .map_err(|_| anyhow!("scheduler queue closed"))?;
        tokio::task::spawn_blocking(move || rx.recv().map_err(|_| anyhow!("worker died"))?)
            .await
            .map_err(|_| anyhow!("blocking join"))?
    }

    pub async fn submit_stream(
        &self,
        prompt_text: String,
        max_new: usize,
    ) -> Result<tokio::sync::mpsc::Receiver<Result<String>>> {
        let (sync_tx, sync_rx) = mpsc::channel::<Result<String>>();
        self.req_tx
            .send(SchedulerReq::Stream {
                prompt: prompt_text,
                max_new,
                chunk_tx: sync_tx,
            })
            .map_err(|_| anyhow!("scheduler queue closed"))?;
        let (async_tx, async_rx) = tokio::sync::mpsc::channel(256);
        tokio::task::spawn_blocking(move || {
            for item in sync_rx {
                if async_tx.blocking_send(item).is_err() {
                    break;
                }
            }
        });
        Ok(async_rx)
    }
}
