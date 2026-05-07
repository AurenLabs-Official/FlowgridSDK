use anyhow::{anyhow, Result};
use flowgrid_tokenizer::FgTokenizer;
use std::sync::mpsc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::completion::{CompletionMeta, PlainOutput, StreamPart};
use crate::engine::{serve_sampling_from_env, serve_seed_from_env, LocalLlm};

#[derive(Clone)]
pub struct Scheduler {
    worker_txs: Arc<Vec<mpsc::SyncSender<SchedulerReq>>>,
    next_worker: Arc<AtomicUsize>,
    stream_buffer: usize,
}

enum SchedulerReq {
    Plain {
        prompt: String,
        max_new: usize,
        respond: mpsc::Sender<Result<PlainOutput>>,
    },
    Stream {
        prompt: String,
        max_new: usize,
        chunk_tx: tokio::sync::mpsc::Sender<Result<StreamPart, anyhow::Error>>,
    },
}

#[derive(Debug)]
pub struct SchedulerConfig {
    pub queue_depth: usize,
    pub worker_threads: usize,
    pub request_timeout_ms: u64,
    pub stream_buffer: usize,
    pub max_new_tokens: Option<usize>,
}

impl Scheduler {
    pub fn start(cfg: SchedulerConfig, mut llm: Option<LocalLlm>) -> Self {
        let tokenizer_path = std::env::var("FLOWGRID_SERVE_TOKENIZER").ok();
        let timeout_ms = cfg.request_timeout_ms.max(1);
        let worker_threads_cfg = cfg.worker_threads.max(1);
        let worker_threads = if llm.is_some() {
            if worker_threads_cfg > 1 {
                tracing::warn!(
                    requested = worker_threads_cfg,
                    "FLOWGRID_SERVE_WORKERS>1 is currently unsupported with local checkpoint inference; using 1 worker"
                );
            }
            1
        } else {
            worker_threads_cfg
        };
        let queue_depth = cfg.queue_depth.max(1);
        let max_new_tokens = cfg.max_new_tokens.filter(|v| *v > 0);
        let mut worker_txs = Vec::with_capacity(worker_threads);
        for worker_idx in 0..worker_threads {
            let (req_tx, req_rx) = mpsc::sync_channel::<SchedulerReq>(queue_depth);
            worker_txs.push(req_tx);
            let llm_worker = if worker_idx == 0 { llm.take() } else { None };
            let tok_worker = tokenizer_path
                .as_ref()
                .and_then(|p| FgTokenizer::from_file(p).ok());
            thread::spawn(move || {
                Self::inference_loop(req_rx, llm_worker, tok_worker, timeout_ms, max_new_tokens);
            });
        }
        Self {
            worker_txs: Arc::new(worker_txs),
            next_worker: Arc::new(AtomicUsize::new(0)),
            stream_buffer: cfg.stream_buffer.max(1),
        }
    }

    fn inference_loop(
        req_rx: mpsc::Receiver<SchedulerReq>,
        llm: Option<LocalLlm>,
        tokenizer: Option<FgTokenizer>,
        timeout_ms: u64,
        max_new_tokens: Option<usize>,
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
                    let max_new = match max_new_tokens {
                        Some(cap) => max_new.min(cap),
                        None => max_new,
                    };
                    let r = Self::run_plain(&llm, tokenizer.as_ref(), &prompt, max_new, deadline);
                    let _ = respond.send(r);
                }
                SchedulerReq::Stream {
                    prompt,
                    max_new,
                    chunk_tx,
                } => {
                    let max_new = match max_new_tokens {
                        Some(cap) => max_new.min(cap),
                        None => max_new,
                    };
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
                                let _ = chunk_tx
                                    .blocking_send(Ok(StreamPart::Delta(piece.to_string())));
                            },
                        );
                        match r {
                            Ok(meta) => {
                                let _ = chunk_tx.blocking_send(Ok(StreamPart::Done(meta)));
                            }
                            Err(e) => {
                                let _ = chunk_tx.blocking_send(Err(e));
                            }
                        }
                    } else {
                        let (text, meta) = Self::echo_fallback_timed(
                            tokenizer.as_ref(),
                            &prompt,
                            max_new,
                            deadline,
                        );
                        let _ = chunk_tx.blocking_send(Ok(StreamPart::Delta(text)));
                        let _ = chunk_tx.blocking_send(Ok(StreamPart::Done(meta)));
                    }
                }
            }
        }
    }

    fn run_plain(
        llm: &Option<LocalLlm>,
        tokenizer: Option<&FgTokenizer>,
        prompt: &str,
        max_new: usize,
        deadline: Option<Instant>,
    ) -> Result<PlainOutput> {
        if let Some(engine) = llm {
            let sampling = serve_sampling_from_env();
            let seed = serve_seed_from_env();
            let (text, meta) = engine.complete(prompt, max_new.max(1), sampling, seed, deadline)?;
            return Ok(PlainOutput { text, meta });
        }
        let (text, meta) = Self::echo_fallback_timed(tokenizer, prompt, max_new, deadline);
        Ok(PlainOutput { text, meta })
    }

    fn echo_fallback_timed(
        tokenizer: Option<&FgTokenizer>,
        prompt: &str,
        max_new: usize,
        deadline: Option<Instant>,
    ) -> (String, CompletionMeta) {
        if let Some(d) = deadline {
            if Instant::now() > d {
                let msg = "inference timeout".to_string();
                let meta = CompletionMeta::heuristic_echo(prompt, &msg, "stop");
                return (msg, meta);
            }
        }
        Self::echo_fallback(tokenizer, prompt, max_new)
    }

    fn echo_fallback(
        tokenizer: Option<&FgTokenizer>,
        prompt: &str,
        max_new: usize,
    ) -> (String, CompletionMeta) {
        let max_new = max_new.max(1);
        match tokenizer {
            Some(tok) => match tok.encode(prompt, true) {
                Ok(ids) => {
                    let start = ids.len().saturating_sub(max_new);
                    let slice = &ids[start..];
                    let body = tok.decode(slice, true).unwrap_or_default();
                    let text = format!("echo: {body}");
                    let meta = CompletionMeta {
                        prompt_tokens: ids.len() as u32,
                        completion_tokens: slice.len() as u32,
                        finish_reason: "stop",
                    };
                    (text, meta)
                }
                Err(_) => {
                    let echo_body = prompt.chars().take(max_new).collect::<String>();
                    let text = format!("echo: {echo_body}");
                    let meta = CompletionMeta::heuristic_echo(prompt, &text, "stop");
                    (text, meta)
                }
            },
            None => {
                let echo_body = prompt.chars().take(max_new).collect::<String>();
                let text = format!("echo: {echo_body}");
                let meta = CompletionMeta::heuristic_echo(prompt, &text, "stop");
                (text, meta)
            }
        }
    }

    pub async fn submit_plain(&self, prompt: String, max_new: usize) -> Result<PlainOutput> {
        let (tx, rx) = std::sync::mpsc::channel();
        let idx = self.next_worker.fetch_add(1, Ordering::Relaxed) % self.worker_txs.len();
        self.worker_txs[idx]
            .send(SchedulerReq::Plain {
                prompt,
                max_new,
                respond: tx,
            })
            .map_err(|_| anyhow!("scheduler queue closed"))?;
        let recv_result = tokio::task::spawn_blocking(move || rx.recv())
            .await
            .map_err(|e| anyhow!(e.to_string()))?;
        match recv_result {
            Ok(plain_result) => plain_result,
            Err(_) => Err(anyhow!("scheduler dropped response")),
        }
    }

    pub async fn submit_stream(
        &self,
        prompt: String,
        max_new: usize,
    ) -> Result<tokio::sync::mpsc::Receiver<Result<StreamPart, anyhow::Error>>> {
        let (chunk_tx, chunk_rx) = tokio::sync::mpsc::channel(self.stream_buffer.max(1));
        let idx = self.next_worker.fetch_add(1, Ordering::Relaxed) % self.worker_txs.len();
        self.worker_txs[idx]
            .send(SchedulerReq::Stream {
                prompt,
                max_new,
                chunk_tx,
            })
            .map_err(|_| anyhow!("scheduler queue closed"))?;
        Ok(chunk_rx)
    }
}
