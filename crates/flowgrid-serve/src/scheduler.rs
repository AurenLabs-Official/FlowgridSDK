use anyhow::{anyhow, Result};
use tokio::sync::{mpsc, oneshot};
use tokio::time::{timeout, Duration};

#[derive(Clone)]
pub struct Scheduler {
    tx: mpsc::Sender<Job>,
}

#[derive(Debug)]
pub struct SchedulerConfig {
    pub queue_depth: usize,
    pub request_timeout_ms: u64,
}

pub struct Job {
    pub prompt_text: String,
    pub max_new: usize,
    pub done: oneshot::Sender<Result<String>>,
}

impl Scheduler {
    pub fn start(cfg: SchedulerConfig) -> Self {
        let (tx, mut rx) = mpsc::channel::<Job>(cfg.queue_depth.max(1));
        tokio::spawn(async move {
            while let Some(job) = rx.recv().await {
                let result = timeout(
                    Duration::from_millis(cfg.request_timeout_ms.max(1)),
                    async move {
                        // Placeholder decode path; real model backend is wired in follow-up.
                        Ok::<String, anyhow::Error>(format!(
                            "{}{}",
                            "echo: ",
                            job.prompt_text.chars().take(job.max_new.max(1)).collect::<String>()
                        ))
                    },
                )
                .await
                .map_err(|_| anyhow!("scheduler timeout"))
                .and_then(|v| v);
                let _ = job.done.send(result);
            }
        });
        Self { tx }
    }

    pub async fn submit(&self, prompt_text: String, max_new: usize) -> Result<String> {
        let (done_tx, done_rx) = oneshot::channel();
        self.tx
            .send(Job {
                prompt_text,
                max_new,
                done: done_tx,
            })
            .await
            .map_err(|_| anyhow!("scheduler queue closed"))?;
        done_rx.await.map_err(|_| anyhow!("scheduler worker dropped"))?
    }
}
