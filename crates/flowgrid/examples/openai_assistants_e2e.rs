//! Minimal Assistants workflow: thread → user message → run → poll until terminal.
//!
//! Requires **`OPENAI_API_KEY`** and **`OPENAI_ASSISTANT_ID`** (an assistant that already exists in
//! your project). Quotas, pricing, and server-side timeouts remain your responsibility; this example
//! only applies **bounded** polling (`max_polls` × sleep) so it cannot spin forever.
//!
//! Run:
//! ```text
//! OPENAI_API_KEY=sk-... OPENAI_ASSISTANT_ID=asst_... \
//!   cargo run -p flowgrid --example openai_assistants_e2e --features openai,assistants
//! ```
//!
//! Offline debugging: see integration tests under `crates/flowgrid/tests/` (e.g. Wiremock) and
//! contract fixtures under `crates/flowgrid/tests/fixtures/contracts/`.

use flowgrid::{ClientBuilder, ThreadRun};
use serde_json::json;
use std::time::Duration;

/// Run reached an end state we stop polling for (success or failure-ish).
fn run_status_terminal(status: Option<&str>) -> bool {
    matches!(
        status,
        Some(
            "completed" | "failed" | "cancelled" | "expired" | "incomplete" | "requires_action"
        )
    )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let assistant_id = std::env::var("OPENAI_ASSISTANT_ID").map_err(|_| {
        "OPENAI_ASSISTANT_ID must be set to an existing assistant id (e.g. asst_...)"
    })?;

    let client = ClientBuilder::from_env()?.build()?;

    let thread = client
        .threads()
        .create_typed(&json!({ "metadata": { "example": "openai_assistants_e2e" } }))
        .await?;
    println!("thread {}", thread.id);

    let _msg = client
        .threads()
        .thread(&thread.id)
        .messages()
        .create_typed(&json!({
            "role": "user",
            "content": "Say hello in one short sentence."
        }))
        .await?;

    let run = client
        .threads()
        .thread(&thread.id)
        .runs()
        .create_typed(&json!({ "assistant_id": assistant_id }))
        .await?;
    println!("run {} status {:?}", run.id, run.status);

    // Bounded poll: adjust for your SLA; this is not a substitute for app-level deadlines.
    const MAX_POLLS: u32 = 120;
    const POLL_SLEEP: Duration = Duration::from_millis(500);

    let mut last: ThreadRun = run;
    for _ in 0..MAX_POLLS {
        if run_status_terminal(last.status.as_deref()) {
            break;
        }
        tokio::time::sleep(POLL_SLEEP).await;
        last = client
            .threads()
            .thread(&thread.id)
            .runs()
            .retrieve_typed(&last.id)
            .await?;
        println!("run {} status {:?}", last.id, last.status);
    }

    if !run_status_terminal(last.status.as_deref()) {
        let max_wait = POLL_SLEEP * MAX_POLLS;
        eprintln!(
            "warning: still non-terminal after {} polls (~{:?} max wait); increase MAX_POLLS or check the run in the dashboard",
            MAX_POLLS, max_wait
        );
    }

    println!("final status: {:?}", last.status);
    Ok(())
}
