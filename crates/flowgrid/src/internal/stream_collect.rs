//! Collect items from a fallible [`futures::Stream`] that is [`Unpin`] (for example
//! [`crate::internal::sse::oai::OpenAiSseEventStream`] or [`crate::internal::sse::clu::AnthropicSseEventStream`]).
//!
//! **Warning:** This buffers the entire stream in memory; do not use on unbounded producer streams.

use futures::{Stream, StreamExt};

/// Drains `stream` until end-of-stream, returning the first error if any item is `Err`.
pub async fn try_collect_unpin<S, T, E>(mut stream: S) -> Result<Vec<T>, E>
where
    S: Stream<Item = Result<T, E>> + Unpin,
{
    let mut out = Vec::new();
    while let Some(item) = stream.next().await {
        out.push(item?);
    }
    Ok(out)
}
